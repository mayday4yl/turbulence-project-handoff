"""Validation-only V0.3 checkpoint adapter around unchanged V0.1 inference.

Run with ``python -m v03.evaluate --config ... --checkpoint ... --output-dir ...``
or torchrun. No test split is accepted or opened. The trainer is not imported.
The original run_rollout owns schedules, candidate selection, all refinement,
network autocast, cropping, and transitions. Extra measurements run afterwards.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


G_GRID = (0.75, 1.5, 2.25, 3.0, 3.75, 4.5)
ALLOWED_K = (0, 1, 3, 5)
METRIC_DIRECTIONS = {"psnr": 1, "ssim": 1, "lpips": -1, "l1": -1}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("K must be a comma-separated subset of 0,1,3,5") from error
    if not ks or len(ks) != len(set(ks)) or not set(ks).issubset(ALLOWED_K):
        raise ValueError("K must be a nonempty unique subset of 0,1,3,5")
    if 0 not in ks:
        raise ValueError("Include K=0 for paired delta and preservation checks")
    return tuple(sorted(ks))


def select_val_records(records: Sequence[Mapping[str, Any]], limit: int | None,
                       selection: str = "sha256-key-v1") -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit-trajectories must be positive; omit for all validation")
    selected = [dict(record) for record in records]
    if selection == "sha256-key-v1":
        selected.sort(key=lambda record: hashlib.sha256(
            ("v03-val-v1:" + str(record["key"])).encode()).hexdigest())
    elif selection != "first-manifest":
        raise ValueError(f"Unknown fixed validation selection: {selection}")
    if not selected:
        raise ValueError("Validation manifest is empty")
    if len({record["key"] for record in selected}) != len(selected):
        raise ValueError("Duplicate validation keys")
    if limit is not None and limit > len(selected):
        raise ValueError(f"Requested {limit} validation trajectories but only {len(selected)} exist")
    return selected if limit is None else selected[:limit]


def verify_val_source_split(dataset_root: Path) -> dict[str, Any]:
    """Only train and val manifests are read, never held-out test metadata."""
    manifests = {split: dataset_root / "manifests" / f"{split}.jsonl"
                 for split in ("train", "val")}
    rows = {split: load_jsonl(path) for split, path in manifests.items()}
    for split, records in rows.items():
        if not records or any(not record.get("source_member") for record in records):
            raise ValueError(f"{split} manifest is empty or lacks source_member")
        if any(record.get("split", split) != split for record in records):
            raise ValueError(f"Wrong split label in {split} manifest")
    sources = {split: {record["source_member"] for record in records}
               for split, records in rows.items()}
    overlap = sources["train"] & sources["val"]
    if overlap:
        raise ValueError(f"Train/val source leakage: {len(overlap)} source images")
    return {"split": "val", "train_val_source_overlap": 0,
            "source_count": {split: len(value) for split, value in sources.items()},
            "manifest_sha256": {split: sha256_file(path) for split, path in manifests.items()},
            "held_out_test_opened": False}


def paired_delta(quality: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, Any]:
    raw = {name: float(quality[name]) - float(baseline[name]) for name in METRIC_DIRECTIONS}
    return {"raw_metric_K_minus_K0": raw,
            "improvement_positive": {name: raw[name] * direction
                                     for name, direction in METRIC_DIRECTIONS.items()}}


def _finite_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(not math.isfinite(float(value)) for value in values):
        raise FloatingPointError("Non-finite validation metric")
    return float(statistics.fmean(values))


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(rows),
        "quality": {name: _finite_mean([row["final_quality"][name] for row in rows])
                    for name in METRIC_DIRECTIONS},
        "delta_K_minus_K0": {name: _finite_mean([
            row["delta_vs_K0"]["raw_metric_K_minus_K0"][name] for row in rows])
            for name in METRIC_DIRECTIONS},
        "improvement_positive": {name: _finite_mean([
            row["delta_vs_K0"]["improvement_positive"][name] for row in rows])
            for name in METRIC_DIRECTIONS},
    }
    # First average transition diagnostics inside each path, then across paths.
    # A zero-transition path has no next-state/refined residual, rather than zero.
    diagnostics = {}
    for field in ("next_state_mse", "observation_residual_before", "observation_residual_after"):
        path_values = [_finite_mean([step[field] for step in row["steps"]])
                       for row in rows if row["steps"]]
        diagnostics[field + "_path_mean"] = _finite_mean(path_values)
        diagnostics[field + "_path_count"] = len(path_values)
    for field in ("final_network_xi_observation_mse_full",
                  "final_network_xi_observation_mse_valid"):
        diagnostics[field] = _finite_mean([row["diagnostics"][field] for row in rows])
    result["mechanism_diagnostics"] = diagnostics
    result["actual_prediction_counts"] = sorted({row["network_forwards"] for row in rows})
    result["physical_rollout_calls"] = sum(not row.get("cache_reused", False) for row in rows)
    result["wall_seconds_total"] = sum(row["evaluation_wall_seconds"] for row in rows)
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str],
                   ks: Sequence[int]) -> dict[str, Any]:
    expected = {(key, severity, k) for key in keys for severity in G_GRID for k in ks}
    indexed = {(row["key"], float(row["G"]), int(row["requested_K"])): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate (trajectory,G,K) validation rows")
    if set(indexed) != expected:
        raise ValueError(f"Validation coverage mismatch: missing={len(expected-set(indexed))}, "
                         f"extra={len(set(indexed)-expected)}")
    by_k: dict[str, Any] = {}
    for k in ks:
        selected = [row for row in rows if int(row["requested_K"]) == k]
        per_g = {f"{g:.2f}": _group_summary([row for row in selected if float(row["G"]) == g])
                 for g in G_GRID}
        by_k[str(k)] = {
            "per_g": per_g,
            "macro_g": {name: _finite_mean([item["quality"][name] for item in per_g.values()])
                        for name in METRIC_DIRECTIONS},
            "delta_macro_g_K_minus_K0": {name: _finite_mean([
                item["delta_K_minus_K0"][name] for item in per_g.values()])
                for name in METRIC_DIRECTIONS},
            "improvement_positive_macro_g": {name: _finite_mean([
                item["improvement_positive"][name] for item in per_g.values()])
                for name in METRIC_DIRECTIONS},
            "all_paths": _group_summary(selected),
        }
    return {"split": "val", "trajectory_count": len(keys), "row_count": len(rows),
            "per_k": by_k,
            "macro_g_and_k": {name: _finite_mean([value["macro_g"][name]
                              for value in by_k.values()]) for name in METRIC_DIRECTIONS},
            "aggregation": "per-image metrics -> equal trajectories within G -> equal six G; "
                           "macro_g_and_k gives equal weight to requested K labels",
            "metric_region": "full image; original SSIM implementation includes its own padding",
            "delta_definition": "metric(K)-metric(K0)",
            "improvement_directions": METRIC_DIRECTIONS,
            "diagnostics_are_not_checkpoint_selection_metrics": True,
            "held_out_test_opened": False}


def frozen_refinement(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import yaml
    from physical_bridge.inference import RefinementConfig

    paths = config["base"]
    inference_path = Path(paths["inference_config"])
    frozen_path = Path(paths["frozen_inference"])
    document = yaml.safe_load(inference_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_hash = config.get("base_v01_hashes", {}).get("configs/inference_v01.yaml")
    if expected_hash and sha256_file(inference_path) != expected_hash:
        raise ValueError("Inference configuration differs from audited V0.1")
    value = document["refinement"]
    if value["optimizer"] != "Adam":
        raise ValueError("Original V0.1 refinement optimizer must be Adam")
    selected_lr = float(frozen["learning_rate"])
    if selected_lr != 0.03:
        raise ValueError("Frozen V0.1 refinement LR changed from audited 0.03")
    refinement = RefinementConfig(
        learning_rate=selected_lr, betas=tuple(value["betas"]),
        first_iterations=int(value["first_iterations"]),
        later_iterations=int(value["later_iterations"]),
        gradient_clip=float(value["gradient_clip"]),
        early_stop_patience=int(value["early_stop_patience"]),
        early_stop_relative_improvement=float(value["early_stop_relative_improvement"]),
        valid_border=int(value["valid_border"]),
        proximal_weight=float(value["proximal_weight"]), prior_weight=float(value["prior_weight"]),
    )
    return refinement, {"config": asdict(refinement), "optimizer": "Adam",
                        "inference_config_sha256": sha256_file(inference_path),
                        "frozen_inference_sha256": sha256_file(frozen_path),
                        "learning_rate_source": "original V0.1 frozen selection, no retuning",
                        "original_frozen_selected_K": frozen.get("K")}


def make_lpips(config: Mapping[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    """Use original metric/weights. Fail if cache missing, never download weights."""
    import torch
    import torchvision
    from physical_bridge.metrics import LPIPSMetric

    val = config.get("validation", {})
    if val.get("lpips_torch_home"):
        torch.hub.set_dir(str(Path(val["lpips_torch_home"]) / "hub"))
    spec = importlib.util.find_spec("lpips")
    if spec is None or spec.origin is None:
        raise ImportError("Original lpips package is unavailable")
    package_weights = Path(spec.origin).parent / "weights" / "v0.1" / "alex.pth"
    weights_enum = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    backbone = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights_enum.url).path).name
    for path in (package_weights, backbone):
        if not path.is_file():
            raise FileNotFoundError(f"Original LPIPS weights must already exist: {path}")
    metric_device = torch.device(val.get("lpips_device", "cpu"))
    if metric_device.type not in ("cpu", device.type):
        raise ValueError("LPIPS device must be CPU or the evaluation device")
    if metric_device.type == device.type:
        metric_device = device
    metric = LPIPSMetric(metric_device)

    def original_metric_on_selected_device(prediction: Any, target: Any) -> Any:
        return metric(prediction.detach().float().to(metric_device),
                      target.detach().float().to(metric_device))

    return original_metric_on_selected_device, {
        "implementation": "physical_bridge.metrics.LPIPSMetric", "network": "alex",
        "device": str(metric_device), "torchvision_version": torchvision.__version__,
        "lpips_weights": str(package_weights), "lpips_weights_sha256": sha256_file(package_weights),
        "alexnet_weights": str(backbone), "alexnet_weights_sha256": sha256_file(backbone),
    }


def _synchronize(device: Any) -> None:
    import torch
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_qa(output_dir: Path, key: str, g: float, k: int, clean: Any,
             observation: Any, final_input: Any, final_prediction: Any) -> dict[str, str]:
    import torch
    from PIL import Image, ImageDraw

    tag = f"{hashlib.sha256(key.encode()).hexdigest()[:12]}_G{g:.2f}_K{k}"
    target = output_dir / "qualitative"
    target.mkdir(parents=True, exist_ok=True)
    tensors = {"target_clean": clean.detach().float().cpu(),
               "observation": observation.detach().float().cpu(),
               "final_input": final_input.detach().float().cpu(),
               "final_prediction": final_prediction.detach().float().cpu()}
    tensor_path = target / f"{tag}.pt"
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save({"key": key, "G": g, "requested_K": k, **tensors}, temporary)
    temporary.replace(tensor_path)
    images = []
    for name, tensor in tensors.items():
        array = tensor[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
        images.append((name, Image.fromarray(array)))
    width, height = images[0][1].size
    canvas = Image.new("RGB", (len(images) * width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, image) in enumerate(images):
        canvas.paste(image, (column * width, 24))
        draw.text((column * width + 3, 4), name, fill="black")
    image_path = target / f"{tag}.png"
    canvas.save(image_path)
    return {"tensor": str(tensor_path.relative_to(output_dir)),
            "image": str(image_path.relative_to(output_dir))}


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint_path: Path, output_dir: Path,
                        limit_trajectories: int | None = None,
                        ks: Sequence[int] = ALLOWED_K) -> dict[str, Any] | None:
    """Validation entry point; ranks independently own (trajectory,G) tasks."""
    import torch

    ks = parse_ks(",".join(str(k) for k in ks))
    base_source = Path(config["base_source"]).resolve()
    sys.path.insert(0, str(base_source))
    from .common import build_model, build_simulator, verify_base
    verify_base(config)
    from physical_bridge.data import BridgeArchive, evenly_spaced_schedule
    from physical_bridge.inference import run_rollout

    world_size, rank, local_rank = (int(os.environ.get(name, default)) for name, default in
                                    (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    val = config.get("validation", {})
    device_type = val.get("device", "npu")
    if device_type == "npu":
        import torch_npu  # noqa: F401
        if world_size > 4 or local_rank not in range(4):
            raise ValueError("Only authorized de36 physical NPU 0-3 are allowed")
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Evaluation device must be npu or cpu")
    owns_group = world_size > 1 and not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(
            backend="hccl" if device.type == "npu" else "gloo",
            timeout=dt.timedelta(seconds=int(val.get("distributed_timeout_seconds", 21600))))
    barrier = torch.distributed.barrier if world_size > 1 else lambda: None
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()
    archive = None
    hook = None
    try:
        dataset_root = Path(config["dataset_root"])
        split_info = verify_val_source_split(dataset_root)
        archive = BridgeArchive(dataset_root, "val")
        records = select_val_records(archive.records, limit_trajectories,
                                     val.get("selection", "sha256-key-v1"))
        keys = [record["key"] for record in records]
        refinement, refinement_info = frozen_refinement(config)
        weight_key = val.get("weight_key", "ema")
        if weight_key not in ("ema", "model"):
            raise ValueError("Checkpoint weight_key must be ema or model")
        checkpoint_sha = sha256_file(checkpoint_path)
        identity = {"version": "v03-val-v1", "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha, "weight_key": weight_key,
                    "config_sha256": document_sha256(config), "ks": list(ks),
                    "selection": val.get("selection", "sha256-key-v1"), "keys": keys,
                    "selected_keys_sha256": document_sha256(keys), "world_size": world_size,
                    "split_info": split_info, "refinement": refinement_info,
                    "inference_source_sha256": sha256_file(base_source / "physical_bridge/inference.py"),
                    "data_source_sha256": sha256_file(base_source / "physical_bridge/data.py"),
                    "metrics_source_sha256": sha256_file(base_source / "physical_bridge/metrics.py")}
        identity["adapter_source_sha256"] = sha256_file(Path(__file__))
        identity_hash = document_sha256(identity)
        identity_path = output_dir / f"rank-{rank:02d}.identity.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text(encoding="utf-8"))
            if previous != identity:
                raise ValueError("Output directory contains a different validation identity")
        else:
            if (output_dir / f"rank-{rank:02d}.jsonl").exists():
                raise ValueError("Cannot resume rows without matching identity manifest")
            atomic_json(identity_path, identity)
        model = build_model(config, device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if weight_key not in checkpoint:
            raise KeyError(f"Checkpoint lacks original V0.1-compatible {weight_key} state dict")
        model.load_state_dict(checkpoint[weight_key], strict=True)
        model.eval().requires_grad_(False)
        checkpoint_update = checkpoint.get("update", checkpoint.get("step"))
        del checkpoint
        simulator = build_simulator(config, device)
        lpips_metric, lpips_info = make_lpips(config, device)
        atomic_json(output_dir / f"rank-{rank:02d}.compatibility.json", {
            "load_state_dict_strict": True, "weight_key": weight_key,
            "checkpoint_update": checkpoint_update, "lpips": lpips_info,
            "inference_source_sha256": identity["inference_source_sha256"],
            "device": str(device), "rank": rank, "world_size": world_size,
            "refinement": refinement_info, "true_states_role": "diagnostics only"})
        last_output: dict[str, Any] = {}

        def capture_final_output(_module: Any, inputs: Any, output: Any) -> None:
            # Forward hooks returning None cannot replace inputs or outputs.
            last_output["input"] = inputs[0]
            last_output["prediction"] = output

        hook = model.register_forward_hook(capture_final_output)
        rank_path = output_dir / f"rank-{rank:02d}.jsonl"
        existing = load_jsonl(rank_path)
        if any(row.get("validation_identity") != identity_hash for row in existing):
            raise ValueError("Existing validation rows have a different identity")
        complete = {(row["key"], float(row["G"]), int(row["requested_K"])): row
                    for row in existing}
        if len(complete) != len(existing):
            raise ValueError("Duplicate existing validation rows")
        capture_keys = set(keys[:int(val.get("capture_trajectories", 1))])
        capture_g = {float(g) for g in val.get("capture_G", [4.5])}
        reuse = bool(val.get("reuse_identical_schedules", True))
        tasks = [(record, g) for record in records for g in G_GRID]
        # Greedy balancing avoids every other rank receiving only the odd/even G
        # subset when six severities are distributed across four ranks.
        def task_cost(task: Any) -> int:
            _, severity = task
            seen = set()
            cost = 0
            for k in ks:
                method = "one_shot" if k == 0 else "bridge_refinement"
                schedule = tuple(evenly_spaced_schedule(severity, k))
                signature = (method, schedule)
                if reuse and signature in seen:
                    continue
                seen.add(signature)
                transitions = len(schedule)-1
                iterations = (refinement.first_iterations +
                              max(0, transitions-1)*refinement.later_iterations) if transitions else 0
                cost += len(schedule) + 2*iterations + 4*transitions + 1
            return cost
        allocations: list[list[Any]] = [[] for _ in range(world_size)]
        loads = [0]*world_size
        for task in sorted(tasks, key=task_cost, reverse=True):
            owner = min(range(world_size), key=lambda worker: (loads[worker], worker))
            allocations[owner].append(task)
            loads[owner] += task_cost(task)
        assigned = allocations[rank]
        for task_number, (record, g) in enumerate(assigned):
            clean = archive.clean(record).unsqueeze(0).to(device)
            states = {level: archive.state(record, level).unsqueeze(0).to(device)
                      for level in G_GRID if level <= g}
            observation = states[g]
            baseline = complete.get((record["key"], g, 0), {}).get("final_quality")
            cache: dict[tuple[Any, ...], dict[str, Any]] = {}
            for k in ks:
                row_key = (record["key"], g, k)
                method = "one_shot" if k == 0 else "bridge_refinement"
                schedule = tuple(evenly_spaced_schedule(g, k))
                cache_key = (method, schedule)
                if row_key in complete:
                    old_row = complete[row_key]
                    cache[cache_key] = old_row
                    if k == 0:
                        baseline = old_row["final_quality"]
                    continue
                if reuse and cache_key in cache:
                    row = copy.deepcopy(cache[cache_key])
                    row.update(requested_K=k, cache_reused=True,
                               cache_source_K=int(cache[cache_key]["requested_K"]),
                               evaluation_wall_seconds=0.0, rollout_wall_seconds=0.0,
                               extra_diagnostics_wall_seconds=0.0)
                    # QA paths intentionally point to the physically computed identical output.
                else:
                    last_output.clear()
                    if device.type == "npu":
                        torch.npu.reset_peak_memory_stats(device)
                    _synchronize(device)
                    evaluation_start = time.monotonic()
                    result = run_rollout(
                        model=model, simulator=simulator, observation=observation,
                        observation_severity=g, transitions=k, method=method,
                        refinement_config=refinement, true_clean=clean,
                        true_realization=None, true_states=states, lpips_metric=lpips_metric,
                        capture_tensors=False)
                    _synchronize(device)
                    rollout_seconds = time.monotonic() - evaluation_start
                    diagnostic_start = time.monotonic()
                    prediction = last_output["prediction"]
                    if not torch.equal(result["final"], prediction.clean):
                        raise AssertionError("Final diagnostic hook does not match rollout output")
                    with torch.no_grad():
                        reconstructed = simulator.render(prediction.clean.float(),
                                                         prediction.realization, g).output
                        full_mse = float((reconstructed.float()-observation.float()).square().mean().cpu())
                        border = refinement.valid_border
                        valid_mse = float((reconstructed[..., border:-border, border:-border].float()
                                           - observation[..., border:-border, border:-border].float())
                                          .square().mean().cpu())
                    _synchronize(device)
                    diagnostics_seconds = time.monotonic() - diagnostic_start
                    qa = None
                    if record["key"] in capture_keys and g in capture_g:
                        qa = _save_qa(output_dir, record["key"], g, k, clean, observation,
                                      last_output["input"], result["final"])
                    result.pop("final")
                    row = {"key": record["key"], "source_member": record["source_member"],
                           "G": g, "manifest_terminal_G": record.get("terminal_G"),
                           "requested_K": k, "validation_identity": identity_hash,
                           **result, "cache_reused": False, "rollout_wall_seconds": rollout_seconds,
                           "extra_diagnostics_wall_seconds": diagnostics_seconds,
                           "evaluation_wall_seconds": time.monotonic()-evaluation_start,
                           "diagnostics": {
                               "final_network_xi_observation_mse_full": full_mse,
                               "final_network_xi_observation_mse_valid": valid_mse,
                               "final_network_xi_observation_mse_valid_border": border,
                               "extra_simulator_forwards": 1,
                               "semantics": "final clean with final network xi, original Y/G anchor; "
                                            "diagnostic only, no final refinement or transition"},
                           "qualitative": qa,
                           "peak_allocated_bytes": int(torch.npu.max_memory_allocated(device))
                           if device.type == "npu" else None,
                           "peak_reserved_bytes": int(torch.npu.max_memory_reserved(device))
                           if device.type == "npu" else None}
                    if k == 0:
                        baseline = row["final_quality"]
                    if baseline is None:
                        raise AssertionError("K0 baseline must be computed before progressive K")
                    row["delta_vs_K0"] = paired_delta(row["final_quality"], baseline)
                    cache[cache_key] = row
                    last_output.clear()
                    del prediction, reconstructed
                append_json(rank_path, row)
                complete[row_key] = row
                atomic_json(output_dir / f"rank-{rank:02d}.progress.json", {
                    "status": "running", "completed_rows": len(complete),
                    "assigned_tasks": len(assigned), "task_number": task_number,
                    "last_key": record["key"], "last_G": g, "last_K": k,
                    "elapsed_seconds": time.monotonic()-overall_start})
            del clean, states, observation
        atomic_json(output_dir / f"rank-{rank:02d}.complete.json", {
            "validation_identity": identity_hash, "rows": len(complete),
            "elapsed_seconds": time.monotonic()-overall_start})
        barrier()
        summary = None
        if rank == 0:
            combined = []
            for worker_rank in range(world_size):
                other_identity = json.loads((output_dir / f"rank-{worker_rank:02d}.identity.json")
                                            .read_text(encoding="utf-8"))
                if other_identity != identity:
                    raise ValueError("Validation rank identities differ")
                combined.extend(load_jsonl(output_dir / f"rank-{worker_rank:02d}.jsonl"))
            combined.sort(key=lambda row: (keys.index(row["key"]), row["G"], row["requested_K"]))
            summary = {**summarize_rows(combined, keys, ks), "status": "complete",
                       "identity": identity, "validation_identity": identity_hash,
                       "checkpoint_update": checkpoint_update, "lpips": lpips_info,
                       "world_size": world_size, "wall_seconds": time.monotonic()-overall_start,
                       "physical_rollout_calls": sum(not row["cache_reused"] for row in combined),
                       "inference_unchanged": True,
                       "timing_note": "Original rollout wall time includes step/final quality metrics; "
                                      "extra final reconstruction diagnostic and QA are separately labeled. "
                                      "First-time compilation is not a steady-state throughput estimate."}
            all_path = output_dir / "all.jsonl"
            temporary = all_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in combined:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            temporary.replace(all_path)
            atomic_json(output_dir / "summary.json", summary)
            print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                              "rows": len(combined), "per_k_macro_g": {
                                  k: value["macro_g"] for k, value in summary["per_k"].items()}},
                             sort_keys=True), flush=True)
        barrier()
        return summary
    except BaseException as error:
        atomic_json(output_dir / f"rank-{rank:02d}.failure.json", {
            "type": type(error).__name__, "message": str(error),
            "elapsed_seconds": time.monotonic()-overall_start})
        raise
    finally:
        if hook is not None:
            hook.remove()
        if archive is not None:
            archive.store.close()
        if owns_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit-trajectories", type=int)
    parser.add_argument("--ks", default="0,1,3,5")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluate_checkpoint(config, args.checkpoint, args.output_dir,
                        limit_trajectories=args.limit_trajectories, ks=parse_ks(args.ks))


if __name__ == "__main__":
    main()

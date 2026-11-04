"""Bounded D0 pilot using the unchanged official DiffUIR-L training definition.

This runner adapts only the frozen FP32 V5 pairs, logging and checkpoint policy.
It never resumes an old optimizer, changes the diffusion equations, or adds a
physical-time/xi/conditioning module. All prerequisite gates must already pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from common import (DATA, ROOT, evaluate, load_array, load_official, seed,
                    setup_npu, sha, val4_pairs, write_json)
from grad_helpers import (capture_rng, cpu_copy, observe_training_indices,
                          restore_rng, state_digest)


EXPECTED = {
    "updates": 100,
    "microbatch": 1,
    "gradient_accumulate_every": 2,
    "lr": 8e-5,
    "adam_betas": [0.9, 0.99],
    "grad_clip_norm": 1.0,
    "precision": "FP32",
    "sampling_seed": 12345,
    "eval_updates": [0, 25, 50, 75, 100],
    "checkpoint_updates": [0, 50, 100],
    "ema_beta": 0.995,
    "ema_update_every": 10,
}


def local_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Pilot artifacts must remain inside the new root: {path}")
    return path


def validate_config(path):
    config = json.loads(path.read_text())
    if config.get("frozen") is not True:
        raise ValueError("PILOT_CONFIG must explicitly be frozen before execution")
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected pilot {key}: {config.get(key)!r}; expected {expected!r}")
    if not isinstance(config.get("seed"), int) or isinstance(config.get("seed"), bool):
        raise ValueError("A frozen integer training seed is required")
    if not isinstance(config.get("selection_sha256"), str) or len(config["selection_sha256"]) != 64:
        raise ValueError("Frozen selection_sha256 is required")
    if config.get("augmentation", "none") != "none":
        raise ValueError("This pilot adapter preserves arrays without online augmentation")
    if config.get("initial_branch", config.get("init_branch", "model")) != "model":
        raise ValueError("Pilot must restart from the official raw model branch")
    if config.get("scheduler", "none") != "none":
        raise ValueError("The official pilot uses no learning-rate scheduler")
    if config.get("weight_decay", 0.0) != 0.0:
        raise ValueError("Expected native Adam weight_decay=0")
    return config


def verify_gates(config):
    paths = config.get("gate_paths", {
        "weights": "evidence/WEIGHTS_STATUS.json",
        "inference": "evidence/INFERENCE_STATUS.json",
        "gradient": "evidence/GRADIENT_STATUS.json",
    })
    if set(paths) != {"weights", "inference", "gradient"}:
        raise ValueError("Exactly weights/inference/gradient prerequisite gates are required")
    results = {}
    for name, value in paths.items():
        path = local_path(value)
        gate = json.loads(path.read_text())
        if gate.get("status") != "PASS":
            raise RuntimeError(f"Required {name} gate has not passed: {gate.get('status')}")
        results[name] = {"path": str(path), "sha256": sha(path), "record": gate}
    return results


def record_id(row):
    return row.get("id", row.get("sample_id"))


def validate_selection(selection):
    groups = {"train": selection.get("train", []), "val50": selection.get("val50", [])}
    train_sources = set()
    for split, rows in groups.items():
        expected_count = 400 if split == "train" else 50
        if len(rows) != expected_count:
            raise ValueError(f"Expected {expected_count} frozen {split} rows, got {len(rows)}")
        ids = [record_id(row) for row in rows]
        if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Missing or repeated {split} sample IDs")
        sources = defaultdict(set)
        categories = defaultdict(set)
        for row in rows:
            sample_id = record_id(row)
            source = row["source_id"]
            category = row["category_id"]
            realization = row["realization_index"]
            strength = row.get("g", row.get("G", row.get("strength")))
            if strength != 4 or isinstance(strength, bool):
                raise ValueError(f"Non-G4 or missing strength: {sample_id}")
            expected_split = "train" if split == "train" else "val"
            if row.get("split", expected_split) != expected_split or not sample_id.startswith(expected_split + "_"):
                raise ValueError(f"Wrong split identity: {sample_id}")
            if split == "val50" and realization != 0:
                raise ValueError("Val50 must use r00 only")
            sources[source].add(realization)
            categories[category].add(source)
        expected_sources = 100 if split == "train" else 50
        expected_per_category = 2 if split == "train" else 1
        if len(sources) != expected_sources or len(categories) != 50:
            raise ValueError(f"Wrong frozen source/category counts for {split}")
        if any(len(values) != expected_per_category for values in categories.values()):
            raise ValueError(f"Unbalanced source counts in {split}")
        if split == "train":
            if any(values != {0, 1, 2, 3} for values in sources.values()):
                raise ValueError("Each training source must have exactly four original realizations")
            train_sources = set(sources)
        elif train_sources & set(sources):
            raise ValueError("Train and validation source identities overlap")
    return groups


def read_pairs(rows, data_root):
    pairs = []
    for row in rows:
        pair = {"id": record_id(row)}
        for key in ("clean", "observation"):
            reference = Path(row[key + "_ref"])
            if reference.is_absolute() or ".." in reference.parts or "test" in reference.parts:
                raise ValueError(f"Unexpected or Test data reference: {reference}")
            path = (data_root / reference).resolve()
            if not path.is_relative_to(data_root.resolve()):
                raise ValueError("Data reference leaves canonical data root")
            pair[key] = load_array(path, row[key + "_sha256"])
        pairs.append(pair)
    return pairs


class FrozenSampler:
    def __init__(self, ids, random_seed):
        self.ids = list(ids)
        self.random_seed = random_seed
        self.epoch = 0
        self.cursor = 0
        self.exposures = 0
        self.order = self._order()

    def _order(self):
        return np.random.default_rng(self.random_seed + self.epoch).permutation(len(self.ids)).tolist()

    def next(self):
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.order = self._order()
        index = self.order[self.cursor]
        self.cursor += 1
        self.exposures += 1
        return index

    def state_dict(self):
        next_index = self.order[self.cursor] if self.cursor < len(self.order) else None
        return {"algorithm": "numpy.default_rng(seed+epoch).permutation",
                "seed": self.random_seed, "epoch": self.epoch, "cursor": self.cursor,
                "sample_exposures": self.exposures, "permutation": self.order,
                "sample_ids": self.ids, "next_index": next_index,
                "next_sample_id": self.ids[next_index] if next_index is not None else None}


@contextmanager
def preserve_training_state_for_evaluation(model, device):
    """Match official test schedule init without changing the training stream."""
    rng = capture_rng(device)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        model.init()
        model.to(device)
        yield
    finally:
        with torch.no_grad():
            current = dict(model.named_buffers())
            if set(current) != set(buffers):
                raise RuntimeError("Evaluation changed the official buffer key set")
            for name, saved in buffers.items():
                current[name].copy_(saved)
        for name, module in model.named_modules():
            module.training = modes[name]
        restore_rng(rng, device)


def eval_panel(model, pairs, device, directory, prefix, random_seed):
    with preserve_training_state_for_evaluation(model, device):
        result = evaluate(model, pairs, device, directory, prefix=prefix, random_seed=random_seed)
    if result.get("status") != "PASS":
        raise RuntimeError(f"Evaluation gate failed for {prefix}")
    if not all(math.isfinite(float(value)) for value in result["mean_metrics"].values()):
        raise RuntimeError(f"Nonfinite evaluation metrics for {prefix}")
    return {"status": result["status"], "sample_count": result["sample_count"],
            "mean_metrics": result["mean_metrics"],
            "mean_input_metrics": result["mean_input_metrics"],
            "result_path": str(Path(directory) / f"{prefix}_RESULT.json")}


def finite_all(tensors):
    tensors = list(tensors)
    return bool(tensors) and bool(torch.stack([torch.isfinite(value).all() for value in tensors]).all().item())


def ema_configuration(ema):
    names = ("beta", "update_after_step", "update_every", "inv_gamma", "power", "min_value",
             "include_online_model", "allow_different_devices", "coerce_dtype", "use_foreach",
             "update_model_with_ema_every", "update_model_with_ema_beta")
    values = {}
    for name in names:
        value = getattr(ema, name, None)
        values[name] = value if isinstance(value, (type(None), str, bool, int, float)) else repr(value)
    return {"package_version": importlib.metadata.version("ema-pytorch"),
            "constructor_signature": str(inspect.signature(type(ema).__init__)),
            "explicit_arguments": {"beta": .995, "update_every": 10},
            "effective_attributes": values,
            "warmup_policy": "All unprovided defaults retained exactly as the installed official EMA class; no warmup override",
            "initialization": "Fresh EMA and step counter from the freshly loaded official raw model; old checkpoint EMA/optimizer counters are not resumed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "protocol/PILOT_CONFIG.json"))
    parser.add_argument("--selection", default=str(ROOT / "protocol/PILOT_SELECTION.json"))
    args = parser.parse_args()
    status_path = ROOT / "evidence/PILOT_STATUS.json"
    run_dir = None
    status = {"schema": "diffuir_l_d0_pilot_v1", "status": "CHECKING_PREREQUISITES",
              "formal_training": False, "short_d0_pilot": True, "test_used": False,
              "physical_simulator": False, "xi_head": False, "physical_time_modulation": False,
              "optimizer_updates_completed": 0}
    # A second invocation is not allowed to overwrite an existing pilot record.
    if status_path.exists():
        raise FileExistsError(f"Pilot status already exists; use a separately authorized new run: {status_path}")
    try:
        config_path = local_path(args.config)
        selection_path = local_path(args.selection)
        config = validate_config(config_path)
        gates = verify_gates(config)
        if sha(selection_path) != config["selection_sha256"]:
            raise RuntimeError("Frozen pilot selection SHA256 mismatch")
        selection = json.loads(selection_path.read_text())
        groups = validate_selection(selection)
        data_root = Path(config.get("data_root", str(DATA)))
        if data_root.resolve() != DATA.resolve():
            raise ValueError("Pilot must use the original canonical V5 data root")
        run_dir = local_path(config.get("run_dir", "runs/d0_pilot_100u"))
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "evaluations"):
            (run_dir / name).mkdir()
        config_sha = sha(config_path)
        selection_sha = sha(selection_path)
        write_json(run_dir / "FROZEN_CONFIG.json", config)
        write_json(run_dir / "FROZEN_SELECTION.json", selection)
        status.update({"status": "PREPARING", "run_dir": str(run_dir),
                       "config_sha256": config_sha, "selection_sha256": selection_sha,
                       "prerequisite_gates": gates, "started_unix": time.time(),
                       "updates_authorized": 100, "precision": "FP32"})
        write_json(status_path, status)
        write_json(run_dir / "STATUS.json", status)

        # Hash and shape-check every selected input before placing a model on NPU.
        train_pairs = read_pairs(groups["train"], data_root)
        validation_pairs = read_pairs(groups["val50"], data_root)
        fixed_val4 = val4_pairs()
        model, checkpoint_audit = load_official("model", run_dir / "CHECKPOINT_AUDIT.json")
        if config.get("official_checkpoint_sha256", checkpoint_audit["sha256"]) != checkpoint_audit["sha256"]:
            raise RuntimeError("Official checkpoint does not match the frozen Pilot configuration")
        device = setup_npu()
        seed(config["seed"])
        model = model.to(device).float()
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=config["lr"],
                                     betas=tuple(config["adam_betas"]), weight_decay=0.)
        from ema_pytorch import EMA
        ema = EMA(model, beta=config["ema_beta"], update_every=config["ema_update_every"]).to(device)
        ema_info = ema_configuration(ema)
        write_json(run_dir / "EMA_CONFIGURATION.json", ema_info)
        sampler = FrozenSampler([pair["id"] for pair in train_pairs], config["seed"])
        initial_digest = state_digest(model.state_dict())
        identity = {"script_sha256": sha(Path(__file__)),
                    "common_sha256": sha(ROOT / "scripts/common.py"),
                    "official_model_source_sha256": sha(ROOT / "upstream/src/model.py"),
                    "checkpoint_audit": checkpoint_audit,
                    "initial_model_state_sha256": initial_digest,
                    "ema": ema_info, "torch_version": torch.__version__,
                    "visible_physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    "data_policy": "Original FP32 RGB arrays, exact SHA verification; no online augmentation, input clamp, rescale, crop or PNG conversion",
                    "training_definition": "Unmodified official forward([Clean,Y4,['blur']]); random single diffusion index; pixel residual L1; original schedule and condition",
                    "optimizer_policy": "Fresh Adam from official raw model; no diagnostic-updated weights or old optimizer state",
                    "metric_policy": "Fixed Val50, original RGB range1 without extra crop/clamp/resize; fixed per-image sampling seed; best by EMA Val50 PSNR",
                    "subset_counts": {"train_pairs": 400, "train_sources": 100, "val50_pairs": 50,
                                      "train_categories": 50, "val50_categories": 50}}
        write_json(run_dir / "RUN_IDENTITY.json", identity)
        evaluation_history = []
        best = None
        checkpoint_records = []
        update_rows = []
        last_checkpoint_update = None

        def actual_coverage():
            by_id = {record_id(row): row for row in groups["train"]}
            seen = [sample_id for row in update_rows for sample_id in row["sample_ids"]]
            unique = sorted(set(seen))
            source_counts = Counter(by_id[sample_id]["source_id"] for sample_id in seen)
            realization_counts = Counter(by_id[sample_id]["realization_index"] for sample_id in seen)
            categories = sorted({by_id[sample_id]["category_id"] for sample_id in seen})
            return {"eligible_pool_pairs": 400, "eligible_pool_sources": 100,
                    "sample_exposures": len(seen), "unique_realizations_seen": len(unique),
                    "unique_sources_seen": len(source_counts), "unique_categories_seen": len(categories),
                    "sample_ids_seen": unique, "source_exposure_counts": dict(source_counts),
                    "realization_index_exposure_counts": dict(realization_counts),
                    "categories_seen": categories,
                    "interpretation": "400 is the frozen eligible pool, not the number of pairs necessarily consumed by this pilot"}

        def save_checkpoint(update):
            nonlocal last_checkpoint_update
            path = run_dir / "checkpoints" / f"U{update:06d}.pt"
            if path.exists():
                return path
            rng = capture_rng(device)
            payload = {"schema": "diffuir_l_d0_pilot_checkpoint_v1", "step": update,
                       "model": cpu_copy(model.state_dict()), "opt0": cpu_copy(optimizer.state_dict()),
                       "ema": cpu_copy(ema.state_dict()), "rng": cpu_copy(rng),
                       "sampler": sampler.state_dict(), "config_sha256": config_sha,
                       "selection_sha256": selection_sha, "config": config,
                       "ema_configuration": ema_info, "evaluation_history": evaluation_history,
                       "best": best, "actual_training_coverage": actual_coverage(),
                       "source_checkpoint_sha256": checkpoint_audit["sha256"]}
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            del payload
            record = {"update": update, "path": str(path), "bytes": path.stat().st_size,
                      "sha256": sha(path), "sample_exposures": sampler.exposures}
            checkpoint_records.append(record)
            last_checkpoint_update = update
            write_json(run_dir / "CHECKPOINTS.json", checkpoint_records)
            return path

        def evaluate_at(update):
            nonlocal best
            directory = run_dir / "evaluations" / f"U{update:06d}"
            panel = {"update": update}
            panel["ema_val50"] = eval_panel(ema.ema_model, validation_pairs, device,
                                             directory / "ema_val50", "ema", config["sampling_seed"])
            if update in {0, config["updates"]}:
                panel["raw_val50"] = eval_panel(model, validation_pairs, device,
                                                 directory / "raw_val50", "raw", config["sampling_seed"])
                panel["ema_val4"] = eval_panel(ema.ema_model, fixed_val4, device,
                                                directory / "ema_val4", "ema", config["sampling_seed"])
                panel["raw_val4"] = eval_panel(model, fixed_val4, device,
                                                directory / "raw_val4", "raw", config["sampling_seed"])
            evaluation_history.append(panel)
            psnr = panel["ema_val50"]["mean_metrics"]["psnr"]
            improved = best is None or psnr > best["ema_val50_psnr"]
            if improved:
                best = {"update": update, "ema_val50_psnr": psnr,
                        "selection_metric": "EMA Val50 original-float RGB mean PSNR",
                        "checkpoint": str(run_dir / "checkpoints" / f"U{update:06d}.pt")}
            write_json(run_dir / "EVALUATION_HISTORY.json", evaluation_history)
            if update in config["checkpoint_updates"] or improved:
                path = save_checkpoint(update)
                if improved:
                    best_path = run_dir / "checkpoints/best.pt"
                    if best_path.is_symlink():
                        best_path.unlink()
                    elif best_path.exists():
                        raise RuntimeError("Refusing to replace a non-symlink best checkpoint")
                    best_path.symlink_to(path.name)
                    write_json(run_dir / "BEST.json", best)
            print(json.dumps({"event": "evaluation", "update": update,
                              "ema_val50": panel["ema_val50"]["mean_metrics"], "best": best}, ensure_ascii=False), flush=True)
            return panel

        evaluate_at(0)
        optimizer.zero_grad(set_to_none=True)
        status["status"] = "RUNNING"
        write_json(status_path, status)
        write_json(run_dir / "STATUS.json", status)
        started_training = time.perf_counter()
        for update in range(1, config["updates"] + 1):
            status["attempting_update"] = update
            model.train()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            began = time.perf_counter()
            total_loss = 0.
            batch_ids = []
            sampled_indices = []
            probe_before = parameters[0].detach().clone()
            for _ in range(config["gradient_accumulate_every"]):
                pair = train_pairs[sampler.next()]
                clean = pair["clean"].to(device)
                observation = pair["observation"].to(device)
                batch_ids.append(pair["id"])
                with observe_training_indices(model) as indices:
                    losses = model([clean, observation, ["blur"]])
                sampled_indices.extend(indices)
                if not isinstance(losses, list) or len(losses) != 1:
                    raise RuntimeError("Native pred_res loss must return exactly one loss")
                loss = losses[0]
                if loss.dtype != torch.float32 or not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"Nonfinite or non-FP32 native loss at update {update}")
                total_loss += float(loss.detach().item()) / config["gradient_accumulate_every"]
                (loss / config["gradient_accumulate_every"]).backward()
                del clean, observation, losses, loss
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError(f"Missing parameter gradients at update {update}")
            gradients_finite = finite_all(parameter.grad for parameter in parameters)
            if not gradients_finite:
                raise RuntimeError(f"Nonfinite gradients at update {update}")
            preclip = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip_norm"])
            preclip_value = float(preclip.item())
            if not math.isfinite(preclip_value) or preclip_value <= 0:
                raise RuntimeError(f"Invalid gradient norm at update {update}: {preclip_value}")
            optimizer.step()
            if not finite_all(parameter.detach() for parameter in parameters):
                raise RuntimeError(f"Nonfinite parameters after optimizer update {update}")
            probe_change = float((parameters[0].detach() - probe_before).abs().max().item())
            del probe_before
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            torch.npu.synchronize()
            elapsed = time.perf_counter() - began
            row = {"update": update, "loss": total_loss, "preclip_grad_l2": preclip_value,
                   "gradients_finite": gradients_finite, "parameters_finite": True,
                   "first_parameter_update_max_abs": probe_change, "lr": optimizer.param_groups[0]["lr"],
                   "sample_ids": batch_ids, "diffusion_indices": sampled_indices,
                   "sample_exposures": sampler.exposures, "update_seconds": elapsed,
                   "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
                   "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
                   "allocated_bytes": int(torch.npu.memory_allocated()),
                   "reserved_bytes": int(torch.npu.memory_reserved()),
                   "ema_step": int(ema.step.item())}
            update_rows.append(row)
            with (run_dir / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            status.update({"optimizer_updates_completed": update, "sample_exposures": sampler.exposures,
                           "last_loss": total_loss, "last_preclip_grad_l2": preclip_value,
                           "last_checkpoint_update": last_checkpoint_update})
            if update % 10 == 0:
                write_json(status_path, status)
                write_json(run_dir / "STATUS.json", status)
                print(json.dumps({"event": "training", **row}, ensure_ascii=False), flush=True)
            if update in config["eval_updates"]:
                evaluate_at(update)
            elif update in config["checkpoint_updates"]:
                save_checkpoint(update)
        final_digest = state_digest(model.state_dict())
        if final_digest == initial_digest:
            raise RuntimeError("Pilot ended without any persistent model-state change")
        status.update({"status": "PASS", "optimizer_updates_completed": config["updates"],
                       "finished_unix": time.time(), "sample_exposures": sampler.exposures,
                       "last_checkpoint_update": last_checkpoint_update, "best": best,
                       "final_state_sha256": final_digest,
                       "initial_state_sha256": initial_digest,
                       "actual_training_coverage": actual_coverage(),
                       "evaluation_history": evaluation_history,
                       "training_update_seconds_sum": sum(row["update_seconds"] for row in update_rows),
                       "training_and_interleaved_evaluation_seconds": time.perf_counter() - started_training,
                       "last10_loss_mean": sum(row["loss"] for row in update_rows[-10:]) / 10,
                       "interpretation": "Bounded single-task fine-tuning feasibility only; not full Val1000, not Test, and not physical-recursion training"})
        write_json(run_dir / "RESULT.json", status)
        write_json(run_dir / "ACTUAL_TRAINING_COVERAGE.json", status["actual_training_coverage"])
        write_json(run_dir / "STATUS.json", status)
        write_json(status_path, status)
        print(json.dumps({"event": "complete", "status": "PASS", "updates": 100,
                          "sample_exposures": sampler.exposures, "best": best}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        status.update({"status": "FAIL", "finished_unix": time.time(), "error": repr(exc),
                       "traceback": traceback.format_exc(),
                       "policy": "Stopped; no automatic retry, budget extension, fallback model or scientific-definition change"})
        write_json(status_path, status)
        if run_dir is not None and run_dir.is_dir():
            write_json(run_dir / "STATUS.json", status)
            write_json(run_dir / "FAILURE.json", status)
            (run_dir / "FAILURE_TRACEBACK.txt").write_text(status["traceback"])
        raise


if __name__ == "__main__":
    main()

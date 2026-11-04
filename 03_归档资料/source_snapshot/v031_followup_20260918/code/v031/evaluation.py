"""Checkpoint-bound calibration and complete-prefix overfit diagnostics.

This module is called by a separate process. Calibration never loads GT arrays;
evaluation alone loads clean targets, and its metrics never select solver values.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from physical_bridge.simulator import Realization

from .data import StartDataset, read_jsonl
from .inference import _frozen_eval
from .refinement import (ITERATION_BUDGETS, LEARNING_RATES, RefinementFailure,
                         clone_xi, per_sample_objective, rank_candidates,
                         select_batched_initialization, solve_adam, synchronize)
from .runtime import save_json, sha256, state_hash


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze_calibration_policy(dataset_root, *, trajectory_manifest="subsets/overfit_16traj_trajectories.jsonl",
                              output=None):
    root = Path(dataset_root).resolve()
    source = root / trajectory_manifest
    ids = []
    for row in read_jsonl(source):
        trajectory = row["trajectory_id"]
        if trajectory not in ids:
            ids.append(trajectory)
        if len(ids) == 4:
            break
    if len(ids) != 4:
        raise ValueError("Frozen trajectory file has fewer than four unique trajectories")
    starts = read_jsonl(root / "start_manifest.jsonl")
    selected = []
    for trajectory in ids:
        for severity in (4, 3, 2):
            found = [row for row in starts if row["trajectory_id"] == trajectory
                     and row["initial_G"] == severity]
            if len(found) != 1:
                raise ValueError("Expected one authoritative start for each trajectory/G")
            selected.append(found[0]["start_id"])
    policy = {"format": "v031_calibration_policy_v1", "calibrate_once_at_update": 100,
              "checkpoint_weights": "raw", "dataset_root": str(root),
              "trajectory_manifest": trajectory_manifest, "trajectory_manifest_sha256": sha256(source),
              "start_manifest_sha256": sha256(root / "start_manifest.jsonl"),
              "trajectory_ids": ids, "start_ids": selected, "expected_probe_count": 24,
              "learning_rates": list(LEARNING_RATES), "iteration_budgets": list(ITERATION_BUDGETS),
              "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
              "objective": "original_observation_single_image_full_RGB_mean_MSE",
              "precision": "fp32", "same_lr_for_both_native_blocks": True,
              "optimizer_state": "reset_every_transition", "return": "best_including_initialization",
              "early_stopping": False, "clipping": False, "normalization": False,
              "projection": False, "clamp": False, "selection_fraction": 0.95,
              "timing_tie_fraction": 0.03, "integration_attempt_limit": 3,
              "low_signal_relative_improvement_threshold": 1e-6,
              "probe_initialization": "network_xi_from_fixed_unrefined_rollout",
              "selection_uses_GT": False}
    if output:
        path = Path(output)
        if path.exists() and json.loads(path.read_text()) != policy:
            raise ValueError("Existing frozen calibration policy differs")
        save_json(path, policy)
    return policy


def validate_policy(policy):
    expected = freeze_calibration_policy(policy["dataset_root"],
                                         trajectory_manifest=policy["trajectory_manifest"])
    if expected != policy:
        raise ValueError("Calibration policy/data identities differ from the frozen rule")


@dataclass
class Prediction:
    clean: torch.Tensor
    current_g: int
    network_redegradation_mse: list[float]
    refinement: list[dict] | None


def rollout_predictions(*, model, simulator, observation, initial_G, mode="unrefined",
                        solver_config=None, probe_callback=None, one_shot=False):
    if mode not in ("unrefined", "refined") or initial_G not in (1, 2, 3, 4):
        raise ValueError("Invalid rollout mode or initial severity")
    if observation.ndim != 4 or observation.shape[1] != 3:
        raise ValueError("Expected [B,3,H,W] observations")
    if mode == "refined" and initial_G > 1 and not one_shot and solver_config is None:
        raise ValueError("Refined inference requires the frozen update100 numeric configuration")
    original = observation.detach().float()
    current, previous, predictions = original, None, []
    path = (initial_G,) if one_shot else tuple(range(initial_G, 0, -1))
    with _frozen_eval(model, simulator):
        for index, current_g in enumerate(path):
            with torch.no_grad(), torch.autocast(device_type=current.device.type, enabled=False):
                prediction = model(current, torch.full((current.shape[0],), float(current_g),
                                                        device=current.device, dtype=torch.float32))
                clean = prediction.clean.detach().float()
                network_xi = clone_xi(prediction.realization)
                if not bool(torch.isfinite(clean).all()):
                    raise FloatingPointError("Nonfinite clean prediction")
            def objective(candidate, fixed_clean=clean):
                return per_sample_objective(simulator=simulator, current_clean=fixed_clean,
                                            realization=candidate, observation=original, initial_G=initial_G)
            with torch.no_grad():
                network_residual = objective(network_xi)
                if not bool(torch.isfinite(network_residual).all()):
                    raise FloatingPointError("Nonfinite predicted-xi re-degradation")
            refinements = None
            if index + 1 < len(path):
                if probe_callback:
                    probe_callback(index + 1, current_g, clean, network_xi, original, initial_G)
                transition_xi = network_xi
                if mode == "refined":
                    selected, sources, old_scores, new_scores = select_batched_initialization(
                        network_xi, previous, objective)
                    result = solve_adam(objective, selected,
                                        learning_rate=solver_config["learning_rate"],
                                        iterations=solver_config["iterations"])
                    previous, transition_xi = result.realization, result.realization
                    refinements = [{"initialization_source": source,
                                    "previous_candidate_objective": old_scores[b] if old_scores else None,
                                    "network_candidate_objective": new_scores[b] if new_scores else None,
                                    "selected_initial_objective": result.initial_objective[b],
                                    "best_objective": result.best_objective[b],
                                    "iterations": solver_config["iterations"],
                                    "batch_solver_seconds": result.elapsed_seconds,
                                    "gradient_connected": True, "finite": True,
                                    "nonzero_zernike_update": any(t["update_l2_zernike"][b] > 0 for t in result.trace),
                                    "nonzero_tilt_update": any(t["update_l2_tilt"][b] > 0 for t in result.trace)}
                                   for b, source in enumerate(sources)]
                with torch.no_grad(), torch.autocast(device_type=current.device.type, enabled=False):
                    current = simulator.render(clean, transition_xi, path[index + 1]).output.detach().float()
                    if not bool(torch.isfinite(current).all()):
                        raise FloatingPointError("Nonfinite predicted transition")
            predictions.append(Prediction(clean, current_g, network_residual.cpu().tolist(), refinements))
    return predictions


def clean_metrics(prediction, target):
    """Float RGB means; exact zero PSNR is represented by a flag and JSON null."""
    values = (prediction.detach().float() - target.detach().float()).square().flatten(1).mean(1)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite clean MSE")
    return [{"clean_mse": value, "clean_psnr_db": -10 * math.log10(value) if value > 0 else None,
             "zero_mse": value == 0, "psnr_infinite": value == 0}
            for value in values.cpu().tolist()]


def paired_gain(first, final, *, tie_psnr_db):
    if tie_psnr_db < 0 or not math.isfinite(tie_psnr_db):
        raise ValueError("Tie tolerance must be a predeclared finite nonnegative value")
    a, b = first["clean_psnr_db"], final["clean_psnr_db"]
    if first["zero_mse"] and final["zero_mse"]:
        gain, flag, outcome = 0.0, None, "tie"
    elif final["zero_mse"]:
        gain, flag, outcome = None, "positive_infinity", "win"
    elif first["zero_mse"]:
        gain, flag, outcome = None, "negative_infinity", "loss"
    else:
        gain, flag = b - a, None
        outcome = "tie" if abs(gain) <= tie_psnr_db else ("win" if gain > 0 else "loss")
    return {"psnr_gain_db": gain, "psnr_gain_nonfinite": flag,
            "mse_gain": first["clean_mse"] - final["clean_mse"], "outcome": outcome}


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _resolved_identity(config):
    # One canonical implementation owns actual code/data/numerical hashes for
    # both training resume and independent calibration/evaluation processes.
    from train_worker import resolved_identity
    return resolved_identity(config, StartDataset(config["dataset_root"]))


def _checkpoint_state(checkpoint, config, weights):
    if weights not in ("raw", "ema"):
        raise ValueError("Weights must explicitly be raw or ema")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "v031_distributed_update_boundary_v1":
        raise ValueError("Unknown checkpoint format")
    if (payload.get("formal_training") is not True or payload.get("smoke_optimizer_updates") != 0
            or payload.get("fixture_label") is not None or payload.get("fixture_mode") not in (None, False)):
        raise ValueError("Toy/smoke checkpoints are not formal evaluation checkpoints")
    if payload.get("identity") != _resolved_identity(config):
        raise ValueError("Checkpoint identity differs from the frozen training config")
    update = payload.get("global_update")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("Checkpoint requires a nonnegative integer formal global_update")
    if payload.get("completed_updates") != update:
        raise ValueError("Checkpoint update counters disagree")
    state = payload["network"] if weights == "raw" else payload["ema"]["network"]
    for value in state.values():
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError("Evaluation requires FP32 master or EMA weights")
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise ValueError("Evaluation checkpoint contains nonfinite state")
    return state, update


def _bound_progress(path, identity):
    if Path(path).exists():
        progress = json.loads(Path(path).read_text())
        if progress["identity"] != identity:
            raise ValueError("Partial/completed output belongs to a different checkpoint or policy")
        return progress
    return {"identity": identity, "status": "running"}


def validate_completed_calibration(calibration, config):
    """Completed means all fixed probes and twelve real integration starts exist."""
    if calibration.get("status") != "complete":
        raise ValueError("Calibration is not complete")
    identity = calibration["identity"]
    if (identity["training_identity"] != _resolved_identity(config) or identity["global_update"] != 100
            or identity["weights"] != "raw"):
        raise ValueError("Calibration is not from this run's formal raw update100")
    probes = calibration.get("probes", [])
    if len(probes) != 24 or len({p["probe_id"] for p in probes}) != 24:
        raise ValueError("Completed calibration lacks 24 unique probes")
    ranking = rank_candidates(probes)
    selected = calibration["selected"]
    eligible = [{"learning_rate": c["learning_rate"], "iterations": c["iterations"]}
                for c in ranking["ranking"][:3]]
    if selected not in eligible:
        raise ValueError("Selected solver is outside the allowed integration ranking")
    matches = [attempt for attempt in calibration.get("integration_attempts", [])
               if attempt.get("status") == "complete"
               and {k: attempt[k] for k in ("learning_rate", "iterations")} == selected]
    if len(matches) != 1 or len(matches[0].get("starts", [])) != 12:
        raise ValueError("Completed calibration lacks all twelve real refined integration starts")
    starts = matches[0]["starts"]
    if len({s["start_id"] for s in starts}) != 12 or not all(s["finite"] and s["network_unchanged"] for s in starts):
        raise ValueError("Calibration integration identities/finite/weight evidence is invalid")


def _observation(dataset, index, device):
    # Do NOT invoke __getitem__: calibration is forbidden to load clean GT.
    row = dataset.records[index]
    return dataset._array(row["initial_state_ref"], (3, 256, 256)).unsqueeze(0).to(device)


def calibrate(*, config, policy, checkpoint, model, simulator, device, output):
    session_started = time.perf_counter()
    validate_policy(policy)
    state, update = _checkpoint_state(checkpoint, config, "raw")
    if update != 100:
        raise ValueError("Calibration is only authorized at formal raw update100")
    model.load_state_dict(state, strict=True)
    del state
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": sha256(checkpoint), "policy_sha256": json_digest(policy),
                "training_identity": _resolved_identity(config), "weights": "raw", "global_update": update}
    path = root / "calibration.json"
    progress = _bound_progress(path, identity)
    if progress["status"] == "complete":
        validate_completed_calibration(progress, config)
        return progress  # Valid completed calibration is never searched again.
    dataset = StartDataset(config["dataset_root"])
    by_id = {row["start_id"]: index for index, row in enumerate(dataset.records)}
    network_hash = state_hash(model)
    simulator_hash = state_hash(simulator)
    progress.setdefault("probes", [])
    existing = {probe["probe_id"]: probe for probe in progress["probes"]}
    for start_id in policy["start_ids"]:
        index = by_id[start_id]
        row = dataset.records[index]
        expected = [f"{start_id}:K{k}" for k in range(1, row["initial_G"])]
        if all(identifier in existing for identifier in expected):
            continue
        def capture(k, current_g, clean, xi, observation, initial_g):
            probe_id = f"{start_id}:K{k}"
            if probe_id in existing:
                return
            filename = "probe_" + hashlib.sha256(probe_id.encode()).hexdigest()[:20] + ".pt"
            target = root / "probes" / filename
            _atomic_torch_save(target, {"clean": clean.cpu(), "zernike": xi.epsilon_zernike.cpu(),
                                        "tilt": xi.epsilon_tilt.cpu(), "observation": observation.cpu(),
                                        "original_G": initial_g})
            probe = {"probe_id": probe_id, "start_id": start_id,
                     "trajectory_id": row["trajectory_id"], "prediction_index": k,
                     "current_g": current_g, "original_G": initial_g,
                     "file": str(target), "sha256": sha256(target)}
            progress["probes"].append(probe)
            existing[probe_id] = probe
            save_json(path, progress)
        rollout_predictions(model=model, simulator=simulator, observation=_observation(dataset, index, device),
                            initial_G=row["initial_G"], probe_callback=capture)
    if len(progress["probes"]) != 24:
        raise RuntimeError("Expected all 24 nonfinal unrefined objective probes")
    for probe in progress["probes"]:
        if sha256(probe["file"]) != probe["sha256"]:
            raise ValueError("Saved probe identity changed")
        tensors = torch.load(probe["file"], map_location=device, weights_only=True)
        initial = Realization(tensors["zernike"], tensors["tilt"])
        def objective(candidate):
            return per_sample_objective(simulator=simulator, current_clean=tensors["clean"],
                                        realization=candidate, observation=tensors["observation"],
                                        initial_G=tensors["original_G"])
        for lr in LEARNING_RATES:
            if str(lr) in probe:
                continue
            with _frozen_eval(model, simulator):
                try:
                    result = solve_adam(objective, initial, learning_rate=lr, iterations=40)
                    record = {"prefixes": result.prefixes, "trace": result.trace, "failure": None}
                    del result
                except RefinementFailure as exc:
                    record = {"prefixes": exc.prefixes, "trace": exc.trace,
                              "failure": str(exc), "failed_update": exc.failed_update}
            record["network_unchanged"] = state_hash(model) == network_hash
            record["simulator_unchanged"] = state_hash(simulator) == simulator_hash
            if not record["simulator_unchanged"]:
                record["network_unchanged"] = False
            probe[str(lr)] = record
            save_json(path, progress)
        del tensors, initial
    try:
        progress["ranking"] = rank_candidates(progress["probes"], expected_probe_count=24,
                                               low_signal_threshold=policy["low_signal_relative_improvement_threshold"])
    except (RuntimeError, ValueError) as exc:
        progress["status"], progress["failure"] = "blocked_solver_candidates", str(exc)
        save_json(path, progress)
        raise
    progress.setdefault("integration_attempts", [])
    save_json(path, progress)
    for candidate in progress["ranking"]["ranking"][:3]:
        previous = next((x for x in progress["integration_attempts"] if x["learning_rate"] == candidate["learning_rate"]
                         and x["iterations"] == candidate["iterations"]), None)
        if previous and previous.get("status") == "failed":
            continue
        attempt = previous or {"learning_rate": candidate["learning_rate"], "iterations": candidate["iterations"],
                               "status": "running", "starts": []}
        if previous is None:
            progress["integration_attempts"].append(attempt)
        try:
            for start_id in policy["start_ids"]:
                if any(row["start_id"] == start_id for row in attempt["starts"]):
                    continue
                index = by_id[start_id]
                g = dataset.records[index]["initial_G"]
                synchronize(device)
                started = time.perf_counter()
                steps = rollout_predictions(model=model, simulator=simulator,
                    observation=_observation(dataset, index, device), initial_G=g,
                    mode="refined", solver_config=candidate)
                synchronize(device)
                if len(steps) != g or steps[-1].refinement is not None:
                    raise RuntimeError("Invalid full path or last-prediction refinement")
                if state_hash(model) != network_hash or state_hash(simulator) != simulator_hash:
                    raise RuntimeError("Network/simulator state changed during calibration")
                attempt["starts"].append({"start_id": start_id, "seconds": time.perf_counter() - started,
                                           "finite": True, "network_unchanged": True,
                                           "severity_path": [step.current_g for step in steps],
                                           "transitions": [step.refinement for step in steps[:-1]]})
                save_json(path, progress)
                del steps
            attempt["status"] = "complete"
            progress["status"] = "complete"
            progress["selected"] = {"learning_rate": candidate["learning_rate"], "iterations": candidate["iterations"]}
            progress["calibration_low_signal"] = progress["ranking"]["calibration_low_signal"]
            progress["completing_process_seconds"] = time.perf_counter() - session_started
            validate_completed_calibration(progress, config)
            save_json(path, progress)
            return progress
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            attempt["status"], attempt["failure"] = "failed", str(exc)
            save_json(path, progress)
    progress["status"] = "blocked_no_valid_integration"
    save_json(path, progress)
    raise RuntimeError("No candidate passed real refined integration in the allowed three attempts")


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_records(records, *, tie_psnr_db):
    curves, pairs = [], []
    for g in (4, 3, 2, 1):
        subset = [x for x in records if x["initial_G"] == g]
        if not subset:
            continue
        for k in range(1, g + 1):
            steps = [x["predictions"][k - 1] for x in subset]
            finite_psnr = [x["clean_psnr_db"] for x in steps if not x["zero_mse"]]
            zero_count = sum(x["zero_mse"] for x in steps)
            curves.append({"initial_G": g, "K": k, "count": len(steps),
                           "mean_clean_mse": _mean([x["clean_mse"] for x in steps]),
                           "mean_clean_psnr_db": _mean(finite_psnr) if not zero_count else None,
                           "psnr_mean_infinite": bool(zero_count), "zero_mse_count": zero_count,
                           "finite_only_mean_psnr_db": _mean(finite_psnr),
                           "mean_network_redegradation_mse": _mean([x["network_redegradation_mse"] for x in steps])})
        gains = [x["full_minus_one_shot"] for x in subset] if g > 1 else []
        finite = [x["psnr_gain_db"] for x in gains if x["psnr_gain_db"] is not None]
        pairs.append({"initial_G": g, "count": len(subset), "multistep_applicable": g > 1,
                      "mean_psnr_gain_db": _mean(finite) if len(finite) == len(gains) else None,
                      "nonfinite_gain_count": len(gains) - len(finite),
                      "win": sum(x["outcome"] == "win" for x in gains),
                      "tie": sum(x["outcome"] == "tie" for x in gains),
                      "loss": sum(x["outcome"] == "loss" for x in gains)})
    return {"G_curves": curves, "paired_full_minus_one_shot": pairs, "tie_psnr_db": tie_psnr_db,
            "metric_policy": "original_float_RGB_mean_MSE; per_image_PSNR_data_range_1_then_average",
            "scope": "overfit training-data diagnostics; G1 has no multistep gain"}


def _preview(path, observation, target, predictions):
    from PIL import Image, ImageDraw
    images = [observation, target] + [step.clean[0] for step in predictions]
    labels = ["original Y", "clean GT"] + [f"K={k}" for k in range(1, len(predictions) + 1)]
    arrays = [(value.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
              for value in images]
    height, width = arrays[0].shape[:2]
    canvas = Image.new("RGB", (width * len(arrays), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(arrays, labels)):
        canvas.paste(Image.fromarray(array), (width * index, 24))
        draw.text((width * index + 4, 4), label, fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(*, config, checkpoint, weights, manifest, model, simulator, device, output,
             batch_size=1, modes=("unrefined", "refined"), calibration=None):
    session_started = time.perf_counter()
    state, update = _checkpoint_state(checkpoint, config, weights)
    model.load_state_dict(state, strict=True)
    del state
    if update == 0 and modes != ("unrefined",):
        raise ValueError("Update0 records only unrefined/one-shot baseline")
    if "refined" in modes:
        if not calibration:
            raise ValueError("Refined evaluation requires completed update100 calibration")
        validate_completed_calibration(calibration, config)
        if update < 100:
            raise ValueError("Refined evaluation is only defined from update100 onwards")
    tie = config["evaluation"]["tie_psnr_db"]
    # Validation of the explicit tie occurs even if no starts have G>1.
    if not math.isfinite(tie) or tie < 0:
        raise ValueError("Invalid frozen PSNR tie tolerance")
    dataset = StartDataset(config["dataset_root"], manifest)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": sha256(checkpoint), "weights": weights, "global_update": update,
                "manifest_sha256": sha256(dataset._safe_path(manifest)), "training_identity": _resolved_identity(config),
                "calibration_identity": calibration["identity"] if calibration else None,
                "numeric_config": calibration["selected"] if calibration else None,
                "modes": list(modes), "tie_psnr_db": tie,
                "fixed_preview_start_ids": config["evaluation"]["fixed_preview_start_ids"]}
    progress_path = root / "progress.json"
    progress = _bound_progress(progress_path, identity)
    if progress["status"] == "complete":
        return json.loads((root / "summary.json").read_text())
    progress.setdefault("batch_size_events", [])
    progress.setdefault("completed_batches", [])
    save_json(progress_path, progress)
    before = state_hash(model)
    simulator_before = state_hash(simulator)
    summaries = {}
    preview_ids = set(config["evaluation"]["fixed_preview_start_ids"])
    for mode in modes:
        row_dir = root / mode / "rows"
        row_dir.mkdir(parents=True, exist_ok=True)
        def row_path(start_id):
            return row_dir / (hashlib.sha256(start_id.encode()).hexdigest() + ".json")
        for g in (4, 3, 2, 1):
            pending = [i for i, row in enumerate(dataset.records)
                       if row["initial_G"] == g and not row_path(row["start_id"]).exists()]
            cursor = 0
            effective_batch = batch_size
            while cursor < len(pending):
                indices = pending[cursor:cursor + effective_batch]
                try:
                    batch_started = time.perf_counter()
                    samples = [dataset[i] for i in indices]
                    observations = torch.stack([x["observation"] for x in samples]).to(device)
                    targets = torch.stack([x["clean"] for x in samples]).to(device)
                    synchronize(device)
                    started = time.perf_counter()
                    steps = rollout_predictions(model=model, simulator=simulator, observation=observations,
                        initial_G=g, mode=mode, solver_config=calibration["selected"] if calibration else None)
                    synchronize(device)
                    seconds = time.perf_counter() - started
                    if state_hash(model) != before or state_hash(simulator) != simulator_before:
                        raise RuntimeError("Evaluation changed network/simulator state before committing rows")
                    per_step = [clean_metrics(step.clean, targets) for step in steps]
                    for b, sample in enumerate(samples):
                        predictions = []
                        for k, step in enumerate(steps):
                            predictions.append({"K": k + 1, "current_g": step.current_g, **per_step[k][b],
                                "network_redegradation_mse": step.network_redegradation_mse[b],
                                "refinement": step.refinement[b] if step.refinement else None})
                        record = {"start_id": sample["start_id"], "trajectory_id": sample["trajectory_id"],
                                  "initial_G": g, "mode": mode, "weights": weights,
                                  "global_update": update, "predictions": predictions,
                                  "full_minus_one_shot": paired_gain(predictions[0], predictions[-1], tie_psnr_db=tie)
                                  if g > 1 else None, "batch_seconds": seconds, "batch_size": len(samples)}
                        if sample["start_id"] in preview_ids:
                            single = [Prediction(s.clean[b:b + 1], s.current_g, [], None) for s in steps]
                            filename = hashlib.sha256(sample["start_id"].encode()).hexdigest()[:20] + ".png"
                            _preview(root / mode / "previews" / filename, observations[b], targets[b], single)
                        save_json(row_path(sample["start_id"]), record)
                    cursor += len(indices)
                    progress["completed_batches"].append({"mode": mode, "G": g, "batch_size": len(indices),
                                                           "seconds": seconds,
                                                           "batch_wall_seconds": time.perf_counter() - batch_started})
                    save_json(progress_path, progress)
                    del steps, observations, targets, per_step
                except (RuntimeError, FloatingPointError, ValueError) as exc:
                    if "out of memory" not in str(exc).lower() or effective_batch <= 1:
                        progress["status"], progress["failure"] = "blocked", str(exc)
                        save_json(progress_path, progress)
                        raise
                    old = effective_batch
                    effective_batch = max(1, effective_batch // 2)
                    progress["batch_size_events"].append({"mode": mode, "G": g, "reason": "evaluation_OOM",
                                                           "from": old, "to": effective_batch})
                    save_json(progress_path, progress)
                    # A wrapped RefinementFailure keeps its original exception,
                    # whose traceback owns native xi, optimizer and graph frames.
                    # Release that chain BEFORE collecting/emptying the cache.
                    import gc
                    import traceback
                    exception_chain = exc
                    seen_exceptions = set()
                    while exception_chain is not None and id(exception_chain) not in seen_exceptions:
                        seen_exceptions.add(id(exception_chain))
                        next_exception = exception_chain.__cause__ or exception_chain.__context__
                        traceback.clear_frames(exception_chain.__traceback__)
                        exception_chain.__traceback__ = None
                        exception_chain.__cause__ = None
                        exception_chain.__context__ = None
                        exception_chain = next_exception
                    steps = observations = targets = per_step = single = step = None
                    gc.collect()
                    backend = getattr(torch, device.type, None)
                    if backend is not None and hasattr(backend, "empty_cache"):
                        backend.empty_cache()
        records = [json.loads(row_path(row["start_id"]).read_text()) for row in dataset.records]
        if len(records) != len(dataset):
            raise RuntimeError("Incomplete evaluation coverage")
        temporary = root / mode / "per_start.jsonl.tmp"
        with temporary.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        os.replace(temporary, root / mode / "per_start.jsonl")
        summaries[mode] = summarize_records(records, tie_psnr_db=tie)
        save_json(root / mode / "G_curves.json", summaries[mode]["G_curves"])
    if state_hash(model) != before or state_hash(simulator) != simulator_before:
        raise RuntimeError("Evaluation changed network/simulator parameters or buffers")
    summary = {"identity": identity, "status": "complete", "start_count_per_mode": len(dataset),
               "network_unchanged": True, "simulator_unchanged": True, "modes": summaries,
               "batch_size_events": progress["batch_size_events"],
               "measured_rollout_seconds": sum(x["seconds"] for x in progress["completed_batches"]),
               "measured_batch_wall_seconds": sum(x["batch_wall_seconds"] for x in progress["completed_batches"]),
               "completing_process_seconds": time.perf_counter() - session_started,
               "timing_scope": "Batch wall time includes reads/transfers/rollout/metrics/state checks/previews; model construction excluded"}
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "max_memory_allocated"):
        summary["peak_allocated_bytes"] = backend.max_memory_allocated()
    save_json(root / "summary.json", summary)
    progress["status"] = "complete"
    save_json(progress_path, progress)
    return summary

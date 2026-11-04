#!/usr/bin/env python3
"""Independent-process V0.31 calibration/evaluation; never restores training RNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "calibrate", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--weights", "--raw-or-ema", choices=("raw", "ema"), default="raw")
    parser.add_argument("--manifest", default="subsets/overfit_16traj_starts.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--numeric-config", help="Completed calibration.json; never an arbitrary LR/steps override")
    parser.add_argument("--rollout-modes", choices=("unrefined", "refined", "both"), default="both")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = json.loads(Path(args.config).read_text())
    sys.path.insert(0, config["base_source"])
    import torch
    from v031.runtime import build, configure_npu, save_json, state_hash
    from v031.evaluation import (_checkpoint_state, calibrate, evaluate, freeze_calibration_policy,
                                 rollout_predictions, validate_policy)
    from v031.refinement import (clone_xi, per_sample_objective, solve_adam,
                                 select_batched_initialization, synchronize)
    from v031.data import StartDataset
    from v031.inference import _frozen_eval
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        configure_npu(config)
    device = torch.device(args.device)
    model, simulator = build(config, device)
    if args.mode == "calibrate":
        if not args.checkpoint or not args.policy or args.weights != "raw":
            parser.error("Calibration requires --checkpoint raw update100 and --policy")
        result = calibrate(config=config, policy=json.loads(Path(args.policy).read_text()),
                           checkpoint=args.checkpoint, model=model, simulator=simulator,
                           device=device, output=args.output)
    elif args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("Evaluation requires an explicit --checkpoint")
        modes = ("unrefined", "refined") if args.rollout_modes == "both" else (args.rollout_modes,)
        calibration = json.loads(Path(args.numeric_config).read_text()) if args.numeric_config else None
        result = evaluate(config=config, checkpoint=args.checkpoint, weights=args.weights,
                          manifest=args.manifest, model=model, simulator=simulator, device=device,
                          output=args.output, batch_size=args.batch_size, modes=modes, calibration=calibration)
    else:
        # Limited real-operator mechanism check at one prespecified numeric
        # point. This does not freeze selected inference numerics or use GT.
        if args.checkpoint:
            state, _ = _checkpoint_state(args.checkpoint, config, args.weights)
            model.load_state_dict(state, strict=True)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        policy = freeze_calibration_policy(config["dataset_root"], output=args.policy or output / "policy.json")
        validate_policy(policy)
        dataset = StartDataset(config["dataset_root"], args.manifest)
        indices = [i for i, row in enumerate(dataset.records) if row["initial_G"] == 4][:2]
        if len(indices) != 2:
            raise ValueError("Solver smoke requires two fixed G4 starts")
        if args.batch_size not in (1, 2):
            parser.error("Solver smoke supports only --batch-size 1 or 2")
        observation = torch.stack([dataset._array(dataset.records[i]["initial_state_ref"], (3, 256, 256))
                                   for i in indices]).to(device)
        from physical_bridge.simulator import Realization
        from v031.refinement import RefinementFailure
        before_model, before_simulator = state_hash(model), state_hash(simulator)
        comparison_kind = "batched_vs_individual" if args.batch_size == 2 else "repeated_single_image"
        result = {"status": "running", "scope": "solver mechanism smoke; no inference selection or quality claim",
                  "learning_rate": 1e-3, "iterations": 10, "batch_size": args.batch_size,
                  "comparison_kind": comparison_kind, "allclose_rtol": 1e-4, "allclose_atol": 1e-5,
                  "start_ids": [dataset.records[i]["start_id"] for i in indices],
                  "policy": policy, "solver_runs": [], "batch_vs_individual": [],
                  "full_refined_runs": [], "formal_training_updates": 0, "restoration_optimizer_updates": 0}
        smoke_path = output / "smoke.json"
        save_json(smoke_path, result)

        def record_run(value, label, start_ids):
            # The trace contains only scalar/list summaries for both native
            # blocks; never serialize native tensors or an autograd graph.
            record = {"label": label, "start_ids": start_ids,
                      "initial_objective": value.initial_objective,
                      "best_objective": value.best_objective,
                      "elapsed_seconds": value.elapsed_seconds, "trace": value.trace,
                      "finite": all(t["finite"] for t in value.trace),
                      "gradient_connected": all(t["gradient_connected"] for t in value.trace),
                      "nonzero_update_zernike": [any(t["update_l2_zernike"][b] > 0 for t in value.trace)
                                                  for b in range(len(start_ids))],
                      "nonzero_update_tilt": [any(t["update_l2_tilt"][b] > 0 for t in value.trace)
                                               for b in range(len(start_ids))]}
            result["solver_runs"].append(record)
            save_json(smoke_path, result)

        def compare(candidate, candidate_index, reference, start_id):
            difference = {"start_id": start_id, "comparison_kind": comparison_kind,
                          "individual_solver_seconds": reference.elapsed_seconds,
                          "candidate_initial_objective": candidate.initial_objective[candidate_index],
                          "reference_initial_objective": reference.initial_objective[0],
                          "candidate_best_objective": candidate.best_objective[candidate_index],
                          "reference_best_objective": reference.best_objective[0]}
            for objective_name in ("initial_objective", "best_objective"):
                a = getattr(candidate, objective_name)[candidate_index]
                b = getattr(reference, objective_name)[0]
                difference[objective_name + "_absolute_difference"] = abs(a - b)
                difference[objective_name + "_relative_difference"] = abs(a - b) / max(abs(b), 1e-12)
            for name in ("epsilon_zernike", "epsilon_tilt"):
                x = getattr(candidate.realization, name)[candidate_index:candidate_index + 1]
                y = getattr(reference.realization, name)
                delta = x - y
                maximum = float(delta.abs().max().cpu())
                denominator = float(y.norm().cpu())
                relative_l2 = float(delta.norm().cpu()) / max(denominator, 1e-12)
                difference[name + "_max_abs"] = maximum
                difference[name + "_relative_l2"] = relative_l2
                difference[name + "_allclose"] = bool(torch.allclose(x, y, rtol=1e-4, atol=1e-5))
                difference[name + "_reference_l2"] = denominator
            difference["passed"] = all(difference[name + "_allclose"] for name in ("epsilon_zernike", "epsilon_tilt"))
            result["batch_vs_individual"].append(difference)
            # Store both blocks and all objective differences BEFORE any gate.
            save_json(smoke_path, result)

        try:
            candidate_seconds = []
            with _frozen_eval(model, simulator):
                for offset in range(0, 2, args.batch_size):
                    current_observation = observation[offset:offset + args.batch_size]
                    current_ids = result["start_ids"][offset:offset + args.batch_size]
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                        prediction = model(current_observation, torch.full((len(current_ids),), 4.0, device=device))
                        clean, xi = prediction.clean.detach().float(), clone_xi(prediction.realization)
                    def objective(candidate):
                        return per_sample_objective(simulator=simulator, current_clean=clean,
                                                    realization=candidate, observation=current_observation, initial_G=4)
                    candidate = solve_adam(objective, xi, learning_rate=1e-3, iterations=10)
                    candidate_seconds.append(candidate.elapsed_seconds)
                    record_run(candidate, "candidate", current_ids)
                    for b, start_id in enumerate(current_ids):
                        initial = Realization(xi.epsilon_zernike[b:b + 1], xi.epsilon_tilt[b:b + 1])
                        reference = solve_adam(lambda value: per_sample_objective(
                            simulator=simulator, current_clean=clean[b:b + 1], realization=value,
                            observation=current_observation[b:b + 1], initial_G=4),
                            initial, learning_rate=1e-3, iterations=10)
                        record_run(reference, "individual_reference", [start_id])
                        compare(candidate, b, reference, start_id)
                        del reference
                    del candidate, prediction, clean, xi
            result["candidate_solver_seconds"] = sum(candidate_seconds)
            result["batched_solver_seconds"] = sum(candidate_seconds) if args.batch_size == 2 else None
            result["total_individual_solver_seconds"] = sum(x["individual_solver_seconds"] for x in result["batch_vs_individual"])
            result["comparison_passed"] = all(x["passed"] for x in result["batch_vs_individual"])
            result["network_simulator_unchanged"] = state_hash(model) == before_model and state_hash(simulator) == before_simulator
            result["evaluation_batch_decision"] = (
                "eligible_pending_full_integration" if result["comparison_passed"] else
                "exclude_batch2_keep_tolerances" if args.batch_size == 2 else "block_single_image_consistency")
            save_json(smoke_path, result)
            if not result["comparison_passed"]:
                result["full_refined_status"] = "not_run_comparison_failed"
                raise RuntimeError("Solver consistency check failed at frozen rtol=1e-4 / atol=1e-5; see smoke.json")
            if not result["network_simulator_unchanged"]:
                raise RuntimeError("Solver smoke changed network/simulator state")
            # Full integration follows the candidate's actual evaluation batch.
            full_refined_seconds = 0.0
            for offset in range(0, 2, args.batch_size):
                synchronize(device)
                full_refined_started = time.perf_counter()
                steps = rollout_predictions(model=model, simulator=simulator,
                    observation=observation[offset:offset + args.batch_size], initial_G=4, mode="refined",
                    solver_config={"learning_rate": 1e-3, "iterations": 10})
                synchronize(device)
                elapsed = time.perf_counter() - full_refined_started
                full_refined_seconds += elapsed
                result["full_refined_runs"].append({
                    "start_ids": result["start_ids"][offset:offset + args.batch_size], "seconds": elapsed,
                    "full_path": [step.current_g for step in steps],
                    "transitions": [step.refinement for step in steps[:-1]],
                    "network_redegradation_mse": [step.network_redegradation_mse for step in steps]})
                del steps
                save_json(smoke_path, result)
            result["full_refined_two_G4_seconds"] = full_refined_seconds
            result["full_path"] = [4, 3, 2, 1]
            result["network_simulator_unchanged"] = state_hash(model) == before_model and state_hash(simulator) == before_simulator
            if not result["network_simulator_unchanged"]:
                raise RuntimeError("Full refined smoke changed network/simulator state")
            result["status"] = "passed"
            result["full_refined_status"] = "passed"
            result["evaluation_batch_decision"] = f"batch{args.batch_size}_mechanism_accepted"
            save_json(smoke_path, result)
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            result["status"], result["failure"] = "failed", str(exc)
            result["network_simulator_unchanged"] = state_hash(model) == before_model and state_hash(simulator) == before_simulator
            if isinstance(exc, RefinementFailure):
                result["failed_solver_trace"] = exc.trace
                result["failed_solver_prefixes"] = exc.prefixes
                result["failed_solver_update"] = exc.failed_update
            save_json(smoke_path, result)
            raise
    print(json.dumps({"status": result["status"], "output": args.output}, allow_nan=False))


if __name__ == "__main__":
    main()

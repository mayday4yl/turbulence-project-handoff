#!/usr/bin/env python3
"""Four-rank HCCL worker; resource guards and calibration belong to the launcher.

Examples (through the authorized guarded distributed environment):
  python -m torch.distributed.run --nproc_per_node=4 code/train_worker.py \
    --mode performance-smoke --config configs/candidate.json --output runs/mb2 \
    --microbatch 2 --warmup-updates 1 --measure-updates 3
  ... --mode train --config configs/train_v031.json --output runs/formal \
    --microbatch 2 --run-until-update 0
  ... --mode train --config configs/train_v031.json --output runs/formal \
    --microbatch 2 --run-until-update 100 --resume runs/formal/checkpoint_000000.pt \
    --authorize-boundary-extension
  ... --mode train --config configs/train_v031.json --output runs/formal \
    --microbatch 2 --run-until-update 200 --resume runs/formal/checkpoint_000100.pt \
    --authorize-boundary-extension

Smoke uses independent fresh output directories, seed, sampler and weights; its
updates never become formal progress. resume-check executes N prefix updates,
saves, executes M continuation updates, reconstructs/restores and repeats M;
physical smoke budget is N+2M, not N+M. Defaults are N=1, M=1 for that mode.

Boundary 0 saves the frozen initialized network/EMA and complete RNG/sampler
state for a separate baseline process, then exits without optimizer updates.
After evaluating it, use --resume checkpoint_000000.pt and an explicitly
authorized boundary extension to 100. To continue beyond 1000 in the future,
the config must contain continuation_authorization with explicit_user_authorization
true, an authorization_id, and authorized_run_until_update matching the new
training.run_until_update. The resume/extension flag is still mandatory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from v031.checkpoint import _cpu_snapshot
from v031.data import StartDataset
from v031.distributed_runtime import configure_local_rank
from v031.runtime import save_json, sha256, state_hash, verify_source
from v031.training import (TrainingEngine, authorized_update_limit, capture_rank_rng, compare_states,
                           load_training_checkpoint, memory_stats,
                           save_training_checkpoint, synchronize)


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["memory-smoke", "performance-smoke", "resume-check", "train"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--measure-updates", type=int, default=None)
    parser.add_argument("--run-until-update", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--authorize-boundary-extension", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = config["training"]
    fixed = {"optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
             "weight_decay": 1e-4, "clip_grad_norm": 1.0, "ema_enabled": False,
             "parameter_group_rule": "bias_and_normalization_no_decay",
             "precision": "fp32",
             "world_size": 4, "effective_batch": 32, "microbatch": args.microbatch,
             "accumulation": 8 // args.microbatch, "checkpoint_every": 100,
             "allow_automatic_updates_beyond_1000": False}
    for key, value in fixed.items():
        if training.get(key) != value:
            raise ValueError(f"training.{key} must explicitly equal {value!r}")
    if config.get("seed") != 20260916 or config.get("lambda_R") != (0.0 if config["task"] == "T1_DIRECT" else 0.2):
        raise ValueError("Training seed and lambda_R are frozen")
    if training.get("sampler_seed", 20260916) != 20260916 or training.get("train_seed", 20260916) != 20260916:
        raise ValueError("Sampler/train seeds differ from authority")
    schedule = config.get("scheduler", {})
    schedule_values = {"optimizer": "AdamW", "peak_lr": 1e-4, "warmup_updates": 50,
                       "cosine_end_update": 1000, "floor_lr": 1e-5, "after_1000": "constant"}
    for key, value in schedule_values.items():
        if key in schedule and schedule[key] != value:
            raise ValueError(f"scheduler.{key} conflicts with frozen update-indexed schedule")
    if not isinstance(config.get("identity"), dict) or not config["identity"]:
        raise ValueError("Explicit nonempty code/config/data identity required")
    authorized_limit = authorized_update_limit(config.get("continuation_authorization"))
    if training.get("run_until_update") != authorized_limit:
        raise ValueError("Configured run limit differs from explicit authorization")
    if args.mode == "train":
        if config.get("mode") != "FOLLOWUP_TO_1000":
            raise ValueError("Formal training requires explicit AUTO_TO_1000 mode")
        for gate in ["targeted_acceptance_passed", "training_config_frozen"]:
            if config.get("gates", {}).get(gate) is not True:
                raise ValueError(f"Formal launch gate is not true: {gate}")
        if args.run_until_update is None or not 0 <= args.run_until_update <= authorized_limit:
            raise ValueError("Formal run requires a bounded --run-until-update within authorization")
        if authorized_limit > 1000 and (args.resume is None or not args.authorize_boundary_extension):
            raise ValueError("Future authorized continuation requires --resume and --authorize-boundary-extension")
        if args.run_until_update % 100:
            raise ValueError("Formal segment boundary must be a fixed 100-update diagnostic boundary")
        if args.warmup_updates is not None or args.measure_updates is not None:
            raise ValueError("Smoke counters are forbidden for formal training")
    else:
        if args.resume is not None or args.run_until_update is not None or args.authorize_boundary_extension:
            raise ValueError("Smoke always starts fresh; resume-check owns its isolated checkpoint")
        args.warmup_updates = 1 if args.warmup_updates is None else args.warmup_updates
        args.measure_updates = (1 if args.mode == "resume-check" else 3) if args.measure_updates is None else args.measure_updates
        if args.warmup_updates < 1 or args.measure_updates < 1:
            raise ValueError("Smoke must include at least one warmup and measured update")
        if args.mode == "performance-smoke" and args.measure_updates < 3:
            raise ValueError("Performance ranking requires at least three measured updates")
        calls = args.warmup_updates + args.measure_updates * (2 if args.mode == "resume-check" else 1)
        if calls > 30:
            raise ValueError("One invocation already exceeds the total allowed smoke budget")


def resolved_identity(config: dict[str, Any], dataset: StartDataset) -> dict[str, Any]:
    """Bind actual worker/method bytes and manifests, plus the supplied audit IDs."""
    code_paths = [ROOT / "code/train_worker.py"] + sorted((ROOT / "code/v031").glob("*.py"))
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    if "code_hashes" in config and config["code_hashes"] != code_hashes:
        raise ValueError("Actual training code differs from frozen code_hashes")
    manifests = {name: sha256(dataset.root / name) for name in
                 ["source_manifest.jsonl", "trajectory_manifest.jsonl", "start_manifest.jsonl"]}
    if "data_manifest_hashes" in config and config["data_manifest_hashes"] != manifests:
        raise ValueError("Actual manifests differ from frozen data hashes")
    asset_dir = Path(config["asset_dir"]).resolve(strict=True)
    asset_hashes = {name: sha256(asset_dir / name) for name in
                    ["P2S_model.pt", "dictionary.pt", "I0_I2.pt"]}
    if "simulator_asset_hashes" in config and config["simulator_asset_hashes"] != asset_hashes:
        raise ValueError("Simulator assets differ from frozen Stage 1 hashes")
    # Stage flags and authorized run length can evolve separately from the
    # immutable numerical configuration; they must still pass launch validation.
    numerical = {key: config[key] for key in ["seed", "lambda_R", "network", "physics", "npu_execution"]}
    numerical["scheduler"] = config.get("scheduler", {})
    numerical["training"] = {key: value for key, value in config["training"].items()
                              if key not in {"run_until_update", "allow_automatic_updates_beyond_1000"}}
    return {"declared": config["identity"], "actual_code_sha256": code_hashes,
            "actual_data_manifests_sha256": manifests,
            "actual_simulator_assets_sha256": asset_hashes, "asset_dir": str(asset_dir),
            "numerical_config_sha256": hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest(),
            "dataset_root": str(dataset.root), "base_hashes": config["base_hashes"],
            "data_hash_manifest_sha256": config["data_hash_manifest_sha256"]}


class SelectedStarts:
    """Read-only G=4 view exclusively for longest-path pressure smoke."""
    def __init__(self, dataset: StartDataset) -> None:
        self.dataset = dataset
        self.indices = [i for i, record in enumerate(dataset.records) if record["initial_G"] == 4]
        self.records = [dataset.records[i] for i in self.indices]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


def make_engine(config: dict[str, Any], dataset: Any, identity: dict[str, Any],
                args: argparse.Namespace, local_rank: int) -> TrainingEngine:
    # The original constructors and initialization are unchanged. Seed only this
    # rank's device rather than runtime.seed_all()'s all-visible-device path.
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    torch.npu.manual_seed(seed)
    device = torch.device(f"npu:{local_rank}")
    if 'FOLLOWUP_OWN_CAP_GIB' not in os.environ:
        raise RuntimeError('Training requires an explicit shared-resource memory cap')
    torch.npu.set_per_process_memory_fraction(float(os.environ['FOLLOWUP_OWN_CAP_GIB']) / 64, device)
    from v031.followup import CleanOnly
    network = DualHeadUNet(**config["network"])
    if state_hash(network) != config["expected_full_initial_network_sha256"]:
        raise ValueError("Full common constructor initialization mismatch")
    shared = torch.load(config["initial_state_path"], map_location="cpu", weights_only=True)
    network.load_state_dict(shared, strict=True)
    if state_hash(network) != config["expected_full_initial_network_sha256"]:
        raise ValueError("Shared saved initialization mismatch")
    if config["task"] == "T1_DIRECT":
        network = CleanOnly(network)
        simulator = torch.nn.Identity().to(device).eval().requires_grad_(False)
    else:
        simulator = PiRNP2SSimulator(config["asset_dir"], **config["physics"], device=device)
        simulator.eval().requires_grad_(False)
    network.followup_task = config["task"]
    network = network.to(device)
    return TrainingEngine(network, simulator, dataset, device=device, microbatch=args.microbatch,
                          rank=local_rank, world_size=4, seed=seed, identity=identity,
                          formal=args.mode == "train")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resume_snapshot(engine: TrainingEngine) -> dict[str, Any]:
    return {"network": _cpu_snapshot(engine.network.state_dict()),
            "ema_enabled": False,
            "optimizer": _cpu_snapshot(engine.optimizer.state_dict()),
            "scheduler": deepcopy(engine.scheduler.state_dict()),
            "sampler": _cpu_snapshot(engine.sampler.state_dict()),
            "rng": capture_rank_rng(engine.device), "counts": deepcopy(engine.counts)}


def compare_resume_reexecution(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Exact state restoration is checked separately, before recomputation.

    Native repeated FP32 backward is not bitwise deterministic on this stack.
    Recomputed floating weights/moments therefore have a declared numerical
    gate. LR, optimizer step, EMA counts, RNG, sampler and every other field
    remain exact. Preserve the bitwise result; never relabel it as exact.
    """
    import math
    result = compare_states(reference, candidate)
    epsilon = torch.finfo(torch.float32).eps
    failures = []
    totals = {name: [0.0, 0.0] for name in ["network", "ema", "optimizer_exp_avg", "optimizer_exp_avg_sq"]}

    def category(path):
        if path.startswith("state.network."):
            return "network"
        if path.startswith("state.ema.network."):
            return "ema"
        if path.startswith("state.optimizer.state.") and path.rsplit(".", 1)[-1] in {"exp_avg", "exp_avg_sq"}:
            return "optimizer_" + path.rsplit(".", 1)[-1]
        return None

    def visit(left, right, path):
        if torch.is_tensor(left):
            if not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
                failures.append(path)
                return
            group = category(path)
            if group is None or not left.is_floating_point():
                if not torch.equal(left, right):
                    failures.append(path)
                return
            if left.dtype != torch.float32 or not (torch.isfinite(left).all() and torch.isfinite(right).all()):
                failures.append(path)
                return
            a, b = left.double(), right.double()
            totals[group][0] += float(a.square().sum())
            totals[group][1] += float((a - b).square().sum())
            if not torch.allclose(left, right, atol=epsilon, rtol=8 * epsilon):
                failures.append(path)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                failures.append(path)
                return
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                failures.append(path)
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{index}]")
        elif left != right:
            failures.append(path)

    visit(reference, candidate, "state")
    relative = {}
    for name, (norm2, difference2) in totals.items():
        value = math.sqrt(difference2 / norm2) if norm2 else (0.0 if not difference2 else None)
        relative[name] = value
        if value is None or value > 8 * epsilon:
            failures.append(f"state.{name}.relative_l2")
    result.update(passed=not failures, criterion="exact_restore_then_bounded_fp32_reexecution_v1",
                  numerical_tolerance={"atol": epsilon, "rtol": 8 * epsilon,
                                       "group_relative_l2_max": 8 * epsilon},
                  exact_fields="all except FP32 network, EMA network and AdamW exp_avg/exp_avg_sq",
                  group_relative_l2=relative, failed_acceptance_fields=failures)
    return result


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, args)
    local_rank = int(os.environ["LOCAL_RANK"])
    output = args.output.resolve()
    # The parent directory must be fresh for each smoke or first formal launch.
    # A resumed formal run is explicitly allowed to append to its same output.
    if local_rank == 0:
        if args.resume:
            if not output.is_dir():
                raise ValueError("Resume output directory does not exist")
        else:
            output.mkdir(parents=True, exist_ok=False)
    execution = configure_local_rank(config, local_rank)
    verify_source(config)
    dist.init_process_group("hccl", timeout=timedelta(minutes=30))
    dist.barrier()
    save_json(output / f"runtime_rank{local_rank}.json", execution)
    started = time.perf_counter()
    engine = None
    physical_steps_before_engine = 0
    step_attempts_before_engine = 0
    result: dict[str, Any] = {"mode": args.mode, "rank": local_rank,
                              "formal_training_started": args.mode == "train" and args.run_until_update != 0,
                              "status": "initializing", "smoke_optimizer_updates": 0}
    try:
        all_starts = StartDataset(config["dataset_root"])
        if len(all_starts) != 1600:
            raise ValueError("Accepted main dataset must contain 1600 start records")
        identity = resolved_identity(config, all_starts)
        dataset = SelectedStarts(all_starts) if args.mode == "memory-smoke" else all_starts
        engine = make_engine(config, dataset, identity, args, local_rank)
        synchronize(engine.device)
        result.update({"identity": identity, "initial_network_sha256": state_hash(engine.network),
                       "initial_simulator_sha256": state_hash(engine.rollout.simulator),
                       "setup_seconds": time.perf_counter() - started,
                       "parameter_groups": engine.group_report,
                       "batch_dependent_layers": engine.batch_dependent_layers,
                       "precision": "fp32", "effective_batch": 32, "world_size": 4,
                       "microbatch": args.microbatch, "accumulation": 8 // args.microbatch,
                       "stream": "shuffled_G4_pressure_only" if args.mode == "memory-smoke" else "uniform_shuffled_1600_no_replacement",
                       "optimizer_states_allocated": False,
                       "ema_enabled": False, "status": "running"})
        initial_simulator = result["initial_simulator_sha256"]
        if (config.get("expected_initial_network_sha256") is not None
                and config["expected_initial_network_sha256"] != result["initial_network_sha256"]):
            raise ValueError("Fresh network differs from accepted initialization")
        if (config.get("expected_simulator_sha256") is not None
                and config["expected_simulator_sha256"] != initial_simulator):
            raise ValueError("Loaded simulator differs from accepted Stage 1 state")
        if args.resume:
            result["resume"] = load_training_checkpoint(
                engine, args.resume, run_until_update=args.run_until_update,
                authorize_boundary_extension=args.authorize_boundary_extension,
                authorization=config.get("continuation_authorization"))
        if local_rank == 0:
            save_json(output / "progress.json", result)
        logs = output / f"updates_rank{local_rank}.jsonl"
        records: list[dict[str, Any]] = []

        def step(phase: str) -> dict[str, Any]:
            record = engine.update()
            record["phase"] = phase
            record["physical_optimizer_steps_this_invocation"] = physical_steps_before_engine + engine.optimizer_step_calls
            record["optimizer_step_attempts_this_invocation"] = step_attempts_before_engine + engine.optimizer_step_attempts
            records.append(record)
            append_jsonl(logs, record)
            if local_rank == 0:
                save_json(output / "progress.json", {
                    "mode": args.mode, "status": "running", "phase": phase,
                    "global_update": record["global_update"],
                    "smoke_optimizer_updates": (0 if args.mode == "train" else record["optimizer_step_attempts_this_invocation"]),
                    "last_update": record, "formal_training_started": args.mode == "train"})
            return record

        if args.mode == "train":
            if args.resume is None:
                result["initial_checkpoint"] = save_training_checkpoint(
                    engine, output / "checkpoint_000000.pt", run_until_update=args.run_until_update,
                    authorization=config.get("continuation_authorization"))
            while engine.completed_updates < args.run_until_update:
                step("formal")
                if engine.completed_updates % 100 == 0:
                    result["checkpoint"] = save_training_checkpoint(
                        engine, output / f"checkpoint_{engine.completed_updates:06d}.pt",
                        run_until_update=args.run_until_update,
                        authorization=config.get("continuation_authorization"))
            if engine.completed_updates == 0:
                result["checkpoint"] = result.get("initial_checkpoint", result.get("resume"))
            result["status"] = "initialization_complete" if engine.completed_updates == 0 else "segment_complete"
            result["global_update"] = engine.completed_updates
            result["evaluation_required_at_boundary"] = True
        elif args.mode == "resume-check":
            for _ in range(args.warmup_updates):
                step("resume_prefix")
            split = engine.completed_updates
            checkpoint = output / f"smoke_resume_{split:06d}.pt"
            # This snapshot is taken before checkpoint capture/publication. An
            # immediate exact restore check separates serialization/RNG defects
            # from any floating-point replay noise in the later suffix.
            boundary_reference = resume_snapshot(engine)
            save_training_checkpoint(engine, checkpoint, run_until_update=split + args.measure_updates)
            for _ in range(args.measure_updates):
                step("resume_uninterrupted_suffix")
            reference = resume_snapshot(engine)
            uninterrupted_ids = [r["global_start_ids"] for r in records[-args.measure_updates:]]
            physical_steps_before_engine += engine.optimizer_step_calls
            step_attempts_before_engine += engine.optimizer_step_attempts
            del engine
            engine = None
            gc.collect()
            torch.npu.empty_cache()
            engine = make_engine(config, dataset, identity, args, local_rank)
            load_training_checkpoint(engine, checkpoint, run_until_update=split + args.measure_updates)
            boundary_comparison = compare_states(boundary_reference, resume_snapshot(engine))
            boundary_comparison["tolerance"] = {"rtol": 0.0, "atol": 0.0}
            boundary_comparison["completed_updates"] = engine.completed_updates
            boundary_comparison["checked_before_any_restored_optimizer_attempt"] = engine.optimizer_step_attempts == 0
            result["resume_boundary_parity"] = boundary_comparison
            save_json(output / f"resume_boundary_parity_rank{local_rank}.json", boundary_comparison)
            boundary_passed = torch.tensor(int(boundary_comparison["exact"] and
                boundary_comparison["checked_before_any_restored_optimizer_attempt"]), device=engine.device)
            dist.all_reduce(boundary_passed, op=dist.ReduceOp.MIN)
            if not boundary_passed.item():
                raise RuntimeError("Checkpoint immediate-boundary restore failed exact parity before suffix update")
            del boundary_reference
            for _ in range(args.measure_updates):
                step("resume_restored_suffix")
            comparison = compare_resume_reexecution(reference, resume_snapshot(engine))
            comparison["identical_suffix_start_ids"] = uninterrupted_ids == [r["global_start_ids"] for r in records[-args.measure_updates:]]
            comparison["bitwise_diagnostic_tolerance"] = {"rtol": 0.0, "atol": 0.0}
            result["resume_parity"] = comparison
            result["checkpoint"] = str(checkpoint)
            save_json(output / f"resume_parity_rank{local_rank}.json", comparison)
            passed = torch.tensor(int(comparison["passed"] and comparison["identical_suffix_start_ids"]), device=engine.device)
            dist.all_reduce(passed, op=dist.ReduceOp.MIN)
            if not passed.item():
                raise RuntimeError("Same-configuration resume failed exact control-state or FP32 reexecution parity")
            result["status"] = "passed"
        else:
            for _ in range(args.warmup_updates):
                step("warmup")
            for _ in range(args.measure_updates):
                step("measure")
            durations = [record["seconds"] for record in records if record["phase"] == "measure"]
            result.update({"status": "passed", "measured_seconds_per_update": durations,
                           "median_seconds_per_update": statistics.median(durations),
                           "min_seconds_per_update": min(durations), "max_seconds_per_update": max(durations),
                           "warmup_updates": args.warmup_updates, "measure_updates": args.measure_updates,
                           "cold_first_update_seconds": records[0]["seconds"],
                           "warmup_seconds": sum(r["seconds"] for r in records if r["phase"] == "warmup"),
                           "timing_scope": "read/transfer + complete rollout/backward/DDP + clip/AdamW + numeric metrics; synchronized slowest rank; excludes JSON/checkpoint I/O"})
        result.update({"smoke_optimizer_updates": (0 if args.mode == "train" else step_attempts_before_engine + engine.optimizer_step_attempts),
                       "physical_optimizer_steps_this_invocation": physical_steps_before_engine + engine.optimizer_step_calls,
                       "optimizer_step_attempts_this_invocation": step_attempts_before_engine + engine.optimizer_step_attempts,
                       "optimizer_states_allocated": bool(engine.optimizer.state),
                       "counts": engine.counts, "final_memory": memory_stats(engine.device),
                       "wall_seconds": time.perf_counter() - started,
                       "final_network_sha256": state_hash(engine.network),
                       "simulator_unchanged": initial_simulator == state_hash(engine.rollout.simulator)})
        if not result["simulator_unchanged"]:
            raise RuntimeError("Frozen simulator state changed")
        save_json(output / f"results_rank{local_rank}.json", result)
        dist.barrier()
        if local_rank == 0:
            result["rank_results"] = [json.loads((output / f"results_rank{rank}.json").read_text()) for rank in range(4)]
            result["max_peak_allocated_bytes"] = max(r["final_memory"]["peak_allocated_bytes"] for r in result["rank_results"])
            result["max_peak_reserved_bytes"] = max(r["final_memory"]["peak_reserved_bytes"] for r in result["rank_results"])
            save_json(output / "results.json", result)
            save_json(output / "progress.json", result)
        return 0
    except BaseException as error:
        calls = physical_steps_before_engine + (engine.optimizer_step_calls if engine is not None else 0)
        attempts = step_attempts_before_engine + (engine.optimizer_step_attempts if engine is not None else 0)
        result.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "physical_optimizer_steps_this_invocation": calls,
                       "optimizer_step_attempts_this_invocation": attempts,
                       "smoke_optimizer_updates": 0 if args.mode == "train" else attempts,
                       "global_update": engine.completed_updates if engine is not None and args.mode == "train" else 0,
                       "wall_seconds": time.perf_counter() - started})
        result["memory_at_failure"] = memory_stats(torch.device(f"npu:{local_rank}"))
        save_json(output / f"error_rank{local_rank}.json", result)
        if local_rank == 0:
            save_json(output / "progress.json", result)
        raise
    finally:
        # No barrier here: a peer might already have failed. torchrun terminates
        # remaining workers on nonzero exit; never try to skip a failed batch.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

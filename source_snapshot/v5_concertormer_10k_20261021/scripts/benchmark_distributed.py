"""Bounded real V5 update benchmark; never creates a formal training run.

Example (launch only after independently checking device occupancy):
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    scripts/benchmark_distributed.py --initial weights/adapted_U0.pt \
    --dataset data/v5_places_c50_s2000_v250_t250_g4_r4_20261021 \
    --output reports/benchmark_R0_w4_m1.json --method R0 --microbatch 1 \
    --checkpointing --talking-backend conv2d_equivalent --warmup 1 --updates 2

Loads only completed, hash-verified training image pairs. Validation/test image
arrays are never opened. Warmup/measurement optimizer updates are discarded.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.common import rows, sha, verify_array
from models import build_model
from runtime.environment import configure, make_simulator, synchronize, memory_stats, SIM_SHA256
from training.core import unroll_training
from training.runner import RunConfig, load_initial_model, save_training_checkpoint
from training.evaluation import evaluate_full_validation
from data_pipeline.dataset import V5Dataset
from torch.utils.data import DataLoader
from training.state import (UpdateBatchSampler, UpdateLRScheduler, atomic_json,
                            capture_rng, gather_rank_state, restore_rng)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("D0", "R0"), default="R0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--checkpoint-policy", choices=("all", "highest", "highest_half", "highest_third", "highres", "alternating", "highres_half_low"), default="all")
    parser.add_argument("--talking-backend", choices=("conv3d", "conv2d_equivalent"), default="conv2d_equivalent")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--updates", type=int, default=2, help="bounded timed updates, 1 to 12")
    parser.add_argument("--data-mode", choices=("cached", "streaming"), default="cached")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--extended-validation", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, help="single process only; torchrun uses local rank")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--timeout-minutes", type=float, default=5.)
    parser.add_argument("--keep-checkpoint-probe", action="store_true")
    args = parser.parse_args(argv)
    if args.microbatch < 1 or args.effective_batch not in (32, 36, 48):
        parser.error("microbatch must be positive; supported effective batches are 32/36/48")
    if args.warmup not in (0, 1, 2, 3) or not 1 <= args.updates <= 12:
        parser.error("warmup must be 0 to 3; timed updates must be 1 to 12")
    if args.workers < 0 or args.val_batch_size < 1:
        parser.error("invalid worker count or validation batch size")
    if args.timeout_minutes <= 0:
        parser.error("timeout must be positive")
    return args


def read_completed_train_pairs(dataset, count=32):
    """Inspect immutable identities and read only complete train observations."""
    root = Path(dataset)
    freeze_path, identity_path = root / "SOURCE_FREEZE.json", root / "GENERATION_IDENTITY.json"
    freeze, identity = json.loads(freeze_path.read_text()), json.loads(identity_path.read_text())
    if sha(root / "sample_plan.jsonl") != freeze["sample_plan_sha256"]:
        raise ValueError("Frozen sample-plan hash mismatch")
    if sha(root / "source_manifest.jsonl") != freeze["source_manifest_sha256"]:
        raise ValueError("Frozen source-manifest hash mismatch")
    if identity["simulator_sha256"] != SIM_SHA256 or identity["source_freeze_sha256"] != sha(freeze_path):
        raise ValueError("Generated observations do not bind the accepted Simulator/source freeze")
    identity_hash = sha(identity_path)
    source_rows = {row["source_id"]: row for row in rows(root / "source_manifest.jsonl") if row["split"] == "train"}
    selected = []
    for row in rows(root / "sample_plan.jsonl"):
        if row["split"] != "train":
            continue
        record_path = root / row["record_ref"]
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        if record["sample_plan_row"] != row or record["generation_identity_sha256"] != identity_hash:
            raise ValueError(f"Generation record identity mismatch: {row['sample_id']}")
        if row["G"] != 4 or not record.get("finite"):
            raise ValueError(f"Invalid fixed-G4 generated record: {row['sample_id']}")
        source = source_rows[row["source_id"]]
        if source["clean"]["ref"] != row["clean_ref"]:
            raise ValueError("Clean reference mismatch")
        # Verify bytes/shape/dtype/finite on CPU. No clamp or conversion.
        clean = verify_array(root / row["clean_ref"], source["clean"]["sha256"])
        observed = verify_array(root / row["observation_ref"], record["observation_sha256"])
        selected.append((row, torch.from_numpy(clean), torch.from_numpy(observed)))
        if len(selected) == count:
            break
    if len(selected) != count or len({entry[0]["sample_id"] for entry in selected}) != count:
        raise RuntimeError(f"Need {count} unique completed train pairs, found {len(selected)}")
    return selected, {"source_freeze_sha256": sha(freeze_path),
                      "generation_identity_sha256": identity_hash,
                      "sample_plan_sha256": freeze["sample_plan_sha256"],
                      "simulator_sha256": identity["simulator_sha256"],
                      "selected_sample_ids": [entry[0]["sample_id"] for entry in selected],
                      "selected_split": "train", "val_or_test_arrays_read": False,
                      "requires_full_dataset_ready": False}


def all_finite(flag, device, world):
    value = torch.tensor([int(bool(flag))], dtype=torch.int32, device=device)
    if world > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())


def checkpoint_compatibility(model, simulator, optimizer, scheduler, args, device,
                             rank, world, accumulation, completed_updates, identity):
    """Use the production full-state writer and strict restore, outside timing."""
    directory = Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_checkpoint_probe")
    if rank == 0 and directory.exists():
        raise RuntimeError(f"Refusing to overwrite an existing checkpoint probe directory: {directory}")
    if world > 1:
        dist.barrier()
    config = RunConfig(method=args.method, output_dir=str(directory), initial_checkpoint=args.initial,
        readiness_report="BENCHMARK_ONLY_NOT_A_FORMAL_READINESS_REPORT", world_size=world,
        microbatch=args.microbatch, accumulation_steps=accumulation, effective_batch=args.effective_batch,
        activation_checkpointing=args.checkpointing, checkpoint_policy=args.checkpoint_policy, talking_backend=args.talking_backend,
        network_precision=args.precision, seed=args.seed, objective=getattr(args, 'objective', 'v5_dual'))
    sampler = UpdateBatchSampler(8000, seed=args.seed, microbatch=args.microbatch,
        accumulation_steps=accumulation, world_size=world, rank=rank)
    synchronize(device)
    started = time.perf_counter()
    # This performs all_gather_object of every rank's Python/NumPy/CPU/NPU RNG,
    # then atomic serialization of model, optimizer and complete scheduler.
    save_training_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
        sampler=sampler, config=config, completed_update=completed_updates,
        counters={"benchmark_only_updates": completed_updates, "formal_optimizer_updates": 0},
        best_psnr=0., best_update=completed_updates, initial_identity=identity, device=device,
        make_best=True, periodic=True, rolling=False)
    synchronize(device)
    write_seconds = time.perf_counter() - started
    read_started = time.perf_counter()
    checkpoint_path = directory / "checkpoints" / "latest.pt"
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "rank_states", "sampler", "config_hash", "initial_identity"}
    if not required.issubset(restored) or restored["world_size"] != world or len(restored["rank_states"]) != world:
        raise RuntimeError("Full checkpoint lost required state or per-rank RNG")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    restore_rng(restored["rank_states"][rank]["rng"], device)
    rng_after = capture_rng(device)
    expected_rng = restored["rank_states"][rank]["rng"]
    for key in ("torch_cpu", "device_rng"):
        if key in expected_rng and not torch.equal(rng_after[key], expected_rng[key]):
            raise RuntimeError(f"Restored {key} differs on rank {rank}")
    if rng_after["python"] != expected_rng["python"]:
        raise RuntimeError("Python RNG restore mismatch")
    numpy_same = all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                     for a, b in zip(rng_after["numpy"], expected_rng["numpy"]))
    if not numpy_same:
        raise RuntimeError("NumPy RNG restore mismatch")
    synchronize(device)
    elapsed = time.perf_counter() - started
    del restored, rng_after, expected_rng
    records = gather_rank_state({"rank": rank, "passed": True,
        "write_seconds": write_seconds, "read_restore_seconds": time.perf_counter() - read_started,
        "write_read_restore_seconds": elapsed, "resources_after_checkpoint_probe": memory_stats(device)})
    result = {"status": "PASS", "all_gather_object_rank_rng": True,
              "atomic_full_checkpoint_write_read": True, "strict_model_optimizer_scheduler_restore": True,
              "rng_restore_exact": True, "path": str(checkpoint_path),
              "bytes": checkpoint_path.stat().st_size, "rank_records": records,
              "kept": args.keep_checkpoint_probe, "outside_update_throughput_timing": True}
    if world > 1:
        dist.barrier()
    if rank == 0 and not args.keep_checkpoint_probe:
        # Remove only two known files created by this invocation, then empty
        # directories. No recursive removal or unrelated path modification.
        checkpoint_path.unlink()
        (checkpoint_path.parent / "best.pt").unlink()
        (checkpoint_path.parent / f"update_{completed_updates:06d}.pt").unlink()
        (checkpoint_path.parent / "LATEST.json").unlink()
        checkpoint_path.parent.rmdir()
        directory.rmdir()
    if world > 1:
        dist.barrier()
    return result


def main(argv=None):
    args = arguments(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (1, 2, 4, 6) or not 0 <= rank < world:
        raise ValueError("Authorized benchmark world sizes are 1, 2, 4, 6")
    if world > 1 and args.device is not None:
        raise ValueError("torchrun assigns npu:LOCAL_RANK; do not pass --device for multiple ranks")
    if args.effective_batch % (world * args.microbatch):
        raise ValueError("microbatch * world_size must divide effective batch exactly")
    accumulation = args.effective_batch // (world * args.microbatch)
    device = args.device or f"npu:{local_rank}"
    if not str(device).startswith("npu"):
        raise ValueError("This real-shape distributed benchmark is NPU-only; use CPU semantic tests separately")
    output = Path(args.output)
    rank_output = output.with_name(output.stem + f".rank{rank}.json")
    if output.exists() or rank_output.exists():
        raise RuntimeError("Refusing to overwrite an existing benchmark report; choose a fresh output path")
    runtime = {"world_size": world, "microbatch": args.microbatch, "accumulation_steps": accumulation,
               "effective_batch": args.effective_batch, "network_precision": args.precision,
               "activation_checkpointing": args.checkpointing, "checkpoint_policy": args.checkpoint_policy, "talking_backend": args.talking_backend}
    report = {"schema": "v5_distributed_update_benchmark_v1", "status": "RUNNING",
        "scope": "real train-pair NPU benchmark; zero formal updates; not a stand-alone Simulator/method Gate",
        "arguments": vars(args), "runtime": runtime, "method": args.method,
        "rank": rank, "local_rank": local_rank, "device": device,
        "started_at_unix": time.time(), "warmup_updates": args.warmup,
        "timed_updates": args.updates, "formal_optimizer_updates": 0,
        "cold_start_only": args.warmup == 0,
        "timing_label": "cold_start_included_no_steady_state_claim" if args.warmup == 0 else "after_bounded_warmup_not_convergence_evidence",
        "measurement_boundary": f"{args.data_mode} train pairs; includes H2D, model, FP32 Simulator/loss, backward, DDP and optimizer; streaming includes production DataLoader; excludes reporting, validation and checkpoint probe",
        "measurements": []}
    atomic_json(report, rank_output)
    success = False
    try:
        report["environment"] = configure(device)
        if world > 1:
            dist.init_process_group("hccl", timeout=timedelta(minutes=args.timeout_minutes))
        random.seed(args.seed + rank)
        np.random.seed((args.seed + rank) % 2 ** 32)
        torch.manual_seed(args.seed + rank)
        torch.npu.manual_seed(args.seed + rank)
        model = build_model(activation_checkpointing=args.checkpointing, checkpoint_policy=args.checkpoint_policy,
                            talking_backend=args.talking_backend, seed=args.seed)
        initial = load_initial_model(model, args.initial)
        initial_sha = sha(args.initial)
        initial_identity = {"path": args.initial, "sha256": initial_sha,
                            "provenance": initial.get("provenance", initial.get("initial_identity"))}
        del initial
        report["initial_checkpoint_sha256"] = initial_sha
        pairs, data_identity = read_completed_train_pairs(args.dataset, count=args.effective_batch)
        report["data_identity"] = data_identity
        # Every rank agrees on the actual initialization and selected data.
        identities = gather_rank_state({"initial_sha256": initial_sha,
            "generation_identity_sha256": data_identity["generation_identity_sha256"],
            "sample_ids": data_identity["selected_sample_ids"]})
        if any(value != identities[0] for value in identities[1:]):
            raise RuntimeError("Ranks do not share initialization/data identity")
        cpu_batches = []
        for micro in range(accumulation):
            start = micro * world * args.microbatch + rank * args.microbatch
            selected = pairs[start:start + args.microbatch]
            cpu_batches.append((torch.stack([entry[1] for entry in selected]),
                                torch.stack([entry[2] for entry in selected]),
                                [entry[0]["sample_id"] for entry in selected]))
        del pairs
        stream = None
        if args.data_mode == "streaming":
            dataset = V5Dataset(args.dataset, "train", require_ready=True, allow_test=False)
            sampler = UpdateBatchSampler(len(dataset), seed=args.seed, microbatch=args.microbatch,
                accumulation_steps=accumulation, world_size=world, rank=rank,
                max_updates=args.warmup + args.updates)
            options = {"num_workers": args.workers,
                       "generator": torch.Generator().manual_seed(args.seed + 991 + rank)}
            if args.workers:
                options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
            stream = iter(DataLoader(dataset, batch_sampler=sampler, **options))
            report["data_identity"]["streaming_sample_order"] = "production UpdateBatchSampler; seed plus epoch"
        report["local_sample_ids"] = [identifier for batch in cpu_batches for identifier in batch[2]]
        model.to(device).train()
        simulator = make_simulator(device).eval()
        for parameter in simulator.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(), betas=(.9, .999), eps=1e-8)
        scheduler = UpdateLRScheduler(optimizer, max_updates=10000, warmup_updates=300)
        if world > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank],
                broadcast_buffers=False, find_unused_parameters=False)
        raw_model = model.module if hasattr(model, "module") else model
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        report["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
        synchronize(device)
        torch.npu.reset_peak_memory_stats(device)
        timed_rank_records = []
        for update_index in range(args.warmup + args.updates):
            phase = "warmup" if update_index < args.warmup else "timed"
            if update_index == args.warmup:
                synchronize(device)
                torch.npu.reset_peak_memory_stats(device)
            if world > 1:
                dist.barrier()
            synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step_for_update(update_index + 1)
            total_loss = torch.zeros(3, dtype=torch.float32, device=device)
            for micro, (cpu_clean, cpu_observed, _) in enumerate(cpu_batches):
                if stream is not None:
                    batch = next(stream)
                    cpu_clean, cpu_observed = batch["clean"], batch["observation"]
                clean = cpu_clean.to(device=device, dtype=torch.float32)
                observed = cpu_observed.to(device=device, dtype=torch.float32)
                context = model.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
                with context:
                    result = unroll_training(model, simulator, observed, clean, method=args.method,
                        path_seed=args.seed + update_index * accumulation * world + micro * world + rank,
                        network_precision=args.precision, retain_trace=False)
                    if not all_finite(torch.isfinite(result.loss).item(), device, world):
                        raise FloatingPointError("Nonfinite benchmark loss on at least one rank")
                    (result.loss / accumulation).backward()
                total_loss += torch.stack((result.loss.detach(), result.clean_loss.detach(),
                                           result.observation_loss.detach())) / accumulation
                del result, clean, observed
            missing = [name for name, parameter in raw_model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            if not all_finite(not missing, device, world):
                raise RuntimeError(f"Missing trainable gradients; local rank {rank}: {missing}")
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=False)
            if not all_finite(torch.isfinite(norm).item(), device, world):
                raise FloatingPointError("Nonfinite global gradient norm on at least one rank")
            optimizer.step()
            synchronize(device)
            seconds = time.perf_counter() - started
            local_record = {"rank": rank, "phase": phase, "benchmark_update": update_index + 1,
                "seconds": seconds, "loss": total_loss[0].item(), "clean_mae": total_loss[1].item(),
                "observation_mae": total_loss[2].item(), "grad_norm_before_clip": norm.item(),
                "all_trainable_gradients_present": True, "finite_gradients": True,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                **memory_stats(device)}
            del total_loss, norm
            rank_records = gather_rank_state(local_record)
            max_elapsed = max(record["seconds"] for record in rank_records)
            record = {"phase": phase, "benchmark_update": update_index + 1,
                "max_rank_elapsed_seconds": max_elapsed, "global_samples_per_second": args.effective_batch / max_elapsed,
                "global_network_forward_exposures_per_second": args.effective_batch * (1 if args.method == "D0" else 4) / max_elapsed,
                "mean_loss_across_ranks": sum(r["loss"] for r in rank_records) / world,
                "rank_records": rank_records}
            report["measurements"].append(record)
            if phase == "timed":
                timed_rank_records.append(record)
            atomic_json(report, rank_output)
            if rank == 0:
                print(json.dumps({"event": "benchmark_update", **record}, allow_nan=False), flush=True)
        # Save compute-phase peak before full checkpoint read/restore allocates
        # additional optimizer state. Do not conflate the two memory figures.
        compute_resources = gather_rank_state({"rank": rank, **memory_stats(device)})
        report["compute_rank_resources"] = compute_resources
        report["timed_summary"] = {
            "timed_updates": len(timed_rank_records),
            "sum_max_rank_elapsed_seconds": sum(r["max_rank_elapsed_seconds"] for r in timed_rank_records),
            "global_samples_per_second": args.effective_batch * len(timed_rank_records) /
                sum(r["max_rank_elapsed_seconds"] for r in timed_rank_records),
            "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in compute_resources),
            "max_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in compute_resources),
            "overall_update_peak_allocated_bytes_including_warmup": max(
                r["peak_allocated_bytes"] for measurement in report["measurements"] for r in measurement["rank_records"]),
            "overall_update_peak_reserved_bytes_including_warmup": max(
                r["peak_reserved_bytes"] for measurement in report["measurements"] for r in measurement["rank_records"]),
            "cold_start_only": args.warmup == 0}
        seconds = [r["max_rank_elapsed_seconds"] for r in timed_rank_records]
        report["timed_summary"].update(mean_seconds=statistics.mean(seconds),
            median_seconds=statistics.median(seconds), min_seconds=min(seconds), max_seconds=max(seconds),
            coefficient_of_variation=statistics.pstdev(seconds) / statistics.mean(seconds),
            stable_throughput_evidence=len(seconds) >= 5 and args.warmup >= 1 and
                statistics.pstdev(seconds) / statistics.mean(seconds) <= .05)
        report["checkpoint_compatibility"] = checkpoint_compatibility(model, simulator, optimizer,
            scheduler, args, device, rank, world, accumulation, args.warmup + args.updates,
            initial_identity)
        if args.full_validation:
            validation_started = time.perf_counter()
            validation = evaluate_full_validation(model, simulator,
                V5Dataset(args.dataset, "val", require_ready=True, allow_test=False),
                device=device, method=args.method, output_dir=output.with_suffix(""), update=0,
                network_precision=args.precision, batch_size=args.val_batch_size,
                workers=args.workers, expected_size=1000, extended_metrics=args.extended_validation)
            if world > 1:
                dist.barrier()
            report["full_validation"] = {**validation,
                "complete_call_seconds_including_report": time.perf_counter() - validation_started,
                "weight_scope": "temporary benchmark weights; not a formal D0/R0 experiment result",
                "test_used": False}
        report["benchmark_optimizer_updates_discarded"] = args.warmup + args.updates
        report["status"] = "PASS"
        success = True
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    report["finished_at_unix"] = time.time()
    report["whole_invocation_elapsed_seconds"] = report["finished_at_unix"] - report["started_at_unix"]
    atomic_json(report, rank_output)
    if rank == 0:
        atomic_json(report, output)
        print(json.dumps({"status": report["status"], "output": str(output),
                          "timed_summary": report.get("timed_summary"),
                          "error": report.get("error")}, allow_nan=False), flush=True)
    if success and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Single frozen 3,000-update Stage A1 pilot."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel

from physical_bridge.data import BridgeArchive
from physical_bridge.model import DualHeadUNet
from physical_bridge.simulator import PiRNP2SSimulator
from physical_bridge.utils import (
    atomic_touch,
    atomic_write_bytes,
    atomic_write_json,
    parameter_grad_norm,
    seed_process,
    sha256_file,
)
from physical_bridge.v02_data import load_batch_manifest, load_rank_batch, stack_loaded
from physical_bridge.v02_losses import (
    clean_loss_per_sample,
    trajectory_bridge_losses_per_sample,
)
from physical_bridge.v02_protocol import cosine_learning_rate
from physical_bridge.v02_training import (
    EMA,
    atomic_checkpoint,
    cpu_tree,
    full_trajectory_loss_sum,
    optimizer_to,
)


def slice_batch(batch: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    result: dict[str, Any] = {"specs": batch["specs"][start:stop]}
    for key, value in batch.items():
        if key != "specs":
            result[key] = value[start:stop]
    return result


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_checkpoint(path: Path) -> Path | None:
    values = sorted(path.glob("update-*.pt"))
    return values[-1] if values else None


def metrics_prefix(path: Path, maximum_update: int) -> bytes:
    if not path.exists():
        return b""
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if int(row["update"]) <= maximum_update:
                rows[int(row["update"])] = row
    return b"".join(
        (json.dumps(rows[index], sort_keys=True) + "\n").encode() for index in sorted(rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a0-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--resume", choices=("none", "auto"), default="auto")
    parser.add_argument(
        "--engineering-probe-only",
        action="store_true",
        help="Run one disposable update to verify the requested microbatch and peak HBM.",
    )
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    a1_config = document["a1"]
    a0 = json.loads(args.a0_report.read_text(encoding="utf-8"))
    if a0["status"] != "GO" or not (args.a0_report.parent / "A0_PASS").exists():
        raise RuntimeError("A0 is not PASS")
    selected = a0["selection"]["selected"]
    lambda_t = float(selected["lambda_T"])
    lambda_b = float(selected["lambda_B"])
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible != document["authorized_physical_npus"]:
        raise RuntimeError(f"Physical device authorization mismatch: {visible}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4:
        raise RuntimeError(f"A1 requires frozen four-NPU execution, got {world_size}")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.distributed.init_process_group(backend="hccl", timeout=dt.timedelta(hours=12))
    barrier = torch.distributed.barrier
    seed_process(int(a1_config["model_optimizer_seed"]))

    model = DualHeadUNet(**document["model"]).to(device)
    initial = torch.load(document["v01_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(initial["model"], strict=True)
    del initial
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(a1_config["learning_rate"]),
        betas=tuple(a1_config["betas"]),
        weight_decay=float(a1_config["weight_decay"]),
    )
    ema = EMA(model, float(a1_config["ema"]))
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    source_root = Path(document["v02_source_root"])
    simulator_document = yaml.safe_load(
        (source_root / document["simulator_config"]).read_text(encoding="utf-8")
    )
    simulator = PiRNP2SSimulator(
        simulator_document["asset_dir"],
        image_size=int(simulator_document["image_size"]),
        rf_range=int(simulator_document["rf_range"]),
        aperture_diameter=float(simulator_document["aperture_diameter"]),
        propagation_distance=float(simulator_document["propagation_distance"]),
        wavelength=float(simulator_document["wavelength"]),
        correlation=float(simulator_document["correlation"]),
        tilt_threshold=float(simulator_document["tilt_threshold"]),
        adjustment=float(simulator_document["adjustment"]),
        device=device,
    )
    archive = BridgeArchive(document["dataset_root"], "train")
    batches = load_batch_manifest(args.manifest)
    maximum_updates = int(a1_config["updates"])
    if len(batches) != maximum_updates:
        raise ValueError(f"A1 expected {maximum_updates} batches, got {len(batches)}")
    effective_batch = int(a1_config["effective_batch_size"])
    local_count = effective_batch // world_size
    if local_count % args.microbatch:
        raise ValueError("microbatch must divide local batch")
    accumulation = local_count // args.microbatch
    checkpoint_dir = args.output_dir / "checkpoints"
    metrics_path = args.output_dir / "train_metrics.jsonl"
    start_update = 0
    if args.resume == "auto":
        resume_path = latest_checkpoint(checkpoint_dir)
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_to(optimizer, device)
            ema.load_state_dict(checkpoint["ema"], device)
            start_update = int(checkpoint["update"])
            del checkpoint
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(metrics_path, metrics_prefix(metrics_path, start_update))
        atomic_write_json(
            args.output_dir / "frozen_config.json",
            {
                "config": document,
                "lambda_T": lambda_t,
                "lambda_B": lambda_b,
                "a0_report_sha256": sha256_file(args.a0_report),
                "manifest_sha256": sha256_file(args.manifest),
                "physical_visible_devices": visible,
                "world_size": world_size,
                "microbatch_per_rank": args.microbatch,
                "gradient_accumulation": accumulation,
                "start_update": start_update,
            },
        )
        atomic_touch(
            args.output_dir
            / ("ENGINEERING_PROBE_RUNNING" if args.engineering_probe_only else "A1_RUNNING")
        )
    barrier()

    training_start = time.monotonic()
    execution_stop = min(
        maximum_updates,
        start_update + 1 if args.engineering_probe_only else maximum_updates,
    )
    for update_index in range(start_update, execution_stop):
        step_start = time.monotonic()
        lr = cosine_learning_rate(
            update_index,
            maximum_updates,
            float(a1_config["learning_rate"]),
            int(a1_config["warmup_updates"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        row = batches[update_index]
        loaded = load_rank_batch(archive, row, rank=rank, world_size=world_size)
        full_batch = stack_loaded(loaded, device)
        global_valid = sum(item.get("h") is not None for item in row["samples"])
        local_sums = torch.zeros(6, device=device, dtype=torch.float32)
        for microstep in range(accumulation):
            start = microstep * args.microbatch
            stop = start + args.microbatch
            batch = slice_batch(full_batch, start, stop)
            sync_context = ddp.no_sync() if microstep < accumulation - 1 else contextlib.nullcontext()
            with sync_context:
                with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                    prediction = ddp(batch["image"], batch["g"])
                clean_values = clean_loss_per_sample(prediction, batch)
                trajectory_values, bridge_values = trajectory_bridge_losses_per_sample(
                    prediction, batch, simulator
                )
                clean_loss = clean_values.sum() / local_count
                trajectory_loss = trajectory_values.sum() / local_count
                bridge_loss = (
                    bridge_values.sum() * world_size / global_valid
                    if bridge_values.numel()
                    else prediction.clean.float().sum() * 0.0
                )
                total_loss = clean_loss + lambda_t * trajectory_loss + lambda_b * bridge_loss
                if not bool(torch.isfinite(total_loss).detach().cpu()):
                    raise FloatingPointError(f"Non-finite loss at A1 update {update_index}")
                total_loss.backward()
            full_t_required = update_index < 200 or update_index >= maximum_updates - 200
            full_t_sum = (
                full_trajectory_loss_sum(prediction, batch, simulator)
                if full_t_required
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            local_sums += torch.tensor(
                [
                    float(clean_values.detach().sum().cpu()),
                    float(trajectory_values.detach().sum().cpu()),
                    float(bridge_values.detach().sum().cpu()) if bridge_values.numel() else 0.0,
                    float(bridge_values.numel()),
                    float(full_t_sum.detach().cpu()),
                    float(clean_values.numel()),
                ],
                device=device,
                dtype=torch.float32,
            )
        preclip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(a1_config["gradient_clip"])
        )
        branches = model.branch_parameters()
        gradient_norms = {
            "preclip": float(preclip.detach().cpu()),
            "shared": parameter_grad_norm(branches["shared"]),
            "clean": parameter_grad_norm(branches["clean"]),
            "realization": parameter_grad_norm(branches["realization"]),
        }
        optimizer.step()
        ema.update(model)
        completed = update_index + 1
        torch.distributed.all_reduce(local_sums)
        clean_mean = float((local_sums[0] / local_sums[5]).cpu())
        trajectory_mean = float((local_sums[1] / local_sums[5]).cpu())
        bridge_mean = float((local_sums[2] / local_sums[3]).cpu())
        full_trajectory_mean = float((local_sums[4] / local_sums[5]).cpu())
        total_mean = clean_mean + lambda_t * trajectory_mean + lambda_b * bridge_mean
        elapsed = time.monotonic() - training_start
        if rank == 0:
            append_jsonl(
                metrics_path,
                {
                    "update": completed,
                    "loss": total_mean,
                    "clean_loss": clean_mean,
                    "trajectory_estimator_loss": trajectory_mean,
                    "full_trajectory_loss": full_trajectory_mean if full_t_required else None,
                    "bridge_loss": bridge_mean,
                    "lambda_T": lambda_t,
                    "lambda_B": lambda_b,
                    "learning_rate": lr,
                    "gradient_norms": gradient_norms,
                    "valid_transitions": int(local_sums[3].cpu()),
                    "step_wall_seconds": time.monotonic() - step_start,
                    "elapsed_wall_seconds": elapsed,
                    "eta_seconds": elapsed / (completed - start_update) * (maximum_updates - completed),
                    "peak_npu_memory_bytes": int(torch.npu.max_memory_allocated(device)),
                },
            )
            if completed % 10 == 0 or completed == 1:
                atomic_write_json(
                    args.output_dir / "progress.json",
                    {
                        "status": "RUNNING",
                        "completed_updates": completed,
                        "total_updates": maximum_updates,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / (completed - start_update) * (maximum_updates - completed),
                        "last_step_seconds": time.monotonic() - step_start,
                    },
                )
        checkpoint_due = (
            not args.engineering_probe_only
            and completed % int(a1_config["checkpoint_interval"]) == 0
        )
        if checkpoint_due or completed == maximum_updates:
            barrier()
            if rank == 0:
                checkpoint_path = checkpoint_dir / f"update-{completed:06d}.pt"
                checkpoint_hash = atomic_checkpoint(
                    checkpoint_path,
                    {
                        "stage": "A1",
                        "update": completed,
                        "model": cpu_tree(model.state_dict()),
                        "ema": ema.state_dict_cpu(),
                        "optimizer": cpu_tree(optimizer.state_dict()),
                        "lambda_T": lambda_t,
                        "lambda_B": lambda_b,
                        "manifest_sha256": sha256_file(args.manifest),
                    },
                )
                atomic_write_json(
                    checkpoint_path.with_suffix(".json"),
                    {"update": completed, "path": str(checkpoint_path), "sha256": checkpoint_hash},
                )
            barrier()
    barrier()
    if args.engineering_probe_only:
        peak = torch.tensor(
            [float(torch.npu.max_memory_allocated(device))],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        if rank == 0:
            atomic_write_json(
                args.output_dir / "engineering_probe.json",
                {
                    "status": "PASS",
                    "microbatch_per_rank": args.microbatch,
                    "gradient_accumulation": accumulation,
                    "completed_updates": execution_stop - start_update,
                    "peak_npu_memory_bytes_max_rank": int(peak.cpu()),
                    "wall_seconds": time.monotonic() - training_start,
                },
            )
            atomic_touch(args.output_dir / "ENGINEERING_PROBE_PASS")
        barrier()
        archive.store.close()
        torch.distributed.destroy_process_group()
        return

    if rank == 0:
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        first = rows[:200]
        last = rows[-200:]
        stability = {
            "full_trajectory_first_200_median": statistics.median(
                row["full_trajectory_loss"] for row in first
            ),
            "full_trajectory_last_200_median": statistics.median(
                row["full_trajectory_loss"] for row in last
            ),
            "bridge_first_200_median": statistics.median(row["bridge_loss"] for row in first),
            "bridge_last_200_median": statistics.median(row["bridge_loss"] for row in last),
        }
        atomic_write_json(
            args.output_dir / "training_summary.json",
            {
                "stage": "A1",
                "completed_updates": maximum_updates,
                "lambda_T": lambda_t,
                "lambda_B": lambda_b,
                "stability": stability,
                "wall_seconds": time.monotonic() - training_start,
            },
        )
        atomic_touch(args.output_dir / "TRAINING_COMPLETE")
        atomic_write_json(
            args.output_dir / "progress.json",
            {
                "status": "TRAINING_COMPLETE",
                "completed_updates": maximum_updates,
                "total_updates": maximum_updates,
                "elapsed_seconds": time.monotonic() - training_start,
                "eta_seconds": 0,
            },
        )
    barrier()
    archive.store.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

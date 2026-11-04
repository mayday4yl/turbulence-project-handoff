#!/usr/bin/env python3
"""Formal four-NPU Stage A0 gradient audit for the frozen V0.2 losses."""

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
from physical_bridge.utils import atomic_touch, atomic_write_bytes, atomic_write_json, parameter_grad_norm
from physical_bridge.v02_data import load_batch_manifest, load_rank_batch, stack_loaded
from physical_bridge.v02_losses import (
    bridge_loss_per_sample,
    clean_loss_per_sample,
    output_zero,
    trajectory_loss_per_sample,
)
from physical_bridge.v02_protocol import select_gradient_weights


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


def branch_norms(model: DualHeadUNet) -> dict[str, float]:
    branches = model.branch_parameters()
    return {
        "total": parameter_grad_norm(model.parameters()),
        "shared": parameter_grad_norm(branches["shared"]),
        "clean": parameter_grad_norm(branches["clean"]),
        "realization": parameter_grad_norm(branches["realization"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root = Path(document["v02_source_root"])
    run_root = Path(document["run_root"])
    if not (run_root / "manifests/MANIFESTS_FROZEN").exists():
        raise RuntimeError("Frozen manifest gate is missing")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible != document["authorized_physical_npus"]:
        raise RuntimeError(f"Physical device authorization mismatch: {visible}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4:
        raise RuntimeError(f"A0 requires frozen four-NPU execution, got {world_size}")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.distributed.init_process_group(backend="hccl", timeout=dt.timedelta(hours=6))
    barrier = torch.distributed.barrier

    model = DualHeadUNet(**document["model"]).to(device)
    checkpoint = torch.load(document["v01_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.train()
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
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
    expected_batches = int(document["a0"]["effective_batches"])
    if len(batches) != expected_batches:
        raise ValueError(f"A0 expected {expected_batches} batches, got {len(batches)}")
    local_count = int(document["a0"]["effective_batch_size"]) // world_size
    if local_count % args.microbatch:
        raise ValueError("microbatch must divide four local samples")
    accumulation = local_count // args.microbatch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "gradient_batches.jsonl"
    if rank == 0:
        atomic_write_bytes(metrics_path, b"")
    barrier()
    start_time = time.monotonic()
    records: list[dict[str, Any]] = []
    for batch_index, row in enumerate(batches):
        loaded = load_rank_batch(archive, row, rank=rank, world_size=world_size)
        full_batch = stack_loaded(loaded, device)
        global_valid = sum(item.get("h") is not None for item in row["samples"])
        row_result: dict[str, Any] = {
            "batch": batch_index,
            "global_valid_transitions": global_valid,
            "losses": {},
        }
        for loss_name in ("clean", "trajectory", "bridge"):
            model.zero_grad(set_to_none=True)
            loss_value = 0.0
            for microstep in range(accumulation):
                start = microstep * args.microbatch
                stop = start + args.microbatch
                batch = slice_batch(full_batch, start, stop)
                sync_context = ddp.no_sync() if microstep < accumulation - 1 else contextlib.nullcontext()
                with sync_context:
                    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                        prediction = ddp(batch["image"], batch["g"])
                    if loss_name == "clean":
                        values = clean_loss_per_sample(prediction, batch)
                        loss = values.sum() / local_count
                    elif loss_name == "trajectory":
                        values = trajectory_loss_per_sample(prediction, batch, simulator)
                        loss = values.sum() / local_count
                    else:
                        values = bridge_loss_per_sample(prediction, batch, simulator)
                        loss = (
                            values.sum() * world_size / global_valid
                            if values.numel()
                            else output_zero(prediction)
                        )
                    if not bool(torch.isfinite(loss).detach().cpu()):
                        raise FloatingPointError(f"Non-finite {loss_name} loss at A0 batch {batch_index}")
                    loss.backward()
                    loss_value += float(loss.detach().cpu())
            norms = branch_norms(model)
            if any(not math.isfinite(value) for value in norms.values()):
                raise FloatingPointError(f"Non-finite {loss_name} gradient at batch {batch_index}")
            reduced_loss = torch.tensor([loss_value], device=device)
            torch.distributed.all_reduce(reduced_loss)
            reduced_loss /= world_size
            row_result["losses"][loss_name] = {
                "loss": float(reduced_loss.cpu()),
                "gradient_norms": norms,
            }
        if rank == 0:
            records.append(row_result)
            append_jsonl(metrics_path, row_result)
        if (batch_index + 1) % 8 == 0:
            barrier()
            if rank == 0:
                atomic_write_json(
                    args.output_dir / "progress.json",
                    {
                        "completed_batches": batch_index + 1,
                        "total_batches": len(batches),
                        "elapsed_seconds": time.monotonic() - start_time,
                    },
                )
    barrier()
    if rank == 0:
        medians: dict[str, dict[str, float]] = {}
        for loss_name in ("clean", "trajectory", "bridge"):
            medians[loss_name] = {
                branch: statistics.median(
                    record["losses"][loss_name]["gradient_norms"][branch] for record in records
                )
                for branch in ("total", "shared", "clean", "realization")
            }
        selection = select_gradient_weights(
            g_clean=medians["clean"]["total"],
            g_trajectory=medians["trajectory"]["total"],
            g_bridge=medians["bridge"]["total"],
            h_clean=medians["clean"]["clean"],
            h_bridge=medians["bridge"]["clean"],
        )
        report = {
            "stage": "A0",
            "status": selection["status"],
            "effective_batches": len(records),
            "effective_batch_size": int(document["a0"]["effective_batch_size"]),
            "physical_visible_devices": visible,
            "microbatch_per_rank": args.microbatch,
            "gradient_accumulation": accumulation,
            "gradient_precision": "BF16 network forward, FP32 simulator/loss, no GradScaler",
            "medians": medians,
            "selection": selection,
            "wall_seconds": time.monotonic() - start_time,
        }
        atomic_write_json(args.output_dir / "v02_gradient_audit.json", report)
        selected = selection.get("selected", {})
        lines = [
            "# V0.2 Stage A0 gradient audit",
            "",
            f"Status: **{selection['status']}**",
            "",
            f"- Effective batches: `{len(records)}`",
            f"- Physical NPUs: `{visible}`",
            f"- Median total norms: C={medians['clean']['total']:.8g}, "
            f"T={medians['trajectory']['total']:.8g}, B={medians['bridge']['total']:.8g}",
        ]
        if selection["status"] == "GO":
            lines.extend(
                [
                    f"- Frozen lambda_T: `{selected['lambda_T']}`",
                    f"- Frozen lambda_B: `{selected['lambda_B']}`",
                    f"- Ratios: r_T={selected['r_T']:.6f}, r_B={selected['r_B']:.6f}, "
                    f"r_phys={selected['r_phys']:.6f}, r_B,C={selected['r_B_clean']:.6f}",
                ]
            )
            atomic_touch(args.output_dir / "A0_PASS")
        else:
            lines.append(f"- STOP reason: `{selection.get('reason')}`")
            atomic_touch(args.output_dir / "A0_STOP")
        (args.output_dir / "v02_gradient_audit.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    barrier()
    archive.store.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

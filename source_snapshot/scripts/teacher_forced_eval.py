#!/usr/bin/env python3
"""Frozen-checkpoint teacher-forced branch evaluation on val or test."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator
    from physical_bridge.utils import atomic_touch
    from physical_bridge.validation import teacher_forced_validation

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="hccl", timeout=dt.timedelta(hours=12)
        )
    barrier = torch.distributed.barrier if world_size > 1 else None
    train_document = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    simulator_document = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    model_config = train_document["model"]
    model = DualHeadUNet(
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        residual_blocks_per_level=int(model_config["residual_blocks_per_level"]),
        group_norm_groups=int(model_config["group_norm_groups"]),
        severity_embedding_dim=int(model_config["severity_embedding_dim"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval().requires_grad_(False)
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
    qualitative = json.loads(
        (args.dataset_root / "manifests" / "qualitative_ids.json").read_text(encoding="utf-8")
    )[args.split]
    teacher_forced_validation(
        model=model,
        simulator=simulator,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=device,
        rank=rank,
        world_size=world_size,
        split=args.split,
        qualitative_keys=qualitative if rank == 0 else None,
        distributed_barrier=barrier,
    )
    if rank == 0:
        atomic_touch(args.output_dir / "TEACHER_FORCED_COMPLETE")
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

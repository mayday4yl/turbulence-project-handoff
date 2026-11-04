#!/usr/bin/env python3
"""Exploratory inference-only decomposition for the completed V0.2 A1 run.

This script is deliberately separate from the frozen gate evaluator. It writes
only to a caller-provided diagnostic directory and does not alter checkpoints,
manifests, official markers, or the scientific STOP decision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml


G_GRID = (0.75, 1.50, 2.25, 3.00, 3.75, 4.50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().flatten(1).mean(1)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--val-subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    from physical_bridge.data import BridgeArchive
    from physical_bridge.metrics import per_sample_psnr
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator, Realization

    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible != document["authorized_physical_npus"]:
        raise RuntimeError(f"Physical device authorization mismatch: {visible}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4:
        raise RuntimeError(f"Expected four diagnostic workers, got {world_size}")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.distributed.init_process_group(backend="hccl", timeout=dt.timedelta(hours=2))
    barrier = torch.distributed.barrier

    simulator_document = yaml.safe_load(
        (Path(document["v02_source_root"]) / document["simulator_config"]).read_text(
            encoding="utf-8"
        )
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
    model = DualHeadUNet(**document["model"]).to(device)
    model.eval().requires_grad_(False)

    v01 = Path(document["v01_checkpoint"])
    a1_checkpoints = Path(document["run_root"]) / "a1" / "checkpoints"
    states = [
        ("v01_raw", v01, "model"),
        ("v01_ema", v01, "ema"),
        ("v02_1k_ema", a1_checkpoints / "update-001000.pt", "ema"),
        ("v02_2k_ema", a1_checkpoints / "update-002000.pt", "ema"),
        ("v02_3k_raw", a1_checkpoints / "update-003000.pt", "model"),
        ("v02_3k_ema", a1_checkpoints / "update-003000.pt", "ema"),
    ]

    subset = json.loads(args.val_subset.read_text(encoding="utf-8"))
    keys = [key for bucket in sorted(subset, key=float) for key in subset[bucket]]
    if len(keys) != 100 or len(set(keys)) != 100:
        raise ValueError("Frozen diagnostic subset must contain 100 unique trajectories")
    archive = BridgeArchive(document["dataset_root"], "val")

    # Cache each rank's 25 records on CPU. This avoids repeated tar reads while
    # retaining one model at a time on the NPU.
    cached: list[dict[str, Any]] = []
    for key in keys[rank::world_size]:
        record = archive.by_key[key]
        terminal_g = float(record["terminal_G"])
        next_g = G_GRID[G_GRID.index(round(terminal_g, 2)) - 1]
        cached.append(
            {
                "key": key,
                "G": terminal_g,
                "next_g": next_g,
                "observation": archive.state(record, terminal_g),
                "true_next": archive.state(record, next_g),
                "clean": archive.clean(record),
                "true_realization": archive.realization(record),
            }
        )

    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for label, checkpoint_path, state_key in states:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint[state_key], strict=True)
        del checkpoint
        for item in cached:
            observation = item["observation"].unsqueeze(0).to(device)
            true_next = item["true_next"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)
            terminal_g = float(item["G"])
            next_g = float(item["next_g"])
            true_realization_cpu = item["true_realization"]
            true_realization = Realization(
                true_realization_cpu.epsilon_zernike.unsqueeze(0).to(device),
                true_realization_cpu.epsilon_tilt.unsqueeze(0).to(device),
            )
            with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
                prediction = model(
                    observation, torch.tensor([terminal_g], device=device)
                )

            # Render three controlled next-state variants in one fixed-shape call:
            # predicted clean + predicted xi (actual no-refine path), predicted
            # clean + true xi, and true clean + predicted xi.
            rendered_clean = torch.cat([prediction.clean, prediction.clean, clean], dim=0)
            rendered_realization = Realization(
                torch.cat(
                    [
                        prediction.realization.epsilon_zernike,
                        true_realization.epsilon_zernike,
                        prediction.realization.epsilon_zernike,
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        prediction.realization.epsilon_tilt,
                        true_realization.epsilon_tilt,
                        prediction.realization.epsilon_tilt,
                    ],
                    dim=0,
                ),
            )
            with torch.no_grad():
                rendered = simulator.render(
                    rendered_clean.float(),
                    rendered_realization,
                    torch.tensor([next_g, next_g, next_g], device=device),
                ).output
            predicted_next, true_xi_next, true_clean_next = rendered.split(1, dim=0)
            final_inputs = torch.cat(
                [predicted_next, true_next, true_xi_next, true_clean_next], dim=0
            )
            with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
                final_clean = model(
                    final_inputs,
                    torch.full((4,), next_g, device=device),
                ).clean
            no_refine_final, teacher_final, true_xi_final, true_clean_final = final_clean.split(
                1, dim=0
            )
            zernike = prediction.realization.epsilon_zernike.float().flatten(1)
            tilt = prediction.realization.epsilon_tilt.float().flatten(1)
            rows.append(
                {
                    "checkpoint": label,
                    "checkpoint_path": str(checkpoint_path),
                    "state_key": state_key,
                    "key": item["key"],
                    "G": terminal_g,
                    "next_g": next_g,
                    "k0_psnr": float(per_sample_psnr(prediction.clean, clean)[0].cpu()),
                    "no_refine_next_mse": float(
                        per_sample_mse(predicted_next, true_next)[0].cpu()
                    ),
                    "no_refine_k1_psnr": float(
                        per_sample_psnr(no_refine_final, clean)[0].cpu()
                    ),
                    "teacher_forced_k1_psnr": float(
                        per_sample_psnr(teacher_final, clean)[0].cpu()
                    ),
                    "predicted_clean_true_xi_next_mse": float(
                        per_sample_mse(true_xi_next, true_next)[0].cpu()
                    ),
                    "predicted_clean_true_xi_k1_psnr": float(
                        per_sample_psnr(true_xi_final, clean)[0].cpu()
                    ),
                    "true_clean_predicted_xi_next_mse": float(
                        per_sample_mse(true_clean_next, true_next)[0].cpu()
                    ),
                    "true_clean_predicted_xi_k1_psnr": float(
                        per_sample_psnr(true_clean_final, clean)[0].cpu()
                    ),
                    "zernike_norm": float(zernike.norm(dim=1)[0].cpu()),
                    "tilt_norm": float(tilt.norm(dim=1)[0].cpu()),
                }
            )
        barrier()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank-{rank:02d}.jsonl"
    append_rows(rank_path, rows)
    barrier()
    if rank == 0:
        combined: list[dict[str, Any]] = []
        for worker in range(world_size):
            combined.extend(
                json.loads(line)
                for line in (args.output_dir / f"rank-{worker:02d}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        expected = len(states) * len(keys)
        if len(combined) != expected:
            raise ValueError(f"Expected {expected} diagnostic rows, got {len(combined)}")
        metric_names = [
            "k0_psnr",
            "no_refine_next_mse",
            "no_refine_k1_psnr",
            "teacher_forced_k1_psnr",
            "predicted_clean_true_xi_next_mse",
            "predicted_clean_true_xi_k1_psnr",
            "true_clean_predicted_xi_next_mse",
            "true_clean_predicted_xi_k1_psnr",
            "zernike_norm",
            "tilt_norm",
        ]
        summaries: list[dict[str, Any]] = []
        for label, checkpoint_path, state_key in states:
            state_rows = [row for row in combined if row["checkpoint"] == label]
            for terminal_g in G_GRID[1:]:
                group = [row for row in state_rows if float(row["G"]) == terminal_g]
                record: dict[str, Any] = {
                    "checkpoint": label,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "state_key": state_key,
                    "G": terminal_g,
                    "n": len(group),
                }
                for name in metric_names:
                    values = [float(row[name]) for row in group]
                    record[name] = statistics.fmean(values)
                    record[f"{name}_median"] = statistics.median(values)
                record["no_refine_k1_minus_k0_db"] = statistics.fmean(
                    row["no_refine_k1_psnr"] - row["k0_psnr"] for row in group
                )
                record["teacher_k1_minus_k0_db"] = statistics.fmean(
                    row["teacher_forced_k1_psnr"] - row["k0_psnr"] for row in group
                )
                record["rollout_state_penalty_db"] = statistics.fmean(
                    row["no_refine_k1_psnr"] - row["teacher_forced_k1_psnr"]
                    for row in group
                )
                summaries.append(record)
        payload = {
            "status": "DIAGNOSTIC_COMPLETE",
            "scientific_gate_replacement": False,
            "authorized_physical_npus": visible,
            "validation_trajectories": len(keys),
            "rows": len(combined),
            "states": [
                {
                    "label": label,
                    "checkpoint": str(path),
                    "state_key": state_key,
                    "sha256": sha256_file(path),
                }
                for label, path, state_key in states
            ],
            "summaries": summaries,
            "wall_seconds": time.monotonic() - start,
        }
        (args.output_dir / "diagnostic_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "DIAGNOSTIC_COMPLETE").write_text("complete\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    barrier()
    archive.store.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

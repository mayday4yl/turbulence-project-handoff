#!/usr/bin/env python3
"""Frozen Stage A1 gate on the preselected 100-trajectory validation subset."""

from __future__ import annotations

import argparse
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

from physical_bridge.data import BridgeArchive
from physical_bridge.metrics import per_sample_psnr
from physical_bridge.model import DualHeadUNet
from physical_bridge.simulator import PiRNP2SSimulator
from physical_bridge.utils import atomic_touch, atomic_write_bytes, atomic_write_json, sha256_file
from physical_bridge.v02_losses import per_sample_mse


def load_model(document: dict[str, Any], state: dict[str, torch.Tensor], device: torch.device) -> DualHeadUNet:
    model = DualHeadUNet(**document["model"]).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def xi_statistics(prediction: Any) -> dict[str, float]:
    z = prediction.realization.epsilon_zernike.float().flatten(1)
    t = prediction.realization.epsilon_tilt.float().flatten(1)
    return {
        "zernike_norm": float(z.norm(dim=1)[0].cpu()),
        "zernike_std": float(z.std(dim=1)[0].cpu()),
        "tilt_norm": float(t.norm(dim=1)[0].cpu()),
        "tilt_std": float(t.std(dim=1)[0].cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--val-subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible != document["authorized_physical_npus"]:
        raise RuntimeError(f"Physical device authorization mismatch: {visible}")
    if not (args.pilot_dir / "TRAINING_COMPLETE").exists():
        raise RuntimeError("A1 training is not complete")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4:
        raise RuntimeError("A1 evaluation requires four NPUs")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.distributed.init_process_group(backend="hccl", timeout=dt.timedelta(hours=6))
    barrier = torch.distributed.barrier

    baseline_checkpoint = torch.load(
        document["v01_checkpoint"], map_location="cpu", weights_only=False
    )
    pilot_checkpoint_path = args.pilot_dir / "checkpoints/update-003000.pt"
    pilot_checkpoint = torch.load(pilot_checkpoint_path, map_location="cpu", weights_only=False)
    baseline = load_model(document, baseline_checkpoint["model"], device)
    pilot = load_model(document, pilot_checkpoint["ema"], device)
    del baseline_checkpoint, pilot_checkpoint
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
    subset = json.loads(args.val_subset.read_text(encoding="utf-8"))
    keys = [key for bucket in sorted(subset, key=float) for key in subset[bucket]]
    if len(keys) != 100 or any(len(value) != 20 for value in subset.values()):
        raise ValueError("Frozen A1 validation subset must be 100 trajectories, 20 per bucket")
    archive = BridgeArchive(document["dataset_root"], "val")
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for key in keys[rank::world_size]:
        record = archive.by_key[key]
        terminal_g = float(record["terminal_G"])
        image = archive.state(record, terminal_g).unsqueeze(0).to(device)
        clean = archive.clean(record).unsqueeze(0).to(device)
        severity = torch.tensor([terminal_g], device=device)
        with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
            baseline_prediction = baseline(image, severity)
            pilot_prediction = pilot(image, severity)
        row: dict[str, Any] = {
            "key": key,
            "G": terminal_g,
            "baseline_k0_psnr": float(
                per_sample_psnr(baseline_prediction.clean, clean)[0].cpu()
            ),
            "pilot_k0_psnr": float(per_sample_psnr(pilot_prediction.clean, clean)[0].cpu()),
            "baseline_xi": xi_statistics(baseline_prediction),
            "pilot_xi": xi_statistics(pilot_prediction),
        }
        if terminal_g == 4.50:
            next_g = 3.75
            true_next = archive.state(record, next_g).unsqueeze(0).to(device)
            with torch.no_grad():
                baseline_next = simulator.render(
                    baseline_prediction.clean.float(), baseline_prediction.realization, next_g
                ).output
                pilot_next = simulator.render(
                    pilot_prediction.clean.float(), pilot_prediction.realization, next_g
                ).output
            with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
                baseline_final = baseline(
                    baseline_next, torch.tensor([next_g], device=device)
                ).clean
                pilot_final = pilot(pilot_next, torch.tensor([next_g], device=device)).clean
            row.update(
                {
                    "baseline_next_state_mse": float(
                        per_sample_mse(baseline_next, true_next)[0].cpu()
                    ),
                    "pilot_next_state_mse": float(per_sample_mse(pilot_next, true_next)[0].cpu()),
                    "baseline_k1_psnr": float(per_sample_psnr(baseline_final, clean)[0].cpu()),
                    "pilot_k1_psnr": float(per_sample_psnr(pilot_final, clean)[0].cpu()),
                }
            )
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank-{rank:02d}.jsonl"
    atomic_write_bytes(
        rank_path,
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows),
    )
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
        if len(combined) != 100:
            raise ValueError(f"Expected 100 evaluation rows, got {len(combined)}")
        g450 = [row for row in combined if float(row["G"]) == 4.50]
        mean = lambda name, values=combined: statistics.fmean(float(row[name]) for row in values)
        baseline_mse = mean("baseline_next_state_mse", g450)
        pilot_mse = mean("pilot_next_state_mse", g450)
        next_reduction = 1.0 - pilot_mse / baseline_mse
        k1_gain = mean("pilot_k1_psnr", g450) - mean("baseline_k1_psnr", g450)
        k0_delta = mean("pilot_k0_psnr") - mean("baseline_k0_psnr")
        xi_ratios: dict[str, float] = {}
        xi_ok = True
        for name in ("zernike_norm", "zernike_std", "tilt_norm", "tilt_std"):
            baseline_median = statistics.median(row["baseline_xi"][name] for row in combined)
            pilot_median = statistics.median(row["pilot_xi"][name] for row in combined)
            ratio = pilot_median / baseline_median if baseline_median > 0 else math.inf
            xi_ratios[name] = ratio
            xi_ok = xi_ok and 0.1 <= ratio <= 10.0 and math.isfinite(ratio)
        training = json.loads((args.pilot_dir / "training_summary.json").read_text())
        stability = training["stability"]
        full_t_ratio = (
            stability["full_trajectory_last_200_median"]
            / stability["full_trajectory_first_200_median"]
        )
        bridge_ratio = (
            stability["bridge_last_200_median"] / stability["bridge_first_200_median"]
        )
        gates = {
            "next_state_mse_reduction_at_least_25pct": next_reduction >= 0.25,
            "k1_psnr_gain_at_least_0p1db": k1_gain >= 0.1,
            "k0_psnr_drop_no_more_than_0p1db": k0_delta >= -0.1,
            "full_trajectory_last200_at_most_95pct_first200": full_t_ratio <= 0.95,
            "bridge_last200_at_most_95pct_first200": bridge_ratio <= 0.95,
            "xi_scale_stable": xi_ok,
            "finite": all(
                math.isfinite(value)
                for value in (baseline_mse, pilot_mse, next_reduction, k1_gain, k0_delta)
            ),
        }
        status = "PASS" if all(gates.values()) else "STOP"
        report = {
            "stage": "A1",
            "status": status,
            "validation_trajectories": len(combined),
            "g450_trajectories": len(g450),
            "metrics": {
                "baseline_next_state_mse": baseline_mse,
                "pilot_next_state_mse": pilot_mse,
                "next_state_mse_reduction": next_reduction,
                "baseline_k1_psnr": mean("baseline_k1_psnr", g450),
                "pilot_k1_psnr": mean("pilot_k1_psnr", g450),
                "k1_psnr_gain_db": k1_gain,
                "baseline_k0_psnr": mean("baseline_k0_psnr"),
                "pilot_k0_psnr": mean("pilot_k0_psnr"),
                "k0_psnr_delta_db": k0_delta,
                "full_trajectory_last_first_ratio": full_t_ratio,
                "bridge_last_first_ratio": bridge_ratio,
                "xi_median_ratios": xi_ratios,
            },
            "gates": gates,
            "pilot_checkpoint": str(pilot_checkpoint_path),
            "pilot_checkpoint_sha256": sha256_file(pilot_checkpoint_path),
            "wall_seconds": time.monotonic() - start,
        }
        atomic_write_json(args.output_dir / "v02_pilot_report.json", report)
        lines = [
            "# V0.2 Stage A1 pilot report",
            "",
            f"Status: **{status}**",
            "",
            f"- Next-state MSE reduction: `{next_reduction:.3%}`",
            f"- K=1 no-refine PSNR gain: `{k1_gain:.4f} dB`",
            f"- K=0 PSNR delta: `{k0_delta:.4f} dB`",
            f"- Full L_T last/first ratio: `{full_t_ratio:.6f}`",
            f"- L_B last/first ratio: `{bridge_ratio:.6f}`",
            f"- Xi scale gate: `{xi_ok}`",
            "",
            "Failed gates: "
            + (", ".join(name for name, passed in gates.items() if not passed) or "none"),
        ]
        (args.output_dir / "v02_pilot_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        atomic_touch(args.output_dir / ("A1_PASS" if status == "PASS" else "A1_STOP"))
    barrier()
    archive.store.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

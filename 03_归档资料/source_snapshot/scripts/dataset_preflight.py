#!/usr/bin/env python3
"""Replay, save/load, indexed-loader, and full model smoke for a frozen mini dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left.detach().float() - right.detach().float()).abs().cpu()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.double().mean())}


def expand_realization(realization: Any, count: int) -> Any:
    from physical_bridge.simulator import Realization

    return Realization(
        realization.epsilon_zernike.unsqueeze(0).expand(count, -1, -1, -1),
        realization.epsilon_tilt.unsqueeze(0).expand(count, -1, -1, -1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    from physical_bridge.data import BridgeArchive, FROZEN_G_GRID, REQUIRED_SUFFIXES
    from physical_bridge.metrics import per_sample_psnr
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator, Realization
    from physical_bridge.utils import atomic_touch, atomic_write_json, parameter_grad_norm

    config = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.reset_peak_memory_stats(device)
    simulator = PiRNP2SSimulator(
        config["asset_dir"],
        image_size=int(config["image_size"]),
        rf_range=int(config["rf_range"]),
        aperture_diameter=float(config["aperture_diameter"]),
        propagation_distance=float(config["propagation_distance"]),
        wavelength=float(config["wavelength"]),
        correlation=float(config["correlation"]),
        tilt_threshold=float(config["tilt_threshold"]),
        adjustment=float(config["adjustment"]),
        device=device,
    )
    archive = BridgeArchive(args.dataset_root, args.split)
    if len(archive.records) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} records, got {len(archive.records)}")
    severities = torch.tensor(FROZEN_G_GRID, device=device, dtype=torch.float32)
    replay_maxima = []
    stored_maxima = []
    seed_reload_maxima = []
    psnr_curves = []
    checksum_failures = []
    start = time.monotonic()
    first_payload = None
    for record in archive.records:
        clean_cpu = archive.clean(record)
        states_cpu = torch.stack([archive.state(record, g) for g in FROZEN_G_GRID])
        realization_cpu = archive.realization(record)
        metadata = archive.metadata(record)
        if tuple(realization_cpu.epsilon_zernike.shape) != (36, 512, 512):
            raise ValueError(f"Bad Zernike shape for {record['key']}")
        if tuple(realization_cpu.epsilon_tilt.shape) != (512, 512, 2):
            raise ValueError(f"Bad tilt shape for {record['key']}")
        for suffix in REQUIRED_SUFFIXES:
            if suffix == "json":
                continue
            raw = archive.store.read(record["shard"], f"{record['key']}.{suffix}")
            digest = hashlib.sha256(raw).hexdigest()
            if digest != metadata["checksums"][suffix]:
                checksum_failures.append(f"{record['key']}.{suffix}")
        clean = clean_cpu.unsqueeze(0).to(device)
        realization = Realization(
            realization_cpu.epsilon_zernike.unsqueeze(0).to(device),
            realization_cpu.epsilon_tilt.unsqueeze(0).to(device),
        )
        seeded = simulator.sample_realization(
            1,
            seed_zernike=int(record["seeds"]["epsilon_zernike"]),
            seed_tilt=int(record["seeds"]["epsilon_tilt"]),
        )
        seed_reload_maxima.append(
            max(
                error(realization.epsilon_zernike, seeded.epsilon_zernike)["max_abs"],
                error(realization.epsilon_tilt, seeded.epsilon_tilt)["max_abs"],
            )
        )
        expanded = Realization(
            realization.epsilon_zernike.expand(len(FROZEN_G_GRID), -1, -1, -1),
            realization.epsilon_tilt.expand(len(FROZEN_G_GRID), -1, -1, -1),
        )
        with torch.no_grad():
            prepared = simulator.prepare_image(clean)
            first = simulator.render_prepared(prepared, expanded, severities).output
            replay = simulator.render_prepared(prepared, expanded, severities).output
        replay_maxima.append(error(first, replay)["max_abs"])
        stored_maxima.append(error(first.cpu(), states_cpu)["max_abs"])
        if not bool(torch.isfinite(first).all().cpu()):
            raise FloatingPointError(f"Non-finite replay for {record['key']}")
        curve = per_sample_psnr(first, clean.expand_as(first)).cpu().tolist()
        psnr_curves.append(curve)
        if first_payload is None:
            first_payload = (clean, states_cpu, realization)
    if checksum_failures:
        raise IOError(f"Checksum failures: {checksum_failures[:5]}")
    if max(replay_maxima) >= 1e-5:
        raise AssertionError(f"Replay threshold failed: {max(replay_maxima)}")
    if max(seed_reload_maxima) != 0:
        raise AssertionError(f"Saved realization differs from seeded source: {max(seed_reload_maxima)}")

    assert first_payload is not None
    clean, states_cpu, realization = first_payload
    zernike = realization.epsilon_zernike.detach().clone().requires_grad_(True)
    tilt = realization.epsilon_tilt.detach().clone().requires_grad_(True)
    differentiable = simulator.render(clean, Realization(zernike, tilt), 4.5).output
    simulator_loss = differentiable.square().mean()
    simulator_loss.backward()
    simulator_backward = {
        "loss": float(simulator_loss.detach().cpu()),
        "zernike_grad_finite": bool(torch.isfinite(zernike.grad).all().cpu()),
        "zernike_grad_norm": float(zernike.grad.float().norm().cpu()),
        "tilt_grad_finite": bool(torch.isfinite(tilt.grad).all().cpu()),
        "tilt_grad_norm": float(tilt.grad.float().norm().cpu()),
    }

    model = DualHeadUNet().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    image = states_cpu[-1:].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        prediction = model(image, torch.tensor([4.5], device=device))
    clean_loss = (prediction.clean - clean).square().mean()
    reconstruction = simulator.render(clean, prediction.realization, 4.5).output
    degradation_loss = (reconstruction - image).square().mean()
    total_loss = clean_loss + 0.1 * degradation_loss
    total_loss.backward()
    gradient_norms = {
        name: parameter_grad_norm(parameters)
        for name, parameters in model.branch_parameters().items()
    }
    preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).cpu())
    optimizer.step()
    if device.type == "npu":
        torch.npu.synchronize(device)
    curves = np.asarray(psnr_curves, dtype=np.float64)
    result = {
        "status": "PASS",
        "scientific_result": False,
        "note": "Mini preflight validates implementation only.",
        "trajectory_count": len(archive.records),
        "simulator_protocol": simulator.protocol(),
        "replay": {
            "threshold": 1e-5,
            "max_abs": float(max(replay_maxima)),
            "mean_of_max_abs": float(np.mean(replay_maxima)),
            "max_abs_per_trajectory": replay_maxima,
        },
        "stored_float16_roundtrip": {
            "max_abs": float(max(stored_maxima)),
            "mean_of_max_abs": float(np.mean(stored_maxima)),
            "max_abs_per_trajectory": stored_maxima,
        },
        "saved_xi_vs_seeded_xi_max_abs": float(max(seed_reload_maxima)),
        "checksum_failures": checksum_failures,
        "severity_psnr_mean": {
            f"{g:.2f}": float(curves[:, index].mean())
            for index, g in enumerate(FROZEN_G_GRID)
        },
        "fraction_psnr_g075_above_g450": float((curves[:, 0] > curves[:, -1]).mean()),
        "simulator_backward": simulator_backward,
        "full_model_step": {
            "clean_loss": float(clean_loss.detach().cpu()),
            "degradation_loss": float(degradation_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "preclip_gradient_norm": preclip,
            "branch_gradient_norms": gradient_norms,
            "finite": bool(torch.isfinite(total_loss).cpu()),
        },
        "simulator_parameter_grads_are_none": all(
            value is None for value in simulator.simulator_parameter_gradients().values()
        ),
        "wall_seconds": time.monotonic() - start,
    }
    if device.type == "npu":
        result["npu_memory"] = {
            "peak_allocated": int(torch.npu.max_memory_allocated(device)),
            "peak_reserved": int(torch.npu.max_memory_reserved(device)),
        }
    atomic_write_json(args.output, result)
    atomic_touch(args.output.parent / "PREFLIGHT_COMPLETE")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

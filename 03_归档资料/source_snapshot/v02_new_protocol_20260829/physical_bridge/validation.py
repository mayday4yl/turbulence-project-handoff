"""Teacher-forced validation for checkpoint selection and branch diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import BridgeArchive, FROZEN_G_GRID
from .metrics import LPIPSMetric, per_sample_l1, per_sample_psnr, per_sample_ssim
from .utils import atomic_write_bytes, atomic_write_json


def _component_diagnostics(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    batch = predicted.shape[0]
    left = predicted.float().reshape(batch, -1)
    right = target.float().reshape(batch, -1)
    difference = (left - right).square().mean(1)
    left_centered = left - left.mean(1, keepdim=True)
    right_centered = right - right.mean(1, keepdim=True)
    correlation = (left_centered * right_centered).mean(1) / (
        left_centered.square().mean(1).sqrt()
        * right_centered.square().mean(1).sqrt()
    ).clamp_min(1e-12)
    return {
        "mse": difference,
        "correlation": correlation,
        "mean": left.mean(1),
        "std": left.std(1),
        "norm": left.square().sum(1).sqrt(),
    }


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )


def _tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


@torch.no_grad()
def save_qualitative_predictions(
    *,
    model: torch.nn.Module,
    archive: BridgeArchive,
    keys: list[str],
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tensor_payload: dict[str, np.ndarray] = {}
    severities = torch.tensor(FROZEN_G_GRID, device=device)
    for key in keys:
        record = archive.by_key[key]
        clean = archive.clean(record)
        inputs = torch.stack([archive.state(record, g) for g in FROZEN_G_GRID]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            prediction = model(inputs, severities).clean
        tensor_payload[key] = prediction.half().cpu().numpy()
        rows.append((key, clean, prediction.cpu()))
    temporary_npz = output_dir / "clean_predictions.npz.tmp"
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **tensor_payload)
    temporary_npz.replace(output_dir / "clean_predictions.npz")

    labels = ("target", "pred .75", "pred 1.50", "pred 2.25", "pred 3.00", "pred 3.75", "pred 4.50")
    tile, header = 256, 24
    canvas = Image.new("RGB", (len(labels) * tile, len(rows) * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (key, clean, prediction) in enumerate(rows):
        images = [_tensor_image(clean), *[_tensor_image(item) for item in prediction]]
        top = row_index * (tile + header)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            left = column * tile
            canvas.paste(image, (left, top + header))
            draw.text((left + 4, top + 4), f"{key} {label}", fill="black")
    temporary_png = output_dir / "montage.tmp.png"
    canvas.save(temporary_png, format="PNG", compress_level=3)
    temporary_png.replace(output_dir / "montage.png")


def teacher_forced_validation(
    *,
    model: torch.nn.Module,
    simulator: torch.nn.Module,
    dataset_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    rank: int,
    world_size: int,
    split: str = "val",
    qualitative_keys: list[str] | None = None,
    distributed_barrier: Any | None = None,
) -> dict[str, Any] | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = BridgeArchive(dataset_root, split)
    lpips_metric = LPIPSMetric(device)
    records = []
    model.eval()
    start = time.monotonic()
    severity_tensor = torch.tensor(FROZEN_G_GRID, device=device, dtype=torch.float32)
    for record in archive.records[rank::world_size]:
        clean_cpu = archive.clean(record)
        clean = clean_cpu.unsqueeze(0).to(device).expand(len(FROZEN_G_GRID), -1, -1, -1)
        inputs = torch.stack([archive.state(record, g) for g in FROZEN_G_GRID]).to(device)
        terminal = archive.state(record, float(record["terminal_G"])).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            prediction = model(inputs, severity_tensor)
        reconstructed = simulator.render(
            clean,
            prediction.realization,
            torch.full_like(severity_tensor, float(record["terminal_G"])),
        ).output
        true_realization = archive.realization(record)
        true_z = true_realization.epsilon_zernike.unsqueeze(0).to(device).expand_as(
            prediction.realization.epsilon_zernike
        )
        true_tilt = true_realization.epsilon_tilt.unsqueeze(0).to(device).expand_as(
            prediction.realization.epsilon_tilt
        )
        z_diag = _component_diagnostics(prediction.realization.epsilon_zernike, true_z)
        tilt_diag = _component_diagnostics(prediction.realization.epsilon_tilt, true_tilt)
        l1 = per_sample_l1(prediction.clean, clean)
        psnr = per_sample_psnr(prediction.clean, clean)
        ssim = per_sample_ssim(prediction.clean, clean)
        lpips_values = lpips_metric(prediction.clean.float(), clean.float())
        degradation_residual = (reconstructed - terminal).square().flatten(1).mean(1)
        for index, severity in enumerate(FROZEN_G_GRID):
            records.append(
                {
                    "key": record["key"],
                    "g": severity,
                    "terminal_G": float(record["terminal_G"]),
                    "l1": float(l1[index].cpu()),
                    "psnr": float(psnr[index].cpu()),
                    "ssim": float(ssim[index].cpu()),
                    "lpips": float(lpips_values[index].cpu()),
                    "degradation_residual": float(degradation_residual[index].cpu()),
                    "predicted_zernike_mean": float(z_diag["mean"][index].cpu()),
                    "predicted_zernike_std": float(z_diag["std"][index].cpu()),
                    "predicted_zernike_norm": float(z_diag["norm"][index].cpu()),
                    "zernike_true_mse": float(z_diag["mse"][index].cpu()),
                    "zernike_true_correlation": float(z_diag["correlation"][index].cpu()),
                    "predicted_tilt_mean": float(tilt_diag["mean"][index].cpu()),
                    "predicted_tilt_std": float(tilt_diag["std"][index].cpu()),
                    "predicted_tilt_norm": float(tilt_diag["norm"][index].cpu()),
                    "tilt_true_mse": float(tilt_diag["mse"][index].cpu()),
                    "tilt_true_correlation": float(tilt_diag["correlation"][index].cpu()),
                }
            )
    atomic_write_bytes(output_dir / f"rank-{rank:02d}.jsonl", _jsonl(records))
    if distributed_barrier is not None:
        distributed_barrier()
    if rank != 0:
        if distributed_barrier is not None:
            distributed_barrier()
        model.train()
        return None

    combined = []
    for worker_rank in range(world_size):
        path = output_dir / f"rank-{worker_rank:02d}.jsonl"
        with path.open(encoding="utf-8") as handle:
            combined.extend(json.loads(line) for line in handle if line.strip())
    expected = len(archive.records) * len(FROZEN_G_GRID)
    if len(combined) != expected:
        raise ValueError(f"Validation expected {expected} rows, got {len(combined)}")
    per_g: dict[str, dict[str, float]] = {}
    metric_names = (
        "l1",
        "psnr",
        "ssim",
        "lpips",
        "degradation_residual",
        "predicted_zernike_mean",
        "predicted_zernike_std",
        "predicted_zernike_norm",
        "zernike_true_mse",
        "zernike_true_correlation",
        "predicted_tilt_mean",
        "predicted_tilt_std",
        "predicted_tilt_norm",
        "tilt_true_mse",
        "tilt_true_correlation",
    )
    for severity in FROZEN_G_GRID:
        rows = [row for row in combined if float(row["g"]) == severity]
        per_g[f"{severity:.2f}"] = {
            name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names
        }
    summary = {
        "trajectory_count": len(archive.records),
        "row_count": len(combined),
        "equal_severity_mean_psnr": float(np.mean([value["psnr"] for value in per_g.values()])),
        "equal_severity_mean_lpips": float(
            np.mean([value["lpips"] for value in per_g.values()])
        ),
        "equal_severity_mean_degradation_residual": float(
            np.mean([value["degradation_residual"] for value in per_g.values()])
        ),
        "per_g": per_g,
        "wall_seconds": time.monotonic() - start,
    }
    atomic_write_bytes(output_dir / "all.jsonl", _jsonl(combined))
    atomic_write_json(output_dir / "summary.json", summary)
    if qualitative_keys:
        save_qualitative_predictions(
            model=model,
            archive=archive,
            keys=qualitative_keys,
            device=device,
            output_dir=output_dir / "qualitative",
        )
    if distributed_barrier is not None:
        distributed_barrier()
    model.train()
    return summary

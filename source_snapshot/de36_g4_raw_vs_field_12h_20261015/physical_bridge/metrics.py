"""Frozen RGB quality metrics and paired bootstrap statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def per_sample_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).abs().flatten(1).mean(1)


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def per_sample_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(per_sample_mse(prediction, target).clamp_min(1e-12))


def _gaussian_window(
    channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords.square()) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).to(dtype=dtype)
    return window.expand(channels, 1, size, size).contiguous()


def per_sample_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    window = _gaussian_window(channels, prediction.device, prediction.dtype)
    mu_x = F.conv2d(prediction, window, padding=5, groups=channels)
    mu_y = F.conv2d(target, window, padding=5, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(prediction.square(), window, padding=5, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(target.square(), window, padding=5, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=5, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(1)


class LPIPSMetric:
    def __init__(self, device: torch.device, net: str = "alex") -> None:
        import lpips

        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(prediction * 2.0 - 1.0, target * 2.0 - 1.0).flatten()


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_bootstrap(
    deltas: list[float] | np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Paired bootstrap requires at least one delta")
    generator = np.random.default_rng(int(seed))
    means = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "n": int(values.size),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "paired_wins": int((values > 0).sum()),
        "paired_ties": int((values == 0).sum()),
        "paired_losses": int((values < 0).sum()),
        "bootstrap_samples": samples,
        "bootstrap_95_ci": [float(lower), float(upper)],
    }


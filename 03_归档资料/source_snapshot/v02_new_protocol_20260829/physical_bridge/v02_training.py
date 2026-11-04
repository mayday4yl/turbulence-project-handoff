"""Training-state helpers shared by frozen V0.2 stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .simulator import Realization
from .utils import sha256_file
from .v02_losses import per_sample_mse


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], device: torch.device) -> None:
        self.shadow = {name: value.to(device) for name, value in state.items()}


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


@torch.no_grad()
def full_trajectory_loss_sum(
    prediction: Any,
    batch: dict[str, Any],
    simulator: torch.nn.Module,
) -> torch.Tensor:
    """Sum per-sample full L_T over unique {G,g,h}; used only as a diagnostic."""

    # Each frozen set {G,g,h} contains at most three severities.  Pad to three
    # rows per sample, render once at a fixed batch shape, and give padded rows
    # zero reduction weight.  The resulting value is exactly the sum of each
    # sample's mean over its unique severities.
    device = batch["image"].device
    slots = 3
    severities: list[float] = []
    targets: list[torch.Tensor] = []
    weights: list[float] = []
    for spec in batch["specs"]:
        targets_by_g = spec["full_targets"]
        values = sorted(targets_by_g)
        if not 1 <= len(values) <= slots:
            raise ValueError(f"Expected 1..{slots} unique trajectory severities, got {values}")
        count = len(values)
        fallback = values[0]
        for slot in range(slots):
            severity = values[slot] if slot < count else fallback
            severities.append(float(severity))
            targets.append(targets_by_g[severity])
            weights.append(1.0 / count if slot < count else 0.0)

    clean = batch["clean"].float().repeat_interleave(slots, dim=0)
    zernike = prediction.realization.epsilon_zernike.detach().repeat_interleave(slots, dim=0)
    tilt = prediction.realization.epsilon_tilt.detach().repeat_interleave(slots, dim=0)
    target = torch.stack(targets).to(device, non_blocking=True)
    rendered = simulator.render(
        clean,
        Realization(zernike, tilt),
        torch.tensor(severities, device=device, dtype=torch.float32),
    ).output
    values = per_sample_mse(rendered, target)
    return (values * torch.tensor(weights, device=device, dtype=torch.float32)).sum()

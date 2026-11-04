"""V0.3 training-only rollout around the unchanged V0.1 network and simulator.

The DDP boundary is the entire rollout. All image and realization tensors remain
connected across predictions; only diagnostic copies are detached.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
from torch import nn


def full_image_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One FP32 MSE per image, with no crop and no averaging across paths."""
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def rank_path_counts(global_paths: int, world_size: int) -> list[int]:
    if global_paths <= 0 or world_size <= 0:
        raise ValueError("Positive path count and world size required")
    quotient, remainder = divmod(global_paths, world_size)
    return [quotient + int(rank < remainder) for rank in range(world_size)]


def ddp_loss_scale(global_paths: int, world_size: int) -> float:
    """DDP averages ranks: scale each local path SUM by world/global paths."""
    return float(world_size) / float(global_paths)


class RolloutModule(nn.Module):
    def __init__(self, network: nn.Module, simulator: nn.Module,
                 lambda_R: float, *, network_bfloat16: bool = True) -> None:
        super().__init__()
        if lambda_R <= 0:
            raise ValueError("V0.3 requires a positive reconstruction coefficient")
        self.network = network
        self.simulator = simulator
        self.lambda_R = float(lambda_R)
        self.network_bfloat16 = bool(network_bfloat16)
        if any(parameter.requires_grad for parameter in simulator.parameters()):
            raise ValueError("The original simulator must remain frozen")

    def train(self, mode: bool = True) -> "RolloutModule":
        super().train(mode)
        self.simulator.eval()
        return self

    def forward(self, clean: torch.Tensor, observation: torch.Tensor,
                severities: torch.Tensor, schedules: Sequence[Sequence[float]],
                path_weights: torch.Tensor | None = None, *,
                retain_intermediates: bool = False) -> dict[str, Any]:
        if clean.shape != observation.shape or clean.ndim != 4:
            raise ValueError("Expected equally shaped [B,C,H,W] clean/observation")
        batch = len(schedules)
        if batch != clean.shape[0] or severities.numel() != batch:
            raise ValueError("One original G and schedule per path required")
        paths = [tuple(float(g) for g in schedule) for schedule in schedules]
        original_g = severities.float().reshape(-1)
        # Schedule authority and admissibility are enforced by the sampler.
        if any(not schedule for schedule in paths):
            raise ValueError("Every path includes at least one prediction")
        lengths = torch.tensor([len(path) for path in paths], device=clean.device,
                               dtype=torch.float32)
        if path_weights is None:
            path_weights = torch.ones(batch, device=clean.device)
        weights = path_weights.to(device=clean.device, dtype=torch.float32)
        if weights.numel() != batch:
            raise ValueError("One contribution weight per path required")
        clean_sums = [clean.new_zeros((), dtype=torch.float32) for _ in paths]
        recon_sums = [clean.new_zeros((), dtype=torch.float32) for _ in paths]
        active = list(range(batch))
        state = observation.float()
        step_records: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        autocast_device = clean.device.type
        for step in range(max(map(len, paths))):
            indices = torch.tensor(active, device=clean.device, dtype=torch.long)
            current_g = torch.tensor([paths[i][step] for i in active],
                                     device=clean.device, dtype=torch.float32)
            context = (torch.autocast(device_type=autocast_device,
                                     dtype=torch.bfloat16)
                       if self.network_bfloat16 else nullcontext())
            with context:
                prediction = self.network(state, current_g)
            # Explicitly leave network autocast: full simulator and losses FP32.
            with torch.autocast(device_type=autocast_device, enabled=False):
                estimated_clean = prediction.clean.float()
                realization = prediction.realization
                prepared = self.simulator.prepare_image(estimated_clean)
                reconstructed = self.simulator.render_prepared(
                    prepared, realization, original_g.index_select(0, indices)).output
                lc = full_image_mse(estimated_clean, clean.index_select(0, indices))
                lr = full_image_mse(reconstructed, observation.index_select(0, indices))
                for local, path_index in enumerate(active):
                    clean_sums[path_index] = clean_sums[path_index] + lc[local]
                    recon_sums[path_index] = recon_sums[path_index] + lr[local]
                active_weights = weights.index_select(0, indices)
                step_records.append({
                    "step": step,
                    "clean_sum": (lc.detach() * active_weights).sum(),
                    "reconstruction_sum": (lr.detach() * active_weights).sum(),
                    "path_count": active_weights.sum(),
                })
                continuing = [j for j, i in enumerate(active)
                              if step + 1 < len(paths[i])]
                next_state = None
                if continuing:
                    selected = torch.tensor(continuing, device=clean.device,
                                            dtype=torch.long)
                    next_active = [active[j] for j in continuing]
                    next_g = torch.tensor([paths[i][step + 1] for i in next_active],
                                          device=clean.device, dtype=torch.float32)
                    # Reusing this linear image preparation is algebraically
                    # identical to a second original render call, with both
                    # gradient contributions accumulated into estimated_clean.
                    next_prepared = type(prepared)(
                        prepared.image_mean.index_select(0, selected),
                        prepared.dictionary_images.index_select(0, selected))
                    next_realization = type(realization)(
                        realization.epsilon_zernike.index_select(0, selected),
                        realization.epsilon_tilt.index_select(0, selected))
                    next_state = self.simulator.render_prepared(
                        next_prepared, next_realization, next_g).output
                if retain_intermediates:
                    retained = {
                        "step": step, "active_path_indices": tuple(active),
                        "input": state, "clean": estimated_clean,
                        "epsilon_zernike": realization.epsilon_zernike,
                        "epsilon_tilt": realization.epsilon_tilt,
                        "next_state": next_state,
                        "clean_losses": lc, "reconstruction_losses": lr,
                    }
                    for value in retained.values():
                        if torch.is_tensor(value) and value.requires_grad:
                            value.retain_grad()
                    trace.append(retained)
                if next_state is not None:
                    # No detach, cloning, GT replacement, or refinement here.
                    state = next_state
                    active = next_active
        path_clean = torch.stack(clean_sums) / lengths
        path_reconstruction = torch.stack(recon_sums) / lengths
        clean_loss_sum = (path_clean * weights).sum()
        reconstruction_loss_sum = (path_reconstruction * weights).sum()
        return {
            "loss_sum": clean_loss_sum + self.lambda_R * reconstruction_loss_sum,
            "clean_loss_sum": clean_loss_sum,
            "reconstruction_loss_sum": reconstruction_loss_sum,
            "path_clean": path_clean.detach(),
            "path_reconstruction": path_reconstruction.detach(),
            "path_count": weights.sum(),
            "prediction_count": (lengths * weights).sum(),
            "steps": step_records,
            "trace": trace,
        }

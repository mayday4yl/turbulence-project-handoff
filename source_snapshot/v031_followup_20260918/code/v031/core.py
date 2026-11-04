"""Full predicted-state BPTT for the fixed V0.31 four-severity method.

This module contains the differentiable method only, without an optimizer or
training launch. Network and simulator definitions are supplied unchanged by
the caller. Every complete path contributes equally, independently of length.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn


SEVERITIES = (4, 3, 2, 1)
LAMBDA_R = 0.2


def severity_path(initial_G: float) -> tuple[int, ...]:
    """The complete training tail; K counts predictions, not transitions."""
    value = float(initial_G)
    if value not in SEVERITIES:
        raise ValueError(f"V0.31 initial_G must be 4, 3, 2 or 1, got {value}")
    return SEVERITIES[SEVERITIES.index(value):]


def full_image_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One FP32 full-image mean squared error per RGB image."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("MSE requires equal [B,3,H,W] tensors")
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def ddp_loss_scale(global_paths: int, world_size: int) -> float:
    """Scale local path SUM before backward when DDP averages rank gradients."""
    if global_paths <= 0 or world_size <= 0:
        raise ValueError("Positive global_paths and world_size required")
    return float(world_size) / float(global_paths)


class RolloutModule(nn.Module):
    def __init__(self, network: nn.Module, simulator: nn.Module,
                 lambda_R: float, *, network_bfloat16: bool = False) -> None:
        super().__init__()
        if not math.isfinite(float(lambda_R)) or float(lambda_R) != LAMBDA_R:
            raise ValueError("The V0.31 addendum freezes lambda_R at 0.2")
        if any(parameter.requires_grad for parameter in simulator.parameters()):
            raise ValueError("Simulator parameters must be frozen before construction")
        if any(parameter.is_floating_point() and parameter.dtype != torch.float32
               for parameter in network.parameters()):
            raise ValueError("Network master parameters must remain FP32")
        self.network = network
        self.simulator = simulator
        self.lambda_R = float(lambda_R)
        self.network_bfloat16 = bool(network_bfloat16)
        self.simulator.eval()

    def train(self, mode: bool = True) -> "RolloutModule":
        super().train(mode)
        self.simulator.eval()
        return self

    def forward(self, clean: torch.Tensor, observation: torch.Tensor,
                severities: torch.Tensor,
                schedules: Sequence[Sequence[float]] | None = None, *,
                retain_intermediates: bool = False,
                collect_step_metrics: bool = False) -> dict[str, Any]:
        if clean.shape != observation.shape or clean.ndim != 4 or clean.shape[1] != 3:
            raise ValueError("Expected equally shaped [B,3,H,W] clean/observation")
        if clean.shape[0] == 0 or severities.numel() != clean.shape[0]:
            raise ValueError("A nonempty batch and one original G per path are required")
        if clean.device != observation.device:
            raise ValueError("Clean and observation must share a device")
        if clean.dtype != torch.float32 or observation.dtype != torch.float32:
            raise ValueError("Authoritative clean and observed states must be FP32")
        original_g = severities.to(device=clean.device, dtype=torch.float32).reshape(-1)
        # Severity metadata is not a learned tensor. Its conversion never touches
        # a predicted clean, native realization, or generated state.
        paths = [severity_path(g) for g in original_g.tolist()]
        if schedules is not None:
            supplied = [tuple(float(g) for g in path) for path in schedules]
            if supplied != paths:
                raise ValueError("Training schedules must equal each original G's complete tail")
        batch = len(paths)
        lengths = torch.tensor([len(path) for path in paths], device=clean.device,
                               dtype=torch.float32)
        clean_sums = [clean.new_zeros(()) for _ in paths]
        reconstruction_sums = [clean.new_zeros(()) for _ in paths]
        active = list(range(batch))
        state = observation
        trace: list[dict[str, Any]] = []
        step_metrics: list[dict[str, Any]] = []
        device_type = clean.device.type
        render_calls = 0

        for step in range(max(map(len, paths))):
            indices = torch.tensor(active, device=clean.device, dtype=torch.long)
            current_g = torch.tensor([paths[i][step] for i in active],
                                     device=clean.device, dtype=torch.float32)
            # Disabled autocast is explicit in reference mode, even if the caller
            # accidentally invokes this module under an outer autocast context.
            network_context = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                               if self.network_bfloat16 else
                               torch.autocast(device_type=device_type, enabled=False))
            with network_context:
                prediction = self.network(state, current_g)
            with torch.autocast(device_type=device_type, enabled=False):
                estimated_clean = prediction.clean.float()
                raw_realization = prediction.realization
                # Cast every native branch explicitly while preserving autograd.
                realization = type(raw_realization)(
                    raw_realization.epsilon_zernike.float(),
                    raw_realization.epsilon_tilt.float())
                prepared = self.simulator.prepare_image(estimated_clean)
                reconstruction_g = original_g.index_select(0, indices)
                target_observation = observation.index_select(0, indices)
                reconstructed = self.simulator.render_prepared(
                    prepared, realization, reconstruction_g).output
                render_calls += 1
                clean_losses = full_image_mse(estimated_clean, clean.index_select(0, indices))
                reconstruction_losses = full_image_mse(reconstructed, target_observation)
                if collect_step_metrics:
                    # Logging copies only; these detached small loss tensors are
                    # never used to compute the differentiable training loss.
                    step_metrics.append({
                        "step": step + 1,
                        "active_path_indices": tuple(active),
                        "current_g": current_g.detach(),
                        "clean_losses": clean_losses.detach(),
                        "reconstruction_losses": reconstruction_losses.detach(),
                    })
                for local, path_index in enumerate(active):
                    clean_sums[path_index] = clean_sums[path_index] + clean_losses[local]
                    reconstruction_sums[path_index] = (reconstruction_sums[path_index]
                                                       + reconstruction_losses[local])
                continuing = [j for j, i in enumerate(active) if step + 1 < len(paths[i])]
                next_state = None
                next_g = None
                next_active: list[int] = []
                if continuing:
                    selected = torch.tensor(continuing, device=clean.device, dtype=torch.long)
                    next_active = [active[j] for j in continuing]
                    next_g = torch.tensor([paths[i][step + 1] for i in next_active],
                                          device=clean.device, dtype=torch.float32)
                    # prepare_image is the unchanged linear image preparation;
                    # reuse accumulates both render gradients into the same clean.
                    next_prepared = type(prepared)(
                        prepared.image_mean.index_select(0, selected),
                        prepared.dictionary_images.index_select(0, selected))
                    next_realization = type(realization)(
                        realization.epsilon_zernike.index_select(0, selected),
                        realization.epsilon_tilt.index_select(0, selected))
                    next_state = self.simulator.render_prepared(
                        next_prepared, next_realization, next_g).output
                    render_calls += 1
                if retain_intermediates:
                    retained = {
                        "step": step, "active_path_indices": tuple(active),
                        "input": state, "clean": estimated_clean,
                        "epsilon_zernike": realization.epsilon_zernike,
                        "epsilon_tilt": realization.epsilon_tilt,
                        "next_state": next_state,
                        "clean_losses": clean_losses,
                        "reconstruction_losses": reconstruction_losses,
                        "current_g": current_g, "original_g": reconstruction_g,
                        "original_observation": target_observation,
                        "reconstructed": reconstructed, "next_g": next_g,
                        "next_active_path_indices": tuple(next_active),
                    }
                    for value in retained.values():
                        if torch.is_tensor(value) and value.requires_grad:
                            value.retain_grad()
                    trace.append(retained)
                if next_state is not None:
                    state = next_state
                    active = next_active

        with torch.autocast(device_type=device_type, enabled=False):
            path_clean = torch.stack(clean_sums) / lengths
            path_reconstruction = torch.stack(reconstruction_sums) / lengths
            path_losses = path_clean + self.lambda_R * path_reconstruction
            loss_sum = path_losses.sum()
            loss_mean = loss_sum / batch
        return {
            "loss_mean": loss_mean, "loss_sum": loss_sum,
            "clean_loss_sum": path_clean.sum(),
            "reconstruction_loss_sum": path_reconstruction.sum(),
            "path_clean": path_clean, "path_reconstruction": path_reconstruction,
            "path_losses": path_losses, "path_count": batch,
            "prediction_count": sum(map(len, paths)),
            "logical_render_count": sum(2 * len(path) - 1 for path in paths),
            "batched_render_call_count": render_calls,
            "training_refinement_count": 0, "trace": trace,
            "step_metrics": step_metrics,
        }

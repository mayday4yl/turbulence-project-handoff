"""Frozen progressive bridge rollout and realization refinement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .data import evenly_spaced_schedule
from .metrics import per_sample_l1, per_sample_psnr, per_sample_ssim
from .simulator import Realization


@dataclass(frozen=True)
class RefinementConfig:
    learning_rate: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    first_iterations: int = 50
    later_iterations: int = 15
    gradient_clip: float = 1.0
    early_stop_patience: int = 5
    early_stop_relative_improvement: float = 1e-4
    valid_border: int = 16
    proximal_weight: float = 1e-2
    prior_weight: float = 1e-4


@dataclass(frozen=True)
class ObjectiveValues:
    total: torch.Tensor
    observation: torch.Tensor
    proximal: torch.Tensor
    prior: torch.Tensor


@dataclass(frozen=True)
class RefinementResult:
    realization: Realization
    initial: dict[str, float]
    best: dict[str, float]
    iterations: int
    simulator_forwards: int
    simulator_backwards: int
    update_norms: dict[str, float]


def clone_realization(value: Realization, *, requires_grad: bool = False) -> Realization:
    return Realization(
        value.epsilon_zernike.detach().clone().requires_grad_(requires_grad),
        value.epsilon_tilt.detach().clone().requires_grad_(requires_grad),
    )


def realization_statistics(value: Realization) -> dict[str, float]:
    result = {}
    for name, tensor in (
        ("zernike", value.epsilon_zernike),
        ("tilt", value.epsilon_tilt),
    ):
        flat = tensor.detach().float().reshape(-1)
        result[f"{name}_mean"] = float(flat.mean().cpu())
        result[f"{name}_std"] = float(flat.std().cpu())
        result[f"{name}_max_abs"] = float(flat.abs().max().cpu())
        result[f"{name}_norm"] = float(flat.square().sum().sqrt().cpu())
    return result


def objective(
    *,
    simulator: torch.nn.Module,
    clean: torch.Tensor,
    realization: Realization,
    initialization: Realization,
    observation: torch.Tensor,
    observation_severity: float,
    config: RefinementConfig,
) -> ObjectiveValues:
    reconstructed = simulator.render(clean, realization, observation_severity).output
    border = config.valid_border
    valid_reconstruction = reconstructed[..., border:-border, border:-border]
    valid_observation = observation[..., border:-border, border:-border]
    observation_loss = (valid_reconstruction - valid_observation).square().mean()
    proximal = 0.5 * (
        realization.epsilon_zernike - initialization.epsilon_zernike
    ).square().mean() + 0.5 * (
        realization.epsilon_tilt - initialization.epsilon_tilt
    ).square().mean()
    prior = 0.5 * realization.epsilon_zernike.square().mean() + 0.5 * (
        realization.epsilon_tilt.square().mean()
    )
    total = (
        observation_loss
        + config.proximal_weight * proximal
        + config.prior_weight * prior
    )
    return ObjectiveValues(total, observation_loss, proximal, prior)


def detached_values(values: ObjectiveValues) -> dict[str, float]:
    return {
        "objective": float(values.total.detach().cpu()),
        "observation_residual": float(values.observation.detach().cpu()),
        "proximal": float(values.proximal.detach().cpu()),
        "prior": float(values.prior.detach().cpu()),
    }


@torch.no_grad()
def initialization_objective(
    *,
    simulator: torch.nn.Module,
    clean: torch.Tensor,
    candidate: Realization,
    observation: torch.Tensor,
    observation_severity: float,
    config: RefinementConfig,
) -> dict[str, float]:
    return detached_values(
        objective(
            simulator=simulator,
            clean=clean,
            realization=candidate,
            initialization=candidate,
            observation=observation,
            observation_severity=observation_severity,
            config=config,
        )
    )


def refine_realization(
    *,
    simulator: torch.nn.Module,
    clean: torch.Tensor,
    initialization: Realization,
    observation: torch.Tensor,
    observation_severity: float,
    maximum_iterations: int,
    config: RefinementConfig,
) -> RefinementResult:
    initial = clone_realization(initialization)
    current = clone_realization(initialization, requires_grad=True)
    optimizer = torch.optim.Adam(
        [current.epsilon_zernike, current.epsilon_tilt],
        lr=config.learning_rate,
        betas=config.betas,
    )
    best = clone_realization(initial)
    with torch.no_grad():
        initial_values = objective(
            simulator=simulator,
            clean=clean,
            realization=current,
            initialization=initial,
            observation=observation,
            observation_severity=observation_severity,
            config=config,
        )
    best_values = detached_values(initial_values)
    previous = best_values["objective"]
    stagnant = 0
    completed = 0
    forward_calls = 1
    for iteration in range(maximum_iterations):
        optimizer.zero_grad(set_to_none=True)
        values = objective(
            simulator=simulator,
            clean=clean,
            realization=current,
            initialization=initial,
            observation=observation,
            observation_severity=observation_severity,
            config=config,
        )
        forward_calls += 1
        if not bool(torch.isfinite(values.total).detach().cpu()):
            raise FloatingPointError(f"Non-finite refinement objective at iteration {iteration}")
        current_value = float(values.total.detach().cpu())
        if current_value < best_values["objective"]:
            best = clone_realization(current)
            best_values = detached_values(values)
        values.total.backward()
        torch.nn.utils.clip_grad_norm_(
            [current.epsilon_zernike, current.epsilon_tilt], config.gradient_clip
        )
        optimizer.step()
        completed = iteration + 1
        relative = (previous - current_value) / max(abs(previous), 1e-12)
        stagnant = stagnant + 1 if relative < config.early_stop_relative_improvement else 0
        previous = current_value
        if stagnant >= config.early_stop_patience:
            break
    with torch.no_grad():
        final_values = objective(
            simulator=simulator,
            clean=clean,
            realization=current,
            initialization=initial,
            observation=observation,
            observation_severity=observation_severity,
            config=config,
        )
    forward_calls += 1
    if float(final_values.total.cpu()) < best_values["objective"]:
        best = clone_realization(current)
        best_values = detached_values(final_values)
    return RefinementResult(
        realization=best,
        initial=detached_values(initial_values),
        best=best_values,
        iterations=completed,
        simulator_forwards=forward_calls,
        simulator_backwards=completed,
        update_norms={
            "zernike": float(
                (best.epsilon_zernike - initial.epsilon_zernike).float().norm().cpu()
            ),
            "tilt": float((best.epsilon_tilt - initial.epsilon_tilt).float().norm().cpu()),
        },
    )


def quality(prediction: torch.Tensor, clean: torch.Tensor, lpips_metric: Any | None) -> dict[str, float]:
    result = {
        "l1": float(per_sample_l1(prediction, clean)[0].cpu()),
        "psnr": float(per_sample_psnr(prediction, clean)[0].cpu()),
        "ssim": float(per_sample_ssim(prediction, clean)[0].cpu()),
    }
    if lpips_metric is not None:
        result["lpips"] = float(lpips_metric(prediction.float(), clean.float())[0].cpu())
    return result


def run_rollout(
    *,
    model: torch.nn.Module,
    simulator: torch.nn.Module,
    observation: torch.Tensor,
    observation_severity: float,
    transitions: int,
    method: str,
    refinement_config: RefinementConfig,
    true_clean: torch.Tensor | None = None,
    true_realization: Realization | None = None,
    true_states: Mapping[float, torch.Tensor] | None = None,
    lpips_metric: Any | None = None,
    capture_tensors: bool = False,
) -> dict[str, Any]:
    allowed = {"one_shot", "bridge_no_refinement", "bridge_refinement", "oracle_xi", "teacher_forced"}
    if method not in allowed:
        raise ValueError(f"Unknown rollout method {method}")
    if method == "oracle_xi" and true_realization is None:
        raise ValueError("oracle_xi requires true_realization")
    if method == "teacher_forced" and true_states is None:
        raise ValueError("teacher_forced requires true_states")
    schedule = (
        (float(observation_severity),)
        if method == "one_shot"
        else evenly_spaced_schedule(observation_severity, transitions)
    )
    start = time.monotonic()
    current = observation
    previous_refined: Realization | None = None
    steps = []
    captured = []
    captured_realization_cache: dict[int, torch.Tensor] = {}

    def capture_realization_tensor(value: torch.Tensor) -> torch.Tensor:
        key = id(value)
        if key not in captured_realization_cache:
            captured_realization_cache[key] = value.detach().float().cpu()
        return captured_realization_cache[key]

    simulator_forwards = 0
    simulator_backwards = 0
    for step_index, (current_g, next_g) in enumerate(zip(schedule[:-1], schedule[1:], strict=True)):
        step_start = time.monotonic()
        forwards_before = simulator_forwards
        backwards_before = simulator_backwards
        if method == "teacher_forced":
            current = true_states[float(current_g)]
        with torch.no_grad(), torch.autocast(
            device_type=observation.device.type, dtype=torch.bfloat16
        ):
            prediction = model(
                current, torch.tensor([current_g], device=observation.device)
            )
        clean_prediction = prediction.clean.detach()
        network_realization = clone_realization(prediction.realization)
        initialization_source = "network"
        initialization_scores = None
        refinement = None
        if method == "bridge_refinement":
            if previous_refined is None:
                initialization = network_realization
            else:
                previous_score = initialization_objective(
                    simulator=simulator,
                    clean=clean_prediction,
                    candidate=previous_refined,
                    observation=observation,
                    observation_severity=observation_severity,
                    config=refinement_config,
                )
                network_score = initialization_objective(
                    simulator=simulator,
                    clean=clean_prediction,
                    candidate=network_realization,
                    observation=observation,
                    observation_severity=observation_severity,
                    config=refinement_config,
                )
                simulator_forwards += 2
                initialization_scores = {
                    "previous": previous_score,
                    "network": network_score,
                }
                if previous_score["objective"] <= network_score["objective"]:
                    initialization = previous_refined
                    initialization_source = "previous_refined"
                else:
                    initialization = network_realization
            refinement = refine_realization(
                simulator=simulator,
                clean=clean_prediction,
                initialization=initialization,
                observation=observation,
                observation_severity=observation_severity,
                maximum_iterations=(
                    refinement_config.first_iterations
                    if step_index == 0
                    else refinement_config.later_iterations
                ),
                config=refinement_config,
            )
            selected_realization = refinement.realization
            previous_refined = selected_realization
            simulator_forwards += refinement.simulator_forwards
            simulator_backwards += refinement.simulator_backwards
        elif method == "oracle_xi":
            selected_realization = true_realization
            initialization_source = "oracle"
        else:
            selected_realization = network_realization

        with torch.no_grad():
            reconstruction_before = simulator.render(
                clean_prediction, network_realization, observation_severity
            ).output
            simulator_forwards += 1
            if selected_realization is network_realization:
                reconstruction_after = reconstruction_before
            else:
                reconstruction_after = simulator.render(
                    clean_prediction, selected_realization, observation_severity
                ).output
                simulator_forwards += 1
            if method == "teacher_forced":
                next_state = true_states[float(next_g)]
            else:
                next_state = simulator.render(
                    clean_prediction, selected_realization, next_g
                ).output.detach()
                simulator_forwards += 1
        step: dict[str, Any] = {
            "step": step_index,
            "current_g": current_g,
            "next_g": next_g,
            "initialization_source": initialization_source,
            "initialization_scores": initialization_scores,
            "network_realization": realization_statistics(network_realization),
            "selected_realization": realization_statistics(selected_realization),
            "observation_residual_before": float(
                (reconstruction_before[..., 16:-16, 16:-16] - observation[..., 16:-16, 16:-16])
                .square()
                .mean()
                .cpu()
            ),
            "observation_residual_after": float(
                (reconstruction_after[..., 16:-16, 16:-16] - observation[..., 16:-16, 16:-16])
                .square()
                .mean()
                .cpu()
            ),
        }
        if refinement is not None:
            step["refinement"] = {
                "initial": refinement.initial,
                "best": refinement.best,
                "iterations": refinement.iterations,
                "update_norms": refinement.update_norms,
                "simulator_forwards": refinement.simulator_forwards,
                "simulator_backwards": refinement.simulator_backwards,
            }
        if true_clean is not None:
            step["clean_quality"] = quality(clean_prediction, true_clean, lpips_metric)
        if true_states is not None:
            step["next_state_mse"] = float(
                (next_state - true_states[float(next_g)]).square().mean().cpu()
            )
        step["simulator_forwards"] = simulator_forwards - forwards_before
        step["simulator_backwards"] = simulator_backwards - backwards_before
        step["wall_seconds"] = time.monotonic() - step_start
        if observation.device.type == "npu":
            step["peak_npu_memory_bytes"] = int(
                torch.npu.max_memory_allocated(observation.device)
            )
        steps.append(step)
        if capture_tensors:
            captured.append(
                {
                    "current": current.detach().half().cpu(),
                    "clean_prediction": clean_prediction.detach().half().cpu(),
                    "network_zernike": capture_realization_tensor(
                        network_realization.epsilon_zernike
                    ),
                    "network_tilt": capture_realization_tensor(
                        network_realization.epsilon_tilt
                    ),
                    "selected_zernike": capture_realization_tensor(
                        selected_realization.epsilon_zernike
                    ),
                    "selected_tilt": capture_realization_tensor(
                        selected_realization.epsilon_tilt
                    ),
                    "reconstruction_before": reconstruction_before.detach().half().cpu(),
                    "reconstruction_after": reconstruction_after.detach().half().cpu(),
                    "next_state": next_state.detach().half().cpu(),
                }
            )
        current = next_state

    final_g = float(schedule[-1])
    if method == "teacher_forced":
        current = true_states[final_g]
    with torch.no_grad(), torch.autocast(
        device_type=observation.device.type, dtype=torch.bfloat16
    ):
        final_prediction = model(
            current, torch.tensor([final_g], device=observation.device)
        ).clean.detach()
    result: dict[str, Any] = {
        "method": method,
        "schedule": list(schedule),
        "transitions": len(schedule) - 1,
        "steps": steps,
        "network_forwards": len(schedule),
        "simulator_forwards": simulator_forwards,
        "simulator_backwards": simulator_backwards,
        "wall_seconds": time.monotonic() - start,
        "final": final_prediction,
    }
    if true_clean is not None:
        result["final_quality"] = quality(final_prediction, true_clean, lpips_metric)
    if capture_tensors:
        result["captured"] = captured
        result["final_input"] = current.detach().half().cpu()
        result["final_prediction"] = final_prediction.detach().half().cpu()
    return result

"""Fixed FP32 native-xi Adam and samplewise whole-pair initialization.

The vector objective contains SINGLE-IMAGE means. Its SUM is differentiated so
batching cannot rescale a sample's gradient (and Adam epsilon behavior).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch
from physical_bridge.simulator import Realization

from .inference import RefinementNumericConfig, RefinementRequest

LEARNING_RATES = (1e-3, 1e-2, 1e-1)
ITERATION_BUDGETS = (10, 20, 40)


def synchronize(device):
    backend = getattr(torch, device.type, None)
    if device.type != "cpu" and backend is not None:
        backend.synchronize()


def clone_xi(value, *, grad=False):
    return Realization(*(x.detach().float().clone().requires_grad_(grad)
                         for x in (value.epsilon_zernike, value.epsilon_tilt)))


def xi_where(mask, first, second):
    """One boolean per sample selects BOTH entire native blocks together."""
    def select(a, b):
        return torch.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
    return Realization(select(first.epsilon_zernike, second.epsilon_zernike),
                       select(first.epsilon_tilt, second.epsilon_tilt))


def per_sample_objective(*, simulator, current_clean, realization, observation, initial_G):
    if initial_G not in (1, 2, 3, 4):
        raise ValueError("Original G must be 1, 2, 3 or 4")
    with torch.autocast(device_type=observation.device.type, enabled=False):
        output = simulator.render(current_clean.detach().float(), realization, initial_G).output
        if output.shape != observation.shape or output.ndim != 4 or output.shape[1] != 3:
            raise ValueError("Expected equal [B,3,H,W] original observation and render")
        return (output.float() - observation.detach().float()).square().flatten(1).mean(1)


def _scores(objective, value, batch):
    result = objective(value)
    if result.shape != (batch,):
        raise ValueError("Objective must return one single-image mean per sample")
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Objective must be finite FP32")
    return result


def select_batched_initialization(network_realization, previous_refined, objective):
    batch = network_realization.epsilon_zernike.shape[0]
    if previous_refined is None:
        return clone_xi(network_realization), ["network"] * batch, None, None
    with torch.no_grad():
        previous = _scores(objective, previous_refined, batch)
        network = _scores(objective, network_realization, batch)
        inherit = previous <= network
        selected = clone_xi(xi_where(inherit, previous_refined, network_realization))
    return (selected, ["previous_refined" if x else "network" for x in inherit.cpu().tolist()],
            previous.cpu().tolist(), network.cpu().tolist())


@dataclass
class RefinementResult:
    realization: Realization
    initial_objective: list[float]
    best_objective: list[float]
    trace: list[dict]
    prefixes: dict[str, dict]
    elapsed_seconds: float


class RefinementFailure(RuntimeError):
    def __init__(self, message, *, trace, prefixes, failed_update):
        super().__init__(message)
        self.trace = trace
        self.prefixes = prefixes
        self.failed_update = failed_update


def solve_adam(objective: Callable, initialization: Realization, *, learning_rate: float,
               iterations: int, checkpoint_steps=ITERATION_BUDGETS) -> RefinementResult:
    """A fresh Adam for every call; no clipping, projection, normalization or stop.

    Nonfinite/disconnected derivatives fail the run. A connected zero derivative
    is recorded distinctly and is not called an improvement. Best includes init.
    Prefixes before a failure remain available to the finite calibration ranking.
    """
    if learning_rate not in LEARNING_RATES or iterations not in ITERATION_BUDGETS:
        raise ValueError("Only the predeclared 3 learning rates and 10/20/40 budgets are allowed")
    current = clone_xi(initialization, grad=True)
    blocks = (current.epsilon_zernike, current.epsilon_tilt)
    batch = blocks[0].shape[0]
    if blocks[1].shape[0] != batch or batch < 1:
        raise ValueError("Native blocks must share a nonempty batch axis")
    optimizer = torch.optim.Adam(blocks, lr=learning_rate, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
    trace, prefixes = [], {}
    synchronize(blocks[0].device)
    started = time.perf_counter()
    step = 0
    try:
        with torch.enable_grad(), torch.autocast(device_type=blocks[0].device.type, enabled=False):
            values = _scores(objective, current, batch)
            initial = values.detach().clone()
            best_values, best = initial.clone(), clone_xi(current)
            for step in range(1, iterations + 1):
                optimizer.zero_grad(set_to_none=True)
                if not values.requires_grad:
                    raise RuntimeError("Objective is disconnected from both native xi blocks")
                values.sum().backward()  # NOT a batch mean.
                gradient_norms = []
                for name, block in zip(("zernike", "tilt"), blocks):
                    if block.grad is None:
                        raise RuntimeError(f"Objective is disconnected from native {name}")
                    if not bool(torch.isfinite(block.grad).all()):
                        raise FloatingPointError(f"Nonfinite native {name} gradient")
                    gradient_norms.append(block.grad.detach().flatten(1).norm(dim=1).cpu().tolist())
                before = [block.detach().clone() for block in blocks]
                before_values = values.detach().cpu().tolist()
                optimizer.step()
                update_norms = []
                for block, old in zip(blocks, before):
                    if not bool(torch.isfinite(block).all()):
                        raise FloatingPointError("Nonfinite xi update")
                    update_norms.append((block.detach() - old).flatten(1).norm(dim=1).cpu().tolist())
                del before, values
                # This graph is reused for the NEXT backward, with no extra forward.
                values = _scores(objective, current, batch)
                with torch.no_grad():
                    better = values.detach() < best_values
                    best = clone_xi(xi_where(better, current, best))
                    best_values = torch.minimum(best_values, values.detach())
                synchronize(blocks[0].device)
                elapsed = time.perf_counter() - started
                trace.append({"update": step, "objective_before": before_values,
                              "objective_after": values.detach().cpu().tolist(),
                              "best_objective": best_values.cpu().tolist(),
                              "gradient_l2_zernike": gradient_norms[0],
                              "gradient_l2_tilt": gradient_norms[1],
                              "update_l2_zernike": update_norms[0],
                              "update_l2_tilt": update_norms[1],
                              "gradient_connected": True, "finite": True,
                              "cumulative_seconds": elapsed})
                if step in checkpoint_steps:
                    prefixes[str(step)] = {"valid": True, "initial_objective": initial.cpu().tolist(),
                                           "best_objective": best_values.cpu().tolist(),
                                           "cumulative_seconds": elapsed}
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        raise RefinementFailure(str(exc), trace=trace, prefixes=prefixes, failed_update=step) from exc
    return RefinementResult(best, initial.cpu().tolist(), best_values.cpu().tolist(), trace,
                            prefixes, time.perf_counter() - started)


def numeric_config(learning_rate, iterations):
    if learning_rate not in LEARNING_RATES or iterations not in ITERATION_BUDGETS:
        raise ValueError("Configuration lies outside the frozen calibration grid")
    return RefinementNumericConfig(
        optimizer="Adam", optimizer_options={"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        learning_rate=learning_rate, first_iterations=iterations, later_iterations=iterations,
        objective_reduction="mean", stopping_policy={"kind": "fixed_budget"},
        return_policy={"kind": "best_including_initialization"},
        optimizer_state_policy={"kind": "reset_every_transition"},
        gradient_clip_policy={"kind": "none"}, projection_policy={"kind": "none"},
        gradient_normalization_policy={"kind": "none"}, nonfinite_policy={"kind": "fail"})


class NativeAdamRefiner:
    """Adapter for the existing single-sample run_inference selection API."""
    def __call__(self, request: RefinementRequest):
        config = request.numeric_config
        config.require_complete()
        if config != numeric_config(config.learning_rate, request.maximum_iterations):
            raise ValueError("Numeric policy differs from the authorized fixed native Adam")
        if request.initialization.epsilon_zernike.shape[0] != 1:
            raise ValueError("Scalar-objective adapter is restricted to one sample")
        return solve_adam(lambda xi: request.objective(xi).reshape(1), request.initialization,
                          learning_rate=config.learning_rate,
                          iterations=request.maximum_iterations).realization


def rank_candidates(probe_runs, *, expected_probe_count=24, low_signal_threshold=1e-6):
    """Rank all nine prefixes without dropping failed probes from any average."""
    if len(probe_runs) != expected_probe_count:
        raise ValueError("Calibration must include every frozen objective probe")
    candidates = []
    for lr in LEARNING_RATES:
        for steps in ITERATION_BUDGETS:
            improvements, seconds, errors = [], 0.0, []
            for probe in probe_runs:
                run = probe.get(str(lr), {})
                prefix = run.get("prefixes", {}).get(str(steps), {})
                if not prefix.get("valid") or run.get("network_unchanged") is not True:
                    errors.append(probe.get("probe_id", "missing_probe"))
                    continue
                j0, jb = prefix["initial_objective"], prefix["best_objective"]
                if len(j0) != 1 or len(jb) != 1 or not all(map(math.isfinite, (j0[0], jb[0]))):
                    errors.append(probe.get("probe_id", "invalid_values"))
                    continue
                improvements.append((j0[0] - jb[0]) / max(j0[0], 1e-12))
                seconds += prefix["cumulative_seconds"]
            valid = not errors and len(improvements) == expected_probe_count
            candidates.append({"learning_rate": lr, "iterations": steps, "valid": valid,
                               "failed_probe_ids": errors, "score": sum(improvements) / len(improvements)
                               if valid else None, "relative_improvements": improvements,
                               "total_cumulative_seconds": seconds if valid else None})
    remaining = [x for x in candidates if x["valid"]]
    if not remaining:
        raise RuntimeError("All frozen calibration candidates have invalid probes")
    best_score = max(x["score"] for x in remaining)
    eligible = [x for x in remaining if x["score"] >= 0.95 * best_score]
    ranking = []
    # Preserve the original 95% gate for the at-most-three integration attempts.
    # Each 3% timing tie is anchored to the fastest remaining retained candidate.
    while eligible:
        fastest = min(x["total_cumulative_seconds"] for x in eligible)
        tied = [x for x in eligible if x["total_cumulative_seconds"] <= 1.03 * fastest]
        chosen = min(tied, key=lambda x: (x["iterations"], x["learning_rate"]))
        ranking.append(chosen)
        eligible.remove(chosen)
    return {"candidates": candidates, "ranking": ranking, "best_score": best_score,
            "calibration_low_signal": best_score <= low_signal_threshold,
            "low_signal_threshold": low_signal_threshold,
            "scope": "Fixed observation-residual numerical calibration; no image-quality claim"}

"""Frozen protocol helpers for the new Known-g Physical Bridge V0.2."""

from __future__ import annotations

import math
from collections.abc import Iterable


G_GRID = (0.75, 1.50, 2.25, 3.00, 3.75, 4.50)
TERMINAL_GRID = (1.50, 2.25, 3.00, 3.75, 4.50)
REQUESTED_K = (0, 1, 3, 5)
LAMBDA_T_CANDIDATES = (0.2, 0.4, 0.8)
LAMBDA_B_CANDIDATES = (0.05, 0.1, 0.2)


def canonical_g(value: float) -> float:
    result = round(float(value), 2)
    if result not in G_GRID:
        raise ValueError(f"Severity {value} is outside the frozen grid")
    return result


def next_weaker(severity: float) -> float | None:
    severity = canonical_g(severity)
    index = G_GRID.index(severity)
    return None if index == 0 else G_GRID[index - 1]


def adjacent_schedule(start: float, requested_k: int) -> tuple[float, ...]:
    """SA-V02-ADJ-1: take at most K adjacent transitions toward 0.75."""

    start = canonical_g(start)
    requested_k = int(requested_k)
    if requested_k < 0:
        raise ValueError("requested_k must be non-negative")
    index = G_GRID.index(start)
    effective = min(requested_k, index)
    return tuple(G_GRID[index - step] for step in range(effective + 1))


def trajectory_severities(terminal_g: float, g: float, h: float | None) -> tuple[float, ...]:
    """Canonical ordered representation of unique {G, g, h}."""

    values: Iterable[float | None] = (terminal_g, g, h)
    return tuple(sorted({canonical_g(value) for value in values if value is not None}))


def cosine_learning_rate(update: int, maximum_updates: int, peak: float, warmup: int) -> float:
    """Frozen warmup plus cosine-to-zero schedule used by A1 and Stage B."""

    completed = int(update) + 1
    if completed <= int(warmup):
        return float(peak) * completed / int(warmup)
    progress = min(
        1.0,
        (completed - int(warmup)) / max(1, int(maximum_updates) - int(warmup)),
    )
    return 0.5 * float(peak) * (1.0 + math.cos(math.pi * progress))


def select_gradient_weights(
    *,
    g_clean: float,
    g_trajectory: float,
    g_bridge: float,
    h_clean: float,
    h_bridge: float,
) -> dict[str, object]:
    """Apply the frozen A0 health gate and deterministic selection rule."""

    denominators = (g_clean, h_clean)
    values = (g_clean, g_trajectory, g_bridge, h_clean, h_bridge)
    if any(not math.isfinite(value) for value in values) or any(value <= 0 for value in denominators):
        return {"status": "STOP", "reason": "non_finite_or_zero_denominator", "candidates": []}

    candidates: list[dict[str, float | bool]] = []
    for lambda_t in LAMBDA_T_CANDIDATES:
        for lambda_b in LAMBDA_B_CANDIDATES:
            r_t = lambda_t * g_trajectory / g_clean
            r_b = lambda_b * g_bridge / g_clean
            r_phys = r_t + r_b
            r_b_clean = lambda_b * h_bridge / h_clean
            healthy = (
                0.20 <= r_t <= 0.40
                and 0.05 <= r_b <= 0.20
                and 0.25 <= r_phys <= 0.50
                and r_b_clean <= 0.25
            )
            distance = abs(math.log2(lambda_t / 0.4)) + abs(math.log2(lambda_b / 0.1))
            candidates.append(
                {
                    "lambda_T": lambda_t,
                    "lambda_B": lambda_b,
                    "r_T": r_t,
                    "r_B": r_b,
                    "r_phys": r_phys,
                    "r_B_clean": r_b_clean,
                    "distance": distance,
                    "healthy": healthy,
                }
            )
    healthy_candidates = [item for item in candidates if bool(item["healthy"])]
    if not healthy_candidates:
        return {"status": "STOP", "reason": "no_healthy_candidate", "candidates": candidates}
    selected = min(
        healthy_candidates,
        key=lambda item: (
            float(item["distance"]),
            float(item["r_phys"]),
            float(item["lambda_B"]),
            float(item["lambda_T"]),
        ),
    )
    return {"status": "GO", "selected": selected, "candidates": candidates}

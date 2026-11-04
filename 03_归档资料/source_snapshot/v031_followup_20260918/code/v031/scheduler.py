"""Explicit update-index scheduler from the V0.31 protocol addendum.

``lr_for_update(u)`` is the LR USED by the 1-based optimizer update ``u``.
At a boundary with ``completed_updates == n``, optimizer groups already contain
the LR for update ``n + 1``. Call ``mark_update_completed()`` only AFTER a real
optimizer update. This class never updates model parameters itself.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Mapping


SCHEDULE_ID = "v031_linear50_cosine1000_constant1e-5_v1"
PEAK_LR = 1e-4
FLOOR_LR = 1e-5
WARMUP_UPDATES = 50
DECAY_END_UPDATE = 1000


def _index(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def lr_for_update(update_index: int) -> float:
    """LR at update index u; u=0 is a curve anchor, never an applied update."""
    u = _index(update_index, "update_index")
    if u <= WARMUP_UPDATES:
        return PEAK_LR * (u / WARMUP_UPDATES)
    if u >= DECAY_END_UPDATE:
        return FLOOR_LR
    fraction = (u - WARMUP_UPDATES) / (DECAY_END_UPDATE - WARMUP_UPDATES)
    return FLOOR_LR + 0.5 * (PEAK_LR - FLOOR_LR) * (1.0 + math.cos(math.pi * fraction))


class UpdateIndexedScheduler:
    """All parameter groups use the protocol LR; other optimizer fields stay intact.

    This is deliberately independent of torch's LRScheduler epoch conventions.
    State is a completed-update count plus fixed schedule identity. There is no
    restart, plateau handling, implicit step, or finite maximum update count.
    """

    def __init__(self, optimizer: Any, *, completed_updates: int = 0) -> None:
        if not optimizer.param_groups:
            raise ValueError("Optimizer must have at least one parameter group")
        self.optimizer = optimizer
        self.completed_updates = _index(completed_updates, "completed_updates")
        self._set_next_lr()

    @property
    def next_update_index(self) -> int:
        return self.completed_updates + 1

    @property
    def next_lr(self) -> float:
        return lr_for_update(self.next_update_index)

    @property
    def last_used_lr(self) -> float | None:
        return lr_for_update(self.completed_updates) if self.completed_updates else None

    def _set_next_lr(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.next_lr

    def mark_update_completed(self, *, update_index: int | None = None) -> None:
        """Record one already-completed update and prepare the following LR.

        Passing the actual 1-based update index catches duplicate/skipped calls.
        The caller must not use this to report a skipped optimizer update.
        """
        expected = self.next_update_index
        if update_index is not None and _index(update_index, "update_index") != expected:
            raise ValueError(f"Expected completed update {expected}, got {update_index}")
        self.assert_optimizer_lr()
        self.completed_updates = expected
        self._set_next_lr()

    def assert_optimizer_lr(self) -> None:
        for index, group in enumerate(self.optimizer.param_groups):
            value = group["lr"]
            if not isinstance(value, (int, float)) or float(value) != self.next_lr:
                raise ValueError(f"Optimizer group {index} LR does not equal next-update LR {self.next_lr}")

    def state_dict(self) -> dict[str, Any]:
        self.assert_optimizer_lr()
        return {
            "version": 1,
            "schedule_id": SCHEDULE_ID,
            "completed_updates": self.completed_updates,
            "next_update_index": self.next_update_index,
            "next_lr": self.next_lr,
            "last_used_lr": self.last_used_lr,
            "parameter_group_count": len(self.optimizer.param_groups),
            "index_semantics": "one_based_applied_update; optimizer_lr_is_next",
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> int:
        n = _index(state["completed_updates"], "completed_updates")
        expected = {
            "version": 1,
            "schedule_id": SCHEDULE_ID,
            "completed_updates": n,
            "next_update_index": n + 1,
            "next_lr": lr_for_update(n + 1),
            "last_used_lr": lr_for_update(n) if n else None,
            "parameter_group_count": len(self.optimizer.param_groups),
            "index_semantics": "one_based_applied_update; optimizer_lr_is_next",
        }
        if dict(state) != expected:
            raise ValueError("Scheduler state identity, counters, or LR fields are inconsistent")
        return n

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.completed_updates = self.validate_state_dict(state)
        self._set_next_lr()

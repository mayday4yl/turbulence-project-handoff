"""Atomic, identity-bound V0.31 update-boundary checkpoints.

No optimizer update or training loop lives here. Callers supply their EMA and
sampler objects through state_dict/load_state_dict. Only checkpoint at a completed
effective-batch boundary: in-flight accumulated gradients are not captured.
"""
from __future__ import annotations

from copy import deepcopy
import json
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import torch

from .scheduler import UpdateIndexedScheduler


CHECKPOINT_FORMAT = "v031_complete_update_boundary_v1"


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("A nonempty experiment/data/code identity mapping is required")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("Identity keys must be strings")
    # Canonical JSON also rejects tensors and arbitrary Python object identities.
    encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    if decoded != dict(value):
        raise ValueError("Identity must contain JSON-native values only")
    return decoded


def _cpu_snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return deepcopy(value)


def _npu_available() -> bool:
    backend = getattr(torch, "npu", None)
    return bool(backend is not None and backend.is_available())


def capture_rng_state(*, include_npu: bool | None = None) -> dict[str, Any]:
    """Capture Python, optional NumPy, torch CPU and optional visible-NPU RNGs.

    include_npu=None detects the already-imported torch.npu backend. This module
    does not import or initialize torch_npu on an otherwise CPU-only process.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "numpy": None,
        "torch_npu": None,
    }
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        name, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
        state["numpy"] = {
            "name": name, "keys": keys.tolist(), "position": int(position),
            "has_gaussian": int(has_gaussian), "cached_gaussian": float(cached_gaussian),
        }
    use_npu = _npu_available() if include_npu is None else include_npu
    if use_npu:
        if not _npu_available():
            raise RuntimeError("NPU RNG capture requested but no torch.npu backend is available")
        state["torch_npu"] = [item.cpu().clone() for item in torch.npu.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Strictly restore every RNG recorded; never silently drop NPU/NumPy state."""
    numpy_state = state["numpy"]
    np = None
    if numpy_state is not None:
        import numpy as np
    npu_states = state["torch_npu"]
    if npu_states is not None:
        if not _npu_available():
            raise RuntimeError("Checkpoint includes NPU RNG but torch.npu is unavailable")
        if len(npu_states) != torch.npu.device_count():
            raise ValueError("Visible NPU count differs from checkpoint RNG topology")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if numpy_state is not None:
        assert np is not None
        np.random.set_state((numpy_state["name"], np.array(numpy_state["keys"], dtype=np.uint32),
                             numpy_state["position"], numpy_state["has_gaussian"],
                             numpy_state["cached_gaussian"]))
    if npu_states is not None:
        torch.npu.set_rng_state_all([item.cpu() for item in npu_states])


def _validate_optimizer(optimizer: torch.optim.Optimizer,
                        scheduler: UpdateIndexedScheduler) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("The V0.31 addendum requires AdamW")
    if scheduler.optimizer is not optimizer:
        raise ValueError("Scheduler belongs to a different optimizer")


def _require_fp32_floating_tensors(value: Any, label: str) -> None:
    """Reject reduced/different precision without converting saved values.

    Integer/bool buffers and counters, and Python scalar metadata, are valid.
    In particular this does not freeze AdamW or EMA scalar hyperparameters.
    """
    if torch.is_tensor(value):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"{label} must remain FP32; found {value.dtype}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_fp32_floating_tensors(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_fp32_floating_tensors(item, f"{label}[{index}]")


def _validate_fp32_training_state(network_state: Mapping[str, Any],
                                  optimizer_state: Mapping[str, Any],
                                  ema_state: Any) -> None:
    _require_fp32_floating_tensors(network_state, "network")
    _require_fp32_floating_tensors(optimizer_state["state"], "optimizer.state")
    _require_fp32_floating_tensors(ema_state, "ema")


def save_checkpoint(path: str | Path, *, network: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: UpdateIndexedScheduler, ema: Any, sampler: Any,
                    completed_updates: int, identity: Mapping[str, Any],
                    include_npu_rng: bool | None = None,
                    overwrite: bool = False,
                    fixture_label: str | None = None) -> dict[str, Any]:
    """Atomically save a complete boundary. Does not freeze unspecified AdamW options.

    fixture_label is persisted so a synthetic state fixture cannot be mistaken
    for a training checkpoint. A normal load rejects fixtures by default.
    """
    _validate_optimizer(optimizer, scheduler)
    scheduler_state = scheduler.state_dict()
    if (isinstance(completed_updates, bool) or not isinstance(completed_updates, Integral)
            or completed_updates != scheduler.completed_updates):
        raise ValueError("Checkpoint and scheduler completed-update counts disagree")
    canonical_identity = _identity(identity)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "identity": canonical_identity,
        "completed_updates": scheduler.completed_updates,
        "network": _cpu_snapshot(network.state_dict()),
        "optimizer_class": "torch.optim.AdamW",
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "scheduler": scheduler_state,
        "ema": _cpu_snapshot(ema.state_dict()),
        "sampler": _cpu_snapshot(sampler.state_dict()),
        "fixture_label": fixture_label,
    }
    _validate_fp32_training_state(payload["network"], payload["optimizer"], payload["ema"])
    payload["rng"] = capture_rng_state(include_npu=include_npu_rng)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{destination.name}.",
                                         suffix=".tmp", dir=destination.parent, delete=False) as stream:
            temporary = stream.name
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic no-clobber publication, even if another writer races us.
            os.link(temporary, destination)
            os.unlink(temporary)
        temporary = None
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return {"path": str(destination), "completed_updates": scheduler.completed_updates,
            "last_used_lr": scheduler.last_used_lr, "next_lr": scheduler.next_lr,
            "fixture_label": fixture_label}


def load_checkpoint(path: str | Path, *, network: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: UpdateIndexedScheduler, ema: Any, sampler: Any,
                    expected_identity: Mapping[str, Any],
                    allow_fixture: bool = False) -> dict[str, Any]:
    """Restore all state, rejecting incompatible identity before any mutation.

    Loading into the same device topology is required for exact RNG replay.
    A failing component restoration rolls back the supplied objects and RNGs.
    """
    _validate_optimizer(optimizer, scheduler)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format")
    if payload.get("identity") != _identity(expected_identity):
        raise ValueError("Checkpoint experiment/data/code identity mismatch")
    if payload.get("fixture_label") is not None and not allow_fixture:
        raise ValueError("Synthetic fixture checkpoint is not authorized for training resume")
    if payload.get("optimizer_class") != "torch.optim.AdamW":
        raise ValueError("Checkpoint optimizer is not AdamW")
    # Validate the serialized dtype BEFORE load_state_dict can silently cast a
    # reduced-precision checkpoint into FP32 receiver tensors.
    _validate_fp32_training_state(payload["network"], payload["optimizer"], payload["ema"])
    _require_fp32_floating_tensors(network.state_dict(), "receiving network")
    completed = scheduler.validate_state_dict(payload["scheduler"])
    if payload.get("completed_updates") != completed:
        raise ValueError("Checkpoint completed-update counts are inconsistent")
    saved_groups = payload["optimizer"]["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer parameter-group count mismatch")
    if any(group["lr"] != payload["scheduler"]["next_lr"] for group in saved_groups):
        raise ValueError("Checkpoint optimizer LR disagrees with scheduler next LR")
    # Save rollback state before touching any mutable caller object.
    rollback = {
        "network": _cpu_snapshot(network.state_dict()),
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "scheduler": deepcopy(scheduler.state_dict()),
        "ema": _cpu_snapshot(ema.state_dict()),
        "sampler": _cpu_snapshot(sampler.state_dict()),
        "rng": capture_rng_state(),
    }
    try:
        network.load_state_dict(payload["network"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.load_state_dict(payload["ema"])
        sampler.load_state_dict(payload["sampler"])
        # Custom EMA/load adapters must not downcast an otherwise valid payload.
        # Keep this inside the transaction so a violation triggers rollback.
        _validate_fp32_training_state(network.state_dict(), optimizer.state_dict(), ema.state_dict())
        restore_rng_state(payload["rng"])
    except BaseException:
        network.load_state_dict(rollback["network"], strict=True)
        optimizer.load_state_dict(rollback["optimizer"])
        scheduler.load_state_dict(rollback["scheduler"])
        ema.load_state_dict(rollback["ema"])
        sampler.load_state_dict(rollback["sampler"])
        restore_rng_state(rollback["rng"])
        raise
    return {"completed_updates": completed,
            "last_used_lr": scheduler.last_used_lr,
            "next_update_index": scheduler.next_update_index,
            "next_lr": scheduler.next_lr,
            "identity": deepcopy(payload["identity"]),
            "fixture_label": payload["fixture_label"]}

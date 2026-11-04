"""Inspection helpers for the isolated DiffUIR-L gradient audit.

These helpers do not replace any model, diffusion equation, or gradient.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time
import types

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def runtime(device):
    device = torch.device(device)
    return getattr(torch, device.type, None) if device.type != "cpu" else None


def synchronize(device):
    backend = runtime(device)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def start_measurement(device):
    synchronize(device)
    backend = runtime(device)
    if backend is not None and hasattr(backend, "reset_peak_memory_stats"):
        backend.reset_peak_memory_stats()
    return time.perf_counter()


def end_measurement(device, started):
    synchronize(device)
    result = {"synchronized_seconds": time.perf_counter() - started}
    backend = runtime(device)
    for name in ("max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"):
        if backend is not None and hasattr(backend, name):
            result[name + "_bytes"] = int(getattr(backend, name)())
    result["memory_scope"] = "Framework allocator; excludes host snapshots and is not total process HBM"
    return result


def seed_rng(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    backend = runtime(device)
    if backend is not None and hasattr(backend, "manual_seed_all"):
        backend.manual_seed_all(seed)


def capture_rng(device):
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    backend = runtime(device)
    if backend is not None and hasattr(backend, "get_rng_state_all"):
        result["device"] = backend.get_rng_state_all()
    return result


def restore_rng(result, device):
    random.setstate(result["python"])
    np.random.set_state(result["numpy"])
    torch.set_rng_state(result["torch"])
    backend = runtime(device)
    if "device" in result:
        backend.set_rng_state_all(result["device"])


def tensor_summary(tensor):
    if tensor is None:
        return {"present": False, "finite": False, "nonzero": False}
    value = tensor.detach().float().cpu()
    finite = bool(torch.isfinite(value).all().item())
    result = {"present": True, "shape": list(value.shape), "dtype": str(tensor.dtype),
              "finite": finite, "nonzero": bool(torch.count_nonzero(value).item()),
              "nonzero_elements": int(torch.count_nonzero(value).item()), "elements": value.numel()}
    if finite:
        result.update({"l2": float(torch.linalg.vector_norm(value.double()).item()),
                       "max_abs": float(value.abs().max().item()),
                       "min": float(value.min().item()), "max": float(value.max().item())})
    return result


def parameter_gradients(model):
    rows = {name: tensor_summary(p.grad) for name, p in model.named_parameters() if p.requires_grad}
    missing = [n for n, r in rows.items() if not r["present"]]
    nonfinite = [n for n, r in rows.items() if r["present"] and not r["finite"]]
    zero = [n for n, r in rows.items() if r["present"] and r["finite"] and not r["nonzero"]]
    summary = {"trainable_tensor_count": len(rows), "missing_names": missing,
               "nonfinite_names": nonfinite, "zero_names": zero,
               "all_present_finite_nonzero": bool(rows) and not (missing or nonfinite or zero)}
    if not (missing or nonfinite):
        summary["total_l2"] = math.sqrt(sum(r["l2"] ** 2 for r in rows.values()))
    return {"summary": summary, "per_parameter": rows}


def parameter_changes(model, initial):
    rows = {}
    for name, p in model.named_parameters():
        new = p.detach().cpu()
        old = initial[name]
        delta = new.double() - old.double()
        finite = bool(torch.isfinite(new).all().item())
        rows[name] = {"finite": finite, "changed_elements": int(torch.count_nonzero(delta).item())}
        if finite:
            rows[name].update({"l2": float(torch.linalg.vector_norm(delta).item()),
                               "max_abs": float(delta.abs().max().item())})
    return {"changed_tensor_count": sum(r["changed_elements"] > 0 for r in rows.values()),
            "parameter_tensor_count": len(rows),
            "all_parameters_finite": all(r["finite"] for r in rows.values()),
            "per_parameter": rows}


@contextmanager
def sampling_gradients_enabled(diffusion):
    """Open precisely the two no_grad decorators on the actual three-step path."""
    if not diffusion.is_ddim_sampling or diffusion.sampling_timesteps != 3:
        raise ValueError("This audit is limited to the official three-step DDIM path")
    names = ("sample", "ddim_sample")
    old_instance_attributes = {}
    try:
        for name in names:
            method = getattr(diffusion, name)
            function = getattr(method, "__func__", method)
            if not hasattr(function, "__wrapped__"):
                raise RuntimeError(f"Expected official no_grad wrapper missing: {name}")
            old_instance_attributes[name] = (name in diffusion.__dict__, diffusion.__dict__.get(name))
            setattr(diffusion, name, types.MethodType(function.__wrapped__, diffusion))
        yield
    finally:
        for name, (existed, value) in old_instance_attributes.items():
            if existed:
                setattr(diffusion, name, value)
            else:
                delattr(diffusion, name)


@contextmanager
def observe_network(diffusion, retain=False):
    calls = []
    outputs = []
    def hook(module, args, output):
        tensors = list(output) if isinstance(output, (list, tuple)) else [output]
        row = {"call_index": len(calls) + 1, "input_shape": list(args[0].shape),
               "input_dtype": str(args[0].dtype), "output_shapes": [list(t.shape) for t in tensors],
               "output_requires_grad": [t.requires_grad for t in tensors],
               "time_conditions": [t.detach().cpu().tolist() for t in args[1]]}
        calls.append(row)
        outputs.extend(tensors)
        if retain:
            for tensor in tensors:
                if tensor.requires_grad:
                    tensor.retain_grad()
    handle = diffusion.model.register_forward_hook(hook)
    try:
        yield calls, outputs
    finally:
        handle.remove()


@contextmanager
def observe_training_indices(diffusion):
    observed = []
    old = diffusion.p_losses
    existed = "p_losses" in diffusion.__dict__
    old_attribute = diffusion.__dict__.get("p_losses")
    def wrapper(self, imgs, t, *args, **kwargs):
        observed.extend(t.detach().cpu().tolist())
        return old(imgs, t, *args, **kwargs)
    diffusion.p_losses = types.MethodType(wrapper, diffusion)
    try:
        yield observed
    finally:
        if existed:
            diffusion.p_losses = old_attribute
        else:
            delattr(diffusion, "p_losses")

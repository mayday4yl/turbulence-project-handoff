"""Numerical checks used by CPU tests and the real-initialization NPU preflight."""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
from types import MethodType

import numpy as np
import torch

from optimizations.psf_adjoint_20261021.adjoint import FrozenPSFConv, adjoint_kernel


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def gradient_record(tensors, gradients, labels):
    result = {}
    for tensor, grad, label in zip(tensors, gradients, labels):
        if grad is None or grad.shape != tensor.shape or not torch.isfinite(grad).all():
            raise RuntimeError(f"Missing/nonfinite gradient at {label}")
        norm = grad.detach().float().norm().item()
        maximum = grad.detach().float().abs().max().item()
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"No effective nonzero gradient at {label}")
        result[label] = {"l2": norm, "max_abs": maximum, "finite": True, "nonzero": True}
    return result


def check_result_gradients(result, stage):
    if not result.trace or len(result.trace) != (1 if stage == "A" else 4):
        raise RuntimeError("Incorrect retained stage trace")
    first = result.trace[0]
    tensors = [first["clean"], first["raw"].epsilon_zernike, first["raw"].epsilon_tilt]
    objective = result.observation_loss if stage == "A" else result.stage_losses[-1]
    gradients = torch.autograd.grad(objective, tensors, allow_unused=True, retain_graph=False)
    return {"source": "current_g_observation_loss" if stage == "A" else "last_stage_loss_to_first_prediction",
            "gradients": gradient_record(tensors, gradients, ["clean", "raw_zernike", "raw_tilt"]),
            "identity_or_surrogate_used": False}


def check_psf_adjoint(device="cpu", dtype=torch.float64):
    """Small independent frozen-33x33 check including reflection boundary VJP.

    The accepted original full real-shape parity evidence remains authoritative.
    This regression check detects an identity or zero-gradient substitution.
    """
    rng = torch.Generator().manual_seed(20261022)
    x = torch.randn(1, 1, 20, 21, generator=rng, dtype=dtype).to(device).requires_grad_(True)
    w = torch.randn(2, 1, 33, 33, generator=rng, dtype=dtype).to(device) / 33
    padded = torch.nn.functional.pad(x, (16, 16, 16, 16), mode="reflect")
    native = torch.nn.functional.conv2d(padded, w)
    candidate = FrozenPSFConv.apply(padded, w, adjoint_kernel(w))
    v = torch.randn(native.shape, generator=rng, dtype=dtype).to(device)
    gn, = torch.autograd.grad((native * v).sum(), x, retain_graph=True)
    gc, = torch.autograd.grad((candidate * v).sum(), x)
    # FP64 is a CPU reference/reduction, never an unsupported NPU cast.
    gn64, gc64 = gn.detach().cpu().double(), gc.detach().cpu().double()
    denominator = gn64.norm().clamp_min(1e-30)
    error = (gn64 - gc64).norm() / denominator
    threshold = 1e-10 if dtype == torch.float64 else 1e-4
    forward_max = (native.detach() - candidate.detach()).abs().max().item()
    if forward_max != 0 or not torch.isfinite(gc).all() or error.item() > threshold:
        raise RuntimeError(f"PSF exact-adjoint regression failed: forward={forward_max}, relL2={error.item()}")
    return {"status": "PASS", "device": str(device), "dtype": str(dtype),
            "forward_max_abs": forward_max, "gradient_relative_l2": error.item(),
            "gradient_relative_l2_threshold": threshold, "reflection_boundary_retained": True,
            "scope": "small independent regression; not a new physical Simulator certification"}


@contextmanager
def record_render_strengths(simulator):
    """Observe calls without changing any values/gradients; CPU copies are diagnostic-only."""
    records, originals = [], {}
    for name in ("render", "render_prepared"):
        if not hasattr(simulator, name):
            continue
        original = getattr(simulator, name)
        originals[name] = original
        def wrapper(self, *args, _name=name, _original=original, **kwargs):
            strength = kwargs.get("g", args[2] if len(args) >= 3 else None)
            if strength is not None:
                records.append({"method": _name, "strength": torch.as_tensor(strength).detach().cpu().clone()})
            return _original(*args, **kwargs)
        setattr(simulator, name, MethodType(wrapper, simulator))
    try:
        yield records
    finally:
        for name, original in originals.items():
            setattr(simulator, name, original)


def verify_render_strengths(records, stage, *, strength=None, paths=None):
    if stage == "A":
        # Inherited render can call render_prepared internally; both must see
        # the same current sample g, and there must be only one outer render.
        outer = [r for r in records if r["method"] == "render"]
        if len(outer) != 1 or not records:
            raise RuntimeError("A must execute exactly one current-g physical reconstruction")
        expected = strength.detach().cpu().reshape(-1)
        if any(not torch.equal(r["strength"].reshape(-1), expected) for r in records):
            raise RuntimeError("A reconstruction did not use its current sample strength")
        return {"outer_render_count": 1, "target": "Yg", "strength": expected.tolist()}
    prepared = [r for r in records if r["method"] == "render_prepared"]
    if len(prepared) != 7 or paths is None:
        raise RuntimeError("B must execute four G4 reconstructions and three predicted transitions")
    expected = []
    for i in range(4):
        expected.append(torch.tensor([4.], dtype=torch.float32))
        if i < 3:
            expected.append(paths.g[:, i + 1].detach().cpu())
    for actual, wanted in zip(prepared, expected):
        if not torch.equal(actual["strength"].reshape(-1), wanted.reshape(-1)):
            raise RuntimeError("B changed its G4 observation or predicted transition strengths")
    t = paths.t.detach().cpu()
    if not ((t[:, 0] == 0).all() and (t[:, -1] == 1).all() and (t[:, 1:] > t[:, :-1]).all()):
        raise RuntimeError("B path must be strict 0<t1<t2<t3<1")
    return {"physical_calls": 7, "observation_strength": 4, "target": "Y4",
            "path_times": t.tolist(), "no_gt_intermediate_states": True}


class ForwardHotspots:
    """One bounded diagnostic forward; synchronous inclusive times are not throughput.

    Counts/times cover model, attention, raw head, simulator preparation,
    FFT/effective-state, P2S mapping and render. Backward is measured separately
    by preflight. Nested times deliberately must not be summed.
    """
    def __init__(self, model, simulator, synchronize):
        self.model, self.simulator, self.synchronize = model, simulator, synchronize
        self.records, self.handles, self.methods, self.functions = {}, [], {}, []

    def _record(self, name, seconds):
        row = self.records.setdefault(name, {"calls": 0, "inclusive_seconds": 0.})
        row["calls"] += 1
        row["inclusive_seconds"] += seconds

    def _hooks(self, module, name):
        stack = []
        def before(_module, _args):
            self.synchronize()
            stack.append(time.perf_counter())
        def after(_module, _args, _output):
            self.synchronize()
            self._record(name, time.perf_counter() - stack.pop())
        self.handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    def __enter__(self):
        self._hooks(self.model, "restoration_network_forward")
        if hasattr(self.model, "raw_head"):
            self._hooks(self.model.raw_head, "raw_xi_head_forward")
        for module in self.model.modules():
            if module.__class__.__name__ in ("CSAttention", "LocalCSAttention"):
                self._hooks(module, "attention_forward")
        if hasattr(self.simulator, "mapping"):
            self._hooks(self.simulator.mapping, "p2s_mapping_forward")
        for name in ("prepare_image", "effective_state", "render_prepared", "sampling_grid"):
            if not hasattr(self.simulator, name):
                continue
            original = getattr(self.simulator, name)
            self.methods[name] = original
            def wrapped(_self, *args, _name=name, _original=original, **kwargs):
                self.synchronize()
                started = time.perf_counter()
                value = _original(*args, **kwargs)
                self.synchronize()
                self._record("simulator_" + _name, time.perf_counter() - started)
                return value
            setattr(self.simulator, name, MethodType(wrapped, self.simulator))
        for owner, name, label in ((torch.fft, "fft2", "fft2_forward"),
                                    (torch.fft, "ifft2", "ifft2_forward"),
                                    (torch.nn.functional, "grid_sample", "warp_grid_sample_forward")):
            original = getattr(owner, name)
            self.functions.append((owner, name, original))
            def measured(*args, _original=original, _label=label, **kwargs):
                self.synchronize()
                started = time.perf_counter()
                value = _original(*args, **kwargs)
                self.synchronize()
                self._record(_label, time.perf_counter() - started)
                return value
            setattr(owner, name, measured)
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        for name, original in self.methods.items():
            setattr(self.simulator, name, original)
        for owner, name, original in self.functions:
            setattr(owner, name, original)

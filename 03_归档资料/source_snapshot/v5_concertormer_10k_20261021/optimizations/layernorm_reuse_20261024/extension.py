"""Reuse the identical x-minus-mean temporary in official LayerNorm forward.

Only explicit installation changes execution. Parameters, buffers, dtype rules,
the pinned official backward function and frozen sources are unchanged. This
candidate has no runner/controller side effects and cannot start training.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MethodType

import torch
from models import official_layer_norm as official


EXPECTED_OFFICIAL_SHA256 = "db935a761114ae04954e0edaba47e9100bd00aa01e2f3779c0141c416ef70852"
_MISSING = object()


def verify_source_identity():
    source = Path(official.__file__).resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != EXPECTED_OFFICIAL_SHA256:
        raise RuntimeError("Pinned official LayerNorm source identity changed")
    return {"official_source": str(source), "official_sha256": actual,
            "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class CenteredReuseLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        # Retain the official dimension contract and failure for non-BCHW input.
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        centered = x - mu
        var = centered.pow(2).mean(1, keepdim=True)
        y = centered / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    # Reference the exact official function, rather than copying or rederiving it.
    backward = staticmethod(official.LayerNormFunction.backward)


def _reuse_forward(self, x):
    return CenteredReuseLayerNormFunction.apply(x, self.weight, self.bias, self.eps)


@dataclass
class LayerNormReuseHandle:
    entries: list
    identity: dict
    restored: bool = False

    @property
    def module_names(self):
        return [name for name, _, _ in self.entries]

    def restore(self):
        if self.restored:
            return
        for _, module, previous_instance_forward in reversed(self.entries):
            if previous_instance_forward is _MISSING:
                delattr(module, "forward")
            else:
                module.forward = previous_instance_forward
        self.restored = True


def install_layernorm_reuse(model):
    """Patch only exact official LayerNorm2d instances; returns reversible handle.

    Call after model construction/loading and before DDP if this candidate is
    selected in a future accepted runtime entrypoint. It is NOT auto-selected.
    No tensors are constructed here and no RNG or optimizer state is touched.
    """
    identity = verify_source_identity()
    targets = [(name, module) for name, module in model.named_modules()
               if type(module) is official.LayerNorm2d]
    if not targets:
        raise ValueError("No exact official LayerNorm2d modules found")
    if any(getattr(module.forward, "__func__", None) is _reuse_forward
           for _, module in targets):
        raise RuntimeError("LayerNorm reuse is already installed; refusing nested install")
    entries = []
    try:
        for name, module in targets:
            previous = module.__dict__.get("forward", _MISSING)
            module.forward = MethodType(_reuse_forward, module)
            entries.append((name, module, previous))
    except BaseException:
        LayerNormReuseHandle(entries, identity).restore()
        raise
    return LayerNormReuseHandle(entries, identity)


@contextmanager
def layernorm_reuse(model):
    """Scope the extension and restore original methods on every exit path."""
    handle = install_layernorm_reuse(model)
    try:
        yield handle
    finally:
        handle.restore()

"""A uses its current-g observation; B retains the accepted recursive science."""
from __future__ import annotations

import torch

from training.core import (TrainingResult, fp32_context, fp32_raw, network_context,
                           render_fp32, prepare_fp32)
from training.core import infer as _recursive_infer
from training.core import unroll_training as _recursive_training


def validate_current_state(observation, target, time, strength):
    if observation.shape != target.shape or observation.ndim != 4 or observation.shape[1] != 3:
        raise ValueError("A requires matching [B,3,H,W] RGB tensors")
    if observation.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("A canonical observations and targets must remain FP32")
    time = torch.as_tensor(time, device=observation.device, dtype=torch.float32)
    strength = torch.as_tensor(strength, device=observation.device, dtype=torch.float32)
    if time.shape != (len(observation),) or strength.shape != time.shape:
        raise ValueError("A requires one scalar time and current strength per sample")
    if not bool((torch.isfinite(time) & (time >= 0) & (time < 1)).all()):
        raise ValueError("A time is t=0 or strictly interior 0<t<1; t=1 is excluded")
    if not bool((torch.isfinite(strength) & (strength > 0) & (strength <= 4)).all()):
        raise ValueError("A strengths must lie in (0,4]")
    # Manifest storage casts t and g to FP32 independently; tolerate only the
    # rounding of g=4*(1-t), not a different physical time convention.
    if not torch.allclose(strength, 4. * (1. - time), rtol=1e-6, atol=4e-7):
        raise ValueError("A cached time/strength must satisfy g=4*(1-t)")
    if not bool(((time == 0) == (strength == 4)).all()):
        raise ValueError("A endpoints must pair exactly t=0 with original G=4")
    return time, strength


def unroll_training(model, simulator, observation, target, *, method, path_seed,
                    network_precision="fp32", retain_trace=False, objective=None,
                    time=None, strength=None):
    if method == "B" and objective == "recursive_alignment":
        if time is not None or strength is not None:
            raise ValueError("B samples its own strict four-step path from original Y4")
        return _recursive_training(model, simulator, observation, target, method="R0",
            path_seed=path_seed, network_precision=network_precision,
            retain_trace=retain_trace, objective="v5_dual")
    if method != "A" or objective != "precomputed_current_g":
        raise ValueError("Use A/precomputed_current_g or B/recursive_alignment")
    time, strength = validate_current_state(observation, target, time, strength)
    with network_context(observation.device, network_precision):
        clean, raw = model(observation, time)
    clean, raw = clean.float(), fp32_raw(raw)
    with fp32_context(observation.device):
        # Exactly one network prediction and one physical reconstruction. The
        # target is the current observation Y_g, never the original Y4.
        matched = render_fp32(simulator, clean, raw, strength)
        lc = (clean - target).abs().mean()
        lo = (matched - observation).abs().mean()
        loss = .9 * lc + .1 * lo
    trace = [{"state": observation, "clean": clean, "raw": raw,
              "matched": matched, "loss": loss, "time": time, "strength": strength}] if retain_trace else None
    return TrainingResult(loss, lc, lo, clean, None, [loss], trace, [lc], [lo])


def infer(model, simulator, observation, *, method, network_precision="fp32",
          N=None, gamma=1., time_grid=None, refinement=None, diagnostic_target=None,
          objective=None):
    if method not in ("A", "B"):
        raise ValueError("A/B unified validation only")
    if refinement is not None:
        raise ValueError("Refinement is OFF for all A/B model-selection validation")
    # Both checkpoints are selected by a real predicted-state rollout on the
    # untouched original Val1000, even though A training is single-step.
    return _recursive_infer(model, simulator, observation, method="R0",
        network_precision=network_precision, N=N, gamma=gamma, time_grid=time_grid,
        refinement=None, diagnostic_target=diagnostic_target, objective="v5_dual")

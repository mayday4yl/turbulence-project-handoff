"""Scientific computation shared by preflight, training and validation.

Only model forwards can use autocast. Simulator calls and loss reductions stay
FP32. Training recursion contains no detach and uses the predicted clean/raw xi.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import Any

import torch


def _load_schedules():
    # Import the separately accepted interface without mutating frozen sources.
    path = Path(__file__).resolve().parents[2] / "v5_v1_simulator_20261020" / "method_interface_20261021" / "schedules.py"
    name = "_v5_accepted_schedules"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load V5 schedules: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def fp32_context(device):
    device = torch.device(device)
    return torch.autocast(device_type=device.type, enabled=False)


def network_context(device, precision="fp32"):
    if precision == "fp32":
        return fp32_context(device)
    if precision != "bf16":
        raise ValueError("network precision must be fp32 or bf16")
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def raw_like(raw, zernike, tilt):
    return type(raw)(epsilon_zernike=zernike, epsilon_tilt=tilt)


def fp32_raw(raw):
    return raw_like(raw, raw.epsilon_zernike.float(), raw.epsilon_tilt.float())


def render_fp32(simulator, clean, raw, strength, *, prepared=None):
    with fp32_context(clean.device):
        clean, raw = clean.float(), fp32_raw(raw)
        if prepared is not None:
            result = simulator.render_prepared(prepared, raw, strength)
        else:
            result = simulator.render(clean, raw, strength)
        output = result.output if hasattr(result, "output") else result
        if output.dtype != torch.float32:
            raise RuntimeError("Simulator did not return FP32 output")
        return output


def prepare_fp32(simulator, clean):
    # Reuse the exact same differentiable image preparation for g=4 matching
    # and the next recursive state. This saves a duplicate convolution, without
    # changing the Simulator's operations, values or gradient semantics.
    if hasattr(simulator, "prepare_image") and hasattr(simulator, "render_prepared"):
        with fp32_context(clean.device):
            return simulator.prepare_image(clean.float())
    return None


@dataclass
class TrainingResult:
    loss: torch.Tensor
    clean_loss: torch.Tensor
    observation_loss: torch.Tensor
    final_clean: torch.Tensor
    paths: Any
    stage_losses: list[torch.Tensor]
    # For bounded preflight only. Production does not retain an extra list.
    trace: list[dict[str, Any]] | None = None
    stage_clean_losses: list[torch.Tensor] | None = None
    stage_observation_losses: list[torch.Tensor] | None = None


def unroll_training(model, simulator, observation, target, *, method: str,
                    path_seed: int, network_precision="fp32", retain_trace=False,
                    objective="v5_dual"):
    if objective == 'clean_only':
        if method != 'D0' or simulator is not None:
            raise ValueError('Clean-only D0 must not construct a Simulator')
        with network_context(observation.device, network_precision):
            clean = model(observation)
        with fp32_context(observation.device):
            loss = (clean.float() - target.float()).abs().mean()
        return TrainingResult(loss, loss, None, clean.float(), None, [loss],
                              stage_clean_losses=[loss], stage_observation_losses=None)
    if method not in ("D0", "R0", "R0-D0Init"):
        raise ValueError(f"Unknown V5 method: {method}")
    if observation.shape != target.shape or observation.ndim != 4 or observation.shape[1] != 3:
        raise ValueError("Training requires matching [B,3,H,W] RGB tensors")
    if observation.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("Training inputs must retain FP32 canonical values")
    schedules = _load_schedules()
    if method == "D0":
        paths = schedules.inference_paths(G=4., batch_size=observation.shape[0], N=1,
                                           device=observation.device)
    else:
        paths = schedules.sample_training_paths(observation.shape[0], seed=path_seed,
                                                 G=4., device=observation.device)
    state = observation
    clean_losses, observation_losses, stage_losses = [], [], []
    trace = [] if retain_trace else None
    for stage in range(paths.nfe):
        with network_context(state.device, network_precision):
            clean, raw = model(state, paths.t[:, stage])
        clean, raw = clean.float(), fp32_raw(raw)
        with fp32_context(state.device):
            prepared = prepare_fp32(simulator, clean)
            matched = render_fp32(simulator, clean, raw, 4., prepared=prepared)
            lc = (clean - target).abs().mean()
            lo = (matched - observation).abs().mean()
            loss = .9 * lc + .1 * lo
            clean_losses.append(lc)
            observation_losses.append(lo)
            stage_losses.append(loss)
            if trace is not None:
                trace.append({"state": state, "clean": clean, "raw": raw,
                              "matched": matched, "loss": loss})
            if stage + 1 < paths.nfe:
                # No detach or teacher-forced ground-truth state is allowed.
                state = render_fp32(simulator, clean, raw, paths.g[:, stage + 1],
                                     prepared=prepared)
    return TrainingResult(torch.stack(stage_losses).mean(),
                          torch.stack(clean_losses).mean(),
                          torch.stack(observation_losses).mean(), clean, paths,
                          stage_losses, trace, clean_losses, observation_losses)


@dataclass(frozen=True)
class RefinementConfig:
    steps: int
    learning_rate: float
    proximity_weight: float
    optimizer: str = "adam"
    proximity_reduction: str = "all_elements_mean"

    def validate(self):
        import math
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("refinement steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("refinement learning_rate must be finite and positive")
        if not math.isfinite(self.proximity_weight) or self.proximity_weight < 0:
            raise ValueError("refinement proximity_weight must be finite and nonnegative")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("refinement optimizer must be adam or sgd")
        if self.proximity_reduction != "all_elements_mean":
            raise ValueError("Only explicit total-raw-element mean proximity is implemented")


def refine_raw(simulator, clean, raw, observation, config: RefinementConfig):
    config.validate()
    # These detach operations are exclusive to inference refinement. Its clean
    # and reference raw are fixed, per the Method Spec. Training never calls it.
    clean = clean.detach().float()
    reference_z = raw.epsilon_zernike.detach().float()
    reference_t = raw.epsilon_tilt.detach().float()
    z = reference_z.clone().requires_grad_(True)
    t = reference_t.clone().requires_grad_(True)
    opt_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    opt = opt_class([z, t], lr=config.learning_rate)
    with torch.no_grad(), fp32_context(clean.device):
        prepared = prepare_fp32(simulator, clean)
    diagnostics = []
    with torch.enable_grad(), fp32_context(clean.device):
        for step in range(config.steps):
            opt.zero_grad(set_to_none=True)
            candidate = raw_like(raw, z, t)
            rendered = render_fp32(simulator, clean, candidate, 4., prepared=prepared)
            match = (rendered - observation).square().mean()
            prox = ((z - reference_z).square().sum() +
                    (t - reference_t).square().sum()) / (z.numel() + t.numel())
            objective = match + config.proximity_weight * prox
            if not torch.isfinite(objective).item():
                raise FloatingPointError("nonfinite refinement objective")
            objective.backward()
            if z.grad is None or t.grad is None or not (torch.isfinite(z.grad).all() & torch.isfinite(t.grad).all()).item():
                raise FloatingPointError("missing or nonfinite raw refinement gradient")
            opt.step()
            diagnostics.append({"step": step + 1, "objective": objective.item(),
                                "observation_mse": match.item(), "proximity": prox.item()})
    refined = raw_like(raw, z.detach(), t.detach())
    with torch.no_grad(), fp32_context(clean.device):
        if not (torch.isfinite(refined.epsilon_zernike).all() &
                torch.isfinite(refined.epsilon_tilt).all()).item():
            raise FloatingPointError("nonfinite final refinement raw xi")
        after = render_fp32(simulator, clean, refined, 4., prepared=prepared)
        if not torch.isfinite(after).all().item():
            raise FloatingPointError("nonfinite final refinement rendered observation")
        final_mse_tensor = (after - observation).square().mean()
        if not torch.isfinite(final_mse_tensor).item():
            raise FloatingPointError("nonfinite final refinement observation MSE")
        final_mse = final_mse_tensor.item()
    return refined, {"steps": diagnostics, "final_observation_mse": final_mse}


def infer(model, simulator, observation, *, method, network_precision="fp32",
          N=None, gamma=1., time_grid=None, refinement: RefinementConfig | None = None,
          diagnostic_target=None, objective="v5_dual"):
    if objective == 'clean_only':
        if method != 'D0' or simulator is not None or refinement is not None:
            raise ValueError('Clean-only inference has no Simulator or refinement')
        with torch.no_grad(), network_context(observation.device, network_precision):
            clean = model(observation).float()
        return clean, {'nfe': 1}
    if method not in ("D0", "R0", "R0-D0Init"):
        raise ValueError(f"Unknown V5 method: {method}")
    if method == "D0":
        if N not in (None, 1) or gamma != 1 or time_grid is not None:
            raise ValueError("D0 validation is a single network call at t=0")
        N = 1
    else:
        N = 4 if N is None and time_grid is None else N
    paths = _load_schedules().inference_paths(G=4., batch_size=observation.shape[0],
        N=N, gamma=gamma, time_grid=time_grid, device=observation.device)
    state = observation.float()
    refinement_records = []
    stage_metrics = []
    for stage in range(paths.nfe):
        with torch.no_grad(), network_context(state.device, network_precision):
            clean, raw = model(state, paths.t[:, stage])
        clean, raw = clean.float(), fp32_raw(raw)
        if refinement is not None:
            raw, record = refine_raw(simulator, clean, raw, observation, refinement)
            refinement_records.append({"stage": stage, **record})
        with torch.no_grad(), fp32_context(clean.device):
            prepared = prepare_fp32(simulator, clean) if diagnostic_target is not None else None
            if diagnostic_target is not None:
                matched = render_fp32(simulator, clean, raw, 4., prepared=prepared)
                clean_mae = (clean - diagnostic_target).abs().flatten(1).mean(1)
                reprojection_mae = (matched - observation).abs().flatten(1).mean(1)
                stage_metrics.append({'rgb_mse': (clean - diagnostic_target).square().flatten(1).mean(1),
                    'clean_loss': clean_mae, 'reprojection_loss': reprojection_mae,
                    'total_loss': .9 * clean_mae + .1 * reprojection_mae})
            if stage + 1 < paths.nfe:
                state = render_fp32(simulator, clean, raw, paths.g[:, stage + 1], prepared=prepared)
    # The accepted g=0 identity gives Z_N = clean; final refinement still runs
    # and is recorded, but cannot improve that returned clean via a zero warp.
    return clean, {"nfe": paths.nfe, "refinement": refinement_records, "stage_metrics": stage_metrics}

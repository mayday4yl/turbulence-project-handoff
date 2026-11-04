"""Additional fixed-direction FD scale diagnostics, never a Stage 1 PASS.

The original failed report/configuration is read-only. This script consumes the
same seed+71 projection and six direction draws as acceptance.py, skips testing
the two clean directions, and tests all four original xi directions at a fixed
smaller-scale ladder. Hooks observe pre-ReLU outputs and never replace them.
No training, numerical refinement, optimizer construction, or updates occur.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import torch

from v031.data import StartDataset
from v031.runtime import build, configure_npu, save_json, sha256, state_hash, verify_source


EPSILONS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
ATOL = 2e-5
RTOL = 0.05
ROOT = Path(__file__).resolve().parents[1]


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rate(mask: torch.Tensor) -> float:
    return int(mask.sum().item()) / mask.numel()


def rms_max(value: torch.Tensor) -> dict:
    value = value.double()
    return {"rms": float(value.square().mean().sqrt()), "max_abs": float(value.abs().max())}


class PreReluCapture:
    """Observe fc1, fc2 invocation 1, fc2 invocation 2 in unchanged _P2S."""

    def __init__(self, mapping):
        self.records = {}
        self.counts = {}
        self.handles = [mapping.fc1.register_forward_hook(self._hook("fc1")),
                        mapping.fc2.register_forward_hook(self._hook("fc2"))]

    def reset(self):
        self.records = {}
        self.counts = {"fc1": 0, "fc2": 0}

    def _hook(self, name):
        def observe(module, inputs, output):
            self.counts[name] += 1
            label = f"{name}_call{self.counts[name]}"
            value = output.detach()
            # Returning None leaves the real forward tensor and graph untouched.
            positive = (value > 0).cpu()
            self.records[label] = {
                "positive_mask": positive,
                "shape": list(value.shape),
                "positive_rate": rate(positive),
                "zero_count": int((value == 0).sum().cpu()),
                "minimum_absolute_preactivation": float(value.abs().min().cpu()),
                "finite": bool(torch.isfinite(value).all().cpu()),
            }
            return None
        return observe

    def take(self):
        if self.counts != {"fc1": 1, "fc2": 2}:
            raise RuntimeError(f"Unexpected original P2S invocation pattern: {self.counts}")
        result = self.records
        self.records = {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def relu_summary(capture):
    return {name: {key: value for key, value in row.items() if key != "positive_mask"}
            for name, row in capture.items()}


def relu_switches(base, plus, minus):
    return {
        name: {
            "plus_vs_base_rate": rate(plus[name]["positive_mask"] != row["positive_mask"]),
            "minus_vs_base_rate": rate(minus[name]["positive_mask"] != row["positive_mask"]),
            "plus_vs_minus_rate": rate(plus[name]["positive_mask"] != minus[name]["positive_mask"]),
            "plus_zero_count": plus[name]["zero_count"],
            "minus_zero_count": minus[name]["zero_count"],
            "plus_min_abs_preactivation": plus[name]["minimum_absolute_preactivation"],
            "minus_min_abs_preactivation": minus[name]["minimum_absolute_preactivation"],
        }
        for name, row in base.items()
    }


def pixel_grid(flow, height, width):
    # Original simulator uses align_corners=False. Its normalized-flow formula
    # uses size-1, so pixel_position+displacement is NOT the sampling coordinate.
    sizes = torch.tensor([width, height], dtype=torch.float64)
    unbounded = ((flow.double() + 1) * sizes - 1) / 2
    upper = sizes - 1
    clipped = torch.minimum(torch.maximum(unbounded, torch.zeros_like(upper)), upper)
    return {
        "unbounded": unbounded,
        "clipped": clipped,
        "cell": torch.floor(clipped).to(torch.int64),
        "low_border": unbounded <= 0,
        "high_border": unbounded >= upper,
        "exact_border": (unbounded == 0) | (unbounded == upper),
    }


def grid_change(base, changed):
    cell = base["cell"] != changed["cell"]
    border = ((base["low_border"] != changed["low_border"])
              | (base["high_border"] != changed["high_border"]))
    return {
        "sampling_coordinate_delta_pixels": rms_max(changed["unbounded"] - base["unbounded"]),
        "clipped_coordinate_delta_pixels": rms_max(changed["clipped"] - base["clipped"]),
        "cell_coordinate_change_rate": rate(cell),
        "cell_pixel_change_rate": rate(cell.any(dim=-1)),
        "border_coordinate_mask_change_rate": rate(border),
        "border_pixel_mask_change_rate": rate(border.any(dim=-1)),
        "exact_border_coordinate_rate": rate(changed["exact_border"]),
        "outside_or_border_coordinate_rate": rate(changed["low_border"] | changed["high_border"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New diagnostic directory under V031_ROOT")
    parser.add_argument("--reference-result", default="artifacts/derivatives_01/results.json")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Diagnostic output must be inside the independent V031_ROOT")
    output.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT / "configs/stage1_acceptance.json"
    cfg = json.loads(cfg_path.read_text())
    if (cfg["acceptance"]["finite_difference_atol"] != ATOL
            or cfg["acceptance"]["finite_difference_rtol"] != RTOL
            or cfg["acceptance"]["finite_difference_directions_per_branch"] != 2):
        raise ValueError("Original tolerance and two-direction protocol must remain unchanged")
    reference_path = (ROOT / args.reference_result).resolve(strict=True)
    reference = json.loads(reference_path.read_text())
    if reference.get("phase") != "derivatives":
        raise ValueError("Reference must be the original directional-derivative report")
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for source in [Path(__file__), cfg_path, *(ROOT / "code/v031").glob("*.py")]:
        destination = snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result = {
        "phase": "additional_fd_scale_diagnostic", "status": "running",
        "scope": "Diagnostic only; does not replace original FD report or declare Stage 1 PASS",
        "formal_training_started": False, "restoration_optimizer_updates": 0,
        "optimizer_created": False, "epsilon_ladder": list(EPSILONS), "atol": ATOL, "rtol": RTOL,
        "configuration": cfg, "config_sha256": sha256(cfg_path),
        "reference_result": {"path": str(reference_path), "sha256": sha256(reference_path),
                             "status": reference.get("status"), "passed": reference.get("passed")},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "code").rglob("*.py")},
        "execution_environment": {name: os.environ.get(name) for name in
                                  ("ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING")},
        "direction_generation_order": [], "directions": [],
    }

    def progress(event, **details):
        row = {"event": event, "time": time.time(), **details}
        result["last_event"] = row
        save_json(output / "results.json", result)
        print(json.dumps(row), flush=True)

    capture = None
    try:
        import torch_npu
        from physical_bridge.simulator import Realization

        torch.set_num_threads(4)
        device = torch.device("npu:0")
        result["npu_execution_declared"] = configure_npu(cfg)
        model, sim = build(cfg, device)
        initial_model = state_hash(model)
        initial_simulator = state_hash(sim)
        result.update({
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "initial_model_sha256": initial_model, "initial_simulator_sha256": initial_simulator,
            "base_hashes_before": verify_source(cfg),
            "npu_execution_effective": {
                "jit_compile": not bool(torch.npu.is_jit_compile_false()),
                "conv_hf32": torch.npu.conv.allow_hf32,
                "matmul_hf32": torch.npu.matmul.allow_hf32,
                "aclnn_hf32": torch.npu.aclnn.allow_hf32 if hasattr(torch.npu, "aclnn") else None,
                **{name: str(torch_npu._C._npu_getOption(name)) for name in
                   ("jitCompile", "ALLOW_INTERNAL_FORMAT", "ACL_PRECISION_MODE", "NPU_FUZZY_COMPILE_BLACKLIST")},
            },
        })
        if initial_model != reference["initial_model_sha256"] or initial_simulator != reference["initial_simulator_sha256"]:
            raise ValueError("Original model/simulator state identity was not reproduced")
        for name in cfg["base_hashes"]:
            source = Path(cfg["base_source"]) / name
            destination = snapshot / "original_base" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        dataset = StartDataset(cfg["dataset_root"], cfg["diagnostic_subset"])
        index = next(i for i, row in enumerate(dataset.records) if row["initial_G"] == 4)
        sample = dataset[index]
        if sample["start_id"] != reference["sample_start_ids"][0]:
            raise ValueError("Original diagnostic starting record was not reproduced")
        clean = sample["clean"].unsqueeze(0).to(device)
        gt = dataset.load_realization(index)
        base = {"clean": clean.detach().clone(),
                "epsilon_zernike": gt.epsilon_zernike.unsqueeze(0).to(device),
                "epsilon_tilt": gt.epsilon_tilt.unsqueeze(0).to(device)}
        result["sample_start_id"] = sample["start_id"]
        result["base_tensors"] = {name: {"shape": list(t.shape), "sha256": tensor_hash(t)}
                                  for name, t in base.items()}
        generator = torch.Generator().manual_seed(cfg["seed"] + 71)
        projection = torch.randn(clean.shape, generator=generator) / math.sqrt(clean.numel())
        projection_double = projection.double()
        result["projection"] = {"seed": cfg["seed"] + 71, "shape": list(projection.shape),
                                "dtype": str(projection.dtype), "sha256": tensor_hash(projection)}
        capture = PreReluCapture(sim.mapping)

        def evaluate(values):
            capture.reset()
            with torch.autocast(device_type=device.type, enabled=False):
                rendered = sim.render(values["clean"],
                                      Realization(values["epsilon_zernike"], values["epsilon_tilt"]), 4.)
            scalar = (rendered.output.to("cpu", dtype=torch.float64) * projection_double).sum()
            activations = capture.take()
            if not all(row["finite"] for row in activations.values()) or not bool(torch.isfinite(scalar)):
                raise FloatingPointError("Non-finite simulator or P2S scalar diagnostic")
            diagnostics = {
                "relu": activations,
                "grid": pixel_grid(rendered.flow.detach().cpu(), clean.shape[-2], clean.shape[-1]),
                "zernike": rendered.effective_state.zernike.detach().cpu(),
                "displacement": rendered.effective_state.displacement.detach().cpu(),
            }
            return scalar, diagnostics

        progress("initialized")
        # Dict insertion order intentionally matches the original acceptance.py.
        for branch in base:
            for direction_index in range(2):
                direction_cpu = torch.randn(base[branch].shape, generator=generator)
                direction_cpu = direction_cpu / direction_cpu.square().mean().sqrt()
                direction_identity = {"branch": branch, "direction_index": direction_index,
                                      "shape": list(direction_cpu.shape), "dtype": str(direction_cpu.dtype),
                                      "sha256": tensor_hash(direction_cpu),
                                      "rms": float(direction_cpu.double().square().mean().sqrt()),
                                      "l2": float(direction_cpu.double().norm()),
                                      "tested": branch != "clean"}
                result["direction_generation_order"].append(direction_identity)
                if branch == "clean":
                    continue
                progress("direction_start", branch=branch, direction_index=direction_index)
                direction = direction_cpu.to(device)
                values = {name: value.detach().clone().requires_grad_(name == branch)
                          for name, value in base.items()}
                f0, base_diagnostics = evaluate(values)
                grad = torch.autograd.grad(f0, values[branch])[0]
                if not bool(torch.isfinite(grad).all().cpu()):
                    raise FloatingPointError("Non-finite original simulator gradient")
                automatic = float((grad.cpu().double() * direction_cpu.double()).sum())
                baseline = float(f0.detach())
                original = next(row for row in reference["directional_derivatives"]
                                if row["branch"] == branch and row["direction_index"] == direction_index)
                row = {**direction_identity, "base_scalar": baseline, "autograd": automatic,
                       "gradient_finite": True, "gradient_norm": float(grad.norm().cpu()),
                       "original_autograd": original["checks"][0]["autograd"],
                       "autograd_abs_difference_from_original": abs(automatic - original["checks"][0]["autograd"]),
                       "base_pre_relu": relu_summary(base_diagnostics["relu"]),
                       "base_grid_exact_border_rate": rate(base_diagnostics["grid"]["exact_border"]),
                       "checks": []}
                result["directions"].append(row)
                del grad, f0
                for epsilon in EPSILONS:
                    progress("scale_start", branch=branch, direction_index=direction_index, epsilon=epsilon)
                    with torch.no_grad():
                        plus = {**values, branch: values[branch] + epsilon * direction}
                        minus = {**values, branch: values[branch] - epsilon * direction}
                        fplus, plus_diagnostics = evaluate(plus)
                        fminus, minus_diagnostics = evaluate(minus)
                    forward_value, backward_value = float(fplus), float(fminus)
                    central = (forward_value - backward_value) / (2 * epsilon)
                    error = abs(automatic - central)
                    tolerance = ATOL + RTOL * max(abs(automatic), abs(central))
                    # Actual representable native perturbations reveal FP32 resolution.
                    base_cpu = values[branch].detach().cpu().double()
                    actual_plus = plus[branch].detach().cpu().double() - base_cpu
                    actual_minus = minus[branch].detach().cpu().double() - base_cpu
                    check = {
                        "epsilon": epsilon, "base_scalar": baseline,
                        "plus_scalar": forward_value, "minus_scalar": backward_value,
                        "autograd": automatic, "central_difference": central,
                        "forward_one_sided_difference": (forward_value - baseline) / epsilon,
                        "backward_one_sided_difference": (baseline - backward_value) / epsilon,
                        "absolute_error": error,
                        "relative_error": error / max(abs(automatic), abs(central), 1e-12),
                        "tolerance": tolerance, "meets_original_tolerance": error <= tolerance,
                        "native_perturbation": {
                            "plus": rms_max(actual_plus), "minus": rms_max(actual_minus),
                            "plus_unchanged_fraction": rate(actual_plus == 0),
                            "minus_unchanged_fraction": rate(actual_minus == 0),
                            "plus_direction_reproduction_error": rms_max(actual_plus.double() / epsilon - direction_cpu.double()),
                            "minus_direction_reproduction_error": rms_max(actual_minus.double() / epsilon + direction_cpu.double()),
                        },
                        "effective_zernike_delta": {
                            "plus": rms_max(plus_diagnostics["zernike"] - base_diagnostics["zernike"]),
                            "minus": rms_max(minus_diagnostics["zernike"] - base_diagnostics["zernike"]),
                        },
                        "effective_displacement_delta": {
                            "plus": rms_max(plus_diagnostics["displacement"] - base_diagnostics["displacement"]),
                            "minus": rms_max(minus_diagnostics["displacement"] - base_diagnostics["displacement"]),
                        },
                        "sampling_grid": {
                            "coordinate_definition": "q=((flow+1)*[W,H]-1)/2; align_corners=False; border clips to [0,size-1]",
                            "mask_evidence_scope": "CPU float64 coordinate analysis of actual FP32 flow tensors; inferred interpolation cells, not instrumentation inside the CANN grid_sample kernel",
                            "plus_vs_base": grid_change(base_diagnostics["grid"], plus_diagnostics["grid"]),
                            "minus_vs_base": grid_change(base_diagnostics["grid"], minus_diagnostics["grid"]),
                            "plus_vs_minus": grid_change(minus_diagnostics["grid"], plus_diagnostics["grid"]),
                        },
                        "pre_relu_switches": relu_switches(base_diagnostics["relu"], plus_diagnostics["relu"], minus_diagnostics["relu"]),
                    }
                    if epsilon == 1e-3:
                        check["original_central_difference"] = original["checks"][0]["central_difference"]
                        check["central_abs_difference_from_original"] = abs(central - original["checks"][0]["central_difference"])
                    row["checks"].append(check)
                    progress("scale_complete", branch=branch, direction_index=direction_index, epsilon=epsilon)
                    del plus, minus, fplus, fminus, plus_diagnostics, minus_diagnostics
                    del base_cpu, actual_plus, actual_minus
                del values, direction, direction_cpu, base_diagnostics
                gc.collect()
                torch.npu.synchronize()
                torch.npu.empty_cache()
        capture.close()
        capture = None
        result["final_model_sha256"] = state_hash(model)
        result["final_simulator_sha256"] = state_hash(sim)
        result["model_and_simulator_unchanged"] = (
            initial_model == result["final_model_sha256"] and initial_simulator == result["final_simulator_sha256"])
        result["base_hashes_after"] = verify_source(cfg)
        if not result["model_and_simulator_unchanged"]:
            raise RuntimeError("Read-only diagnostic unexpectedly changed module state")
        result["status"] = "completed_diagnostic"
        result["stage1_acceptance_decision"] = "not_made_by_this_diagnostic"
        progress("complete")
    except BaseException as error:
        result.update({"status": "error", "exception": repr(error), "traceback": traceback.format_exc()})
        progress("error", error=repr(error))
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    main()

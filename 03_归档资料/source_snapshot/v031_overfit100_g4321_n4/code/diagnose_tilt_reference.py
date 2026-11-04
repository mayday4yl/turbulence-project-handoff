"""Same-point tilt diagnostics; no production replacement or Stage 1 decision.

Capture one original NPU render, then diagnose grid and literal real-input
irfft2 VJPs independently on CPU. Reproduce with --fixture without NPU/assets.
All reference constants are actual FP32 values promoted to FP64. In particular
the final size-two input axis remains real; it is NEVER reinterpreted as complex.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import torch
from torch.nn import functional as F

from v031.runtime import build, configure_npu, save_json, sha256, state_hash, verify_source

ROOT = Path(__file__).resolve().parents[1]
EPSILONS = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
COMMON_WINDOW = (3e-6, 1e-6)
REPLAY_EPSILONS = (1e-3, 3e-4, 1e-4)
ATOL, RTOL = 2e-5, 0.05
VJP_COSINE_MIN, VJP_RELATIVE_L2_MAX = 0.99999, 0.001


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cpu(value):
    return value.detach().cpu().contiguous()


def rms_max(value):
    value = value.double()
    return {"rms": float(value.square().mean().sqrt()), "max_abs": float(value.abs().max())}


def comparison(candidate, reference):
    a, b = candidate.detach().double().reshape(-1), reference.detach().double().reshape(-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError("Non-finite operator comparison")
    an, bn = float(a.norm()), float(b.norm())
    return {"candidate_norm": an, "reference_norm": bn,
            "relative_l2": float((a - b).norm()) / max(bn, 1e-30),
            "gradient_cosine": float(torch.dot(a, b)) / max(an * bn, 1e-30),
            "max_abs_error": float((a - b).abs().max())}


def vjp_comparison(candidate, reference):
    row = comparison(candidate, reference)
    nonzero = row["reference_norm"] > 0 or row["candidate_norm"] > 0
    row.update({"nonzero_vjp": nonzero, "all_values_finite": True,
                "meets_predeclared_operator_target": not nonzero or (
                    row["gradient_cosine"] >= VJP_COSINE_MIN
                    and row["relative_l2"] <= VJP_RELATIVE_L2_MAX)})
    return row


def scalar_error(a, b):
    error = abs(a - b)
    tolerance = ATOL + RTOL * max(abs(a), abs(b))
    return {"absolute_error": error, "relative_error": error / max(abs(a), abs(b), 1e-12),
            "tolerance": tolerance, "meets_original_tolerance": error <= tolerance}


def fd_row(base, plus, minus, epsilon, automatic):
    central = (plus - minus) / (2 * epsilon)
    return {"epsilon": epsilon, "base_scalar": base, "plus_scalar": plus,
            "minus_scalar": minus, "autograd": automatic, "central_difference": central,
            "forward_one_sided_difference": (plus - base) / epsilon,
            "backward_one_sided_difference": (base - minus) / epsilon,
            **scalar_error(automatic, central)}


def grid(value, flow):
    return F.grid_sample(value, flow, mode="bilinear", padding_mode="border", align_corners=False)


def grid_probe(blur, flow, projection):
    flow = flow.detach().clone().requires_grad_(True)
    output = grid(blur, flow)
    scalar = (output.double() * projection.double()).sum()
    grad, = torch.autograd.grad(scalar, flow)
    return float(scalar.detach()), cpu(grad), cpu(output)


def displacement(fixture, tilt):
    """Literal original mapping with frozen actual q32 and FP32 scalar constant."""
    q = fixture["q32"].to(dtype=tilt.dtype)
    constant = fixture["tilt_constant32"].to(dtype=tilt.dtype)
    size, rf_range = fixture["image_size"], fixture["rf_range"]
    return (torch.fft.irfft2(q * tilt, s=(rf_range, rf_range), dim=(1, 2))
            * constant)[:, :size, :size, :]


def coordinate_analysis(flow, grad32, grad64):
    """Inference from documented align_corners=False arithmetic, not kernel probes."""
    size = torch.tensor([flow.shape[2], flow.shape[1]], dtype=torch.float32)
    q32 = ((flow.float() + 1) * size - 1) / 2
    q64 = ((flow.double() + 1) * size.double() - 1) / 2
    upper = size.double() - 1
    clip32 = q32.double().clamp_min(0).minimum(upper)
    clip64 = q64.clamp_min(0).minimum(upper)
    cell = clip32.floor() != clip64.floor()
    border = ((q32.double() <= 0) != (q64 <= 0)) | ((q32.double() >= upper) != (q64 >= upper))
    affected = cell | border
    delta = grad32.double() - grad64.double()
    total = float(delta.square().sum())
    return {
        "scope": "CPU inferred q32/q64 arithmetic; CANN internal coordinates are not instrumented",
        "q_definition": "((flow+1)*[W,H]-1)/2; border clamp [0,size-1]",
        "q32_minus_q64_pixels": rms_max(q32.double() - q64),
        "cell_coordinate_change_rate": float(cell.float().mean()),
        "cell_pixel_change_rate": float(cell.any(-1).float().mean()),
        "border_coordinate_change_rate": float(border.float().mean()),
        "border_pixel_change_rate": float(border.any(-1).float().mean()),
        "q32_exact_integer_rate": float((q32.double() == q32.double().floor()).float().mean()),
        "q64_exact_integer_rate": float((q64 == q64.floor()).float().mean()),
        "q32_exact_border_rate": float(((q32 == 0) | (q32.double() == upper)).float().mean()),
        "q64_exact_border_rate": float(((q64 == 0) | (q64 == upper)).float().mean()),
        "grid_vjp_difference_energy_on_cell_or_border_changed_coordinates":
            float(delta[affected].square().sum()) / max(total, 1e-300),
        "grid_vjp_difference_energy_on_cell_or_border_changed_pixels":
            float(delta[affected.any(-1)].square().sum()) / max(total, 1e-300),
    }, q32


def run_cpu_reference(fixture, result, progress):
    blur, flow, projection = fixture["blur"], fixture["flow"], fixture["projection"]
    tilt = fixture["tilt"].double()
    npu = fixture["npu_grads"]
    factor = 2.0 / (fixture["image_size"] - 1)
    base_d = displacement(fixture, tilt).detach()
    base32, grad_grid32, output32 = grid_probe(blur, flow, projection)
    base64, grad_grid64, output64 = grid_probe(blur.double(), flow.double(), projection)
    coordinates, q32 = coordinate_analysis(flow, grad_grid32, grad_grid64)
    # Separate diagnostic at q32, never substituted into the same-flow reference.
    sizes64 = torch.tensor([flow.shape[2], flow.shape[1]], dtype=torch.float64)
    flow_q32 = (2 * q32.double() + 1) / sizes64 - 1
    base_q32, grad_q32, _ = grid_probe(blur.double(), flow_q32, projection)
    result["grid_operator"] = {
        "npu_base_scalar": fixture["npu_scalar"], "cpu32_base_scalar": base32,
        "cpu64_same_flow_base_scalar": base64, "cpu64_q32_coordinate_base_scalar": base_q32,
        "cpu64_vs_npu_vjp": vjp_comparison(grad_grid64, npu["flow"]),
        "cpu32_vs_npu_vjp": vjp_comparison(grad_grid32, npu["flow"]),
        "cpu32_vs_cpu64_vjp": vjp_comparison(grad_grid32, grad_grid64),
        "cpu64_q32_coordinates_vs_npu_vjp": vjp_comparison(grad_q32, npu["flow"]),
        "cpu64_q32_coordinates_vs_same_flow_vjp": vjp_comparison(grad_q32, grad_grid64),
        "cpu64_q32_coordinate_flow_difference": rms_max(flow_q32 - flow.double()),
        "cpu64_vs_cpu32_output": comparison(output64, output32),
        "coordinate_rounding": coordinates,
    }
    progress("cpu_grid_complete")

    # FFT VJP uses EXACTLY the NPU displacement cotangent in every comparison.
    fft64_tilt = tilt.clone().requires_grad_(True)
    fft64_d = displacement(fixture, fft64_tilt)
    fft64_grad, = torch.autograd.grad((fft64_d * npu["displacement"].double()).sum(), fft64_tilt)
    fft32_tilt = fixture["tilt"].clone().requires_grad_(True)
    fft32_d = displacement(fixture, fft32_tilt)
    fft32_grad, = torch.autograd.grad((fft32_d.double() * npu["displacement"].double()).sum(), fft32_tilt)
    result["linear_tilt_operator"] = {
        "cotangent": "identical original NPU d(loss)/d(displacement), promoted only for CPU64",
        "literal_fft": "irfft2(real q32_promoted * real tilt, s=(512,512), dim=(1,2)); last axis stays real",
        "cpu64_vs_npu_tilt_vjp": vjp_comparison(fft64_grad, npu["tilt"]),
        "cpu32_vs_npu_tilt_vjp": vjp_comparison(fft32_grad, npu["tilt"]),
        "cpu64_vs_cpu32_tilt_vjp": vjp_comparison(fft64_grad, fft32_grad),
        "cpu64_vs_actual_displacement": comparison(base_d, fixture["displacement"]),
        "cpu32_vs_actual_displacement": comparison(fft32_d.detach(), fixture["displacement"]),
        "npu_displacement_vjp_vs_flow_vjp_times_2_over_255":
            vjp_comparison(npu["displacement"], npu["flow"].double() * factor),
    }
    anchored_tilt = tilt.clone().requires_grad_(True)
    anchored_flow = flow.double() + factor * (displacement(fixture, anchored_tilt) - base_d)
    anchored_loss = (grid(blur.double(), anchored_flow) * projection.double()).sum()
    anchored_grad, = torch.autograd.grad(anchored_loss, anchored_tilt)
    result["anchored_reference"] = {
        "formula": "grid_sample(blur32.double, flow32.double + (2/255)*(D64(tilt)-D64(base_tilt)))",
        "actual_base_flow_difference": rms_max(anchored_flow.detach() - flow.double()),
        "actual_base_flow_exact_equal": bool(torch.equal(anchored_flow.detach(), flow.double())),
        "base_scalar": float(anchored_loss.detach()),
        "same_flow_grid_base_scalar_difference": abs(float(anchored_loss.detach()) - base64),
        "cpu64_vs_original_npu_tilt_vjp": vjp_comparison(anchored_grad, npu["tilt"]),
        "scope": "Higher precision conditional diagnostic anchored at actual FP32 flow; not production simulator",
    }
    if not result["anchored_reference"]["actual_base_flow_exact_equal"]:
        raise RuntimeError("Reference base flow is not the exact original point")

    def evaluate(value):
        with torch.no_grad():
            changed_flow = flow.double() + factor * (displacement(fixture, value) - base_d)
            return float((grid(blur.double(), changed_flow) * projection.double()).sum())

    result["directions"] = []
    for direction_index, direction32 in enumerate(fixture["directions"]):
        direction = direction32.double()
        original_ad = float((npu["tilt"].double() * direction).sum())
        reference_ad = float((anchored_grad * direction).sum())
        tangent = displacement(fixture, direction)
        direct_fft = float((tangent * npu["displacement"].double()).sum())
        adjoint_fft = float((fft64_grad * direction).sum())
        row = {"direction_index": direction_index, "sha256": tensor_hash(direction32),
               "original_npu_autograd": original_ad, "cpu64_anchored_autograd": reference_ad,
               "cpu64_vs_npu_autograd": scalar_error(reference_ad, original_ad),
               "fft_common_cotangent_direct_tangent_dot": direct_fft,
               "fft_common_cotangent_cpu64_vjp_dot": adjoint_fft,
               "fft_adjoint_identity": scalar_error(direct_fft, adjoint_fft),
               "fft_cpu64_vs_original_npu_directional": scalar_error(adjoint_fft, original_ad),
               "grid_cpu64_direct_tangent_dot": float((grad_grid64 * factor * tangent).sum()),
               "grid_cpu32_direct_tangent_dot": float((grad_grid32.double() * factor * tangent).sum()),
               "grid_npu_direct_tangent_dot": float((npu["flow"].double() * factor * tangent).sum()),
               "displacement_direction_tangent_pixels": rms_max(tangent), "checks": [],
               "original_fp32_perturbation_replay": []}
        result["directions"].append(row)
        for epsilon in EPSILONS:
            plus, minus = evaluate(tilt + epsilon * direction), evaluate(tilt - epsilon * direction)
            check = fd_row(base64, plus, minus, epsilon, reference_ad)
            check["versus_original_npu_autograd"] = scalar_error(original_ad, check["central_difference"])
            row["checks"].append(check)
        for replay in fixture["representable_perturbations"][direction_index]:
            epsilon = replay["epsilon"]
            actual_plus, actual_minus = replay["plus"].double(), replay["minus"].double()
            plus, minus = evaluate(actual_plus), evaluate(actual_minus)
            check = fd_row(base64, plus, minus, epsilon, reference_ad)
            original = replay["original_check"]
            exact_plus = evaluate(tilt + epsilon * direction)
            exact_minus = evaluate(tilt - epsilon * direction)
            exact_fd = (exact_plus - exact_minus) / (2 * epsilon)
            check.update({
                "original_fp32_check": original,
                "original_fp32_minus_reference_fd": original["central_difference"] - check["central_difference"],
                "exact_input_cpu64_fd": exact_fd,
                "representable_input_minus_exact_input_fd": check["central_difference"] - exact_fd,
                "original_vs_reference_plus_scalar_change":
                    (original["plus_scalar"] - original["base_scalar"]) - (plus - base64),
                "original_vs_reference_minus_scalar_change":
                    (original["minus_scalar"] - original["base_scalar"]) - (minus - base64),
                "plus_input_sha256": tensor_hash(replay["plus"]),
                "minus_input_sha256": tensor_hash(replay["minus"]),
                "plus_native_direction_error": rms_max((actual_plus - tilt) / epsilon - direction),
                "minus_native_direction_error": rms_max((actual_minus - tilt) / epsilon + direction),
                "versus_original_npu_autograd": scalar_error(original_ad, check["central_difference"]),
            })
            row["original_fp32_perturbation_replay"].append(check)
        progress("cpu_direction_complete", direction_index=direction_index)
    window = [check for row in result["directions"] for check in row["checks"]
              if check["epsilon"] in COMMON_WINDOW]
    result["preselected_common_window_diagnostic"] = {
        "epsilons": list(COMMON_WINDOW), "num_checks": len(window),
        "all_cpu64_ad_vs_fd_meet_original_tolerance": all(c["meets_original_tolerance"] for c in window),
        "all_npu_ad_vs_cpu64_fd_meet_original_tolerance":
            all(c["versus_original_npu_autograd"]["meets_original_tolerance"] for c in window),
        "both_cpu64_ad_vs_npu_ad_meet_original_tolerance":
            all(row["cpu64_vs_npu_autograd"]["meets_original_tolerance"] for row in result["directions"]),
        "scope": "Diagnostic comparisons only; no original acceptance result replaced",
    }


def capture_npu(cfg, reference, result, progress):
    import torch_npu
    from physical_bridge.simulator import Realization
    from v031.data import StartDataset

    device = torch.device("npu:0")
    result["npu_execution_declared"] = configure_npu(cfg)
    model, sim = build(cfg, device)
    model_hash, sim_hash = state_hash(model), state_hash(sim)
    result.update({"torch_npu": torch_npu.__version__, "initial_model_sha256": model_hash,
                   "initial_simulator_sha256": sim_hash, "base_hashes_before": verify_source(cfg),
                   "npu_execution_effective": {
                       "jit_compile": not bool(torch.npu.is_jit_compile_false()),
                       "conv_hf32": torch.npu.conv.allow_hf32,
                       "matmul_hf32": torch.npu.matmul.allow_hf32,
                       "aclnn_hf32": torch.npu.aclnn.allow_hf32 if hasattr(torch.npu, "aclnn") else None,
                       **{key: str(torch_npu._C._npu_getOption(key)) for key in
                          ("jitCompile", "ALLOW_INTERNAL_FORMAT", "ACL_PRECISION_MODE", "NPU_FUZZY_COMPILE_BLACKLIST")}}})
    if model_hash != reference["initial_model_sha256"] or sim_hash != reference["initial_simulator_sha256"]:
        raise ValueError("Original module state identity was not reproduced")
    dataset = StartDataset(cfg["dataset_root"], cfg["diagnostic_subset"])
    index = next(i for i, row in enumerate(dataset.records) if row["initial_G"] == 4)
    sample, gt = dataset[index], dataset.load_realization(index)
    if sample["start_id"] != reference["sample_start_id"]:
        raise ValueError("Original FD starting record was not reproduced")
    clean = sample["clean"].unsqueeze(0).to(device)
    zernike = gt.epsilon_zernike.unsqueeze(0).to(device)
    tilt = gt.epsilon_tilt.unsqueeze(0).to(device).detach().requires_grad_(True)
    base = {"clean": clean, "epsilon_zernike": zernike, "epsilon_tilt": tilt}
    result["sample_start_id"] = sample["start_id"]
    result["base_tensors"] = {name: {"sha256": tensor_hash(value), "shape": list(value.shape)}
                               for name, value in base.items()}
    for name in base:
        if result["base_tensors"][name]["sha256"] != reference["base_tensors"][name]["sha256"]:
            raise ValueError("Original FD input hash differs")
    generator = torch.Generator().manual_seed(cfg["seed"] + 71)
    projection = torch.randn(clean.shape, generator=generator) / math.sqrt(clean.numel())
    if tensor_hash(projection) != reference["projection"]["sha256"]:
        raise ValueError("Original projection hash differs")
    result["projection"] = {"sha256": tensor_hash(projection), "seed": cfg["seed"] + 71}
    directions = []
    result["direction_generation_order"] = []
    for branch, value in base.items():
        for direction_index in range(2):
            direction = torch.randn(value.shape, generator=generator)
            direction = direction / direction.square().mean().sqrt()
            identity = {"branch": branch, "direction_index": direction_index, "sha256": tensor_hash(direction)}
            old = next(r for r in reference["direction_generation_order"]
                       if r["branch"] == branch and r["direction_index"] == direction_index)
            if identity["sha256"] != old["sha256"]:
                raise ValueError("Original six-direction RNG sequence differs")
            result["direction_generation_order"].append(identity)
            if branch == "epsilon_tilt":
                directions.append(direction)
    progress("capture_original_forward")
    with torch.autocast(device_type="npu", enabled=False):
        rendered = sim.render(clean, Realization(zernike, tilt), 4.)
        scalar = (rendered.output.to("cpu", dtype=torch.float64) * projection.double()).sum()
    gradient_tilt, gradient_flow, gradient_displacement = torch.autograd.grad(
        scalar, (tilt, rendered.flow, rendered.effective_state.displacement))
    with torch.no_grad():
        unitpair = torch.stack((sim.tilt_s_half_unit.real, sim.tilt_s_half_unit.imag), dim=-1)
        amplitude = sim._strength_tensor(4., tilt.shape[0]).pow(5.0 / 6.0)
        q32 = unitpair[None] * amplitude[:, None, None, None]
        constant32 = torch.tensor(sim.tilt_constant, dtype=torch.float32, device=device)
        perturbations = []
        for direction_index, direction in enumerate(directions):
            old = next(r for r in reference["directions"]
                       if r["branch"] == "epsilon_tilt" and r["direction_index"] == direction_index)
            rows = []
            for epsilon in REPLAY_EPSILONS:
                check = next(r for r in old["checks"] if r["epsilon"] == epsilon)
                rows.append({"epsilon": epsilon, "plus": cpu(tilt + epsilon * direction.to(device)),
                             "minus": cpu(tilt - epsilon * direction.to(device)),
                             "original_check": {key: check[key] for key in
                                                ("epsilon", "base_scalar", "plus_scalar", "minus_scalar",
                                                 "autograd", "central_difference", "absolute_error",
                                                 "relative_error", "tolerance", "meets_original_tolerance")}})
            perturbations.append(rows)
    fixture = {"format": "v031_tilt_same_point_reference_v1", "configuration": cfg,
               "image_size": sim.image_size, "rf_range": sim.rf_range,
               "blur": cpu(rendered.blur_only), "flow": cpu(rendered.flow),
               "displacement": cpu(rendered.effective_state.displacement),
               "projection": projection, "unitpair": cpu(unitpair), "amplitude": cpu(amplitude),
               "q32": cpu(q32), "tilt_constant32": cpu(constant32),
               "tilt_constant_python": sim.tilt_constant, "tilt": cpu(tilt), "directions": directions,
               "npu_scalar": float(scalar.detach()),
               "npu_grads": {"tilt": cpu(gradient_tilt), "flow": cpu(gradient_flow),
                             "displacement": cpu(gradient_displacement)},
               "representable_perturbations": perturbations,
               "provenance": {key: result[key] for key in
                              ("sample_start_id", "base_tensors", "projection", "direction_generation_order",
                               "initial_model_sha256", "initial_simulator_sha256", "reference_result",
                               "base_hashes_before", "config_sha256", "npu_execution_effective")}}
    result["constants"] = {"tilt_constant_python": sim.tilt_constant,
                            "tilt_constant_fp32_promoted": float(cpu(constant32).double()),
                            "amplitude_fp32": cpu(amplitude).tolist(),
                            "q32_sha256": tensor_hash(q32), "unitpair_sha256": tensor_hash(unitpair),
                            "rule": "Freeze actual NPU unitpair32*amplitude32; promote q32, do not recompute product in FP64"}
    result["reference_base_scalar_differences"] = [
        {"direction_index": i, "new_minus_previous": fixture["npu_scalar"] - perturbations[i][0]["original_check"]["base_scalar"]}
        for i in range(2)]
    result["final_model_sha256"], result["final_simulator_sha256"] = state_hash(model), state_hash(sim)
    result["base_hashes_after"] = verify_source(cfg)
    result["model_and_simulator_unchanged"] = (result["final_model_sha256"] == model_hash
                                              and result["final_simulator_sha256"] == sim_hash)
    if not result["model_and_simulator_unchanged"]:
        raise RuntimeError("Original module state changed")
    torch.npu.synchronize()
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory under V031_ROOT; overwrite forbidden")
    parser.add_argument("--reference-result", default="artifacts/fd_scales_01/results.json")
    parser.add_argument("--fixture", help="CPU-only replay of a previously exported torch fixture")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Output must be inside independent V031_ROOT")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    result = {"phase": "tilt_same_point_reference_diagnostic", "status": "running",
              "formal_training_started": False, "restoration_optimizer_updates": 0, "optimizer_created": False,
              "stage1_acceptance_decision": "not_made_by_this_diagnostic", "torch": torch.__version__,
              "epsilon_ladder": list(EPSILONS), "preselected_common_window": list(COMMON_WINDOW),
              "replay_epsilons": list(REPLAY_EPSILONS), "atol": ATOL, "rtol": RTOL,
              "operator_cosine_min": VJP_COSINE_MIN, "operator_relative_l2_max": VJP_RELATIVE_L2_MAX,
              "scope": "Original failed reports remain unchanged; conditional CPU64 reference only",
              "execution_environment": {name: os.environ.get(name) for name in
                                        ("ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING")}}

    def progress(event, **details):
        result["last_event"] = {"event": event, "time": time.time(), **details}
        save_json(output / "results.json", result)
        print(json.dumps(result["last_event"]), flush=True)

    try:
        snapshot = output / "code_snapshot"
        snapshot.mkdir()
        local_sources = [Path(__file__), ROOT / "code/acceptance.py", ROOT / "code/diagnose_fd_scales.py",
                         *(ROOT / "code/v031").glob("*.py")]
        result["code_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in local_sources}
        for source in local_sources:
            destination = snapshot / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        protocol_path = ROOT / "docs/TILT_REFERENCE_DIAGNOSTIC_PROTOCOL.md"
        if protocol_path.is_file():
            result["diagnostic_protocol_sha256"] = sha256(protocol_path)
            shutil.copy2(protocol_path, snapshot / protocol_path.name)
        if args.fixture:
            fixture_path = (ROOT / args.fixture).resolve(strict=True)
            fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
            if fixture["format"] != "v031_tilt_same_point_reference_v1":
                raise ValueError("Unsupported fixture format")
            cfg = fixture["configuration"]
            result["fixture"] = {"path": str(fixture_path), "sha256": sha256(fixture_path)}
            result["fixture_provenance"] = fixture["provenance"]
            result["execution_mode"] = "cpu_fixture_replay_no_npu_import"
            save_json(snapshot / "configuration.json", cfg)
        else:
            cfg_path = ROOT / "configs/stage1_acceptance.json"
            cfg = json.loads(cfg_path.read_text())
            result["config_sha256"] = sha256(cfg_path)
            shutil.copy2(cfg_path, snapshot / cfg_path.name)
            reference_path = (ROOT / args.reference_result).resolve(strict=True)
            reference = json.loads(reference_path.read_text())
            if (reference["phase"] != "additional_fd_scale_diagnostic"
                    or reference["status"] != "completed_diagnostic"
                    or reference["config_sha256"] != result["config_sha256"]):
                raise ValueError("Completed original-scale diagnostic and identical configuration required")
            result["reference_result"] = {"path": str(reference_path), "sha256": sha256(reference_path)}
            shutil.copy2(reference_path, snapshot / "reference_fd_scales_results.json")
            for name in cfg["base_hashes"]:
                source = Path(cfg["base_source"]) / name
                destination = snapshot / "original_base" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            result["configuration"] = cfg
            if (cfg["acceptance"]["finite_difference_atol"] != ATOL
                    or cfg["acceptance"]["finite_difference_rtol"] != RTOL
                    or cfg["acceptance"]["finite_difference_directions_per_branch"] != 2):
                raise ValueError("Original tolerance/direction protocol changed")
            fixture = capture_npu(cfg, reference, result, progress)
            torch.npu.empty_cache()
            fixture_path = output / "tilt_reference_fixture.pt"
            torch.save(fixture, fixture_path)
            result["fixture"] = {"path": str(fixture_path), "sha256": sha256(fixture_path),
                                 "bytes": fixture_path.stat().st_size}
            result["execution_mode"] = "one_original_npu_forward_then_cpu_reference"
        result["configuration"] = cfg
        result["fixture_tensor_sha256"] = {
            name: tensor_hash(fixture[name]) for name in
            ("blur", "flow", "displacement", "projection", "unitpair", "amplitude", "q32", "tilt_constant32", "tilt")}
        result["fixture_tensor_sha256"].update({"npu_grad_" + name: tensor_hash(value)
                                                for name, value in fixture["npu_grads"].items()})
        progress("fixture_ready")
        run_cpu_reference(fixture, result, progress)
        result["status"] = "completed_diagnostic"
        progress("complete")
    except BaseException as error:
        result.update({"status": "error", "exception": repr(error), "traceback": traceback.format_exc()})
        progress("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()

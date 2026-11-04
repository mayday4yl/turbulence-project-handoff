#!/usr/bin/env python3
"""Independent, no-update differentiability audit for the frozen V0.2 simulator.

The audit deliberately does not trust method-document claims.  It checks the
executed PyTorch graph, input VJPs, local conditioning across severity, and
central finite differences on a frozen validation sample.  It never calls an
optimizer and never changes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


G_GRID = (0.75, 1.50, 2.25, 3.00, 3.75, 4.50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {
            "present": False,
            "finite_fraction": None,
            "nonzero_fraction": None,
            "l2_norm": None,
            "rms": None,
            "mean_abs": None,
            "max_abs": None,
        }
    detached = value.detach().float()
    finite = torch.isfinite(detached)
    absolute = detached.abs()
    return {
        "present": True,
        "shape": list(detached.shape),
        "finite_fraction": float(finite.float().mean()),
        "nonzero_fraction": float((absolute > 0).float().mean()),
        "l2_norm": float(torch.linalg.vector_norm(detached)),
        "rms": float(torch.sqrt(detached.square().mean())),
        "mean_abs": float(absolute.mean()),
        "max_abs": float(absolute.max()),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().flatten(1).mean(1)


def mapping_activation_stats(mapping: torch.nn.Module, zernike: torch.Tensor) -> dict[str, float]:
    value = zernike.permute(0, 2, 3, 1).reshape(-1, 36)
    first = torch.relu(mapping.fc1(value))
    second = torch.relu(mapping.fc2(first))
    third = torch.relu(mapping.fc2(second))
    return {
        "relu1_active_fraction": float((first > 0).float().mean()),
        "relu2_active_fraction": float((second > 0).float().mean()),
        "relu3_active_fraction": float((third > 0).float().mean()),
    }


def flow_stats(flow: torch.Tensor, displacement: torch.Tensor) -> dict[str, float]:
    detached = flow.detach().float()
    displacement = displacement.detach().float()
    any_out = (detached.abs() > 1.0).any(dim=-1)
    return {
        "coordinate_out_of_bounds_fraction": float((detached.abs() > 1.0).float().mean()),
        "pixel_out_of_bounds_fraction": float(any_out.float().mean()),
        "coordinate_near_border_fraction": float((detached.abs() > 0.98).float().mean()),
        "coordinate_abs_max": float(detached.abs().max()),
        "displacement_rms_pixels": float(torch.sqrt(displacement.square().mean())),
        "displacement_abs_max_pixels": float(displacement.abs().max()),
    }


def finite_difference_series(
    *,
    name: str,
    base: torch.Tensor,
    gradient: torch.Tensor,
    evaluate: Callable[[torch.Tensor], float],
    base_loss: float,
    relative_steps: tuple[float, ...] = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4),
) -> dict[str, Any]:
    grad = gradient.detach().float()
    grad_norm = float(torch.linalg.vector_norm(grad))
    if not math.isfinite(grad_norm) or grad_norm == 0.0:
        return {
            "variable": name,
            "status": "ZERO_OR_NONFINITE_AUTOGRAD",
            "autograd_directional_derivative": grad_norm,
            "rows": [],
        }
    direction = grad / grad_norm
    base_norm = float(torch.linalg.vector_norm(base.detach().float()))
    scale = max(base_norm, 1.0)
    rows = []
    for relative_step in relative_steps:
        absolute_step = relative_step * scale
        plus_loss = evaluate(base.detach() + absolute_step * direction)
        minus_loss = evaluate(base.detach() - absolute_step * direction)
        finite_difference = (plus_loss - minus_loss) / (2.0 * absolute_step)
        denominator = max(abs(finite_difference), abs(grad_norm), 1e-12)
        relative_error = abs(finite_difference - grad_norm) / denominator
        rows.append(
            {
                "relative_step": relative_step,
                "absolute_l2_step": absolute_step,
                "autograd_directional_derivative": grad_norm,
                "central_difference": finite_difference,
                "relative_error": relative_error,
                "sign_match": bool(finite_difference * grad_norm > 0),
                "plus_loss": plus_loss,
                "minus_loss": minus_loss,
                "plus_change": plus_loss - base_loss,
                "minus_change": minus_loss - base_loss,
            }
        )
    finite_rows = [row for row in rows if math.isfinite(row["relative_error"])]
    best = min(finite_rows, key=lambda row: row["relative_error"]) if finite_rows else None
    return {
        "variable": name,
        "status": "CHECKED" if best is not None else "NO_FINITE_RESULT",
        "base_l2_norm": base_norm,
        "autograd_directional_derivative": grad_norm,
        "best": best,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--val-subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-key", default="ema")
    parser.add_argument("--sample-key")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(20260901)
    started = time.monotonic()

    # Import the exact frozen implementation selected by --source-root.
    import sys

    sys.path.insert(0, str(args.source_root))
    from physical_bridge.data import BridgeArchive
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator, Realization

    simulator_document = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    simulator = PiRNP2SSimulator(
        simulator_document["asset_dir"],
        image_size=int(simulator_document["image_size"]),
        rf_range=int(simulator_document["rf_range"]),
        aperture_diameter=float(simulator_document["aperture_diameter"]),
        propagation_distance=float(simulator_document["propagation_distance"]),
        wavelength=float(simulator_document["wavelength"]),
        correlation=float(simulator_document["correlation"]),
        tilt_threshold=float(simulator_document["tilt_threshold"]),
        adjustment=float(simulator_document["adjustment"]),
        device="cpu",
    )

    v02_document = yaml.safe_load((args.source_root / "configs/v02.yaml").read_text())
    model = DualHeadUNet(**v02_document["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint
    model.eval().requires_grad_(False)

    subset = json.loads(args.val_subset.read_text(encoding="utf-8"))
    sample_key = args.sample_key or subset["4.50"][0]
    archive = BridgeArchive(args.dataset_root, "val")
    record = archive.by_key[sample_key]
    clean = archive.clean(record).unsqueeze(0)
    true_realization_cpu = archive.realization(record)
    true_realization = Realization(
        true_realization_cpu.epsilon_zernike.unsqueeze(0),
        true_realization_cpu.epsilon_tilt.unsqueeze(0),
    )

    audit: dict[str, Any] = {
        "status": "DIFFERENTIABILITY_AUDIT_COMPLETE",
        "scientific_gate_replacement": False,
        "parameter_updates": 0,
        "device": "cpu",
        "cpu_threads": args.threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sample_key": sample_key,
        "terminal_G": float(record["terminal_G"]),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "state_key": args.state_key,
        "simulator_source": str(args.source_root / "physical_bridge/simulator.py"),
        "simulator_source_sha256": sha256_file(args.source_root / "physical_bridge/simulator.py"),
        "simulator_config": str(args.simulator_config),
        "simulator_config_sha256": sha256_file(args.simulator_config),
        "all_simulator_parameters_frozen": all(
            not parameter.requires_grad for parameter in simulator.parameters()
        ),
        "simulator_parameter_count": sum(parameter.numel() for parameter in simulator.parameters()),
        "true_realization": {
            "epsilon_zernike": tensor_stats(true_realization.epsilon_zernike),
            "epsilon_tilt": tensor_stats(true_realization.epsilon_tilt),
        },
        "severity_records": [],
    }

    predictions: dict[float, Any] = {}
    for g in G_GRID:
        observation = archive.state(record, g).unsqueeze(0)
        with torch.no_grad():
            prediction = model(observation, torch.tensor([g], dtype=torch.float32))
        predictions[g] = prediction
        target = observation

        zernike = prediction.realization.epsilon_zernike.detach().requires_grad_(True)
        tilt = prediction.realization.epsilon_tilt.detach().requires_grad_(True)
        trajectory_result = simulator.render(clean, Realization(zernike, tilt), g)
        trajectory_loss = per_sample_mse(trajectory_result.output, target).mean()
        trajectory_grad_z, trajectory_grad_t = torch.autograd.grad(
            trajectory_loss, (zernike, tilt), retain_graph=False
        )
        with torch.no_grad():
            true_replay = simulator.render(clean, true_realization, g)

        severity_record: dict[str, Any] = {
            "g": g,
            "prediction": {
                "clean": tensor_stats(prediction.clean),
                "epsilon_zernike": tensor_stats(prediction.realization.epsilon_zernike),
                "epsilon_tilt": tensor_stats(prediction.realization.epsilon_tilt),
                "zernike_rms_over_true": float(
                    torch.sqrt(prediction.realization.epsilon_zernike.detach().square().mean())
                    / torch.sqrt(true_realization.epsilon_zernike.square().mean())
                ),
                "tilt_rms_over_true": float(
                    torch.sqrt(prediction.realization.epsilon_tilt.detach().square().mean())
                    / torch.sqrt(true_realization.epsilon_tilt.square().mean())
                ),
            },
            "trajectory_probe": {
                "loss": float(trajectory_loss.detach()),
                "prediction_clean_used": False,
                "grad_prediction_clean": tensor_stats(None),
                "grad_epsilon_zernike": tensor_stats(trajectory_grad_z),
                "grad_epsilon_tilt": tensor_stats(trajectory_grad_t),
                "grad_z_cosine_with_pred_minus_true": cosine(
                    trajectory_grad_z,
                    prediction.realization.epsilon_zernike.detach()
                    - true_realization.epsilon_zernike,
                ),
                "grad_tilt_cosine_with_pred_minus_true": cosine(
                    trajectory_grad_t,
                    prediction.realization.epsilon_tilt.detach() - true_realization.epsilon_tilt,
                ),
            },
            "graph": {
                "output_requires_grad": bool(trajectory_result.output.requires_grad),
                "output_grad_fn": type(trajectory_result.output.grad_fn).__name__,
                "blur_grad_fn": type(trajectory_result.blur_only.grad_fn).__name__,
                "flow_grad_fn": type(trajectory_result.flow.grad_fn).__name__,
                "weights_grad_fn": type(trajectory_result.weights.grad_fn).__name__,
                "zernike_state_grad_fn": type(
                    trajectory_result.effective_state.zernike.grad_fn
                ).__name__,
                "displacement_grad_fn": type(
                    trajectory_result.effective_state.displacement.grad_fn
                ).__name__,
            },
            "flow": flow_stats(
                trajectory_result.flow,
                trajectory_result.effective_state.displacement,
            ),
            "true_replay": {
                "mse_to_archived_state": float(per_sample_mse(true_replay.output, target).mean()),
                "predicted_effective_zernike_rms": float(
                    torch.sqrt(
                        trajectory_result.effective_state.zernike.detach().square().mean()
                    )
                ),
                "true_effective_zernike_rms": float(
                    torch.sqrt(true_replay.effective_state.zernike.square().mean())
                ),
                "predicted_over_true_effective_zernike_rms": float(
                    torch.sqrt(
                        trajectory_result.effective_state.zernike.detach().square().mean()
                    )
                    / torch.sqrt(true_replay.effective_state.zernike.square().mean())
                ),
                "predicted_over_true_displacement_rms": float(
                    torch.sqrt(
                        trajectory_result.effective_state.displacement.detach().square().mean()
                    )
                    / torch.sqrt(true_replay.effective_state.displacement.square().mean())
                ),
                "flow": flow_stats(
                    true_replay.flow,
                    true_replay.effective_state.displacement,
                ),
            },
            "mapping_activations": mapping_activation_stats(
                simulator.mapping, trajectory_result.effective_state.zernike.detach()
            ),
        }

        if g > G_GRID[0]:
            h = G_GRID[G_GRID.index(g) - 1]
            target_h = archive.state(record, h).unsqueeze(0)
            clean_prediction = prediction.clean.detach().requires_grad_(True)
            bridge_z = prediction.realization.epsilon_zernike.detach().requires_grad_(True)
            bridge_t = prediction.realization.epsilon_tilt.detach().requires_grad_(True)
            bridge_result = simulator.render(
                clean_prediction, Realization(bridge_z, bridge_t), h
            )
            bridge_loss = per_sample_mse(bridge_result.output, target_h).mean()
            grad_clean, grad_z, grad_t = torch.autograd.grad(
                bridge_loss, (clean_prediction, bridge_z, bridge_t), retain_graph=False
            )
            severity_record["bridge_probe"] = {
                "h": h,
                "loss": float(bridge_loss.detach()),
                "grad_prediction_clean": tensor_stats(grad_clean),
                "grad_epsilon_zernike": tensor_stats(grad_z),
                "grad_epsilon_tilt": tensor_stats(grad_t),
                "grad_clean_cosine_with_pred_minus_true": cosine(
                    grad_clean, prediction.clean.detach() - clean
                ),
                "grad_z_cosine_with_pred_minus_true": cosine(
                    grad_z,
                    prediction.realization.epsilon_zernike.detach()
                    - true_realization.epsilon_zernike,
                ),
                "grad_tilt_cosine_with_pred_minus_true": cosine(
                    grad_t,
                    prediction.realization.epsilon_tilt.detach() - true_realization.epsilon_tilt,
                ),
            }
        audit["severity_records"].append(severity_record)
        print(
            json.dumps(
                {
                    "g": g,
                    "trajectory_loss": severity_record["trajectory_probe"]["loss"],
                    "z_grad": severity_record["trajectory_probe"]["grad_epsilon_zernike"][
                        "l2_norm"
                    ],
                    "tilt_grad": severity_record["trajectory_probe"]["grad_epsilon_tilt"][
                        "l2_norm"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Central finite-difference check on the hardest frozen validation state.
    g = 4.50
    h = 3.75
    prediction = predictions[g]
    target_h = archive.state(record, h).unsqueeze(0)
    base_clean = prediction.clean.detach().clone().requires_grad_(True)
    base_z = prediction.realization.epsilon_zernike.detach().clone().requires_grad_(True)
    base_t = prediction.realization.epsilon_tilt.detach().clone().requires_grad_(True)
    base_result = simulator.render(base_clean, Realization(base_z, base_t), h)
    base_loss_tensor = per_sample_mse(base_result.output, target_h).mean()
    grad_clean, grad_z, grad_t = torch.autograd.grad(
        base_loss_tensor, (base_clean, base_z, base_t), retain_graph=False
    )
    base_loss = float(base_loss_tensor.detach())

    fixed_prepared = simulator.prepare_image(base_clean.detach())

    def image_loss(value: torch.Tensor) -> float:
        with torch.no_grad():
            rendered = simulator.render(
                value, Realization(base_z.detach(), base_t.detach()), h
            ).output
            return float(per_sample_mse(rendered, target_h).mean())

    def zernike_loss(value: torch.Tensor) -> float:
        with torch.no_grad():
            rendered = simulator.render_prepared(
                fixed_prepared, Realization(value, base_t.detach()), h
            ).output
            return float(per_sample_mse(rendered, target_h).mean())

    def tilt_loss(value: torch.Tensor) -> float:
        with torch.no_grad():
            rendered = simulator.render_prepared(
                fixed_prepared, Realization(base_z.detach(), value), h
            ).output
            return float(per_sample_mse(rendered, target_h).mean())

    finite_difference = {
        "g": g,
        "h": h,
        "base_loss": base_loss,
        "direction": "unit-L2 autograd direction",
        "precision": "float32 exact frozen implementation",
        "variables": [
            finite_difference_series(
                name="predicted_clean",
                base=base_clean,
                gradient=grad_clean,
                evaluate=image_loss,
                base_loss=base_loss,
            ),
            finite_difference_series(
                name="epsilon_zernike",
                base=base_z,
                gradient=grad_z,
                evaluate=zernike_loss,
                base_loss=base_loss,
            ),
            finite_difference_series(
                name="epsilon_tilt",
                base=base_t,
                gradient=grad_t,
                evaluate=tilt_loss,
                base_loss=base_loss,
            ),
        ],
    }
    finite_difference["all_best_sign_match"] = all(
        item.get("best", {}).get("sign_match", False)
        for item in finite_difference["variables"]
    )
    finite_difference["max_best_relative_error"] = max(
        item.get("best", {}).get("relative_error", math.inf)
        for item in finite_difference["variables"]
    )
    audit["finite_difference"] = finite_difference

    # The upstream code uses an (N-1) normalization together with
    # align_corners=False.  Quantify the resulting zero-tilt resampling rather
    # than treating it as a differentiability failure.
    with torch.no_grad():
        zero_tilt = torch.zeros_like(base_t)
        zero_tilt_result = simulator.render(
            base_clean.detach(), Realization(base_z.detach(), zero_tilt), h
        )
        size = simulator.image_size
        corrected_identity = (
            2.0 * (simulator.pixel_position + 0.5) / float(size) - 1.0
        )
        corrected_output = torch.nn.functional.grid_sample(
            zero_tilt_result.blur_only,
            corrected_identity,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        audit["coordinate_convention_probe"] = {
            "implemented_normalization": "2 * pixel / (N - 1) - 1",
            "grid_sample_align_corners": False,
            "mathematically_matching_identity_normalization": "2 * (pixel + 0.5) / N - 1",
            "implemented_zero_tilt_mse_to_blur": float(
                per_sample_mse(zero_tilt_result.output, zero_tilt_result.blur_only).mean()
            ),
            "matching_identity_mse_to_blur": float(
                per_sample_mse(corrected_output, zero_tilt_result.blur_only).mean()
            ),
            "note": "Forward-coordinate convention mismatch; not a broken autograd edge.",
        }

    audit["simulator_parameter_gradients_after_input_backward"] = {
        name: tensor_stats(parameter.grad) for name, parameter in simulator.named_parameters()
    }
    audit["all_simulator_parameter_gradients_none"] = all(
        parameter.grad is None for parameter in simulator.parameters()
    )
    audit["elapsed_seconds"] = time.monotonic() - started

    trajectory_z_norms = [
        row["trajectory_probe"]["grad_epsilon_zernike"]["l2_norm"]
        for row in audit["severity_records"]
    ]
    trajectory_t_norms = [
        row["trajectory_probe"]["grad_epsilon_tilt"]["l2_norm"]
        for row in audit["severity_records"]
    ]
    audit["summary"] = {
        "all_graph_outputs_require_grad": all(
            row["graph"]["output_requires_grad"] for row in audit["severity_records"]
        ),
        "all_trajectory_input_gradients_finite": all(
            row["trajectory_probe"][name]["finite_fraction"] == 1.0
            for row in audit["severity_records"]
            for name in ("grad_epsilon_zernike", "grad_epsilon_tilt")
        ),
        "all_bridge_input_gradients_finite_nonzero": all(
            row["bridge_probe"][name]["finite_fraction"] == 1.0
            and row["bridge_probe"][name]["l2_norm"] > 0.0
            for row in audit["severity_records"]
            if "bridge_probe" in row
            for name in (
                "grad_prediction_clean",
                "grad_epsilon_zernike",
                "grad_epsilon_tilt",
            )
        ),
        "trajectory_z_grad_norm_min": min(trajectory_z_norms),
        "trajectory_z_grad_norm_median": statistics.median(trajectory_z_norms),
        "trajectory_z_grad_norm_max": max(trajectory_z_norms),
        "trajectory_tilt_grad_norm_min": min(trajectory_t_norms),
        "trajectory_tilt_grad_norm_median": statistics.median(trajectory_t_norms),
        "trajectory_tilt_grad_norm_max": max(trajectory_t_norms),
        "finite_difference_all_best_sign_match": finite_difference[
            "all_best_sign_match"
        ],
        "finite_difference_max_best_relative_error": finite_difference[
            "max_best_relative_error"
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    archive.store.close()
    print(json.dumps(audit["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

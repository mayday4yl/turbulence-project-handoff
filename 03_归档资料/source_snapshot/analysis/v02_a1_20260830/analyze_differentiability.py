#!/usr/bin/env python3
"""Derive auditable tables for the V0.2 simulator differentiability audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DERIVED = ROOT / "derived"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "zero_count": sum(value == 0 for value in values),
        "nonfinite_count": sum(not math.isfinite(value) for value in values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    audit = read_json(DATA / "differentiability/differentiability_audit_cpu.json")
    cohort = read_json(DATA / "differentiability/differentiability_cohort_g450_cpu.json")
    a0 = read_jsonl(DATA / "a0/gradient_batches.jsonl")
    a1 = read_jsonl(DATA / "a1/train_metrics.jsonl")
    source = ROOT / "source/simulator.py"
    source_text = source.read_text(encoding="utf-8")
    upstream = ROOT / "source/upstream_simulator_new.py"

    assert len(a0) == 128
    assert len(a1) == 3000
    assert "self.requires_grad_(False)" in source_text
    assert "output = F.grid_sample(" in source_text
    assert "align_corners=False" in source_text
    assert "def render(" in source_text
    assert "@torch.no_grad()\n    def render" not in source_text

    a0_rows: list[dict[str, Any]] = []
    for loss in ("clean", "trajectory", "bridge"):
        for branch in ("shared", "clean", "realization", "total"):
            values = [
                float(row["losses"][loss]["gradient_norms"][branch]) for row in a0
            ]
            a0_rows.append({"loss": loss, "branch": branch, **stats(values)})

    a1_rows: list[dict[str, Any]] = []
    for branch in ("shared", "clean", "realization", "preclip"):
        values = [float(row["gradient_norms"][branch]) for row in a1]
        a1_rows.append({"branch": branch, **stats(values)})

    per_g_rows = []
    for row in audit["severity_records"]:
        trajectory = row["trajectory_probe"]
        bridge = row.get("bridge_probe")
        true_replay = row["true_replay"]
        per_g_rows.append(
            {
                "g": row["g"],
                "trajectory_loss": trajectory["loss"],
                "trajectory_z_grad_l2": trajectory["grad_epsilon_zernike"]["l2_norm"],
                "trajectory_tilt_grad_l2": trajectory["grad_epsilon_tilt"]["l2_norm"],
                "bridge_clean_grad_l2": None
                if bridge is None
                else bridge["grad_prediction_clean"]["l2_norm"],
                "bridge_z_grad_l2": None
                if bridge is None
                else bridge["grad_epsilon_zernike"]["l2_norm"],
                "bridge_tilt_grad_l2": None
                if bridge is None
                else bridge["grad_epsilon_tilt"]["l2_norm"],
                "effective_zernike_rms_ratio": true_replay[
                    "predicted_over_true_effective_zernike_rms"
                ],
                "displacement_rms_ratio": true_replay[
                    "predicted_over_true_displacement_rms"
                ],
                "relu3_active_fraction": row["mapping_activations"][
                    "relu3_active_fraction"
                ],
                "predicted_flow_oob_pixel_fraction": row["flow"][
                    "pixel_out_of_bounds_fraction"
                ],
                "true_flow_oob_pixel_fraction": true_replay["flow"][
                    "pixel_out_of_bounds_fraction"
                ],
                "true_replay_mse": true_replay["mse_to_archived_state"],
            }
        )

    finite_difference_rows = []
    for variable in audit["finite_difference"]["variables"]:
        best = variable["best"]
        finite_difference_rows.append(
            {
                "variable": variable["variable"],
                "autograd_directional_derivative": variable[
                    "autograd_directional_derivative"
                ],
                "best_relative_step": best["relative_step"],
                "central_difference": best["central_difference"],
                "best_relative_error": best["relative_error"],
                "sign_match": best["sign_match"],
                "minus_change": best["minus_change"],
                "plus_change": best["plus_change"],
            }
        )

    cohort_rows = [dict(row) for row in cohort["records"]]

    a0_log = (DATA / "differentiability/a0_npu.log").read_text(errors="replace")
    a1_log = (DATA / "differentiability/a1_npu_training.log").read_text(
        errors="replace"
    )
    warning_text = "An output with one or more elements was resized"
    summary = {
        "status": "COMPLETE_WITH_NPU_FINITE_DIFFERENCE_PENDING",
        "question": "Is V0.2 failure caused by a non-differentiable simulator?",
        "answer": (
            "No evidence of a broken or detached first-order graph. CPU central finite "
            "differences agree with autograd, and real NPU A0/A1 gradients have the "
            "expected branch fingerprint. A gradient-conditioning issue in the P2S "
            "Zernike path remains a plausible secondary contributor, not the primary "
            "failure locus established by the rollout controls."
        ),
        "binary_non_differentiability_supported": False,
        "cpu_gradient_accuracy": {
            "device": audit["device"],
            "sample_key": audit["sample_key"],
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "all_sign_match": audit["finite_difference"]["all_best_sign_match"],
            "max_best_relative_error": audit["finite_difference"][
                "max_best_relative_error"
            ],
            "all_graph_outputs_require_grad": audit["summary"][
                "all_graph_outputs_require_grad"
            ],
            "all_bridge_input_gradients_finite_nonzero": audit["summary"][
                "all_bridge_input_gradients_finite_nonzero"
            ],
        },
        "real_npu_gradient_presence": {
            "a0_batches": len(a0),
            "a1_updates": len(a1),
            "a0_expected_branch_fingerprint": {
                "clean_loss_realization_zero_all_batches": next(
                    row for row in a0_rows if row["loss"] == "clean" and row["branch"] == "realization"
                )["zero_count"]
                == len(a0),
                "trajectory_loss_clean_zero_all_batches": next(
                    row for row in a0_rows if row["loss"] == "trajectory" and row["branch"] == "clean"
                )["zero_count"]
                == len(a0),
                "bridge_all_three_branches_nonzero_all_batches": all(
                    next(
                        row
                        for row in a0_rows
                        if row["loss"] == "bridge" and row["branch"] == branch
                    )["zero_count"]
                    == 0
                    for branch in ("shared", "clean", "realization")
                ),
            },
            "a1_all_three_branch_norms_finite_nonzero": all(
                row["zero_count"] == 0 and row["nonfinite_count"] == 0
                for row in a1_rows
                if row["branch"] in ("shared", "clean", "realization")
            ),
            "a0_backward_resize_warning_processes": a0_log.count(warning_text),
            "a1_backward_resize_warning_processes": a1_log.count(warning_text),
            "warning_interpretation": (
                "One warning per rank reports a [16,17,32] to [16,32,32] output resize. "
                "That shape resembles an N=32 real-FFT half spectrum, but the log does not "
                "identify the operator. It is a compatibility risk, not proof of wrong gradients."
            ),
            "direct_npu_finite_difference_checked": False,
            "direct_npu_finite_difference_blocker": (
                "Authorized physical NPUs 0,2,4,6 were occupied by unrelated jobs during "
                "the 2026-09-01 audit, so no co-scheduled diagnostic was launched."
            ),
        },
        "g450_conditioning": {
            "sample_count": cohort["sample_count"],
            "effective_zernike_rms_ratio": cohort["summary"][
                "effective_zernike_rms_ratio"
            ],
            "displacement_rms_ratio": cohort["summary"]["displacement_rms_ratio"],
            "relu3_active_fraction": cohort["summary"]["relu3_active_fraction"],
            "zernike_gradient_l2": cohort["summary"]["zernike_gradient_l2"],
            "tilt_gradient_l2": cohort["summary"]["tilt_gradient_l2"],
            "all_gradients_finite_nonzero": cohort["summary"][
                "all_zernike_gradients_finite_nonzero"
            ]
            and cohort["summary"]["all_tilt_gradients_finite_nonzero"],
            "interpretation": (
                "The Zernike/P2S path is active but operates far from the true effective-state "
                "scale on this 10-sample high-severity cohort. This can make gradients poorly "
                "conditioned or semantically weak without making the renderer non-differentiable."
            ),
        },
        "forward_replay": {
            "true_replay_mse": cohort["summary"]["true_replay_mse"],
            "interpretation": (
                "True clean plus true realization replays archived states to near numerical "
                "precision, arguing against a gross forward-model mismatch."
            ),
        },
        "coordinate_convention_probe": audit["coordinate_convention_probe"],
        "source_audit": {
            "simulator_sha256": sha256_file(source),
            "upstream_simulator_sha256": sha256_file(upstream),
            "upstream_git_commit": "12b9ef6a32c75143c23402a7ab8f843624ca99e8",
            "frozen_parameters_do_not_detach_inputs": True,
            "render_is_not_decorated_no_grad": True,
            "training_precision": "BF16 network forward; FP32 simulator and loss",
            "path_ops": [
                "einsum",
                "fft2/ifft2/irfft2",
                "conv2d",
                "linear/ReLU P2S",
                "bilinear grid_sample with border padding",
            ],
        },
        "causal_boundary": {
            "ruled_out": [
                "simulator output detached from predicted xi",
                "simulator parameter freezing disabled input gradients",
                "all-zero or non-finite physical-loss gradients",
                "high-severity total gradient collapse on the audited cohort",
                "gross mismatch between simulator and archived forward states",
            ],
            "secondary_plausible": [
                "P2S Zernike-path scale/activation conditioning",
                "border-gradient loss for the small out-of-bounds flow fraction",
                "inherited N-1 plus align_corners=False coordinate convention",
                "NPU backward resize warning pending operator localization and direct NPU finite differences",
            ],
            "strongest_existing_failure_locus": (
                "Predicted-clean intermediate-state error amplification, based on prior "
                "teacher-forced and controlled-render routes."
            ),
        },
    }

    write_csv(DERIVED / "differentiability_a0_branch_gradients.csv", a0_rows)
    write_csv(DERIVED / "differentiability_a1_branch_gradients.csv", a1_rows)
    write_csv(DERIVED / "differentiability_per_g.csv", per_g_rows)
    write_csv(DERIVED / "differentiability_finite_difference.csv", finite_difference_rows)
    write_csv(DERIVED / "differentiability_cohort_g450.csv", cohort_rows)
    (DERIVED / "differentiability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired statistics, plots, qualitative rendering, and final scientific report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from physical_bridge.metrics import paired_bootstrap
from physical_bridge.utils import (
    atomic_touch,
    atomic_write_bytes,
    atomic_write_json,
    derive_seed,
    sha256_file,
)


METHODS = (
    "one_shot",
    "bridge_no_refinement",
    "bridge_refinement",
    "oracle_xi",
    "teacher_forced",
)


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle if line.strip())
    return result


def method_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        result[row["config"]][row["key"]] = row
    return result


def group_keys(index: dict[str, dict[str, dict[str, Any]]], predicate: Any) -> list[str]:
    return sorted(
        key for key, row in index["one_shot"].items() if predicate(float(row["terminal_G"]))
    )


def paired_comparison(
    index: dict[str, dict[str, dict[str, Any]]],
    method: str,
    keys: list[str],
    seed: int,
) -> dict[str, Any]:
    metrics = {}
    directions = {"psnr": 1.0, "ssim": 1.0, "l1": -1.0, "lpips": -1.0}
    for metric, direction in directions.items():
        deltas = [
            direction
            * (
                float(index[method][key]["final_quality"][metric])
                - float(index["one_shot"][key]["final_quality"][metric])
            )
            for key in keys
        ]
        metrics[metric] = {
            **paired_bootstrap(
                deltas, seed=derive_seed(seed, "bootstrap", method, metric, len(keys))
            ),
            "positive_means_method_is_better": True,
        }
    return metrics


def table_by_terminal(index: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    table = {}
    for terminal in (1.5, 2.25, 3.0, 3.75, 4.5):
        terminal_key = f"{terminal:.2f}"
        table[terminal_key] = {}
        for method in METHODS:
            rows = [
                row
                for row in index[method].values()
                if float(row["terminal_G"]) == terminal
            ]
            table[terminal_key][method] = {
                metric: float(np.mean([row["final_quality"][metric] for row in rows]))
                for metric in ("l1", "psnr", "ssim", "lpips")
            }
    return table


def paired_by_terminal(
    index: dict[str, dict[str, dict[str, Any]]], seed: int
) -> dict[str, Any]:
    result = {}
    for terminal in (1.5, 2.25, 3.0, 3.75, 4.5):
        keys = group_keys(index, lambda value, target=terminal: value == target)
        result[f"{terminal:.2f}"] = {
            method: paired_comparison(index, method, keys, seed)
            for method in METHODS
            if method != "one_shot"
        }
    return result


def paired_delta_rows(
    index: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    directions = {"psnr": 1.0, "ssim": 1.0, "l1": -1.0, "lpips": -1.0}
    rows = []
    for key, baseline in sorted(index["one_shot"].items()):
        for method in METHODS:
            if method == "one_shot":
                continue
            row: dict[str, Any] = {
                "key": key,
                "terminal_G": float(baseline["terminal_G"]),
                "method": method,
            }
            for metric, direction in directions.items():
                raw = float(index[method][key]["final_quality"][metric]) - float(
                    baseline["final_quality"][metric]
                )
                row[f"raw_{metric}_method_minus_one_shot"] = raw
                row[f"improvement_{metric}"] = direction * raw
            rows.append(row)
    return rows


def rollout_diagnostics(
    index: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    result = {}
    for terminal in (1.5, 2.25, 3.0, 3.75, 4.5):
        result[f"{terminal:.2f}"] = {}
        for method in METHODS:
            selected = [
                row
                for row in index[method].values()
                if float(row["terminal_G"]) == terminal
            ]
            steps = [step for row in selected for step in row["steps"]]
            values: dict[str, Any] = {
                "count": len(selected),
                "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in selected])),
                "mean_network_forwards": float(
                    np.mean([row["network_forwards"] for row in selected])
                ),
                "mean_simulator_forwards": float(
                    np.mean([row["simulator_forwards"] for row in selected])
                ),
                "mean_simulator_backwards": float(
                    np.mean([row["simulator_backwards"] for row in selected])
                ),
                "peak_npu_memory_bytes": int(
                    max(row["peak_npu_memory_bytes"] for row in selected)
                ),
            }
            if steps:
                values.update(
                    {
                        "mean_observation_residual_before": float(
                            np.mean([step["observation_residual_before"] for step in steps])
                        ),
                        "mean_observation_residual_after": float(
                            np.mean([step["observation_residual_after"] for step in steps])
                        ),
                        "mean_step_wall_seconds": float(
                            np.mean([step["wall_seconds"] for step in steps])
                        ),
                    }
                )
            result[f"{terminal:.2f}"][method] = values
    return result


def plot_test(
    table: dict[str, Any],
    paired: dict[str, Any],
    index: dict[str, dict[str, dict[str, Any]]],
    diagnostics: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    terminals = [float(value) for value in table]
    plt.figure(figsize=(8, 5))
    for method in METHODS:
        plt.plot(
            terminals,
            [table[f"{value:.2f}"][method]["psnr"] for value in terminals],
            marker="o",
            label=method,
        )
    plt.xlabel("Terminal severity G")
    plt.ylabel("Final RGB PSNR (dB)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "test_psnr_by_terminal.png", dpi=180)
    plt.close()

    names = [name for name in METHODS if name != "one_shot"]
    means = [paired["g450"][name]["psnr"]["mean_delta"] for name in names]
    lowers = [paired["g450"][name]["psnr"]["bootstrap_95_ci"][0] for name in names]
    uppers = [paired["g450"][name]["psnr"]["bootstrap_95_ci"][1] for name in names]
    plt.figure(figsize=(8, 5))
    positions = np.arange(len(names))
    plt.errorbar(
        positions,
        means,
        yerr=[np.asarray(means) - lowers, np.asarray(uppers) - means],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(positions, names, rotation=20, ha="right")
    plt.ylabel("Paired PSNR improvement over one-shot (dB)")
    plt.tight_layout()
    plt.savefig(output_dir / "g450_paired_psnr_ci.png", dpi=180)
    plt.close()

    g450_keys = group_keys(index, lambda terminal: terminal == 4.5)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in ("bridge_no_refinement", "bridge_refinement", "oracle_xi"):
        psnr_delta = [
            index[method][key]["final_quality"]["psnr"]
            - index["one_shot"][key]["final_quality"]["psnr"]
            for key in g450_keys
        ]
        lpips_improvement = [
            index["one_shot"][key]["final_quality"]["lpips"]
            - index[method][key]["final_quality"]["lpips"]
            for key in g450_keys
        ]
        axes[0].hist(psnr_delta, bins=30, alpha=0.45, label=method)
        axes[1].hist(lpips_improvement, bins=30, alpha=0.45, label=method)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("G=4.50 paired PSNR improvement (dB)")
    axes[1].set_xlabel("G=4.50 paired LPIPS improvement")
    for axis in axes:
        axis.set_ylabel("Trajectories")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "g450_paired_improvement_histograms.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    runtime = [diagnostics["4.50"][method]["mean_wall_seconds"] for method in METHODS]
    network = [diagnostics["4.50"][method]["mean_network_forwards"] for method in METHODS]
    simulator_forward = [
        diagnostics["4.50"][method]["mean_simulator_forwards"] for method in METHODS
    ]
    simulator_backward = [
        diagnostics["4.50"][method]["mean_simulator_backwards"] for method in METHODS
    ]
    memory = [
        diagnostics["4.50"][method]["peak_npu_memory_bytes"] / (1 << 30)
        for method in METHODS
    ]
    positions = np.arange(len(METHODS))
    axes[0, 0].bar(positions, runtime)
    axes[0, 0].set_ylabel("Wall seconds / trajectory")
    axes[0, 1].bar(positions, network)
    axes[0, 1].set_ylabel("Network forwards (NFE)")
    axes[1, 0].bar(positions, simulator_forward, label="forward")
    axes[1, 0].bar(
        positions, simulator_backward, bottom=simulator_forward, label="backward"
    )
    axes[1, 0].set_ylabel("Simulator calls")
    axes[1, 0].legend()
    axes[1, 1].bar(positions, memory)
    axes[1, 1].set_ylabel("Peak NPU allocated memory (GiB)")
    for axis in axes.ravel():
        axis.set_xticks(positions, METHODS, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("G=4.50 runtime, NFE, simulator calls, and memory")
    figure.tight_layout()
    figure.savefig(output_dir / "g450_runtime_nfe_simulator_memory.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for method in METHODS:
        selected = [
            row for row in index[method].values() if float(row["terminal_G"]) == 4.5
        ]
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            for step in row["steps"]:
                buckets[int(step["step"])].append(step)
        if not buckets:
            continue
        ordered = sorted(buckets)
        labels = [float(buckets[step][0]["current_g"]) for step in ordered]
        axes[0, 0].plot(
            labels,
            [np.mean([item["clean_quality"]["psnr"] for item in buckets[step]]) for step in ordered],
            marker="o",
            label=method,
        )
        axes[0, 1].plot(
            labels,
            [np.mean([item["clean_quality"]["lpips"] for item in buckets[step]]) for step in ordered],
            marker="o",
            label=method,
        )
        axes[1, 0].plot(
            labels,
            [np.mean([item["next_state_mse"] for item in buckets[step]]) for step in ordered],
            marker="o",
            label=method,
        )
    refinement_rows = [
        row
        for row in index["bridge_refinement"].values()
        if float(row["terminal_G"]) == 4.5
    ]
    refinement_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in refinement_rows:
        for step in row["steps"]:
            refinement_buckets[int(step["step"])].append(step)
    ordered = sorted(refinement_buckets)
    labels = [float(refinement_buckets[step][0]["current_g"]) for step in ordered]
    axes[1, 1].plot(
        labels,
        [
            np.mean([item["observation_residual_before"] for item in refinement_buckets[step]])
            for step in ordered
        ],
        marker="o",
        label="before",
    )
    axes[1, 1].plot(
        labels,
        [
            np.mean([item["observation_residual_after"] for item in refinement_buckets[step]])
            for step in ordered
        ],
        marker="o",
        label="after",
    )
    axes[0, 0].set_ylabel("Predicted clean PSNR (dB)")
    axes[0, 1].set_ylabel("Predicted clean LPIPS")
    axes[1, 0].set_ylabel("Next-state MSE")
    axes[1, 1].set_ylabel("Observation residual")
    for axis in axes.ravel():
        axis.set_xlabel("Current severity g")
        axis.invert_xaxis()
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("G=4.50 per-step trajectory diagnostics")
    figure.tight_layout()
    figure.savefig(output_dir / "g450_per_step_diagnostics.png", dpi=180)
    plt.close(figure)


def plot_training(run_dir: Path, output_dir: Path) -> None:
    metrics = load_jsonl([run_dir / "train_metrics.jsonl"])
    if metrics:
        updates = [row["update"] for row in metrics]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(updates, [row["loss"] for row in metrics], label="total")
        axes[0, 0].plot(updates, [row["clean_loss"] for row in metrics], label="clean")
        axes[0, 0].plot(
            updates, [row["degradation_loss"] for row in metrics], label="degradation"
        )
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylabel("Loss")
        for field, label in (
            ("shared_gradient_norm", "shared"),
            ("clean_branch_gradient_norm", "clean branch"),
            ("degradation_branch_gradient_norm", "realization branch"),
        ):
            axes[0, 1].plot(updates, [row[field] for row in metrics], label=label)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_ylabel("Gradient norm")
        axes[1, 0].plot(updates, [row["learning_rate"] for row in metrics])
        axes[1, 0].set_ylabel("Learning rate")
        axes[1, 1].plot(updates, [row["lambda_degradation"] for row in metrics])
        axes[1, 1].set_ylabel("lambda_D")
        for axis in axes.ravel():
            axis.set_xlabel("Optimizer update")
            axis.grid(alpha=0.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "training_losses_gradients_schedule.png", dpi=180)
        plt.close(figure)

    validation_paths = sorted((run_dir / "val").glob("update-*/summary.json"))
    validation = [json.loads(path.read_text(encoding="utf-8")) for path in validation_paths]
    if validation:
        updates = [int(path.parent.name.split("-", 1)[1]) for path in validation_paths]
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for severity in (0.75, 1.5, 2.25, 3.0, 3.75, 4.5):
            key = f"{severity:.2f}"
            axes[0].plot(
                updates,
                [item["per_g"][key]["psnr"] for item in validation],
                marker="o",
                label=key,
            )
            axes[1].plot(
                updates,
                [item["per_g"][key]["lpips"] for item in validation],
                marker="o",
                label=key,
            )
        axes[2].plot(
            updates,
            [item["equal_severity_mean_degradation_residual"] for item in validation],
            marker="o",
        )
        axes[0].set_ylabel("Val clean PSNR (dB)")
        axes[1].set_ylabel("Val clean LPIPS")
        axes[2].set_ylabel("Val terminal reconstruction residual")
        for axis in axes:
            axis.set_xlabel("Optimizer update")
            axis.grid(alpha=0.2)
        axes[0].legend(title="g", fontsize=7)
        axes[1].legend(title="g", fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "validation_metrics_per_g.png", dpi=180)
        plt.close(figure)


def plot_selections(run_dir: Path, output_dir: Path) -> None:
    sources = (
        (run_dir / "val" / "refinement_lr" / "summary.json", "learning_rate", "LR"),
        (run_dir / "val" / "k_selection" / "summary.json", "K", "K"),
    )
    if not all(path.exists() for path, _, _ in sources):
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, (path, field, title) in zip(axes, sources, strict=True):
        summary = json.loads(path.read_text(encoding="utf-8"))
        configurations = {item["name"]: item for item in summary["configurations"]}
        names = list(summary["results"])
        x_values = [configurations[name][field] for name in names]
        psnr = [summary["results"][name]["mean"]["psnr"] for name in names]
        lpips = [summary["results"][name]["mean"]["lpips"] for name in names]
        axis.plot(x_values, psnr, marker="o", color="tab:blue", label="PSNR")
        axis.set_xlabel(title)
        axis.set_ylabel("Final PSNR (dB)", color="tab:blue")
        twin = axis.twinx()
        twin.plot(x_values, lpips, marker="s", color="tab:orange", label="LPIPS")
        twin.set_ylabel("Final LPIPS", color="tab:orange")
        axis.grid(alpha=0.2)
    figure.suptitle("Validation-only inference hyperparameter selections")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_lr_and_k_selection.png", dpi=180)
    plt.close(figure)


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    value = tensor.squeeze(0).float().clamp(0, 1)
    array = value.mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def render_qualitative(input_root: Path, output_root: Path) -> None:
    for method_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for tensor_path in sorted(method_dir.glob("*.pt")):
            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            steps = payload["captured"]
            columns = (
                "current",
                "clean_prediction",
                "reconstruction_before",
                "reconstruction_after",
                "next_state",
            )
            rows = len(steps) + 1
            tile, header = 256, 24
            canvas = Image.new(
                "RGB", (len(columns) * tile, rows * (tile + header)), "white"
            )
            draw = ImageDraw.Draw(canvas)
            context = (
                ("target clean", payload["target_clean"]),
                ("observation", payload["observation"]),
                ("final clean", payload["final_prediction"]),
            )
            for column_index, (label, value) in enumerate(context):
                left = column_index * tile
                canvas.paste(tensor_image(value), (left, header))
                draw.text((left + 4, 4), label, fill="black")
            for row_index, step in enumerate(steps):
                top = (row_index + 1) * (tile + header)
                for column_index, name in enumerate(columns):
                    image = tensor_image(step[name])
                    left = column_index * tile
                    canvas.paste(image, (left, top + header))
                    draw.text((left + 4, top + 4), f"step {row_index} {name}", fill="black")
            destination = output_root / method_dir.name / f"{tensor_path.stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp.png")
            canvas.save(temporary, format="PNG", compress_level=3)
            temporary.replace(destination)


def markdown_table(table: dict[str, Any], metric: str) -> str:
    lines = ["| G | " + " | ".join(METHODS) + " |", "|---:|" + "---:|" * len(METHODS)]
    for terminal, methods in table.items():
        values = " | ".join(f"{methods[method][metric]:.4f}" for method in METHODS)
        lines.append(f"| {terminal} | {values} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--test-rollout-dir", type=Path, required=True)
    parser.add_argument("--teacher-forced-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    rows = load_jsonl(sorted(args.test_rollout_dir.glob("rank-*.jsonl")))
    index = method_index(rows)
    missing = set(METHODS) - set(index)
    if missing:
        raise ValueError(f"Missing test methods: {sorted(missing)}")
    reference_keys = set(index["one_shot"])
    if any(set(index[method]) != reference_keys for method in METHODS):
        raise ValueError("Comparator keys are not exactly paired")
    groups = {
        "g450": group_keys(index, lambda terminal: terminal == 4.5),
        "high_g300_to_g450": group_keys(index, lambda terminal: terminal >= 3.0),
        "all": sorted(reference_keys),
    }
    paired = {
        group: {
            method: paired_comparison(index, method, keys, args.seed)
            for method in METHODS
            if method != "one_shot"
        }
        for group, keys in groups.items()
    }
    table = table_by_terminal(index)
    terminal_paired = paired_by_terminal(index, args.seed)
    delta_rows = paired_delta_rows(index)
    diagnostics = rollout_diagnostics(index)
    frozen = json.loads((args.run_dir / "frozen_inference.json").read_text(encoding="utf-8"))
    bridge_g450 = paired["g450"]["bridge_refinement"]
    bridge_supported = (
        int(frozen["K"]) > 0
        and
        bridge_g450["psnr"]["mean_delta"] > 0
        and bridge_g450["psnr"]["bootstrap_95_ci"][0] > 0
        and bridge_g450["lpips"]["mean_delta"] >= 0
    )
    oracle_delta = paired["g450"]["oracle_xi"]["psnr"]["mean_delta"]
    refinement_delta = bridge_g450["psnr"]["mean_delta"]
    no_refinement_delta = paired["g450"]["bridge_no_refinement"]["psnr"]["mean_delta"]
    if int(frozen["K"]) == 0:
        mechanism = "validation_selected_one_shot_progressive_not_supported"
    elif oracle_delta <= 0:
        mechanism = "physical_path_not_supported"
    elif refinement_delta <= 0:
        mechanism = "realization_estimation_or_refinement_bottleneck"
    elif no_refinement_delta > refinement_delta:
        mechanism = "adam_refinement_hurts_relative_to_no_refinement"
    else:
        mechanism = "progressive_bridge_supported" if bridge_supported else "mixed_or_inconclusive"
    training = json.loads((args.run_dir / "training_summary.json").read_text(encoding="utf-8"))
    dataset = json.loads((args.dataset_root / "generation_summary.json").read_text(encoding="utf-8"))
    dataset_statistics_path = args.dataset_root / "statistics" / "summary.json"
    dataset_statistics = json.loads(dataset_statistics_path.read_text(encoding="utf-8"))
    teacher_forced = json.loads((args.teacher_forced_dir / "summary.json").read_text(encoding="utf-8"))
    result = {
        "bridge_supported": bridge_supported,
        "bridge_success_rule": {
            "requires_progressive_k_above_zero": True,
            "g450_psnr_mean_delta_above_zero": True,
            "g450_psnr_bootstrap_ci_lower_above_zero": True,
            "g450_lpips_improvement_nonnegative": True,
        },
        "mechanism_judgment": mechanism,
        "frozen_inference": frozen,
        "paired": paired,
        "paired_by_terminal": terminal_paired,
        "per_terminal": table,
        "rollout_diagnostics_by_terminal": diagnostics,
        "teacher_forced_branch": teacher_forced,
        "dataset_statistics": {
            "path": str(dataset_statistics_path),
            "sha256": sha256_file(dataset_statistics_path),
            "trajectory_count": dataset_statistics["trajectory_count"],
        },
        "test_row_count": len(rows),
    }
    plots = args.run_dir / "plots"
    plot_test(table, paired, index, diagnostics, plots)
    plot_training(args.run_dir, plots)
    plot_selections(args.run_dir, plots)
    qualitative_root = args.test_rollout_dir / "qualitative_tensors"
    if qualitative_root.exists():
        render_qualitative(qualitative_root, args.run_dir / "qualitative" / "test_rollouts")
    delta_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in delta_rows
    )
    atomic_write_bytes(args.run_dir / "test" / "paired_deltas.jsonl", delta_bytes)
    atomic_write_json(args.run_dir / "test" / "paired_statistics.json", result)

    delta = bridge_g450["psnr"]
    conclusion = (
        "progressive bridge 获得冻结判据支持"
        if bridge_supported
        else "progressive bridge 未达到冻结成功判据"
    )
    report = f"""# Known-g Physical Turbulence Bridge v0.1 最终报告

## 结论

{conclusion}。G=4.50 上 Bridge + refinement 相对 one-shot 的 paired PSNR 变化为
{delta['mean_delta']:.4f} dB，95% bootstrap CI 为
[{delta['bootstrap_95_ci'][0]:.4f}, {delta['bootstrap_95_ci'][1]:.4f}]；机制分类为
`{mechanism}`。冻结 K={frozen['K']}，refinement learning rate={frozen['learning_rate']}。

## 原始假设

沿固定 realization 的 PiRN/P2S 六级 severity trajectory 逐步恢复，可能优于同一网络直接 one-shot 恢复。

## 执行与冻结

- 数据：Places365-Standard，{dataset['total_trajectories']} 条 trajectory，六级状态与 simulator-native FP32 ξ 完整保存。
- 训练：完成 {training['completed_updates']} updates；使用 EMA checkpoint；effective batch 固定 16。
- checkpoint：`{frozen['checkpoint']}`，SHA-256 `{frozen['checkpoint_sha256']}`。
- test 前冻结：checkpoint、refinement LR、K 均来自 val；test 未调参。
- implementation fixes：NPU device port、真实 512 ξ shape、batched scalar-G、decoder 512 上采样、indexed tar、BF16/FP32 边界、DDP/checkpoint/resume、数值与磁盘保护。
- method changes：无。

## Frozen Test L1

{markdown_table(table, 'l1')}

## Frozen Test PSNR

{markdown_table(table, 'psnr')}

## Frozen Test SSIM

{markdown_table(table, 'ssim')}

## Frozen Test LPIPS

{markdown_table(table, 'lpips')}

## 机制解释

- Oracle-ξ 相对 one-shot 的 G=4.50 PSNR 变化：{oracle_delta:.4f} dB。
- Bridge no-refinement 相对 one-shot：{no_refinement_delta:.4f} dB。
- Bridge + refinement 相对 one-shot：{refinement_delta:.4f} dB。
- G=4.50 Bridge + refinement paired PSNR wins/ties/losses：{delta['paired_wins']}/{delta['paired_ties']}/{delta['paired_losses']}。
- G=4.50 Bridge + refinement 的 LPIPS improvement：{bridge_g450['lpips']['mean_delta']:.6f}（正值表示优于 one-shot）。
- teacher-forced branch 的完整六级指标见 `test/teacher_forced_branch/summary.json`。

## 过程证据

- 数据退化曲线、replay histogram、range/clipping、物理 descriptor 与 split 分布：`data/places_p2s_bridge_v01/statistics/`。
- 训练 loss、gradient norm、LR、lambda_D 与 val per-g 曲线：`plots/training_losses_gradients_schedule.png`、`plots/validation_metrics_per_g.png`。
- LR/K 冻结曲线：`plots/validation_lr_and_k_selection.png`。
- paired delta histogram、per-step quality/next-state/residual、runtime/NFE/simulator/memory：`plots/`。
- 逐条 paired delta：`test/paired_deltas.jsonl`；完整分 terminal bootstrap 与 residual/runtime 统计：`test/paired_statistics.json`。

## 可复现性与安全

所有 manifest、source/checkpoint hash、环境、逐步 tensor、每步 residual/NFE/runtime、paired rows 和完成标记均保存在本 run 目录。训练和评估仅使用物理 NPU 4–7；数据仅写入 `/data*/yanglin`。数据 tar、必要 checkpoint、完整 ξ 和固定定性样本被保留；只清理 checksum 验证后的临时 spool。
"""
    atomic_write_bytes(args.run_dir / "final_report.md", report.encode("utf-8"))
    result["final_report_sha256"] = sha256_file(args.run_dir / "final_report.md")
    atomic_write_json(args.run_dir / "final_summary.json", result)
    atomic_touch(args.run_dir / "FINAL_REPORT_COMPLETE")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

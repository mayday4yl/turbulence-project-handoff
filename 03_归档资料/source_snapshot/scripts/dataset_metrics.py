#!/usr/bin/env python3
"""Compute and plot the frozen full-dataset degradation statistics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def aggregate(rows: list[dict[str, Any]], severities: tuple[float, ...]) -> dict[str, Any]:
    metric_names = (
        "l1",
        "psnr",
        "ssim",
        "lpips",
        "image_min",
        "image_max",
        "below_zero_fraction",
        "above_one_fraction",
        "clipping_fraction",
        "zernike_rms",
        "tilt_component_rms",
        "warp_mean_magnitude",
        "warp_max_magnitude",
        "blur_weight_rms",
    )
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        per_g: dict[str, Any] = {}
        for severity in severities:
            values = [
                state
                for row in split_rows
                for state in row["states"]
                if float(state["g"]) == severity
            ]
            per_g[f"{severity:.2f}"] = {
                name: {
                    "mean": float(np.mean([float(item[name]) for item in values])),
                    "std": float(np.std([float(item[name]) for item in values])),
                    "q05": float(np.quantile([float(item[name]) for item in values], 0.05)),
                    "median": float(np.median([float(item[name]) for item in values])),
                    "q95": float(np.quantile([float(item[name]) for item in values], 0.95)),
                }
                for name in metric_names
            }
        result[split] = {
            "trajectories": len(split_rows),
            "source_images": len({row["source_image_id"] for row in split_rows}),
            "terminal_counts": dict(
                sorted(Counter(f"{float(row['terminal_G']):.2f}" for row in split_rows).items())
            ),
            "category_trajectory_counts": dict(
                sorted(
                    Counter(str(row["source_category_id"]) for row in split_rows).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "per_g": per_g,
        }
    return result


def plot_degradation(summary: dict[str, Any], output: Path) -> None:
    severities = [float(value) for value in summary["train"]["per_g"]]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric, label in zip(
        axes.ravel(),
        ("l1", "psnr", "ssim", "lpips"),
        ("L1", "PSNR (dB)", "SSIM", "LPIPS-Alex"),
        strict=True,
    ):
        for split in ("train", "val", "test"):
            means = [summary[split]["per_g"][f"{g:.2f}"][metric]["mean"] for g in severities]
            axis.plot(severities, means, marker="o", label=split)
        axis.set_xlabel("Severity g")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Original degradation curves: X_g versus clean")
    figure.tight_layout()
    figure.savefig(output / "original_degradation_curves.png", dpi=180)
    plt.close(figure)


def plot_physical(summary: dict[str, Any], output: Path) -> None:
    severities = [float(value) for value in summary["train"]["per_g"]]
    metrics = (
        ("zernike_rms", "Effective Zernike RMS"),
        ("tilt_component_rms", "Effective tilt component RMS"),
        ("warp_mean_magnitude", "Mean warp magnitude"),
        ("blur_weight_rms", "P2S blur-weight RMS"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        for split in ("train", "val", "test"):
            means = [summary[split]["per_g"][f"{g:.2f}"][metric]["mean"] for g in severities]
            axis.plot(severities, means, marker="o", label=split)
        axis.set_xlabel("Severity g")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Simulator physical descriptor curves")
    figure.tight_layout()
    figure.savefig(output / "physical_descriptor_curves.png", dpi=180)
    plt.close(figure)


def plot_ranges(summary: dict[str, Any], output: Path) -> None:
    severities = [float(value) for value in summary["train"]["per_g"]]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for split in ("train", "val", "test"):
        minimum = [summary[split]["per_g"][f"{g:.2f}"]["image_min"]["mean"] for g in severities]
        maximum = [summary[split]["per_g"][f"{g:.2f}"]["image_max"]["mean"] for g in severities]
        clipping = [
            summary[split]["per_g"][f"{g:.2f}"]["clipping_fraction"]["mean"]
            for g in severities
        ]
        axes[0].plot(severities, minimum, marker="o", label=split)
        axes[1].plot(severities, maximum, marker="o", label=split)
        axes[2].plot(severities, clipping, marker="o", label=split)
    for axis, label in zip(
        axes, ("Mean image minimum", "Mean image maximum", "Fraction outside [0,1]"), strict=True
    ):
        axis.set_xlabel("Severity g")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / "image_range_and_clipping.png", dpi=180)
    plt.close(figure)


def plot_distributions(summary: dict[str, Any], output: Path) -> None:
    splits = ("train", "val", "test")
    terminals = sorted(summary["train"]["terminal_counts"], key=float)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(terminals))
    width = 0.25
    for index, split in enumerate(splits):
        counts = [summary[split]["terminal_counts"].get(value, 0) for value in terminals]
        axes[0].bar(x + (index - 1) * width, counts, width=width, label=split)
        category_counts = summary[split]["category_trajectory_counts"]
        axes[1].plot(
            [int(value) for value in category_counts],
            [category_counts[value] for value in category_counts],
            linewidth=1,
            label=split,
        )
    axes[0].set_xticks(x, terminals)
    axes[0].set_xlabel("Terminal G")
    axes[0].set_ylabel("Trajectories")
    axes[0].legend()
    axes[1].set_xlabel("Places365 category id")
    axes[1].set_ylabel("Trajectories")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "split_terminal_category_distributions.png", dpi=180)
    plt.close(figure)


def plot_replay(preflight: dict[str, Any], output: Path) -> None:
    replay = preflight["replay"]["max_abs_per_trajectory"]
    stored = preflight["stored_float16_roundtrip"]["max_abs_per_trajectory"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(replay, bins=max(1, min(16, len(replay))))
    axes[0].axvline(preflight["replay"]["threshold"], color="red", linestyle="--", label="threshold")
    axes[0].set_xlabel("Deterministic replay max absolute error")
    axes[0].set_ylabel("Trajectories")
    axes[0].legend()
    axes[1].hist(stored, bins=max(1, min(16, len(stored))))
    axes[1].set_xlabel("Stored float16 roundtrip max absolute error")
    axes[1].set_ylabel("Trajectories")
    figure.suptitle(f"Mini32 replay evidence (n={len(replay)})")
    figure.tight_layout()
    figure.savefig(output / "replay_error_histogram.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    from physical_bridge.data import BridgeArchive, FROZEN_G_GRID
    from physical_bridge.metrics import LPIPSMetric, per_sample_l1, per_sample_psnr, per_sample_ssim
    from physical_bridge.utils import atomic_touch, atomic_write_json

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    if world_size > 1:
        torch.distributed.init_process_group(backend="hccl", timeout=dt.timedelta(hours=6))
    barrier = torch.distributed.barrier if world_size > 1 else lambda: None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank-{rank:02d}.jsonl"
    existing = load_jsonl([rank_path])
    complete = {(row["split"], row["key"]) for row in existing}
    metric = LPIPSMetric(device)
    start = time.monotonic()
    assigned = 0
    for split in ("train", "val", "test"):
        archive = BridgeArchive(args.dataset_root, split)
        for position, record in enumerate(archive.records):
            if position % world_size != rank or (split, record["key"]) in complete:
                continue
            clean_cpu = archive.clean(record)
            states_cpu = torch.stack([archive.state(record, g) for g in FROZEN_G_GRID])
            clean = clean_cpu.unsqueeze(0).expand_as(states_cpu).to(device)
            states = states_cpu.to(device)
            with torch.no_grad():
                l1 = per_sample_l1(states, clean)
                psnr = per_sample_psnr(states, clean)
                ssim = per_sample_ssim(states, clean)
                lpips = metric(states, clean)
            generation = {float(item["g"]): item for item in record["severity_statistics"]}
            state_rows = []
            for index, severity in enumerate(FROZEN_G_GRID):
                item = generation[severity]
                below = float(item["below_zero_fraction"])
                above = float(item["above_one_fraction"])
                state_rows.append(
                    {
                        "g": severity,
                        "l1": float(l1[index].cpu()),
                        "psnr": float(psnr[index].cpu()),
                        "ssim": float(ssim[index].cpu()),
                        "lpips": float(lpips[index].cpu()),
                        "image_min": float(item["min"]),
                        "image_max": float(item["max"]),
                        "below_zero_fraction": below,
                        "above_one_fraction": above,
                        "clipping_fraction": below + above,
                        "zernike_rms": float(item["zernike_rms"]),
                        "tilt_component_rms": float(item["tilt_component_rms"]),
                        "warp_mean_magnitude": float(item["warp_mean_magnitude"]),
                        "warp_max_magnitude": float(item["warp_max_magnitude"]),
                        "blur_weight_rms": float(item["blur_weight_rms"]),
                    }
                )
            row = {
                "split": split,
                "key": record["key"],
                "source_image_id": record["source_image_id"],
                "source_category_id": int(record["source_category_id"]),
                "terminal_G": float(record["terminal_G"]),
                "states": state_rows,
            }
            append_jsonl(rank_path, row)
            complete.add((split, record["key"]))
            assigned += 1
            if assigned % 64 == 0:
                torch.npu.empty_cache()
    atomic_touch(args.output_dir / f"rank-{rank:02d}.complete")
    barrier()
    if rank == 0:
        rows = load_jsonl(sorted(args.output_dir.glob("rank-*.jsonl")))
        expected = sum(
            len(BridgeArchive(args.dataset_root, split).records)
            for split in ("train", "val", "test")
        )
        identities = {(row["split"], row["key"]) for row in rows}
        if len(rows) != expected or len(identities) != expected:
            raise ValueError(
                f"Dataset statistics expected {expected} unique trajectories, got "
                f"{len(rows)} rows/{len(identities)} unique"
            )
        summary = aggregate(rows, FROZEN_G_GRID)
        preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        result = {
            "trajectory_count": expected,
            "g_grid": list(FROZEN_G_GRID),
            "splits": summary,
            "preflight_replay": {
                "scope": "frozen mini32 preflight",
                "trajectory_count": int(preflight["trajectory_count"]),
                "threshold": float(preflight["replay"]["threshold"]),
                "max_abs": float(preflight["replay"]["max_abs"]),
                "stored_float16_max_abs": float(
                    preflight["stored_float16_roundtrip"]["max_abs"]
                ),
            },
            "wall_seconds_rank0_after_resume": time.monotonic() - start,
        }
        atomic_write_json(args.output_dir / "summary.json", result)
        plot_degradation(summary, args.output_dir)
        plot_physical(summary, args.output_dir)
        plot_ranges(summary, args.output_dir)
        plot_distributions(summary, args.output_dir)
        plot_replay(preflight, args.output_dir)
        atomic_touch(args.output_dir / "DATASET_STATISTICS_COMPLETE")
        print(json.dumps(result, indent=2, sort_keys=True))
    barrier()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

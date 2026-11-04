#!/usr/bin/env python3
"""Reproducible, read-only analysis of the completed V0.2 Stage A1 artifacts.

The script never imports the training project and never writes to the remote run.
It reads the locally copied JSON/JSONL evidence and emits small audit tables beside
this file for the companion notebook and technical report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "derived"
BOOTSTRAP_SEED = 2026082899
BOOTSTRAP_DRAWS = 10_000
G_GRID = (0.75, 1.50, 2.25, 3.00, 3.75, 4.50)
TERMINAL_GRID = (1.50, 2.25, 3.00, 3.75, 4.50)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def percentile_interval(values: Sequence[float]) -> list[float]:
    return [quantile(values, 0.025), quantile(values, 0.975)]


def paired_bootstrap(
    rows: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float],
    *,
    seed_offset: int = 0,
) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    return [
        statistic([rows[rng.randrange(len(rows))] for _ in rows])
        for _ in range(BOOTSTRAP_DRAWS)
    ]


def stratified_bootstrap(
    groups: dict[float, list[dict]],
    statistic: Callable[[Sequence[dict]], float],
    *,
    seed_offset: int = 0,
) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample: list[dict] = []
        for group in sorted(groups):
            rows = groups[group]
            sample.extend(rows[rng.randrange(len(rows))] for _ in rows)
        draws.append(statistic(sample))
    return draws


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = stats.fmean(left)
    right_mean = stats.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else math.nan


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def summary(values: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": stats.fmean(values),
        "median": stats.median(values),
        "std": stats.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "max": max(values),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_manifest(batches: Sequence[dict]) -> dict:
    errors: list[str] = []
    if len(batches) != 3000:
        errors.append(f"expected 3000 batches, got {len(batches)}")
    updates = [int(batch["update"]) for batch in batches]
    if updates != list(range(len(batches))):
        errors.append("update indices are not exactly sequential from zero")
    batch_sizes = [len(batch["samples"]) for batch in batches]
    if set(batch_sizes) != {16}:
        errors.append(f"batch sizes are not all 16: {sorted(set(batch_sizes))}")

    samples = [sample for batch in batches for sample in batch["samples"]]
    key_to_g: dict[str, set[float]] = defaultdict(set)
    terminal_counts = Counter()
    current_counts = Counter()
    h_counts = Counter()
    s_t_counts = Counter()
    legal_transition_errors = 0
    severity_membership_errors = 0
    for sample in samples:
        key = str(sample["key"])
        terminal_g = float(sample["G"])
        current_g = float(sample["g"])
        h = None if sample["h"] is None else float(sample["h"])
        s_t = float(sample["s_T"])
        key_to_g[key].add(terminal_g)
        terminal_counts[terminal_g] += 1
        current_counts[(terminal_g, current_g)] += 1
        h_counts[(current_g, h)] += 1
        s_t_counts[(terminal_g, current_g, s_t)] += 1
        allowed = [value for value in G_GRID if value <= terminal_g]
        if current_g not in allowed or s_t not in allowed:
            severity_membership_errors += 1
        expected_h = None if current_g == G_GRID[0] else G_GRID[G_GRID.index(current_g) - 1]
        if h != expected_h:
            legal_transition_errors += 1

    inconsistent_keys = [key for key, values in key_to_g.items() if len(values) != 1]
    unique_by_terminal = Counter(next(iter(key_to_g[key])) for key in key_to_g)
    if inconsistent_keys:
        errors.append(f"{len(inconsistent_keys)} keys map to multiple terminal severities")
    if unique_by_terminal != Counter({value: 400 for value in TERMINAL_GRID}):
        errors.append(f"unexpected unique trajectory balance: {dict(unique_by_terminal)}")
    if legal_transition_errors:
        errors.append(f"{legal_transition_errors} samples violate adjacent-h semantics")
    if severity_membership_errors:
        errors.append(f"{severity_membership_errors} samples use severity outside trajectory")

    return {
        "batch_count": len(batches),
        "sample_count": len(samples),
        "batch_sizes": dict(Counter(batch_sizes)),
        "unique_trajectory_count": len(key_to_g),
        "unique_trajectories_by_terminal_G": {
            f"{key:.2f}": unique_by_terminal[key] for key in TERMINAL_GRID
        },
        "sample_counts_by_terminal_G": {
            f"{key:.2f}": terminal_counts[key] for key in TERMINAL_GRID
        },
        "current_g_counts": {
            f"G={terminal:.2f},g={current:.2f}": count
            for (terminal, current), count in sorted(current_counts.items())
        },
        "legal_transition_errors": legal_transition_errors,
        "severity_membership_errors": severity_membership_errors,
        "errors": errors,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_rows: list[dict] = []
    rank_counts: dict[str, int] = {}
    for path in sorted((DATA / "eval").glob("rank-*.jsonl")):
        rank_rows = read_jsonl(path)
        rank_counts[path.name] = len(rank_rows)
        eval_rows.extend(rank_rows)
    report = read_json(DATA / "eval" / "v02_pilot_report.json")
    val_subset = read_json(DATA / "val_subset.json")
    train_rows = read_jsonl(DATA / "a1" / "train_metrics.jsonl")
    batches = read_jsonl(DATA / "batches.jsonl")
    a0 = read_json(DATA / "a0" / "v02_gradient_audit.json")

    by_g: dict[float, list[dict]] = defaultdict(list)
    for row in eval_rows:
        by_g[float(row["G"])].append(row)
    g450 = by_g[4.50]

    expected_val_keys = {
        key for bucket in val_subset.values() for key in bucket
    }
    observed_val_keys = {row["key"] for row in eval_rows}
    all_numeric_finite = True
    for row in eval_rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                all_numeric_finite = False
            if isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, (int, float)) and not math.isfinite(float(nested)):
                        all_numeric_finite = False

    data_quality = {
        "rank_counts": rank_counts,
        "evaluation_row_count": len(eval_rows),
        "evaluation_unique_key_count": len(observed_val_keys),
        "val_manifest_unique_key_count": len(expected_val_keys),
        "eval_exactly_matches_val_manifest": observed_val_keys == expected_val_keys,
        "duplicate_eval_key_count": len(eval_rows) - len(observed_val_keys),
        "bucket_counts": {f"{g:.2f}": len(by_g[g]) for g in TERMINAL_GRID},
        "k1_rows": sum("pilot_k1_psnr" in row for row in eval_rows),
        "k1_only_on_g450": all(
            ("pilot_k1_psnr" in row) == (float(row["G"]) == 4.50)
            for row in eval_rows
        ),
        "all_numeric_values_finite": all_numeric_finite,
        "training_metric_rows": len(train_rows),
        "training_updates_exact_1_to_3000": [row["update"] for row in train_rows]
        == list(range(1, 3001)),
        "full_trajectory_non_null_rows": sum(
            row["full_trajectory_loss"] is not None for row in train_rows
        ),
        "nonfinite_training_rows": sum(
            not all(
                math.isfinite(float(row[key]))
                for key in ("loss", "clean_loss", "trajectory_estimator_loss", "bridge_loss")
            )
            for row in train_rows
        ),
        "manifest": profile_manifest(batches),
    }

    overall_k0_stat = lambda rows: stats.fmean(
        row["pilot_k0_psnr"] - row["baseline_k0_psnr"] for row in rows
    )
    overall_k0_boot = stratified_bootstrap(by_g, overall_k0_stat, seed_offset=1)
    per_g_table: list[dict] = []
    per_g_details: dict[str, dict] = {}
    for index, terminal_g in enumerate(TERMINAL_GRID):
        rows = by_g[terminal_g]
        deltas = [row["pilot_k0_psnr"] - row["baseline_k0_psnr"] for row in rows]
        boot = paired_bootstrap(rows, overall_k0_stat, seed_offset=10 + index)
        record = {
            "G": terminal_g,
            "n": len(rows),
            "baseline_k0_psnr": stats.fmean(row["baseline_k0_psnr"] for row in rows),
            "pilot_k0_psnr": stats.fmean(row["pilot_k0_psnr"] for row in rows),
            "paired_delta_db": stats.fmean(deltas),
            "paired_delta_median_db": stats.median(deltas),
            "bootstrap_ci_low_db": percentile_interval(boot)[0],
            "bootstrap_ci_high_db": percentile_interval(boot)[1],
            "pilot_win_count": sum(delta > 0 for delta in deltas),
            "pilot_win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        }
        per_g_table.append(record)
        per_g_details[f"{terminal_g:.2f}"] = {**summary(deltas), "bootstrap_ci": percentile_interval(boot)}

    def k1_gain_stat(rows: Sequence[dict]) -> float:
        return stats.fmean(row["pilot_k1_psnr"] - row["baseline_k1_psnr"] for row in rows)

    def pilot_gap_stat(rows: Sequence[dict]) -> float:
        return stats.fmean(row["pilot_k1_psnr"] - row["pilot_k0_psnr"] for row in rows)

    def baseline_gap_stat(rows: Sequence[dict]) -> float:
        return stats.fmean(row["baseline_k1_psnr"] - row["baseline_k0_psnr"] for row in rows)

    def gap_closure_stat(rows: Sequence[dict]) -> float:
        return stats.fmean(
            (row["pilot_k1_psnr"] - row["pilot_k0_psnr"])
            - (row["baseline_k1_psnr"] - row["baseline_k0_psnr"])
            for row in rows
        )

    def mse_reduction_stat(rows: Sequence[dict]) -> float:
        return 1.0 - stats.fmean(row["pilot_next_state_mse"] for row in rows) / stats.fmean(
            row["baseline_next_state_mse"] for row in rows
        )

    g450_metrics: dict[str, dict | float | int | list[float]] = {}
    for index, (name, function) in enumerate(
        [
            ("k1_gain_vs_v01_db", k1_gain_stat),
            ("pilot_k1_minus_k0_db", pilot_gap_stat),
            ("baseline_k1_minus_k0_db", baseline_gap_stat),
            ("progressive_gap_closure_db", gap_closure_stat),
            ("next_state_mse_reduction_ratio_of_means", mse_reduction_stat),
        ]
    ):
        boot = paired_bootstrap(g450, function, seed_offset=100 + index)
        g450_metrics[name] = {
            "estimate": function(g450),
            "bootstrap_ci": percentile_interval(boot),
        }

    k1_gain = [row["pilot_k1_psnr"] - row["baseline_k1_psnr"] for row in g450]
    pilot_gap = [row["pilot_k1_psnr"] - row["pilot_k0_psnr"] for row in g450]
    baseline_gap = [row["baseline_k1_psnr"] - row["baseline_k0_psnr"] for row in g450]
    gap_closure = [pilot - baseline for pilot, baseline in zip(pilot_gap, baseline_gap)]
    mse_reduction = [
        1.0 - row["pilot_next_state_mse"] / row["baseline_next_state_mse"]
        for row in g450
    ]
    k0_change_g450 = [
        row["pilot_k0_psnr"] - row["baseline_k0_psnr"] for row in g450
    ]
    g450_metrics.update(
        {
            "n": len(g450),
            "k1_gain_distribution": summary(k1_gain),
            "pilot_k1_minus_k0_distribution": summary(pilot_gap),
            "progressive_gap_closure_distribution": summary(gap_closure),
            "per_sample_mse_reduction_distribution": summary(mse_reduction),
            "pearson_mse_reduction_vs_k1_gain": pearson(mse_reduction, k1_gain),
            "spearman_mse_reduction_vs_k1_gain": spearman(mse_reduction, k1_gain),
            "pearson_mse_reduction_vs_pilot_gap": pearson(mse_reduction, pilot_gap),
            "spearman_mse_reduction_vs_pilot_gap": spearman(mse_reduction, pilot_gap),
            "pearson_k0_change_vs_k1_gain": pearson(k0_change_g450, k1_gain),
            "pilot_k1_better_than_k0_count": sum(value > 0 for value in pilot_gap),
            "gap_closed_count": sum(value > 0 for value in gap_closure),
            "mse_improved_count": sum(value > 0 for value in mse_reduction),
        }
    )

    # Leave-one-out ranges quantify whether any single trajectory drives a mean.
    for name, function in (
        ("k1_gain_vs_v01_db", k1_gain_stat),
        ("pilot_k1_minus_k0_db", pilot_gap_stat),
        ("progressive_gap_closure_db", gap_closure_stat),
        ("next_state_mse_reduction_ratio_of_means", mse_reduction_stat),
    ):
        values = [function(g450[:index] + g450[index + 1 :]) for index in range(len(g450))]
        g450_metrics[name]["leave_one_out_range"] = [min(values), max(values)]

    per_sample_table: list[dict] = []
    for row in sorted(g450, key=lambda value: value["key"]):
        per_sample_table.append(
            {
                "key": row["key"],
                "G": row["G"],
                "baseline_k0_psnr": row["baseline_k0_psnr"],
                "pilot_k0_psnr": row["pilot_k0_psnr"],
                "k0_change_db": row["pilot_k0_psnr"] - row["baseline_k0_psnr"],
                "baseline_k1_psnr": row["baseline_k1_psnr"],
                "pilot_k1_psnr": row["pilot_k1_psnr"],
                "k1_gain_vs_v01_db": row["pilot_k1_psnr"] - row["baseline_k1_psnr"],
                "baseline_k1_minus_k0_db": row["baseline_k1_psnr"]
                - row["baseline_k0_psnr"],
                "pilot_k1_minus_k0_db": row["pilot_k1_psnr"] - row["pilot_k0_psnr"],
                "progressive_gap_closure_db": (
                    row["pilot_k1_psnr"] - row["pilot_k0_psnr"]
                )
                - (row["baseline_k1_psnr"] - row["baseline_k0_psnr"]),
                "baseline_next_state_mse": row["baseline_next_state_mse"],
                "pilot_next_state_mse": row["pilot_next_state_mse"],
                "next_state_mse_reduction": 1.0
                - row["pilot_next_state_mse"] / row["baseline_next_state_mse"],
            }
        )

    windows = [
        ("updates 1-200", train_rows[:200]),
        ("updates 201-1000", train_rows[200:1000]),
        ("updates 1001-2000", train_rows[1000:2000]),
        ("updates 2001-2800", train_rows[2000:2800]),
        ("updates 2801-3000", train_rows[2800:]),
    ]
    training_window_table: list[dict] = []
    for name, rows in windows:
        training_window_table.append(
            {
                "window": name,
                "n": len(rows),
                "lr_start": rows[0]["learning_rate"],
                "lr_end": rows[-1]["learning_rate"],
                "clean_loss_median": stats.median(row["clean_loss"] for row in rows),
                "trajectory_loss_median": stats.median(
                    row["trajectory_estimator_loss"] for row in rows
                ),
                "bridge_loss_median": stats.median(row["bridge_loss"] for row in rows),
                "total_loss_median": stats.median(row["loss"] for row in rows),
                "grad_shared_median": stats.median(
                    row["gradient_norms"]["shared"] for row in rows
                ),
                "grad_clean_median": stats.median(
                    row["gradient_norms"]["clean"] for row in rows
                ),
                "grad_realization_median": stats.median(
                    row["gradient_norms"]["realization"] for row in rows
                ),
                "step_seconds_median": stats.median(row["step_wall_seconds"] for row in rows),
            }
        )

    first = train_rows[:200]
    last = train_rows[-200:]
    training_dynamics = {
        "clipped_step_count": sum(row["gradient_norms"]["preclip"] > 1.0 for row in train_rows),
        "peak_npu_memory_bytes": max(row["peak_npu_memory_bytes"] for row in train_rows),
        "median_step_seconds": stats.median(row["step_wall_seconds"] for row in train_rows),
        "valid_transitions": summary([row["valid_transitions"] for row in train_rows]),
        "first_to_last_median_ratios": {
            "clean": stats.median(row["clean_loss"] for row in last)
            / stats.median(row["clean_loss"] for row in first),
            "trajectory_estimator": stats.median(
                row["trajectory_estimator_loss"] for row in last
            )
            / stats.median(row["trajectory_estimator_loss"] for row in first),
            "bridge": stats.median(row["bridge_loss"] for row in last)
            / stats.median(row["bridge_loss"] for row in first),
            "full_trajectory": stats.median(
                row["full_trajectory_loss"] for row in last
            )
            / stats.median(row["full_trajectory_loss"] for row in first),
            "total": stats.median(row["loss"] for row in last)
            / stats.median(row["loss"] for row in first),
        },
    }

    # Downsample the noisy training series into non-overlapping 50-update bins.
    training_series: list[dict] = []
    for start in range(0, len(train_rows), 50):
        rows = train_rows[start : start + 50]
        training_series.append(
            {
                "update": rows[-1]["update"],
                "clean_loss_median": stats.median(row["clean_loss"] for row in rows),
                "trajectory_loss_median": stats.median(
                    row["trajectory_estimator_loss"] for row in rows
                ),
                "bridge_loss_median": stats.median(row["bridge_loss"] for row in rows),
                "total_loss_median": stats.median(row["loss"] for row in rows),
                "learning_rate": stats.fmean(row["learning_rate"] for row in rows),
            }
        )

    recomputed = {
        "baseline_next_state_mse": stats.fmean(
            row["baseline_next_state_mse"] for row in g450
        ),
        "pilot_next_state_mse": stats.fmean(row["pilot_next_state_mse"] for row in g450),
        "next_state_mse_reduction": mse_reduction_stat(g450),
        "baseline_k1_psnr": stats.fmean(row["baseline_k1_psnr"] for row in g450),
        "pilot_k1_psnr": stats.fmean(row["pilot_k1_psnr"] for row in g450),
        "k1_psnr_gain_db": k1_gain_stat(g450),
        "baseline_k0_psnr": stats.fmean(row["baseline_k0_psnr"] for row in eval_rows),
        "pilot_k0_psnr": stats.fmean(row["pilot_k0_psnr"] for row in eval_rows),
        "k0_psnr_delta_db": overall_k0_stat(eval_rows),
    }
    report_reconciliation = {
        key: {
            "reported": report["metrics"][key],
            "recomputed": value,
            "absolute_difference": abs(report["metrics"][key] - value),
        }
        for key, value in recomputed.items()
    }

    local_hashes = {
        path.name: sha256(path)
        for path in sorted((ROOT / "source").iterdir())
        if path.is_file()
    }
    expected_hashes: dict[str, str] = {}
    for line in (DATA / "a1" / "source_snapshot.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            expected_hashes[Path(path).name] = digest
    hash_reconciliation = {
        name: {
            "expected": expected_hashes.get(name),
            "local": digest,
            "matches": expected_hashes.get(name) == digest,
        }
        for name, digest in local_hashes.items()
        if name in expected_hashes
    }

    result = {
        "analysis_version": "v02_a1_exploratory_20260830",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS},
        "data_quality": data_quality,
        "report_reconciliation": report_reconciliation,
        "hash_reconciliation": hash_reconciliation,
        "overall_k0": {
            "estimate_db": overall_k0_stat(eval_rows),
            "stratified_bootstrap_ci_db": percentile_interval(overall_k0_boot),
            "distribution": summary(
                [row["pilot_k0_psnr"] - row["baseline_k0_psnr"] for row in eval_rows]
            ),
        },
        "per_g_k0": per_g_details,
        "g450": g450_metrics,
        "training_dynamics": training_dynamics,
        "a0": {
            "selected": a0["selection"]["selected"],
            "medians": a0["medians"],
        },
        "aggregation_warning": {
            "reported_k0_population": "all 100 fast-val trajectories across five terminal-G buckets",
            "reported_k1_population": "20 G=4.50 trajectories only",
            "direct_k1_vs_k0_requires": "paired comparison on the same 20 G=4.50 trajectories",
        },
    }

    write_csv(OUT / "per_g_k0_summary.csv", per_g_table)
    write_csv(OUT / "g450_per_sample.csv", per_sample_table)
    write_csv(OUT / "training_windows.csv", training_window_table)
    write_csv(OUT / "training_series_50update.csv", training_series)
    (OUT / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

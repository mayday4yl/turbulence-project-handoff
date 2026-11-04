#!/usr/bin/env python3
"""Reconcile implementation deviations and mechanism controls from diagnostics."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "derived"
SEED = 2026082899
DRAWS = 10_000
G_VALUES = (1.50, 2.25, 3.00, 3.75, 4.50)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(folder: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(folder.glob("rank-*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return rows


def q(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap(
    rows: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float],
    *,
    seed_offset: int,
) -> list[float]:
    rng = random.Random(SEED + seed_offset)
    return [
        statistic([rows[rng.randrange(len(rows))] for _ in rows])
        for _ in range(DRAWS)
    ]


def stratified_bootstrap(
    groups: dict[float, list[dict]],
    statistic: Callable[[Sequence[dict]], float],
    *,
    seed_offset: int,
) -> list[float]:
    rng = random.Random(SEED + seed_offset)
    output: list[float] = []
    for _ in range(DRAWS):
        sample: list[dict] = []
        for severity in G_VALUES:
            rows = groups[severity]
            sample.extend(rows[rng.randrange(len(rows))] for _ in rows)
        output.append(statistic(sample))
    return output


def ci(values: Sequence[float]) -> list[float]:
    return [q(values, 0.025), q(values, 0.975)]


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = stats.fmean(left)
    right_mean = stats.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else math.nan


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
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


def scenario_metrics(rows: Sequence[dict], baseline: str, label: str) -> dict:
    by_key = {(row["checkpoint"], row["key"]): row for row in rows}
    pilot = [row for row in rows if row["checkpoint"] == "v02_3k_ema"]
    g450 = [row for row in pilot if float(row["G"]) == 4.50]

    def k0_stat(sample: Sequence[dict]) -> float:
        return stats.fmean(
            row["k0_psnr"] - by_key[(baseline, row["key"])]["k0_psnr"] for row in sample
        )

    def k1_stat(sample: Sequence[dict]) -> float:
        return stats.fmean(
            row["no_refine_k1_psnr"]
            - by_key[(baseline, row["key"])]["no_refine_k1_psnr"]
            for row in sample
        )

    def mse_reduction(sample: Sequence[dict]) -> float:
        pilot_mean = stats.fmean(row["no_refine_next_mse"] for row in sample)
        baseline_mean = stats.fmean(
            by_key[(baseline, row["key"])]["no_refine_next_mse"] for row in sample
        )
        return 1.0 - pilot_mean / baseline_mean

    groups = defaultdict(list)
    for row in pilot:
        groups[float(row["G"])].append(row)
    k0_boot = stratified_bootstrap(groups, k0_stat, seed_offset=100)
    k1_boot = bootstrap(g450, k1_stat, seed_offset=101)
    mse_boot = bootstrap(g450, mse_reduction, seed_offset=102)
    k0 = k0_stat(pilot)
    k1 = k1_stat(g450)
    mse = mse_reduction(g450)
    return {
        "scenario": label,
        "baseline_state": baseline,
        "k0_delta_db": k0,
        "k0_ci_low_db": ci(k0_boot)[0],
        "k0_ci_high_db": ci(k0_boot)[1],
        "k0_gate_pass": k0 >= -0.1,
        "k1_gain_db": k1,
        "k1_ci_low_db": ci(k1_boot)[0],
        "k1_ci_high_db": ci(k1_boot)[1],
        "k1_gate_pass": k1 >= 0.1,
        "next_state_mse_reduction": mse,
        "mse_ci_low": ci(mse_boot)[0],
        "mse_ci_high": ci(mse_boot)[1],
        "mse_gate_pass": mse >= 0.25,
        "overall_gate": "PASS" if k0 >= -0.1 and k1 >= 0.1 and mse >= 0.25 else "STOP",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual = read_rows(DATA / "diagnostic")
    correct = read_rows(DATA / "diagnostic_seed2026082897")
    actual_keys = {row["key"] for row in actual if row["checkpoint"] == "v01_raw"}
    correct_keys = {row["key"] for row in correct if row["checkpoint"] == "v01_raw"}
    scenarios = [
        scenario_metrics(actual, "v01_raw", "executed seed 2026082801 + raw baseline"),
        scenario_metrics(correct, "v01_raw", "protocol seed 2026082897 + raw baseline"),
        scenario_metrics(actual, "v01_ema", "executed seed 2026082801 + EMA baseline"),
        scenario_metrics(correct, "v01_ema", "protocol seed 2026082897 + EMA baseline"),
    ]

    by_key = {(row["checkpoint"], row["key"]): row for row in correct}
    pilot = [row for row in correct if row["checkpoint"] == "v02_3k_ema"]
    per_g: list[dict] = []
    mechanism: list[dict] = []
    correlations: dict[str, dict[str, float]] = {}
    for severity in G_VALUES:
        group = [row for row in pilot if float(row["G"]) == severity]
        base = [by_key[("v01_ema", row["key"])] for row in group]
        k0_delta = [row["k0_psnr"] - old["k0_psnr"] for row, old in zip(group, base)]
        k1_delta = [
            row["no_refine_k1_psnr"] - old["no_refine_k1_psnr"]
            for row, old in zip(group, base)
        ]
        pilot_gap = [row["no_refine_k1_psnr"] - row["k0_psnr"] for row in group]
        base_gap = [row["no_refine_k1_psnr"] - row["k0_psnr"] for row in base]
        teacher_gain = [
            row["teacher_forced_k1_psnr"] - row["k0_psnr"] for row in group
        ]
        rollout_penalty = [
            row["no_refine_k1_psnr"] - row["teacher_forced_k1_psnr"]
            for row in group
        ]
        next_mse = [row["no_refine_next_mse"] for row in group]
        base_mse = [row["no_refine_next_mse"] for row in base]
        true_clean_gain = [
            row["true_clean_predicted_xi_k1_psnr"] - row["k0_psnr"] for row in group
        ]
        predicted_clean_true_xi_gain = [
            row["predicted_clean_true_xi_k1_psnr"] - row["k0_psnr"] for row in group
        ]
        per_g.append(
            {
                "G": severity,
                "n": len(group),
                "k0_delta_vs_v01_ema_db": stats.fmean(k0_delta),
                "k1_delta_vs_v01_ema_db": stats.fmean(k1_delta),
                "next_state_mse_reduction_vs_v01_ema": 1.0
                - stats.fmean(next_mse) / stats.fmean(base_mse),
                "baseline_k1_minus_k0_db": stats.fmean(base_gap),
                "pilot_k1_minus_k0_db": stats.fmean(pilot_gap),
                "progressive_gap_closure_db": stats.fmean(
                    new - old for new, old in zip(pilot_gap, base_gap)
                ),
                "pilot_k1_win_vs_k0_count": sum(value > 0 for value in pilot_gap),
                "pilot_k0_win_vs_v01_ema_count": sum(value > 0 for value in k0_delta),
                "pilot_k1_win_vs_v01_ema_count": sum(value > 0 for value in k1_delta),
            }
        )
        mechanism.append(
            {
                "G": severity,
                "teacher_true_state_gain_db": stats.fmean(teacher_gain),
                "rollout_state_penalty_db": stats.fmean(rollout_penalty),
                "net_k1_minus_k0_db": stats.fmean(pilot_gap),
                "true_clean_predicted_xi_gain_db": stats.fmean(true_clean_gain),
                "predicted_clean_true_xi_gain_db": stats.fmean(predicted_clean_true_xi_gain),
                "identity_check_residual": stats.fmean(
                    teacher + penalty - net
                    for teacher, penalty, net in zip(teacher_gain, rollout_penalty, pilot_gap)
                ),
            }
        )
        correlations[f"{severity:.2f}"] = {
            "pearson_next_mse_vs_rollout_penalty": pearson(next_mse, rollout_penalty),
            "spearman_next_mse_vs_rollout_penalty": spearman(next_mse, rollout_penalty),
            "pearson_next_mse_vs_k1_minus_k0": pearson(next_mse, pilot_gap),
            "spearman_next_mse_vs_k1_minus_k0": spearman(next_mse, pilot_gap),
        }

    checkpoint_order = ("v01_ema", "v02_1k_ema", "v02_2k_ema", "v02_3k_ema")
    checkpoint_progression: list[dict] = []
    for label in checkpoint_order:
        group = [
            row
            for row in correct
            if row["checkpoint"] == label and float(row["G"]) == 4.50
        ]
        checkpoint_progression.append(
            {
                "checkpoint": label,
                "update": {"v01_ema": 0, "v02_1k_ema": 1000, "v02_2k_ema": 2000, "v02_3k_ema": 3000}[label],
                "k0_psnr": stats.fmean(row["k0_psnr"] for row in group),
                "k1_psnr": stats.fmean(row["no_refine_k1_psnr"] for row in group),
                "k1_minus_k0_db": stats.fmean(
                    row["no_refine_k1_psnr"] - row["k0_psnr"] for row in group
                ),
                "next_state_mse": stats.fmean(row["no_refine_next_mse"] for row in group),
                "teacher_forced_k1_psnr": stats.fmean(
                    row["teacher_forced_k1_psnr"] for row in group
                ),
                "rollout_state_penalty_db": stats.fmean(
                    row["no_refine_k1_psnr"] - row["teacher_forced_k1_psnr"]
                    for row in group
                ),
            }
        )

    batches = [
        json.loads(line)
        for line in (DATA / "batches.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    samples = [sample for batch in batches for sample in batch["samples"]]
    current_counts = Counter(float(sample["g"]) for sample in samples)
    terminal_state_counts = Counter(
        float(sample["G"]) for sample in samples if float(sample["g"]) == float(sample["G"])
    )
    sampling: list[dict] = []
    for severity in (0.75, *G_VALUES):
        count = current_counts[severity]
        share = count / len(samples)
        sampling.append(
            {
                "g": severity,
                "v02_a1_count": count,
                "v02_a1_share": share,
                "v01_sampler_expected_share": 1 / 6,
                "v02_to_v01_exposure_ratio": share / (1 / 6),
                "terminal_state_count": terminal_state_counts[severity],
            }
        )

    gradient = read_json(DATA / "gradient_cosines.json")
    gradient_rows: list[dict] = []
    for checkpoint, branches in gradient["medians"].items():
        for branch_name, pairs in branches.items():
            gradient_rows.append(
                {"checkpoint": checkpoint, "branch": branch_name, **pairs}
            )

    drift = read_json(DATA / "v02_a1_checkpoint_drift_20260830.json")
    drift_rows: list[dict] = []
    for state, groups in drift["relative_to_initial"].items():
        for group_name, values in groups.items():
            drift_rows.append({"state": state, "group": group_name, **values})

    output = {
        "analysis_version": "v02_a1_diagnostics_20260830",
        "validation_seed_audit": {
            "protocol_seed": 2026082897,
            "executed_seed": 2026082801,
            "executed_exact_match_to_2026082801": True,
            "executed_exact_match_to_2026082897": False,
            "unique_keys_each": 100,
            "overlap_keys": len(actual_keys & correct_keys),
            "overlap_rate": len(actual_keys & correct_keys) / 100,
        },
        "comparator_audit": {
            "executed_baseline_state": "v01_raw",
            "frozen_v01_inference_state": "v01_ema",
            "v01_raw_to_ema_parameter_relative_l2": drift["relative_to_initial"]["v01_ema"],
        },
        "gate_sensitivity": scenarios,
        "corrected_gate_status": scenarios[-1]["overall_gate"],
        "per_g": per_g,
        "mechanism_decomposition": mechanism,
        "correlations": correlations,
        "checkpoint_progression_g450": checkpoint_progression,
        "sampling_exposure": sampling,
        "gradient_cosine_medians": gradient["medians"],
        "checkpoint_drift_v02_3k_ema": drift["relative_to_initial"]["v02_3k_ema"],
        "limitations": [
            "Corrected-seed diagnostics are exploratory and do not replace the official A1 marker.",
            "Gradient-direction audit uses four frozen batches and is directional evidence, not a population estimate.",
            "Teacher-forced and controlled-render routes diagnose failure location; they are not proposed methods.",
        ],
    }
    write_csv(OUT / "gate_sensitivity.csv", scenarios)
    write_csv(OUT / "corrected_per_g.csv", per_g)
    write_csv(OUT / "mechanism_decomposition.csv", mechanism)
    write_csv(OUT / "checkpoint_progression_g450.csv", checkpoint_progression)
    write_csv(OUT / "sampling_exposure.csv", sampling)
    write_csv(OUT / "gradient_cosine_medians.csv", gradient_rows)
    write_csv(OUT / "checkpoint_drift_summary.csv", drift_rows)
    (OUT / "diagnostic_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

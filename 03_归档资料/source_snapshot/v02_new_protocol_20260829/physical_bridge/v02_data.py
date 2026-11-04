"""Frozen V0.2 manifest generation and indexed batch loading."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import BridgeArchive, load_jsonl
from .utils import atomic_write_bytes, atomic_write_json, sha256_file
from .v02_protocol import G_GRID, TERMINAL_GRID, next_weaker, trajectory_severities


def _derived_seed(base_seed: int, namespace: str) -> int:
    payload = f"{int(base_seed)}\x1f{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _bucketed_records(dataset_root: Path, split: str) -> dict[float, list[dict[str, Any]]]:
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(dataset_root / "manifests" / f"{split}.jsonl"):
        buckets[round(float(record["terminal_G"]), 2)].append(record)
    if set(buckets) != set(TERMINAL_GRID):
        raise ValueError(f"Unexpected terminal buckets for {split}: {sorted(buckets)}")
    return dict(buckets)


def select_balanced_subset(
    dataset_root: str | Path,
    *,
    split: str,
    per_bucket: int,
    seed: int,
) -> dict[float, list[dict[str, Any]]]:
    buckets = _bucketed_records(Path(dataset_root), split)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    selected: dict[float, list[dict[str, Any]]] = {}
    for terminal in TERMINAL_GRID:
        rows = sorted(buckets[terminal], key=lambda item: item["key"])
        if per_bucket > len(rows):
            raise ValueError(f"Requested {per_bucket} from {terminal}, only {len(rows)} exist")
        indices = generator.permutation(len(rows))[:per_bucket]
        selected[terminal] = [rows[int(index)] for index in indices]
    return selected


def _cycle_record(
    bucket: list[dict[str, Any]],
    state: dict[str, Any],
    generator: np.random.Generator,
) -> dict[str, Any]:
    if state["position"] >= len(state["order"]):
        state["order"] = generator.permutation(len(bucket)).tolist()
        state["position"] = 0
        state["cycle"] += 1
    index = int(state["order"][state["position"]])
    state["position"] += 1
    return bucket[index]


def build_training_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    stage: str,
    updates: int,
    effective_batch: int,
    subset_per_bucket: int,
    data_seed: int,
    g_seed: int,
    trajectory_seed: int,
    guarantee_transition: bool = False,
) -> dict[str, Any]:
    """Build balanced batches before any gradient or validation result is observed."""

    if effective_batch < len(TERMINAL_GRID):
        raise ValueError("Effective batch must cover all terminal buckets")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = select_balanced_subset(
        dataset_root,
        split="train",
        per_bucket=subset_per_bucket,
        seed=data_seed,
    )
    data_rng = np.random.Generator(np.random.PCG64(int(data_seed)))
    g_rng = np.random.Generator(np.random.PCG64(int(g_seed)))
    trajectory_rng = np.random.Generator(np.random.PCG64(int(trajectory_seed)))
    states = {
        terminal: {
            "order": data_rng.permutation(len(buckets[terminal])).tolist(),
            "position": 0,
            "cycle": 0,
        }
        for terminal in TERMINAL_GRID
    }
    base = effective_batch // len(TERMINAL_GRID)
    remainder = effective_batch % len(TERMINAL_GRID)
    if remainder != 1:
        raise ValueError("Frozen effective batch 16 must yield one rotating extra bucket")

    lines: list[bytes] = []
    guard_count = 0
    for update in range(int(updates)):
        labels = [terminal for terminal in TERMINAL_GRID for _ in range(base)]
        labels.append(TERMINAL_GRID[update % len(TERMINAL_GRID)])
        data_rng.shuffle(labels)
        samples: list[dict[str, Any]] = []
        for terminal in labels:
            record = _cycle_record(buckets[terminal], states[terminal], data_rng)
            eligible_g = [value for value in G_GRID if value <= terminal]
            g = float(eligible_g[int(g_rng.integers(0, len(eligible_g)))])
            h = next_weaker(g)
            severities = trajectory_severities(terminal, g, h)
            s_t = float(severities[int(trajectory_rng.integers(0, len(severities)))])
            samples.append(
                {
                    "key": record["key"],
                    "G": float(terminal),
                    "g": g,
                    "h": h,
                    "s_T": s_t,
                }
            )
        if guarantee_transition and not any(item["h"] is not None for item in samples):
            guard_count += 1
            item = samples[0]
            item["g"] = 1.50
            item["h"] = 0.75
            severities = trajectory_severities(item["G"], item["g"], item["h"])
            item["s_T"] = float(
                severities[int(trajectory_rng.integers(0, len(severities)))]
            )
            item["a0_transition_guard"] = True
        row = {"update": update, "samples": samples}
        lines.append((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())

    batches_path = output_dir / "batches.jsonl"
    atomic_write_bytes(batches_path, b"".join(lines))
    subset = {
        f"{terminal:.2f}": [record["key"] for record in buckets[terminal]]
        for terminal in TERMINAL_GRID
    }
    atomic_write_json(output_dir / "train_subset.json", subset)
    metadata = {
        "stage": stage,
        "dataset_root": str(Path(dataset_root).resolve()),
        "updates": int(updates),
        "effective_batch": int(effective_batch),
        "subset_per_terminal_bucket": int(subset_per_bucket),
        "terminal_grid": list(TERMINAL_GRID),
        "g_grid": list(G_GRID),
        "generator": "numpy.random.Generator(PCG64)",
        "seeds": {
            "data_order": int(data_seed),
            "g_h": int(g_seed),
            "trajectory_loss": int(trajectory_seed),
        },
        "transition_guard_batches": guard_count,
        "batches_sha256": sha256_file(batches_path),
    }
    atomic_write_json(output_dir / "manifest.json", metadata)
    return metadata


def build_a0_manifest(dataset_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    base = 2026082800
    return build_training_manifest(
        dataset_root,
        output_dir,
        stage="A0",
        updates=128,
        effective_batch=16,
        subset_per_bucket=1600,
        data_seed=_derived_seed(base, "data_order"),
        g_seed=_derived_seed(base, "g_h"),
        trajectory_seed=_derived_seed(base, "trajectory_loss"),
        guarantee_transition=True,
    )


def build_a1_manifests(dataset_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train = build_training_manifest(
        dataset_root,
        output_dir / "train",
        stage="A1",
        updates=3000,
        effective_batch=16,
        subset_per_bucket=400,
        data_seed=2026082801,
        g_seed=2026082802,
        trajectory_seed=2026082803,
    )
    validation = select_balanced_subset(
        dataset_root,
        split="val",
        per_bucket=20,
        seed=2026082801,
    )
    validation_keys = {
        f"{terminal:.2f}": [record["key"] for record in validation[terminal]]
        for terminal in TERMINAL_GRID
    }
    atomic_write_json(output_dir / "val_subset.json", validation_keys)
    summary = {
        "train": train,
        "validation_per_terminal_bucket": 20,
        "validation_trajectories": 100,
        "validation_subset_sha256": sha256_file(output_dir / "val_subset.json"),
        "model_optimizer_seed": 2026082805,
        "refinement_seed": 2026082804,
    }
    atomic_write_json(output_dir / "manifest.json", summary)
    return summary


def load_batch_manifest(path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def load_rank_batch(
    archive: BridgeArchive,
    batch_row: dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    samples = batch_row["samples"]
    if len(samples) % world_size:
        raise ValueError("Effective batch is not divisible by world size")
    local = len(samples) // world_size
    selected = samples[rank * local : (rank + 1) * local]
    loaded: list[dict[str, Any]] = []
    for item in selected:
        record = archive.by_key[item["key"]]
        cache: dict[float, torch.Tensor] = {}

        def state(severity: float) -> torch.Tensor:
            severity = round(float(severity), 2)
            if severity not in cache:
                cache[severity] = archive.state(record, severity)
            return cache[severity]

        h = item.get("h")
        full_severities = trajectory_severities(item["G"], item["g"], h)
        loaded.append(
            {
                **item,
                "image": state(item["g"]),
                "clean": archive.clean(record),
                "target_T": state(item["s_T"]),
                "target_h": state(h) if h is not None else None,
                "full_targets": {severity: state(severity) for severity in full_severities},
            }
        )
    return loaded


def stack_loaded(samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    valid = torch.tensor([item["h"] is not None for item in samples], device=device)
    h_values = [item["h"] if item["h"] is not None else item["g"] for item in samples]
    fallback = [item["target_h"] if item["target_h"] is not None else item["image"] for item in samples]
    return {
        "specs": samples,
        "image": torch.stack([item["image"] for item in samples]).to(device, non_blocking=True),
        "clean": torch.stack([item["clean"] for item in samples]).to(device, non_blocking=True),
        "target_T": torch.stack([item["target_T"] for item in samples]).to(device, non_blocking=True),
        "target_h": torch.stack(fallback).to(device, non_blocking=True),
        "g": torch.tensor([item["g"] for item in samples], device=device, dtype=torch.float32),
        "G": torch.tensor([item["G"] for item in samples], device=device, dtype=torch.float32),
        "h": torch.tensor(h_values, device=device, dtype=torch.float32),
        "s_T": torch.tensor([item["s_T"] for item in samples], device=device, dtype=torch.float32),
        "valid_h": valid,
    }

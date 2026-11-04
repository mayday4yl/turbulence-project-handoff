"""Deterministic stratified time plans; no accelerator or training side effects."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from data_pipeline.common import sha, write_json, write_rows
from .identity import SCHEMA, load_base

DEFAULT_SEED = 20261022
TIME_BINS = 10


def stable_seed(*parts):
    raw = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def sampled_time(seed, slot, bins=TIME_BINS):
    """Reject boundary hits *after* FP32 conversion and also require 0 < g < 4."""
    if not 0 <= slot < bins:
        raise ValueError("Invalid time stratum")
    rng = random.Random(seed)
    low, high = slot / bins, (slot + 1) / bins
    for _ in range(10000):
        t = np.float32(low + (high - low) * rng.random())
        g = np.float32(4.0) * (np.float32(1.0) - t)
        if low < float(t) < high and 0 < t < 1 and 0 < g < 4:
            return float(t), float(g)
    raise RuntimeError("Cannot draw an interior FP32 time")


def cache_directory(cache_root, split, cycle=0):
    if split == "train":
        if cycle < 0:
            raise ValueError("Negative cache cycle")
        return Path(cache_root) / "train" / f"cycle_{cycle:06d}"
    if split == "val":
        if cycle != 0:
            raise ValueError("Val-A is fixed: only cycle 0 exists")
        return Path(cache_root) / "val_A"
    raise ValueError("Alignment cache permits train/val only; Test remains locked")


def make_plan(base_samples, split, cycle=0, seed=DEFAULT_SEED):
    if split not in {"train", "val"} or cycle < 0 or (split == "val" and cycle != 0):
        raise ValueError("Invalid split/cycle")
    samples = [r for r in base_samples if r["split"] == split]
    result = []
    val_endpoint = {}
    if split == "val":
        by_source = collections.defaultdict(list)
        for sample in samples:
            by_source[sample["source_id"]].append(sample)
        for source_id, members in by_source.items():
            members = sorted(members, key=lambda x: x["realization_index"])
            if [r["realization_index"] for r in members] != [0, 1, 2, 3]:
                raise ValueError("Val-A requires four original realizations per source")
            selected = random.Random(stable_seed(seed, "val-endpoint", source_id)).randrange(4)
            val_endpoint[source_id] = members[selected]["sample_id"]
        internal_ids = [r["sample_id"] for r in samples if r["sample_id"] != val_endpoint[r["source_id"]]]
        shuffled = list(internal_ids)
        random.Random(stable_seed(seed, "val-time-strata")).shuffle(shuffled)
        val_slots = {sample_id: i % TIME_BINS for i, sample_id in enumerate(shuffled)}
    for base_index, row in enumerate(samples):
        slots = range(TIME_BINS) if split == "train" else [val_slots.get(row["sample_id"], -1)]
        for slot in slots:
            endpoint = slot == -1
            draw_seed = stable_seed(seed, split, cycle, row["sample_id"], slot)
            t, g = (0.0, 4.0) if endpoint else sampled_time(draw_seed, slot)
            key = f"{row['sample_id']}_c{cycle:06d}_t{slot:02d}" if split == "train" else row["sample_id"]
            result.append({"schema": SCHEMA, "sample_id": key, "base_sample_id": row["sample_id"],
                           "base_index": base_index, "source_id": row["source_id"],
                           "realization_id": row["realization_id"], "realization_index": row["realization_index"],
                           "split": split, "cache_cycle": cycle, "time_slot": slot,
                           "time_seed": draw_seed, "t": t, "strength": g, "G": 4.0,
                           "is_endpoint": endpoint, "seed_zernike": row["seed_zernike"],
                           "seed_tilt": row["seed_tilt"], "clean_ref": row["clean_ref"],
                           "clean_sha256": row["clean_sha256"], "base_observation_ref": row["observation_ref"],
                           "observation_ref": row["observation_ref"] if endpoint else f"observations/{key}.npy",
                           "observation_root": "base" if endpoint else "cache",
                           "record_ref": f"records/{key}.json"})
    return result


def prepare_plan(base_root, cache_root, split, cycle=0, seed=DEFAULT_SEED):
    base, sources, samples, _ = load_base(base_root)
    cache_root = Path(cache_root)
    identity = {"schema": SCHEMA, "base": base, "seed": seed, "time_bins": TIME_BINS,
                "train_endpoint_fraction": 0.25, "effective_batch": 48,
                "per_update_endpoint_count": 12, "per_update_internal_count": 36,
                "train_cache_cycle_updates": 2000, "train_cache_cycle_count": 3,
                "internal_items_consumed_per_cycle": 72000, "unused_internal_items_per_cycle": 8000,
                "time_rule": "one independent uniform draw inside each equal-width stratum; reject FP32 boundaries",
                "state_rule": "F(clean, original raw realization, g); no RGB mixing; g=4*(1-t)",
                "dtype": "float32", "clamp": False, "test_model_evaluation": False}
    write_json(cache_root / "CACHE_IDENTITY.json", identity)
    directory = cache_directory(cache_root, split, cycle)
    plan = make_plan(samples, split, cycle, seed)
    write_rows(directory / "PLAN.jsonl", plan)
    summary = {"schema": SCHEMA, "cache_identity_sha256": sha(cache_root / "CACHE_IDENTITY.json"),
               "plan_sha256": sha(directory / "PLAN.jsonl"), "split": split, "cache_cycle": cycle,
               "records": len(plan), "endpoints": sum(r["is_endpoint"] for r in plan),
               "internal": sum(not r["is_endpoint"] for r in plan),
               "time_bin_counts": dict(sorted(collections.Counter(r["time_slot"] for r in plan if not r["is_endpoint"]).items())),
               "sources": len({r["source_id"] for r in plan}), "base_samples": len({r["base_sample_id"] for r in plan}),
               "estimated_new_float32_bytes": sum(not r["is_endpoint"] for r in plan) * (3 * 256 * 256 * 4 + 128)}
    write_json(directory / "PLAN_IDENTITY.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--cycle", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args()
    print(json.dumps(prepare_plan(a.base_root, a.cache_root, a.split, a.cycle, a.seed), indent=2))


if __name__ == "__main__":
    main()

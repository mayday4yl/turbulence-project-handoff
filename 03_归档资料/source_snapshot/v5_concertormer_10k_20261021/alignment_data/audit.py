"""Audit immutable base data without constructing a restoration model or using NPU."""
import argparse
import collections
import hashlib
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data_pipeline.common import digest, sha, verify_array, write_json
from data_pipeline.source_preprocess import preprocess
from .identity import ROOT, SCHEMA, load_base, load_legacy_generator


def audit_data(base_root, output_root, *, verify_preprocessing=True):
    started = time.perf_counter()
    root, output = Path(base_root), Path(output_root)
    base, sources, samples, records = load_base(root)
    source_counts = collections.Counter(r["split"] for r in sources)
    sample_counts = collections.Counter(r["split"] for r in samples)
    if dict(source_counts) != base["source_counts"] or dict(sample_counts) != base["sample_counts"]:
        raise ValueError("Base source/sample counts differ")
    uniqueness = {}
    for field in ("source_id", "archive_member", "original_sha256", "clean_pixel_sha256"):
        uniqueness[field] = len({r[field] for r in sources})
        if uniqueness[field] != len(sources):
            raise ValueError(f"Duplicate/leaked source identity: {field}")
    category_counts, preprocessing_counts = {}, {}
    clean_stats, observation_stats = {}, {}
    for split in ("train", "val", "test"):
        members = [r for r in sources if r["split"] == split]
        category_counts[split] = dict(sorted(collections.Counter(r["category_name"] for r in members).items()))
        preprocessing_counts[split] = dict(collections.Counter(json.dumps({k: v for k, v in r["preprocessing"].items()
            if k not in {"pillow_version", "original_size_wh", "oriented_size_wh"}}, sort_keys=True) for r in members))
        clean_stats[split] = {"minimum": float("inf"), "maximum": float("-inf"), "channel_sum": np.zeros(3),
                              "channel_square_sum": np.zeros(3), "pixels_per_channel": 0}
        observation_stats[split] = {"minimum": float("inf"), "maximum": float("-inf"),
                                    "below_zero_pixels": 0, "above_one_pixels": 0, "elements": 0}
    categories = set(category_counts["train"])
    if any(set(counts) != categories for counts in category_counts.values()):
        raise ValueError("Train/Val/Test scene category sets differ")
    target = {"train": 40, "val": 5, "test": 5}
    category_balanced = all(len(v) == 50 and set(v.values()) == {target[k]} for k, v in category_counts.items())
    if not category_balanced:
        raise ValueError("Unexpected category split: expected 40/5/5 sources in each of 50 categories")
    checked_preprocessing = 0
    for source in sources:
        clean = verify_array(root / source["clean"]["ref"], source["clean"]["sha256"])
        if digest(clean.tobytes()) != source["clean_pixel_sha256"]:
            raise ValueError("Clean pixel hash changed")
        raw_path = root / source["original_relative_path"]
        if sha(raw_path) != source["original_sha256"]:
            raise ValueError("Original source bytes changed")
        if not (0.0 <= float(clean.min()) <= float(clean.max()) <= 1.0):
            raise ValueError("Clean image out of [0,1]")
        if verify_preprocessing:
            replay, provenance = preprocess(raw_path.read_bytes())
            if not np.array_equal(replay, clean):
                raise ValueError(f"Preprocessing replay differs: {source['source_id']}")
            if any(v != source["preprocessing"].get(k) for k, v in provenance.items() if k != "pillow_version"):
                raise ValueError("Preprocessing provenance changed")
            checked_preprocessing += 1
        stats = clean_stats[source["split"]]
        values = clean.astype(np.float64).reshape(3, -1)
        stats["minimum"] = min(stats["minimum"], float(clean.min()))
        stats["maximum"] = max(stats["maximum"], float(clean.max()))
        stats["channel_sum"] += values.sum(axis=1)
        stats["channel_square_sum"] += (values * values).sum(axis=1)
        stats["pixels_per_channel"] += values.shape[1]
    by_source = collections.defaultdict(list)
    raw_hashes = set()
    seeds = set()
    for row in samples:
        by_source[row["source_id"]].append(row)
        if row["G"] != 4.0 or row["sample_id"] != row["realization_id"]:
            raise ValueError("Base realization/G rule changed")
        record = records[row["sample_id"]]
        disk_record = json.loads((root / row["record_ref"]).read_text())
        if disk_record != record:
            raise ValueError("Per-sample record differs from frozen sample manifest")
        y = verify_array(root / row["observation_ref"], record["observation_sha256"])
        if digest(y.tobytes()) != record["observation_pixel_sha256"]:
            raise ValueError("Y4 pixel hash changed")
        if record["generation_identity_sha256"] != base["files_sha256"]["GENERATION_IDENTITY.json"]:
            raise ValueError("Observation simulator generation identity differs")
        for key, value in record["raw_sha256"].items():
            if value in raw_hashes:
                raise ValueError("Duplicate raw realization field")
            raw_hashes.add(value)
        for key in ("seed_zernike", "seed_tilt"):
            if row[key] in seeds:
                raise ValueError("Repeated raw RNG seed")
            seeds.add(row[key])
        stats = observation_stats[row["split"]]
        stats["minimum"] = min(stats["minimum"], float(y.min()))
        stats["maximum"] = max(stats["maximum"], float(y.max()))
        stats["below_zero_pixels"] += int((y < 0).sum())
        stats["above_one_pixels"] += int((y > 1).sum())
        stats["elements"] += y.size
    if any(sorted(r["realization_index"] for r in members) != [0, 1, 2, 3] for members in by_source.values()):
        raise ValueError("Expected four independent realizations for every clean source")
    for stats in clean_stats.values():
        mean = stats.pop("channel_sum") / stats["pixels_per_channel"]
        variance = stats.pop("channel_square_sum") / stats["pixels_per_channel"] - mean * mean
        stats["channel_mean"] = mean.tolist()
        stats["channel_std"] = np.sqrt(np.maximum(variance, 0)).tolist()
    # Read-only perceptual duplicate screen, not a claim of exhaustive semantic deduplication.
    near = []
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            if a["split"] == b["split"]:
                continue
            d = (int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count()
            if d <= 3:
                p = (int(a["phash64"], 16) ^ int(b["phash64"], 16)).bit_count()
                if p <= 3:
                    near.append({"source_a": a["source_id"], "source_b": b["source_id"], "dhash_hamming": d, "phash_hamming": p})
    if near:
        raise ValueError(f"Potential cross-split near duplicates require review: {near[:5]}")
    result = {"schema": SCHEMA, "status": "PASS", "base_identity": base,
              "source_counts": dict(source_counts), "sample_counts": dict(sample_counts),
              "source_uniqueness": uniqueness, "source_disjoint_exact": True,
              "category_counts": category_counts, "category_distributions_identical": True,
              "source_origin": "All splits use disjoint Places365 val_256 images; fixed 50 classes; same-category unseen-image generalization",
              "content_limitation": "Exact duplicate and dHash/phash screens are passed; category balance does not prove identical image content distributions",
              "cross_split_near_duplicate_candidates": near,
              "preprocessing_counts": preprocessing_counts, "preprocessing_exact_replays": checked_preprocessing,
              "clean_statistics": clean_stats, "observation_statistics": observation_stats,
              "verified_rgb_arrays": len(sources) + len(samples), "unique_raw_hashes": len(raw_hashes),
              "unique_raw_seeds": len(seeds), "raw_hash_replay_status": "SEPARATE_RAW_REPLAY_REPORT",
              "test_used_for_model_evaluation": False, "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "DATA_AUDIT.json", result, frozen=False)
    return result


_RAW_HELPER = None


def _initialize_raw_worker():
    import torch
    global _RAW_HELPER
    torch.set_num_threads(1)
    _RAW_HELPER = load_legacy_generator()


def _replay_raw_one(item):
    row, expected = item
    raw, actual = _RAW_HELPER.raw_noise(row)
    if actual != expected:
        raise ValueError(f"Raw realization hash mismatch: {row['sample_id']}")
    del raw
    return row["sample_id"], row["split"], actual


def replay_raw(base_root, output_root, *, splits=("train", "val", "test"), workers=1):
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    root, output = Path(base_root), Path(output_root)
    base, _, samples, records = load_base(root)
    if str(torch.__version__) != base["raw_torch_version"]:
        raise ValueError(f"Raw replay must use frozen torch {base['raw_torch_version']}; got {torch.__version__}")
    if not 1 <= workers <= 8:
        raise ValueError("CPU raw replay is bounded to 1--8 workers")
    generation = json.loads((root / "GENERATION_IDENTITY.json").read_text())
    if sha(ROOT / "data_pipeline/generate.py") != generation["generator_file_sha256"]:
        raise ValueError("Original raw RNG implementation changed")
    if sha(ROOT / "data_pipeline/common.py") != generation["raw_rng_helper_file_sha256"]:
        raise ValueError("Original RNG hashing helper changed")
    count = collections.Counter()
    rolling = hashlib.sha256()
    selected = [r for r in samples if r["split"] in splits]
    work = [(row, records[row["sample_id"]]["raw_sha256"]) for row in selected]
    if workers == 1:
        _initialize_raw_worker()
        iterator = map(_replay_raw_one, work)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_initialize_raw_worker)
        iterator = pool.map(_replay_raw_one, work, chunksize=5)
    try:
        for sample_id, split, hashes in iterator:
            count[split] += 1
            rolling.update(json.dumps([sample_id, hashes], sort_keys=True).encode())
            if sum(count.values()) % 500 == 0:
                print(json.dumps({"phase": "raw_replay", "verified": sum(count.values()), "workers": workers,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    result = {"schema": SCHEMA, "status": "PASS", "base_identity": base, "torch": str(torch.__version__),
              "counts": dict(count), "verified_raw_fields": 2 * sum(count.values()),
              "ordered_replay_result_sha256": rolling.hexdigest(), "all_hashes_exact": True, "cpu_workers": workers,
              "test_used_for_model_evaluation": False, "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "RAW_REPLAY.json", result, frozen=False)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--mode", choices=("data", "raw", "all"), default="all")
    p.add_argument("--raw-workers", type=int, default=1)
    a = p.parse_args()
    if a.mode in {"data", "all"}:
        value = audit_data(a.base_root, a.output_root)
        print(json.dumps({"status": value["status"], "phase": "data_audit", "seconds": value["elapsed_seconds"]}), flush=True)
    if a.mode in {"raw", "all"}:
        value = replay_raw(a.base_root, a.output_root, workers=a.raw_workers)
        print(json.dumps({"status": value["status"], "phase": "raw_replay", "seconds": value["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

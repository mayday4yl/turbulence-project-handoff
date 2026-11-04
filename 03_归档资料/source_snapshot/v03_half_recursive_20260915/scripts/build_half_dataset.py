#!/usr/bin/env python3
"""Build and fully audit a source-level half subset of the existing V0.1 data.

No simulator, model, NPU, or test metric is used. Original archives are opened
read-only. The published manifests differ only in their absolute shard paths.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

SEED = 2026091504
TARGETS = {"train": 1000, "val": 100, "test": 250}
REPEATS = {"train": 4, "val": 2, "test": 2}
G_GRID = (.75, 1.5, 2.25, 3., 3.75, 4.5)
STATE_SUFFIXES = tuple(f"x_g{int(g * 100):03d}.npy" for g in G_GRID)
SUFFIXES = ("clean.png", *STATE_SUFFIXES, "epsilon_zernike.npy", "epsilon_tilt.npy", "json")
SOURCE_KEYS = ("source_image_id", "source_category_id", "source_member", "source_split",
               "source_bytes", "source_index")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode())


def object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(value):
    def nonfinite(token):
        raise ValueError(f"Non-finite JSON numeric token: {token}")
    result = json.loads(value, object_pairs_hook=object_no_duplicates, parse_constant=nonfinite)
    def check_numbers(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Non-finite JSON numeric value")
        if isinstance(item, dict):
            for child in item.values():
                check_numbers(child)
        elif isinstance(item, list):
            for child in item:
                check_numbers(child)
    check_numbers(result)
    return result


def load_jsonl(path):
    return [parse_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def order_hash(seed, split, kind, value):
    # This exact serialization is frozen in run_manifest.json.
    return sha_bytes(json.dumps([seed, split, kind, value], separators=(",", ":")).encode())


def select_sources(sources, target, split, seed=SEED, categories_reliable=True):
    if len(sources) != 2 * target or len({s["source_image_id"] for s in sources}) != len(sources):
        raise ValueError(f"{split}: expected exactly {2 * target} unique original sources")
    if not categories_reliable:
        selected = sorted(sources, key=lambda s: (order_hash(seed, split, "source", s["source_image_id"]),
                                                   s["source_image_id"]))[:target]
        return selected, {"strategy": "uniform-hash-order", "category_quotas": None}
    buckets = defaultdict(list)
    for source in sources:
        buckets[source["source_category_id"]].append(source)
    quotas = {category: len(rows) // 2 for category, rows in buckets.items()}
    remaining = target - sum(quotas.values())
    odd = [category for category, rows in buckets.items() if len(rows) % 2]
    odd.sort(key=lambda c: (order_hash(seed, split, "category", c), c))
    if not 0 <= remaining <= len(odd):
        raise AssertionError("Invalid half-subset remainder allocation")
    for category in odd[:remaining]:
        quotas[category] += 1
    chosen = []
    for category in sorted(buckets):
        ordered = sorted(buckets[category], key=lambda s: (
            order_hash(seed, split, "source", s["source_image_id"]), s["source_image_id"]))
        chosen.extend(ordered[:quotas[category]])
    assert len(chosen) == target
    return chosen, {"strategy": "stratified-largest-remainder-half",
                    "category_quotas": {str(c): quotas[c] for c in sorted(quotas)},
                    "odd_categories_awarded_extra": odd[:remaining]}


def inspect_sources(rows_by_split, metadata_by_split):
    """Reject source/split/trajectory corruption; category uncertainty falls back."""
    seen_sources, seen_members, seen_keys = {}, {}, set()
    result = {}
    for split, rows in rows_by_split.items():
        groups = defaultdict(list)
        for row in rows:
            source = row["source_image_id"]
            if row.get("source_split") != split or row.get("split", split) != split:
                raise ValueError(f"Wrong source/split identity: {row.get('key')}")
            if not isinstance(source, str) or not source or not row.get("source_member"):
                raise ValueError("Missing source image/member identity")
            if row["key"] in seen_keys:
                raise ValueError(f"Duplicate global trajectory key {row['key']}")
            seen_keys.add(row["key"])
            if source in seen_sources and seen_sources[source] != split:
                raise ValueError(f"Source leakage: {source}")
            member = row["source_member"]
            if member in seen_members and seen_members[member] != (split, source):
                raise ValueError(f"Source member reused under different source/split: {member}")
            seen_sources[source], seen_members[member] = split, (split, source)
            groups[source].append(row)
        if len(groups) != TARGETS[split] * 2:
            raise ValueError(f"Unexpected original source count for {split}: {len(groups)}")
        metadata = metadata_by_split[split]
        metadata_index = {row["source_image_id"]: row for row in metadata}
        if len(metadata_index) != len(metadata) or set(metadata_index) != set(groups):
            raise ValueError(f"{split} source metadata membership is inconsistent")
        issues, sources = [], []
        for source, trajectories in groups.items():
            if len(trajectories) != REPEATS[split]:
                raise ValueError(f"{source}: incomplete realization count")
            if sorted(row["realization_index"] for row in trajectories) != list(range(REPEATS[split])):
                raise ValueError(f"{source}: duplicated/missing realization indices")
            first = trajectories[0]
            meta = metadata_index[source]
            for field in ("source_member", "source_split"):
                if any(row.get(field) != first.get(field) for row in trajectories) or meta.get(field) != first.get(field):
                    raise ValueError(f"Inconsistent source identity {source}/{field}")
            categories = [row.get("source_category_id") for row in trajectories] + [meta.get("source_category_id")]
            valid = all(isinstance(category, int) and not isinstance(category, bool) and 0 <= category < 365
                        for category in categories)
            if not valid or len({str(category) for category in categories}) != 1:
                issues.append({"source_image_id": source, "reason": "category missing, invalid, or inconsistent",
                               "observed_categories": categories})
            sources.append(dict(meta))
        result[split] = {"sources": sources, "groups": groups, "categories_reliable": not issues,
                         "category_issues": issues}
    return result


def file_identity(path, include_hash=False):
    path = Path(path)
    stat = path.stat()
    value = {"path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
             "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
             "mode": stat.st_mode, "uid": stat.st_uid, "gid": stat.st_gid,
             "atime_ns_observed_only": stat.st_atime_ns}
    if include_hash:
        value["sha256"] = sha_file(path)
    return value


def identity_equal(before, after):
    # Reading may change atime; it is not evidence of original data mutation.
    return {k: v for k, v in before.items() if k != "atime_ns_observed_only"} == {
        k: v for k, v in after.items() if k != "atime_ns_observed_only"}


def read_member(handle, shard_bytes, index, name):
    if name not in index["members"]:
        raise KeyError(f"Missing indexed member {name}")
    location = index["members"][name]
    offset, size = location["offset"], location["size"]
    if (not isinstance(offset, int) or not isinstance(size, int) or offset < 512
            or offset % 512 or size <= 0 or offset + size > shard_bytes):
        raise ValueError(f"Invalid indexed bounds {name}: {location}")
    handle.seek(offset - 512)
    header_bytes = handle.read(512)
    header = tarfile.TarInfo.frombuf(header_bytes, encoding="utf-8", errors="strict")
    if not header.isfile() or header.name != name or header.size != size:
        raise ValueError(f"Index/header mismatch for {name}: {header.name}/{header.size}")
    # offset-512 addresses the regular file header, even when PAX headers precede it.
    value = handle.read(size)
    if len(value) != size:
        raise IOError(f"Short member read: {name}")
    return value, {"offset": offset, "header_offset": offset - 512, "size": size,
                   "sha256": sha_bytes(value), "tar_header_checksum_valid": True}


def audit_record(row, shard, index, deadline):
    import numpy as np
    from PIL import Image
    results = {}
    with Path(shard).open("rb") as handle:
        for suffix in SUFFIXES:
            if time.monotonic() > deadline:
                raise TimeoutError("Dataset audit exceeded its predeclared wall budget")
            payload, report = read_member(handle, index["bytes"], index, row["key"] + "." + suffix)
            if suffix != "json":
                expected = row["member_checksums"].get(suffix)
                if not isinstance(expected, str) or len(expected) != 64 or report["sha256"] != expected:
                    raise ValueError(f"Member SHA mismatch: {row['key']}.{suffix}")
                report["original_sha256_verified"] = True
            if suffix == "clean.png":
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (256, 256):
                        raise ValueError(f"Unexpected clean image shape/mode: {row['key']}")
                    array = np.asarray(image)
                    if array.shape != (256, 256, 3) or not np.isfinite(array).all():
                        raise ValueError(f"Invalid clean pixels: {row['key']}")
                    report.update(shape=list(array.shape), dtype=str(array.dtype), finite=True)
            elif suffix.endswith(".npy"):
                stream = io.BytesIO(payload)
                array = np.load(stream, allow_pickle=False)
                expected_shape = ((36, 512, 512) if suffix == "epsilon_zernike.npy" else
                                  (512, 512, 2) if suffix == "epsilon_tilt.npy" else (3, 256, 256))
                expected_dtype = np.dtype("float16" if suffix in STATE_SUFFIXES else "float32")
                if array.shape != expected_shape or array.dtype != expected_dtype or not np.isfinite(array).all():
                    raise ValueError(f"Invalid shape/dtype/finite array: {row['key']}.{suffix}")
                if stream.tell() != len(payload):
                    raise ValueError(f"Trailing bytes after array: {row['key']}.{suffix}")
                report.update(shape=list(array.shape), dtype=str(array.dtype), finite=True)
            else:
                metadata = parse_json(payload)
                for field in ("key", *SOURCE_KEYS, "realization_index"):
                    if metadata.get(field) != row.get(field):
                        raise ValueError(f"Embedded JSON identity mismatch: {row['key']}/{field}")
                if tuple(metadata.get("g_grid", ())) != G_GRID:
                    raise ValueError(f"Missing six G states in metadata: {row['key']}")
                if metadata.get("checksums") != row["member_checksums"]:
                    raise ValueError(f"Embedded JSON checksum inventory mismatch: {row['key']}")
                report.update(decoded=True, finite_json_numbers=True, original_sha256_verified=False)
            results[suffix] = report
            del payload
    return {"key": row["key"], "source_image_id": row["source_image_id"],
            "source_split": row["source_split"], "shard": str(shard), "members": results,
            "read_bytes": sum(item["size"] + 512 for item in results.values()), "passed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Predeclared source_root, output_root, subset_seed, workers, maximum_wall_seconds")
    args = parser.parse_args(argv)
    config = parse_json(args.config.read_text())
    if config.get("subset_seed") != SEED:
        raise ValueError(f"Subset seed is frozen at {SEED}")
    workers = int(config["workers"])
    seconds = float(config["maximum_wall_seconds"])
    if workers not in (2, 3, 4) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Require 2--4 CPU workers and a positive finite wall budget")
    original = Path(config["source_root"]).resolve()
    output = Path(config["output_root"]).resolve()
    if output == original or output.is_relative_to(original):
        raise ValueError("Output must be an independent directory outside original data")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + seconds
    run_manifest = {"config": config, "config_sha256": canonical_hash(config),
                    "config_file_sha256": sha_file(args.config), "script_sha256": sha_file(__file__),
                    "started_unix": time.time(), "targets": TARGETS, "realizations_per_source": REPEATS,
                    "hash_order_serialization": "JSON compact [seed,split,kind,value], SHA256 hexadecimal ascending",
                    "data_access": "CPU-only indexed read and decode; no model, simulator, NPU, or test metrics",
                    "budget": {"workers": workers, "maximum_wall_seconds": seconds}}
    atomic_json(output / "run_manifest.json", run_manifest)
    before, after = {}, {}
    try:
        files = [original / "manifests" / f"{prefix}{split}.jsonl"
                 for split in TARGETS for prefix in ("", "source_")]
        for path in files:
            before[str(path)] = file_identity(path, include_hash=True)
        rows = {split: load_jsonl(original / "manifests" / f"{split}.jsonl") for split in TARGETS}
        metadata = {split: load_jsonl(original / "manifests" / f"source_{split}.jsonl") for split in TARGETS}
        source_info = inspect_sources(rows, metadata)
        selected_rows, selected_sources, selection = {}, {}, {}
        indexes, shard_paths = {}, {}
        for split in TARGETS:
            info = source_info[split]
            chosen, details = select_sources(info["sources"], TARGETS[split], split,
                                             categories_reliable=info["categories_reliable"])
            ids = {row["source_image_id"] for row in chosen}
            selected_sources[split] = [row for row in metadata[split] if row["source_image_id"] in ids]
            selected_rows[split] = [dict(row) for row in rows[split] if row["source_image_id"] in ids]
            if len(selected_rows[split]) != TARGETS[split] * REPEATS[split]:
                raise AssertionError("Incomplete source-level trajectory selection")
            before_categories = Counter(str(row.get("source_category_id")) for row in info["sources"])
            after_categories = Counter(str(row.get("source_category_id")) for row in chosen)
            selection[split] = {**details, "categories_reliable": info["categories_reliable"],
                                "category_issues": info["category_issues"],
                                "source_count_before": len(info["sources"]), "source_count_after": len(chosen),
                                "trajectory_count_after": len(selected_rows[split]),
                                "categories_before": dict(sorted(before_categories.items())),
                                "categories_after": dict(sorted(after_categories.items())),
                                "category_count_before": len(before_categories), "category_count_after": len(after_categories),
                                "all_original_categories_retained": set(before_categories) == set(after_categories)}
            for row in selected_rows[split]:
                shard = (original / row["shard"]).resolve()
                if not shard.is_relative_to(original) or shard.suffix != ".tar":
                    raise ValueError(f"Unexpected original shard reference: {shard}")
                row["shard"] = str(shard)
                shard_paths[str(shard)] = shard
        for name, shard in shard_paths.items():
            index_path = shard.with_suffix(".index.json")
            before[name] = file_identity(shard)
            before[str(index_path)] = file_identity(index_path, include_hash=True)
            index = parse_json(index_path.read_text())
            if index["bytes"] != before[name]["size"] or index["member_count"] != len(index["members"]):
                raise ValueError(f"Index size/member inventory mismatch: {shard}")
            indexes[name] = index
        flat = [row for split in TARGETS for row in selected_rows[split]]
        flat.sort(key=lambda row: (row["shard"], indexes[row["shard"]]["members"][row["key"] + ".clean.png"]["offset"]))
        expected_read_bytes = sum(indexes[row["shard"]]["members"][row["key"] + "." + suffix]["size"] + 512
                                  for row in flat for suffix in SUFFIXES)
        run_manifest["budget"].update(expected_trajectories=len(flat), expected_members=len(flat) * 10,
                                      estimated_member_read_bytes=expected_read_bytes)
        atomic_json(output / "run_manifest.json", run_manifest)
        atomic_json(output / "selection_plan.json", selection)
        atomic_json(output / "original_files_before.json", before)
        for split in TARGETS:
            write_jsonl(output / "selection" / f"source_{split}.jsonl", selected_sources[split])
            write_jsonl(output / "selection" / f"trajectories_{split}.jsonl", [
                {"key": row["key"], "source_image_id": row["source_image_id"],
                 "realization_index": row["realization_index"]} for row in selected_rows[split]])
        completed = read_bytes = 0
        audited_by_split = Counter()
        pending, cursor = {}, iter(flat)
        with (output / "member_audit.jsonl").open("x") as audit_log, ThreadPoolExecutor(max_workers=workers) as executor:
            def refill():
                while len(pending) < workers * 2:
                    row = next(cursor, None)
                    if row is None:
                        break
                    future = executor.submit(audit_record, row, row["shard"], indexes[row["shard"]], deadline)
                    pending[future] = row["key"]
            refill()
            while pending:
                if time.monotonic() > deadline:
                    raise TimeoutError("Dataset audit exceeded predeclared wall budget")
                ready, _ = wait(pending, timeout=min(2, max(.01, deadline - time.monotonic())), return_when=FIRST_COMPLETED)
                for future in ready:
                    result = future.result()
                    pending.pop(future)
                    audit_log.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                    completed += 1
                    read_bytes += result["read_bytes"]
                    audited_by_split[result["source_split"]] += 1
                audit_log.flush()
                progress = {"status": "auditing", "completed_trajectories": completed,
                            "expected_trajectories": len(flat), "completed_members": completed * 10,
                            "read_bytes": read_bytes, "elapsed_seconds": time.monotonic() - started,
                            "audited_by_split": dict(audited_by_split)}
                atomic_json(output / "progress.json", progress)
                refill()
        if completed != len(flat):
            raise AssertionError("Audit did not visit all selected records")
        for name, identity in before.items():
            after[name] = file_identity(name, include_hash="sha256" in identity)
        atomic_json(output / "original_files_after.json", after)
        changed = [name for name in before if not identity_equal(before[name], after[name])]
        if changed:
            raise RuntimeError(f"Original file identity changed during read-only audit: {changed}")
        # Publish usable manifests only after every selected member passed audit.
        for split in TARGETS:
            write_jsonl(output / "manifests" / f"{split}.jsonl", selected_rows[split])
            write_jsonl(output / "manifests" / f"source_{split}.jsonl", selected_sources[split])
        summary = {"status": "completed", "subset_seed": SEED, "config_sha256": canonical_hash(config),
                   "source_root": str(original), "output_root": str(output), "selection": selection,
                   "all_selected_members_audited": completed * 10,
                   "existing_member_sha256_verified": completed * 9,
                   "json_members_decoded_and_new_sha256_recorded": completed,
                   "read_bytes": read_bytes, "elapsed_seconds": time.monotonic() - started,
                   "source_split_overlap": 0, "trajectory_key_duplicates": 0,
                   "original_file_identities_unchanged": True,
                   "readonly_evidence_limits": "Manifest/index SHA256 and tar stat unchanged; all selected data member SHA256 verified. Whole tar files were not rehashed; atime may change on reads.",
                   "test_access": "Selected held-out members were read only for integrity; no model or quality metric evaluated.",
                   "row_transform": "Only shard field replaced by original absolute archive path; all other fields preserved.",
                   "archives_copied_or_symlinked": False,
                   "category_caveat": "Half selection can drop singleton categories; val starts with 200 singleton categories and retains 100."}
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", summary)
        output_hashes = {str(path.relative_to(output)): sha_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
        atomic_json(output / "artifact_sha256.json", output_hashes)
        print(json.dumps(summary, sort_keys=True), flush=True)
        atomic_json(output / "DATASET_AUDIT_PASSED", {
            "status": "passed", "summary_file": "summary.json", "summary_sha256": sha_file(output / "summary.json"),
            "config_sha256": canonical_hash(config), "artifact_hashes_file": "artifact_sha256.json",
            "artifact_hashes_sha256": sha_file(output / "artifact_sha256.json"),
            "manifest_sha256": {f"{prefix}{split}.jsonl": sha_file(output / "manifests" / f"{prefix}{split}.jsonl")
                                for split in TARGETS for prefix in ("", "source_")}})
    except BaseException as exc:
        for name, identity in before.items():
            try:
                after[name] = file_identity(name, include_hash="sha256" in identity)
            except OSError as identity_error:
                after[name] = {"error": repr(identity_error)}
        atomic_json(output / "original_files_after_failure.json", after)
        failure = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                   "elapsed_seconds": time.monotonic() - started, "usable_manifests_published": False}
        atomic_json(output / "failure.json", failure)
        atomic_json(output / "progress.json", failure)
        raise


if __name__ == "__main__":
    main()

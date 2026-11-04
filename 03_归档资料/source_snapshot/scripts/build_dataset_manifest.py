#!/usr/bin/env python3
"""Freeze disjoint Places365 source splits and all trajectory identities."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from physical_bridge.data import TERMINAL_GRID
from physical_bridge.utils import atomic_write_bytes, atomic_write_json, derive_seed, sha256_file


SPLIT_ORDER = ("train", "val", "test")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )


def text_member(archive: tarfile.TarFile, pattern: str) -> str:
    candidates = [
        member
        for member in archive.getmembers()
        if member.isfile() and pattern.lower() in Path(member.name).name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern!r} in {archive.name}")
    exact = [member for member in candidates if Path(member.name).name.lower() == pattern]
    member = exact[0] if exact else sorted(candidates, key=lambda item: item.name)[0]
    extracted = archive.extractfile(member)
    if extracted is None:
        raise IOError(f"Could not extract {member.name}")
    return extracted.read().decode("utf-8")


def parse_filelist(filelist_archive: Path) -> tuple[dict[str, int], dict[int, str], str]:
    with tarfile.open(filelist_archive, "r:*") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        label_candidates = [
            name
            for name in names
            if "val" in Path(name).name.lower() and Path(name).suffix.lower() == ".txt"
        ]
        if not label_candidates:
            raise FileNotFoundError(f"No validation label list in {filelist_archive}")
        preferred = [name for name in label_candidates if "places365_val" in name.lower()]
        label_name = sorted(preferred or label_candidates)[0]
        label_file = archive.extractfile(label_name)
        if label_file is None:
            raise IOError(f"Could not extract {label_name}")
        label_text = label_file.read().decode("utf-8")
        category_names: dict[int, str] = {}
        category_candidates = [name for name in names if "categories_places365" in name]
        if category_candidates:
            category_file = archive.extractfile(sorted(category_candidates)[0])
            if category_file is not None:
                for line in category_file.read().decode("utf-8").splitlines():
                    fields = line.split()
                    if len(fields) >= 2:
                        category_names[int(fields[-1])] = " ".join(fields[:-1])

    labels: dict[str, int] = {}
    for line_number, line in enumerate(label_text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed validation label line {line_number}: {line!r}")
        labels[Path(fields[0]).name] = int(fields[-1])
    return labels, category_names, label_name


def source_members(image_archive: Path) -> dict[str, tuple[str, int]]:
    result = {}
    with tarfile.open(image_archive, "r:") as archive:
        for member in archive:
            if member.isfile() and Path(member.name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                basename = Path(member.name).name
                if basename in result:
                    raise ValueError(f"Duplicate source basename in archive: {basename}")
                result[basename] = (member.name, int(member.size))
    return result


def category_quotas(total: int, category_count: int, seed: int) -> dict[int, int]:
    base, extra = divmod(total, category_count)
    category_ids = list(range(category_count))
    random.Random(seed).shuffle(category_ids)
    quotas = {category: base for category in category_ids}
    for category in category_ids[:extra]:
        quotas[category] += 1
    return quotas


def freeze_sources(
    labels: dict[str, int],
    members: dict[str, tuple[str, int]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_category: dict[int, list[str]] = defaultdict(list)
    for basename, label in labels.items():
        if basename in members:
            by_category[label].append(basename)
    category_ids = sorted(by_category)
    if len(category_ids) != 365:
        raise ValueError(f"Expected 365 categories with images, found {len(category_ids)}")

    master_seed = int(config["master_seed"])
    for category, images in by_category.items():
        random.Random(derive_seed(master_seed, "source-order", category)).shuffle(images)

    quotas = {
        split: category_quotas(
            int(config["splits"][split]["source_images"]),
            len(category_ids),
            derive_seed(master_seed, "quota", split),
        )
        for split in SPLIT_ORDER
    }
    cursors = {category: 0 for category in category_ids}
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_ORDER:
        records = []
        for category in category_ids:
            count = quotas[split][category]
            start = cursors[category]
            selected = by_category[category][start : start + count]
            if len(selected) != count:
                raise ValueError(f"Category {category} lacks source images")
            cursors[category] += count
            for basename in selected:
                member_name, member_size = members[basename]
                records.append(
                    {
                        "source_image_id": basename,
                        "source_member": member_name,
                        "source_bytes": member_size,
                        "source_category_id": category,
                        "source_split": split,
                    }
                )
        random.Random(derive_seed(master_seed, "split-order", split)).shuffle(records)
        for index, record in enumerate(records):
            record["source_index"] = index
        expected = int(config["splits"][split]["source_images"])
        if len(records) != expected:
            raise AssertionError(f"{split}: expected {expected}, found {len(records)}")
        result[split] = records
    all_ids = [record["source_image_id"] for records in result.values() for record in records]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Source-image leakage across splits")
    return result


def freeze_trajectories(
    source_records: list[dict[str, Any]], split: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    master_seed = int(config["master_seed"])
    per_source = int(config["splits"][split]["realizations_per_source"])
    trajectory_count = len(source_records) * per_source
    terminals = [TERMINAL_GRID[index % len(TERMINAL_GRID)] for index in range(trajectory_count)]
    random.Random(derive_seed(master_seed, "terminal-order", split)).shuffle(terminals)
    records = []
    index = 0
    for source in source_records:
        for realization_index in range(per_source):
            key = f"{split}_{index:06d}"
            records.append(
                {
                    **source,
                    "key": key,
                    "trajectory_index": index,
                    "realization_index": realization_index,
                    "terminal_G": terminals[index],
                    "shard_index": index // int(config["shard_size"]),
                    "seeds": {
                        "preprocess": derive_seed(master_seed, split, index, "preprocess"),
                        "epsilon_zernike": derive_seed(master_seed, split, index, "zernike"),
                        "epsilon_tilt": derive_seed(master_seed, split, index, "tilt"),
                    },
                }
            )
            index += 1
    return records


def balanced_qualitative_keys(
    records: list[dict[str, Any]], count: int, master_seed: int, label: str
) -> list[str]:
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[float(record["terminal_G"])].append(record)
    for terminal, bucket in buckets.items():
        random.Random(derive_seed(master_seed, "qualitative", label, terminal)).shuffle(bucket)
    selected = []
    offset = 0
    while len(selected) < count:
        progressed = False
        for terminal in TERMINAL_GRID:
            if offset < len(buckets[terminal]) and len(selected) < count:
                selected.append(buckets[terminal][offset]["key"])
                progressed = True
        if not progressed:
            raise ValueError(f"Cannot select {count} qualitative keys for {label}")
        offset += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    image_archive = Path(config["source"]["archive_path"])
    filelist_archive = Path(config["source"]["filelist_archive_path"])
    labels, category_names, label_member = parse_filelist(filelist_archive)
    members = source_members(image_archive)
    sources = freeze_sources(labels, members, config)

    planned: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_ORDER:
        planned[split] = freeze_trajectories(sources[split], split, config)
        atomic_write_bytes(manifest_root / f"source_{split}.jsonl", jsonl_bytes(sources[split]))
        atomic_write_bytes(manifest_root / f"planned_{split}.jsonl", jsonl_bytes(planned[split]))

    qualitative = {
        "generation": balanced_qualitative_keys(
            planned["train"],
            int(config["qualitative_ids"]["generation"]),
            int(config["master_seed"]),
            "generation",
        ),
        "val": balanced_qualitative_keys(
            planned["val"],
            int(config["qualitative_ids"]["val"]),
            int(config["master_seed"]),
            "val",
        ),
        "test": balanced_qualitative_keys(
            planned["test"],
            int(config["qualitative_ids"]["test"]),
            int(config["master_seed"]),
            "test",
        ),
        "refinement_lr_val_g450": [
            record["key"]
            for record in sorted(
                (
                    item
                    for item in planned["val"]
                    if float(item["terminal_G"]) == 4.50
                ),
                key=lambda item: derive_seed(
                    int(config["master_seed"]), "refinement-lr", item["key"]
                ),
            )[:20]
        ],
    }
    atomic_write_json(manifest_root / "qualitative_ids.json", qualitative)
    split_hash_payload = {
        split: sha256_file(manifest_root / f"source_{split}.jsonl") for split in SPLIT_ORDER
    }
    summary = {
        "dataset_version": config["version"],
        "master_seed": int(config["master_seed"]),
        "source_archive": {
            "path": str(image_archive),
            "bytes": image_archive.stat().st_size,
            "sha256": sha256_file(image_archive),
        },
        "filelist_archive": {
            "path": str(filelist_archive),
            "bytes": filelist_archive.stat().st_size,
            "sha256": sha256_file(filelist_archive),
            "label_member": label_member,
        },
        "category_count": len(set(labels.values())),
        "category_names_available": len(category_names),
        "source_counts": {split: len(sources[split]) for split in SPLIT_ORDER},
        "trajectory_counts": {split: len(planned[split]) for split in SPLIT_ORDER},
        "source_split_hashes": split_hash_payload,
        "planned_manifest_hashes": {
            split: sha256_file(manifest_root / f"planned_{split}.jsonl")
            for split in SPLIT_ORDER
        },
        "cross_split_source_overlap": 0,
        "qualitative_ids": qualitative,
    }
    atomic_write_json(manifest_root / "split_summary.json", summary)
    atomic_write_bytes(manifest_root / "SPLIT_FROZEN", b"frozen\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

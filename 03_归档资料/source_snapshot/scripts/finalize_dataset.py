#!/usr/bin/env python3
"""Verify all generated shards, freeze global manifests, and build summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from physical_bridge.data import BridgeArchive, FROZEN_G_GRID, iter_jsonl
from physical_bridge.utils import atomic_touch, atomic_write_bytes, atomic_write_json, sha256_file


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )


def verify_split(
    output_root: Path,
    planned_manifest: Path,
    split: str,
    shard_size: int,
    verify_tar_hashes: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planned = list(iter_jsonl(planned_manifest))
    expected_shards = math.ceil(len(planned) / shard_size)
    generated = []
    shard_summaries = []
    for shard_index in range(expected_shards):
        shard_path = output_root / split / f"shard-{shard_index:06d}.tar"
        index_path = shard_path.with_suffix(".index.json")
        complete_path = shard_path.with_suffix(".complete")
        manifest_path = output_root / "manifests" / "shards" / (
            f"{split}-{shard_index:06d}.jsonl"
        )
        for required in (shard_path, index_path, complete_path, manifest_path):
            if not required.exists():
                raise FileNotFoundError(f"Incomplete dataset: missing {required}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if shard_path.stat().st_size != int(index["bytes"]):
            raise IOError(f"Size mismatch for {shard_path}")
        if verify_tar_hashes and sha256_file(shard_path) != index["sha256"]:
            raise IOError(f"SHA-256 mismatch for {shard_path}")
        shard_records = list(iter_jsonl(manifest_path))
        if len(shard_records) != int(index["sample_count"]):
            raise ValueError(f"Manifest count mismatch for {shard_path}")
        generated.extend(shard_records)
        shard_summaries.append(
            {
                "path": str(shard_path.relative_to(output_root)),
                "bytes": int(index["bytes"]),
                "samples": int(index["sample_count"]),
                "sha256": index["sha256"],
            }
        )
    generated.sort(key=lambda record: int(record["trajectory_index"]))
    planned_keys = [record["key"] for record in planned]
    generated_keys = [record["key"] for record in generated]
    if planned_keys != generated_keys:
        raise AssertionError(f"Generated keys do not exactly match the frozen plan for {split}")
    terminal_counts = Counter(f"{float(record['terminal_G']):.2f}" for record in generated)
    source_ids = {record["source_image_id"] for record in generated}
    categories = Counter(str(record["source_category_id"]) for record in generated)
    severity_values: dict[float, dict[str, list[float]]] = {
        g: defaultdict(list) for g in FROZEN_G_GRID
    }
    generation_seconds = []
    for record in generated:
        generation_seconds.append(float(record["generation_wall_seconds"]))
        for item in record["severity_statistics"]:
            target = severity_values[float(item["g"])]
            for name, value in item.items():
                if name != "g":
                    target[name].append(float(value))
    severity_summary = {}
    for severity, metrics in severity_values.items():
        severity_summary[f"{severity:.2f}"] = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in metrics.items()
        }
    summary = {
        "split": split,
        "trajectory_count": len(generated),
        "source_image_count": len(source_ids),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "category_trajectory_counts": dict(sorted(categories.items(), key=lambda item: int(item[0]))),
        "shard_count": len(shard_summaries),
        "bytes": sum(item["bytes"] for item in shard_summaries),
        "shards": shard_summaries,
        "generation_wall_seconds_per_trajectory": {
            "mean": float(np.mean(generation_seconds)),
            "median": float(np.median(generation_seconds)),
            "max": float(np.max(generation_seconds)),
        },
        "severity_summary": severity_summary,
    }
    return generated, summary


def tensor_to_image(tensor: Any) -> Image.Image:
    array = tensor.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def make_preview(archive: BridgeArchive, keys: list[str], output: Path) -> None:
    labels = ("Clean", "g=.75", "g=1.50", "g=2.25", "g=3.00", "g=3.75", "g=4.50")
    tile, label_height = 256, 24
    canvas = Image.new("RGB", (len(labels) * tile, len(keys) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, key in enumerate(keys):
        record = archive.by_key[key]
        images = [tensor_to_image(archive.clean(record))]
        images.extend(tensor_to_image(archive.state(record, g)) for g in FROZEN_G_GRID)
        top = row * (tile + label_height)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            left = column * tile
            canvas.paste(image, (left, top + label_height))
            draw.text((left + 4, top + 4), f"{key} {label}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.png")
    canvas.save(temporary, format="PNG", compress_level=3)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--planned-root", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--skip-tar-hash", action="store_true")
    parser.add_argument("--qualitative-ids", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["output_root"])
    planned_root = args.planned_root or output_root / "manifests"
    shard_size = args.shard_size or int(config["shard_size"])
    summaries = {}
    source_sets = {}
    for split in args.splits:
        records, summary = verify_split(
            output_root,
            planned_root / f"planned_{split}.jsonl",
            split,
            shard_size,
            not args.skip_tar_hash,
        )
        manifest = output_root / "manifests" / f"{split}.jsonl"
        atomic_write_bytes(manifest, jsonl_bytes(records))
        summary["manifest_sha256"] = sha256_file(manifest)
        summaries[split] = summary
        source_sets[split] = {record["source_image_id"] for record in records}
    for left_index, left in enumerate(args.splits):
        for right in args.splits[left_index + 1 :]:
            overlap = source_sets[left] & source_sets[right]
            if overlap:
                raise AssertionError(f"Source leakage between {left} and {right}: {len(overlap)}")
    result = {
        "dataset_version": config["version"],
        "g_grid": list(FROZEN_G_GRID),
        "splits": summaries,
        "total_trajectories": sum(item["trajectory_count"] for item in summaries.values()),
        "total_tar_bytes": sum(item["bytes"] for item in summaries.values()),
        "cross_split_source_overlap": 0,
        "tar_hashes_verified": not args.skip_tar_hash,
    }
    atomic_write_json(output_root / "generation_summary.json", result)

    qualitative_path = args.qualitative_ids or planned_root / "qualitative_ids.json"
    if qualitative_path.exists() and "train" in args.splits:
        qualitative = json.loads(qualitative_path.read_text(encoding="utf-8"))
        archive = BridgeArchive(output_root, "train")
        make_preview(
            archive,
            qualitative["generation"],
            output_root / "previews" / "fixed_16_trajectories.png",
        )
    atomic_touch(output_root / "DATASET_COMPLETE")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

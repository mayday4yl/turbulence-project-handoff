#!/usr/bin/env python3
"""Generate resumable, self-contained PiRN/P2S trajectory tar shards."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from physical_bridge.data import FROZEN_G_GRID, REQUIRED_SUFFIXES, iter_jsonl
from physical_bridge.utils import (
    atomic_touch,
    atomic_write_bytes,
    atomic_write_json,
    ensure_free_space,
    sha256_file,
)


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def preprocess_image(value: bytes, seed: int, size: int) -> tuple[np.ndarray, dict[str, Any]]:
    generator = random.Random(int(seed))
    with Image.open(io.BytesIO(value)) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        original_size = list(image.size)
        if min(image.size) != size:
            scale = size / min(image.size)
            resized = tuple(max(size, int(round(dimension * scale))) for dimension in image.size)
            image = image.resize(resized, Image.Resampling.BICUBIC)
        resized_size = list(image.size)
        max_left = image.width - size
        max_top = image.height - size
        left = generator.randint(0, max_left) if max_left else 0
        top = generator.randint(0, max_top) if max_top else 0
        image = image.crop((left, top, left + size, top + size))
        rotation = generator.choice((0, 90, 180, 270))
        if rotation:
            image = image.rotate(rotation, resample=Image.Resampling.NEAREST, expand=False)
        horizontal_flip = bool(generator.getrandbits(1))
        vertical_flip = bool(generator.getrandbits(1))
        if horizontal_flip:
            image = ImageOps.mirror(image)
        if vertical_flip:
            image = ImageOps.flip(image)
        array = np.asarray(image, dtype=np.uint8).copy()
    metadata = {
        "original_size_wh": original_size,
        "resized_size_wh": resized_size,
        "crop_xywh": [left, top, size, size],
        "rotation_degrees": rotation,
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "resize_interpolation": "bicubic",
    }
    return array, metadata


class SourceTar:
    def __init__(self, path: Path) -> None:
        self.archive = tarfile.open(path, "r:")
        self.members = {member.name: member for member in self.archive if member.isfile()}

    def read(self, name: str) -> bytes:
        extracted = self.archive.extractfile(self.members[name])
        if extracted is None:
            raise IOError(f"Could not extract source member {name}")
        return extracted.read()


def expand_realization(realization: Any, count: int) -> Any:
    from physical_bridge.simulator import Realization

    return Realization(
        realization.epsilon_zernike.expand(count, -1, -1, -1),
        realization.epsilon_tilt.expand(count, -1, -1, -1),
    )


def project_bytes(*roots: Path) -> int:
    total = 0

    def handle_walk_error(error: OSError) -> None:
        if isinstance(error, FileNotFoundError):
            return
        raise error

    for root in roots:
        if not root.exists():
            continue
        for directory, _, filenames in os.walk(root, onerror=handle_walk_error):
            for filename in filenames:
                try:
                    total += (Path(directory) / filename).stat().st_size
                except FileNotFoundError:
                    # Workers atomically publish files and remove completed spool
                    # trees. A vanished temporary entry is counted at its durable
                    # destination by this or the next periodic safety check.
                    continue
    return total


def completed_shard_outputs(
    output_root: Path, split: str, shard_index: int
) -> tuple[Path, Path, Path, Path]:
    shard_path = output_root / split / f"shard-{shard_index:06d}.tar"
    return (
        shard_path,
        shard_path.with_suffix(".index.json"),
        shard_path.with_suffix(".complete"),
        output_root / "manifests" / "shards" / f"{split}-{shard_index:06d}.jsonl",
    )


def check_storage(
    output_root: Path, temporary_root: Path, minimum_free: int, project_stop: int
) -> None:
    ensure_free_space(output_root, minimum_free)
    ensure_free_space(temporary_root, minimum_free)
    used = project_bytes(output_root, temporary_root)
    if used >= project_stop:
        raise RuntimeError(f"Safety stop: project uses {used} bytes, limit is {project_stop}")


def generate_record(
    *,
    record: dict[str, Any],
    record_dir: Path,
    source: SourceTar,
    simulator: Any,
    source_identity: dict[str, Any],
    image_size: int,
) -> dict[str, Any]:
    metadata_path = record_dir / "json"
    if metadata_path.exists() and all((record_dir / suffix).exists() for suffix in REQUIRED_SUFFIXES):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    record_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source_bytes = source.read(record["source_member"])
    clean_uint8, transform = preprocess_image(
        source_bytes, int(record["seeds"]["preprocess"]), image_size
    )
    clean = (
        torch.from_numpy(clean_uint8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(simulator.device_spec, dtype=torch.float32)
        .div_(255.0)
    )
    realization = simulator.sample_realization(
        1,
        seed_zernike=int(record["seeds"]["epsilon_zernike"]),
        seed_tilt=int(record["seeds"]["epsilon_tilt"]),
    )
    severity = torch.tensor(FROZEN_G_GRID, device=simulator.device_spec, dtype=torch.float32)
    with torch.no_grad():
        prepared = simulator.prepare_image(clean)
        rendered = simulator.render_prepared(
            prepared, expand_realization(realization, len(FROZEN_G_GRID)), severity
        )
        output = rendered.output.detach().float()
        if not bool(torch.isfinite(output).all().cpu()):
            raise FloatingPointError(f"Non-finite simulator output for {record['key']}")
        clean_expanded = clean.expand_as(output)
        mse = (output - clean_expanded).square().flatten(1).mean(1)
        l1 = (output - clean_expanded).abs().flatten(1).mean(1)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        displacement_magnitude = rendered.effective_state.displacement.square().sum(-1).sqrt()
        severity_stats = []
        for index, g_value in enumerate(FROZEN_G_GRID):
            current = output[index]
            severity_stats.append(
                {
                    "g": g_value,
                    "l1": float(l1[index].cpu()),
                    "psnr": float(psnr[index].cpu()),
                    "min": float(current.min().cpu()),
                    "max": float(current.max().cpu()),
                    "below_zero_fraction": float((current < 0).float().mean().cpu()),
                    "above_one_fraction": float((current > 1).float().mean().cpu()),
                    "zernike_rms": float(
                        rendered.effective_state.zernike[index].square().mean().sqrt().cpu()
                    ),
                    "tilt_component_rms": float(
                        rendered.effective_state.displacement[index]
                        .square()
                        .mean()
                        .sqrt()
                        .cpu()
                    ),
                    "warp_mean_magnitude": float(displacement_magnitude[index].mean().cpu()),
                    "warp_max_magnitude": float(displacement_magnitude[index].max().cpu()),
                    "blur_weight_rms": float(rendered.weights[index].square().mean().sqrt().cpu()),
                }
            )
        states = output.half().cpu().numpy()
    clean_value = png_bytes(clean_uint8)
    atomic_write_bytes(record_dir / "clean.png", clean_value)
    for index, suffix in enumerate(
        ("x_g075.npy", "x_g150.npy", "x_g225.npy", "x_g300.npy", "x_g375.npy", "x_g450.npy")
    ):
        atomic_numpy(record_dir / suffix, states[index])
    atomic_numpy(
        record_dir / "epsilon_zernike.npy",
        realization.epsilon_zernike[0].detach().float().cpu().numpy(),
    )
    atomic_numpy(
        record_dir / "epsilon_tilt.npy",
        realization.epsilon_tilt[0].detach().float().cpu().numpy(),
    )
    checksums = {
        suffix: sha256_file(record_dir / suffix)
        for suffix in REQUIRED_SUFFIXES
        if suffix != "json"
    }
    metadata = {
        **record,
        "crop_and_augmentation": transform,
        "all_random_seeds": record["seeds"],
        "g_grid": list(FROZEN_G_GRID),
        "simulator_protocol": simulator.protocol(),
        "simulator_source_hash": source_identity["simulator_source_hash"],
        "P2S_checkpoint_hash": source_identity["P2S_checkpoint_hash"],
        "tensor_shapes": {
            "clean": [3, image_size, image_size],
            "trajectory_state": [3, image_size, image_size],
            **simulator.realization_shapes,
        },
        "tensor_dtypes": {
            "clean": "uint8",
            "trajectory_state": "float16",
            "epsilon_zernike": "float32",
            "epsilon_tilt": "float32",
        },
        "checksums": checksums,
        "severity_statistics": severity_stats,
        "generation_wall_seconds": time.monotonic() - start,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def pack_shard(
    *,
    split: str,
    shard_index: int,
    records: list[dict[str, Any]],
    spool: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    shard_path = split_root / f"shard-{shard_index:06d}.tar"
    index_path = shard_path.with_suffix(".index.json")
    complete_path = shard_path.with_suffix(".complete")
    if shard_path.exists() and index_path.exists() and complete_path.exists():
        return shard_path, json.loads(index_path.read_text(encoding="utf-8"))
    temporary = shard_path.with_suffix(".tar.tmp")
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        for record in records:
            record_dir = spool / record["key"]
            for suffix in REQUIRED_SUFFIXES:
                path = record_dir / suffix
                archive.add(path, arcname=f"{record['key']}.{suffix}", recursive=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    members = {}
    with tarfile.open(temporary, "r:") as archive:
        for member in archive:
            if member.isfile():
                members[member.name] = {"offset": int(member.offset_data), "size": int(member.size)}
    expected_members = len(records) * len(REQUIRED_SUFFIXES)
    if len(members) != expected_members:
        raise IOError(f"Shard {shard_index}: expected {expected_members} members, got {len(members)}")
    shard_hash = sha256_file(temporary)
    index = {
        "split": split,
        "shard_index": shard_index,
        "sample_count": len(records),
        "member_count": len(members),
        "bytes": temporary.stat().st_size,
        "sha256": shard_hash,
        "members": members,
    }
    temporary.replace(shard_path)
    atomic_write_json(index_path, index)
    atomic_touch(complete_path)
    return shard_path, index


def shard_manifest_records(
    records: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], shard_rel: str
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        item = metadata[record["key"]]
        result.append(
            {
                **record,
                "shard": shard_rel,
                "member_checksums": item["checksums"],
                "severity_statistics": item["severity_statistics"],
                "generation_wall_seconds": item["generation_wall_seconds"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--simulator-source", type=Path, required=True)
    parser.add_argument("--p2s-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    from physical_bridge.simulator import PiRNP2SSimulator

    dataset_config = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    simulator_config = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    records = list(iter_jsonl(args.manifest))
    output_root = args.output_root or Path(dataset_config["output_root"])
    temporary_root = args.temporary_root or Path(dataset_config["temporary_root"])
    shard_size = args.shard_size or int(dataset_config["shard_size"])
    minimum_free = int(dataset_config["storage"]["minimum_free_bytes"])
    project_stop = int(dataset_config["storage"]["project_stop_bytes"])
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    check_storage(output_root, temporary_root, minimum_free, project_stop)

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    simulator = PiRNP2SSimulator(
        simulator_config["asset_dir"],
        image_size=int(simulator_config["image_size"]),
        rf_range=int(simulator_config["rf_range"]),
        aperture_diameter=float(simulator_config["aperture_diameter"]),
        propagation_distance=float(simulator_config["propagation_distance"]),
        wavelength=float(simulator_config["wavelength"]),
        correlation=float(simulator_config["correlation"]),
        tilt_threshold=float(simulator_config["tilt_threshold"]),
        adjustment=float(simulator_config["adjustment"]),
        device=device,
    )
    source_identity = {
        "simulator_source_hash": sha256_file(args.simulator_source),
        "P2S_checkpoint_hash": sha256_file(args.p2s_checkpoint),
    }
    source = SourceTar(Path(dataset_config["source"]["archive_path"]))
    total_shards = math.ceil(len(records) / shard_size)
    worker_log = output_root / "manifests" / "workers" / f"{args.split}-rank{args.rank}.jsonl"
    worker_log.parent.mkdir(parents=True, exist_ok=True)

    for shard_index in range(args.rank, total_shards, args.world_size):
        start_index = shard_index * shard_size
        shard_records = records[start_index : start_index + shard_size]
        if not shard_records:
            continue
        durable_outputs = completed_shard_outputs(output_root, args.split, shard_index)
        if all(path.exists() for path in durable_outputs):
            print(
                json.dumps(
                    {
                        "event": "shard_resume_skip",
                        "split": args.split,
                        "rank": args.rank,
                        "shard_index": shard_index,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        spool = temporary_root / args.split / f"shard-{shard_index:06d}"
        metadata = {}
        shard_start = time.monotonic()
        for local_index, record in enumerate(shard_records):
            if local_index % 8 == 0:
                check_storage(output_root, temporary_root, minimum_free, project_stop)
            record_dir = spool / record["key"]
            metadata[record["key"]] = generate_record(
                record=record,
                record_dir=record_dir,
                source=source,
                simulator=simulator,
                source_identity=source_identity,
                image_size=int(simulator_config["image_size"]),
            )
        shard_path, shard_index_data = pack_shard(
            split=args.split,
            shard_index=shard_index,
            records=shard_records,
            spool=spool,
            output_root=output_root,
        )
        shard_rel = str(shard_path.relative_to(output_root))
        manifest_records = shard_manifest_records(shard_records, metadata, shard_rel)
        shard_manifest = output_root / "manifests" / "shards" / (
            f"{args.split}-{shard_index:06d}.jsonl"
        )
        shard_manifest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            shard_manifest,
            b"".join(
                (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for item in manifest_records
            ),
        )
        event = {
            "event": "shard_complete",
            "split": args.split,
            "rank": args.rank,
            "shard_index": shard_index,
            "sample_count": len(shard_records),
            "bytes": shard_index_data["bytes"],
            "sha256": shard_index_data["sha256"],
            "wall_seconds": time.monotonic() - shard_start,
            "terminal_counts": dict(Counter(str(item["terminal_G"]) for item in shard_records)),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with worker_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # The tar and per-member checksums are durable; spool is only a redundant copy.
        resolved_spool = spool.resolve()
        resolved_temp = temporary_root.resolve()
        if resolved_temp in resolved_spool.parents and resolved_spool.name.startswith("shard-"):
            shutil.rmtree(resolved_spool)
        if device.type == "npu":
            torch.npu.empty_cache()
        print(json.dumps(event, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

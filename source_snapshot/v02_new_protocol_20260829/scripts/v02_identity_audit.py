#!/usr/bin/env python3
"""Read-only binding of the real V0.1 implementation and baseline assets."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from physical_bridge.data import BridgeArchive, load_jsonl
from physical_bridge.metrics import LPIPSMetric, per_sample_l1, per_sample_psnr, per_sample_ssim
from physical_bridge.model import DualHeadUNet, SeverityEmbedding
from physical_bridge.simulator import PiRNP2SSimulator
from physical_bridge.utils import atomic_touch, atomic_write_json, sha256_file
from physical_bridge.v02_protocol import G_GRID, TERMINAL_GRID


EXPECTED = {
    "base_source_tree": "1e3bda1e6c6063ab104904c6e17187ffd5432db632d6bfd4e96874fb92ff1605",
    "checkpoint": "34dc4dce03bd13ba4079100d4b195fcc0e8c3b07699e06ae12ca91e0aba48f47",
    "simulator": "5c5023ed0cd8db1c759a3eca7dd047dd5169d9f761133ba0f29a4ceb8e20ddd4",
    "p2s": "578c3cd09c5c44e66c1086ffc0aadcab99542b74d27b932e9ca601a3faf6a4d5",
    "dictionary": "2204ace3e4b99f31d62d21d0ba19bae95c606393ba7df5111e2526172e6b24c6",
    "i0_i2": "aa99db54950df99465e5c8b7b7b16b7ed9236603b5c1a63d82dd8831b5712f2e",
    "train_manifest": "2c059ce420acf4bd299d602110e6470dc5a458262277393a7a1f3ae9ec64d94f",
    "split_summary": "fa0bb142095a60afb90a470a1e4025549193dd5cc74a2b0fa6de71a8d18d6198",
}


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for directory in ("physical_bridge", "scripts", "configs", "tests", "docs"):
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file())
    paths.extend([root / "pyproject.toml", root / "README.md"])
    for path in sorted(paths):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def callsite(symbol: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(symbol) or "").resolve()
    _, line = inspect.getsourcelines(symbol)
    return {"path": str(path), "line": line}


def dataset_audit(dataset_root: Path) -> dict[str, Any]:
    expected = {"train": (8000, 1600), "val": (400, 80), "test": (1000, 200)}
    required = {
        "clean.png",
        "epsilon_zernike.npy",
        "epsilon_tilt.npy",
        "x_g075.npy",
        "x_g150.npy",
        "x_g225.npy",
        "x_g300.npy",
        "x_g375.npy",
        "x_g450.npy",
    }
    all_keys: set[str] = set()
    all_sources: set[str] = set()
    result: dict[str, Any] = {"splits": {}}
    for split, (expected_rows, bucket_rows) in expected.items():
        rows = load_jsonl(dataset_root / "manifests" / f"{split}.jsonl")
        keys = [str(row["key"]) for row in rows]
        sources = {str(row["source_image_id"]) for row in rows}
        buckets = Counter(float(row["terminal_G"]) for row in rows)
        schema_failures = sum(set(row.get("member_checksums", {})) != required for row in rows)
        state_failures = sum(
            {float(item["g"]) for item in row.get("severity_statistics", [])} != set(G_GRID)
            for row in rows
        )
        seed_failures = sum(
            not {"epsilon_zernike", "epsilon_tilt", "preprocess"}.issubset(row.get("seeds", {}))
            for row in rows
        )
        split_result = {
            "rows": len(rows),
            "unique_keys": len(set(keys)),
            "terminal_buckets": {f"{key:.2f}": value for key, value in sorted(buckets.items())},
            "source_images": len(sources),
            "cross_split_key_overlap": len(all_keys & set(keys)),
            "cross_split_source_overlap": len(all_sources & sources),
            "member_schema_failures": schema_failures,
            "six_state_failures": state_failures,
            "replay_seed_failures": seed_failures,
        }
        split_result["pass"] = (
            len(rows) == expected_rows
            and len(set(keys)) == expected_rows
            and set(buckets) == set(TERMINAL_GRID)
            and all(value == bucket_rows for value in buckets.values())
            and not split_result["cross_split_key_overlap"]
            and not split_result["cross_split_source_overlap"]
            and not schema_failures
            and not state_failures
            and not seed_failures
        )
        result["splits"][split] = split_result
        all_keys.update(keys)
        all_sources.update(sources)

    samples: dict[str, Any] = {}
    for split in expected:
        archive = BridgeArchive(dataset_root, split)
        record = archive.records[0]
        clean = archive.clean(record)
        states = {f"{g:.2f}": archive.state(record, g) for g in G_GRID}
        realization = archive.realization(record)
        samples[split] = {
            "key": record["key"],
            "clean_shape": list(clean.shape),
            "clean_dtype": str(clean.dtype),
            "state_shapes": {key: list(value.shape) for key, value in states.items()},
            "state_dtypes": {key: str(value.dtype) for key, value in states.items()},
            "epsilon_zernike_shape": list(realization.epsilon_zernike.shape),
            "epsilon_zernike_dtype": str(realization.epsilon_zernike.dtype),
            "epsilon_tilt_shape": list(realization.epsilon_tilt.shape),
            "epsilon_tilt_dtype": str(realization.epsilon_tilt.dtype),
        }
        archive.store.close()
    result["sample_reads"] = samples
    result["trajectory_count"] = len(all_keys)
    result["pass"] = all(item["pass"] for item in result["splits"].values()) and len(all_keys) == 9400
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root = Path(document["v02_source_root"])
    base_root = Path(document["base_project_root"])
    dataset_root = Path(document["dataset_root"])
    checkpoint_path = Path(document["v01_checkpoint"])
    simulator_config_path = source_root / document["simulator_config"]
    simulator_document = yaml.safe_load(simulator_config_path.read_text(encoding="utf-8"))
    vendor_root = Path("/data0/yanglin/vendor/PiRN")

    hashes = {
        "base_source_tree": source_tree_hash(base_root),
        "checkpoint": sha256_file(checkpoint_path),
        "simulator": sha256_file(vendor_root / "assets/simulator_new.py"),
        "p2s": sha256_file(vendor_root / "assets/utils/P2S_model.pt"),
        "dictionary": sha256_file(vendor_root / "assets/utils/dictionary.pt"),
        "i0_i2": sha256_file(vendor_root / "assets/utils/I0_I2.pt"),
        "train_manifest": sha256_file(dataset_root / "manifests/train.jsonl"),
        "split_summary": sha256_file(dataset_root / "manifests/split_summary.json"),
        "method_document": sha256_file(source_root / "docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_METHOD.md"),
        "training_document": sha256_file(source_root / "docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_TRAINING_PLAN_CODEX.md"),
        "metric_source": sha256_file(base_root / "physical_bridge/metrics.py"),
        "model_source": sha256_file(base_root / "physical_bridge/model.py"),
        "simulator_wrapper_source": sha256_file(base_root / "physical_bridge/simulator.py"),
    }
    expected_all = {
        **EXPECTED,
        "method_document": document["method_document_sha256"],
        "training_document": document["training_document_sha256"],
    }
    hash_checks = {
        name: {"actual": hashes[name], "expected": expected, "pass": hashes[name] == expected}
        for name, expected in expected_all.items()
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DualHeadUNet(**document["model"])
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_check = {
        "update": int(checkpoint["update"]),
        "strict_load_missing": list(incompatible.missing_keys),
        "strict_load_unexpected": list(incompatible.unexpected_keys),
        "pass": int(checkpoint["update"]) == 20000
        and not incompatible.missing_keys
        and not incompatible.unexpected_keys,
    }
    del checkpoint, model

    tracked_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=base_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=base_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    data = dataset_audit(dataset_root)
    interface = {
        "model_forward": callsite(DualHeadUNet.forward),
        "severity_embedding": callsite(SeverityEmbedding.forward),
        "simulator_effective_state": callsite(PiRNP2SSimulator.effective_state),
        "simulator_render": callsite(PiRNP2SSimulator.render),
        "metric_psnr": callsite(per_sample_psnr),
        "metric_ssim": callsite(per_sample_ssim),
        "metric_l1": callsite(per_sample_l1),
        "metric_lpips": callsite(LPIPSMetric),
        "forward_signature": str(inspect.signature(DualHeadUNet.forward)),
        "g_conditioning": "SeverityEmbedding -> FiLMResBlock in encoder, middle, and both decoders",
        "realization_shapes": {
            "epsilon_zernike": [36, int(simulator_document["rf_range"]), int(simulator_document["rf_range"])],
            "epsilon_tilt": [int(simulator_document["rf_range"]), int(simulator_document["rf_range"]), 2],
        },
        "realization_dtype": "float32",
        "renderer_gradient_path": "No detach between predicted realization and effective_state/render; formal backward is gated by A0",
        "metrics": {
            "image_range": "[0,1] RGB",
            "clamp_before_numeric_metrics": False,
            "valid_mask_or_crop": None,
            "psnr": "per-image RGB mean MSE, data_range 1",
            "ssim": "11x11 Gaussian, sigma 1.5, zero padding 5, RGB mean",
            "lpips": "lpips 0.1.4, alex backbone, input mapped [0,1] to [-1,1]",
            "aggregation": "per-image then arithmetic mean",
        },
    }
    report = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "resolved_paths": {
            "base_root": str(base_root.resolve()),
            "v02_source_root": str(source_root.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "simulator_config": str(simulator_config_path.resolve()),
        },
        "authorization": {
            "server": "liteserver-40b3",
            "physical_npus": document["authorized_physical_npus"],
        },
        "base_git": {
            "commit": git_commit,
            "tracked_status": tracked_status,
            "tracked_clean": not tracked_status,
        },
        "hashes": hashes,
        "hash_checks": hash_checks,
        "checkpoint": checkpoint_check,
        "data": data,
        "interface": interface,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    failures = [name for name, check in hash_checks.items() if not check["pass"]]
    if failures or not checkpoint_check["pass"] or not data["pass"] or tracked_status:
        report["status"] = "STOP"
        report["implementation_report_mismatch"] = {
            "hash_failures": failures,
            "checkpoint_failure": not checkpoint_check["pass"],
            "data_failure": not data["pass"],
            "tracked_source_dirty": tracked_status,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "v02_identity_audit.json", report)
    lines = [
        "# V0.2 identity audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- V0.1 commit: `{git_commit}`",
        f"- V0.1 tracked source clean: `{not tracked_status}`",
        f"- Checkpoint update: `{checkpoint_check['update']}`",
        f"- Checkpoint SHA-256: `{hashes['checkpoint']}`",
        f"- Dataset trajectories: `{data['trajectory_count']}`",
        f"- Split/data identity pass: `{data['pass']}`",
        f"- Forward signature: `{interface['forward_signature']}`",
        f"- Explicit g conditioning: `{interface['g_conditioning']}`",
        "- Simulator predicted-xi backward: deferred to the formal A0 gate.",
        "",
        "No V0.1 tracked source, data, checkpoint, result, or protocol file was modified.",
    ]
    (args.output_dir / "v02_identity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_touch(args.output_dir / ("IDENTITY_PASS" if report["status"] == "PASS" else "IDENTITY_STOP"))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

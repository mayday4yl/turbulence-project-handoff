#!/usr/bin/env python3
"""Freeze A0 and A1 manifests before observing V0.2 gradients."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from physical_bridge.utils import atomic_touch, atomic_write_json
from physical_bridge.v02_data import build_a0_manifest, build_a1_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    identity = Path(document["run_root"]) / "identity" / "IDENTITY_PASS"
    if not identity.exists():
        raise RuntimeError(f"Identity gate is not PASS: {identity}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "a0": build_a0_manifest(document["dataset_root"], args.output_dir / "a0"),
        "a1": build_a1_manifests(document["dataset_root"], args.output_dir / "a1"),
    }
    atomic_write_json(args.output_dir / "manifest_summary.json", summary)
    atomic_touch(args.output_dir / "MANIFESTS_FROZEN")


if __name__ == "__main__":
    main()

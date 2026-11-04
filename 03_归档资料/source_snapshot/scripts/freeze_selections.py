#!/usr/bin/env python3
"""Apply the frozen checkpoint, refinement-LR, and K selection rules."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from physical_bridge.utils import atomic_write_json, sha256_file


def checkpoint_selection(run_dir: Path) -> dict[str, Any]:
    state = json.loads((run_dir / "val" / "selection_state.json").read_text(encoding="utf-8"))
    best = state["best"]
    source = Path(best["checkpoint"])
    destination = run_dir / "checkpoints" / "frozen_checkpoint.pt"
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(f"Existing frozen checkpoint differs: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    result = {
        "selection_rule": "equal-severity PSNR; within 0.05 dB LPIPS; then L_D residual",
        "source_checkpoint": str(source),
        "frozen_checkpoint": str(destination),
        "sha256": sha256_file(destination),
        "validation": best,
    }
    atomic_write_json(run_dir / "frozen_checkpoint.json", result)
    return result


def lr_selection(summary_path: Path, frozen_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary["results"]
    entries = []
    for name, value in results.items():
        entries.append(
            {
                "name": name,
                "learning_rate": float(name.split("_", 1)[1]),
                "psnr": float(value["mean"]["psnr"]),
                "lpips": float(value["mean"]["lpips"]),
            }
        )
    best = max(entries, key=lambda item: (item["psnr"], -item["lpips"]))
    default = next(item for item in entries if abs(item["learning_rate"] - 0.01) < 1e-12)
    selected = default if best["psnr"] - default["psnr"] <= 0.05 else best
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    frozen.update(
        {
            "learning_rate": selected["learning_rate"],
            "learning_rate_selection": {
                "rule": "keep 0.01 when within 0.05 dB of the best final PSNR",
                "candidates": entries,
                "best": best,
                "selected": selected,
                "source": str(summary_path),
            },
        }
    )
    atomic_write_json(frozen_path, frozen)
    return frozen


def k_selection(summary_path: Path, frozen_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = []
    for name, value in summary["results"].items():
        entries.append(
            {
                "name": name,
                "K": int(name.split("_", 1)[1]),
                "psnr": float(value["mean"]["psnr"]),
                "lpips": float(value["mean"]["lpips"]),
            }
        )
    maximum_psnr = max(item["psnr"] for item in entries)
    eligible = [item for item in entries if maximum_psnr - item["psnr"] <= 0.05]
    minimum_lpips = min(item["lpips"] for item in eligible)
    lpips_tied = [item for item in eligible if item["lpips"] - minimum_lpips <= 1e-6]
    selected = min(lpips_tied, key=lambda item: item["K"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen.update(
        {
            "K": selected["K"],
            "k_selection": {
                "rule": "max PSNR; within 0.05 dB lower LPIPS; then smaller K",
                "candidates": entries,
                "selected": selected,
                "source": str(summary_path),
            },
            "status": "FROZEN_FOR_TEST",
        }
    )
    atomic_write_json(frozen_path, frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("checkpoint", "lr", "k"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    frozen_path = args.run_dir / "frozen_inference.json"
    if args.mode == "checkpoint":
        result = checkpoint_selection(args.run_dir)
        frozen = {
            "checkpoint": result["frozen_checkpoint"],
            "checkpoint_sha256": result["sha256"],
            "status": "CHECKPOINT_FROZEN",
        }
        atomic_write_json(frozen_path, frozen)
    elif args.mode == "lr":
        if args.summary is None:
            raise ValueError("--summary is required for lr selection")
        result = lr_selection(args.summary, frozen_path)
    else:
        if args.summary is None:
            raise ValueError("--summary is required for K selection")
        result = k_selection(args.summary, frozen_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

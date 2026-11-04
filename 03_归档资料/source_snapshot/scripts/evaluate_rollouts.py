#!/usr/bin/env python3
"""Distributed, resumable validation/test rollouts for all frozen comparators."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def refinement_config(document: dict[str, Any], learning_rate: float) -> Any:
    from physical_bridge.inference import RefinementConfig

    value = document["refinement"]
    return RefinementConfig(
        learning_rate=float(learning_rate),
        betas=tuple(value["betas"]),
        first_iterations=int(value["first_iterations"]),
        later_iterations=int(value["later_iterations"]),
        gradient_clip=float(value["gradient_clip"]),
        early_stop_patience=int(value["early_stop_patience"]),
        early_stop_relative_improvement=float(value["early_stop_relative_improvement"]),
        valid_border=int(value["valid_border"]),
        proximal_weight=float(value["proximal_weight"]),
        prior_weight=float(value["prior_weight"]),
    )


def experiment_configs(
    phase: str, inference_document: dict[str, Any], frozen: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if phase == "lr":
        return [
            {
                "name": f"lr_{float(lr):.3f}",
                "method": "bridge_refinement",
                "K": 5,
                "learning_rate": float(lr),
            }
            for lr in inference_document["refinement"]["candidate_learning_rates"]
        ]
    if phase == "k":
        if frozen is None or "learning_rate" not in frozen:
            raise ValueError("K validation requires a frozen refinement learning rate")
        result = []
        for value in inference_document["candidate_k"]:
            result.append(
                {
                    "name": f"k_{int(value)}",
                    "method": "one_shot" if int(value) == 0 else "bridge_refinement",
                    "K": int(value),
                    "learning_rate": float(frozen["learning_rate"]),
                }
            )
        return result
    if phase == "test":
        if frozen is None or not {"learning_rate", "K"}.issubset(frozen):
            raise ValueError("Frozen test requires frozen learning rate and K")
        learning_rate = float(frozen["learning_rate"])
        transitions = int(frozen["K"])
        return [
            {"name": "one_shot", "method": "one_shot", "K": 0, "learning_rate": learning_rate},
            {
                "name": "bridge_no_refinement",
                "method": "bridge_no_refinement",
                "K": transitions,
                "learning_rate": learning_rate,
            },
            {
                "name": "bridge_refinement",
                "method": "bridge_refinement",
                "K": transitions,
                "learning_rate": learning_rate,
            },
            {
                "name": "oracle_xi",
                "method": "oracle_xi",
                "K": transitions,
                "learning_rate": learning_rate,
            },
            {
                "name": "teacher_forced",
                "method": "teacher_forced",
                "K": transitions,
                "learning_rate": learning_rate,
            },
        ]
    raise ValueError(f"Unknown phase: {phase}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)
    for name, config_rows in sorted(by_config.items()):
        by_terminal: dict[str, dict[str, float]] = {}
        for terminal in (1.5, 2.25, 3.0, 3.75, 4.5):
            selected = [row for row in config_rows if float(row["terminal_G"]) == terminal]
            if not selected:
                continue
            by_terminal[f"{terminal:.2f}"] = {
                metric: float(np.mean([row["final_quality"][metric] for row in selected]))
                for metric in ("l1", "psnr", "ssim", "lpips")
            }
        per_step: dict[str, dict[str, float]] = {}
        step_buckets: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
        for row in config_rows:
            for step in row["steps"]:
                step_buckets[(int(step["step"]), float(step["current_g"]))].append(step)
        for (step_index, severity), step_rows in sorted(step_buckets.items()):
            values = {
                "count": len(step_rows),
                "observation_residual_before": float(
                    np.mean([item["observation_residual_before"] for item in step_rows])
                ),
                "observation_residual_after": float(
                    np.mean([item["observation_residual_after"] for item in step_rows])
                ),
            }
            if "clean_quality" in step_rows[0]:
                values.update(
                    {
                        f"clean_{metric}": float(
                            np.mean([item["clean_quality"][metric] for item in step_rows])
                        )
                        for metric in ("l1", "psnr", "ssim", "lpips")
                    }
                )
            if "next_state_mse" in step_rows[0]:
                values["next_state_mse"] = float(
                    np.mean([item["next_state_mse"] for item in step_rows])
                )
            values.update(
                {
                    "mean_wall_seconds": float(
                        np.mean([item["wall_seconds"] for item in step_rows])
                    ),
                    "mean_simulator_forwards": float(
                        np.mean([item["simulator_forwards"] for item in step_rows])
                    ),
                    "mean_simulator_backwards": float(
                        np.mean([item["simulator_backwards"] for item in step_rows])
                    ),
                    "peak_npu_memory_bytes": int(
                        max(item.get("peak_npu_memory_bytes", 0) for item in step_rows)
                    ),
                }
            )
            refinement_rows = [item["refinement"] for item in step_rows if "refinement" in item]
            if refinement_rows:
                values.update(
                    {
                        "mean_refinement_iterations": float(
                            np.mean([item["iterations"] for item in refinement_rows])
                        ),
                        "mean_zernike_update_norm": float(
                            np.mean([item["update_norms"]["zernike"] for item in refinement_rows])
                        ),
                        "mean_tilt_update_norm": float(
                            np.mean([item["update_norms"]["tilt"] for item in refinement_rows])
                        ),
                    }
                )
            per_step[f"step{step_index}_g{severity:.2f}"] = values
        result[name] = {
            "count": len(config_rows),
            "mean": {
                metric: float(np.mean([row["final_quality"][metric] for row in config_rows]))
                for metric in ("l1", "psnr", "ssim", "lpips")
            },
            "by_terminal_G": by_terminal,
            "per_step": per_step,
            "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in config_rows])),
            "mean_network_forwards": float(
                np.mean([row["network_forwards"] for row in config_rows])
            ),
            "mean_simulator_forwards": float(
                np.mean([row["simulator_forwards"] for row in config_rows])
            ),
            "mean_simulator_backwards": float(
                np.mean([row["simulator_backwards"] for row in config_rows])
            ),
            "peak_npu_memory_bytes": int(max(row["peak_npu_memory_bytes"] for row in config_rows)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("lr", "k", "test"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--frozen-inference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    from physical_bridge.data import BridgeArchive, FROZEN_G_GRID
    from physical_bridge.inference import run_rollout
    from physical_bridge.metrics import LPIPSMetric
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator, Realization
    from physical_bridge.utils import atomic_touch, atomic_write_json

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="hccl", timeout=dt.timedelta(hours=12)
        )
    barrier = torch.distributed.barrier if world_size > 1 else lambda: None
    train_document = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    simulator_document = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    inference_document = yaml.safe_load(args.inference_config.read_text(encoding="utf-8"))
    frozen = (
        json.loads(args.frozen_inference.read_text(encoding="utf-8"))
        if args.frozen_inference
        else None
    )
    configurations = experiment_configs(args.phase, inference_document, frozen)
    model_config = train_document["model"]
    model = DualHeadUNet(
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        residual_blocks_per_level=int(model_config["residual_blocks_per_level"]),
        group_norm_groups=int(model_config["group_norm_groups"]),
        severity_embedding_dim=int(model_config["severity_embedding_dim"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval().requires_grad_(False)
    simulator = PiRNP2SSimulator(
        simulator_document["asset_dir"],
        image_size=int(simulator_document["image_size"]),
        rf_range=int(simulator_document["rf_range"]),
        aperture_diameter=float(simulator_document["aperture_diameter"]),
        propagation_distance=float(simulator_document["propagation_distance"]),
        wavelength=float(simulator_document["wavelength"]),
        correlation=float(simulator_document["correlation"]),
        tilt_threshold=float(simulator_document["tilt_threshold"]),
        adjustment=float(simulator_document["adjustment"]),
        device=device,
    )
    archive = BridgeArchive(args.dataset_root, args.split)
    qualitative = json.loads(
        (args.dataset_root / "manifests" / "qualitative_ids.json").read_text(encoding="utf-8")
    )
    if args.phase == "lr":
        allowed_keys = set(qualitative["refinement_lr_val_g450"])
        capture_keys = allowed_keys
    else:
        allowed_keys = None
        capture_keys = set(qualitative[args.split])
    lpips_metric = LPIPSMetric(device)
    rank_path = args.output_dir / f"rank-{rank:02d}.jsonl"
    existing = load_rows(rank_path)
    complete_pairs = {(row["key"], row["config"]) for row in existing}
    records = [
        record
        for record in archive.records
        if allowed_keys is None or record["key"] in allowed_keys
    ]
    start = time.monotonic()
    for record in records[rank::world_size]:
        clean = archive.clean(record).unsqueeze(0).to(device)
        terminal_g = float(record["terminal_G"])
        observation = archive.state(record, terminal_g).unsqueeze(0).to(device)
        true_realization_cpu = archive.realization(record)
        true_realization = Realization(
            true_realization_cpu.epsilon_zernike.unsqueeze(0).to(device),
            true_realization_cpu.epsilon_tilt.unsqueeze(0).to(device),
        )
        true_states = {
            g: archive.state(record, g).unsqueeze(0).to(device)
            for g in FROZEN_G_GRID
            if g <= terminal_g
        }
        for experiment in configurations:
            pair = (record["key"], experiment["name"])
            if pair in complete_pairs:
                continue
            torch.npu.reset_peak_memory_stats(device)
            capture = record["key"] in capture_keys
            result = run_rollout(
                model=model,
                simulator=simulator,
                observation=observation,
                observation_severity=terminal_g,
                transitions=int(experiment["K"]),
                method=experiment["method"],
                refinement_config=refinement_config(
                    inference_document, float(experiment["learning_rate"])
                ),
                true_clean=clean,
                true_realization=true_realization,
                true_states=true_states,
                lpips_metric=lpips_metric,
                capture_tensors=capture,
            )
            if capture:
                payload = {
                    "key": record["key"],
                    "terminal_G": terminal_g,
                    "config": experiment,
                    "target_clean": clean.detach().half().cpu(),
                    "observation": observation.detach().half().cpu(),
                    "captured": result.pop("captured"),
                    "final_input": result.pop("final_input"),
                    "final_prediction": result.pop("final_prediction"),
                }
                atomic_torch_save(
                    args.output_dir
                    / "qualitative_tensors"
                    / experiment["name"]
                    / f"{record['key']}.pt",
                    payload,
                )
            result.pop("final")
            row = {
                "key": record["key"],
                "terminal_G": terminal_g,
                "config": experiment["name"],
                "experiment": experiment,
                **result,
                "peak_npu_memory_bytes": int(torch.npu.max_memory_allocated(device)),
            }
            append_jsonl(rank_path, row)
            complete_pairs.add(pair)
        torch.npu.empty_cache()
    atomic_touch(args.output_dir / f"rank-{rank:02d}.complete")
    barrier()
    if rank == 0:
        combined = []
        for worker_rank in range(world_size):
            combined.extend(load_rows(args.output_dir / f"rank-{worker_rank:02d}.jsonl"))
        expected = len(records) * len(configurations)
        if len(combined) != expected:
            raise ValueError(f"Expected {expected} rollout rows, got {len(combined)}")
        summary = {
            "phase": args.phase,
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "configurations": configurations,
            "records": len(records),
            "rows": len(combined),
            "results": summarize(combined),
            "wall_seconds": time.monotonic() - start,
        }
        atomic_write_json(args.output_dir / "summary.json", summary)
        atomic_touch(args.output_dir / "PHASE_COMPLETE")
        print(json.dumps(summary, indent=2, sort_keys=True))
    barrier()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

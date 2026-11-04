#!/usr/bin/env python3
"""Exact-resume four-NPU training for Known-g Physical Turbulence Bridge v0.1."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def lambda_degradation(update: int, config: dict[str, Any]) -> float:
    zero_until = int(config["degradation_zero_until"])
    ramp_until = int(config["degradation_ramp_until"])
    final = float(config["lambda_degradation_final"])
    if update < zero_until:
        return 0.0
    if update < ramp_until:
        return final * (update - zero_until) / (ramp_until - zero_until)
    return final


def learning_rate(update: int, maximum: int, config: dict[str, Any]) -> float:
    peak = float(config["learning_rate"])
    final = float(config["final_learning_rate"])
    warmup = int(config["warmup_updates"])
    completed = update + 1
    if completed <= warmup:
        return peak * completed / warmup
    progress = min(1.0, (completed - warmup) / max(1, maximum - warmup))
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))


def select_checkpoint(history: list[dict[str, Any]], tolerance_db: float) -> dict[str, Any]:
    """Apply the frozen checkpoint rule globally without sequential tie drift."""

    if not history:
        raise ValueError("Checkpoint selection requires validation history")
    maximum_psnr = max(float(item["psnr"]) for item in history)
    eligible = [
        item for item in history if maximum_psnr - float(item["psnr"]) <= tolerance_db
    ]
    minimum_lpips = min(float(item["lpips"]) for item in eligible)
    lpips_best = [
        item for item in eligible if float(item["lpips"]) == minimum_lpips
    ]
    minimum_residual = min(float(item["degradation_residual"]) for item in lpips_best)
    residual_best = [
        item
        for item in lpips_best
        if float(item["degradation_residual"]) == minimum_residual
    ]
    return min(residual_best, key=lambda item: int(item["update"]))


def validation_selection_state(
    history: list[dict[str, Any]], tolerance_db: float
) -> tuple[dict[str, Any] | None, int, float]:
    if not history:
        return None, 0, -math.inf
    reference = -math.inf
    last_significant = 0
    for item in sorted(history, key=lambda value: int(value["update"])):
        if float(item["psnr"]) > reference + tolerance_db:
            reference = float(item["psnr"])
            last_significant = int(item["update"])
    return select_checkpoint(history, tolerance_db), last_significant, reference


def validation_candidate(update: int, checkpoint_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "update": int(update),
        "checkpoint": str(checkpoint_dir / f"update-{int(update):06d}.pt"),
        "psnr": float(summary["equal_severity_mean_psnr"]),
        "lpips": float(summary["equal_severity_mean_lpips"]),
        "degradation_residual": float(
            summary["equal_severity_mean_degradation_residual"]
        ),
    }


def metrics_prefix_bytes(path: Path, maximum_update: int) -> bytes:
    if not path.exists():
        return b""
    latest: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            update = int(row["update"])
            if update <= maximum_update:
                latest[update] = row
    return b"".join(
        (json.dumps(latest[update], sort_keys=True) + "\n").encode()
        for update in sorted(latest)
    )


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in state.items():
            temporary = value.detach().clone()
            value.copy_(self.shadow[name])
            self.shadow[name].copy_(temporary)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], device: torch.device) -> None:
        self.shadow = {name: value.to(device) for name, value in state.items()}


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def source_tree_hash(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    paths = []
    for directory in ("physical_bridge", "scripts", "configs", "tests", "docs"):
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file())
    paths.extend([root / "pyproject.toml", root / "README.md"])
    for path in sorted(paths):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("update-*.pt"))
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--maximum-updates", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    from physical_bridge.data import RankSampleSampler, StatelessTrainingDataset
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator
    from physical_bridge.utils import (
        atomic_touch,
        atomic_write_bytes,
        atomic_write_json,
        parameter_grad_norm,
        seed_process,
        sha256_file,
    )
    from physical_bridge.validation import teacher_forced_validation

    train_document = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    simulator_document = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    config = train_document["training"]
    model_config = train_document["model"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group(
            backend="hccl", timeout=dt.timedelta(hours=6)
        )
    barrier = torch.distributed.barrier if distributed else None
    master_seed = int(train_document["master_seed"])
    seed_process(master_seed)
    effective_batch = int(config["effective_batch_size"])
    if effective_batch % world_size:
        raise ValueError("Effective batch size is not divisible by world size")
    local_per_update = effective_batch // world_size
    if local_per_update % args.microbatch:
        raise ValueError(
            f"microbatch {args.microbatch} does not divide local batch {local_per_update}"
        )
    accumulation = local_per_update // args.microbatch
    maximum_updates = args.maximum_updates or int(config["maximum_updates"])
    validation_interval = args.validation_interval or int(config["full_validation_interval"])
    checkpoint_interval = args.checkpoint_interval or int(config["checkpoint_interval"])
    schedule_maximum = (
        maximum_updates if args.maximum_updates is not None else int(config["maximum_updates"])
    )
    dataset_root = Path(train_document["dataset_root"])
    run_dir = Path(train_document["run_root"]) / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        frozen_config = {
            "train": train_document,
            "simulator": simulator_document,
            "runtime": {
                "world_size": world_size,
                "physical_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "microbatch_per_rank": args.microbatch,
                "gradient_accumulation": accumulation,
                "effective_batch_size": effective_batch,
                "maximum_updates": maximum_updates,
            },
        }
        atomic_write_json(run_dir / "frozen_config.json", frozen_config)
        identity = {
            "project_source_hash": source_tree_hash(Path.cwd()),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None,
            "git_status": subprocess.run(
                ["git", "status", "--porcelain=v1"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines(),
            "train_config_sha256": sha256_file(args.train_config),
            "simulator_config_sha256": sha256_file(args.simulator_config),
            "dataset_manifest_sha256": sha256_file(dataset_root / "manifests" / "train.jsonl"),
            "split_summary_sha256": sha256_file(dataset_root / "manifests" / "split_summary.json"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "npu_smi": subprocess.run(
                ["npu-smi", "info"], capture_output=True, text=True, check=False
            ).stdout,
            "pip_freeze": subprocess.run(
                [str(Path(os.sys.executable)), "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines(),
        }
        atomic_write_json(run_dir / "source_identity.json", identity)
    if barrier:
        barrier()

    model = DualHeadUNet(
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        residual_blocks_per_level=int(model_config["residual_blocks_per_level"]),
        group_norm_groups=int(model_config["group_norm_groups"]),
        severity_embedding_dim=int(model_config["severity_embedding_dim"]),
    ).to(device)
    raw_model = model
    ddp_model: torch.nn.Module
    if distributed:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        ddp_model = model
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
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(config["betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    ema = EMA(raw_model, float(config["ema"]))
    start_update = 0
    validation_history: list[dict[str, Any]] = []
    best_selection: dict[str, Any] | None = None
    last_significant_improvement = 0
    significant_reference_psnr = -math.inf
    resume_path: Path | None = None
    if args.resume == "auto":
        resume_path = latest_checkpoint(checkpoint_dir)
    elif args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_to(optimizer, device)
        ema.load_state_dict(checkpoint["ema"], device)
        start_update = int(checkpoint["update"])
        validation_history = checkpoint.get("validation_history", [])
        best_selection = checkpoint.get("best_selection")
        last_significant_improvement = int(
            checkpoint.get("last_significant_improvement", 0)
        )
        significant_reference_psnr = float(
            checkpoint.get("significant_reference_psnr", -math.inf)
        )
        selection_path = run_dir / "val" / "selection_state.json"
        if selection_path.exists():
            selection_state = json.loads(selection_path.read_text(encoding="utf-8"))
            history = selection_state.get("history", [])
            if not history or int(history[-1]["update"]) <= start_update:
                validation_history = history
                best_selection = selection_state.get("best")
                last_significant_improvement = int(
                    selection_state.get("last_significant_improvement", 0)
                )
                significant_reference_psnr = float(
                    selection_state.get("significant_reference_psnr", -math.inf)
                )
        if not math.isfinite(significant_reference_psnr) and validation_history:
            eligible_history = [
                item
                for item in validation_history
                if int(item["update"]) <= last_significant_improvement
            ]
            if eligible_history:
                significant_reference_psnr = max(
                    float(item["psnr"]) for item in eligible_history
                )
    dataset = StatelessTrainingDataset(dataset_root, master_seed)
    sampler_end = (
        int(config["extension_updates"])
        if args.maximum_updates is None
        else maximum_updates
    )
    sampler = RankSampleSampler(
        start_update=start_update,
        end_update=sampler_end,
        effective_batch_size=effective_batch,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.microbatch,
        sampler=sampler,
        num_workers=int(config["num_workers_per_rank"]),
        prefetch_factor=int(config["prefetch_factor"]),
        persistent_workers=int(config["num_workers_per_rank"]) > 0,
        drop_last=True,
    )
    iterator = iter(loader)
    qualitative = json.loads(
        (dataset_root / "manifests" / "qualitative_ids.json").read_text(encoding="utf-8")
    )["val"]
    training_start = time.monotonic()
    stopped_early = False
    current_maximum = maximum_updates
    update = start_update
    tolerance = float(config["early_stop_min_delta_db"])
    if rank == 0:
        rebuilt_history = []
        for summary_path in sorted((run_dir / "val").glob("update-*/summary.json")):
            summary_update = int(summary_path.parent.name.split("-", 1)[1])
            if summary_update <= start_update:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rebuilt_history.append(
                    validation_candidate(summary_update, checkpoint_dir, summary)
                )
        validation_history = rebuilt_history
        (
            best_selection,
            last_significant_improvement,
            significant_reference_psnr,
        ) = validation_selection_state(validation_history, tolerance)
        atomic_write_bytes(
            run_dir / "train_metrics.jsonl",
            metrics_prefix_bytes(run_dir / "train_metrics.jsonl", start_update),
        )
        if validation_history:
            atomic_write_json(
                run_dir / "val" / "selection_state.json",
                {
                    "history": validation_history,
                    "best": best_selection,
                    "last_significant_improvement": last_significant_improvement,
                    "significant_reference_psnr": significant_reference_psnr,
                },
            )
    if barrier:
        barrier()

    pending_validation = torch.zeros(1, device=device, dtype=torch.int64)
    if rank == 0:
        pending_summary = run_dir / "val" / f"update-{start_update:06d}" / "summary.json"
        pending_validation[0] = int(
            resume_path is not None
            and not args.no_validation
            and start_update > 0
            and start_update % validation_interval == 0
            and not pending_summary.exists()
        )
    if distributed:
        torch.distributed.broadcast(pending_validation, src=0)
    if bool(pending_validation.item()):
        ema.swap(raw_model)
        pending_summary = teacher_forced_validation(
            model=raw_model,
            simulator=simulator,
            dataset_root=dataset_root,
            output_dir=run_dir / "val" / f"update-{start_update:06d}",
            device=device,
            rank=rank,
            world_size=world_size,
            qualitative_keys=qualitative if rank == 0 else None,
            distributed_barrier=barrier,
        )
        ema.swap(raw_model)
        if rank == 0 and pending_summary is not None:
            validation_history.append(
                validation_candidate(start_update, checkpoint_dir, pending_summary)
            )
            (
                best_selection,
                last_significant_improvement,
                significant_reference_psnr,
            ) = validation_selection_state(validation_history, tolerance)
            atomic_write_json(
                run_dir / "val" / "selection_state.json",
                {
                    "history": validation_history,
                    "best": best_selection,
                    "last_significant_improvement": last_significant_improvement,
                    "significant_reference_psnr": significant_reference_psnr,
                },
            )
        if barrier:
            barrier()
    while update < current_maximum:
        step_start = time.monotonic()
        # A val-authorized extension continues from 30k at the already reached
        # final LR; it must not create a second LR rise.
        lr = learning_rate(update, schedule_maximum, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        degradation_weight = lambda_degradation(update, config)
        optimizer.zero_grad(set_to_none=True)
        accumulated = {"clean": 0.0, "degradation": 0.0, "total": 0.0}
        for microstep in range(accumulation):
            batch = next(iterator)
            image = batch["image"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            terminal = batch["terminal"].to(device, non_blocking=True)
            severity = batch["g"].to(device, non_blocking=True)
            terminal_severity = batch["G"].to(device, non_blocking=True)
            sync_context = (
                ddp_model.no_sync()
                if distributed and microstep < accumulation - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                    prediction = ddp_model(image, severity)
                clean_loss = (prediction.clean.float() - clean.float()).square().mean()
                if degradation_weight > 0:
                    reconstructed = simulator.render(
                        clean.float(), prediction.realization, terminal_severity
                    ).output
                    degradation_loss = (
                        reconstructed.float() - terminal.float()
                    ).square().mean()
                else:
                    degradation_loss = torch.zeros((), device=device)
                total_loss = (
                    float(config["lambda_clean"]) * clean_loss
                    + degradation_weight * degradation_loss
                )
                if not bool(torch.isfinite(total_loss).detach().cpu()):
                    raise FloatingPointError(f"Non-finite loss at update {update}")
                (total_loss / accumulation).backward()
            accumulated["clean"] += float(clean_loss.detach().cpu()) / accumulation
            accumulated["degradation"] += float(degradation_loss.detach().cpu()) / accumulation
            accumulated["total"] += float(total_loss.detach().cpu()) / accumulation
        preclip_norm = float(
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(config["gradient_clip"])).cpu()
        )
        branch_norms = {
            name: parameter_grad_norm(parameters)
            for name, parameters in raw_model.branch_parameters().items()
        }
        optimizer.step()
        ema.update(raw_model)
        update += 1
        reduced = torch.tensor(
            [
                accumulated["total"],
                accumulated["clean"],
                accumulated["degradation"],
                preclip_norm,
                branch_norms["shared"],
                branch_norms["clean"],
                branch_norms["realization"],
            ],
            device=device,
            dtype=torch.float32,
        )
        if distributed:
            torch.distributed.all_reduce(reduced)
            reduced /= world_size
        if rank == 0:
            append_jsonl(
                run_dir / "train_metrics.jsonl",
                {
                    "update": update,
                    "loss": float(reduced[0].cpu()),
                    "clean_loss": float(reduced[1].cpu()),
                    "degradation_loss": float(reduced[2].cpu()),
                    "lambda_degradation": degradation_weight,
                    "learning_rate": lr,
                    "preclip_gradient_norm": float(reduced[3].cpu()),
                    "shared_gradient_norm": float(reduced[4].cpu()),
                    "clean_branch_gradient_norm": float(reduced[5].cpu()),
                    "degradation_branch_gradient_norm": float(reduced[6].cpu()),
                    "step_wall_seconds": time.monotonic() - step_start,
                    "elapsed_wall_seconds": time.monotonic() - training_start,
                },
            )

        checkpoint_due = update % checkpoint_interval == 0 or update == current_maximum
        validation_due = (
            not args.no_validation
            and (update % validation_interval == 0 or update == current_maximum)
        )
        if checkpoint_due:
            if barrier:
                barrier()
            checkpoint_path = checkpoint_dir / f"update-{update:06d}.pt"
            if rank == 0:
                checkpoint_value = {
                    "update": update,
                    "model": cpu_tree(raw_model.state_dict()),
                    "ema": ema.state_dict_cpu(),
                    "optimizer": cpu_tree(optimizer.state_dict()),
                    "validation_history": validation_history,
                    "best_selection": best_selection,
                    "last_significant_improvement": last_significant_improvement,
                    "significant_reference_psnr": significant_reference_psnr,
                    "runtime": {
                        "world_size": world_size,
                        "microbatch": args.microbatch,
                        "accumulation": accumulation,
                    },
                }
                atomic_checkpoint(checkpoint_path, checkpoint_value)
                atomic_write_json(
                    checkpoint_path.with_suffix(".json"),
                    {
                        "update": update,
                        "path": str(checkpoint_path),
                        "sha256": sha256_file(checkpoint_path),
                    },
                )
            if barrier:
                barrier()

        if validation_due:
            if not checkpoint_due:
                raise AssertionError("Full validation must coincide with a checkpoint")
            ema.swap(raw_model)
            summary = teacher_forced_validation(
                model=raw_model,
                simulator=simulator,
                dataset_root=dataset_root,
                output_dir=run_dir / "val" / f"update-{update:06d}",
                device=device,
                rank=rank,
                world_size=world_size,
                qualitative_keys=qualitative if rank == 0 else None,
                distributed_barrier=barrier,
            )
            ema.swap(raw_model)
            if rank == 0 and summary is not None:
                candidate = validation_candidate(update, checkpoint_dir, summary)
                validation_history.append(candidate)
                (
                    best_selection,
                    last_significant_improvement,
                    significant_reference_psnr,
                ) = validation_selection_state(validation_history, tolerance)
                atomic_write_json(
                    run_dir / "val" / "selection_state.json",
                    {
                        "history": validation_history,
                        "best": best_selection,
                        "last_significant_improvement": last_significant_improvement,
                        "significant_reference_psnr": significant_reference_psnr,
                    },
                )
            if barrier:
                barrier()
            control = torch.zeros(2, device=device, dtype=torch.int64)
            if rank == 0:
                minimum = int(config["minimum_updates"])
                patience = int(config["early_stop_patience_updates"])
                should_stop = update >= minimum and update - last_significant_improvement >= patience
                should_extend = False
                if update == maximum_updates and maximum_updates == int(config["maximum_updates"]):
                    recent = [item for item in validation_history if item["update"] >= update - 5000]
                    earlier = [item for item in validation_history if item["update"] <= update - 5000]
                    if recent and earlier:
                        should_extend = max(item["psnr"] for item in recent) > max(
                            item["psnr"] for item in earlier
                        ) + float(config["early_stop_min_delta_db"])
                control[0] = int(should_stop)
                control[1] = int(should_extend)
            if distributed:
                torch.distributed.broadcast(control, src=0)
            if bool(control[0].item()):
                stopped_early = True
                break
            if bool(control[1].item()):
                current_maximum = int(config["extension_updates"])

    if barrier:
        barrier()
    if rank == 0:
        if best_selection is None and args.no_validation:
            best_selection = {
                "update": update,
                "checkpoint": str(checkpoint_dir / f"update-{update:06d}.pt"),
                "smoke_only": True,
            }
        atomic_write_json(
            run_dir / "training_summary.json",
            {
                "completed_updates": update,
                "stopped_early": stopped_early,
                "best_selection": best_selection,
                "validation_history": validation_history,
                "last_significant_improvement": last_significant_improvement,
                "significant_reference_psnr": significant_reference_psnr,
                "wall_seconds": time.monotonic() - training_start,
            },
        )
        atomic_touch(run_dir / "TRAINING_COMPLETE")
    if barrier:
        barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

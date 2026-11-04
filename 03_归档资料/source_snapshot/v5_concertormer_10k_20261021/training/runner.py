"""Single-process or HCCL/DDP optimizer-update budget runner.

Environment setup, accepted Simulator construction and readiness/probe reports
belong to the runtime factory. This module never selects a different backbone.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import importlib
import json
import math
import os
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .core import unroll_training
from .evaluation import (append_jsonl, distributed_info, evaluate_full_validation,
                         json_safe, memory_stats, synchronize)
from .state import (UpdateBatchSampler, UpdateLRScheduler, atomic_json,
                    atomic_torch_save, canonical_hash, capture_rng, file_sha256,
                    gather_rank_state, restore_rng)


@dataclass
class RunConfig:
    method: str
    output_dir: str
    initial_checkpoint: str
    readiness_report: str
    max_updates: int = 10000
    warmup_updates: int = 300
    microbatch: int = 1
    accumulation_steps: int = 8
    world_size: int = 4
    effective_batch: int = 32
    backbone_lr: float = 5e-5
    new_lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    network_precision: str = "fp32"
    activation_checkpointing: bool = True
    checkpoint_policy: str = "all"
    talking_backend: str = "conv3d"
    seed: int = 20261021
    train_log_interval: int = 10
    val_interval: int = 250
    checkpoint_interval: int = 250
    keep_recent_checkpoints: int = 2
    extended_validation_metrics: bool = False
    # Historical save_interval now denotes immutable 1000-update milestones.
    save_interval: int = 1000
    val_batch_size: int = 1
    workers: int = 2
    resume_checkpoint: str | None = None
    # Operational pause only: the LR horizon remains max_updates on every stage.
    stop_after_update: int | None = None
    psf_backend: str = "native"
    parent_training_summary: str | None = None
    ema: bool = False
    # Authoritative scientific and implementation identities from runtime.
    experiment_metadata: dict | None = None
    objective: str = "v5_dual"

    def validate(self):
        if self.objective not in ('v5_dual', 'clean_only') or (self.objective == 'clean_only' and self.method != 'D0'):
            raise ValueError('Clean-only is a distinct D0 objective')
        if self.method not in ("D0", "R0", "R0-D0Init"):
            raise ValueError("method must be D0, R0 or R0-D0Init")
        for name in ("max_updates", "microbatch", "accumulation_steps", "world_size",
                     "effective_batch", "val_batch_size", "train_log_interval", "val_interval", "checkpoint_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_updates > 10000:
            raise ValueError("The authorized first-round per-group budget is at most 10000 updates")
        if self.world_size not in (1, 2, 4, 6):
            raise ValueError("Supported accelerator counts are one, two, four or six")
        if self.stop_after_update is not None and (isinstance(self.stop_after_update, bool) or
                not isinstance(self.stop_after_update, int) or not 1 <= self.stop_after_update <= self.max_updates):
            raise ValueError("Stage endpoint must be an absolute update within the fixed LR horizon")
        if self.psf_backend not in ("native", "adjoint"):
            raise ValueError("Unsupported PSF backward implementation")
        if self.microbatch * self.accumulation_steps * self.world_size != self.effective_batch:
            raise ValueError("microbatch * accumulation * world_size must equal effective_batch")
        if self.save_interval != 1000 or self.keep_recent_checkpoints != 2:
            raise ValueError("Keep 1000-update milestone checkpoints and the latest two rolling checkpoints")
        if self.ema:
            raise ValueError("EMA is OFF for this experiment")
        if self.network_precision not in ("fp32", "bf16"):
            raise ValueError("network_precision must be fp32 or bf16; Simulator/loss always FP32")
        if self.talking_backend not in ("conv3d", "conv2d_equivalent"):
            raise ValueError("Unknown Concertormer talking-attention convolution backend")
        if self.checkpoint_policy not in ("all", "highest", "highest_half", "highest_third", "highres", "alternating", "highres_half_low"):
            raise ValueError("Unknown activation checkpoint policy")
        if self.gradient_clip != 1.:
            raise ValueError("global gradient clipping is frozen at 1.0")
        for name in ("backbone_lr", "new_lr", "weight_decay"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.workers < 0 or self.warmup_updates < 0 or self.warmup_updates >= self.max_updates:
            raise ValueError("invalid worker count or warmup budget")
        if not self.initial_checkpoint or not self.readiness_report:
            raise ValueError("initial checkpoint and successful readiness report are required")
        if self.method == "R0-D0Init" and not self.parent_training_summary:
            raise ValueError("R0-D0Init requires the complete D0 selection-run cost summary")


TrainConfig = RunConfig


def scientific_config(config):
    value = asdict(config)
    # The resume path and status artifact's file location are operational, not
    # permission to change method, batching, precision, identities or schedule.
    for key in ("resume_checkpoint", "readiness_report", "stop_after_update",
                "train_log_interval", "val_interval", "checkpoint_interval", "keep_recent_checkpoints"):
        value.pop(key)
    if value.get('extended_validation_metrics') is False:
        value.pop('extended_validation_metrics')
    if value.get('objective') == 'v5_dual':
        value.pop('objective')
    if value.get("psf_backend") == "native":
        value.pop("psf_backend")  # Preserve legacy native checkpoint identity.
    if value.get('experiment_metadata'):
        value['experiment_metadata'].pop('execution_plan_sha256', None)
    return value


def load_initial_model(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Initial checkpoint must be a dictionary")
    state = checkpoint.get("model", checkpoint.get("model_state", checkpoint.get("state_dict")))
    if state is None:
        raise ValueError("Adapted full model state not found in initial checkpoint")
    model.load_state_dict(state, strict=True)
    return checkpoint


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _alias_checkpoint(source, destination):
    # Immutable inode aliases avoid serializing hundreds of MiB three times.
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(*, model, optimizer, scheduler, sampler, config,
                             completed_update, counters, best_psnr, best_update,
                             initial_identity, device, make_best=False, periodic=False, rolling=True):
    rank, world = distributed_info()
    rank_states = gather_rank_state({"rng": capture_rng(device),
                                    "resources": memory_stats(device)})
    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        state = {"schema": "v5_full_training_checkpoint_v1", "method": config.method,
                 "model": _cpu_tree(raw_model.state_dict()),
                 "optimizer": _cpu_tree(optimizer.state_dict()),
                 "scheduler": scheduler.state_dict(),
                 "completed_update": completed_update,
                 "sampler": sampler.state_dict(completed_update),
                 "rank_states": rank_states, "world_size": world,
                 "counters": counters.copy(), "best_psnr": best_psnr,
                 "best_update": best_update, "config": asdict(config),
                 "config_hash": canonical_hash(scientific_config(config)),
                 "initial_identity": initial_identity,
                 "cost_cutoff": "before_this_checkpoint_serialization; see checkpoint_events.jsonl and STATUS.json for complete wall time",
                 "saved_unix_time": time.time()}
        folder = Path(config.output_dir) / "checkpoints"
        latest = folder / "latest.pt"
        atomic_torch_save(state, latest)
        if periodic:
            _alias_checkpoint(latest, folder / f"update_{completed_update:06d}.pt")
        if make_best:
            _alias_checkpoint(latest, folder / "best.pt")
        if rolling:
            _alias_checkpoint(latest, folder / f"rolling_update_{completed_update:06d}.pt")
            for stale in sorted(folder.glob("rolling_update_*.pt"))[:-config.keep_recent_checkpoints]:
                stale.unlink()  # Only our rolling aliases; best/milestones remain linked.
        atomic_json(json_safe({"schema": state["schema"], "completed_update": completed_update,
                              "best_psnr": best_psnr, "best_update": best_update,
                              "latest": str(latest), "periodic": periodic,
                              "made_best": make_best, "counters": counters,
                              "rank_resources": [s["resources"] for s in rank_states],
                              "config_hash": state["config_hash"]}), folder / "LATEST.json")
    if world > 1:
        dist.barrier()


def _finite_every_rank(value, device):
    flag = torch.tensor([1 if bool(value) else 0], dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _validate_readiness(config):
    report = json.loads(Path(config.readiness_report).read_text())
    if report.get("status") not in ("READY", "PASS", "PASSED", "READY_FOR_TRAINING"):
        raise RuntimeError("Automatic full-chain readiness did not pass")
    # The factory/controller owns detailed weight, gradient, finite and memory
    # checks. Bind the pass to the actual adapted initialization and runtime.
    expected = report.get("initial_checkpoint_sha256")
    if expected is None or expected != file_sha256(config.initial_checkpoint):
        raise RuntimeError("Readiness report must bind the actual initial checkpoint SHA256")
    runtime = report.get("runtime", {})
    methods = report.get("validated_methods", [report.get("method")])
    if report.get('objective', 'v5_dual') != config.objective:
        raise RuntimeError('Readiness objective does not match the actual training loss')
    if config.method not in methods:
        raise RuntimeError("Readiness report does not validate this method's complete unrolled path")
    for field in ("microbatch", "world_size", "network_precision", "activation_checkpointing", "talking_backend", "checkpoint_policy"):
        actual = runtime.get(field, "all" if field == "checkpoint_policy" else None)
        if actual != getattr(config, field):
            raise RuntimeError(f"Readiness runtime mismatch or missing field: {field}")
    if config.world_size == 6 or config.psf_backend != "native":
        for field in ("effective_batch", "accumulation_steps", "psf_backend"):
            if runtime.get(field) != getattr(config, field):
                raise RuntimeError(f"Six-card/optimized readiness must bind {field}")
    return report


def train(config: RunConfig, *, model, simulator, train_dataset, val_dataset, device):
    config.validate()
    stage_end = config.stop_after_update or config.max_updates
    rank, world = distributed_info()
    if world != config.world_size:
        raise ValueError("Initialized process group size differs from selected configuration")
    if len(train_dataset) != 8000 or len(val_dataset) != 1000:
        raise ValueError("Formal V5 data must be Train8000 and full Val1000")
    if getattr(train_dataset, "split", "train") != "train" or getattr(val_dataset, "split", "val") not in ("val", "validation"):
        raise ValueError("Incorrect data split supplied to training")
    device = torch.device(device)
    readiness = _validate_readiness(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_hash(scientific_config(config))
    if rank == 0:
        frozen_path = output / "RUN_CONFIG.json"
        if frozen_path.exists() and not config.resume_checkpoint:
            raise RuntimeError("Existing run directory requires explicit resume; refusing to overwrite")
        if not frozen_path.exists():
            atomic_json({"config": asdict(config), "config_hash": config_hash,
                         "readiness_report_sha256": file_sha256(config.readiness_report)}, frozen_path)
    random.seed(config.seed + rank)
    np.random.seed((config.seed + rank) % (2 ** 32))
    torch.manual_seed(config.seed + rank)
    if device.type in ("npu", "cuda"):
        getattr(torch, device.type).manual_seed(config.seed + rank)
    model.to(device)
    if getattr(model, "activation_checkpointing", config.activation_checkpointing) != config.activation_checkpointing:
        raise ValueError("Model activation checkpointing differs from the preflight-selected runtime")
    if getattr(model, "checkpoint_policy", config.checkpoint_policy) != config.checkpoint_policy:
        raise ValueError("Model checkpoint policy differs from the preflight-selected runtime")
    if getattr(model, "talking_backend", config.talking_backend) != config.talking_backend:
        raise ValueError("Model convolution backend differs from the preflight-selected runtime")
    initial = load_initial_model(model, config.initial_checkpoint)
    initial_identity = {"path": config.initial_checkpoint,
                        "sha256": file_sha256(config.initial_checkpoint),
                        "provenance": initial.get("provenance", initial.get("initial_identity")),
                        "parent_completed_update": initial.get("completed_update", 0)}
    parent_counters = {}
    if config.method == "R0-D0Init":
        if initial.get("method") != "D0" or initial.get("best_update") != initial.get("completed_update"):
            raise ValueError("R0-D0Init must start from a D0 full-validation best checkpoint")
        parent_summary = json.loads(Path(config.parent_training_summary).read_text())
        if (parent_summary.get("method") != "D0" or
                not str(parent_summary.get("status", "")).startswith("TRAINING_COMPLETE") or
                parent_summary.get("best_update") != initial.get("completed_update")):
            raise ValueError("D0 must finish selection and identify this full-validation best before R0-D0Init")
        parent_counters = parent_summary["counters"]
        if parent_counters.get("optimizer_updates", -1) != parent_summary.get("completed_update"):
            raise ValueError("Parent completion/cost counters disagree")
        initial_identity["parent_checkpoint_counters"] = initial.get("counters", {})
        initial_identity["parent_selection_run_total_counters"] = parent_counters
        initial_identity["parent_training_summary_sha256"] = file_sha256(config.parent_training_summary)
        initial_identity["optimizer_reset"] = True
    del initial
    if config.objective == 'clean_only' and simulator is not None:
        raise ValueError('Clean-only D0 must not instantiate the Simulator')
    if simulator is not None and any(parameter.requires_grad for parameter in simulator.parameters()):
        raise ValueError("Simulator parameters must be frozen; input derivatives remain enabled")
    groups = model.optimizer_parameter_groups(backbone_lr=config.backbone_lr,
        new_lr=config.new_lr, weight_decay=config.weight_decay)
    parameter_ids = [id(p) for group in groups for p in group["params"]]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != trainable_ids:
        raise ValueError("Optimizer groups must cover each trainable parameter exactly once")
    optimizer = torch.optim.AdamW(groups, lr=config.backbone_lr, weight_decay=config.weight_decay)
    scheduler = UpdateLRScheduler(optimizer, max_updates=config.max_updates,
                                  warmup_updates=config.warmup_updates)
    completed_update = 0
    best_psnr, best_update = -math.inf, None
    counters = {"optimizer_updates": 0, "sample_exposures": 0,
                "network_forward_exposures": 0, "training_seconds": 0.,
                "validation_seconds": 0., "checkpoint_seconds": 0., "run_elapsed_seconds": 0.,
                "parent_optimizer_updates": parent_counters.get("optimizer_updates", 0),
                "parent_run_elapsed_seconds": parent_counters.get("run_elapsed_seconds", 0.)}
    resume_rng = None
    if config.resume_checkpoint:
        state = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=False)
        if state.get("schema") != "v5_full_training_checkpoint_v1" or state["config_hash"] != config_hash:
            raise ValueError("Resume schema or immutable run configuration mismatch")
        if state["initial_identity"]["sha256"] != initial_identity["sha256"] or state["world_size"] != world:
            raise ValueError("Resume initialization or distributed size mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed_update, best_psnr, best_update = state["completed_update"], state["best_psnr"], state["best_update"]
        counters = state["counters"]
        # Checkpoint serialization cannot record its own future duration. The
        # atomic run record preserves that completed cost across a restart.
        event_path = output / "checkpoint_events.jsonl"
        if event_path.exists():
            for line in event_path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A power loss may truncate the final append.
                if event.get("completed_update") == completed_update and event.get("config_hash") == config_hash:
                    counters = event["counters_after_save"]
        resume_rng = state["rank_states"][rank]["rng"]
        if scheduler.last_update != completed_update or state["sampler"]["completed_update"] != completed_update:
            raise ValueError("Resume update/scheduler/sampler counters disagree")
        del state
    if completed_update > stage_end:
        raise ValueError("Cannot resume backwards to an earlier stage endpoint")
    if rank == 0:
        append_jsonl(output / 'INTERVAL_HISTORY.jsonl', {
            'effective_after_update': completed_update, 'train_log_interval': config.train_log_interval,
            'val_interval': config.val_interval, 'checkpoint_interval': config.checkpoint_interval,
            'stage_endpoint': stage_end, 'scheduler_horizon': config.max_updates, 'unix_time': time.time()})
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type != "cpu" else None,
                    broadcast_buffers=False, find_unused_parameters=False)
    sampler = UpdateBatchSampler(len(train_dataset), seed=config.seed,
        microbatch=config.microbatch, accumulation_steps=config.accumulation_steps,
        world_size=world, rank=rank, start_update=completed_update + 1, max_updates=stage_end)
    loader_options = {"num_workers": config.workers, "persistent_workers": config.workers > 0,
                      "generator": torch.Generator().manual_seed(config.seed + 991 + rank)}
    if config.workers:
        loader_options["prefetch_factor"] = 2
        # Do not fork a process after the accelerator runtime was initialized.
        loader_options["multiprocessing_context"] = "spawn"
    loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_options)
    batches = iter(loader)
    if resume_rng:
        restore_rng(resume_rng, device)
    if device.type in ("npu", "cuda"):
        getattr(torch, device.type).reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    prior_elapsed = counters["run_elapsed_seconds"]
    model.train()
    log_window = []
    if rank == 0:
        append_jsonl(output / "metrics.jsonl", {"event": "training_started", "completed_update": completed_update,
            "method": config.method, "config_hash": config_hash, "initial_identity": initial_identity,
            "world_size": world, "effective_batch": config.effective_batch,
            "preflight_updates_counted": 0, "unix_time": time.time()})
    try:
        for update in range(completed_update + 1, stage_end + 1):
            synchronize(device)
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step_for_update(update)
            nfe = 1 if config.method == 'D0' else 4
            clean_only = config.objective == 'clean_only'
            totals = torch.zeros(2 if clean_only else 3 + 2 * nfe, device=device, dtype=torch.float32)
            for micro_index in range(config.accumulation_steps):
                batch = next(batches)
                if not bool((torch.as_tensor(batch.get("G", batch.get("strength", float("nan")))) == 4).all()):
                    raise ValueError("Training data strength must be G=4")
                target = batch["clean"].to(device=device, dtype=torch.float32)
                observation = batch["observation"].to(device=device, dtype=torch.float32)
                if len(target) != config.microbatch:
                    raise ValueError("Incomplete training microbatch would change loss scaling")
                sync_context = model.no_sync() if world > 1 and micro_index + 1 < config.accumulation_steps else nullcontext()
                # Independent reproducible paths for every rank/microbatch.
                path_seed = config.seed + (update - 1) * config.accumulation_steps * world + micro_index * world + rank
                with sync_context:
                    result = unroll_training(model, simulator, observation, target, method=config.method,
                        path_seed=path_seed, network_precision=config.network_precision, objective=config.objective)
                    if not _finite_every_rank(torch.isfinite(result.loss).item(), device):
                        raise FloatingPointError("Nonfinite training loss on at least one rank")
                    (result.loss / config.accumulation_steps).backward()
                components = [result.loss.detach(), result.clean_loss.detach()]
                if not clean_only:
                    components += [result.observation_loss.detach(), *[v.detach() for v in result.stage_clean_losses],
                                   *[v.detach() for v in result.stage_observation_losses]]
                totals += torch.stack(components) / config.accumulation_steps
                del result
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            completed_update = update
            synchronize(device)
            duration = time.monotonic() - update_started
            if world > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                totals /= world
            counters["optimizer_updates"] = update
            counters["sample_exposures"] = update * config.effective_batch
            counters["network_forward_exposures"] = counters["sample_exposures"] * (1 if config.method == "D0" else 4)
            counters["training_seconds"] += duration
            counters["run_elapsed_seconds"] = prior_elapsed + time.monotonic() - wall_started
            counters["cumulative_optimizer_updates_including_parent"] = update + counters["parent_optimizer_updates"]
            counters["cumulative_run_elapsed_seconds_including_parent"] = counters["run_elapsed_seconds"] + counters["parent_run_elapsed_seconds"]
            values = totals.cpu().tolist()
            record = {"event": "optimizer_update", "update": update, "method": config.method,
                      "loss": values[0], "clean_mae": values[1],
                      "grad_norm_before_clip": float(gradient_norm),
                      "learning_rates": [group["lr"] for group in optimizer.param_groups],
                      "elapsed_seconds": duration, "samples_per_second": config.effective_batch / duration,
                      "anomaly": None,
                      "counters": counters.copy(), **memory_stats(device)}
            if not clean_only:
                record.update(observation_mae=values[2], stage_clean_loss=values[3:3+nfe],
                              stage_reprojection_loss=values[3+nfe:])
            if rank == 0:
                atomic_json(json_safe({"status": "TRAINING", **record}), output / "STATUS.json")
                log_window.append(record)
                if update % config.train_log_interval == 0 or update == stage_end:
                    count = len(log_window)
                    logged = {**record, 'event': 'train_log', 'update_start': log_window[0]['update'],
                        'updates_aggregated': count, 'seconds_per_update': sum(x['elapsed_seconds'] for x in log_window) / count,
                        'backbone_lr': record['learning_rates'][0],
                        'gradient_norm_max': max(x['grad_norm_before_clip'] for x in log_window)}
                    for key in ('loss', 'clean_mae', 'grad_norm_before_clip'):
                        logged[key] = sum(x[key] for x in log_window) / count
                    logged.update(total_loss=logged['loss'], clean_loss=logged['clean_mae'])
                    if not clean_only:
                        logged['new_head_lr'] = record['learning_rates'][1]
                        logged['observation_mae'] = sum(x['observation_mae'] for x in log_window) / count
                        logged['reprojection_loss'] = logged['observation_mae']
                        for key in ('stage_clean_loss', 'stage_reprojection_loss'):
                            logged[key] = [sum(x[key][s] for x in log_window) / count for s in range(nfe)]
                    append_jsonl(output / 'metrics.jsonl', logged)
                    log_window.clear()
            make_best = False
            # An early authorized budget ending before its next scheduled 500
            # also receives a complete Val1000, never a surrogate subset.
            if update % config.val_interval == 0 or update == stage_end:
                validation = evaluate_full_validation(model, simulator, val_dataset,
                    device=device, method=config.method, output_dir=output, update=update,
                    network_precision=config.network_precision, batch_size=config.val_batch_size,
                    workers=config.workers, expected_size=1000, extended_metrics=config.extended_validation_metrics,
                    objective=config.objective)
                counters["validation_seconds"] += validation["elapsed_seconds"]
                if validation["psnr"] > best_psnr:
                    best_psnr, best_update = validation["psnr"], update
                    make_best = True
                if rank == 0:
                    append_jsonl(output / "metrics.jsonl", validation)
            if make_best or update % config.checkpoint_interval == 0 or update % config.save_interval == 0 or update == stage_end:
                counters["run_elapsed_seconds"] = prior_elapsed + time.monotonic() - wall_started
                counters["cumulative_run_elapsed_seconds_including_parent"] = counters["run_elapsed_seconds"] + counters["parent_run_elapsed_seconds"]
                checkpoint_started = time.monotonic()
                save_training_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                    sampler=sampler, config=config, completed_update=update, counters=counters,
                    best_psnr=best_psnr, best_update=best_update, initial_identity=initial_identity,
                    device=device, make_best=make_best, periodic=update % config.save_interval == 0 or update == stage_end)
                counters["checkpoint_seconds"] += time.monotonic() - checkpoint_started
                counters["run_elapsed_seconds"] = prior_elapsed + time.monotonic() - wall_started
                counters["cumulative_run_elapsed_seconds_including_parent"] = counters["run_elapsed_seconds"] + counters["parent_run_elapsed_seconds"]
                if rank == 0:
                    append_jsonl(output / "checkpoint_events.jsonl", {
                        "event": "checkpoint_saved", "completed_update": update,
                        "config_hash": config_hash,
                        "checkpoint_elapsed_seconds": time.monotonic() - checkpoint_started,
                        "made_best": make_best, "periodic": update % config.save_interval == 0 or update == stage_end,
                        "counters_after_save": counters.copy()})
            if update in (2000, 5000, 10000) and rank == 0:
                curve = [json.loads(line) for line in (output / 'validation_curve.jsonl').read_text().splitlines() if line.strip()]
                latest_val = curve[-1]
                best_val = next(row for row in curve if row['update'] == best_update)
                milestone = {'method': config.method, 'update': update, 'best_update': best_update,
                    'best_psnr': best_psnr, 'latest_validation': latest_val, 'best_validation': best_val,
                    'counters': counters.copy(), 'scheduler_horizon': config.max_updates,
                    'health': 'NORMAL', 'test_used': False,
                    'next_action': 'continue same group' if update < stage_end else 'handoff to approved next group'}
                atomic_json(json_safe(milestone), output / f'MILESTONE_{update:06d}.json')
                (output / f'MILESTONE_{update:06d}.md').write_text(
                    f"# {config.method} U{update}\n\n"
                    f"Best: U{best_update}, PSNR {best_psnr:.4f} dB; latest: U{latest_val['update']}, {latest_val['psnr']:.4f} dB.\n\n"
                    f"Input gain (best/latest): {best_val.get('psnr_gain_vs_input')} / {latest_val.get('psnr_gain_vs_input')} dB.\n\n"
                    f"Latest stage PSNR: {latest_val.get('stage_psnr')}.\n\n"
                    f"Elapsed active run time: {counters['run_elapsed_seconds']/3600:.2f} hours. Health: NORMAL. "
                    f"Next: {milestone['next_action']}. No manual approval gate.\n", encoding='utf8')
        counters["run_elapsed_seconds"] = prior_elapsed + time.monotonic() - wall_started
        counters["cumulative_run_elapsed_seconds_including_parent"] = counters["run_elapsed_seconds"] + counters["parent_run_elapsed_seconds"]
        if rank == 0:
            summary = {"status": ("STAGE_COMPLETE_HANDOFF_READY" if stage_end < config.max_updates else
                                  "TRAINING_COMPLETE_REFINEMENT_EVALUATION_PENDING"), "method": config.method,
                       "stage_endpoint": stage_end, "scheduler_horizon": config.max_updates,
                       "completed_update": completed_update, "best_update": best_update,
                       "best_psnr": best_psnr, "counters": counters,
                       "best_checkpoint": str(output / "checkpoints" / "best.pt"),
                       "test_used": False, **memory_stats(device)}
            atomic_json(json_safe(summary), output / "STATUS.json")
            atomic_json(json_safe(summary), output / f"STAGE_{stage_end:06d}.json")
            append_jsonl(output / "metrics.jsonl", summary)
        return {"completed_update": completed_update, "best_update": best_update,
                "best_psnr": best_psnr, "counters": counters}
    except BaseException as exc:
        if rank == 0:
            atomic_json({"status": "FAILED", "completed_update": completed_update,
                         "error": repr(exc), "traceback": traceback.format_exc(),
                         "counters": counters, "test_used": False}, output / "STATUS.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--factory", default="runtime.factories:training_components")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    from runtime.authorization import require_formal_training_confirmation
    confirmation = require_formal_training_confirmation()
    config = RunConfig(**json.loads(Path(args.config).read_text()))
    if config.world_size == 6 or config.stop_after_update is not None:
        from runtime.authorization import require_stage_confirmation
        require_stage_confirmation(confirmation, config.stop_after_update, [config.method], worker=True)
        if (config.experiment_metadata or {}).get("physical_cards") != [2, 3, 4, 5, 6, 7] and config.world_size == 6:
            raise RuntimeError("This six-card experiment must use de36 physical cards 2 through 7")
    if ((config.experiment_metadata or {}).get("execution_plan_sha256") !=
            confirmation["approved_execution_plan_sha256"]):
        raise RuntimeError("Training configuration does not bind the execution plan confirmed by the user")
    # Factory configures the device and HCCL, then returns device and components.
    module, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    components = factory(config, args.device)
    train(config, **components)


if __name__ == "__main__":
    main()

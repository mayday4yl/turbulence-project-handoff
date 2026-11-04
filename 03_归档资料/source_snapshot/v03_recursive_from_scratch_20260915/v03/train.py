"""Fresh V0.3 trainer. Run with torchrun -m v03.train --config ... --run-dir ... .

Only --resume accepts weights, and it accepts an identical-protocol V0.3
checkpoint from this run. Pilot weights cannot initialize a formal run.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from .core import RolloutModule, ddp_loss_scale, rank_path_counts


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


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode() + b"\0")
        digest.update(str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone()
                       for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value)


class UpdateScheduler:
    """Explicit update-indexed linear warmup then cosine decay; resume auditable."""
    def __init__(self, optimizer: torch.optim.Optimizer, training: dict) -> None:
        self.optimizer = optimizer
        self.maximum = int(training["maximum_updates"])
        self.warmup = int(training.get("warmup_updates", 0))
        self.initial = float(training["learning_rate"])
        self.final = float(training.get("final_learning_rate", self.initial))
        self.completed_updates = 0

    def lr_at(self, update: int) -> float:
        if self.warmup and update < self.warmup:
            return self.initial * (update + 1) / self.warmup
        fraction = min(1.0, max(0.0, (update - self.warmup) /
                                max(1, self.maximum - self.warmup - 1)))
        return self.final + .5 * (self.initial - self.final) * (1 + math.cos(math.pi * fraction))

    def set_for_update(self, update: int) -> float:
        if update != self.completed_updates:
            raise ValueError("Scheduler/update counter disagreement")
        rate = self.lr_at(update)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def state_dict(self) -> dict:
        return {key: getattr(self, key) for key in
                ("maximum", "warmup", "initial", "final", "completed_updates")}

    def load_state_dict(self, state: dict) -> None:
        for key in ("maximum", "warmup", "initial", "final"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume scheduler protocol mismatch: {key}")
        self.completed_updates = int(state["completed_updates"])


class TaggedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: tuple[int, int, bool]) -> dict:
        sample_id, update, valid = index
        value = dict(self.dataset[(sample_id, update)])
        value.update(sample_id=sample_id, update=update, valid=valid)
        return value


class UpdateBatchSampler(Sampler):
    """All ranks execute equal microstep counts, including an uneven tail.

    A zero-contribution duplicate is only used when a rank has no real paths in
    its last microstep. Its graph lets every rank join the final DDP reduction.
    It is excluded from losses, metrics, exposure counts and sampler progress.
    """
    def __init__(self, start: int, end: int, effective_batch: int,
                 microbatch: int, rank: int, world_size: int,
                 maximum_paths: int | None = None, *, config: dict | None = None,
                 path_selector=None) -> None:
        self.start, self.end = start, end
        self.effective_batch, self.microbatch = effective_batch, microbatch
        self.rank, self.world_size = rank, world_size
        self.maximum_paths = maximum_paths
        self.config = config
        training = config.get("training", {}) if config else {}
        self.balance_paths = bool(training.get("balance_paths_by_length", False))
        budget = training.get("microbatch_prediction_budget")
        self.prediction_budget = int(budget) if budget is not None else None
        if self.prediction_budget is not None and self.prediction_budget <= 0:
            raise ValueError("microbatch_prediction_budget must be positive")
        self.path_selector = path_selector
        self._plans: OrderedDict[int, list] = OrderedDict()

    def global_count(self, update: int) -> int:
        if self.maximum_paths is None:
            return self.effective_batch
        return max(0, min(self.effective_batch,
                          self.maximum_paths - update * self.effective_batch))

    def all_rank_chunks(self, update: int) -> list[list[list[tuple[int, int, bool]]]]:
        """Deterministic layout of exactly the original global sample ID set.

        No sampling occurs in assignment: choose_sample is a stateless lookup of
        the schedule already associated with each existing global sample ID.
        """
        if update in self._plans:
            return self._plans[update]
        counts = rank_path_counts(self.global_count(update), self.world_size)
        base = update * self.effective_batch
        global_ids = list(range(base, base + sum(counts)))
        assignments = [global_ids[sum(counts[:rank]):sum(counts[:rank+1])]
                       for rank in range(self.world_size)]
        metadata = {}
        if self.balance_paths or self.prediction_budget is not None:
            selector = self.path_selector
            if selector is None:
                from .common import choose_sample
                selector = lambda sample_id, index: choose_sample(self.config, sample_id, index)
            metadata = {sample_id: tuple(selector(sample_id, update)["schedule"])
                        for sample_id in global_ids}
            if any(not path for path in metadata.values()):
                raise ValueError("Every selected path must contain a prediction")
            if self.prediction_budget is not None and any(
                    len(path) > self.prediction_budget for path in metadata.values()):
                raise ValueError("One intact path exceeds microbatch_prediction_budget")
        if self.balance_paths:
            assignments = [[] for _ in range(self.world_size)]
            loads = [0] * self.world_size
            for sample_id in sorted(global_ids, key=lambda item: (-len(metadata[item]), item)):
                rank = min(range(self.world_size),
                           key=lambda item: (loads[item], len(assignments[item]), item))
                assignments[rank].append(sample_id)
                loads[rank] += len(metadata[sample_id])
            for assignment in assignments:
                assignment.sort(key=lambda item: (-len(metadata[item]), metadata[item], item))
        result = []
        for assignment in assignments:
            chunks, pending, prediction_count = [], [], 0
            for sample_id in assignment:
                length = len(metadata[sample_id]) if metadata else 0
                exceeds = (self.prediction_budget is not None
                           and prediction_count + length > self.prediction_budget)
                if pending and (len(pending) >= self.microbatch or exceeds):
                    chunks.append(pending)
                    pending, prediction_count = [], 0
                pending.append((sample_id, update, True))
                prediction_count += length
            if pending:
                chunks.append(pending)
            result.append(chunks)
        microsteps = max(map(len, result))
        dummy_id = min(global_ids, key=lambda item: (len(metadata[item]), item)) if metadata else base
        for chunks in result:
            chunks.extend([[(dummy_id, update, False)]] * (microsteps - len(chunks)))
        self._plans[update] = result
        # DataLoader may prefetch future updates. Keep a small bounded cache;
        # recomputation is deterministic if prefetch exceeds this window.
        while len(self._plans) > 16:
            self._plans.popitem(last=False)
        return result

    def chunks(self, update: int) -> list[list[tuple[int, int, bool]]]:
        return self.all_rank_chunks(update)[self.rank]

    def __iter__(self):
        for update in range(self.start, self.end):
            if self.global_count(update) <= 0:
                break
            yield from self.chunks(update)

    def __len__(self) -> int:
        return sum(len(self.chunks(update)) for update in range(self.start, self.end)
                   if self.global_count(update) > 0)


def identity_collate(samples: list[dict]) -> list[dict]:
    return samples


def capture_rng(device: torch.device) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if device.type == "npu":
        state["accelerator"] = torch.npu.get_rng_state(device).cpu()
    elif device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "npu":
        torch.npu.set_rng_state(state["accelerator"], device)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)


def atomic_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def repair_metrics_prefix(path: Path, maximum_update: int) -> dict | None:
    """Keep a byte-exact backup before removing metrics beyond a resumed state."""
    if not path.exists():
        return None
    original = path.read_bytes()
    rows = original.splitlines()
    latest: dict[int, dict] = {}
    for index, line in enumerate(rows):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(rows) - 1:
                raise
            # A killed process can leave one incomplete trailing append.
            continue
        update = int(row["update"])
        if update <= maximum_update:
            latest[update] = row
    prefix = b"".join((json.dumps(latest[update], sort_keys=True) + "\n").encode()
                       for update in sorted(latest))
    backup = path.with_name(f"{path.stem}.before-resume-{time.time_ns()}.jsonl")
    with backup.open("xb") as stream:
        stream.write(original)
    temporary = path.with_name(path.name + ".resume.tmp")
    temporary.write_bytes(prefix)
    temporary.replace(path)
    return {"path": str(path), "backup": str(backup),
            "maximum_update": maximum_update, "retained_updates": len(latest)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int,
                        help="Stop at this absolute completed update, preserving full LR horizon")
    parser.add_argument("--mode", choices=("benchmark", "pilot", "formal", "acceptance"),
                        required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    training = config["training"]
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    sys.path.insert(0, str(config["base_source"]))
    from .common import (TrainingDataset, append_jsonl, atomic_json, build_model,
                         build_simulator, seed_all, verify_base)
    from physical_bridge.data import evenly_spaced_schedule
    import torch_npu  # noqa: F401

    verify_base(config)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(int(training.get("cpu_threads", 4)))
    if distributed:
        dist.init_process_group("hccl", timeout=dt.timedelta(minutes=30))
    run_dir = args.run_dir.resolve()
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    if args.mode == "formal" and world_size != 4:
        raise ValueError("The authorized formal run uses exactly four NPU ranks")
    seed_all(int(config["seed"]))
    network = build_model(config, device)
    initial_hash = state_digest(network.state_dict())
    simulator = build_simulator(config, device)
    module = RolloutModule(network, simulator, float(training["lambda_R"]),
                           network_bfloat16=bool(training.get("network_bfloat16", True)))
    wrapped = (DistributedDataParallel(module, device_ids=[local_rank],
                                       broadcast_buffers=False,
                                       find_unused_parameters=False)
               if distributed else module)
    optimizer = torch.optim.AdamW(network.parameters(),
                                 lr=float(training["learning_rate"]),
                                 betas=tuple(training.get("betas", (.9, .99))),
                                 weight_decay=float(training.get("weight_decay", 1e-4)))
    scheduler = UpdateScheduler(optimizer, training)
    ema = EMA(network, float(training.get("ema_decay", .999)))
    start = 0
    resumed_rng = None
    cumulative_paths = cumulative_predictions = 0
    cumulative_compute_seconds = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        required = {"format": "v03_recursive_from_scratch", "config_hash": config_hash,
                    "run_dir": str(run_dir), "mode": args.mode, "world_size": world_size}
        for key, expected in required.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Forbidden/incompatible resume: {key}")
        network.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.shadow = {key: value.to(device) for key, value in checkpoint["ema"].items()}
        if float(checkpoint["ema_decay"]) != ema.decay:
            raise ValueError("EMA decay changed on resume")
        start = int(checkpoint["update"])
        if start != scheduler.completed_updates:
            raise ValueError("Resume update/scheduler disagreement")
        expected_position = min(start * int(training["effective_batch"]),
                                int(training.get("maximum_paths", 2**63 - 1)))
        if checkpoint["sampler"]["next_global_sample_id"] != expected_position:
            raise ValueError("Resume sampler position disagreement")
        if checkpoint["curriculum_counter"] != start:
            raise ValueError("Resume curriculum counter disagreement")
        resumed_rng = checkpoint["rng_by_rank"][rank]
        cumulative_paths = int(checkpoint["counters"]["paths"])
        cumulative_predictions = int(checkpoint["counters"]["predictions"])
        cumulative_compute_seconds = float(checkpoint["counters"]["compute_seconds"])
        if checkpoint["initial_model_hash"] != initial_hash:
            raise ValueError("Random initialization identity changed")
        del checkpoint
    elif (run_dir / "identity.json").exists():
        raise FileExistsError("Existing run requires explicit identical-run resume")

    effective_batch = int(training["effective_batch"])
    microbatch = int(training["microbatch"])
    if min(effective_batch, microbatch) <= 0:
        raise ValueError("Positive effective batch and microbatch required")
    end = min(int(training["maximum_updates"]), args.stop_after or 2**63 - 1)
    if not start < end:
        raise ValueError("No updates remain in the requested segment")
    sampler = UpdateBatchSampler(start, end, effective_batch, microbatch, rank,
                                 world_size, training.get("maximum_paths"), config=config)
    dataset = TaggedDataset(TrainingDataset(config))
    workers = int(training.get("num_workers", 2))
    loader_generator = torch.Generator().manual_seed(int(config["seed"]) + 900000 + rank)
    loader_kwargs = dict(dataset=dataset, batch_sampler=sampler,
                         collate_fn=identity_collate, num_workers=workers,
                         generator=loader_generator, persistent_workers=workers > 0)
    if workers:
        loader_kwargs.update(prefetch_factor=int(training.get("prefetch_factor", 2)),
                             multiprocessing_context="spawn")
    loader = DataLoader(**loader_kwargs)
    iterator = iter(loader)
    # Loader worker seeding is isolated from model RNG. Restore after all setup.
    if resumed_rng is not None:
        restore_rng(resumed_rng, device)
    grid = (.75, 1.5, 2.25, 3., 3.75, 4.5)
    path_tuples = sorted({tuple(evenly_spaced_schedule(g, k))
                          for g in grid for k in (0, 1, 3, 5)})
    path_index = {path: index for index, path in enumerate(path_tuples)}
    identity = {"format": "v03_recursive_from_scratch", "config_hash": config_hash,
                "initial_model_hash": initial_hash, "seed": config["seed"],
                "sampling_seed": config["sampling_seed"], "mode": args.mode,
                "run_dir": str(run_dir), "world_size": world_size,
                "physical_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "microbatch": microbatch, "effective_batch": effective_batch,
                "balance_paths_by_length": sampler.balance_paths,
                "microbatch_prediction_budget": sampler.prediction_budget,
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "source_model": str(sys.modules[type(network).__module__].__file__),
                "source_simulator": str(sys.modules[type(simulator).__module__].__file__)}
    if rank == 0:
        if not args.resume:
            atomic_json(run_dir / "identity.json", identity)
            atomic_json(run_dir / "frozen_config.json", config)
        else:
            for metric_name in ("train_metrics.jsonl", "timing.jsonl"):
                repair = repair_metrics_prefix(run_dir / metric_name, start)
                if repair:
                    append_jsonl(run_dir / "events.jsonl",
                                 {"event": "resume_metrics_prefix", **repair})
        append_jsonl(run_dir / "events.jsonl", {"event": "segment_start",
                     "time": time.time(), "start_update": start, "end_update": end,
                     "resume": str(args.resume) if args.resume else None})

    def save(update: int) -> None:
        synchronization_check = None
        if args.mode == "acceptance":
            local_digest = {"rank": rank, "model": state_digest(network.state_dict()),
                            "ema": state_digest(ema.shadow)}
            rank_digests: list[Any] = [None] * world_size
            if distributed:
                dist.all_gather_object(rank_digests, local_digest)
            else:
                rank_digests[0] = local_digest
            synchronization_check = {
                "update": update, "world_size": world_size,
                "rank_digests": rank_digests,
                "model_equal": len({item["model"] for item in rank_digests}) == 1,
                "ema_equal": len({item["ema"] for item in rank_digests}) == 1}
            if rank == 0:
                atomic_json(run_dir / "acceptance" / f"checkpoint_sync_{update:06d}.json",
                            synchronization_check)
            if not (synchronization_check["model_equal"] and synchronization_check["ema_equal"]):
                raise AssertionError("DDP acceptance failed: model/EMA differ across ranks")
        rng = capture_rng(device)
        rng_states: list[Any] = [None] * world_size
        if distributed:
            dist.all_gather_object(rng_states, rng)
        else:
            rng_states[0] = rng
        if rank == 0:
            state = {**identity, "model": cpu_tree(network.state_dict()),
                     "ema": cpu_tree(ema.shadow), "ema_decay": ema.decay,
                     "optimizer": cpu_tree(optimizer.state_dict()),
                     "scheduler": scheduler.state_dict(), "update": update,
                     "acceptance_rank_state_digests": synchronization_check,
                     "rng_by_rank": rng_states, "curriculum_counter": update,
                     "sampler": {"next_global_sample_id": cumulative_paths,
                                 "sampling_seed": config["sampling_seed"]},
                     "config": config, "model_config": config["model"],
                     "counters": {"paths": cumulative_paths,
                                  "predictions": cumulative_predictions,
                                  "compute_seconds": cumulative_compute_seconds}}
            path = run_dir / "checkpoints" / f"update-{update:06d}.pt"
            atomic_checkpoint(path, state)
            atomic_json(run_dir / "latest_checkpoint.json",
                        {"path": str(path), "update": update})
        if distributed:
            dist.barrier()

    if training.get("save_initial_checkpoint", False) and not args.resume:
        save(0)
    module.train()
    segment_start = time.monotonic()
    segment_start_paths = cumulative_paths
    segment_start_predictions = cumulative_predictions
    diagnostic_interval = int(training.get("diagnostic_interval",
                                           1 if args.mode == "acceptance" else 50))
    if diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be positive")
    update = start
    while update < end and sampler.global_count(update) > 0:
        torch.npu.synchronize(device)
        torch.npu.reset_peak_memory_stats(device)
        step_start = time.monotonic()
        rate = scheduler.set_for_update(update)
        optimizer.zero_grad(set_to_none=True)
        chunks = sampler.chunks(update)
        global_paths = sampler.global_count(update)
        # 3 path-level loss sums, 18 step sums/counts, histogram of exact paths.
        totals = torch.zeros(21 + len(path_tuples), device=device, dtype=torch.float32)
        loaded_microbatches = []
        for microstep in range(len(chunks)):
            samples = next(iterator)
            if any(sample["update"] != update for sample in samples):
                raise AssertionError("Data loader crossed an optimizer-update boundary")
            if [(sample["sample_id"], sample["update"], sample["valid"]) for sample in samples] != chunks[microstep]:
                raise AssertionError("Loaded sample IDs differ from deterministic rank assignment")
            clean = torch.stack([sample["C"] for sample in samples]).to(device)
            observation = torch.stack([sample["Y"] for sample in samples]).to(device)
            severities = torch.tensor([float(sample["G"]) for sample in samples],
                                      device=device, dtype=torch.float32)
            schedules = [tuple(sample["schedule"]) for sample in samples]
            loaded_microbatches.append([
                len(samples), sum(int(sample["valid"]) for sample in samples),
                sum(map(len, schedules)),
                sum(len(path) for sample, path in zip(samples, schedules, strict=True)
                    if sample["valid"])])
            if sampler.prediction_budget is not None and loaded_microbatches[-1][2] > sampler.prediction_budget:
                raise AssertionError("Loaded microbatch exceeds the frozen prediction budget")
            valid = torch.tensor([float(sample["valid"]) for sample in samples],
                                 device=device, dtype=torch.float32)
            sync_context = (wrapped.no_sync() if distributed and microstep < len(chunks) - 1
                            else nullcontext())
            with sync_context:
                output = wrapped(clean, observation, severities, schedules, valid)
                finite = torch.isfinite(output["loss_sum"]).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite.item()):
                    raise FloatingPointError(f"Nonfinite loss at update {update}")
                (output["loss_sum"] * ddp_loss_scale(global_paths, world_size)).backward()
            totals[0] += output["clean_loss_sum"].detach()
            totals[1] += output["reconstruction_loss_sum"].detach()
            totals[2] += output["prediction_count"].detach()
            for entry in output["steps"]:
                offset = 3 + entry["step"] * 3
                totals[offset] += entry["clean_sum"]
                totals[offset + 1] += entry["reconstruction_sum"]
                totals[offset + 2] += entry["path_count"]
            histogram = [0.] * len(path_tuples)
            for sample, path in zip(samples, schedules, strict=True):
                if sample["valid"]:
                    histogram[path_index[path]] += 1
            totals[21:] += torch.tensor(histogram, device=device)
            del output, clean, observation, severities, valid
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(), float(training.get("grad_clip", 1.)),
            error_if_nonfinite=True)
        branch_norms = {}
        diagnostic_due = update % diagnostic_interval == 0
        if diagnostic_due and hasattr(network, "branch_parameters"):
            for name, parameters in network.branch_parameters().items():
                norms = [parameter.grad.detach().float().norm()
                         for parameter in parameters if parameter.grad is not None]
                branch_norms[name] = float(torch.stack(norms).norm().item()) if norms else 0.
        optimizer.step()
        ema.update(network)
        scheduler.completed_updates += 1
        update += 1
        if distributed:
            dist.all_reduce(totals)
        local_layout = torch.tensor(loaded_microbatches, device=device, dtype=torch.int64)
        layout_by_rank = [torch.empty_like(local_layout) for _ in range(world_size)]
        if distributed:
            dist.all_gather(layout_by_rank, local_layout)
        else:
            layout_by_rank[0] = local_layout
        torch.npu.synchronize(device)
        local_seconds = time.monotonic() - step_start
        local_runtime = torch.tensor(local_seconds, device=device, dtype=torch.float32)
        peak_memory = torch.tensor([torch.npu.max_memory_allocated(device),
                                    torch.npu.max_memory_reserved(device)],
                                   device=device, dtype=torch.int64)
        if distributed:
            dist.all_reduce(local_runtime, op=dist.ReduceOp.MAX)
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        seconds = float(local_runtime.item())
        allocated, reserved = peak_memory.cpu().tolist()
        values = totals.cpu().tolist()
        actual_layout = [value.cpu().tolist() for value in layout_by_rank]
        if round(sum(values[21:])) != global_paths:
            raise AssertionError("Global path normalization/count disagreement")
        predictions = round(values[2])
        cumulative_paths += global_paths
        cumulative_predictions += predictions
        cumulative_compute_seconds += seconds
        record = {"update": update, "time": time.time(),
                  "clean_loss": values[0] / global_paths,
                  "reconstruction_loss": values[1] / global_paths,
                  "loss": (values[0] + float(training["lambda_R"]) * values[1]) / global_paths,
                  "lambda_R": training["lambda_R"], "learning_rate": rate,
                  "gradient_norm_before_clip": float(gradient_norm.item()),
                  "branch_gradient_norms_after_clip": branch_norms,
                  "branch_diagnostics_executed": diagnostic_due,
                  "actual_global_paths": global_paths, "predictions": predictions,
                  "microbatch": microbatch, "microsteps": len(chunks),
                  "microbatch_is_configured_upper_bound": True,
                  "balance_paths_by_length": sampler.balance_paths,
                  "microbatch_prediction_budget": sampler.prediction_budget,
                  "actual_rank_microbatches": [
                      {"rank": rank_index, "chunks": [
                          {"microstep": microstep,
                           "physical_paths_including_dummy": values[0],
                           "real_paths": values[1],
                           "physical_predictions_including_dummy": values[2],
                           "real_predictions": values[3],
                           "real_sample_ids": [sample_id for sample_id, _, valid in
                                               sampler.all_rank_chunks(update - 1)[rank_index][microstep]
                                               if valid]}
                          for microstep, values in enumerate(rank_values)]}
                      for rank_index, rank_values in enumerate(actual_layout)],
                  "world_size": world_size, "step_seconds": seconds,
                  "step_seconds_scope": "data_load_through_optimizer_ema_and_loss_reduction; excludes reporting_and_checkpoint_io",
                  "trajectories_per_second": global_paths / seconds,
                  "predictions_per_second": predictions / seconds,
                  "updates_per_second": 1 / seconds,
                  "peak_allocated_bytes_max_rank": int(allocated),
                  "peak_reserved_bytes_max_rank": int(reserved),
                  "cumulative_paths": cumulative_paths,
                  "cumulative_predictions": cumulative_predictions,
                  "compute_npu_hours": cumulative_compute_seconds * world_size / 3600,
                  "segment_wall_seconds": time.monotonic() - segment_start,
                  "steps": [{"step": k, "paths": round(values[3 + 3*k + 2]),
                             "clean_mse": values[3 + 3*k] / values[3 + 3*k + 2],
                             "reconstruction_mse": values[3 + 3*k + 1] / values[3 + 3*k + 2]}
                            for k in range(6) if values[3 + 3*k + 2] > 0],
                  "actual_path_counts": {str(path): round(values[21+i])
                                         for i, path in enumerate(path_tuples)
                                         if values[21+i] > 0}}
        if rank == 0:
            append_jsonl(run_dir / "train_metrics.jsonl", record)
            atomic_json(run_dir / "progress.json", record)
            print(json.dumps(record, sort_keys=True), flush=True)
        due = update % int(training.get("checkpoint_interval", 1000)) == 0
        if due or update == end or sampler.global_count(update) <= 0:
            save(update)
        if rank == 0:
            # Includes data input, all compute/collectives, CPU summaries, normal
            # JSON logs, progress writes and any checkpoint at this update.
            # Only this small timing-record append itself follows the timestamp.
            loop_seconds = time.monotonic() - step_start
            append_jsonl(run_dir / "timing.jsonl", {
                "update": update, "time": time.time(),
                "full_update_wall_seconds": loop_seconds,
                "actual_global_paths": global_paths, "predictions": predictions,
                "trajectories_per_second": global_paths / loop_seconds,
                "predictions_per_second": predictions / loop_seconds,
                "optimizer_updates_per_second": 1 / loop_seconds,
                "checkpoint_due": due or update == end or sampler.global_count(update) <= 0,
                "scope": "full_update_including_reporting_and_checkpoint_before_this_timing_append"})
    if rank == 0:
        segment_seconds = time.monotonic() - segment_start
        atomic_json(run_dir / "segment_complete.json", {
            "status": "completed", "update": update,
            "formal_budget_complete": update == int(training["maximum_updates"]),
            "config_hash": config_hash, "paths": cumulative_paths,
            "predictions": cumulative_predictions,
            "segment_wall_seconds": segment_seconds,
            "segment_sampled_paths": cumulative_paths - segment_start_paths,
            "segment_predictions": cumulative_predictions - segment_start_predictions,
            "segment_trajectories_per_second": (cumulative_paths - segment_start_paths) / segment_seconds,
            "segment_predictions_per_second": (cumulative_predictions - segment_start_predictions) / segment_seconds,
            "segment_updates_per_second": (update - start) / segment_seconds,
            "segment_wall_npu_hours": segment_seconds * world_size / 3600})
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

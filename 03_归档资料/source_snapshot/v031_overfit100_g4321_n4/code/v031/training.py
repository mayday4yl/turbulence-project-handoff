"""Bounded FP32 V0.31 optimizer updates and exact distributed boundary state.

The launcher owns resource permission and stage gates. This module never chooses
devices, changes the method, retries a failed update, or performs evaluation.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import (_cpu_snapshot, _identity, _validate_fp32_training_state,
                         capture_rng_state, restore_rng_state)
from .core import RolloutModule, ddp_loss_scale, severity_path
from .data import collate_starts
from .scheduler import UpdateIndexedScheduler


FORMAT = "v031_distributed_update_boundary_v1"
EMA_DECAY = 0.996
GLOBAL_BATCH = 32


def authorized_update_limit(authorization: Mapping[str, Any] | None = None) -> int:
    """Current limit is 1000; future extensions require an explicit audit record.

    This is an execution boundary, never part of the fixed learning-rate curve.
    Passing a record is an API assertion by the authorizing launcher, not a way
    for this engine to infer permission from elapsed time or checkpoint state.
    """
    if authorization is None:
        return 1000
    boundary = authorization.get("authorized_run_until_update")
    reference = authorization.get("authorization_id")
    if (authorization.get("explicit_user_authorization") is not True
            or type(boundary) is not int or boundary <= 1000
            or not isinstance(reference, str) or not reference.strip()):
        raise ValueError("Future continuation requires an explicit user authorization record")
    _identity(authorization)
    return boundary


class GlobalStartSampler:
    """One dedicated CPU RNG, one global permutation, no replacement or drops.

    All ranks independently replay this same sampler and take their fixed slice
    of the global batch. The formal 1600/32 stream has exactly 50 updates/epoch.
    A smaller pressure-test population may cross an epoch within a batch.
    """
    def __init__(self, start_ids: Sequence[str], *, seed: int = 20260916,
                 batch_size: int = GLOBAL_BATCH) -> None:
        self.start_ids = tuple(start_ids)
        if not self.start_ids or len(set(self.start_ids)) != len(self.start_ids):
            raise ValueError("Sampler requires nonempty unique start IDs")
        if batch_size < 1:
            raise ValueError("Positive global batch required")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.epoch = 0
        self.position = 0
        self.consumed = 0
        self.order = torch.randperm(len(self.start_ids), generator=self.generator).tolist()
        self.start_ids_sha256 = hashlib.sha256(
            json.dumps(self.start_ids, ensure_ascii=False).encode()).hexdigest()

    def next_indices(self) -> list[int]:
        chosen: list[int] = []
        while len(chosen) < self.batch_size:
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self.order = torch.randperm(len(self.start_ids), generator=self.generator).tolist()
            take = min(self.batch_size - len(chosen), len(self.order) - self.position)
            chosen.extend(self.order[self.position:self.position + take])
            self.position += take
        self.consumed += self.batch_size
        return chosen

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "seed": self.seed, "batch_size": self.batch_size,
                "population": len(self.start_ids), "start_ids_sha256": self.start_ids_sha256,
                "epoch": self.epoch, "position": self.position, "consumed": self.consumed,
                "order": list(self.order), "rng": self.generator.get_state().clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {"version": 1, "seed": self.seed, "batch_size": self.batch_size,
                    "population": len(self.start_ids), "start_ids_sha256": self.start_ids_sha256}
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Sampler identity mismatch")
        n = len(self.start_ids)
        order = list(state["order"])
        epoch, position, consumed = state["epoch"], state["position"], state["consumed"]
        if (any(type(v) is not int for v in [epoch, position, consumed]) or epoch < 0
                or not 0 <= position <= n or consumed != epoch * n + position
                or consumed % self.batch_size or sorted(order) != list(range(n))):
            raise ValueError("Invalid sampler permutation/counters")
        trial = torch.Generator(device="cpu")
        trial.set_state(state["rng"].cpu())
        self.order, self.epoch, self.position, self.consumed = order, epoch, position, consumed
        self.generator.set_state(trial.get_state())


def parameter_groups(network: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bias and normalization parameters get zero decay; all others get 1e-4."""
    normalization_types = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm,
                           nn.GroupNorm, nn.modules.instancenorm._InstanceNorm)
    if hasattr(nn, "RMSNorm"):
        normalization_types = normalization_types + (nn.RMSNorm,)
    no_decay_ids: set[int] = set()
    for module in network.modules():
        if isinstance(module, normalization_types):
            no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias":
                no_decay_ids.add(id(parameter))
    buckets: dict[str, list[tuple[str, nn.Parameter]]] = {"decay": [], "no_decay": []}
    for name, parameter in sorted(network.named_parameters()):
        if parameter.requires_grad:
            if parameter.dtype != torch.float32:
                raise ValueError(f"Non-FP32 master parameter: {name}")
            buckets["no_decay" if id(parameter) in no_decay_ids else "decay"].append((name, parameter))
    if not any(buckets.values()):
        raise ValueError("No trainable network parameters")
    groups, report = [], {}
    for label, decay in [("decay", 1e-4), ("no_decay", 0.0)]:
        pairs = buckets[label]
        if not pairs:
            continue
        groups.append({"params": [p for _, p in pairs], "weight_decay": decay})
        report[label] = {"weight_decay": decay, "names": [name for name, _ in pairs],
                         "tensor_count": len(pairs), "parameter_count": sum(p.numel() for _, p in pairs)}
    grouped = [id(p) for group in groups for p in group["params"]]
    expected = [id(p) for p in network.parameters() if p.requires_grad]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(expected):
        raise AssertionError("Parameter groups omit or duplicate trainable parameters")
    report["rule"] = "bias_and_normalization_no_decay; all_groups_same_scheduler_lr"
    report["total_trainable_parameters"] = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return groups, report


class FP32EMA:
    """EMA applies to parameters; every buffer is copied after each update."""
    def __init__(self, network: nn.Module) -> None:
        self.parameter_names = tuple(name for name, _ in network.named_parameters())
        self.network = {name: value.detach().clone() for name, value in network.state_dict().items()}
        self.updates = 0
        self._validate()

    def _validate(self) -> None:
        for name, value in self.network.items():
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError(f"EMA requires FP32 floating state: {name}")

    @torch.no_grad()
    def update(self, network: nn.Module) -> None:
        state = network.state_dict()
        if state.keys() != self.network.keys():
            raise ValueError("EMA/network state keys changed")
        parameters = set(self.parameter_names)
        for name, value in state.items():
            if name in parameters:
                self.network[name].mul_(EMA_DECAY).add_(value.detach(), alpha=1.0 - EMA_DECAY)
            else:
                self.network[name].copy_(value.detach())
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {"decay": EMA_DECAY, "buffer_rule": "copy_after_each_update",
                "parameter_names": list(self.parameter_names), "updates": self.updates,
                "network": self.network}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (state.get("decay") != EMA_DECAY or state.get("buffer_rule") != "copy_after_each_update"
                or tuple(state["parameter_names"]) != self.parameter_names
                or state["network"].keys() != self.network.keys()
                or type(state["updates"]) is not int or state["updates"] < 0):
            raise ValueError("EMA identity mismatch")
        for name, value in state["network"].items():
            receiving = self.network[name]
            if value.dtype != receiving.dtype or value.shape != receiving.shape:
                raise ValueError(f"EMA tensor mismatch: {name}")
        for name, value in state["network"].items():
            self.network[name].copy_(value)
        self.updates = state["updates"]
        self._validate()


def empty_counts() -> dict[str, Any]:
    return {"paths": 0, "predictions": 0, "true_inputs": 0, "predicted_inputs": 0,
            "by_initial_G": {str(g): 0 for g in [4, 3, 2, 1]},
            "by_current_g": {str(g): {"predictions": 0, "true_inputs": 0, "predicted_inputs": 0}
                             for g in [4, 3, 2, 1]}}


def advance_counts(counts: dict[str, Any], severities: Sequence[int]) -> None:
    for initial in severities:
        path = severity_path(initial)
        counts["paths"] += 1
        counts["predictions"] += len(path)
        counts["true_inputs"] += 1
        counts["predicted_inputs"] += len(path) - 1
        counts["by_initial_G"][str(initial)] += 1
        for step, g in enumerate(path):
            item = counts["by_current_g"][str(g)]
            item["predictions"] += 1
            item["true_inputs" if step == 0 else "predicted_inputs"] += 1


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device) -> dict[str, int]:
    backend = getattr(torch, device.type, None)
    if device.type not in {"npu", "cuda"}:
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0}
    return {"allocated_bytes": int(backend.memory_allocated(device)),
            "reserved_bytes": int(backend.memory_reserved(device)),
            "peak_allocated_bytes": int(backend.max_memory_allocated(device)),
            "peak_reserved_bytes": int(backend.max_memory_reserved(device))}


class TrainingEngine:
    """A real optimizer update always consumes a complete global batch of 32.

    Production world_size is four. A world_size-one CPU fixture is allowed for
    semantics tests. There is one DDP wrapper around the complete rollout and
    no_sync encloses BOTH forward and backward on nonfinal microbatches.
    """
    def __init__(self, network: nn.Module, simulator: nn.Module, dataset: Any, *,
                 device: torch.device | str, microbatch: int, rank: int = 0,
                 world_size: int = 1, seed: int = 20260916,
                 identity: Mapping[str, Any], formal: bool = False) -> None:
        self.device = torch.device(device)
        self.rank, self.world_size, self.microbatch = rank, world_size, microbatch
        if world_size not in {1, 4} or not 0 <= rank < world_size:
            raise ValueError("Only four production ranks or one CPU fixture rank supported")
        if formal and world_size != 4:
            raise ValueError("Formal training requires four ranks")
        self.local_paths = GLOBAL_BATCH // world_size
        if microbatch not in {1, 2, 4, 8} or self.local_paths % microbatch:
            raise ValueError("Unsupported effective-batch factorization")
        self.accumulation = self.local_paths // microbatch
        self.dataset, self.network, self.formal = dataset, network, formal
        self.identity = _identity(identity)
        self.records = dataset.records
        if formal and len(self.records) != 1600:
            raise ValueError("Formal training requires all 1600 starts")
        self.sampler = GlobalStartSampler([r["start_id"] for r in self.records], seed=seed)
        self.counts = empty_counts()
        self.at_boundary = True
        self.poisoned = False
        self.optimizer_step_calls = 0
        self.optimizer_step_attempts = 0
        self.batch_dependent_layers = [name for name, module in network.named_modules()
                                       if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        if self.batch_dependent_layers:
            raise ValueError(f"Batch-dependent layers require prior semantics approval: {self.batch_dependent_layers}")
        self.rollout = RolloutModule(network, simulator, 0.2, network_bfloat16=False).to(self.device)
        self.rollout.train()
        if world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != world_size or dist.get_rank() != rank:
                raise RuntimeError("Process group topology disagrees with engine")
            kwargs = {"device_ids": [self.device.index]} if self.device.type != "cpu" else {}
            self.module = DistributedDataParallel(self.rollout, broadcast_buffers=False,
                                                   find_unused_parameters=False, **kwargs)
        else:
            self.module = self.rollout
        groups, self.group_report = parameter_groups(network)
        self.optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                                           weight_decay=1e-4, foreach=False)
        self.scheduler = UpdateIndexedScheduler(self.optimizer)
        self.ema = FP32EMA(network)

    @property
    def completed_updates(self) -> int:
        return self.scheduler.completed_updates

    def _all_reduce(self, value: torch.Tensor, op: Any = None) -> None:
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM if op is None else op)

    def update(self) -> dict[str, Any]:
        if self.poisoned or not self.at_boundary:
            raise RuntimeError("An incomplete/failed update must never be retried in-place")
        self.at_boundary = False
        try:
            result = self._update()
        except BaseException:
            self.poisoned = True
            raise
        self.at_boundary = True
        return result

    def _update(self) -> dict[str, Any]:
        synchronize(self.device)
        start_time = time.perf_counter()
        indices = self.sampler.next_indices()
        global_records = [self.records[i] for i in indices]
        local_indices = indices[self.rank * self.local_paths:(self.rank + 1) * self.local_paths]
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.assert_optimizer_lr()
        used_lr = self.scheduler.next_lr
        # [original G - 1, step, LC sum / LR sum / path count]. No padded steps
        # contribute. Each rank aggregates detached metrics without extra renders.
        metric_values = [[[0.0, 0.0, 0.0] for _ in range(4)] for _ in range(4)]
        path_sums = torch.zeros(3, dtype=torch.float32, device=self.device)
        for offset in range(0, self.local_paths, self.microbatch):
            batch = collate_starts([self.dataset[i] for i in local_indices[offset:offset + self.microbatch]])
            clean = batch["clean"].to(self.device, non_blocking=False)
            observation = batch["observation"].to(self.device, non_blocking=False)
            gs = batch["G"].to(self.device, non_blocking=False)
            final_microbatch = offset + self.microbatch == self.local_paths
            context = (self.module.no_sync() if self.world_size > 1 and not final_microbatch
                       else nullcontext())
            with context:
                output = self.module(clean, observation, gs, batch["severity_path"],
                                     collect_step_metrics=True)
                if not torch.isfinite(output["path_losses"]).all().item():
                    raise FloatingPointError("Nonfinite path loss; no optimizer update committed")
                loss = output["loss_sum"] * ddp_loss_scale(GLOBAL_BATCH, self.world_size)
                loss.backward()
            with torch.no_grad():
                path_sums += torch.stack([output["loss_sum"].detach(), output["clean_loss_sum"].detach(),
                                          output["reconstruction_loss_sum"].detach()])
                for entry in output["step_metrics"]:
                    detached = torch.stack([entry["clean_losses"], entry["reconstruction_losses"]], 1).cpu().tolist()
                    for local, path_index in enumerate(entry["active_path_indices"]):
                        original = int(batch["G"][path_index].item())
                        slot = metric_values[original - 1][entry["step"] - 1]
                        slot[0] += detached[local][0]
                        slot[1] += detached[local][1]
                        slot[2] += 1
            del output, loss, clean, observation, gs, batch
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        missing = [name for name, p in self.network.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError(f"Missing trainable gradients: {missing}")
        # DDP already averaged rank gradients once, after all local sums. Do not
        # all-reduce their norm again: that would introduce a sqrt(world_size).
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
        preclip = float(norm.item())
        if not math.isfinite(preclip):
            raise FloatingPointError("Nonfinite averaged gradient norm")
        self.optimizer_step_attempts += 1
        self.optimizer.step()
        self.optimizer_step_calls += 1
        if not torch.stack([torch.isfinite(p).all() for p in parameters]).all().item():
            raise FloatingPointError("Nonfinite parameter after update")
        self.ema.update(self.network)
        self.scheduler.mark_update_completed()
        self._all_reduce(path_sums)
        metrics = torch.tensor(metric_values, dtype=torch.float32, device=self.device)
        self._all_reduce(metrics)
        synchronize(self.device)
        elapsed = time.perf_counter() - start_time
        # Slowest rank determines real distributed throughput, including I/O,
        # synchronization, optimizer, EMA, and production diagnostics above.
        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float32, device=self.device)
        self._all_reduce(elapsed_tensor, dist.ReduceOp.MAX)
        severities = [int(record["initial_G"]) for record in global_records]
        update_counts = empty_counts()
        advance_counts(update_counts, severities)
        advance_counts(self.counts, severities)
        values = metrics.cpu().tolist()
        per_step = []
        for g in [4, 3, 2, 1]:
            for step in range(g):
                lc, lr, count = values[g - 1][step]
                if count:
                    per_step.append({"initial_G": g, "K": step + 1, "current_g": g - step,
                                     "paths": int(count), "LC_mean": lc / count, "LR_mean": lr / count,
                                     "per_sample_LC_weight": 1.0 / (GLOBAL_BATCH * g),
                                     "per_sample_LR_weight": 0.2 / (GLOBAL_BATCH * g)})
        sums = path_sums.cpu().tolist()
        return {"global_update": self.completed_updates if self.formal else 0,
                "smoke_optimizer_updates": 0 if self.formal else self.completed_updates,
                "optimizer_update_index": self.completed_updates, "formal": self.formal,
                "loss": sums[0] / GLOBAL_BATCH, "LC": sums[1] / GLOBAL_BATCH,
                "LR": sums[2] / GLOBAL_BATCH, "finite_loss": True, "finite_gradients": True,
                "preclip_global_norm": preclip, "clip_max_norm": 1.0,
                "lr_used": used_lr, "next_lr": self.scheduler.next_lr,
                "seconds": float(elapsed_tensor.item()), "memory": memory_stats(self.device),
                "microbatch": self.microbatch, "accumulation": self.accumulation,
                "ddp_local_sum_scale": ddp_loss_scale(GLOBAL_BATCH, self.world_size),
                "global_start_ids": [record["start_id"] for record in global_records],
                "path_weights": [{"start_id": record["start_id"], "G": int(record["initial_G"]),
                                  "path_weight": 1.0 / GLOBAL_BATCH,
                                  "step_weight": 1.0 / (GLOBAL_BATCH * int(record["initial_G"])),
                                  "lambda_R": 0.2} for record in global_records],
                "per_step": per_step, "update_counts": update_counts,
                "cumulative_counts": deepcopy(self.counts),
                "sampler": {"epoch": self.sampler.epoch, "position": self.sampler.position,
                            "consumed": self.sampler.consumed}}


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication; an existing checkpoint is never replaced.
        temporary.unlink()
        temporary = None
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_rank_rng(device: torch.device) -> dict[str, Any]:
    """Capture only this process's selected NPU, never initialize other devices."""
    state = capture_rng_state(include_npu=False)
    state["torch_npu_local"] = (torch.npu.get_rng_state(device).cpu().clone()
                                if device.type == "npu" else None)
    state["local_device_type"] = device.type
    return state


def restore_rank_rng(state: Mapping[str, Any], device: torch.device) -> None:
    if state["local_device_type"] != device.type:
        raise ValueError("Rank RNG device type mismatch")
    restore_rng_state(state)
    if device.type == "npu":
        torch.npu.set_rng_state(state["torch_npu_local"].cpu(), device)


def save_training_checkpoint(engine: TrainingEngine, path: str | Path, *,
                             run_until_update: int,
                             authorization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Collect every rank RNG and save raw + EMA separately at a synced boundary."""
    if not engine.at_boundary or engine.poisoned:
        raise RuntimeError("Cannot checkpoint an incomplete/failed update")
    limit = authorized_update_limit(authorization)
    if not engine.completed_updates <= run_until_update <= limit:
        raise ValueError("Saved boundary lies outside completed progress or authorized limit")
    synchronize(engine.device)
    local_rng = capture_rank_rng(engine.device)
    rngs = [None] * engine.world_size
    if engine.world_size > 1:
        dist.all_gather_object(rngs, local_rng)
    else:
        rngs[0] = local_rng
    payload = None
    checkpoint_sha256 = None
    if engine.rank == 0:
        payload = {"format": FORMAT, "identity": engine.identity,
                   "completed_updates": engine.completed_updates,
                   "global_update": engine.completed_updates if engine.formal else 0,
                   "smoke_optimizer_updates": 0 if engine.formal else engine.completed_updates,
                   "formal_training": engine.formal,
                   "world_size": engine.world_size, "microbatch": engine.microbatch,
                   "effective_batch": GLOBAL_BATCH, "run_until_update": run_until_update,
                   "run_authorization": deepcopy(authorization),
                   "network": _cpu_snapshot(engine.network.state_dict()),
                   "ema": _cpu_snapshot(engine.ema.state_dict()),
                   "optimizer": _cpu_snapshot(engine.optimizer.state_dict()),
                   "optimizer_class": "torch.optim.AdamW",
                   "parameter_groups": engine.group_report,
                   "scheduler": engine.scheduler.state_dict(),
                   "sampler": engine.sampler.state_dict(), "rng_by_rank": rngs,
                   "counts": deepcopy(engine.counts)}
        _validate_fp32_training_state(payload["network"], payload["optimizer"], payload["ema"])
        atomic_torch_save(path, payload)
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for part in iter(lambda: handle.read(8 << 20), b""):
                digest.update(part)
        checkpoint_sha256 = digest.hexdigest()
    if engine.world_size > 1:
        dist.barrier()
    return {"path": str(path), "completed_updates": engine.completed_updates,
            "formal_training": engine.formal, "run_until_update": run_until_update,
            "sha256": checkpoint_sha256}


def load_training_checkpoint(engine: TrainingEngine, path: str | Path, *,
                             run_until_update: int,
                             authorize_boundary_extension: bool = False,
                             authorization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Restore only compatible complete state; authorization is not in identity.

    The supplied engine must be fresh. Failed restoration poisons it and the
    process must exit; a partial restoration can never continue or save state.
    """
    if engine.completed_updates or engine.sampler.consumed or engine.poisoned:
        raise ValueError("Restore requires a fresh engine")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format": FORMAT, "identity": engine.identity, "world_size": engine.world_size,
                "microbatch": engine.microbatch, "effective_batch": GLOBAL_BATCH,
                "formal_training": engine.formal, "optimizer_class": "torch.optim.AdamW",
                "parameter_groups": engine.group_report}
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("Checkpoint code/config/data, topology, groups or formal/smoke identity mismatch")
    n = engine.scheduler.validate_state_dict(payload["scheduler"])
    if (payload["completed_updates"] != n or payload["ema"]["updates"] != n
            or payload["sampler"]["consumed"] != n * GLOBAL_BATCH
            or payload["counts"]["paths"] != n * GLOBAL_BATCH
            or len(payload["rng_by_rank"]) != engine.world_size
            or payload["global_update"] != (n if engine.formal else 0)
            or payload["smoke_optimizer_updates"] != (0 if engine.formal else n)):
        raise ValueError("Checkpoint update/sampler/EMA/count inconsistency")
    limit = authorized_update_limit(authorization)
    if run_until_update < n or run_until_update > limit:
        raise ValueError("Requested run boundary is before checkpoint or exceeds current authorization")
    if run_until_update > 1000 and not authorize_boundary_extension:
        raise ValueError("Future continuation requires the explicit boundary-extension flag")
    if run_until_update > payload["run_until_update"] and not authorize_boundary_extension:
        raise ValueError("Increasing a saved run boundary requires explicit authorization")
    _validate_fp32_training_state(payload["network"], payload["optimizer"], payload["ema"])
    saved_groups = payload["optimizer"]["param_groups"]
    current_groups = engine.optimizer.state_dict()["param_groups"]
    if len(saved_groups) != len(current_groups):
        raise ValueError("Saved optimizer parameter-group count mismatch")
    for saved, current in zip(saved_groups, current_groups):
        saved_options = {key: value for key, value in saved.items() if key not in {"lr", "params"}}
        current_options = {key: value for key, value in current.items() if key not in {"lr", "params"}}
        if saved_options != current_options or saved["params"] != current["params"]:
            raise ValueError("Saved optimizer grouping/options disagree with frozen configuration")
    if any(group["lr"] != payload["scheduler"]["next_lr"] for group in saved_groups):
        raise ValueError("Saved optimizer/scheduler LR mismatch")
    expected_counts = empty_counts()
    for g in [4, 3, 2, 1]:
        count = payload["counts"]["by_initial_G"][str(g)]
        if type(count) is not int or count < 0:
            raise ValueError("Saved exposure counts are invalid")
        advance_counts(expected_counts, [g] * count)
    if payload["counts"] != expected_counts:
        raise ValueError("Saved per-G input/prediction exposure counters are inconsistent")
    try:
        engine.network.load_state_dict(payload["network"], strict=True)
        engine.optimizer.load_state_dict(payload["optimizer"])
        engine.scheduler.load_state_dict(payload["scheduler"])
        engine.ema.load_state_dict(payload["ema"])
        engine.sampler.load_state_dict(payload["sampler"])
        engine.counts = deepcopy(payload["counts"])
        _validate_fp32_training_state(engine.network.state_dict(), engine.optimizer.state_dict(),
                                     engine.ema.state_dict())
        restore_rank_rng(payload["rng_by_rank"][engine.rank], engine.device)
        engine.optimizer.zero_grad(set_to_none=True)
    except BaseException:
        engine.poisoned = True
        raise
    return {"path": str(path), "completed_updates": n, "next_lr": engine.scheduler.next_lr,
            "previous_run_until_update": payload["run_until_update"],
            "run_until_update": run_until_update,
            "boundary_extension_authorized": authorize_boundary_extension}


def compare_states(first: Any, second: Any, *, path: str = "state") -> dict[str, Any]:
    """Exact resume comparison with max absolute drift, including integer/RNG state."""
    mismatches: list[str] = []
    max_abs = 0.0
    def visit(a: Any, b: Any, label: str) -> None:
        nonlocal max_abs
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                mismatches.append(label)
            elif not torch.equal(a.cpu(), b.cpu()):
                mismatches.append(label)
                if a.numel() and (a.is_floating_point() or a.dtype != torch.bool):
                    max_abs = max(max_abs, float((a.cpu().double() - b.cpu().double()).abs().max()))
        elif isinstance(a, Mapping):
            if not isinstance(b, Mapping) or a.keys() != b.keys():
                mismatches.append(label)
            else:
                for key in a:
                    visit(a[key], b[key], f"{label}.{key}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                mismatches.append(label)
            else:
                for index, (av, bv) in enumerate(zip(a, b)):
                    visit(av, bv, f"{label}[{index}]")
        elif a != b:
            mismatches.append(label)
    visit(first, second, path)
    return {"exact": not mismatches, "max_absolute_tensor_drift": max_abs,
            "mismatched_fields": mismatches}

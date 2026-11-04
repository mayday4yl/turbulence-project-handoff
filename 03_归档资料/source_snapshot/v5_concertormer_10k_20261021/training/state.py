"""Deterministic update-indexed batches and atomic resumable state."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist


def canonical_hash(value):
    if is_dataclass(value):
        value = asdict(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng(device):
    device = torch.device(device)
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(), "device_type": device.type}
    if device.type in ("npu", "cuda"):
        state["device_rng"] = getattr(torch, device.type).get_rng_state(device).cpu()
    return state


def restore_rng(state, device):
    device = torch.device(device)
    if state["device_type"] != device.type:
        raise ValueError("Exact resume requires the same device type")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "device_rng" in state:
        getattr(torch, device.type).set_rng_state(state["device_rng"], device)


def gather_rank_state(local_state):
    if not (dist.is_available() and dist.is_initialized()):
        return [local_state]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, local_state)
    return values


class UpdateBatchSampler:
    """Rank shards of one deterministic global sample stream per optimizer update.

    The next batch is a pure function of dataset size/seed/update/micro-index.
    DataLoader prefetch cannot advance the checkpoint's logical sampler state.
    There is no stochastic dataset transform in this V5 data protocol.
    """
    def __init__(self, dataset_size, *, seed, microbatch, accumulation_steps,
                 world_size=1, rank=0, start_update=1, max_updates=10000):
        if min(dataset_size, microbatch, accumulation_steps, world_size, start_update) < 1:
            raise ValueError("positive sampler dimensions required")
        if rank < 0 or rank >= world_size or max_updates < start_update - 1:
            raise ValueError("invalid rank or update interval")
        self.dataset_size, self.seed = dataset_size, seed
        self.microbatch, self.accumulation_steps = microbatch, accumulation_steps
        self.world_size, self.rank = world_size, rank
        self.start_update, self.max_updates = start_update, max_updates
        self.effective_batch = microbatch * accumulation_steps * world_size
        self._cached_epoch, self._cached_permutation = None, None

    def _index(self, exposure):
        epoch, offset = divmod(exposure, self.dataset_size)
        if self._cached_epoch != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._cached_permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
            self._cached_epoch = epoch
        return self._cached_permutation[offset]

    def indices_for(self, update, micro_index):
        start = ((update - 1) * self.effective_batch +
                 micro_index * self.world_size * self.microbatch + self.rank * self.microbatch)
        return [self._index(i) for i in range(start, start + self.microbatch)]

    def __iter__(self):
        for update in range(self.start_update, self.max_updates + 1):
            for micro_index in range(self.accumulation_steps):
                yield self.indices_for(update, micro_index)

    def __len__(self):
        return max(0, self.max_updates - self.start_update + 1) * self.accumulation_steps

    def state_dict(self, completed_update):
        return {"kind": "update_indexed_global_epoch_shuffle_v1", "seed": self.seed,
                "dataset_size": self.dataset_size, "completed_update": completed_update,
                "next_global_exposure": completed_update * self.effective_batch,
                "effective_batch": self.effective_batch, "microbatch": self.microbatch,
                "accumulation_steps": self.accumulation_steps, "world_size": self.world_size}


class UpdateLRScheduler:
    """Set LR immediately before optimizer update u, with u starting at one."""
    def __init__(self, optimizer, *, max_updates=10000, warmup_updates=300):
        if max_updates < 1 or warmup_updates < 0 or warmup_updates >= max_updates:
            raise ValueError("require 0 <= warmup_updates < max_updates")
        self.optimizer, self.max_updates, self.warmup_updates = optimizer, max_updates, warmup_updates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_update = 0

    def multiplier(self, update):
        if update < 1 or update > self.max_updates:
            raise ValueError("LR update outside budget")
        if self.warmup_updates and update <= self.warmup_updates:
            return update / self.warmup_updates
        phase = (update - self.warmup_updates) / (self.max_updates - self.warmup_updates)
        return .5 * (1. + math.cos(math.pi * phase))

    def step_for_update(self, update):
        if update != self.last_update + 1:
            raise ValueError("scheduler must advance by exactly one optimizer update")
        factor = self.multiplier(update)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        self.last_update = update

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_update": self.last_update,
                "max_updates": self.max_updates, "warmup_updates": self.warmup_updates,
                "cosine_terminal_lr": 0., "step_timing": "before_optimizer_step_1_based"}

    def load_state_dict(self, state):
        if state["max_updates"] != self.max_updates or state["warmup_updates"] != self.warmup_updates:
            raise ValueError("cannot change LR schedule during exact resume")
        if len(state["base_lrs"]) != len(self.base_lrs):
            raise ValueError("optimizer group count mismatch")
        self.base_lrs = state["base_lrs"]
        self.last_update = state["last_update"]
        factor = self.multiplier(self.last_update) if self.last_update else 1.
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

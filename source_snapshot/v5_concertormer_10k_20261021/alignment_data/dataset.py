"""Stateless update-indexed A sampling; DataLoader prefetch cannot advance RNG."""
import functools
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from data_pipeline.common import rows, sha, verify_array
from .identity import SCHEMA, load_base
from .plans import DEFAULT_SEED, TIME_BINS, cache_directory, stable_seed


class AlignmentBatchSampler(Sampler):
    """Indices are pure functions of (seed, completed update, microbatch, rank).

    An update consumes exactly 12 endpoint and 36 internal global state items.
    Internal items are visited without replacement within each 80k cache cycle.
    A fresh cycle is used every 2000 updates: 72k of its 80k states are consumed.
    Public updates are 1-based: indices_for(1, 0) starts optimizer update 1.
    completed_update is a count. Segment stop changes never alter sampler identity.
    """
    def __init__(self, sample_count, *, seed=DEFAULT_SEED, microbatch=2, accumulation_steps=4,
                 world_size=6, rank=0, start_update=1, max_updates=6000, data_identity=None):
        if microbatch * accumulation_steps * world_size != 48:
            raise ValueError("Authorized A effective batch is exactly 48")
        if not 0 <= rank < world_size or not 1 <= start_update <= max_updates + 1 <= 6001:
            raise ValueError("Invalid rank or A update budget")
        if sample_count <= 0:
            raise ValueError("Empty training source")
        self.sample_count = sample_count
        self.seed = seed
        self.microbatch = microbatch
        self.accumulation_steps = accumulation_steps
        self.world_size = world_size
        self.rank = rank
        self.start_update = start_update
        self.max_updates = max_updates
        self.data_identity = data_identity
        self.cache_size = sample_count * TIME_BINS
        self.cycle_stride_updates = 2000

    @functools.lru_cache(maxsize=4)
    def _permutation(self, kind, epoch):
        count = self.sample_count if kind == "endpoint" else self.cache_size
        values = list(range(count))
        random.Random(stable_seed(self.seed, "sampler-v1", kind, epoch)).shuffle(values)
        return values

    def global_indices_for(self, update):
        if not 1 <= update <= 6000:
            raise IndexError("Update outside authorized A budget")
        update = update - 1
        indices = []
        for position in range(update * 12, (update + 1) * 12):
            epoch, offset = divmod(position, self.sample_count)
            indices.append(("endpoint", self._permutation("endpoint", epoch)[offset]))
        cycle, update_in_cycle = divmod(update, self.cycle_stride_updates)
        for offset in range(update_in_cycle * 36, (update_in_cycle + 1) * 36):
            if offset >= self.cache_size:
                raise ValueError("Cache is too small for this 2000-update segment")
            indices.append(("internal", cycle, self._permutation("internal", cycle)[offset]))
        random.Random(stable_seed(self.seed, "global-batch-order-v1", update)).shuffle(indices)
        return indices

    def indices_for(self, update, micro_index):
        if not 0 <= micro_index < self.accumulation_steps:
            raise IndexError("Invalid accumulation index")
        start = (micro_index * self.world_size + self.rank) * self.microbatch
        return self.global_indices_for(update)[start:start + self.microbatch]

    def __iter__(self):
        for update in range(self.start_update, self.max_updates + 1):
            for micro_index in range(self.accumulation_steps):
                yield self.indices_for(update, micro_index)

    def __len__(self):
        return max(0, self.max_updates - self.start_update + 1) * self.accumulation_steps

    def state_dict(self, completed_update):
        if not 0 <= completed_update <= 6000:
            raise ValueError("Completed update outside A budget")
        return {"schema": SCHEMA, "sampler": "stateless-update-indexed-v1", "seed": self.seed,
                "sample_count": self.sample_count, "cache_size": self.cache_size,
                "microbatch": self.microbatch, "accumulation_steps": self.accumulation_steps,
                "world_size": self.world_size, "scheduler_horizon": 6000,
                "completed_update": completed_update, "next_update_index": completed_update + 1,
                "endpoint_items_consumed": completed_update * 12,
                "internal_items_consumed": completed_update * 36,
                "cycle_stride_updates": self.cycle_stride_updates,
                "next_internal_cycle": completed_update // self.cycle_stride_updates,
                "next_internal_offset": (completed_update % self.cycle_stride_updates) * 36,
                "data_identity": self.data_identity}

    def load_state_dict(self, state):
        expected = self.state_dict(state["completed_update"])
        if state != expected:
            raise ValueError("Sampler resume identity/geometry does not match")
        self.start_update = state["completed_update"] + 1

    def required_cycles(self, start_update=None, end_update=None):
        start = self.start_update if start_update is None else start_update
        end = self.max_updates if end_update is None else end_update
        if not 1 <= start <= end + 1 <= 6001:
            raise ValueError("Invalid update interval")
        return list(range((start - 1) // self.cycle_stride_updates, (end - 1) // self.cycle_stride_updates + 1)) if end >= start else []


class AlignmentDataset(Dataset):
    def __init__(self, base_root, cache_root, split="train", cycle=0, require_ready=True):
        if split not in {"train", "val"}:
            raise ValueError("A cache exposes Train/Val only; Test remains locked")
        self.base_root, self.cache_root = Path(base_root), Path(cache_root)
        self.split = split
        self.cycle = cycle
        self.require_ready = require_ready
        self.base_identity, _, samples, self.base_records = load_base(self.base_root)
        self.samples = [r for r in samples if r["split"] == split]
        self.cache_identity = json.loads((self.cache_root / "CACHE_IDENTITY.json").read_text())
        if self.cache_identity["base"] != self.base_identity:
            raise ValueError("Cache belongs to different frozen base data")
        if self.cache_identity["time_bins"] != TIME_BINS:
            raise ValueError("Cache time-bin definition changed")
        self.data_identity = {"base": self.base_identity,
                              "cache_identity_sha256": sha(self.cache_root / "CACHE_IDENTITY.json"),
                              "cache_seed": self.cache_identity["seed"], "split": split}
        if split == "val" or require_ready:
            self._manifest(0 if split == "val" else cycle)

    @functools.lru_cache(maxsize=2)
    def _manifest(self, cycle):
        directory = cache_directory(self.cache_root, self.split, cycle)
        ready_path = directory / "READY.json"
        if not ready_path.is_file():
            raise RuntimeError(f"Required physical-state cache is not READY: {directory}; do not substitute another cycle")
        ready = json.loads(ready_path.read_text())
        if ready["status"] != "READY" or ready["cache_identity_sha256"] != self.data_identity["cache_identity_sha256"]:
            raise ValueError("Cache READY identity mismatch")
        if sha(directory / "PLAN.jsonl") != ready["plan_sha256"]:
            raise ValueError("Cache time plan changed")
        if sha(directory / "MANIFEST.jsonl") != ready["manifest_sha256"]:
            raise ValueError("Cache output manifest changed")
        records = rows(directory / "MANIFEST.jsonl")
        expected = len(self.samples) * (TIME_BINS if self.split == "train" else 1)
        if len(records) != expected or ready["records"] != expected:
            raise ValueError("Cache record count differs")
        return directory, records

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.split == "val":
            if not isinstance(index, int):
                raise TypeError("Val-A uses integer indices")
            directory, records = self._manifest(0)
            record = records[index]
            row = record["plan_row"]
        elif isinstance(index, (tuple, list)) and index[0] == "endpoint":
            base = self.samples[index[1]]
            row = {"sample_id": base["sample_id"] + "_t0", "base_sample_id": base["sample_id"],
                   "source_id": base["source_id"], "realization_id": base["realization_id"],
                   "realization_index": base["realization_index"], "clean_ref": base["clean_ref"],
                   "clean_sha256": base["clean_sha256"], "observation_ref": base["observation_ref"],
                   "observation_root": "base", "t": 0.0, "strength": 4.0, "cache_cycle": -1,
                   "time_slot": -1, "is_endpoint": True}
            record = self.base_records[base["sample_id"]]
            directory = self.base_root
        elif isinstance(index, (tuple, list)) and index[0] == "internal":
            _, cycle, record_index = index
            directory, records = self._manifest(cycle)
            record = records[record_index]
            row = record["plan_row"]
        else:
            raise TypeError("Train A requires the immutable update-indexed batch sampler")
        clean = verify_array(self.base_root / row["clean_ref"], row["clean_sha256"])
        observation_root = self.base_root if row["observation_root"] == "base" else directory
        observation = verify_array(observation_root / row["observation_ref"], record["observation_sha256"])
        t = torch.tensor(row["t"], dtype=torch.float32)
        g = torch.tensor(row["strength"], dtype=torch.float32)
        if g.item() != (4.0 * (1.0 - t)).item():
            raise ValueError("Cached float32 strength/time mapping changed")
        if row["is_endpoint"]:
            if t.item() != 0.0 or g.item() != 4.0:
                raise ValueError("Invalid endpoint")
        elif not (0.0 < t.item() < 1.0 and 0.0 < g.item() < 4.0):
            raise ValueError("Internal state reached endpoint after float32 conversion")
        return {"clean": torch.from_numpy(clean), "observation": torch.from_numpy(observation),
                "t": t, "time": t, "strength": g, "G": torch.tensor(4.0, dtype=torch.float32),
                "sample_id": row["sample_id"], "base_sample_id": row["base_sample_id"],
                "source_id": row["source_id"], "realization_id": row["realization_id"],
                "realization_index": row["realization_index"], "cache_cycle": row["cache_cycle"],
                "time_slot": row["time_slot"], "is_endpoint": row["is_endpoint"]}

    def make_batch_sampler(self, seed=DEFAULT_SEED, microbatch=2, accumulation_steps=4,
                           world_size=6, rank=0, start_update=1, max_updates=6000):
        if self.split != "train":
            raise ValueError("Training sampler is unavailable for Val-A")
        return AlignmentBatchSampler(len(self.samples), seed=seed, microbatch=microbatch,
                                     accumulation_steps=accumulation_steps, world_size=world_size,
                                     rank=rank, start_update=start_update, max_updates=max_updates,
                                     data_identity=self.data_identity)

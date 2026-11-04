"""Frozen manifests and indexed, uncompressed tar records for bridge data."""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .simulator import FROZEN_G_GRID, Realization
from .utils import derive_seed


G_TO_SUFFIX = {
    0.75: "x_g075.npy",
    1.50: "x_g150.npy",
    2.25: "x_g225.npy",
    3.00: "x_g300.npy",
    3.75: "x_g375.npy",
    4.50: "x_g450.npy",
}
TERMINAL_GRID = (1.50, 2.25, 3.00, 3.75, 4.50)
REQUIRED_SUFFIXES = (
    "clean.png",
    *G_TO_SUFFIX.values(),
    "epsilon_zernike.npy",
    "epsilon_tilt.npy",
    "json",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def severity_suffix(g: float) -> str:
    key = round(float(g), 2)
    if key not in G_TO_SUFFIX:
        raise KeyError(f"Severity {g} is outside the frozen grid")
    return G_TO_SUFFIX[key]


@dataclass(frozen=True)
class MemberLocation:
    offset: int
    size: int


class IndexedTarStore:
    """Direct-offset reader that avoids loading archived xi during ordinary training."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root)
        self._handles: dict[Path, Any] = {}
        self._indexes: dict[Path, dict[str, MemberLocation]] = {}

    def __getstate__(self) -> dict[str, Any]:
        return {"root": self.root, "_handles": {}, "_indexes": {}}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _load_index(self, shard: Path) -> dict[str, MemberLocation]:
        index_path = shard.with_suffix(".index.json")
        if index_path not in self._indexes:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            self._indexes[index_path] = {
                name: MemberLocation(int(value["offset"]), int(value["size"]))
                for name, value in raw["members"].items()
            }
        return self._indexes[index_path]

    def read(self, relative_shard: str, member: str) -> bytes:
        shard = self.root / relative_shard
        index = self._load_index(shard)
        if member not in index:
            raise KeyError(f"Missing {member} in {shard}")
        if shard not in self._handles:
            self._handles[shard] = shard.open("rb", buffering=0)
        location = index[member]
        handle = self._handles[shard]
        handle.seek(location.offset)
        value = handle.read(location.size)
        if len(value) != location.size:
            raise IOError(f"Short read for {member} from {shard}")
        return value


def decode_png(value: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(value)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def decode_npy(value: bytes, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    array = np.load(io.BytesIO(value), allow_pickle=False)
    tensor = torch.from_numpy(array.copy())
    return tensor.to(dtype=dtype) if dtype is not None else tensor


class BridgeArchive:
    def __init__(self, dataset_root: str | Path, split: str) -> None:
        self.root = Path(dataset_root)
        self.split = split
        manifest = self.root / "manifests" / f"{split}.jsonl"
        self.records = load_jsonl(manifest)
        self.by_key = {record["key"]: record for record in self.records}
        if len(self.by_key) != len(self.records):
            raise ValueError(f"Duplicate keys in {manifest}")
        self.store = IndexedTarStore(self.root)

    def _member(self, record: Mapping[str, Any], suffix: str) -> bytes:
        return self.store.read(record["shard"], f"{record['key']}.{suffix}")

    def clean(self, record: Mapping[str, Any]) -> torch.Tensor:
        return decode_png(self._member(record, "clean.png"))

    def state(self, record: Mapping[str, Any], severity: float) -> torch.Tensor:
        return decode_npy(self._member(record, severity_suffix(severity)), dtype=torch.float32)

    def realization(self, record: Mapping[str, Any]) -> Realization:
        zernike = decode_npy(
            self._member(record, "epsilon_zernike.npy"), dtype=torch.float32
        )
        tilt = decode_npy(self._member(record, "epsilon_tilt.npy"), dtype=torch.float32)
        return Realization(zernike, tilt)

    def metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(self._member(record, "json"))


class StatelessTrainingDataset(Dataset[dict[str, Any]]):
    """Implements the frozen uniform-g then uniform-eligible-G sampler."""

    def __init__(self, dataset_root: str | Path, master_seed: int = 20260826) -> None:
        self.archive = BridgeArchive(dataset_root, "train")
        self.master_seed = int(master_seed)
        self.buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for record in self.archive.records:
            self.buckets[round(float(record["terminal_G"]), 2)].append(record)
        expected = len(self.archive.records) // len(TERMINAL_GRID)
        for terminal in TERMINAL_GRID:
            if len(self.buckets[terminal]) != expected:
                raise ValueError(f"Unbalanced terminal bucket {terminal}: {len(self.buckets[terminal])}")

    def __len__(self) -> int:
        return 2**31 - 1

    def selection(self, sample_id: int) -> tuple[float, float, dict[str, Any]]:
        g_index = derive_seed(self.master_seed, "train-g", sample_id) % len(FROZEN_G_GRID)
        severity = FROZEN_G_GRID[g_index]
        eligible = [terminal for terminal in TERMINAL_GRID if terminal >= severity]
        terminal_index = derive_seed(self.master_seed, "train-G", sample_id) % len(eligible)
        terminal = eligible[terminal_index]
        bucket = self.buckets[terminal]
        record_index = derive_seed(self.master_seed, "train-record", sample_id) % len(bucket)
        return severity, terminal, bucket[record_index]

    def __getitem__(self, sample_id: int) -> dict[str, Any]:
        severity, terminal, record = self.selection(int(sample_id))
        return {
            "sample_id": int(sample_id),
            "key": record["key"],
            "image": self.archive.state(record, severity),
            "clean": self.archive.clean(record),
            "terminal": self.archive.state(record, terminal),
            "g": torch.tensor(severity, dtype=torch.float32),
            "G": torch.tensor(terminal, dtype=torch.float32),
        }


class RankSampleSampler(Sampler[int]):
    """Resume-stable sample IDs for one DDP rank."""

    def __init__(
        self,
        *,
        start_update: int,
        end_update: int,
        effective_batch_size: int,
        rank: int,
        world_size: int,
    ) -> None:
        if effective_batch_size % world_size:
            raise ValueError("Effective batch size must be divisible by world size")
        self.start_update = int(start_update)
        self.end_update = int(end_update)
        self.effective_batch_size = int(effective_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_per_update = self.effective_batch_size // self.world_size

    def __len__(self) -> int:
        return (self.end_update - self.start_update) * self.local_per_update

    def __iter__(self) -> Iterator[int]:
        for update in range(self.start_update, self.end_update):
            base = update * self.effective_batch_size + self.rank * self.local_per_update
            yield from range(base, base + self.local_per_update)


def evenly_spaced_schedule(start: float, transitions: int) -> tuple[float, ...]:
    grid = [value for value in FROZEN_G_GRID if value <= round(float(start), 2)]
    if not grid or grid[-1] != round(float(start), 2):
        raise ValueError(f"Start severity {start} is outside the frozen grid")
    available = len(grid) - 1
    effective = min(int(transitions), available)
    if effective == 0:
        return (grid[-1],)
    raw = np.linspace(available, 0, effective + 1)
    # Floor the intermediate grid indices so the explicitly frozen K=3 case is
    # 4.50 -> 3.00 -> 1.50 -> 0.75 (rather than introducing 2.25).
    indices = [int(math.floor(index)) for index in raw]
    indices[0], indices[-1] = available, 0
    deduplicated = []
    for index in indices:
        if not deduplicated or index != deduplicated[-1]:
            deduplicated.append(index)
    return tuple(grid[index] for index in deduplicated)


def expected_shards(trajectories: int, shard_size: int) -> int:
    return math.ceil(trajectories / shard_size)


def iter_assigned_shards(total: int, rank: int, world_size: int) -> Iterable[int]:
    return range(rank, total, world_size)

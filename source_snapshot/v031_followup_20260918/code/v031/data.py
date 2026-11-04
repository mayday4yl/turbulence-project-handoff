"""Read-only adapter for accepted V0.31 starting records and FP32 arrays.

Ordinary access reads only C and the one original observation Y. Native GT xi
is exposed through a separate diagnostics method, never a training item.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .core import severity_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object in {path}:{line_number}")
                records.append(record)
    return records


def collate_starts(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate complete starting paths without padding their variable-length tails.

    Use ``DataLoader(dataset, ..., collate_fn=collate_starts)``. The returned
    clean/observation/G tensors have shapes [B,3,H,W], [B,3,H,W], [B]. Metadata
    preserves input order: severity_path is a tuple of complete tail tuples;
    start_id and trajectory_id are lists. No GT xi or weaker state is loaded.

    Pass ``batch['severity_path']`` as RolloutModule's optional ``schedules``
    argument, or let the module derive the same complete tails from ``batch['G']``.
    Default PyTorch collation cannot handle these unequal-length paths.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch of starts")
    paths = []
    for sample in samples:
        clean, observation, initial_g = sample["clean"], sample["observation"], sample["G"]
        if not all(torch.is_tensor(tensor) for tensor in (clean, observation, initial_g)):
            raise ValueError("Start samples require tensor clean, observation and G")
        if clean.ndim != 3 or clean.shape[0] != 3 or clean.shape != observation.shape:
            raise ValueError("Start images must have equal [3,H,W] shapes")
        if any(tensor.dtype != torch.float32 for tensor in (clean, observation, initial_g)):
            raise ValueError("Start tensors must retain their authoritative FP32 dtype")
        if initial_g.ndim != 0:
            raise ValueError("Each start must have one scalar G")
        path = tuple(sample["severity_path"])
        if path != severity_path(initial_g.item()):
            raise ValueError("Start severity_path must be the complete tail for G")
        paths.append(path)
    return {
        "clean": torch.stack([sample["clean"] for sample in samples]),
        "observation": torch.stack([sample["observation"] for sample in samples]),
        "G": torch.stack([sample["G"] for sample in samples]),
        "severity_path": tuple(paths),
        "start_id": [sample["start_id"] for sample in samples],
        "trajectory_id": [sample["trajectory_id"] for sample in samples],
    }


class StartDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path,
                 start_manifest: str | Path = "start_manifest.jsonl") -> None:
        self.root = Path(root).resolve(strict=True)
        self.sources = self._unique(read_jsonl(self._safe_path("source_manifest.jsonl")), "source_id")
        self.trajectories = self._unique(
            read_jsonl(self._safe_path("trajectory_manifest.jsonl")), "trajectory_id")
        main_path = self._safe_path("start_manifest.jsonl")
        main_records = read_jsonl(main_path)
        by_start = self._unique(main_records, "start_id")
        chosen_path = self._safe_path(start_manifest)
        self.records = main_records if chosen_path == main_path else read_jsonl(chosen_path)
        self._unique(self.records, "start_id")
        if not self.records:
            raise ValueError("Start manifest is empty")
        for record in self.records:
            if record.get("start_id") not in by_start or record != by_start[record["start_id"]]:
                raise ValueError("Subset records must match authoritative start records exactly")
            self._validate_record(record)

    @staticmethod
    def _unique(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
        indexed = {record[key]: record for record in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate {key} in manifest")
        return indexed

    def _safe_path(self, reference: str | Path) -> Path:
        relative = Path(reference)
        if relative.is_absolute():
            raise ValueError(f"Dataset reference must be relative: {reference}")
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Dataset reference escapes root: {reference}")
        return candidate

    def _validate_record(self, record: dict[str, Any]) -> None:
        expected = severity_path(record["initial_G"])
        if tuple(record["severity_path"]) != expected or record["num_predictions"] != len(expected):
            raise ValueError(f"Invalid complete severity tail: {record['start_id']}")
        source = self.sources[record["source_id"]]
        trajectory = self.trajectories[record["trajectory_id"]]
        if trajectory["source_id"] != record["source_id"]:
            raise ValueError("Trajectory and start source identities disagree")
        if record["clean_ref"] != source["clean_ref"] or record["clean_ref"] != trajectory["clean_ref"]:
            raise ValueError("Source, trajectory and start clean references disagree")
        if record["initial_state_ref"] != trajectory[f"x{int(record['initial_G'])}_ref"]:
            raise ValueError("Initial state reference disagrees with original G")
        # Validate containment without opening arrays or GT realization data.
        self._safe_path(record["clean_ref"])
        self._safe_path(record["initial_state_ref"])

    def _array(self, reference: str, shape: tuple[int, ...]) -> torch.Tensor:
        array = np.load(self._safe_path(reference), allow_pickle=False)
        if array.dtype != np.float32 or tuple(array.shape) != shape:
            raise ValueError(f"Expected FP32 {shape} at {reference}; got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite authoritative data: {reference}")
        return torch.from_numpy(array)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "clean": self._array(record["clean_ref"], (3, 256, 256)),
            "observation": self._array(record["initial_state_ref"], (3, 256, 256)),
            "G": torch.tensor(record["initial_G"], dtype=torch.float32),
            "severity_path": severity_path(record["initial_G"]),
            "start_id": record["start_id"],
            "trajectory_id": record["trajectory_id"],
        }

    def load_realization(self, index: int) -> Any:
        """Diagnostics only: native GT xi does not enter normal item loading."""
        from physical_bridge.simulator import Realization

        trajectory = self.trajectories[self.records[index]["trajectory_id"]]
        return Realization(
            self._array(trajectory["epsilon_zernike_ref"], (36, 512, 512)),
            self._array(trajectory["epsilon_tilt_ref"], (512, 512, 2)))

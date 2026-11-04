"""Reproducibility, hashing, and safe atomic-write helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ONE_TIB = 1 << 40


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable positive 63-bit seed from a master seed and identifiers."""

    payload = "\x1f".join([str(int(master_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write_bytes(path: str | Path, value: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def atomic_touch(path: str | Path, text: str = "complete\n") -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def ensure_free_space(path: str | Path, minimum_free_bytes: int = ONE_TIB) -> int:
    free = shutil.disk_usage(Path(path)).free
    if free < minimum_free_bytes:
        raise RuntimeError(
            f"Safety stop: {Path(path)} has {free} free bytes, below {minimum_free_bytes}"
        )
    return free


def seed_process(seed: int) -> None:
    seed32 = int(seed) % (2**32)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed32)


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


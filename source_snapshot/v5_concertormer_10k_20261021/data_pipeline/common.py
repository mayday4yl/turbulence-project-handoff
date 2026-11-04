"""Data identity and atomic IO helpers; no model or accelerator side effects."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

DATASET_ID = 'v5_places_c50_s2000_v250_t250_g4_r4_20261021'
SIMULATOR_SHA256 = '0f2f6f54a578b3c36cb69861815c7eee5476f75ccd91cac863bd001345bd9636'
COUNTS = {'train': 2000, 'val': 250, 'test': 250}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic(path, raw, *, frozen=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Frozen identity mismatch: {path}')
        return
    tmp = path.with_name(path.name + f'.partial.{os.getpid()}')
    with tmp.open('wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value, *, frozen=True):
    atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode(), frozen=frozen)


def write_rows(path, values):
    atomic(path, ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in values).encode())


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def verify_array(path, expected_hash=None):
    if expected_hash and sha(path) != expected_hash:
        raise ValueError(f'Array file hash mismatch: {path}')
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (3, 256, 256) or not np.isfinite(array).all():
        raise ValueError(f'Invalid RGB FP32 array: {path}')
    return array

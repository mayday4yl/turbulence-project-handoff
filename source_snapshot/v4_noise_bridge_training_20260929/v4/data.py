"""Identity-checked reader for the independent A100 physical dataset.

Training and statistics share the same full, fixed 256 x 256 RGB FP32 view.
Physical states are already in RGB uint8/255 units and may be outside [0,1].
No rescaling, clamping, augmentation, or raw-xi supervision is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_array(root: str | Path, record: dict) -> np.ndarray:
    """Verify content identity before returning the unmodified FP32 array."""
    root = Path(root).resolve()
    path = (root / record['ref']).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Array reference escapes dataset directory')
    if sha256_file(path) != record['sha256']:
        raise ValueError(f'Array hash differs from manifest: {record["ref"]}')
    value = np.load(path, allow_pickle=False)
    if (value.dtype != np.float32 or record.get('dtype', 'float32') != 'float32'
            or list(value.shape) != record['shape'] or not np.isfinite(value).all()):
        raise ValueError(f'Array dtype/shape/finite-value contract failed: {record["ref"]}')
    return value


class PhysicalDataset:
    """Read only an explicit train population; never silently admit val/test rows."""

    preprocessing = {
        'layout': 'CHW', 'channels': ['R', 'G', 'B'], 'input_dtype': 'float32',
        'source_units': 'RGB uint8 / 255.0; physical simulator output not clamped',
        'network_units': 'same as source_units', 'view_id': 'full_256_fixed',
        'shape': [3, 256, 256], 'crop_box_xyxy': [0, 0, 256, 256],
        'resize': None, 'crop_seed': None, 'random_crop': False,
        'interpolation': None, 'boundary': 'all stored pixels',
        'clamp': False, 'quantization': False, 'per_image_standardization': False,
    }

    def __init__(self, root: str | Path, population: str = 'overfit100',
                 asset_lock: str | Path | None = None):
        self.root = Path(root).resolve()
        if population not in {'overfit100', 'train'}:
            raise ValueError('Statistics/training population must be overfit100 or train')
        self.population = population
        self.sources = _read_jsonl(self.root / 'source_manifest.jsonl')
        self.trajectories = _read_jsonl(self.root / 'trajectory_manifest.jsonl')
        self.source_by_id = self._unique(self.sources, 'source_id')
        self.by_id = self._unique(self.trajectories, 'trajectory_id')
        identity_path = self.root / 'BUILD_IDENTITY.json'
        self.identity = json.loads(identity_path.read_text())
        self.dataset_id = self.identity['config']['dataset_id']
        if self.identity['config'].get('additional_bridge_noise_baked_into_dataset') is not False:
            raise ValueError('Dataset must explicitly contain no additional bridge noise')
        if self.identity['config'].get('clamp') is not False:
            raise ValueError('Dataset must explicitly preserve unclamped physical states')
        source_sha = sha256_file(self.root / 'source_manifest.jsonl')
        # BUILD_IDENTITY.source_manifest_sha256 identifies the pre-generation
        # plan, whereas the generated source manifest is enriched with array hashes.
        # They are distinct identities; bind generated manifests using ASSET_LOCK.
        self.manifest_hashes = {
            'BUILD_IDENTITY.json': sha256_file(identity_path),
            'source_manifest.jsonl': source_sha,
            'trajectory_manifest.jsonl': sha256_file(self.root / 'trajectory_manifest.jsonl'),
        }
        if population == 'overfit100':
            subset_path = self.root / 'subsets' / 'overfit100.json'
            subset = json.loads(subset_path.read_text())
            self.manifest_hashes['subsets/overfit100.json'] = sha256_file(subset_path)
            selected_ids = subset['trajectory_ids']
            selected_sources = subset['source_ids']
            if len(set(selected_ids)) != len(selected_ids) or len(set(selected_sources)) != len(selected_sources):
                raise ValueError('Duplicate entries in selected train population')
            if not set(selected_ids) <= self.by_id.keys():
                raise ValueError('Unknown trajectory in selected train population')
            if not set(selected_sources) <= self.source_by_id.keys():
                raise ValueError('Unknown source in selected train population')
            for sid in selected_sources:
                if self.source_by_id[sid]['split'] != 'train':
                    raise ValueError('Non-train source in selected train population')
            if {self.by_id[tid]['source_id'] for tid in selected_ids} != set(selected_sources):
                raise ValueError('Train subset source and trajectory identities conflict')
        else:
            selected_ids = [r['trajectory_id'] for r in self.trajectories if r['split'] == 'train']
        if asset_lock is not None:
            lock = json.loads(Path(asset_lock).read_text())
            identities = lock['dataset_identities']
            for name, actual_hash in self.manifest_hashes.items():
                if name not in identities or identities[name]['sha256'] != actual_hash:
                    raise ValueError(f'Generated manifest differs from asset lock: {name}')
            self.manifest_hashes['ASSET_LOCK.json'] = sha256_file(asset_lock)
        self.asset_lock_verified = asset_lock is not None
        self.trajectory_ids = sorted(selected_ids)
        if not self.trajectory_ids:
            raise ValueError('Empty selected train population')
        self.selected = [self.by_id[tid] for tid in self.trajectory_ids]
        for row in self.selected:
            self._validate_train_row(row)
        self.train_manifest = [{
            'trajectory_id': row['trajectory_id'], 'source_id': row['source_id'],
            'split': row['split'], 'clean': row['arrays']['clean'],
            'states': {str(g): row['arrays'][f'X_{g}'] for g in (1, 2, 3, 4)},
        } for row in self.selected]
        self.train_manifest_hash = canonical_hash(self.train_manifest)

    @staticmethod
    def _unique(rows: list[dict], key: str) -> dict:
        result = {}
        for row in rows:
            identity = row[key]
            if identity in result:
                raise ValueError(f'Duplicate or conflicting {key}: {identity}')
            result[identity] = row
        return result

    def _validate_train_row(self, row: dict):
        source = self.source_by_id.get(row['source_id'])
        if row['split'] != 'train' or source is None or source['split'] != 'train':
            raise ValueError('Non-train trajectory/source in selected train population')
        if row.get('source', source) != source:
            raise ValueError('Embedded source record conflicts with source manifest')
        plan = row.get('plan', {})
        for key in ('trajectory_id', 'source_id', 'split'):
            if key in plan and plan[key] != row[key]:
                raise ValueError(f'Conflicting plan {key}')
        if set(plan.get('g_values', [1, 2, 3, 4])) != {1, 2, 3, 4}:
            raise ValueError('Train trajectory must declare exactly g=1,2,3,4')
        if row['arrays']['clean'] != source['clean']:
            raise ValueError('Clean array reference conflicts with source manifest')
        expected_states = {f'X_{g}' for g in (1, 2, 3, 4)}
        if {name for name in row['arrays'] if name.startswith('X_')} != expected_states:
            raise ValueError('Train trajectory must contain exactly g=1,2,3,4')
        for name in ['clean', *sorted(expected_states)]:
            record = row['arrays'][name]
            if record.get('dtype') != 'float32' or record['shape'] != [3, 256, 256]:
                raise ValueError('Training/statistical view requires RGB FP32 full 256 x 256')
        preprocessing = source.get('preprocessing', {})
        if (preprocessing.get('layout') != 'CHW'
                or preprocessing.get('convert_mode') != 'RGB'
                or preprocessing.get('dtype') != 'float32'
                or preprocessing.get('scale') != 'uint8 / 255.0'):
            raise ValueError('Source preprocessing is incompatible with RGB units')

    def sample(self, trajectory_id: str, G: int) -> dict:
        if trajectory_id not in self.trajectory_ids:
            raise ValueError('Sample outside selected train population')
        if isinstance(G, bool) or G not in (1, 2, 3, 4):
            raise ValueError('G must be 1,2,3,4')
        row = self.by_id[trajectory_id]
        return {
            'C': torch.from_numpy(read_array(self.root, row['arrays']['clean'])),
            'Y': torch.from_numpy(read_array(self.root, row['arrays'][f'X_{G}'])),
            'G': int(G), 'trajectory_id': trajectory_id, 'source_id': row['source_id'],
        }

    def iter_stat_states(self) -> Iterator[tuple[dict, np.ndarray]]:
        for row in self.selected:
            for g in (1, 2, 3, 4):
                record = row['arrays'][f'X_{g}']
                state = {
                    'trajectory_id': row['trajectory_id'], 'source_id': row['source_id'],
                    'split': 'train', 'g': g, 'view_id': self.preprocessing['view_id'],
                    'crop_box_xyxy': [0, 0, 256, 256], 'crop_seed': None,
                    'array': record,
                }
                yield state, read_array(self.root, record)

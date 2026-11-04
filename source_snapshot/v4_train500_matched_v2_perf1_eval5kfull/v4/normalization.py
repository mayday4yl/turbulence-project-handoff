"""Train-only, four-strength RGB population statistics and frozen noise mixing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .data import PhysicalDataset, canonical_hash, sha256_file

PROTOCOL = 'v4_rgb_train_equal_g_full_view_chan_fp64_v1'


class ChannelMoments:
    """FP64 Chan merge of pixel blocks; variance always has correction=0."""
    def __init__(self):
        self.n = 0
        self.mean = np.zeros(3, dtype=np.float64)
        self.M2 = np.zeros(3, dtype=np.float64)

    def update(self, pixels: np.ndarray):
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[0] != 3 or pixels.shape[1] == 0:
            raise ValueError('A pixel block must have nonempty shape [3,N]')
        if not np.isfinite(pixels).all():
            raise ValueError('Non-finite statistics input')
        count = pixels.shape[1]
        block_mean = pixels.mean(axis=1, dtype=np.float64)
        centered = pixels - block_mean[:, None]
        block_M2 = np.sum(centered * centered, axis=1, dtype=np.float64)
        delta = block_mean - self.mean
        total = self.n + count
        self.M2 += block_M2 + delta * delta * (self.n * count / total)
        self.mean += delta * (count / total)
        self.n = total
        if not np.isfinite(self.mean).all() or not np.isfinite(self.M2).all():
            raise ValueError('Non-finite statistics accumulator')

    @property
    def variance(self):
        if self.n == 0:
            raise ValueError('Cannot compute variance of an empty group')
        return self.M2 / self.n


class StratifiedRGBStatistics:
    """Count each train (trajectory_id,g) once, pixels within g, equal g weights."""
    def __init__(self, allowed_train_trajectories: Iterable[str], block_pixels: int = 65536):
        ids = list(allowed_train_trajectories)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Train trajectory identities must be nonempty and unique')
        if block_pixels < 1:
            raise ValueError('block_pixels must be positive')
        self.allowed = set(ids)
        self.block_pixels = block_pixels
        self.groups = {g: ChannelMoments() for g in (1, 2, 3, 4)}
        self.state_counts = {g: 0 for g in self.groups}
        self.seen = set()

    def add(self, trajectory_id: str, g: int, image: np.ndarray, *, split: str):
        if split != 'train' or trajectory_id not in self.allowed:
            raise ValueError('Non-train or unauthorized statistics state')
        if isinstance(g, bool) or g not in self.groups:
            raise ValueError('Statistics strength must be g=1,2,3,4')
        key = (trajectory_id, g)
        if key in self.seen:
            raise ValueError('Duplicate or conflicting statistics state')
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[0] != 3 or image.size == 0:
            raise ValueError('State must be nonempty RGB CHW')
        if not np.isfinite(image).all():
            raise ValueError('Non-finite statistics input')
        pixels = image.reshape(3, -1)
        for start in range(0, pixels.shape[1], self.block_pixels):
            self.groups[g].update(pixels[:, start:start + self.block_pixels])
        self.seen.add(key)
        self.state_counts[g] += 1

    def finalize(self) -> dict:
        expected = {(tid, g) for tid in self.allowed for g in self.groups}
        if self.seen != expected:
            raise ValueError('Missing statistics states: every train trajectory requires all four g')
        means = np.stack([self.groups[g].mean for g in self.groups])
        variances = np.stack([self.groups[g].variance for g in self.groups])
        mu = means.mean(axis=0, dtype=np.float64)
        variance = (variances + (means - mu) ** 2).mean(axis=0, dtype=np.float64)
        std = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-6)
        if not np.isfinite(np.stack([mu, variance, std])).all():
            raise ValueError('Non-finite finalized statistics')
        return {
            'mu': mu.tolist(), 'variance': variance.tolist(), 'std': std.tolist(),
            'groups': {str(g): {
                'mean': self.groups[g].mean.tolist(),
                'variance': self.groups[g].variance.tolist(),
                'M2': self.groups[g].M2.tolist(),
                'pixel_count_per_channel': [self.groups[g].n] * 3,
                'unique_state_count': self.state_counts[g], 'weight': 0.25,
            } for g in self.groups},
            'accumulator_dtype': 'float64', 'algorithm': 'Welford/Chan block merge',
            'correction': 0, 'std_floor': 1e-6, 'within_g_weight': 'pixel',
            'g_weights': [0.25, 0.25, 0.25, 0.25],
            'buffer_dtype': 'float32', 'state_count': len(self.seen),
        }


def save_artifact(payload: dict, path: str | Path) -> dict:
    payload = {key: value for key, value in payload.items() if key != 'artifact_hash'}
    payload['artifact_hash'] = canonical_hash(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + '\n')
    return payload


def load_artifact(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    payload = {key: item for key, item in value.items() if key != 'artifact_hash'}
    if value.get('artifact_hash') != canonical_hash(payload):
        raise ValueError('Normalization artifact content hash mismatch')
    if value.get('protocol') != PROTOCOL:
        raise ValueError('Unknown normalization protocol')
    mu = np.asarray(value['mu'], dtype=np.float64)
    variance = np.asarray(value['variance'], dtype=np.float64)
    std = np.asarray(value['std'], dtype=np.float64)
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (mu, variance, std)):
        raise ValueError('Invalid normalization vectors')
    if (variance < 0).any() or (std < 1e-6).any():
        raise ValueError('Invalid variance/std in normalization artifact')
    if not np.allclose(std, np.maximum(np.sqrt(variance), 1e-6), rtol=1e-14, atol=0):
        raise ValueError('Normalization std differs from variance')
    if value.get('correction') != 0 or value.get('accumulator_dtype') != 'float64':
        raise ValueError('Invalid statistics accumulation protocol')
    if value.get('g_weights') != [0.25] * 4:
        raise ValueError('Normalization must use equal g weights')
    groups = value['groups']
    if set(groups) != {'1', '2', '3', '4'}:
        raise ValueError('Normalization artifact missing a strength group')
    counts = []
    for group in groups.values():
        if (group['unique_state_count'] < 1
                or min(group['pixel_count_per_channel']) < 1
                or group['weight'] != 0.25):
            raise ValueError('Invalid group counts or weight')
        counts.append(group['unique_state_count'])
    if sum(counts) != value['state_count']:
        raise ValueError('Normalization state counts disagree')
    group_means = np.asarray([groups[str(g)]['mean'] for g in (1, 2, 3, 4)])
    group_vars = np.asarray([groups[str(g)]['variance'] for g in (1, 2, 3, 4)])
    if (not np.isfinite(group_means).all() or not np.isfinite(group_vars).all()
            or (group_vars < 0).any()):
        raise ValueError('Invalid group moments')
    if not np.allclose(mu, group_means.mean(0), rtol=1e-14, atol=1e-15):
        raise ValueError('Artifact mean differs from equal-g moment reduction')
    expected_variance = (group_vars + (group_means - mu) ** 2).mean(0)
    if not np.allclose(variance, expected_variance, rtol=1e-14, atol=1e-15):
        raise ValueError('Artifact variance omits or corrupts between-g variation')
    return value


def build_artifact(dataset_root: str | Path, output_path: str | Path,
                   population: str = 'overfit100', block_pixels: int = 65536,
                   asset_lock: str | Path | None = None) -> dict:
    if asset_lock is None:
        raise ValueError('Real-data statistics require an explicit generated-manifest asset lock')
    dataset = PhysicalDataset(dataset_root, population, asset_lock=asset_lock)
    accumulator = StratifiedRGBStatistics(dataset.trajectory_ids, block_pixels)
    states = []
    for state, array in dataset.iter_stat_states():
        accumulator.add(state['trajectory_id'], state['g'], array, split=state['split'])
        states.append(state)
    payload = {
        'protocol': PROTOCOL, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_id': dataset.dataset_id, 'dataset_root': str(dataset.root),
        'population': population, 'split': 'train',
        'manifest_sha256': dataset.manifest_hashes,
        'asset_lock_verified': dataset.asset_lock_verified,
        'train_manifest': dataset.train_manifest,
        'train_manifest_hash': dataset.train_manifest_hash,
        'statistics_state_manifest': states, 'statistics_state_manifest_hash': canonical_hash(states),
        'preprocessing': dataset.preprocessing,
        'source_preprocessing': {
            sid: dataset.source_by_id[sid]['preprocessing']
            for sid in sorted({row['source_id'] for row in dataset.selected})
        },
        'statistics_code_sha256': {
            'normalization.py': sha256_file(__file__),
            'data.py': sha256_file(Path(__file__).with_name('data.py')),
        },
        'block_pixels': block_pixels, **accumulator.finalize(),
    }
    saved = save_artifact(payload, output_path)
    load_artifact(output_path)
    return saved


class FrozenRGBStats(nn.Module):
    """Persistent nontrainable statistics; direct RGB mixing preserves input units."""
    def __init__(self, mu, std, *, artifact_hash: str = ''):
        super().__init__()
        mu = torch.as_tensor(mu, dtype=torch.float32).reshape(1, 3, 1, 1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)
        if not torch.isfinite(mu).all() or not torch.isfinite(std).all() or (std < 1e-6).any():
            raise ValueError('Non-finite or invalid frozen RGB statistics')
        if artifact_hash and len(artifact_hash) != 64:
            raise ValueError('Artifact hash must be SHA256 hex')
        self.register_buffer('mu', mu)
        self.register_buffer('s', std)
        self.register_buffer('artifact_hash_bytes', torch.tensor(
            list(bytes.fromhex(artifact_hash or '0' * 64)), dtype=torch.uint8))

    @classmethod
    def from_artifact(cls, path: str | Path):
        artifact = load_artifact(path)
        return cls(artifact['mu'], artifact['std'], artifact_hash=artifact['artifact_hash'])

    @property
    def artifact_hash(self) -> str:
        return bytes(self.artifact_hash_bytes.detach().cpu().tolist()).hex()

    def mix(self, X: torch.Tensor, eta: torch.Tensor, t) -> torch.Tensor:
        if X.ndim != 4 or X.shape[1] != 3 or eta.shape != X.shape:
            raise ValueError('Noise and physical RGB state must share [B,3,H,W] shape')
        if not torch.isfinite(X).all() or not torch.isfinite(eta).all():
            raise ValueError('Non-finite noise mixing input')
        t = torch.as_tensor(t, dtype=X.dtype, device=X.device)
        if t.ndim == 0:
            t = t.expand(X.shape[0])
        if t.numel() != X.shape[0] or not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError('One finite t in [0,1] is required per sample')
        t = t.reshape(-1, 1, 1, 1)
        mu, s = self.mu.to(dtype=X.dtype), self.s.to(dtype=X.dtype)
        Z = (1 - t) * X + t * mu + s * t * eta
        if not torch.isfinite(Z).all():
            raise ValueError('Non-finite noise mixing output')
        return Z

    def forward(self, X: torch.Tensor, eta: torch.Tensor, t) -> torch.Tensor:
        return self.mix(X, eta, t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--asset-lock', required=True)
    parser.add_argument('--population', choices=['overfit100', 'train'], default='overfit100')
    parser.add_argument('--block-pixels', type=int, default=65536)
    args = parser.parse_args()
    artifact = build_artifact(args.dataset_root, args.output, args.population, args.block_pixels, args.asset_lock)
    print(json.dumps({key: artifact[key] for key in
                      ['protocol', 'dataset_id', 'population', 'mu', 'variance', 'std',
                       'state_count', 'artifact_hash']}, indent=2))
    print(json.dumps({'artifact_file_sha256': sha256_file(args.output), 'output': args.output}))


if __name__ == '__main__':
    main()

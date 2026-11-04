"""Deterministic clean/Y4 dataset. True raw xi is not exposed to training."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .common import rows, sha, verify_array
except ImportError:
    from common import rows, sha, verify_array


class V5Dataset(Dataset):
    def __init__(self, root, split, require_ready=True, allow_test=False):
        self.root = Path(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        if split == 'test' and not allow_test:
            raise ValueError('Test is locked until final selection; explicit allow_test is required')
        self.split = split
        freeze = json.loads((self.root / 'SOURCE_FREEZE.json').read_text())
        if sha(self.root / 'sample_plan.jsonl') != freeze['sample_plan_sha256']:
            raise ValueError('Sample plan hash mismatch')
        if sha(self.root / 'source_manifest.jsonl') != freeze['source_manifest_sha256']:
            raise ValueError('Source manifest hash mismatch')
        if require_ready:
            ready = json.loads((self.root / 'DATA_READY.json').read_text())
            if ready['status'] != 'DATA_READY' or ready['source_freeze_sha256'] != sha(self.root / 'SOURCE_FREEZE.json'):
                raise ValueError('Dataset READY identity mismatch')
            if ready['sample_manifest_sha256'] != sha(self.root / 'sample_manifest.jsonl'):
                raise ValueError('Generated observation manifest hash mismatch')
        self.samples = [row for row in rows(self.root / 'sample_plan.jsonl') if row['split'] == split]
        if len(self.samples) != freeze['expected_samples'][split]:
            raise ValueError('Dataset size mismatch')
        if any(row['G'] != 4 for row in self.samples):
            raise ValueError('V5-v1 active dataset requires fixed G=4')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        # No online crop, flip, clamp, augmentation, or random RNG is used here.
        clean = verify_array(self.root / row['clean_ref'])
        observation = verify_array(self.root / row['observation_ref'])
        return {'clean': torch.from_numpy(clean), 'observation': torch.from_numpy(observation),
                'G': 4.0, 'strength': 4.0, 'sample_id': row['sample_id'],
                'realization_id': row['sample_id'], 'source_id': row['source_id'],
                'realization_index': row['realization_index']}

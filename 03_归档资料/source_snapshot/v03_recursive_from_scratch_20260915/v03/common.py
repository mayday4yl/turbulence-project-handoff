"""V0.3 adapters around the unchanged, hash-pinned V0.1 implementation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def import_base(config):
    source = Path(config['base_source']).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import physical_bridge
    if not Path(physical_bridge.__file__).resolve().is_relative_to(source):
        raise RuntimeError('Imported a different physical_bridge source')


def verify_base(config):
    import_base(config)
    expected = config['base_v01_hashes']
    actual = {name: sha256_file(Path(config['base_source']) / name) for name in expected}
    if actual != expected:
        raise RuntimeError('V0.1 authority differs from audited hashes')
    return actual


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if hasattr(torch, 'npu'):
        torch.npu.manual_seed_all(int(seed))


def build_model(config, device):
    import_base(config)
    from physical_bridge.model import DualHeadUNet
    spec = config['model']
    return DualHeadUNet(**{key: spec[key] for key in (
        'base_channels', 'channel_multipliers', 'residual_blocks_per_level',
        'group_norm_groups', 'severity_embedding_dim')}).to(device)


def build_simulator(config, device):
    import_base(config)
    from physical_bridge.simulator import PiRNP2SSimulator
    spec = config['simulator']
    kwargs = {name: spec[name] for name in (
        'image_size', 'rf_range', 'aperture_diameter', 'propagation_distance',
        'wavelength', 'correlation', 'tilt_threshold', 'adjustment')}
    simulator = PiRNP2SSimulator(spec['asset_dir'], **kwargs, device=device)
    simulator.eval()
    for parameter in simulator.parameters():
        parameter.requires_grad_(False)
    return simulator


def derived_rng(seed, kind, sample_id):
    value = f'{int(seed)}:{kind}:{int(sample_id)}'.encode()
    return random.Random(int.from_bytes(hashlib.sha256(value).digest(), 'big'))


def unique_schedules(config, severity):
    import_base(config)
    from physical_bridge.data import evenly_spaced_schedule
    return tuple(dict.fromkeys(tuple(evenly_spaced_schedule(severity, k)) for k in (0, 1, 3, 5)))


def curriculum_stage(config, update):
    fraction = int(update) / max(1, int(config['training']['maximum_updates']))
    for index, stage in enumerate(config['curriculum']):
        if fraction < float(stage['until_fraction']):
            return index, stage
    return len(config['curriculum']) - 1, config['curriculum'][-1]


_RECORD_CACHE = {}


def train_records(config):
    path = Path(config['dataset_root']) / 'manifests/train.jsonl'
    key = str(path.resolve())
    if key not in _RECORD_CACHE:
        _RECORD_CACHE[key] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return _RECORD_CACHE[key]


def choose_sample(config, sample_id, update):
    import_base(config)
    records = train_records(config)
    seed = int(config['sampling_seed'])
    record_index = derived_rng(seed, 'trajectory', sample_id).randrange(len(records))
    severity = (0.75, 1.5, 2.25, 3.0, 3.75, 4.5)[derived_rng(seed, 'G', sample_id).randrange(6)]
    override = config.get('workload', {})
    if override.get('kind') == 'longest':
        severity = 4.5
        from physical_bridge.data import evenly_spaced_schedule
        schedule = tuple(evenly_spaced_schedule(severity, 5))
        stage_index = -1
    else:
        candidates = unique_schedules(config, severity)
        stage_index, stage = curriculum_stage(config, update)
        weights = [float(stage['weights_by_effective_K'].get(str(len(path)-1), 0)) for path in candidates]
        if sum(weights) <= 0:
            raise ValueError('Every G must have positive mass on a legal actual schedule')
        schedule = derived_rng(seed, 'path', sample_id).choices(candidates, weights=weights, k=1)[0]
    return {'sample_id': int(sample_id), 'key': records[record_index]['key'],
            'record_index': record_index, 'G': severity, 'schedule': schedule,
            'curriculum_stage': stage_index}


class TrainingDataset:
    """Uniform existing trajectory and independent uniform terminal G, train split only."""
    def __init__(self, config):
        import_base(config)
        from physical_bridge.data import BridgeArchive
        self.config = config
        self.archive = BridgeArchive(config['dataset_root'], 'train')

    def __len__(self):
        return 2**31 - 1

    def __getitem__(self, index):
        sample_id, update = index
        selected = choose_sample(self.config, sample_id, update)
        record = self.archive.records[selected['record_index']]
        return {**selected, 'C': self.archive.clean(record),
                'Y': self.archive.state(record, selected['G'])}

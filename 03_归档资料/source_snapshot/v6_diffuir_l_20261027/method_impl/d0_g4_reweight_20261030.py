"""Bookkeeping for a finite, paired sampler experiment; native D0 is unchanged.

Serial execution retains the old four logical rank slots and three microsteps:
the same 60 positions, microbatch 5, and per-slot native diffusion seeds.
"""
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from grad_helpers import capture_rng, restore_rng
from method_impl.d0 import derive_seed
from method_impl.training_io import PositionDataset
from scripts.multistrength_order_20261029 import MixedOrder
from scripts.runtime_utils import ROOT, sha

EX = ROOT / 'execution_20261030_d0_g4_reweight_5k'
OLD_EX = ROOT / 'execution_20261029_d0_multistrength160k'
BASE = 157505


def config():
    return json.loads((EX / 'CONFIG.json').read_text())


def freeze():
    for filename, base in [('SOURCE_MANIFEST.json', EX), ('SOURCE_MANIFEST.json', OLD_EX)]:
        for name, expected in json.loads((base / filename).read_text())['files'].items():
            assert sha(ROOT / name) == expected, ('Source identity', name)
    assert sha(EX / 'CONFIG.json') == json.loads((EX / 'CONFIG_MANIFEST.json').read_text())['config_sha256']
    for name, expected in json.loads((EX / 'SUPPORT_SOURCE_MANIFEST.json').read_text())['files'].items():
        assert sha(Path(name)) == expected, ('Support source identity', name)


def load_pool_and_lookup():
    from scripts.multistrength_support_20261029 import load_pool
    pool, identity = load_pool()
    lookup = {}
    for i, row in enumerate(pool):
        # Uppercase G is frozen legacy source metadata; lowercase g is the
        # generated state strength, as documented in DATA_FIELD_CONVENTION.
        key = (row['sample_id'], int(row['g']))
        assert key not in lookup
        lookup[key] = i
    assert Counter(int(r['g']) for r in pool) == {g: 40000 for g in [1, 2, 3, 4]}
    for sample_id in {r['sample_id'] for r in pool}:
        rows = [pool[lookup[(sample_id, g)]] for g in [1, 2, 3, 4]]
        for field in ['source_id', 'realization_index', 'clean_path', 'clean_sha256',
                      'seed_tilt', 'seed_zernike', 'raw_sha256']:
            assert all(r[field] == rows[0][field] for r in rows), (sample_id, field)
    assert len(lookup) == 160000
    return pool, lookup, identity


class ReweightedOrder:
    def __init__(self, pool, lookup, branch, cfg):
        assert branch in ['Control', 'G4_biased']
        self.pool, self.lookup, self.branch, self.cfg = pool, lookup, branch, cfg
        self.original = MixedOrder(pool, cfg['seed'])

    def pair(self, position):
        original_index = self.original.at(position)
        row = self.pool[original_index]
        index = original_index
        if self.branch == 'G4_biased' and int(row['g']) != 4:
            if derive_seed(self.cfg['reweight_seed'], 'promote_nonG4', position) % 3 == 0:
                index = self.lookup[(row['sample_id'], 4)]
        selected = self.pool[index]
        assert selected['sample_id'] == row['sample_id']
        assert selected['clean_sha256'] == row['clean_sha256'] and selected['raw_sha256'] == row['raw_sha256']
        return original_index, index

    def at(self, position):
        return self.pair(position)[1]


class SerialDataset(PositionDataset):
    def __init__(self, pool, lookup, cfg, branch, cursor, count):
        self.pool = pool
        self.order = ReweightedOrder(pool, lookup, branch, cfg)
        self.cursor, self.count = cursor, count
        self.rank, self.world, self.microbatch = 0, 1, 5
        self.verified = set()


def restore_virtual(rngs, virtual_rank, device, verify=False):
    saved = rngs[virtual_rank]
    assert len(saved['device']) == 4
    mapped = copy.deepcopy(saved)
    mapped['device'] = [saved['device'][virtual_rank].clone()]
    restore_rng(mapped, device)
    if verify:
        from scripts.audit_ddp_resume import equal_tree
        assert equal_tree(capture_rng(device), mapped), 'Virtual RNG restore mismatch'


def capture_virtual(rngs, virtual_rank, device):
    actual = capture_rng(device)
    assert len(actual['device']) == 1
    devices = copy.deepcopy(rngs[virtual_rank]['device'])
    devices[virtual_rank] = actual['device'][0].clone()
    actual['device'] = devices
    rngs[virtual_rank] = actual


def logical_slot(serial_micro):
    assert 0 <= serial_micro < 12
    return serial_micro // 4, serial_micro % 4


def branch_identity(parent, cfg, branch, physical):
    return dict(parent['identity'], physical_devices=[physical], logical_parent_devices=[4, 5, 6, 7],
                original_virtual_ranks=4, physical_world_size=1,
                rng_layout='four parent logical-rank states; active device index remapped to local0',
                experiment='D0_Mixed_G4_reweight_5k', branch=branch,
                config_sha256=sha(EX / 'CONFIG.json'), source_manifest_sha256=sha(EX / 'SOURCE_MANIFEST.json'),
                parent_checkpoint_sha256=cfg['parent_sha256'])


def image_pair(pool, index):
    row = pool[index]
    values = []
    for field in ['observation', 'clean']:
        path = Path(row[field + '_path'])
        assert sha(path) == row[field + '_sha256']
        a = np.load(path, allow_pickle=False)
        assert a.dtype == np.float32 and a.shape == (3, 256, 256) and np.isfinite(a).all()
        if field == 'clean':
            assert a.min() >= 0 and a.max() <= 1
        values.append(torch.from_numpy(a))
    return values


def parent_s3_probe(model, ema, device, destination):
    """Five existing fixed-source parent outputs; no fresh full-Val claim."""
    from scripts.multistrength_support_20261029 import val_rows
    from method_impl.evaluation import BASE_SEED
    from scripts.runtime_utils import write_json
    valdir = ROOT / 'distributed/V6_D0_U95544_multistrength160k_formal01/validation/U157505_1000'
    records = [json.loads(line) for line in (valdir / 'PER_IMAGE.jsonl').read_text().splitlines()]
    visual_ids = sorted({r['sample_id'] for r in records if 'prediction_npy' in r})
    sample_id = visual_ids[0]
    row = next(r for r in val_rows(True) if r['sample_id'] == sample_id)
    native, sampling = model.native.model, model.sampling.model
    model.native.model = ema.model
    model.sampling.model = ema.model
    calls = []
    hook = ema.model.register_forward_hook(lambda m, a, o: calls.append(1))
    result = []
    try:
        for g in [.5, 1, 2, 3, 4]:
            ref = next(r for r in records if r['sample_id'] == sample_id and r['g'] == g)
            saved = Path(ref['prediction_npy'])
            assert sha(saved) == ref['prediction_sha256']
            entry = row['states'][f'{g:g}']
            assert sha(entry['path']) == entry['sha256']
            x = torch.from_numpy(np.load(entry['path'], allow_pickle=False)[None]).to(device)
            seed = derive_seed(BASE_SEED, row['source_id'], row['sample_id'], 'outer', 0)
            calls.clear()
            with torch.no_grad():
                y = model.sample_clean(x, seed=seed, differentiable=False)
            assert len(calls) == 3 and bool(torch.isfinite(y).all())
            error = float(np.max(np.abs(y[0].cpu().numpy() - np.load(saved, allow_pickle=False))))
            assert error <= 1e-6, ('Parent inference changed', g, error)
            result.append(dict(g=g, sample_id=sample_id, nfe=3, maxabs=error, reference_sha256=sha(saved)))
    finally:
        hook.remove()
        model.native.model, model.sampling.model = native, sampling
    write_json(destination, dict(status='PASS_FIVE_FIXED_S3_PARENT_REPLAYS', records=result,
                                 fresh_full_5000_validation_executed=False, optimizer_updates=0))
    return result

from collections import OrderedDict
import json
from pathlib import Path
import random

import numpy as np
import torch
from v4.data import PhysicalDataset, read_array
from v4.normalization import load_artifact
from .common import ROOT, atomic_json, canonical, digest, seed_for


class Store:
    def __init__(self, config):
        self.dataset = PhysicalDataset(config['dataset_root'], 'train', Path(config['dataset_root'])/'ASSET_LOCK.json')
        self.root = self.dataset.root
        if self.dataset.dataset_id != config['dataset_id']: raise ValueError('Dataset identity mismatch')
        stats = load_artifact(ROOT/config['normalization_artifact'])
        if (stats['population'] != 'train' or stats['state_count'] != 32000 or
            stats['train_manifest_hash'] != self.dataset.train_manifest_hash or
            stats['manifest_sha256'] != self.dataset.manifest_hashes):
            raise ValueError('Train2000 statistics binding mismatch')
        self.sources = self.dataset.source_by_id
        self.train = self.records(self.dataset.selected)
        vals = [r for r in self.dataset.trajectories if r['split'] == 'val']
        for r in vals:
            source = self.sources[r['source_id']]
            if source['split'] != 'val' or source['clean'] != r['arrays']['clean']:
                raise ValueError('Validation identity mismatch')
        self.val = self.records(vals)
        if len(self.train) != 32000 or len(self.val) != 2000: raise ValueError('Unexpected population sizes')
        train_ids = {r['source_id'] for r in self.train}; val_ids = {r['source_id'] for r in self.val}
        if len(train_ids) != 2000 or len(val_ids) != 250 or train_ids & val_ids:
            raise ValueError('Source split/count invalid')
        self.by_id = {r['start_id']:r for r in self.train+self.val}
        self.lists = {'train32000':self.train, 'val2000':self.val}
        self.cache = OrderedDict(); self.cache_bytes = 0
        self.cache_limit = int(config.get('cache_gib', 2) * 1024**3)
        self.seed = config.get('stream_seed', 20261003)

    @staticmethod
    def records(trajectories):
        return [{'start_id':f"{r['trajectory_id']}::G{g}", 'trajectory_id':r['trajectory_id'],
                 'source_id':r['source_id'], 'split':r['split'], 'G':g,
                 'clean':r['arrays']['clean'], 'state':r['arrays'][f'X_{g}']}
                for r in sorted(trajectories, key=lambda r:r['trajectory_id']) for g in (1,2,3,4)]

    def array(self, record):
        key = (record['ref'], record['sha256'])
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        value = torch.from_numpy(read_array(self.root, record)); size=value.numel()*value.element_size()
        while self.cache and self.cache_bytes+size > self.cache_limit:
            _, old=self.cache.popitem(last=False); self.cache_bytes-=old.numel()*old.element_size()
        self.cache[key]=value; self.cache_bytes+=size
        return value

    def batch(self, rows, device):
        if not rows or any(r['split'] not in ('train','val') for r in rows): raise ValueError('Test access forbidden')
        return (torch.stack([self.array(r['clean']) for r in rows]).to(device),
                torch.stack([self.array(r['state']) for r in rows]).to(device),
                torch.tensor([r['G'] for r in rows], device=device, dtype=torch.float32))


class Stream:
    """Same G-balanced deterministic starts for all methods, one pass per epoch."""
    def __init__(self, store):
        self.store=store; self.cached_epoch=None; self.batches=None
        self.identity=canonical({'seed':store.seed,'algorithm':'sha256_python_random_balanced_G_v1',
                                 'train_ids':[r['start_id'] for r in store.train],
                                 'epochs':20,'updates_per_epoch':1000,'batch':32})

    def rows(self, completed_updates):
        if not 0 <= completed_updates < 20000: raise ValueError('Stream exhausted')
        epoch=completed_updates//1000+1
        if self.cached_epoch != epoch:
            orders={g:[r for r in self.store.train if r['G']==g] for g in (1,2,3,4)}
            for g in orders:
                random.Random(seed_for(f'stream|{self.store.seed}|epoch{epoch}|G{g}')).shuffle(orders[g])
            batches=[]
            for i in range(1000):
                rows=[r for g in orders for r in orders[g][i*8:(i+1)*8]]
                random.Random(seed_for(f'stream|{self.store.seed}|epoch{epoch}|batch{i}')).shuffle(rows)
                batches.append(rows)
            self.batches=batches;self.cached_epoch=epoch
        return self.batches[completed_updates%1000]


def eta_batch(rows, device, epoch=None, seed=20261003):
    # CPU keyed generators make eta independent of microbatch shape and device RNG.
    namespace='eval' if epoch is None else f'train|epoch{epoch}'
    return torch.stack([torch.randn((3,256,256),dtype=torch.float32,
        generator=torch.Generator(device='cpu').manual_seed(seed_for(f'{namespace}|{seed}|{r["start_id"]}')))
        for r in rows]).to(device)

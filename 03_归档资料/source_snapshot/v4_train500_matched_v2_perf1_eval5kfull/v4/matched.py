"""Matched-update protocol. No training loop or automatic launch on import."""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from physical_bridge.simulator import PiRNP2SSimulator
from .data import PhysicalDataset, canonical_hash, read_array
from .model import NoiseBridgeUNet
from .normalization import FrozenRGBStats, load_artifact
from .runtime import sha256, save_json
from .training import RecordStore as HistoricalRecordStore, TrainingObjective

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260929
GROUP_GPU = {'R08':5, 'D0':5, 'R0':6, 'L08':6}
INITIAL_SHA = '2e6a29eb895de090bbaeaf1cfdd1aedd5550703fa1bb9bb80e099ca2fd920fc2'
ETA_NAMESPACE = 'train|20260929'


def seed_for(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63-1)


def eta_batch(rows, device, epoch=None):
    keys = [f'eval|{SEED}|{r["start_id"]}' if epoch is None else
            f'{ETA_NAMESPACE}|{epoch}|{r["start_id"]}' for r in rows]
    seeds = [seed_for(k) for k in keys]
    return torch.stack([torch.randn((3,256,256), device=device, dtype=torch.float32,
                                   generator=torch.Generator(device=device).manual_seed(s)) for s in seeds])


def lr_at(update):
    if not 1 <= update <= 10000:
        raise ValueError('Authorized LR schedule ends at U10000')
    if update <= 500:
        return 1e-4*update/500
    return 1e-5 + .5*(1e-4-1e-5)*(1+math.cos(math.pi*(update-500)/9500))


class Store:
    def __init__(self, config):
        self.dataset = PhysicalDataset(config['dataset_root'], 'train', config['asset_lock'])
        self.root = self.dataset.root
        stats = load_artifact(ROOT/config['normalization_artifact'])
        if (stats['population'] != 'train' or stats['state_count'] != 8000 or
            stats['train_manifest_hash'] != self.dataset.train_manifest_hash or
            stats['manifest_sha256'] != self.dataset.manifest_hashes):
            raise ValueError('Train500 normalization binding failed')
        self.train = HistoricalRecordStore.records(self.dataset.selected)
        vals = [r for r in self.dataset.trajectories if r['split']=='val']
        for r in vals:
            source = self.dataset.source_by_id[r['source_id']]
            if source['split']!='val' or source['clean']!=r['arrays']['clean']:
                raise ValueError('Validation identity mismatch')
        self.val = HistoricalRecordStore.records(vals)
        assert len(self.train)==8000 and len(self.val)==800
        assert len({r['source_id'] for r in self.train})==500
        assert len({r['source_id'] for r in self.val})==100
        assert not {r['source_id'] for r in self.train}&{r['source_id'] for r in self.val}
        self.by_id = {r['start_id']:r for r in self.train+self.val}
        plan = json.loads((ROOT/'protocol/PLANNED_MONITOR_LISTS.json').read_text())
        def resolve(key):
            result=[]
            for item in plan[key]:
                r=self.by_id[item['start_id']]
                if any(r[k]!=v for k,v in item.items()):raise ValueError('Fixed-list identity mismatch')
                result.append(r)
            return result
        self.lists = {'train8000':self.train, 'val800':self.val,
                      'train_monitor400':resolve('train_monitor400')}
        self.visual = resolve('visual_train20')+resolve('visual_val20')
        assert len(self.visual)==40
        self.visual_ids = {r['start_id'] for r in self.visual}
        self.cache=OrderedDict();self.cache_bytes=0
        self.cache_limit=2*1024**3 # pageable CPU only, zero pinned retention

    def array(self, record):
        key=(record['ref'],record['sha256'])
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        value=torch.from_numpy(read_array(self.root,record))
        size=value.numel()*value.element_size()
        while self.cache and self.cache_bytes+size>self.cache_limit:
            _,old=self.cache.popitem(last=False);self.cache_bytes-=old.numel()*old.element_size()
        self.cache[key]=value;self.cache_bytes+=size
        return value

    def batch(self, rows, device):
        if not rows or any(r['split'] not in ('train','val') for r in rows):
            raise ValueError('Empty/test batch forbidden')
        return (torch.stack([self.array(r['clean']) for r in rows]).to(device),
                torch.stack([self.array(r['state']) for r in rows]).to(device),
                torch.tensor([r['G'] for r in rows],device=device,dtype=torch.float32))


def make_stream(store):
    """Registered deterministic ordering; full actual stream saved before freeze."""
    by_g={g:[r['start_id'] for r in store.train if r['G']==g] for g in range(1,5)}
    stream=[]
    for epoch in range(1,41):
        orders={g:ids.copy() for g,ids in by_g.items()}
        for g in orders:
            random.Random(seed_for(f'stream|{SEED}|epoch{epoch}|G{g}')).shuffle(orders[g])
        for batch in range(250):
            chunk=[sid for g in range(1,5) for sid in orders[g][batch*8:(batch+1)*8]]
            random.Random(seed_for(f'stream|{SEED}|epoch{epoch}|batch{batch}')).shuffle(chunk)
            assert len(chunk)==32 and len(set(chunk))==32
            assert all(sum(store.by_id[s]['G']==g for s in chunk)==8 for g in range(1,5))
            stream.append(chunk)
        assert sorted(s for chunk in stream[-250:] for s in chunk)==sorted(r['start_id'] for r in store.train)
    return {'seed':SEED,'epoch_first':1,'updates':stream,'content_hash':canonical_hash(stream),
            'shuffle':'Python Random seeded by SHA256 stream|seed|epochN|Gg and stream|seed|epochN|batch0based',
            'eta_namespace':ETA_NAMESPACE,'effective_batch':32}


def build(config, group, device='cuda'):
    path=ROOT/'protocol/common_init.pt'
    if sha256(path)!=INITIAL_SHA:raise ValueError('Common initial weights identity mismatch')
    stats=FrozenRGBStats.from_artifact(ROOT/config['normalization_artifact'])
    model=NoiseBridgeUNet(stats,**config['network'])
    initial=torch.load(path,map_location='cpu',weights_only=True)
    # Replace only the old train100 buffers, never learned parameters.
    for key,value in model.state_dict().items():
        if key.startswith('stats.'):initial[key]=value.clone()
    model.load_state_dict(initial,strict=True)
    model=model.to(device)
    sim=PiRNP2SSimulator(config['asset_dir'],**config['physics'],device=device).to(device).eval()
    if any(p.requires_grad for p in sim.parameters()):raise ValueError('Simulator is not frozen')
    return model,sim,TrainingObjective(model,sim,group)


def backward32(objective, store, rows, microbatch, epoch):
    if len(rows)!=32 or microbatch not in (1,2):raise ValueError('Expected B32 and microbatch1/2')
    if any(sum(r['G']==g for r in rows)!=8 for g in range(1,5)):raise ValueError('Unbalanced G batch')
    device=next(objective.network.parameters()).device
    values=[]
    for offset in range(0,32,microbatch):
        micro=rows[offset:offset+microbatch];C,Y,G=store.batch(micro,device)
        result=objective(C,Y,G,eta=eta_batch(micro,device,epoch))
        if not torch.isfinite(result['path_losses']).all():raise FloatingPointError('Nonfinite training loss')
        (result['loss_mean']*(len(micro)/32)).backward()
        values.extend(zip(result['path_clean'].detach().cpu().tolist(),
                          result['path_reconstruction'].detach().cpu().tolist(),
                          result['path_losses'].detach().cpu().tolist()))
        del C,Y,G,result
    return np.asarray(values,dtype=np.float64).mean(0).tolist()

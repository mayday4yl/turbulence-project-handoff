"""Formal comparison primitives. Physical/model mathematics use the frozen V4 core."""
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .core import RolloutModule, mse, validate_inputs
from .data import PhysicalDataset, read_array, canonical_hash
from .normalization import load_artifact


GROUPS = {'R08': .8, 'R0': 0., 'D0': 0., 'L08': .8}


def learning_rate(exposures):
    if not 0 < exposures <= 32000:
        raise ValueError('First-segment LR requires 0 < E <= 32000')
    if exposures <= 1600:
        return 1e-4 * exposures / 1600
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * (exposures-1600)/30400))


def epoch_stream(size=1600, epochs=20, seed=20260916):
    rng = random.Random(seed)
    stream = []
    for _ in range(epochs):
        order = list(range(size))
        rng.shuffle(order)
        stream.extend(order)
    return stream


def evaluation_seed(start_id, seed=20260916):
    return int.from_bytes(hashlib.sha256(f'eval:{seed}:{start_id}'.encode()).digest()[:8], 'little') % (2**63-1)


class RecordStore:
    """Explicit train and validation manifests; only train enters statistics/updates."""
    def __init__(self, config):
        self.dataset = PhysicalDataset(config['dataset_root'], 'overfit100', config['asset_lock'])
        artifact=load_artifact(Path(__file__).resolve().parents[1]/config['normalization_artifact'])
        if (artifact['train_manifest_hash']!=self.dataset.train_manifest_hash or
            artifact['manifest_sha256']!=self.dataset.manifest_hashes or artifact['population']!='overfit100'):
            raise ValueError('Normalization artifact is not bound to the frozen training population')
        self.root = self.dataset.root
        self.cache = {}
        train = self.dataset.selected
        val = [r for r in self.dataset.trajectories if r['split'] == 'val']
        for row in val:
            source = self.dataset.source_by_id[row['source_id']]
            if source['split'] != 'val' or row['arrays']['clean'] != source['clean']:
                raise ValueError('Validation source/trajectory identity mismatch')
        self.train = self.records(train)
        self.val = self.records(val)
        if len(self.train) != 1600 or len(self.val) != 800:
            raise ValueError('Unexpected frozen comparison populations')
        if {r['source_id'] for r in self.train} & {r['source_id'] for r in self.val}:
            raise ValueError('Train/validation source overlap')
        self.lists = {'train1600':self.train, 'val800':self.val,
                      'train64':self.monitor(self.train), 'val64':self.monitor(self.val)}

    @staticmethod
    def records(trajectories):
        return [{'start_id':f"{r['trajectory_id']}::G{g}", 'trajectory_id':r['trajectory_id'],
                 'source_id':r['source_id'], 'split':r['split'], 'G':g,
                 'clean':r['arrays']['clean'], 'state':r['arrays'][f'X_{g}']}
                for r in sorted(trajectories,key=lambda r:r['trajectory_id']) for g in (1,2,3,4)]

    @staticmethod
    def monitor(records):
        sources = sorted({r['source_id'] for r in records})
        random.Random(20260916).shuffle(sources)
        selected = []
        for sid in sources[:16]:
            tid = min(r['trajectory_id'] for r in records if r['source_id']==sid)
            selected.extend(r for r in records if r['trajectory_id']==tid)
        return selected

    def array(self, record):
        key = record['ref']
        if key not in self.cache:
            self.cache[key] = torch.from_numpy(read_array(self.root,record)).pin_memory()
        return self.cache[key]

    def batch(self, records, device):
        if not records:
            raise ValueError('Empty batches are not supported by this frozen protocol')
        return (torch.stack([self.array(r['clean']) for r in records]).to(device,non_blocking=True),
                torch.stack([self.array(r['state']) for r in records]).to(device,non_blocking=True),
                torch.tensor([r['G'] for r in records],device=device,dtype=torch.float32))

    def manifest(self):
        return {'data_manifest_hashes':self.dataset.manifest_hashes,
                'lists':{key:[{k:r[k] for k in ('start_id','trajectory_id','source_id','split','G')} for r in rows]
                         for key,rows in self.lists.items()},
                'list_hashes':{key:canonical_hash(rows) for key,rows in self.lists.items()}}


class TrainingObjective(RolloutModule):
    def __init__(self, network, simulator, group):
        if group not in GROUPS:
            raise ValueError('Unknown comparison group')
        super().__init__(network,simulator,T0=GROUPS[group],path='linear' if group=='L08' else 'physical')
        self.group = group

    def forward(self, clean, observation, severities, *, noise_generator=None, eta=None,
                retain_intermediates=False):
        if self.group != 'D0':
            return super().forward(clean,observation,severities,noise_generator=noise_generator,
                                   eta=eta,retain_intermediates=retain_intermediates)
        G = validate_inputs(observation,severities)
        if clean.shape != observation.shape or clean.dtype != torch.float32 or not torch.isfinite(clean).all():
            raise ValueError('Invalid D0 clean target')
        with torch.autocast(device_type=observation.device.type,enabled=False):
            pred = self.network(torch.cat((observation,observation),1),torch.stack((G,torch.zeros_like(G)),1))
            rendered = self.simulator.render(pred.clean,pred.realization,G).output
            lc,lr = mse(pred.clean,clean),mse(rendered,observation)
        losses = lc+.2*lr
        return {'loss_mean':losses.mean(),'path_losses':losses,'path_clean':lc,'path_reconstruction':lr,
                'final_clean':pred.clean,'first_clean':pred.clean,'prediction_count':len(G),'transition_count':0,
                'observation_render_count':len(G),'training_refinement_count':0,'trace':[]}


def make_optimizer(network):
    excluded = set()
    for module in network.modules():
        if isinstance(module,(torch.nn.GroupNorm,torch.nn.LayerNorm,torch.nn.BatchNorm2d)):
            excluded.update(id(p) for p in module.parameters(recurse=False))
    decay,no_decay = [],[]
    for name,param in network.named_parameters():
        if param.requires_grad:
            (no_decay if name.endswith('bias') or id(param) in excluded else decay).append(param)
    return torch.optim.AdamW([{'params':decay,'weight_decay':1e-4},
                             {'params':no_decay,'weight_decay':0.}],
                            lr=1e-4,betas=(.9,.999),eps=1e-8)


def make_ema(network):
    ema = copy.deepcopy(network).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, network, batch_size):
    decay = .996 ** (batch_size/32)
    for target,source in zip(ema.parameters(),network.parameters(),strict=True):
        target.mul_(decay).add_(source,alpha=1-decay)
    for target,source in zip(ema.buffers(),network.buffers(),strict=True):
        target.copy_(source)
    return decay


def rng_state(noise_generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(),'eta':noise_generator.get_state()}


def set_rng_state(state,noise_generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu()); torch.cuda.set_rng_state(state['cuda'].cpu())
    noise_generator.set_state(state['eta'].cpu())


def backward_batch(wrapped, store, records, microbatch, noise_generator, *, etas=None):
    """DDP averages ranks; each rank contributes its equal share of the global sample mean."""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(records) % world or len(records)//world % microbatch:
        raise ValueError('Frozen batches must divide equally across ranks and microbatches; no padding/drop')
    local = records[rank::world]
    model = wrapped.module if hasattr(wrapped,'module') else wrapped
    device = next(model.network.parameters()).device
    totals = torch.zeros(4,4,device=device,dtype=torch.float64)
    for offset in range(0,len(local),microbatch):
        rows = local[offset:offset+microbatch]
        last = offset+microbatch == len(local)
        context = nullcontext() if last or not hasattr(wrapped,'no_sync') else wrapped.no_sync()
        with context:
            C,Y,G = store.batch(rows,device)
            eta = None if etas is None else etas[rank::world][offset:offset+microbatch].to(device)
            result = wrapped(C,Y,G,noise_generator=noise_generator,eta=eta)
            (result['loss_mean'] * (len(rows)/len(local))).backward()
        with torch.no_grad():
            for g in (1,2,3,4):
                mask = G==g
                totals[g-1,0] += mask.sum()
                for j,key in enumerate(('path_clean','path_reconstruction','path_losses'),1):
                    totals[g-1,j] += result[key][mask].double().sum()
        del result,C,Y,G,eta
    if world > 1:
        dist.all_reduce(totals)
    return totals


def finite_gradients(network):
    missing = [name for name,p in network.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise RuntimeError(f'Missing trainable gradients: {missing}')
    if any(not torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite training gradient')


def atomic_torch_save(payload,path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def evaluation_groups(records, batch_size, world_size=2):
    # Each global chunk belongs to one rank; manifest explicitly fixes tail handling.
    return [{'rank':index%world_size,'records':records[start:start+batch_size]}
            for index,start in enumerate(range(0,len(records),batch_size))]

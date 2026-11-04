"""V6 D0: exact official residual training plus true differentiable S3.

This module deliberately implements D0 only. A/B require the separate physical
adapter and the unresolved xi-head time-modulation definition.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import copy
import torch
from torch import nn
from common import build_model
from grad_helpers import capture_rng, restore_rng, seed_rng, sampling_gradients_enabled

OFFICIAL_SHA='b9d07ab6b32d37d895be65c50e99bbe6b2ae6e32823e7304ec2167bc5f18843f'

def derive_seed(*parts):
    import json
    return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':'),ensure_ascii=True).encode()).digest()[:8],'big') & ((1<<63)-1)

@contextmanager
def isolated_rng(seed, device):
    previous=capture_rng(device)
    try:
        seed_rng(seed,device)
        yield
    finally:
        restore_rng(previous,device)

class D0Restorer(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        path=Path(checkpoint_path);h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        if h.hexdigest()!=OFFICIAL_SHA:raise RuntimeError('Wrong official checkpoint')
        data=torch.load(path,map_location='cpu',weights_only=True)
        state={k.removeprefix('ema_model.'):v for k,v in data['ema'].items() if k.startswith('ema_model.')}
        self.native=build_model()
        self.native.load_state_dict(state,strict=True)
        assert all(torch.equal(v,state[k]) for k,v in self.native.state_dict().items())
        self.sampling=build_model()
        self.sampling.load_state_dict(state,strict=True)
        self.sampling.init()  # Exactly official inference lifecycle, on CPU once.
        self.sampling.model=self.native.model  # SAME current parameters in both branches.
        assert sum(p.numel() for p in self.parameters())==36261187
        assert len(list(self.parameters()))==276
        assert self.sampling.model is self.native.model
        self.initialization={'checkpoint_sha256':OFFICIAL_SHA,'branch':'official_ema_network',
            'native_schedule':'checkpoint_ema_buffers','sampling_schedule':'official_init_on_cpu_once',
            'shared_network':True,'physical_adapter':False,'xi_head':False,'simulator':False}

    @property
    def network(self):return self.native.model

    def residual_loss(self, observation, clean, *, seed):
        with isolated_rng(seed,observation.device):
            tau=torch.randint(0,1000,(len(observation),),device=observation.device).long()
            eps=torch.randn_like(clean)
            values=self.native.p_losses([clean,observation,['blur']*len(clean)],tau,noise=eps)
        if len(values)!=1:raise RuntimeError('Unexpected objective')
        return .5*values[0],tau

    def sample_clean(self, observation, *, seed, differentiable=True):
        with isolated_rng(seed,observation.device):
            if differentiable:
                with sampling_gradients_enabled(self.sampling):
                    images=self.sampling.sample(observation,last=True,task='blur')
            else:
                images=self.sampling.sample(observation,last=True,task='blur')
        if len(images)!=4:raise RuntimeError('Official S3 list changed')
        return images[-1]

    def forward(self, observation, clean, *, seed):
        if observation.shape!=clean.shape or observation.shape[1:]!=(3,256,256):raise ValueError('RGB256 pairs required')
        if observation.dtype!=torch.float32 or clean.dtype!=torch.float32:raise ValueError('FP32 only')
        residual,tau=self.residual_loss(observation,clean,seed=derive_seed(seed,'q'))
        predicted=self.sample_clean(observation,seed=derive_seed(seed,'s'),differentiable=True)
        clean_loss=(predicted-clean).abs().mean()
        return {'loss':residual+clean_loss,'residual_loss':residual,'clean_loss':clean_loss,'prediction':predicted,'tau':tau}

class ImmediateEMA:
    """Explicit beta=.995 at every optimizer step, with no copy warmup."""
    def __init__(self, network, beta=.995):
        self.model=copy.deepcopy(network).eval().requires_grad_(False)
        self.beta=float(beta);self.updates=0

    @torch.no_grad()
    def update(self, network):
        source=dict(network.named_parameters())
        for name,value in self.model.named_parameters():value.mul_(self.beta).add_(source[name],alpha=1-self.beta)
        buffers=dict(network.named_buffers())
        for name,value in self.model.named_buffers():value.copy_(buffers[name])
        self.updates+=1

    def state_dict(self):return {'model':self.model.state_dict(),'beta':self.beta,'updates':self.updates}

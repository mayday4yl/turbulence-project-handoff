"""Unmodified official UniDB network/loss/process with new run bookkeeping."""
import hashlib
import importlib.util
import json
import os
import random
import math
import types
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from vendor.unidb_gou.DenoisingUNet_arch import ConditionalUNet
from vendor.unidb_gou.loss import MatchingLoss
from vendor.unidb_gou import sde_utils

ROOT = Path(__file__).resolve().parent
sde_utils.tqdm = lambda iterable, **kwargs: iterable  # Progress display only.


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def derive(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8], 'little') % (2**63-1)


def sources_ok():
    manifest = json.loads((ROOT/'protocol/UPSTREAM_SOURCE.json').read_text())
    for name, record in manifest['files'].items():
        assert sha(ROOT/'vendor/unidb_gou'/name) == record['sha256'], name
    frozen = ROOT/'protocol/RUN_SOURCE.json'
    if frozen.exists():
        for name, expected in json.loads(frozen.read_text())['files'].items():
            assert sha(ROOT/name) == expected, name


def initialize(device='cpu'):
    sources_ok()
    asset = json.loads((ROOT/'evidence/WEIGHT_DOWNLOAD.json').read_text())
    assert asset['file_id'] == '1kIrvZ1X46jFRAYIScverDD3dT1F40Kip'
    assert sha(asset['path']) == asset['sha256']
    state = torch.load(asset['path'], map_location='cpu', weights_only=True)
    wrapper = 'raw_state_dict'
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']; wrapper = 'state_dict'
    assert isinstance(state, dict) and all(isinstance(v,torch.Tensor) for v in state.values())
    if all(k.startswith('module.') for k in state):
        state = {k[7:]:v for k,v in state.items()}; wrapper += '+module_prefix_removed'
    net = ConditionalUNet(in_nc=3,out_nc=3,nf=64,depth=4)
    result = net.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert sum(p.numel() for p in net.parameters()) == 137147523
    assert all(torch.isfinite(v).all() for v in state.values())
    gelu_count=0
    if str(device).startswith('npu'):
        # Ascend 2.7.1's default GELU empirically follows the tanh approximation
        # even for approximate='none'. Preserve the official exact GELU formula
        # and derivative using FP32 erf. No module/parameter/schema is added.
        for module in net.modules():
            if isinstance(module,torch.nn.GELU):
                assert module.approximate=='none'
                module.forward=types.MethodType(exact_gelu_forward,module)
                gelu_count+=1
        assert gelu_count==1
    return net.to(device), {'wrapper':wrapper,'parameter_count':137147523,
        'tensor_count':len(state),'weight_sha256':asset['sha256'],'strict_load':True,
        'equivalent_exact_GELU_runtime_count':gelu_count}


def exact_gelu_forward(module,value):
    return .5*value*(1+torch.erf(value/math.sqrt(2)))


def configure_npu(local):
    import torch_npu
    torch.set_num_threads(2)
    torch.npu.set_device(local)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
    torch.npu.conv.allow_hf32=False
    torch.npu.matmul.allow_hf32=False
    if hasattr(torch.npu,'aclnn'): torch.npu.aclnn.allow_hf32=False
    torch.npu.config.allow_internal_format=True
    torch.npu.set_per_process_memory_fraction(.74)
    return torch.device(f'npu:{local}')


def process(device):
    return sde_utils.UniDB(lambda_square=30,gamma_inv=1e-7,T=100,
        schedule='cosine',eps=.005,device=device)


class NativeObjective(torch.nn.Module):
    def __init__(self, net, device):
        super().__init__()
        self.network=net
        self.sde=process(device)
        self.sde.set_model(net)
        self.criterion=MatchingLoss(loss_type='l1',is_weighted=False)

    def forward(self, degraded, clean):
        # Exact official training procedure. reshape(-1) preserves batch=1,
        # unlike upstream squeeze(), without changing any time or loss value.
        tau, state=self.sde.generate_random_states(x0=clean,mu=degraded)
        tau=tau.to(degraded.device)
        noise=self.sde.noise_fn(state,tau.reshape(-1))
        score=self.sde.get_score_from_noise(noise,tau)
        predicted=self.sde.reverse_sde_step_mean(state,score,tau)
        optimum=self.sde.reverse_optimum_step(state,clean,tau)
        return self.criterion(predicted,optimum),tau.detach()


def make_ema(net,device):
    return EMA(net,beta=.995,update_every=10).to(device)


def rng_capture(device):
    return {'torch':torch.get_rng_state(),'numpy':np.random.get_state(),
        'python':random.getstate(),'npu':torch.npu.get_rng_state(device).cpu()}


def rng_restore(state,device):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    torch.npu.set_rng_state(state['npu'],device)


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def ema_save(ema):
    return {'network':cpu_tree(ema.ema_model.state_dict()),'initted':ema.initted.cpu(),
        'step':ema.step.cpu(),'beta':.995,'update_every':10,'update_after_step':ema.update_after_step}


def ema_restore(ema,state):
    assert state['beta']==.995 and state['update_every']==10
    assert state['update_after_step']==ema.update_after_step
    ema.ema_model.load_state_dict(state['network'],strict=True)
    ema.initted.copy_(state['initted']);ema.step.copy_(state['step'])


def parameter_digest(net,full=False):
    h=hashlib.sha256()
    for name,value in net.state_dict().items():
        h.update(name.encode());flat=value.detach().reshape(-1)
        if not full and flat.numel()>32:flat=torch.cat((flat[:16],flat[-16:]))
        h.update(flat.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics_module():
    path=Path('/data2/yanglin/v5_concertormer_10k_20261021/training/image_metrics.py')
    spec=importlib.util.spec_from_file_location('unidb_fixed_project_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

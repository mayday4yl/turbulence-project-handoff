"""Independent complete-state CPU readback and restart comparison."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from core import ROOT,sha,write


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value:yield from tensors(v)


def audit(path):
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['format']=='UNIDB_NATIVE_FULL_STATE_V1'
    assert p['cursor']==p['update']*p['config']['effective_batch']
    assert sum(p['counts_by_g'].values())==p['cursor']
    assert len(p['rng_by_rank'])==p['config']['world_size']
    assert int(p['ema']['step'])==p['update']
    steps=[int(s['step']) for s in p['optimizer']['state'].values()]
    if p['update']:assert len(steps)==len(p['network']) and set(steps)=={p['update']}
    checked=0
    for tensor in tensors(p):
        if tensor.is_floating_point():assert bool(torch.isfinite(tensor).all())
        checked+=1
    return p,{'path':str(path),'sha256':sha(path),'update':p['update'],'cursor':p['cursor'],
        'tensors_finite_checked':checked,'Adam_states':len(steps),'RNG_slots':len(p['rng_by_rank'])}


def compare(a,b):
    assert a['update']==b['update'] and a['cursor']==b['cursor'] and a['counts_by_g']==b['counts_by_g']
    maxima={}
    def recurse(x,y,label,atol,rtol):
        if isinstance(x,torch.Tensor):
            assert x.shape==y.shape and x.dtype==y.dtype,(label,'shape/dtype')
            if x.is_floating_point():
                error=float((x-y).abs().max()) if x.numel() else 0.
                maxima[label.split('.')[0]]=max(maxima.get(label.split('.')[0],0.),error)
                assert torch.allclose(x,y,atol=atol,rtol=rtol),(label,error)
            else:assert torch.equal(x,y),label
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),label
            for k in x:recurse(x[k],y[k],label+'.'+str(k),atol,rtol)
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y)
            for i,(u,v) in enumerate(zip(x,y)):recurse(u,v,label+'.'+str(i),atol,rtol)
        elif hasattr(x,'shape'):
            import numpy as np
            assert np.array_equal(x,y),label
        else:assert x==y,(label,x,y)
    recurse(a['network'],b['network'],'network',5e-7,5e-5)
    recurse(a['optimizer'],b['optimizer'],'optimizer',1e-8,1e-4)
    recurse(a['ema'],b['ema'],'ema',5e-7,5e-5)
    recurse(a['rng_by_rank'],b['rng_by_rank'],'rng',0,0)
    return maxima


if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--compare');parser.add_argument('--output',required=True)
    args=parser.parse_args();payload,report=audit(args.checkpoint)
    if args.compare:
        other,second=audit(args.compare);report['compared']=second;report['max_abs_errors']=compare(payload,other)
        report['RNG_exact']=True
    report['status']='PASS';write(Path(args.output),report);print(json.dumps(report),flush=True)

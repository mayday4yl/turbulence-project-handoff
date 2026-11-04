"""Bounded real-size NPU checks, never a formal optimizer-update budget."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback
import numpy as np
import torch
from models import build_model
from runtime.environment import configure, make_simulator, synchronize, memory_stats
from training.core import unroll_training, network_context


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='npu:0')
    p.add_argument('--mode',choices=['network','chain'],default='network')
    p.add_argument('--initial',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['D0','R0'],default='R0')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--checkpointing',action='store_true')
    p.add_argument('--checkpoint-policy',choices=['all','highest','highest_half','highest_third','highres','alternating','highres_half_low'],default='all')
    p.add_argument('--talking-backend',choices=['conv3d','conv2d_equivalent'],default='conv3d')
    p.add_argument('--cross-step',action='store_true')
    p.add_argument('--psf-backend', choices=['native','adjoint'], default='native')
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report={'arguments':vars(a),'scope':'bounded preflight/benchmark; zero formal updates',
            'status':'RUNNING','measurements':[],'started_at_unix':time.time()}
    def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        report['environment']=configure(a.device)
        # These are our own hash-bound U0/training checkpoints; full resume
        # records also contain NumPy/Python RNG state, not only tensors.
        initial=torch.load(a.initial,map_location='cpu',weights_only=False)
        model=build_model(activation_checkpointing=a.checkpointing,checkpoint_policy=a.checkpoint_policy,talking_backend=a.talking_backend)
        model.load_state_dict(initial['model'],strict=True)
        report['initial_sha256']=hashlib.sha256(Path(a.initial).read_bytes()).hexdigest()
        model.to(a.device).train()
        rows=[json.loads(line) for line in (Path(a.dataset)/'sample_plan.jsonl').read_text().splitlines()]
        selected=[r for r in rows if r['split']=='train' and (Path(a.dataset)/r['record_ref']).is_file()][:a.microbatch]
        if len(selected)!=a.microbatch:
            raise RuntimeError('Insufficient completed train observations for preflight')
        c=torch.stack([torch.from_numpy(np.load(Path(a.dataset)/r['clean_ref'],allow_pickle=False)) for r in selected]).to(a.device)
        y=torch.stack([torch.from_numpy(np.load(Path(a.dataset)/r['observation_ref'],allow_pickle=False)) for r in selected]).to(a.device)
        report['sample_ids']=[r['sample_id'] for r in selected]
        sim=make_simulator(a.device) if a.mode=='chain' else None
        if sim is not None and a.psf_backend == 'adjoint':
            from optimizations.psf_adjoint_20261021.adjoint import install
            sim=install(sim)
        optimizer=torch.optim.AdamW(model.optimizer_parameter_groups(),betas=(.9,.999),eps=1e-8)
        if a.device.startswith('npu'):torch.npu.reset_peak_memory_stats(a.device)
        for repeat in range(a.repeats):
            optimizer.zero_grad(set_to_none=True)
            synchronize(a.device);start=time.perf_counter()
            if a.mode=='network':
                t=torch.zeros(a.microbatch,device=a.device)
                with network_context(a.device,a.precision):
                    clean,xi=model(y,t)
                    loss=(clean.float()-c).abs().mean()+.01*(xi.epsilon_zernike.float().square().mean()+xi.epsilon_tilt.float().square().mean())
                if repeat==0:
                    with torch.no_grad(),network_context(a.device,a.precision):
                        official=model.backbone(y)[-1]
                    difference=(clean-official).abs()
                    report['official_same_backend_parity']={'max_abs':difference.max().item(),'mean_abs':difference.mean().item()}
                    if difference.max().item()>1e-6:
                        raise RuntimeError('Initial zero-time-adapter output does not match official final RGB')
                    assert tuple(xi.epsilon_zernike.shape)==(a.microbatch,36,512,512)
                    assert tuple(xi.epsilon_tilt.shape)==(a.microbatch,512,512,2)
            else:
                result=unroll_training(model,sim,y,c,method=a.method,path_seed=20261021+repeat,
                                       network_precision=a.precision,retain_trace=a.cross_step and repeat==0)
                loss=result.loss
                if a.cross_step and repeat==0 and a.method!='D0':
                    first=result.trace[0]
                    terminal=(result.final_clean-c).square().mean()
                    gradients=torch.autograd.grad(terminal,[first['clean'],first['raw'].epsilon_zernike,
                                                            first['raw'].epsilon_tilt],retain_graph=True,allow_unused=False)
                    record={}
                    for name,g in zip(['first_clean','first_raw_zernike','first_raw_tilt'],gradients):
                        value={'finite':bool(torch.isfinite(g).all().item()),'norm':g.float().norm().item(),
                               'max_abs':g.abs().max().item(),'nonzero':int(torch.count_nonzero(g).item())}
                        if not value['finite'] or value['nonzero']==0:
                            raise RuntimeError(f'Invalid terminal cross-step gradient: {name} {value}')
                        record[name]=value
                    report['terminal_cross_step_gradients']=record
                    del gradients
            if not torch.isfinite(loss).item():raise FloatingPointError('nonfinite loss')
            loss.backward()
            missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise RuntimeError(f'Missing trainable parameter gradients: {missing}')
            # A finite global norm establishes finiteness without one device
            # synchronization per parameter; error_if_nonfinite enforces it.
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            optimizer.step()
            synchronize(a.device)
            report['measurements'].append({'repeat':repeat,'loss':loss.item(),'grad_norm_before_clip':norm.item(),
                                           'seconds':time.perf_counter()-start,**memory_stats(a.device)})
            if a.mode=='chain':del result
            del loss
            save()
        report['status']='PASS'
    except Exception as exc:
        report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    report['finished_at_unix']=time.time();save()
    print(json.dumps(report,allow_nan=False),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

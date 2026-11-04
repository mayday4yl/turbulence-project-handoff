"""Independent FP32 Simulator VJPs, including zero/interior/max strengths."""
import argparse
from pathlib import Path
import json
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import torch
from adjoint import install
from probe import error
from runtime.environment import configure,make_simulator
from training.state import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='npu:0')
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('Preserve previous evidence')
    report={'status':'RUNNING','formal_optimizer_updates':0,'cases':[],'started_unix':time.time()}
    def save():atomic_json(report,out)
    save()
    try:
        report['environment']=configure(args.device)
        sim=make_simulator(args.device).eval()
        for p in sim.parameters():p.requires_grad_(False)
        generator=torch.Generator().manual_seed(193203)
        raw=sim.sample_realization(1,seed_zernike=719,seed_tilt=127)
        raw.epsilon_zernike.requires_grad_();raw.epsilon_tilt.requires_grad_()
        image=torch.rand(1,3,256,256,generator=generator).to(args.device).requires_grad_()
        v=torch.randn(image.shape,generator=generator).to(args.device)/image.numel()
        reference={}
        for backend in ['native','adjoint']:
            if backend=='adjoint':install(sim)
            for g in [0.,1e-4,2.5,4.]:
                y=sim.render_prepared(sim.prepare_image(image),raw,g).output
                grad=torch.autograd.grad(y,[image,raw.epsilon_zernike,raw.epsilon_tilt],v)
                current={'output':y.detach().cpu(),'gradients':[z.detach().cpu() for z in grad]}
                if backend=='native':reference[g]=current
                else:
                    comparisons={name:error(a,b) for name,a,b in zip(['clean','zernike','tilt'],current['gradients'],reference[g]['gradients'])}
                    row={'g':g,'forward_exact':torch.equal(current['output'],reference[g]['output']),'gradients':comparisons,
                        'allclose':all(torch.allclose(a,b,atol=2e-6,rtol=2e-5) for a,b in zip(current['gradients'],reference[g]['gradients']))}
                    if g==0:
                        row['raw_gradients_zero']=all(torch.count_nonzero(z).item()==0 for z in current['gradients'][1:])
                        row['clean_identity_vjp']=error(current['gradients'][0],v)
                    row['pass']=row['forward_exact'] and row['allclose'] and all(e['finite'] and e['relative_l2']<=2e-5 for e in comparisons.values())
                    if g==0:row['pass']=row['pass'] and row['raw_gradients_zero'] and row['clean_identity_vjp']['relative_l2']<=2e-5
                    report['cases'].append(row);save()
                del y,grad,current
        report['status']='PASS' if all(row['pass'] for row in report['cases']) else 'FAIL'
    except BaseException as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    report['finished_unix']=time.time();save();print(json.dumps(report),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

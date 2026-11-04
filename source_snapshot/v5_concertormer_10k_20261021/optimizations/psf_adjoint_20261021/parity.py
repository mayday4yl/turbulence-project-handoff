"""Full V5 forward/backward parity at common U0; no optimizer exists here."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import torch
from adjoint import install
from probe import error
from models import build_model
from runtime.environment import configure,make_simulator,synchronize,memory_stats
from scripts.benchmark_distributed import read_completed_train_pairs
from training.runner import load_initial_model
from training.state import atomic_json
import training.core as core


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='npu:0')
    p.add_argument('--initial',default='weights/adapted_U0.pt')
    p.add_argument('--dataset',default='data/v5_places_c50_s2000_v250_t250_g4_r4_20261021')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise ValueError('Preserve previous evidence')
    report={'status':'RUNNING','formal_optimizer_updates':0,'started_unix':time.time(),'arguments':vars(args)}
    def save():atomic_json(report,out)
    save()
    try:
        report['environment']=configure(args.device)
        pairs,report['data_identity']=read_completed_train_pairs(args.dataset)
        clean=torch.stack([r[1] for r in pairs[:2]]).to(args.device)
        obs=torch.stack([r[2] for r in pairs[:2]]).to(args.device)
        del pairs
        sim=make_simulator(args.device).eval()
        for v in sim.parameters():v.requires_grad_(False)
        model=build_model(activation_checkpointing=True,checkpoint_policy='highest_half',talking_backend='conv2d_equivalent')
        load_initial_model(model,args.initial)
        model.to(args.device).train()
        values={}
        for backend in ['native','adjoint']:
            if backend=='adjoint':install(sim)
            model.zero_grad(set_to_none=True)
            torch.manual_seed(20261023);torch.npu.manual_seed(20261023)
            synchronize(args.device);start=time.perf_counter()
            result=core.unroll_training(model,sim,obs,clean,method='R0',path_seed=20261023,network_precision='bf16',retain_trace=True)
            first=result.trace[0]
            cg=torch.autograd.grad((result.final_clean-clean).square().mean(),
                [first['clean'],first['raw'].epsilon_zernike,first['raw'].epsilon_tilt],retain_graph=True)
            cross={name:g.detach().cpu() for name,g in zip(['clean','zernike','tilt'],cg)}
            del cg
            forward=[{k:r[k].detach().cpu() for k in ['clean','matched','loss']} for r in result.trace]
            # The original graph is intact throughout the measured gradient check.
            result.loss.backward()
            grads={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad and v.grad is not None}
            missing=[n for n,v in model.named_parameters() if v.requires_grad and v.grad is None]
            if missing:raise RuntimeError('Missing parameter gradients: '+repr(missing))
            values[backend]={'cross':cross,'forward':forward,'grads':grads,'loss':result.loss.item()}
            report[backend]={'loss':result.loss.item(),'seconds_including_CPU_capture':time.perf_counter()-start,
                'gradient_count':len(grads),'all_parameter_gradients_finite':all(torch.isfinite(v).all().item() for v in grads.values()),
                'cross_step':{n:{'finite':bool(torch.isfinite(g).all()),'norm':g.norm().item(),'nonzero':torch.count_nonzero(g).item()} for n,g in cross.items()},
                **memory_stats(args.device)}
            if backend=='adjoint':
                # Explicit negative control, after the intact gradient check only.
                cut=result.trace[1]['state'].detach()
                with core.network_context(args.device,'bf16'):
                    cut_clean,_=model(cut,result.paths.t[:,1])
                disconnected=torch.autograd.grad(cut_clean.float().square().mean(),
                    [first['clean'],first['raw'].epsilon_zernike,first['raw'].epsilon_tilt],allow_unused=True)
                report['negative_control_detached_state_disconnects_first_prediction']=all(g is None for g in disconnected)
                del cut,cut_clean,disconnected
            del result,first,grads,cross,forward
            model.zero_grad(set_to_none=True);gc.collect();torch.npu.empty_cache()
            save()
        report['forward_exact']=all(torch.equal(a[k],b[k]) for a,b in zip(values['native']['forward'],values['adjoint']['forward']) for k in a)
        report['loss_exact']=values['native']['loss']==values['adjoint']['loss']
        report['cross_step_parity']={n:error(values['adjoint']['cross'][n],v) for n,v in values['native']['cross'].items()}
        per_parameter=[];diff_sq=0.;reference_sq=0.
        for n,ref in values['native']['grads'].items():
            actual=values['adjoint']['grads'][n]
            e=error(actual,ref);e['name']=n;per_parameter.append(e)
            diff_sq+=((actual.double()-ref.double())**2).sum().item()
            reference_sq+=(ref.double()**2).sum().item()
        report['parameter_gradient_relative_l2']=(diff_sq/max(reference_sq,1e-60))**.5
        report['maximum_parameter_gradient_absolute_error']=max(r['max_absolute'] for r in per_parameter)
        report['worst_parameter_relative_l2']=max(per_parameter,key=lambda r:r['relative_l2'])
        atomic_json(per_parameter,out.with_name(out.stem+'_parameters.json'))
        report['status']='PASS' if (report['forward_exact'] and report['loss_exact']
            and report['parameter_gradient_relative_l2']<=2e-3
            and all(r['finite'] for r in per_parameter)
            and all(r['finite'] and r['relative_l2']<=2e-3 for r in report['cross_step_parity'].values())
            and all(r['finite'] and r['nonzero']>0 for r in report['adjoint']['cross_step'].values())
            and report['negative_control_detached_state_disconnects_first_prediction']) else 'FAIL'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    report['finished_unix']=time.time();save();print(json.dumps(report),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

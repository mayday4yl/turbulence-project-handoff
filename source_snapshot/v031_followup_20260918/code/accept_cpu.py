"""Targeted CPU mechanism tests; fixture updates never become formal weights."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from copy import deepcopy
import json, tempfile
import torch
from torch import nn
from physical_bridge.model import DualHeadUNet
from v031.followup import CleanOnly, RolloutModule
from v031.core import severity_path, full_image_mse
from v031.training import (TrainingEngine, GlobalStartSampler, empty_counts, advance_counts,
    save_training_checkpoint, load_training_checkpoint, compare_states)
from v031.runtime import save_json
from train_worker import resume_snapshot

ROOT=Path(__file__).resolve().parents[1]

class FixtureSimulator(nn.Module):
    def render(self,c,xi,g):
        scale=torch.as_tensor(g,device=c.device).reshape(-1,1,1,1)
        z=xi.epsilon_zernike.mean((1,2,3),keepdim=True)
        t=xi.epsilon_tilt.mean((1,2,3),keepdim=True)
        return SimpleNamespace(output=c+scale*(0.03*z+0.02*t))

class FixtureDataset:
    def __init__(self):
        self.records=[{'start_id':str(i),'initial_G':4-i%4,'trajectory_id':str(i)} for i in range(32)]
        gen=torch.Generator().manual_seed(44)
        self.c=torch.rand(32,3,16,16,generator=gen)
        self.y=torch.rand(32,3,16,16,generator=gen)
    def __getitem__(self,i):
        r=self.records[i]
        return dict(clean=self.c[i],observation=self.y[i],G=torch.tensor(float(r['initial_G'])),
                    severity_path=severity_path(r['initial_G']),start_id=r['start_id'],trajectory_id=r['trajectory_id'])

def fixture_network(task):
    torch.manual_seed(20260916)
    n=DualHeadUNet(base_channels=4,channel_multipliers=(1,2),residual_blocks_per_level=1,
                  group_norm_groups=2,severity_embedding_dim=8)
    if task=='T1_DIRECT':n=CleanOnly(n)
    n.followup_task=task
    return n

def engine(task,data):
    return TrainingEngine(fixture_network(task),FixtureSimulator(),data,device='cpu',
        microbatch=8,identity={'fixture':True,'task':task},seed=20260916)

def test_feedback():
    n=fixture_network('T2_FEEDBACK_R'); sim=FixtureSimulator()
    data=FixtureDataset(); c,y=data.c[:4],data.y[:4];g=torch.tensor([4.,3.,2.,1.])
    out=RolloutModule(n,sim)(c,y,g,retain_intermediates=True)
    manual=[]
    for i in range(4):
        z=y[i:i+1];loss=[]
        for s in severity_path(g[i]):
            p=n(z,torch.tensor([s],dtype=torch.float32))
            loss.append(full_image_mse(p.clean,c[i:i+1])+0.2*full_image_mse(sim.render(p.clean,p.realization,g[i:i+1]).output,y[i:i+1]))
            z=p.clean
        manual.append(torch.stack(loss).mean())
    assert torch.allclose(out['path_losses'],torch.stack(manual),atol=1e-7,rtol=1e-5)
    trace=out['trace'];terminal=trace[-1]['clean_losses'].sum()
    early=[x['clean'] for x in trace[:-1]]
    gradients=torch.autograd.grad(terminal,early,retain_graph=True)
    assert all(x is not None and x.norm()>0 for x in gradients)
    early_xi=[x[k] for x in trace[:-1] for k in ('epsilon_zernike','epsilon_tilt')]
    assert all(x is None for x in torch.autograd.grad(terminal,early_xi,allow_unused=True,retain_graph=True))
    for step in trace:
        grads=torch.autograd.grad(step['reconstruction_losses'].sum(),[step[k] for k in
            ('clean','epsilon_zernike','epsilon_tilt')],retain_graph=True,allow_unused=True)
        assert all(x is not None and torch.isfinite(x).all() for x in grads)
    # Independent negative-control graphs retain values but sever each early clean feedback.
    negative=[]
    for cut in range(3):
        z=y[:1];pred=[]
        for k,s in enumerate((4,3,2,1)):
            p=n(z,torch.tensor([float(s)]));pred.append(p.clean)
            z=p.clean.detach() if k==cut else p.clean
        grad=torch.autograd.grad(full_image_mse(pred[-1],c[:1]).sum(),pred[cut],allow_unused=True)[0]
        assert grad is None
        assert torch.allclose(pred[-1],trace[-1]['clean'],atol=1e-6,rtol=1e-5)
        negative.append(True)
    assert out['prediction_count']==10 and out['logical_render_count']==10
    return {'loss_denominator':True,'early_clean_connected':[float(x.norm()) for x in gradients],
            'terminal_clean_to_early_xi':'absent_as_required','clean_detach_negative_controls':negative,
            'own_LR_to_both_xi_blocks':'connected','fixture_only':True}

def main():
    torch.set_num_threads(4)
    result={'scope':'CPU fixtures; native backend acceptance still required','tasks':{}}
    stream=GlobalStartSampler([str(i) for i in range(1600)])
    seen=Counter()
    for _ in range(1000):seen.update(stream.next_indices())
    assert len(seen)==1600 and set(seen.values())=={20}
    result['sampler_1000_updates_exact_20_exposures']=True
    result['T2_graph']=test_feedback()
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        counts=empty_counts(task);advance_counts(counts,[4,3,2,1]*8000)
        assert counts['paths']==32000 and counts['predictions']==(32000 if task=='T1_DIRECT' else 80000)
        assert counts['LR_simulator_renders']==(0 if task=='T1_DIRECT' else 80000)
        d=FixtureDataset();e=engine(task,d)
        e.update(); boundary=resume_snapshot(e)
        with tempfile.TemporaryDirectory(dir=ROOT/'acceptance') as td:
            p=Path(td)/'checkpoint.pt'
            save_training_checkpoint(e,p,run_until_update=2)
            e.update(); suffix=resume_snapshot(e)
            restored=engine(task,d);load_training_checkpoint(restored,p,run_until_update=2)
            assert compare_states(boundary,resume_snapshot(restored))['exact']
            restored.update()
            exact=compare_states(suffix,resume_snapshot(restored))
            assert exact['exact'],exact
            raw=torch.load(p,map_location='cpu',weights_only=True)
            assert raw['ema_enabled'] is False and 'ema' not in raw
        result['tasks'][task]={'counts':counts,'exact_boundary_restore':True,'exact_cpu_suffix':True,
                              'fixture_optimizer_updates':3,'formal_updates':0,'no_ema':True}
    save_json(ROOT/'acceptance/cpu_mechanisms.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()

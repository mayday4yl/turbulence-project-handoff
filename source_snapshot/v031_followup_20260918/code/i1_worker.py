"""Single-card raw1000 I1 worker; each output binds configuration and source IDs."""
from pathlib import Path
from dataclasses import asdict
import argparse, json, math, os, time
import numpy as np
import torch
from physical_bridge.model import DualHeadUNet
from physical_bridge.simulator import PiRNP2SSimulator
from v031.followup import CleanOnly
from v031.followup_refinement import Objectives,candidates,solve,clone_xi,blocks
from v031.data import StartDataset
from v031.runtime import configure_npu,verify_source,state_hash,sha256,save_json
from v031.training import synchronize,memory_stats
from v031.evaluation import rollout_predictions

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data2/yanglin/v031_overfit100_g4321_n4')

def init():
    cfg=json.loads((ROOT/'shared/original_train_config.json').read_text())
    import torch_npu
    configure_npu(cfg)
    cap=float(os.environ['FOLLOWUP_OWN_CAP_GIB'])
    torch.npu.set_per_process_memory_fraction(cap/64,torch.device('npu:0'))
    verify_source(cfg)
    p=OLD/'runs/formal/checkpoint_001000.pt'
    if sha256(p)!='6eb3de6a064a0215a9a5b02935145528b23209d2f633b6dd50b60a999bad2454':
        raise RuntimeError('Old raw1000 identity changed')
    ck=torch.load(p,map_location='cpu',weights_only=True)
    if ck['global_update']!=1000:raise RuntimeError('Wrong old update')
    model=DualHeadUNet(**cfg['network']);model.load_state_dict(ck['network'],strict=True)
    del ck
    model=model.to('npu:0').eval().requires_grad_(False)
    clean=CleanOnly(model).eval().requires_grad_(False)
    sim=PiRNP2SSimulator(cfg['asset_dir'],**cfg['physics'],device='npu:0').eval().requires_grad_(False)
    data=StartDataset(cfg['dataset_root'])
    return cfg,model,clean,sim,data

def quality(c,gt):
    mse=float((c-gt).square().mean())
    if not math.isfinite(mse):raise FloatingPointError('Nonfinite full RGB MSE')
    return {'MSE':mse,'PSNR':-10*math.log10(mse) if mse>0 else None,'zero_MSE':mse==0}

def run_path(model,clean,sim,item,candidate,save_arrays=None):
    y=item['observation'].unsqueeze(0).to('npu:0');gt=item['clean'].unsqueeze(0).to('npu:0')
    G=int(item['G']);state=y;previous=None;rows=[];traces=[];preds=[]
    calls={'network':0,'F':0,'VJP':0};started=time.perf_counter()
    for k,s in enumerate(range(G,0,-1),1):
        with torch.no_grad():
            p=model(state,state.new_tensor([float(s)]));calls['network']+=1
            c=p.clean;xi=clone_xi(p.realization)
            r=sim.render(c,xi,G).output;calls['F']+=1
        rows.append({'k':k,'G':G,'current_g':s,**quality(c,gt),'LR':float((r-y).square().mean())})
        preds.append(c.detach().cpu())
        if s>1:
            result=None
            if candidate.family!='U0':
                obj=Objectives(sim,clean,c,y,G,s-1,xi)
                result=solve(candidate=candidate,objectives=obj,anchor=xi,previous=previous)
                xi=result.pop('xi');previous=xi
                for key,value in result['calls'].items():calls[key]+=value
                traces.append({'k':k,**result})
            with torch.no_grad():state=sim.render(c,xi,s-1).output;calls['F']+=1
            if save_arrays is not None:
                save_arrays.mkdir(parents=True,exist_ok=True)
                np.savez(save_arrays/f'k{k}.npz',c=c.cpu().numpy(),next_state=state.cpu().numpy(),
                    **({} if result is None else {'Ua':obj.ua.cpu().numpy()}))
    synchronize(y.device)
    return {'start_id':item['start_id'],'trajectory_id':item['trajectory_id'],'candidate':asdict(candidate),
        'weights':'raw','checkpoint_update':1000,'metrics':rows,'input':quality(y,gt),
        'transitions':traces,'calls':calls,'seconds':time.perf_counter()-started},preds

def grad_stats(gs):
    return [float(g.double().norm()) for g in gs]

def preflight(model,clean,sim,data,index):
    item=data[index];y=item['observation'].unsqueeze(0).to('npu:0')
    gt=item['clean'].unsqueeze(0).to('npu:0');G=int(item['G'])
    if G!=4:raise ValueError('Preflight uses frozen dev first G4')
    result={}
    with torch.no_grad():p=model(y,y.new_tensor([4.]));u=clean(y,y.new_tensor([4.]))
    result['clean_forward_max_abs']=float((p.clean-u).abs().max())
    assert torch.allclose(p.clean,u,atol=1e-6,rtol=1e-5)
    inputs=y.detach().clone().requires_grad_(True)
    a=model(inputs,inputs.new_tensor([4.])).clean
    ga=torch.autograd.grad(a.square().mean(),inputs)[0]
    inputs2=y.detach().clone().requires_grad_(True)
    b=clean(inputs2,inputs2.new_tensor([4.]))
    gb=torch.autograd.grad(b.square().mean(),inputs2)[0]
    result['clean_input_gradient_relative_L2']=float((ga-gb).double().norm()/ga.double().norm())
    assert result['clean_input_gradient_relative_L2']<=1e-4
    xi=clone_xi(p.realization,True);obj=Objectives(sim,clean,p.clean,y,4,3,xi)
    hf=obj.objective(xi,'L');full=torch.autograd.grad(hf,blocks(xi))
    ui=obj.U(xi)
    hd=(obj.F(ui.detach(),xi,4)-y).square().mean()
    direct=torch.autograd.grad(hd,blocks(xi),retain_graph=True)
    hi=(obj.F(ui,clone_xi(xi),4)-y).square().mean()
    indirect=torch.autograd.grad(hi,blocks(xi))
    error=math.sqrt(sum(float((f-d-i).double().square().sum()) for f,d,i in zip(full,direct,indirect)))
    norm=math.sqrt(sum(float(f.double().square().sum()) for f in full))
    result['H_gradient']={'full':grad_stats(full),'direct':grad_stats(direct),
                         'indirect':grad_stats(indirect),'relative_sum_error':error/max(norm,1e-30)}
    assert all(torch.isfinite(g).all() for g in full+direct+indirect)
    assert sum(grad_stats(direct))>0 and sum(grad_stats(indirect))>0
    assert result['H_gradient']['relative_sum_error']<=1e-4
    result['solvers']={}
    for name in ('P_r0.005','L_r0.005'):
        candidate=next(x for x in candidates() if x.name==name)
        obj=Objectives(sim,clean,p.clean,y,4,3,p.realization)
        solved=solve(candidate=candidate,objectives=obj,anchor=p.realization)
        solved.pop('xi');result['solvers'][name]=solved
    result['reference_paths']={}
    sid=data.records[index]['source_id'];tid=data.records[index]['trajectory_id']
    for g in (4,3,2,1):
        i=next(i for i,r in enumerate(data.records) if r['trajectory_id']==tid and r['initial_G']==g)
        sample=data[i]
        record,predictions=run_path(model,clean,sim,sample,candidates()[0])
        old=rollout_predictions(model=model,simulator=sim,observation=sample['observation'].unsqueeze(0).to('npu:0'),initial_G=g,mode='unrefined')
        diffs=[float((x-z.clean.cpu()).abs().max()) for x,z in zip(predictions,old)]
        assert len(predictions)==g and all(d<=1e-5 for d in diffs)
        result['reference_paths'][f'U0_G{g}']={'max_abs':diffs,'no_refinement':not record['transitions']}
    record,predictions=run_path(model,clean,sim,item,candidates()[1])
    old=rollout_predictions(model=model,simulator=sim,observation=y,initial_G=4,mode='refined',
                           solver_config={'learning_rate':0.01,'iterations':40})
    diffs=[float((x-z.clean.cpu()).abs().max()) for x,z in zip(predictions,old)]
    assert all(torch.allclose(x,z.clean.cpu(),atol=1e-5,rtol=1e-4) for x,z in zip(predictions,old))
    result['reference_paths']['legacy_G4']={'max_abs':diffs}
    result['memory']=memory_stats(torch.device('npu:0'))
    result['status']='passed'
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','candidate','diagnostic'],required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--start-id');ap.add_argument('--candidate')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg,model,clean,sim,data=init();before={'network':state_hash(model),'simulator':state_hash(sim)}
    splits=json.loads((ROOT/'shared/manifests/FROZEN_SPLITS.json').read_text())
    sid=a.start_id or splits['splits']['refine_dev8']['start_ids'][0]
    i=next(i for i,r in enumerate(data.records) if r['start_id']==sid)
    try:
        if a.mode=='preflight':result=preflight(model,clean,sim,data,i)
        elif a.mode=='diagnostic':
            from v031.followup_diagnostics import diagnose
            source=data.records[i]['source_id']
            result=diagnose(model=model,clean=clean,sim=sim,data=data,index=i,
                include_direction=source in splits['splits']['refine_dev8']['source_ids'][:4],
                arrays=out/'arrays' if source in splits['diagnostic_arrays_dev_sources'] else None,
                save_full_xi=source in splits['full_xi_saved_sources'])
        else:
            candidate=next(x for x in candidates() if x.name==a.candidate)
            record=data.records[i]
            subset=splits['splits']
            allowed=subset['refine_dev8']['start_ids']
            if sid not in allowed:raise ValueError('Confirmation/heldout require a frozen selection, not this dev worker')
            arrays=out/'arrays' if record['source_id'] in splits['diagnostic_arrays_dev_sources'] else None
            result,predictions=run_path(model,clean,sim,data[i],candidate,arrays)
            result.update(source_id=record['source_id'],status='complete')
            if arrays is not None:
                for k,c in enumerate(predictions,1):np.save(arrays/f'clean_k{k}.npy',c.numpy())
        result.update(initial_state=before,final_state={'network':state_hash(model),'simulator':state_hash(sim)},
            source_hashes={p.name:sha256(p) for p in [Path(__file__),ROOT/'code/v031/followup_refinement.py']},
            card=os.environ['ASCEND_RT_VISIBLE_DEVICES'],host=os.uname().nodename,memory=memory_stats(torch.device('npu:0')))
        assert result['initial_state']==result['final_state'],'Frozen model/simulator changed'
        save_json(out/'result.json',result)
        print(json.dumps({'status':result['status'],'output':str(out),'memory':result['memory']}),flush=True)
    except BaseException as e:
        import traceback
        save_json(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()

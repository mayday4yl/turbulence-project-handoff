"""Actual chosen microbatch/accumulation, six-rank I/O + restart equivalence."""
import argparse,contextlib,json,os
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from grad_helpers import capture_rng,restore_rng,seed_rng,state_digest
from method_impl.d0 import D0Restorer,ImmediateEMA,derive_seed
from method_impl.training_io import load_training_pool,PositionDataset,new_snapshot,restore_snapshot,atomic_torch_save,learning_rate
from scripts.runtime_utils import ROOT,write_json,process_identity,sha

def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def compare(payload,reference,initial):
    result={}
    for group in ['network','ema']:
        actual=payload[group] if group=='network' else payload[group]['model']
        target=reference[group] if group=='network' else reference[group]['model']
        old=initial[group] if group=='network' else initial[group]['model']
        square_error=0.;square_update=0.;maximum=0.
        for k,x in actual.items():
            assert torch.isfinite(x).all() and torch.isfinite(target[k]).all()
            difference=x.double()-target[k].double();delta=target[k].double()-old[k].double()
            maximum=max(maximum,float(difference.abs().max()));square_error+=float(difference.square().sum());square_update+=float(delta.square().sum())
        relative=(square_error/max(square_update,1e-30))**.5
        assert maximum<=3e-6 and relative<=1e-3,(group,maximum,relative)
        result[group]={'max_abs':maximum,'relative_l2_to_reference_update':relative,'absolute_bound':3e-6,'relative_bound':1e-3}
    actual=payload['optimizer'];target=reference['optimizer'];assert actual['param_groups']==target['param_groups']
    for field,atol in [('step',0.),('exp_avg',1e-8),('exp_avg_sq',1e-10)]:
        maximum=0.
        for key,state in actual['state'].items():
            a=state[field];b=target['state'][key][field];assert torch.allclose(a,b,atol=atol,rtol=0. if field=='step' else 1e-5),(field,key)
            maximum=max(maximum,float((a-b).abs().max()))
        result[field]={'max_abs':maximum,'atol':atol,'rtol':0. if field=='step' else 1e-5}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--safety-file',required=True)
    ap.add_argument('--part',choices=['reference','resume'],required=True);a=ap.parse_args()
    import torch_npu
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(2);torch.npu.set_device(local);device=torch.device(f'npu:{local}')
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
    safety=json.load(open(a.safety_file));physical=safety['physical_devices'][local]
    free,total=torch.npu.mem_get_info();torch.npu.set_per_process_memory_fraction(safety['cards'][str(physical)]['allocator_cap']/total)
    dist.init_process_group('hccl',timeout=timedelta(minutes=10))
    run=ROOT/'distributed'/a.run;write_json(run/f'RANK_{rank}.json',{**process_identity(os.getpid()),'physical_device':physical})
    gate=json.loads((ROOT/'gates/G3_D0.json').read_text());assert gate['status']=='PASS' and world==gate['world_size']
    mb=gate['microbatch'];acc=gate['accumulation'];effective=mb*world*acc
    config={'updates':2,'microbatch':mb,'accumulation':acc,'world_size':world,'effective_batch':effective,'seed':735004,
        'learning_rate_peak':2e-5,'lr_policy':'same_warmup_cosine_function_two_update_engineering_probe'}
    pool,identity=load_training_pool(True);identity['gate_sha']=sha(ROOT/'gates/G3_D0.json')
    seed_rng(derive_seed(config['seed'],'process',rank),device)
    model=D0Restorer(ROOT.parent/'v5_diffuir_l_stage1_20261026/weights/model-300.pt').to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=2e-5,betas=(.9,.99),weight_decay=0);ema=ImmediateEMA(model.network)
    out=ROOT/'audits/ddp_resume01';out.mkdir(parents=True,exist_ok=True);start=0;payload=None
    if a.part=='resume':
        payload=torch.load(out/'U1.pt',map_location='cpu',weights_only=False)
        assert payload['config']==config and payload['identity']==identity
        restore_snapshot(payload,model,opt,ema)
        assert state_digest(model.network.state_dict())==state_digest(payload['network'])
        assert equal_tree(opt.state_dict(),payload['optimizer']) and equal_tree(ema.state_dict(),payload['ema'])
        start=payload['update'];assert start==1 and payload['cursor']==effective
    else:assert not (out/'U1.pt').exists()
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    ds=PositionDataset(pool,config['seed'],start*effective,(2-start)*acc*mb,rank,world,mb)
    generator=torch.Generator(device='cpu').manual_seed(derive_seed(config['seed'],'loader_workers',rank))
    loader=DataLoader(ds,batch_size=mb,shuffle=False,drop_last=False,num_workers=2,multiprocessing_context='spawn',
        persistent_workers=True,prefetch_factor=2,generator=generator)
    iterator=iter(loader)
    if payload is not None:
        restore_rng(payload['rng_by_rank'][rank],device);assert equal_tree(capture_rng(device),payload['rng_by_rank'][rank])
    def save(step,name):
        atomic_torch_save(out/f'{a.part}_rng_rank{rank}.pt',capture_rng(device));dist.barrier()
        if rank==0:
            rng=[torch.load(out/f'{a.part}_rng_rank{i}.pt',map_location='cpu',weights_only=False) for i in range(world)]
            value=new_snapshot(model,opt,ema,update=step,cursor=step*effective,rng=rng,config=config,identity=identity)
            digest=atomic_torch_save(out/name,value)
            readback=torch.load(out/name,map_location='cpu',weights_only=False)
            assert equal_tree(value,readback);write_json(out/(name+'.sha.json'),{'sha256':digest,'bytes':(out/name).stat().st_size})
        dist.barrier()
    for update in range(start+1,3):
        lr=learning_rate(update,config['updates'])
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);total_loss=0.;indices=[];positions=[]
        for micro in range(acc):
            x,c,ii,pp=next(iterator);indices+=ii.tolist();positions+=pp.tolist()
            with ddp.no_sync() if micro+1<acc else contextlib.nullcontext():
                value=ddp(x.to(device),c.to(device),seed=derive_seed(config['seed'],update,micro,rank))
                (value['loss']/acc).backward();total_loss+=float(value['loss'])/acc
            del value,x,c
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert bool(norm>0)
        opt.step();ema.update(model.network)
        assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
        record={'update':update,'indices':indices,'positions':positions,'loss':total_loss,'grad_norm':float(norm),'lr':lr,
            'network_sha':state_digest(model.network.state_dict()),'ema_updates':ema.updates}
        write_json(out/f'{a.part}_u{update}_rank{rank}.json',record)
        if update==1:save(1,'U1.pt')
        if update==2:save(2,'reference_U2.pt' if a.part=='reference' else 'resumed_U2.pt')
    dist.barrier()
    if rank==0 and a.part=='resume':
        current=torch.load(out/'resumed_U2.pt',map_location='cpu',weights_only=False)
        reference=torch.load(out/'reference_U2.pt',map_location='cpu',weights_only=False)
        first=torch.load(out/'U1.pt',map_location='cpu',weights_only=False)
        differences=compare(current,reference,first)
        ranks=[]
        for i in range(world):
            before=json.loads((out/f'reference_u2_rank{i}.json').read_text());after=json.loads((out/f'resume_u2_rank{i}.json').read_text())
            assert before['indices']==after['indices'] and before['positions']==after['positions']
            assert before['lr']==after['lr']==learning_rate(2,config['updates'])
            assert abs(before['loss']-after['loss'])<=1e-6 and abs(before['grad_norm']-after['grad_norm'])<=1e-5
            ranks.append({'rank':i,'data_indices_equal':True,'loss_abs_difference':abs(before['loss']-after['loss'])})
        assert len({json.loads((out/f'resume_u2_rank{i}.json').read_text())['network_sha'] for i in range(world)})==1
        assert equal_tree(current['rng_by_rank'],reference['rng_by_rank'])
        write_json(ROOT/'evidence/D0_DDP_RESUME_AUDIT.json',{'status':'PASS','scope':'chosen six-card microbatch/accumulation + actual full-pool DataLoader, new processes resume one matching update',
            'microbatch':mb,'accumulation':acc,'world_size':world,'effective_batch':effective,'full_v6_pool':True,
            'numerical_comparison':differences,'rank_comparisons':ranks,'all_rank_rng_after_update_equal_exact':True,
            'restored_before_update_model_ema_optimizer_rng_equal_exact':True,'rank_parameters_equal_exact':True,
            'lr_scheduler_next_update_equal':True,'scheduler_scope':'same production warmup/cosine function on a two-update diagnostic horizon; not formal updates',
            'cpu_full_payload_readback_equal':True,'formal_updates':0,'diagnostic_global_optimizer_updates_executed':3,
            'G3_gate_sha':identity['gate_sha'],'source_manifest_identity':identity,'test_used':False,
            'source_sha256':{p:sha(ROOT/p) for p in ['scripts/audit_ddp_resume.py','scripts/run_d0.py','method_impl/d0.py','method_impl/training_io.py']},
            'checkpoint_sha':{name:sha(out/name) for name in ['U1.pt','reference_U2.pt','resumed_U2.pt']}})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()

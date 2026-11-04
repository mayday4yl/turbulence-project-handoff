"""V6 A/B genuine full-loss DDP runner, frozen budget and validation contract."""
import os
os.environ["TASK_QUEUE_ENABLE"]="1"
import argparse,contextlib,json,math,os,signal,statistics,time,traceback
from collections import Counter
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from grad_helpers import capture_rng,restore_rng,state_digest,observe_network,seed_rng
from method_impl.ab import ABRestorer
from method_impl.ab_data import full_pool,ABPositionDataset
from method_impl.ab_evaluation import evaluate_rank,summarize
from method_impl.ab_state import snapshot as model_snapshot,restore
from method_impl.d0 import derive_seed,ImmediateEMA
from method_impl.training_io import learning_rate,atomic_torch_save,BalancedOrder
from method_impl.evaluation import choose_candidate
from optimizations.dispatch_20261027.aclnn_runtime import configure
from optimizations.dispatch_20261027.guards import require_finite
from method_impl.continuation_factory import build_model
from method_impl.psf_adjoint import install
from scripts.runtime_utils import ROOT,sha,write_json,process_identity

STOP=False
def signal_stop(*args):
    global STOP
    STOP=True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--safety-file',required=True)
    ap.add_argument('--config',required=True);ap.add_argument('--resume');ap.add_argument('--stop-after',type=int)
    a=ap.parse_args();config=json.load(open(a.config));stage=config['stage'];mode=config['mode']
    execution_policy=json.loads((ROOT/'gates/EXECUTION_POLICY_CURRENT.json').read_text())
    assert execution_policy.get('allow_ab_pilot') and execution_policy.get('allow_ab_formal_training'), 'User requested Benchmark/configuration only; A/B training is disabled'
    assert stage in ['A','B'] and mode in ['resume_probe','pilot','formal']
    assert config['precision']=='FP32' and not config['amp'] and not config['checkpointing']
    assert config['loss']==('residual+0.9*Clean_S3+0.1*physics_current_g' if stage=='A' else '0.9*mean4_Clean+0.1*mean4_physics_G4+0.1*TF_residual')
    assert config['lr_peaks']==([1e-5,5e-5,5e-5] if stage=='A' else [5e-6,1e-5,1e-5])
    total_updates=config['updates'];mb=config['microbatch'];acc=config['accumulation']
    if mode=='formal':assert total_updates==execution_policy[stage+'_target_updates']
    if mode=='pilot':assert 200<=total_updates<=500 if stage=='A' else 100<=total_updates<=500
    if mode=='resume_probe':assert total_updates==2
    for gate in [f'G2_{stage}.json',f'G3_{stage}.json']:
        assert json.loads((ROOT/'execution_20261027_continuation'/gate).read_text())['status']=='PASS',gate
    graph_gate=json.loads((ROOT/'execution_20261027_continuation'/f'G2_{stage}.json').read_text())
    for path,digest in graph_gate['model_code_sha256'].items():
        if path.startswith('method_impl/'):
            assert sha(ROOT/path)==digest,('Graph source identity drift',path)
    frozen=json.loads((ROOT/'execution_20261027_continuation'/f'{stage}_FROZEN.json').read_text())
    assert config==frozen[mode]
    if mode!='resume_probe':assert json.loads((ROOT/'execution_20261027_continuation'/f'{stage}_DDP_RESUME_AUDIT.json').read_text())['status']=='PASS'
    if mode=='formal':assert json.loads((ROOT/'execution_20261027_continuation'/f'{stage}_PILOT.json').read_text())['status']=='PASS'
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==config['world_size'] and world*mb*acc==config['effective_batch'];effective=world*mb*acc
    device=torch.device(f'npu:{local}');runtime=configure(device);torch.set_num_threads(2)
    safety=json.load(open(a.safety_file));physical=safety['physical_devices'][local];limit=safety['cards'][str(physical)]
    free,capacity=torch.npu.mem_get_info();torch.npu.set_per_process_memory_fraction(limit['allocator_cap']/capacity)
    dist.init_process_group('hccl',timeout=timedelta(minutes=20));out=ROOT/'distributed'/a.run
    write_json(out/f'RANK_{rank}.json',{**process_identity(os.getpid()),'physical_device':physical})
    pool,data_identity=full_pool()
    names=['method_impl/psf_adjoint.py','method_impl/ab.py','method_impl/ab_data.py','method_impl/ab_state.py','method_impl/ab_evaluation.py',
           'method_impl/npu_runtime.py','method_impl/d0.py','method_impl/training_io.py','scripts/run_ab_authorized.py','method_impl/continuation_factory.py','optimizations/dispatch_20261027/aclnn_runtime.py','optimizations/dispatch_20261027/guards.py','scripts/run_env.sh']
    identity={**data_identity,'config_sha256':sha(a.config),'code_sha256':{p:sha(ROOT/p) for p in names},
        'authorities':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'authority').glob('*.md'))},
        'physical_devices':safety['physical_devices'],'task_queue':os.environ['TASK_QUEUE_ENABLE'],'runtime':runtime,'torch':str(torch.__version__),'torch_npu':str(__import__('torch_npu').__version__)}
    initial=config['initial_checkpoint'];assert sha(initial)==config['initial_checkpoint_sha256']
    seed_rng(derive_seed(config['seed'],'initialization'),device)
    model=build_model(stage,initial).to(device).train();model.simulator.eval()
    if config['psf_backend']=='exact_frozen_psf_adjoint':
        proof=json.loads((ROOT/'gates/G2_AB_PSF_ADJOINT_EQUIVALENCE.json').read_text())
        assert proof['status']=='PASS' and proof['adjoint_source_sha256']==sha(ROOT/'method_impl/psf_adjoint.py') and proof['model_sha256']==sha(ROOT/'method_impl/ab.py')
        install(model.simulator)
    else:assert config['psf_backend']=='native'
    opt=torch.optim.Adam(model.parameter_groups(stage),betas=(.9,.99),weight_decay=0);ema=ImmediateEMA(model.core)
    update=0;cursor=0;best=None;payload=None;started=time.time();params=tuple(model.core.parameters())
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in params}
    if a.resume:
        payload=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert payload['identity']==identity and payload['config']==config
        restore(payload,model,opt,ema);update=payload['update'];cursor=payload['cursor']
    assert cursor==update*effective
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    dataset=ABPositionDataset(pool,config['seed'],cursor,(total_updates-update)*acc*mb,rank,world,mb)
    gen=torch.Generator().manual_seed(derive_seed(config['seed'],'loader',rank))
    loader=DataLoader(dataset,batch_size=mb,num_workers=2,multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True,generator=gen)
    iterator=iter(loader)
    if payload is not None:restore_rng(payload['rng_by_rank'][rank],device)
    else:seed_rng(derive_seed(config['seed'],'process',rank),device)
    if rank==0:
        write_json(out/'CONFIG.json',config);write_json(out/'IDENTITY.json',identity)
        if (out/'BEST.json').exists():best=json.loads((out/'BEST.json').read_text())['validation']
        elif a.resume:
            prior_best=__import__('pathlib').Path(a.resume).parent.parent/'BEST.json'
            if prior_best.exists():
                chosen=json.loads(prior_best.read_text());assert chosen['update']<=update
                assert sha(chosen['path'])==chosen['sha256']
                write_json(out/'BEST.json',chosen);best=chosen['validation']
            write_json(out/'RESUME_PARENT.json',{'checkpoint':a.resume,'sha256':sha(a.resume),'cursor':cursor,'update':update})
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    order=BalancedOrder(pool,config['seed']);seen=set();class_counts=Counter();origins=Counter();realizations=Counter()
    if rank==0:
        for pos in range(cursor):
            idx=order.at(pos);seen.add(idx);p=pool[idx]
            class_counts[p['category_id']]+=1;origins[p['origin']]+=1;realizations[p['realization_index']]+=1
    def coverage():
        return {'unique_trajectories':len(seen),'unique_clean':len({pool[i]['source_id'] for i in seen}),
                'class_presentations':dict(class_counts),'origin_presentations':dict(origins),'realization_presentations':dict(realizations)}
    def save(step):
        folder=out/'checkpoints';folder.mkdir(exist_ok=True)
        atomic_torch_save(folder/f'rng_rank{rank}.pt',capture_rng(device));dist.barrier()
        if rank==0:
            rng=[torch.load(folder/f'rng_rank{r}.pt',map_location='cpu',weights_only=False) for r in range(world)]
            p=model_snapshot(model,opt,ema,update=step,cursor=step*effective,rng=rng,config=config,identity=identity)
            p['scheduler']['total_updates']=config['scheduler_total_updates']
            p['initialization']=model.initialization
            path=folder/f'U{step:06d}.pt';assert not path.exists(),'Never overwrite a checkpoint'
            digest=atomic_torch_save(path,p);check=torch.load(path,map_location='cpu',weights_only=False)
            assert check['cursor']==step*effective and state_digest(check['core'])==state_digest(p['core'])
            assert state_digest(check['ema']['model'])==state_digest(p['ema']['model'])
            write_json(out/'LAST.json',{'path':str(path),'sha256':digest,'update':step,'cursor':step*effective,'cpu_readback':True})
        dist.barrier()
    def validate(step,full):
        nonlocal best
        opt.zero_grad(set_to_none=True);torch.npu.empty_cache();dist.barrier()
        location=out/'validation'/f'U{step:06d}_{1000 if full else 100}'
        if rank==0:write_json(out/'PHASE.json',{'phase':'VALIDATING_FIXED_FOUR_STEP','update':step,'count':1000 if full else 100})
        evaluate_rank(model,ema,device,location,rank=rank,world=world,full=full,stage=stage);dist.barrier()
        if rank==0:
            summary=summarize(location,world,full,stage);summary['update']=step
            if full:
                chosen,reason=choose_candidate(summary,best)
                write_json(location/'SELECTION.json',{'selected':chosen,'reason':reason,'previous_best_update':None if best is None else best['update'],'scope':'fixed four-step final PSNR'})
                if chosen:
                    best=summary;last=json.loads((out/'LAST.json').read_text());assert last['update']==step
                    write_json(out/'BEST.json',{**last,'validation':summary,'selection_reason':reason})
        dist.barrier();model.train();model.simulator.eval()
    records=[]
    try:
        if update==0:
            save(0)
            if mode!='resume_probe':validate(0,True)
        torch.npu.reset_peak_memory_stats()
        for step in range(update+1,total_updates+1):
            begin=time.perf_counter();opt.zero_grad(set_to_none=True);values=torch.zeros(4,device=device);wait=0.;indices_all=[];positions_all=[];protocols=[]
            if rank==0:write_json(out/'PHASE.json',{'phase':'TRAINING','update':step})
            for group,peak in zip(opt.param_groups,config['lr_peaks']):group['lr']=learning_rate(step,config['scheduler_total_updates'],peak=peak)
            for micro in range(acc):
                start=time.perf_counter();*inputs,indices,positions=next(iterator);tensors=[x.to(device) for x in inputs]
                torch.npu.synchronize();wait+=time.perf_counter()-start;indices_all.extend(indices.tolist());positions_all.extend(positions.tolist())
                context=ddp.no_sync() if micro+1<acc else contextlib.nullcontext()
                inspect=micro==0 and (step==1 or step%50==0 or step==total_updates)
                with context:
                    if inspect:
                        with observe_network(model.restorer.native,retain=True) as (calls,outputs):r=ddp(*tensors,stage=stage,seed=derive_seed(config['seed'],step,micro,rank))
                        assert len(calls)==(4 if stage=='A' else 13)
                    else:r=ddp(*tensors,stage=stage,seed=derive_seed(config['seed'],step,micro,rank))
                    assert bool(torch.isfinite(r['loss']));(r['loss']/acc).backward()
                    protocols.append({'seed':r['seed'],'sampling_seeds':r['sampling_seeds'],
                        'physical_time':r['physical_time'].detach().cpu().tolist(),
                        'g':r['g'].detach().cpu().tolist() if stage=='A' else None,
                        'TF_index':r['tf_index'].detach().cpu().tolist() if stage=='B' else None})
                    if inspect:
                        call_grad=[{'finite':o.grad is not None and bool(torch.isfinite(o.grad).all()),'norm':None if o.grad is None else float(o.grad.norm())} for o in outputs]
                        assert all(g['finite'] and g['norm']>0 for g in call_grad);outputs.clear()
                values+=torch.stack([r[k].detach() for k in ['loss','residual_loss','clean_loss','physical_loss']])/acc
                del r,tensors,inputs
            require_finite(p.grad for p in params)
            group_norms={g['name']:float(torch.stack([p.grad.float().square().sum() for p in g['params']]).sum().sqrt()) for g in opt.param_groups}
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);assert bool(norm>0)
            opt.step();ema.update(model.core)
            require_finite((*params,*ema.model.parameters()))
            dist.all_reduce(values);values/=world
            stopping=STOP or __import__('pathlib').Path(config['stop_request']).exists() or (a.stop_after is not None and step>=a.stop_after)
            stop=torch.tensor(int(stopping),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            dist.barrier();torch.npu.synchronize();free,capacity=torch.npu.mem_get_info();assert free>=limit['min_free']
            item={'update':step,'seconds':time.perf_counter()-begin,'data_wait_seconds':wait,'loss':float(values[0]),
                'residual_loss':float(values[1]),'clean_loss':float(values[2]),'physical_loss':float(values[3]),'grad_norm':float(norm),
                'group_norms':group_norms,'learning_rates':{g['name']:g['lr'] for g in opt.param_groups},'finite':True,
                'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved(),
                'device_used':capacity-free,'device_free':free,'actual_pool_indices':indices_all,'actual_global_positions':positions_all,'protocols':protocols,
                'denoiser_gradients':call_grad if step==1 or step%50==0 or step==total_updates else None}
            records.append(item)
            with (out/f'UPDATES_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(item)+'\n');f.flush()
            cursor=step*effective;update=step
            if rank==0:
                for pos in range(cursor-effective,cursor):
                    idx=order.at(pos);seen.add(idx);p=pool[idx]
                    class_counts[p['category_id']]+=1;origins[p['origin']]+=1;realizations[p['realization_index']]+=1
                if step==1 or step%10==0:
                    write_json(out/'STATUS.json',{'status':'RUNNING','stage':stage,'mode':mode,'update':step,'target_updates':total_updates,
                        'presentations':cursor,'last_loss':item,'seconds':time.time()-started,**coverage()})
                    print(json.dumps({'stage':stage,'update':step,'loss':item['loss'],'seconds':item['seconds']}),flush=True)
            checkpoint_due=mode=='resume_probe' or step%config['checkpoint_every']==0 or step==total_updates or bool(stop)
            if checkpoint_due:save(step)
            if mode!='resume_probe' and not bool(stop):
                if step%1000==0 or step==total_updates:validate(step,True)
                elif step%250==0:validate(step,False)
            if bool(stop):break
        digests=[None]*world;dist.all_gather_object(digests,state_digest(model.core.state_dict()));assert len(set(digests))==1
        summary={'status':'PASS' if update==total_updates else 'INTERRUPTED_CHECKPOINT_SAVED','stage':stage,'mode':mode,
            'completed_updates':update,'formal_updates':update if mode=='formal' else 0,'presentations':cursor,
            'finite_grad_rate':1.,'rank_parameters_equal_exact':True,'measured_updates':len(records),
            'images_per_sec':effective*len(records)/sum(x['seconds'] for x in records),'update_time_p50':statistics.median(x['seconds'] for x in records),
            'update_time_p90':float(np.percentile([x['seconds'] for x in records],90)),
            'mean_data_wait':statistics.mean(x['data_wait_seconds'] for x in records),'clipping_rate':sum(x['grad_norm']>1 for x in records)/len(records),
            'peak_allocated':max(x['peak_allocated'] for x in records),'peak_reserved':max(x['peak_reserved'] for x in records),
            'min_device_free':min(x['device_free'] for x in records),'seconds':time.time()-started,'test_used':False,
            'quality_based_stop_rules':False,'effective_batch':effective,'microbatch':mb,'accumulation':acc,'world_size':world}
        write_json(out/f'RESULT_rank{rank}.json',summary);dist.barrier()
        if rank==0:write_json(out/'RESULT.json',{**summary,**coverage()});write_json(out/'STATUS.json',{**summary,**coverage()})
        dist.barrier()
    except BaseException as exc:
        write_json(out/f'FAILURE_rank{rank}.json',{'error':repr(exc),'traceback':traceback.format_exc(),'last_completed_update':update,'cursor':cursor})
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()

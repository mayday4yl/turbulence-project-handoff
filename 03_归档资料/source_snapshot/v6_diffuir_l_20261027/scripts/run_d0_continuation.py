"""V6 D0-only joint-loss runner. No A/B method is inferred here."""
from pathlib import Path
from collections import Counter
import argparse,contextlib,json,os,signal,statistics,time,traceback
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from grad_helpers import capture_rng,restore_rng,state_digest,observe_network,seed_rng
from method_impl.d0 import D0Restorer,ImmediateEMA,derive_seed
from method_impl.training_io import (load_training_pool,PositionDataset,BalancedOrder,learning_rate,
    new_snapshot,atomic_torch_save,restore_snapshot)
from method_impl.evaluation import evaluate_rank,summarize_validation,choose_candidate
from scripts.prepare_sources import sha
from scripts.runtime_utils import ROOT,write_json,process_identity

STOP=False
def signal_stop(sig,frame):
    global STOP
    STOP=True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--safety-file',required=True)
    ap.add_argument('--config',required=True);ap.add_argument('--resume');args=ap.parse_args()
    import torch_npu
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(2);torch.npu.set_device(local);device=torch.device(f'npu:{local}')
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
    safety=json.load(open(args.safety_file));physical=safety['physical_devices'][local];limit=safety['cards'][str(physical)]
    free,total=torch.npu.mem_get_info();torch.npu.set_per_process_memory_fraction(limit['allocator_cap']/total)
    dist.init_process_group('hccl',timeout=timedelta(minutes=10))
    out=ROOT/'distributed'/args.run;write_json(out/f'RANK_{rank}.json',{**process_identity(os.getpid()),'physical_device':physical})
    config=json.load(open(args.config));mode=config['mode'];assert mode in ['resume_probe','formal']
    assert config['stage']=='D0' and config['world_size']==world and config['precision']=='FP32' and config['loss']=='residual_RGB_L1+actual_S3_Clean_L1'
    assert config['ema_beta']==.995 and config['global_clip']==1. and not config['amp'] and not config['gradient_checkpointing']
    assert config['native_diffusion_times']==1000 and config['sampling_nfe']==3
    total_updates=config['updates'];mb=config['microbatch'];acc=config['accumulation'];effective=world*mb*acc
    assert effective==config['effective_batch'] and mb>0 and acc>0
    execution_policy=json.loads((ROOT/'gates/EXECUTION_POLICY_CURRENT.json').read_text())
    assert execution_policy['allow_d0_continuation']
    frozen=json.loads((ROOT/'execution_20261027_continuation/D0_FROZEN.json').read_text())
    assert config==frozen[mode]
    for gate in ['G2_D0.json','G3_D0.json','G4_D0_PILOT.json']:
        assert json.loads((ROOT/'gates'/gate).read_text())['status']=='PASS'
    if mode=='formal':
        assert json.loads((ROOT/'execution_20261027_continuation/D0_RESUME_AUDIT.json').read_text())['status']=='PASS'
    pool,identity=load_training_pool(require_v6=config['full_v6_pool'])
    # Only executed D0 dependencies bind a resume. Adding unrelated A/B source
    # files later must not invalidate an otherwise identical D0 checkpoint.
    dependency_names=['method_impl/d0.py','method_impl/training_io.py','method_impl/evaluation.py',
        'scripts/run_d0.py','scripts/run_d0_continuation.py','scripts/runtime_utils.py','scripts/prepare_sources.py']
    code_identity={name:sha(ROOT/name) for name in dependency_names}
    identity.update(config_sha256=sha(args.config),code_sha256=code_identity,
        method_spec_sha256=sha(ROOT/'authority/V6_DiffUIR_L_D0_A_B_METHOD_SPEC_v1.0.md'),
        training_handoff_sha256=sha(ROOT/'authority/V6_TRAINING_DATA_HANDOFF_v1.0_20261027.md'),
        user_overrides_sha256=sha(ROOT/'authority/USER_EXECUTION_OVERRIDES_20261027.md'),
        torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),physical_devices=safety['physical_devices'])
    identity['runtime_sources_sha256']={str(p):sha(p) for p in [ROOT/'scripts/run_env.sh',
        ROOT.parent/'v5_diffuir_l_stage1_20261026/upstream/src/model.py',ROOT.parent/'v5_diffuir_l_stage1_20261026/scripts/common.py',
        ROOT.parent/'v5_diffuir_l_stage1_20261026/scripts/grad_helpers.py',
        ROOT.parent/'v5_concertormer_10k_20261021/training/image_metrics.py']}
    seed_rng(derive_seed(config['seed'],'process',rank),device)
    model=D0Restorer(ROOT.parent/'v5_diffuir_l_stage1_20261026/weights/model-300.pt').to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=2e-5,betas=(.9,.99),weight_decay=0);ema=ImmediateEMA(model.network)
    update=0;cursor=0;best=None;start_time=time.time();timings=[];local_pairs=set();local_sources=set()
    payload=None
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu',weights_only=False)
        if payload['config']==config:
            assert payload['identity']==identity,'Resume identity drift'
        else:
            assert sha(args.resume)==config['initial_checkpoint_sha256']
            assert payload['config']==json.loads((ROOT/'TRAINING_FROZEN_CONFIG.json').read_text())['D0']['formal']
            assert payload['update']==10000 and payload['cursor']==600000
            for name,digest in payload['identity']['code_sha256'].items():assert sha(ROOT/name)==digest
            assert payload['identity']['runtime_sources_sha256']==identity['runtime_sources_sha256']
            for key in ['fingerprint_sha256','pair_manifest_sha256','method_spec_sha256','training_handoff_sha256','torch','torch_npu','physical_devices']:
                assert payload['identity'][key]==identity[key],key
        restore_snapshot(payload,model,opt,ema);update=payload['update'];cursor=payload['cursor']
        if (out/'BEST.json').exists():best=json.loads((out/'BEST.json').read_text())['validation']
        else:
            inherited=json.loads((ROOT/'distributed/D0_formal10000_01/BEST.json').read_text())
            assert sha(inherited['path'])==inherited['sha256']
            best=inherited['validation']
            if rank==0:write_json(out/'BEST.json',inherited)
    assert args.resume and cursor==update*effective and update>=10000
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    if rank==0:
        if (out/'CONFIG.json').exists():assert json.loads((out/'CONFIG.json').read_text())==config
        else:write_json(out/'CONFIG.json',config);write_json(out/'IDENTITY.json',identity)
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    ds=PositionDataset(pool,config['seed'],cursor,(total_updates-update)*acc*mb,rank,world,mb)
    kwargs={'num_workers':config.get('loader_workers',2)}
    if kwargs['num_workers']:kwargs.update(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
    loader_generator=torch.Generator(device='cpu').manual_seed(derive_seed(config['seed'],'loader_workers',rank))
    loader=DataLoader(ds,batch_size=mb,shuffle=False,drop_last=False,generator=loader_generator,**kwargs)
    iterator=iter(loader);global_order=BalancedOrder(pool,config['seed'])
    if payload is not None:restore_rng(payload['rng_by_rank'][rank],device)
    seen_pairs=set(global_order.at(i) for i in range(min(cursor,len(pool))))
    # The fixed traversal is periodic by full population; after one cycle all identities were seen.
    if cursor>=len(pool):seen_pairs=set(range(len(pool)))
    seen_sources={pool[i]['source_id'] for i in seen_pairs}
    class_presentations=Counter();origin_presentations=Counter();realization_presentations=Counter()
    if rank==0 and cursor:
        whole,remainder=divmod(cursor,len(pool))
        for row in pool:
            class_presentations[row['category_id']]+=whole;origin_presentations[row['origin']]+=whole;realization_presentations[row['realization_index']]+=whole
        for pos in range(whole*len(pool),cursor):
            row=pool[global_order.at(pos)];class_presentations[row['category_id']]+=1;origin_presentations[row['origin']]+=1;realization_presentations[row['realization_index']]+=1
    source_category={p['source_id']:p['category_id'] for p in pool}
    def coverage():
        return {'class_presentations':dict(class_presentations),'origin_presentations':dict(origin_presentations),
            'realization_presentations':dict(realization_presentations),
            'class_unique_pairs':dict(Counter(pool[i]['category_id'] for i in seen_pairs)),
            'class_unique_sources':dict(Counter(source_category[s] for s in seen_sources))}
    checkpoint_dir=out/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    def save_checkpoint(step):
        rng_file=checkpoint_dir/f'rng_rank{rank}.pt';atomic_torch_save(rng_file,capture_rng(device));dist.barrier()
        if rank==0:
            rng=[torch.load(checkpoint_dir/f'rng_rank{r}.pt',map_location='cpu',weights_only=False) for r in range(world)]
            payload=new_snapshot(model,opt,ema,update=step,cursor=step*effective,rng=rng,config=config,identity=identity)
            payload['scheduler']={'type':'constant_terminal_LR_continuation','last_update':step,'total_updates':total_updates,
                'base_update':10000,'constant_lr':config['continuation_lr'],'parent_scheduler':{'total_updates':10000,'peak':2e-5,'warmup_fraction':.02,'end_ratio':.1}}
            path=checkpoint_dir/f'U{step:06d}.pt' 
            if path.exists():
                existing=torch.load(path,map_location='cpu',weights_only=False)
                assert existing['update']==step and state_digest(existing['network'])==state_digest(payload['network'])
                digest=sha(path)
            else:digest=atomic_torch_save(path,payload)
            check=torch.load(path,map_location='cpu',weights_only=False)
            assert check['cursor']==step*effective and state_digest(check['network'])==state_digest(payload['network']) and state_digest(check['ema']['model'])==state_digest(payload['ema']['model'])
            write_json(out/'LAST.json',{'path':str(path),'sha256':digest,'update':step,'cursor':step*effective,'cpu_readback':True})
        dist.barrier()
    def validation(step,full):
        nonlocal best
        opt.zero_grad(set_to_none=True);torch.npu.empty_cache();dist.barrier()
        location=out/'validation'/f'U{step:06d}_{1000 if full else 100}'
        evaluate_rank(model,ema,device,location,rank=rank,world=world,full=full);dist.barrier()
        if rank==0:
            summary=summarize_validation(location,world,full);summary['update']=step
            if full:
                select,reason=choose_candidate(summary,best)
                write_json(location/'SELECTION.json',{'selected':select,'reason':reason,'previous_best_update':None if best is None else best['update']})
                if select:
                    best=summary;last=json.loads((out/'LAST.json').read_text());assert last['update']==step
                    write_json(out/'BEST.json',{**last,'validation':summary,'selection_reason':reason})
        dist.barrier();model.train()
    if not (checkpoint_dir/f'U{update:06d}.pt').exists():save_checkpoint(update)
    torch.npu.reset_peak_memory_stats();previous_end=time.perf_counter()
    try:
        for step in range(update+1,total_updates+1):
            begin=time.perf_counter();opt.zero_grad(set_to_none=True);data_wait=0.;loss_values=torch.zeros(3,device=device);actual_indices=[];actual_positions=[];call_gradients=None
            lr=config['continuation_lr']
            for group in opt.param_groups:group['lr']=lr
            for micro in range(acc):
                data_start=time.perf_counter();x,c,indices,positions=next(iterator)
                x=x.to(device);c=c.to(device);torch.npu.synchronize();data_wait+=time.perf_counter()-data_start
                for idx in indices.tolist():local_pairs.add(idx);local_sources.add(pool[idx]['source_id'])
                actual_indices.extend(indices.tolist());actual_positions.extend(positions.tolist())
                context=ddp.no_sync() if micro+1<acc else contextlib.nullcontext()
                with context:
                    audit_calls=micro==0 and (step==1 or step%50==0 or step==total_updates)
                    if audit_calls:
                        with observe_network(model.native,retain=True) as (calls,outputs):
                            result=ddp(x,c,seed=derive_seed(config['seed'],step,micro,rank))
                        assert len(calls)==4
                    else:result=ddp(x,c,seed=derive_seed(config['seed'],step,micro,rank))
                    assert bool(torch.isfinite(result['loss']))
                    (result['loss']/acc).backward()
                    if audit_calls:
                        assert all(o.grad is not None and bool(torch.isfinite(o.grad).all()) for o in outputs)
                        call_gradients=[{'call':i,'branch':'residual' if i==0 else 'S3','activation_grad_l2':float(o.grad.float().norm()),
                                         'finite':True,'nonzero':bool(o.grad.abs().max()>0)} for i,o in enumerate(outputs)]
                        outputs.clear()
                loss_values+=torch.stack([result[k].detach() for k in ['loss','residual_loss','clean_loss']])/acc
                del result,x,c
            assert all(p.grad is not None for p in model.parameters())
            assert bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert bool(norm>0),'All-zero joint gradient'
            opt.step();ema.update(model.network)
            assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
            assert bool(torch.stack([torch.isfinite(p).all() for p in ema.model.parameters()]).all())
            dist.all_reduce(loss_values);loss_values/=world
            stop=torch.tensor(int(STOP or Path(config['stop_request']).exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            dist.barrier();torch.npu.synchronize();end=time.perf_counter();free,total=torch.npu.mem_get_info()
            assert free>=limit['min_free'],'Physical HBM safety reserve violated'
            record={'update':step,'seconds':end-begin,'interval_seconds':end-previous_end,'data_wait_seconds':data_wait,
                'loss':float(loss_values[0]),'residual_loss':float(loss_values[1]),'clean_loss':float(loss_values[2]),'grad_norm':float(norm),
                'lr':lr,'ema_updates':ema.updates,'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved(),
                'device_used':total-free,'device_free':free,'finite':True,'local_unique_pairs':len(local_pairs),'local_unique_sources':len(local_sources),
                'actual_pool_indices':actual_indices,'actual_global_positions':actual_positions,'denoiser_output_gradient_audit':call_gradients}
            with (out/f'UPDATES_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
            timings.append(record);cursor=step*effective;update=step
            if rank==0:
                for pos in range(cursor-effective,cursor):
                    idx=global_order.at(pos);seen_pairs.add(idx);seen_sources.add(pool[idx]['source_id'])
                    row=pool[idx];class_presentations[row['category_id']]+=1;origin_presentations[row['origin']]+=1;realization_presentations[row['realization_index']]+=1
                if step%10==0 or step==1:
                    write_json(out/'STATUS.json',{'status':'RUNNING','mode':mode,'update':step,'target_updates':total_updates,
                        'presentations':cursor,'unique_pairs':len(seen_pairs),'unique_sources':len(seen_sources),'completed_population_cycles':cursor//len(pool),
                        'last_loss':record,'seconds':time.time()-start_time,'no_quality_stop_rules':True,**coverage()})
                    print(json.dumps({'run':args.run,'step':step,'loss':record['loss'],'seconds':record['seconds'],'presentations':cursor}),flush=True)
            if mode=='resume_probe' or step%config['checkpoint_every']==0 or step==total_updates or bool(stop):
                save_checkpoint(step)
            if mode=='formal' and not bool(stop):
                if step%1000==0 or step==total_updates:validation(step,True)
                elif step%250==0:validation(step,False)
            previous_end=time.perf_counter()
            if bool(stop):break
        assert update==total_updates or STOP or bool(stop)
        digest=state_digest(model.network.state_dict());all_digests=[None]*world;dist.all_gather_object(all_digests,digest)
        assert len(set(all_digests))==1,'Rank weights diverged'
        measure=[r for r in timings if r['update']>3] if mode=='benchmark' else timings
        summary={'status':'PASS' if update==total_updates else 'INTERRUPTED_CHECKPOINT_SAVED','mode':mode,'stage':'D0',
            'completed_updates':update,'formal_updates':update-config['continuation_base_update'] if mode=='formal' else 0,'microbatch':mb,'accumulation':acc,'world_size':world,
            'effective_batch':effective,'full_v6_pool':config['full_v6_pool'],'rank_parameters_equal_exact':True,
            'per_rank':rank,'measured_updates':len(measure),'update_time_p50':statistics.median(r['seconds'] for r in measure),
            'update_time_p90':float(np.percentile([r['seconds'] for r in measure],90)),
            'images_per_sec':effective*len(measure)/sum(r['interval_seconds'] for r in measure),
            'mean_data_wait':statistics.mean(r['data_wait_seconds'] for r in measure),'peak_allocated':max(r['peak_allocated'] for r in timings),
            'peak_reserved':max(r['peak_reserved'] for r in timings),'min_device_free':min(r['device_free'] for r in timings),
            'finite_grad_rate':1.,'clipping_rate':sum(r['grad_norm']>1 for r in measure)/len(measure),
            'presentations':cursor,'unique_sources':len(seen_sources) if rank==0 else None,'unique_pairs':len(seen_pairs) if rank==0 else None,
            'local_unique_sources':len(local_sources),'local_unique_pairs':len(local_pairs),'test_used':False,
            'negative_validation_results_are_not_blockers':True,'model_discarded':mode=='benchmark'}
        if rank==0:summary.update(coverage())
        write_json(out/f'RESULT_rank{rank}.json',summary)
        dist.barrier()
        if rank==0:
            write_json(out/'RESULT.json',summary)
            write_json(out/'STATUS.json',{**summary,'seconds':time.time()-start_time})
        dist.barrier()
    except BaseException as exc:
        write_json(out/f'FAILURE_rank{rank}.json',{'error':repr(exc),'traceback':traceback.format_exc(),'last_completed_update':update,'cursor':cursor})
        raise
    finally:dist.destroy_process_group()

if __name__=='__main__':main()

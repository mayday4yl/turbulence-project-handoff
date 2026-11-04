"""Full-state D0 continuation on a new fixed mixed-g manifest. Native loss unchanged."""
import argparse,contextlib,json,os,signal,time,traceback
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from grad_helpers import capture_rng,restore_rng,state_digest,observe_network,seed_rng
from method_impl.d0 import ImmediateEMA,derive_seed
from method_impl.d0_official_residual import D0OfficialResidualRestorer
from method_impl.training_io import new_snapshot,atomic_torch_save,restore_snapshot
from scripts.multistrength_order_20261029 import MixedPositionDataset as PositionDataset
from scripts.audit_ddp_resume import equal_tree
from scripts.runtime_utils import ROOT,sha,write_json,process_identity
from scripts.multistrength_support_20261029 import EX,BASE,PARENT,PARENT_SHA,freeze,load_pool,evaluate_rank,summarize,select_best

STOP=False
def signal_stop(sig,frame):
    global STOP
    STOP=True
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--safety-file',required=True)
    ap.add_argument('--mode',choices=['baseline','probe','formal'],required=True);ap.add_argument('--end-update',type=int)
    ap.add_argument('--resume',default=str(PARENT));a=ap.parse_args()
    import torch_npu
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==4
    torch.set_num_threads(2);torch.npu.set_device(local);device=torch.device(f'npu:{local}')
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
    safety=json.load(open(a.safety_file));assert safety['physical_devices']==[4,5,6,7]
    physical=safety['physical_devices'][rank];limit=safety['cards'][str(physical)]
    free,total=torch.npu.mem_get_info();torch.npu.set_per_process_memory_fraction(limit['allocator_cap']/total)
    dist.init_process_group('hccl',timeout=timedelta(minutes=10))
    out=ROOT/'distributed'/a.run;write_json(out/f'RANK_{rank}.json',{**process_identity(os.getpid()),'physical_device':physical})
    freeze();cfg=json.load(open(EX/'CONFIG.json'));assert cfg['microbatch']==5 and cfg['accumulation']==3 and cfg['effective_batch']==60
    policy=json.load(open(ROOT/'gates/EXECUTION_POLICY_CURRENT.json'))
    assert policy['active_execution']==EX.name and policy['allow_d0_continuation'] and not policy['allow_ab_formal_training']
    if a.mode=='formal':assert json.load(open(EX/'RESUME_GATE.json'))['status']=='PASS'
    pool,identity=load_pool();identity.update(config_sha256=sha(EX/'CONFIG.json'),torch=str(torch.__version__),torch_npu=str(torch_npu.__version__))
    seed_rng(derive_seed(cfg['seed'],'process',rank),device)
    model=D0OfficialResidualRestorer(ROOT.parent/'v5_diffuir_l_stage1_20261026/weights/model-300.pt').to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=2e-6,betas=(.9,.99),weight_decay=0);ema=ImmediateEMA(model.network)
    payload=torch.load(a.resume,map_location='cpu',weights_only=False)
    migrating=payload['config']!=cfg
    if migrating:
        assert sha(a.resume)==PARENT_SHA and payload['update']==BASE and payload['cursor']==BASE*60
        assert payload['identity']['physical_devices']==[4,5,6,7] and payload['scaler'] is None
        assert all(g['lr']==2e-6 and tuple(g['betas'])==(.9,.99) and g['weight_decay']==0 for g in payload['optimizer']['param_groups'])
        cursor=0;parent_scheduler=payload['scheduler']
    else:
        assert payload['identity']==identity
        cursor=payload['cursor'];assert cursor==(payload['update']-BASE)*60
        parent_scheduler=payload['parent_scheduler']
    restore_snapshot(payload,model,opt,ema);update=payload['update'];initial_update=update
    assert state_digest(model.network.state_dict())==state_digest(payload['network'])
    assert equal_tree(ema.state_dict(),payload['ema']) and equal_tree(opt.state_dict(),payload['optimizer'])
    assert all(int(v['step'])==update for v in opt.state_dict()['state'].values())
    write_json(out/f'RESUME_INHERITANCE_rank{rank}.json',{'status':'PASS','parent_sha256':sha(a.resume),
        'update':update,'new_manifest_cursor':cursor,'old_G4_cursor':BASE*60,'raw_EMA_Adam_buffers_exact':True,
        'constant_lr':2e-6,'optimizer_state_count':len(opt.state_dict()['state'])})
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    if rank==0:write_json(out/'CONFIG.json',cfg);write_json(out/'IDENTITY.json',identity)
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    if a.mode=='baseline':
        restore_rng(payload['rng_by_rank'][rank],device)
        location=out/'validation/U095544_G05_1000'
        evaluate_rank(model,ema,device,location,rank,world,True,[.5],True);dist.barrier()
        if rank==0:
            summary=summarize(location,world,True,BASE,[.5])
            write_json(out/'RESULT.json',{'status':'PASS','optimizer_updates_executed':0,'new_g05_count':1000,'validation':summary})
        dist.barrier();dist.destroy_process_group();return
    target=a.end_update or cfg['updates'];assert target>update
    ds=PositionDataset(pool,cfg['seed'],cursor,(target-update)*15,rank,world,5)
    generator=torch.Generator(device='cpu').manual_seed(derive_seed(cfg['seed'],'loader_workers',rank))
    loader=DataLoader(ds,batch_size=5,shuffle=False,drop_last=False,num_workers=2,multiprocessing_context='spawn',
        prefetch_factor=2,persistent_workers=True,generator=generator);iterator=iter(loader)
    restore_rng(payload['rng_by_rank'][rank],device);assert equal_tree(capture_rng(device),payload['rng_by_rank'][rank])
    write_json(out/f'RNG_RESTORE_rank{rank}.json',{'status':'PASS','physical_device':physical,'exact':True})
    del payload
    cp=out/'checkpoints';cp.mkdir(exist_ok=True);schedule_source=parent_scheduler
    def save(step):
        atomic_torch_save(cp/f'rng_rank{rank}.pt',capture_rng(device));dist.barrier()
        if rank==0:
            rng=[torch.load(cp/f'rng_rank{i}.pt',map_location='cpu',weights_only=False) for i in range(world)]
            value=new_snapshot(model,opt,ema,update=step,cursor=(step-BASE)*60,rng=rng,config=cfg,identity=identity)
            value['parent_scheduler']=schedule_source
            value['scheduler']=schedule_source if step==BASE else {'type':'mixed_manifest_constant_LR_continuation','last_update':step,'constant_lr':2e-6,'parent_scheduler':schedule_source}
            value.update(parent_checkpoint_sha256=PARENT_SHA,previous_data_cursor=BASE*60,total_presentations_lineage=step*60,
                stage_updates=step-BASE,stage_presentations=(step-BASE)*60)
            path=cp/f'U{step:06d}.pt'
            if path.exists():
                check=torch.load(path,map_location='cpu',weights_only=False);assert equal_tree(check['network'],value['network']);digest=sha(path)
            else:digest=atomic_torch_save(path,value)
            check=torch.load(path,map_location='cpu',weights_only=False);assert equal_tree(check,value),'Full-state CPU readback mismatch'
            write_json(out/'LAST.json',{'path':str(path),'sha256':digest,'update':step,'cursor':(step-BASE)*60,
                'stage_updates':step-BASE,'old_G4_cursor':BASE*60,'cpu_readback':True})
        dist.barrier()
    def validate(step,full):
        opt.zero_grad(set_to_none=True);torch.npu.empty_cache();dist.barrier()
        location=out/'validation'/f'U{step:06d}_{1000 if full else 100}'
        evaluate_rank(model,ema,device,location,rank,world,full);dist.barrier()
        if rank==0:
            summary=summarize(location,world,full,step)
            if full:
                candidates=json.load(open(out/'CANDIDATES.json'));last=json.load(open(out/'LAST.json'));assert last['update']==step
                candidates.append({**last,'validation':summary,'validation_path':str(location)})
                best=select_best(candidates);write_json(out/'CANDIDATES.json',candidates);write_json(out/'BEST.json',best)
                write_json(location/'SELECTION.json',{'G4_eligible':summary['by_g']['4']['psnr']>=21.16514236838766-.2,'best_update':best['update'],'criterion':best['selection']})
            budget=json.load(open(EX/'TIMING_BUDGET.json'))
            write_json(EX/'TIMING_BUDGET.json',{**budget,'latest_validation_seconds':summary['seconds'],'latest_full':full,
                'updated_unix':time.time(),'training_stop_unix':None,'noon_is_review_not_stop':True})
        dist.barrier();model.train()
    def requested_stop():
        return STOP or (EX/'STOP_REQUEST.json').exists()
    def deadline():return False  # Latest human authority: no automatic noon stop or fixed update budget.
    save(update)
    if rank==0 and a.mode=='formal':
        baseline=json.load(open(EX/'BASELINE_CANDIDATE.json'));write_json(out/'CANDIDATES.json',[baseline]);write_json(out/'BEST.json',baseline)
    dist.barrier();started=time.time();previous=time.perf_counter();torch.npu.reset_peak_memory_stats();stop=False
    try:
        for step in range(update+1,target+1):
            flag=torch.tensor(int(requested_stop() or (a.mode=='formal' and deadline())),device=device)
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if bool(flag):save(update);stop=True;break
            begin=time.perf_counter();opt.zero_grad(set_to_none=True);losses=torch.zeros(1,device=device)
            positions=[];indices=[];gs=[];audits=[];taus=[]
            for micro in range(3):
                x,c,ii,pp=next(iterator);positions+=pp.tolist();indices+=ii.tolist();gs += [pool[i]['g'] for i in ii.tolist()]
                x=x.to(device);c=c.to(device)
                audit=(step==initial_update+1 or step%50==0) and micro==0
                with ddp.no_sync() if micro<2 else contextlib.nullcontext():
                    if audit:
                        with observe_network(model.native,retain=True) as (calls,outputs):
                            result=ddp(x,c,seed=derive_seed(cfg['seed'],step,micro,rank))
                        assert len(calls)==len(outputs)==1
                        direct=(outputs[0]-(2*x-1-(2*c-1))).abs().mean()
                        assert torch.allclose(result['loss'],direct,atol=3e-8,rtol=1e-6),'Native raw residual L1 changed'
                    else:result=ddp(x,c,seed=derive_seed(cfg['seed'],step,micro,rank))
                    assert bool(torch.isfinite(result['loss']));(result['loss']/3).backward()
                    losses+=result['loss'].detach().reshape(1)/3;taus+=result['tau'].tolist()
                    if audit:
                        for value in outputs:
                            assert value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool(value.grad.abs().max()>0)
                            audits.append({'finite':True,'nonzero':True,'activation_grad_l2':float(value.grad.norm()),'nfe':1,'manual_L1_abs':float((direct-result['loss']).abs())})
                        del direct;outputs.clear()
                    del result,x,c
            assert all(p.grad is not None for p in model.parameters())
            assert bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert bool(norm>0)
            assert all(g['lr']==2e-6 for g in opt.param_groups)
            opt.step();ema.update(model.network)
            assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
            assert bool(torch.stack([torch.isfinite(p).all() for p in ema.model.parameters()]).all())
            dist.all_reduce(losses);losses/=world;dist.barrier();torch.npu.synchronize()
            end=time.perf_counter();free,total=torch.npu.mem_get_info();assert free>=limit['min_free']
            update=step;cursor=(step-BASE)*60
            record={'update':step,'stage_update':step-BASE,'stage_presentations':cursor,'loss':float(losses[0]),'residual_loss':float(losses[0]),
                'grad_norm':float(norm),'lr':2e-6,'ema_updates':ema.updates,'training_nfe_per_sample':1,'sampling_training_calls':0,
                'training_microbatches':3,'actual_pool_indices':indices,'actual_global_positions':positions,'g_values':gs,'tau':taus,
                'denoiser_output_gradient_audit':audits,'seconds':end-begin,'interval_seconds':end-previous,
                'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved(),'device_free':free,'finite':True}
            with (out/f'UPDATES_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
            if rank==0 and (step-BASE)%10==0:
                write_json(out/'STATUS.json',{'status':'RUNNING','unix':time.time(),'update':update,'last_loss':record})
                print(json.dumps({'update':update,'loss':record['loss'],'seconds':record['seconds']}),flush=True)
            stop_flag=torch.tensor(int(requested_stop() or (a.mode=='formal' and deadline())),device=device);dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX)
            stop=bool(stop_flag);u=step-BASE
            if a.mode=='probe' or u%1000==0 or step==target or stop:save(step)
            if a.mode=='formal' and not stop:
                noon_flag=torch.tensor(int(time.time()>=cfg['noon_review_unix'] and not (EX/'NOON_REVIEW.json').exists()),device=device)
                dist.all_reduce(noon_flag,op=dist.ReduceOp.MAX);noon_due=bool(noon_flag)
                if u%2000==0 or noon_due:
                    save(step);validate(step,True)
                    if rank==0 and noon_due:
                        write_json(EX/'NOON_REVIEW.json',{'status':'PASS','update':step,'stage_updates':u,'unix':time.time(),
                            'validation':json.load(open(out/'validation'/f'U{step:06d}_1000/SUMMARY.json')),
                            'last':json.load(open(out/'LAST.json')),'best':json.load(open(out/'BEST.json')),
                            'training_continues':True,'automatic_noon_stop':False})
                    dist.barrier()
            previous=time.perf_counter()
            if stop:break
        final_validation=False
        if a.mode=='formal':
            save(update)
            validate(update,True);final_validation=True
        digest=state_digest(model.network.state_dict());digests=[None]*world;dist.all_gather_object(digests,digest);assert len(set(digests))==1
        result={'status':'PASS','mode':a.mode,'completed_updates':update,'segment_updates':update-initial_update,'stage_updates':update-BASE,
            'stage_presentations':cursor,'full_checkpoint_saved':True,'final_five_g_validation':final_validation,
            'user_stopped':requested_stop(),'training_deadline_reached':a.mode=='formal' and deadline(),
            'seconds':time.time()-started,'rank_parameters_equal':True,'formal_weights_from_probe':False,'test_used':False}
        write_json(out/f'RESULT_rank{rank}.json',result);dist.barrier()
        if rank==0:write_json(out/'RESULT.json',result);write_json(out/'STATUS.json',result)
        dist.barrier()
    except BaseException as exc:
        write_json(out/f'FAILURE_rank{rank}.json',{'error':repr(exc),'traceback':traceback.format_exc(),'last_completed_update':update,'stage_cursor':cursor});raise
    finally:dist.destroy_process_group()
if __name__=='__main__':main()

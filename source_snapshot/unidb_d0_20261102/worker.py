"""Eight-rank native UniDB training, fixed validation and complete snapshots."""
import argparse
import contextlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from core import (ROOT,NativeObjective,configure_npu,cpu_tree,derive,ema_restore,
    ema_save,initialize,make_ema,metrics_module,parameter_digest,process,
    rng_capture,rng_restore,sha,sources_ok,write)
from data import Positions,load_pool,validation_groups


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def shared_decision(value,device,rank):
    flag=torch.tensor(int(bool(value)) if rank==0 else 0,device=device,dtype=torch.int32)
    dist.broadcast(flag,src=0)
    return bool(flag.item())


def measure(pred,clean,metric):
    p=pred.detach().float().cpu();c=clean.detach().float().cpu()
    ss,lp=metric.image_metrics(p,c)
    a=p.numpy().astype(np.float64);b=c.numpy().astype(np.float64)
    mse=np.mean((a-b)**2,axis=(1,2,3));mae=np.mean(np.abs(a-b),axis=(1,2,3))
    return [{'psnr':-10*math.log10(max(float(e),1e-30)),'ssim':s,'lpips':l,'mae':float(m)}
        for e,m,s,l in zip(mse,mae,ss,lp)]


def evaluate(net,device,out,rank,world,full,update):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    previous_rng=rng_capture(device);was_training=net.training;net.eval()
    sde=process(device);sde.set_model(net);metric=metrics_module();records=[];started=time.time()
    groups=validation_groups(full);visual_sources=sorted({s for _,s,_ in groups})[:4]
    try:
        with torch.no_grad():
            for group_index in range(rank,len(groups),world):
                g,source,rows=groups[group_index];key=f'{g:g}'
                for row in rows:
                    assert sha(row['clean_path'])==row['clean_sha256']
                    assert sha(row['states'][key]['path'])==row['states'][key]['sha256']
                y=torch.from_numpy(np.stack([np.load(r['states'][key]['path'],allow_pickle=False) for r in rows])).to(device)
                c=torch.from_numpy(np.stack([np.load(r['clean_path'],allow_pickle=False) for r in rows])).to(device)
                # Groups always consist of the same source's four realizations.
                # Seeds and batching are identical for monitor/full/all checkpoints.
                seed=derive(843001,source,'four_realizations_native_SDE')
                torch.manual_seed(seed);torch.npu.manual_seed(seed)
                calls=[0];hook=net.register_forward_hook(lambda m,a,v:calls.__setitem__(0,calls[0]+1))
                sde.set_mu(y);torch.npu.synchronize();begin=time.perf_counter()
                try:pred=sde.reverse_sde(y,save_states=False)
                finally:hook.remove()
                torch.npu.synchronize();seconds=time.perf_counter()-begin
                assert calls[0]==100 and bool(torch.isfinite(pred).all())
                measured=measure(pred,c,metric);inp=measure(y,c,metric)
                for i,row in enumerate(rows):
                    item={'sample_id':row['sample_id'],'source_id':source,'g':g,
                        'group_seed':seed,'group_batch':4,'group_position':i,'nfe':100,
                        'update':update,'metrics':measured[i],'input_metrics':inp[i],
                        'input_path':row['states'][key]['path'],'input_sha256':row['states'][key]['sha256'],
                        'clean_path':row['clean_path'],'group_inference_seconds':seconds}
                    if full or source in visual_sources:
                        path=out/'predictions'/f'G{key}'/(row['sample_id']+'.npy')
                        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,pred[i].cpu().numpy(),allow_pickle=False)
                        item.update(prediction=str(path),prediction_sha256=sha(path))
                    assert all(math.isfinite(v) for d in [measured[i],inp[i]] for v in d.values())
                    records.append(item)
                if len(records)%80==0:
                    write(out/f'PROGRESS_rank{rank}.json',{'records':len(records),'seconds':time.time()-started})
                del y,c,pred
    finally:
        net.train(was_training);rng_restore(previous_rng,device)
    path=out/f'rank{rank}.jsonl'
    with path.open('w') as f:
        for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
    write(out/f'DONE_rank{rank}.json',{'status':'PASS','count':len(records),'seconds':time.time()-started,'sha256':sha(path)})
    dist.barrier()
    summary=None
    if rank==0:
        data=[]
        for r in range(world):
            done=json.loads((out/f'DONE_rank{r}.json').read_text());assert done['status']=='PASS'
            data.extend(json.loads(s) for s in (out/f'rank{r}.jsonl').read_text().splitlines())
        expected={(row['sample_id'],g) for g,_,rows in groups for row in rows}
        assert len(data)==len(expected) and {(r['sample_id'],r['g']) for r in data}==expected
        by={f'{g:g}':{key:float(np.mean([r['metrics'][key] for r in data if r['g']==g]))
            for key in ['psnr','ssim','lpips','mae']} for g in [.5,1,2,3,4]}
        summary={'status':'PASS','update':update,'weights':'EMA','full':full,
            'count':len(data),'sources_per_g':250 if full else 25,'per_g':1000 if full else 100,
            'nfe_per_image':100,'by_g':by,'seconds':max(json.loads((out/f'DONE_rank{r}.json').read_text())['seconds'] for r in range(world)),
            'test_used':False,'image_clip_crop_resize':False,'selection_role':'full_validation' if full else 'monitor_only'}
        write(out/'SUMMARY.json',summary)
    answer=[summary];dist.broadcast_object_list(answer,src=0)
    return answer[0]


def checkpoint(out,update,cursor,objective,optimizer,ema,device,rank,world,cfg,counts):
    sources_ok();local_rng=rng_capture(device);all_rng=[None]*world
    dist.all_gather_object(all_rng,local_rng)
    digest=parameter_digest(objective.network);digests=[None]*world
    dist.all_gather_object(digests,digest);assert len(set(digests))==1
    if rank==0:
        folder=out/'checkpoints';folder.mkdir(exist_ok=True)
        path=folder/f'U{update:08d}.pt'
        if not path.exists():
            payload={'format':'UNIDB_NATIVE_FULL_STATE_V1','update':update,'cursor':cursor,
                'network':cpu_tree(objective.network.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
                'ema':ema_save(ema),'rng_by_rank':all_rng,'config':cfg,'counts_by_g':dict(counts),
                'identity':{'weight':json.loads((ROOT/'evidence/WEIGHT_DOWNLOAD.json').read_text()),
                    'data':json.loads((ROOT/'evidence/DATA_GATE.json').read_text()),
                    'source_manifest_sha256':sha(ROOT/'protocol/RUN_SOURCE.json')},
                'scheduler':{'type':'constant','lr':cfg['lr']},'scaler':None}
            temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
            identity={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
                'update':update,'cursor':cursor,'rank_sample_digests_equal':True}
            write(path.with_suffix('.json'),identity)
        identity=json.loads(path.with_suffix('.json').read_text());write(out/'LATEST.json',identity)
    dist.barrier();rng_restore(local_rng,device)


def restore(path,objective,optimizer,ema,device,rank,cfg):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['format']=='UNIDB_NATIVE_FULL_STATE_V1'
    for key in ['microbatch','accumulation','world_size','effective_batch','lr','seed','gamma_inv']:
        assert payload['config'][key]==cfg[key],key
    objective.network.load_state_dict(payload['network'],strict=True)
    optimizer.load_state_dict(payload['optimizer']);ema_restore(ema,payload['ema'])
    rng_restore(payload['rng_by_rank'][rank],device)
    return payload['update'],payload['cursor'],Counter(payload['counts_by_g'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--mode',choices=['bench','probe','pilot','formal'],required=True)
    parser.add_argument('--microbatch',type=int,default=4);parser.add_argument('--accumulation',type=int,default=1)
    parser.add_argument('--steps',type=int,default=20);parser.add_argument('--resume')
    parser.add_argument('--full-eval',action='store_true');args=parser.parse_args()
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    device=configure_npu(local);dist.init_process_group('hccl',timeout=timedelta(hours=2))
    write(out/f'RANK_{rank}.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        'rank':rank,'local':local,'world':world,'device':str(device)})
    net,load_report=initialize(device);objective=NativeObjective(net,device).train()
    cfg={'microbatch':args.microbatch,'accumulation':args.accumulation,'world_size':world,
        'effective_batch':args.microbatch*args.accumulation*world,'lr':1e-5,'seed':843101,
        'gamma_inv':1e-7,'T':100,'precision':'FP32','HF32':False,'loss':'official_native_matching_L1',
        'loss_weight':1.,'training_NFE':1,'ema_beta':.995,'ema_update_every':10,'gradient_clipping':False,
        'deadline_unix':json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())['deadline_unix']}
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['lr'],betas=(.9,.99),weight_decay=0.)
    ema=make_ema(net,device)
    torch.manual_seed(derive(cfg['seed'],'rank',rank));torch.npu.manual_seed(derive(cfg['seed'],'rank',rank))
    np.random.seed(843101+rank);random.seed(843101+rank)
    update,cursor,counts=0,0,Counter()
    if args.resume:
        update,cursor,counts=restore(args.resume,objective,optimizer,ema,device,rank,cfg)
        log=out/f'UPDATES_rank{rank}.jsonl'
        if log.exists():
            committed=[];tail=[]
            for line in log.read_text().splitlines():
                record=json.loads(line)
                (committed if record['update']<=update else tail).append(record)
            if tail:
                stamp=int(time.time());archive=out/f'UNCOMMITTED_RECOVERY_rank{rank}_{stamp}.jsonl'
                archive.write_text(''.join(json.dumps(r)+'\n' for r in tail))
                log.write_text(''.join(json.dumps(r)+'\n' for r in committed))
    ddp=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    pool,gate=load_pool()
    dataset=Positions(pool,cfg['seed'],cursor,10**10,rank,world,args.microbatch)
    generator=torch.Generator().manual_seed(derive(cfg['seed'],'loader',rank))
    loader=DataLoader(dataset,batch_size=args.microbatch,num_workers=2,prefetch_factor=2,
        persistent_workers=True,multiprocessing_context='spawn',generator=generator,shuffle=False)
    saved_rng=rng_capture(device);iterator=iter(loader);rng_restore(saved_rng,device)
    if rank==0:write(out/'CONFIG.json',cfg);write(out/'STRICT_LOAD.json',load_report)
    dist.barrier();torch.npu.reset_peak_memory_stats();start=time.time();times=[];start_update=update
    candidates=[]
    if (out/'FULL_CANDIDATES.json').exists():candidates=json.loads((out/'FULL_CANDIDATES.json').read_text())
    if args.mode=='formal' and not args.resume:
        checkpoint(out,0,0,objective,optimizer,ema,device,rank,world,cfg,counts)
        baseline=evaluate(ema.ema_model,device,out/'validation/U00000000_full',rank,world,True,0)
        candidates.append({'update':0,'validation':baseline})
        if rank==0:write(out/'FULL_CANDIDATES.json',candidates)
    stop_reason='step_budget'
    while True:
        if args.mode!='formal' and update-start_update>=args.steps:break
        if args.mode=='formal' and shared_decision(time.time()>=cfg['deadline_unix'] or (ROOT/'STOP_REQUEST.json').exists(),device,rank):
            stop_reason='deadline' if time.time()>=cfg['deadline_unix'] else 'stop_request';break
        started_unix=time.time();begin=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=device)
        local_counts=torch.zeros(4,device=device,dtype=torch.int64);positions=[];indices=[];tau_min=100;tau_max=0
        calls=[0];hook=net.register_forward_hook(lambda m,a,v:calls.__setitem__(0,calls[0]+1))
        for micro in range(args.accumulation):
            y,c,index,position,g=next(iterator);y=y.to(device);c=c.to(device)
            positions+=position.tolist();indices+=index.tolist()
            for value in g.tolist():local_counts[int(value)-1]+=1
            with ddp.no_sync() if micro+1<args.accumulation else contextlib.nullcontext():
                loss,tau=ddp(y,c);assert bool(torch.isfinite(loss))
                (loss/args.accumulation).backward();loss_sum+=loss.detach()/args.accumulation
            tau_min=min(tau_min,int(tau.min()));tau_max=max(tau_max,int(tau.max()))
            del y,c,loss,tau
        hook.remove();assert calls[0]==args.accumulation and 1<=tau_min<=tau_max<=99
        gradients=[p.grad for p in net.parameters()];assert all(p is not None for p in gradients)
        norm=torch.nn.utils.get_total_norm(gradients,error_if_nonfinite=True)
        assert bool(norm>0)
        optimizer.step();ema.update();update+=1
        dist.all_reduce(loss_sum);loss_sum/=world;dist.all_reduce(local_counts)
        for g,n in enumerate(local_counts.cpu().tolist(),1):counts[str(g)]+=n
        expected=[cursor+micro*world*args.microbatch+rank*args.microbatch+k for micro in range(args.accumulation) for k in range(args.microbatch)]
        assert positions==expected;cursor+=cfg['effective_batch']
        torch.npu.synchronize();elapsed=time.perf_counter()-begin;times.append(elapsed)
        free,total=torch.npu.mem_get_info();assert free>=8*(1<<30),('NPU physical reserve',rank,free)
        record={'update':update,'cursor':cursor,'loss':float(loss_sum),'grad_norm':float(norm),
            'started_unix':started_unix,'completed_unix':time.time(),
            'seconds':elapsed,'rank':rank,'positions':positions,'indices':indices,'tau_min':tau_min,'tau_max':tau_max,
            'nfe_per_sample':1,'lr':cfg['lr'],'counts_by_g':dict(counts),'free':free,
            'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved()}
        append(out/f'UPDATES_rank{rank}.jsonl',record)
        if rank==0 and (update<=20 or update%10==0):
            write(out/'STATUS.json',{'status':'TRAINING' if args.mode=='formal' else args.mode.upper(),
                'update':update,'presentations':cursor,'formal_updates':update if args.mode=='formal' else 0,
                'seconds_per_update_recent':statistics.mean(times[-50:]),'counts_by_g':dict(counts),
                'deadline_unix':cfg['deadline_unix'],'last_unix':time.time()})
            print(args.mode,update,float(loss_sum),elapsed,flush=True)
        if args.mode=='formal' and update%500==0:
            checkpoint(out,update,cursor,objective,optimizer,ema,device,rank,world,cfg,counts)
        if args.mode=='formal' and update%1000==0 and shared_decision(time.time()<cfg['deadline_unix'],device,rank):
            full=update%10000==0
            summary=evaluate(ema.ema_model,device,out/'validation'/f'U{update:08d}_{"full" if full else "monitor"}',rank,world,full,update)
            if full:
                candidates.append({'update':update,'validation':summary})
                if rank==0:write(out/'FULL_CANDIDATES.json',candidates)
    if rank==0 and args.mode=='formal':
        write(out/'TRAINING_STOPPED.json',{'update':update,'cursor':cursor,'stop_reason':stop_reason,
            'detected_unix':time.time(),'deadline_unix':cfg['deadline_unix'],'collective_rank0_stop_decision':True})
    checkpoint(out,update,cursor,objective,optimizer,ema,device,rank,world,cfg,counts)
    if args.full_eval and args.mode!='formal':
        evaluate(ema.ema_model,device,out/'pilot_monitor',rank,world,False,update)
    if args.mode=='formal':
        path=out/'validation'/f'U{update:08d}_full'
        if (path/'SUMMARY.json').exists():summary=json.loads((path/'SUMMARY.json').read_text())
        else:summary=evaluate(ema.ema_model,device,path,rank,world,True,update)
        if not any(c['update']==update for c in candidates):candidates.append({'update':update,'validation':summary})
        if rank==0:
            best=max(candidates,key=lambda c:(c['validation']['by_g']['4']['psnr'],-c['validation']['by_g']['4']['lpips'],-c['update']))
            write(out/'FULL_CANDIDATES.json',candidates);write(out/'BEST.json',best)
    digests=[None]*world;dist.all_gather_object(digests,parameter_digest(net));assert len(set(digests))==1
    result={'status':'PASS','mode':args.mode,'completed_updates':update-start_update,'global_update':update,
        'formal_updates':update if args.mode=='formal' else 0,'presentations':cursor,'config':cfg,
        'mean_seconds':statistics.mean(times[3:] or times) if times else 0.,'timings':times,
        'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved(),
        'rank_parameters_equal':True,'stop_reason':stop_reason,'seconds':time.time()-start,'counts_by_g':dict(counts)}
    write(out/f'RESULT_rank{rank}.json',result);dist.barrier()
    if rank==0:
        allresults=[json.loads((out/f'RESULT_rank{r}.json').read_text()) for r in range(world)]
        if times:
            critical=[max(r['timings'][i] for r in allresults) for i in range(len(times))]
            result['critical_mean_seconds']=statistics.mean(critical[3:] or critical)
            result['images_per_second']=cfg['effective_batch']/result['critical_mean_seconds']
        result['peak_allocated']=max(r['peak_allocated'] for r in allresults)
        result['peak_reserved']=max(r['peak_reserved'] for r in allresults)
        write(out/'RESULT.json',result)
        if args.mode=='formal':write(out/'TRAINING_AND_VALIDATION_COMPLETE.json',{'status':'PASS','update':update,'stop_reason':stop_reason,'final_val':str(path/'SUMMARY.json')})
    del iterator,loader;dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

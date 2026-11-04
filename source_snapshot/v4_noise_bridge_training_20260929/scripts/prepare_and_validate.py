"""Bounded two-GPU acceptance + throughput probes, then freeze one shared protocol."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import torch.distributed as dist
from train_comparison import setup,build,save_checkpoint,load_checkpoint
from v4.model import NoiseBridgeUNet
from v4.normalization import FrozenRGBStats
from v4.runtime import verify_imports,save_json,sha256
from v4.refinement import module_hash
from v4.training import (RecordStore,epoch_stream,evaluation_groups,make_optimizer,make_ema,
    backward_batch,finite_gradients,update_ema,learning_rate,rng_state,set_rng_state,atomic_torch_save)
from v4.evaluation import predict,summarize,psnr
from v4.core import mse


GRAD_RELATIVE_LIMIT=1e-4
FORWARD_ATOL=2e-5
FORWARD_RTOL=2e-4


def assert_equal_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal_tree(x,y)
    else:assert a==b


def gradient_diff(reference,model):
    error=sum(float((p.grad.double()-reference[name].double()).square().sum())
              for name,p in model.named_parameters())
    norm=sum(float(v.double().square().sum()) for v in reference.values())
    relative=(error/max(norm,1e-300))**.5
    if relative>GRAD_RELATIVE_LIMIT:
        raise AssertionError(f'Gradient parity relative L2 {relative} > {GRAD_RELATIVE_LIMIT}')
    return relative


def checkpoint_test(model,sim,objective,wrapped,store,rows,optimizer,ema,noise,device,group):
    temporary=ROOT/'reports'/f'temporary_{group}.pt'
    p={'protocol_hash':'TEMPORARY_ACCEPTANCE','batch':8,'normalization_artifact_sha256':'test',
       'stream_sha256':'test'}
    original_hash=module_hash(model);sim_hash=module_hash(sim)
    optimizer.zero_grad(set_to_none=True)
    backward_batch(wrapped,store,rows,1,noise)
    finite_gradients(model);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for pg in optimizer.param_groups:pg['lr']=learning_rate(8)
    optimizer.step();update_ema(ema,model,8)
    assert module_hash(model)!=original_hash and module_hash(sim)==sim_hash
    saved_hash=save_checkpoint(temporary,model=model,ema=ema,optimizer=optimizer,noise=noise,
        protocol=p,group=group,E=8,u=1)
    payload=torch.load(temporary,map_location='cpu',weights_only=False)
    # Compute the next update uninterrupted, then restore and repeat it.
    optimizer.zero_grad(set_to_none=True);backward_batch(wrapped,store,rows,1,noise)
    finite_gradients(model)
    reference={name:v.grad.detach().cpu().clone() for name,v in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for pg in optimizer.param_groups:pg['lr']=learning_rate(16)
    optimizer.step();update_ema(ema,model,8)
    next_params={name:v.detach().cpu().clone() for name,v in model.named_parameters()}
    end_rng=noise.get_state().clone()
    E,u=load_checkpoint(temporary,model,ema,optimizer,noise,p,group)
    assert (E,u)==(8,1)
    assert_equal_tree(model.state_dict(),payload['model']);assert_equal_tree(ema.state_dict(),payload['ema'])
    assert_equal_tree(optimizer.state_dict(),payload['optimizer'])
    optimizer.zero_grad(set_to_none=True);backward_batch(wrapped,store,rows,1,noise)
    finite_gradients(model)
    relative=gradient_diff({name:v.to(device) for name,v in reference.items()},model)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for pg in optimizer.param_groups:pg['lr']=learning_rate(16)
    optimizer.step();update_ema(ema,model,8)
    max_error=0.
    for name,param in model.named_parameters():
        expected=next_params[name].to(device)
        max_error=max(max_error,float((param-expected).abs().max()))
        if not torch.allclose(param,expected,atol=FORWARD_ATOL,rtol=FORWARD_RTOL):
            raise AssertionError(f'Resumed next update differs: {name}')
    assert torch.equal(noise.get_state(),end_rng)
    assert module_hash(sim)==sim_hash
    dist.barrier()
    if dist.get_rank()==0:temporary.unlink();temporary.with_suffix('.identity.json').unlink()
    return {'checkpoint_state_exact':True,'next_update_gradient_relative_L2':relative,
            'next_update_parameter_max_abs':max_error,'next_eta_rng_exact':True,
            'temporary_optimizer_updates':3,'simulator_unchanged':True,'checkpoint_sha256':saved_hash}


def bptt_check(objective,store,row,device):
    C,Y,G=store.batch([row],device)
    eta=torch.randn(Y.shape,device=device,generator=torch.Generator(device=device).manual_seed(3344))
    answers=[]
    for cut in (False,True):
        objective.zero_grad(set_to_none=True);count=[0]
        def detach_second(module,inputs):
            count[0]+=1
            return (inputs[0].detach(),inputs[1]) if count[0]==2 else None
        hook=objective.network.register_forward_pre_hook(detach_second) if cut else None
        try:result=objective(C,Y,G,eta=eta,retain_intermediates=True)
        finally:
            if hook:hook.remove()
        mse(result['final_clean'],C).mean().backward()
        stages=[]
        for index,record in enumerate(result['trace'][:-1]):
            values={}
            for key in ('clean','epsilon_zernike','epsilon_tilt'):
                grad=record[key].grad
                values[key]={'none':grad is None,'finite':None if grad is None else bool(torch.isfinite(grad).all()),
                             'norm':None if grad is None else float(grad.double().norm())}
                if cut and index==0:assert grad is None
                else:assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
            stages.append(values)
        answers.append({'detach_negative_control':cut,'stages':stages})
        del result
    return answers


def validate_group(group,config,store,initial,device):
    model,sim,objective,wrapped=build(config,device,group,initial)
    noise=torch.Generator(device=device).manual_seed(91810000+dist.get_rank())
    gs=[1,2,3,4,4,3,2,1]
    rows=[store.train[i*16+g-1] for i,g in enumerate(gs)]
    eta=torch.randn((8,3,256,256),generator=torch.Generator().manual_seed(2929))
    objective.zero_grad(set_to_none=True)
    totals=backward_batch(wrapped,store,rows,1,noise,etas=eta)
    finite_gradients(model)
    ddp_grads={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    objective.zero_grad(set_to_none=True)
    calls=[]
    for i,row in enumerate(rows):
        C,Y,G=store.batch([row],device)
        result=objective(C,Y,G,eta=eta[i:i+1].to(device),retain_intermediates=True)
        assert result['prediction_count']==(1 if group=='D0' else row['G'])
        assert result['transition_count']==(0 if group=='D0' else row['G']-1)
        assert result['training_refinement_count']==0
        (result['loss_mean']/len(rows)).backward()
        calls.append({'G':row['G'],'network_calls':result['prediction_count'],'transitions':result['transition_count']})
        if group!='D0':
            for k,record in enumerate(result['trace']):
                assert float(record['g'])==row['G']-k
                assert torch.equal(record['Y'],Y) and torch.equal(record['eta'],eta[i:i+1].to(device))
                if group=='R0':assert torch.equal(record['Z'],record['state'])
        del result
    finite_gradients(model);relative=gradient_diff(ddp_grads,model)
    del ddp_grads
    assert totals[:,0].tolist()==[2.,2.,2.,2.]
    report={'group':group,'DDP_accumulation_vs_single_gradient_relative_L2':relative,'paths':calls}
    if group!='D0':
        report['terminal_only_BPTT']=bptt_check(objective,store,rows[3],device)
    # Fresh common init for the temporary parameter/EMA/checkpoint updates.
    model.load_state_dict(initial,strict=True)
    optimizer=make_optimizer(model);ema=make_ema(model)
    report['checkpoint']=checkpoint_test(model,sim,objective,wrapped,store,rows,optimizer,ema,noise,device,group)
    model.load_state_dict(initial,strict=True);model.eval()
    eval_rows=rows[:4];C,Y,G=store.batch(eval_rows,device)
    on=predict(model,sim,Y[:1],G[:1],eval_rows[:1],group,group!='D0')
    off=predict(model,sim,Y[:1],G[:1],eval_rows[:1],group,False)
    assert torch.equal(on['clean'],off['clean'])
    report['G1_refinement_on_off_exact']=True
    del on,off
    # Exercise the actual per-sample refinement code once per rank at G=2.
    if group=='R08':
        refined=predict(model,sim,Y[1:3],G[1:3],eval_rows[1:3],group,True)
        assert refined['refinement_sample_count']==3
        record=refined['refinements'][0]
        assert record.optimizer['execution']=='per_sample_serial' and len(record.candidates)==41
        report['refinement']={'candidates':len(record.candidates),'execution':record.optimizer['execution'],
                              'invariance':record.invariance,'optimizer':record.optimizer}
        del refined
    del wrapped,objective,model,sim,optimizer,ema,C,Y,G
    gc.collect();torch.cuda.empty_cache();dist.barrier()
    return report


def benchmark(config,store,initial,device):
    results=[];begin_all=time.monotonic()
    for batch,micro in [(16,1),(16,2),(32,2),(32,4)]:
        if time.monotonic()-begin_all>1500:break
        for group in ('R08','R0','D0'):
            model,sim,objective,wrapped=build(config,device,group,initial)
            optimizer=make_optimizer(model);noise=torch.Generator(device=device).manual_seed(4182+dist.get_rank())
            times=[];peaks=[]
            # Include worst-case G=4 in the memory envelope; formal mixed sample stream remains unchanged.
            worst=[store.train[i*4+3] for i in range(2*micro)]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            backward_batch(wrapped,store,worst,micro,noise);finite_gradients(model)
            worst_peak=torch.cuda.max_memory_allocated()/2**30
            optimizer.zero_grad(set_to_none=True)
            rows=[store.train[i*4+i%4] for i in range(batch)]
            for iteration in range(3):
                torch.cuda.reset_peak_memory_stats();dist.barrier();torch.cuda.synchronize();begin=time.monotonic()
                optimizer.zero_grad(set_to_none=True);backward_batch(wrapped,store,rows,micro,noise)
                finite_gradients(model);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                seconds=torch.tensor(time.monotonic()-begin,device=device);dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
                peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
                if iteration:times.append(float(seconds));peaks.append(float(peak))
            item={'batch':batch,'microbatch':micro,'accumulation':batch//(2*micro),'group':group,
                  'measured_update_seconds':times,'starts_per_second':batch/(sum(times)/len(times)),
                  'peak_GiB':max(peaks),'worst_G4_peak_GiB':worst_peak,'temporary_optimizer_updates':3,
                  'precision':'FP32_autocast_and_TF32_off'}
            results.append(item)
            if dist.get_rank()==0:
                save_json(ROOT/'reports/benchmark_progress.json',{'results':results})
                print(json.dumps(item),flush=True)
            del wrapped,objective,model,sim,optimizer
            gc.collect();torch.cuda.empty_cache();dist.barrier()
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['validation','benchmark','freeze'],required=True)
    args=parser.parse_args();rank,device=setup()
    config=json.loads((ROOT/'configs/correctness.json').read_text());imports=verify_imports(ROOT,config)
    store=RecordStore(config)
    initial_path=ROOT/'protocol/common_init.pt'
    if not initial_path.exists():
        if rank==0:
            torch.manual_seed(20260916)
            model=NoiseBridgeUNet(FrozenRGBStats.from_artifact(ROOT/config['normalization_artifact']),**config['network'])
            atomic_torch_save(model.state_dict(),initial_path);del model
            save_json(ROOT/'protocol/stream.json',epoch_stream())
            save_json(ROOT/'protocol/data_and_evaluation_lists.json',store.manifest())
        dist.barrier()
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    if args.phase=='validation':
        reports=[]
        for group in ('D0','R0','R08'):
            reports.append(validate_group(group,config,store,initial,device))
            save_json(ROOT/f'reports/validation_rank{rank}.json',{'status':'IN_PROGRESS','groups':reports})
        save_json(ROOT/f'reports/validation_rank{rank}.json',{'status':'PASSED','groups':reports,
            'gradient_relative_limit':GRAD_RELATIVE_LIMIT,'forward_atol':FORWARD_ATOL,'forward_rtol':FORWARD_RTOL,
            'temporary_optimizer_updates_per_rank':9,'imports':imports,'gpu':torch.cuda.get_device_name(device)})
        dist.barrier()
        if rank==0:
            report={'status':'PASSED','world_size':2,'precision':'FP32_autocast_and_TF32_off',
                    'ranks':[json.loads((ROOT/f'reports/validation_rank{i}.json').read_text()) for i in range(2)]}
            save_json(ROOT/'reports/training_validation.json',report)
    elif args.phase=='benchmark':
        result=benchmark(config,store,initial,device)
        if rank==0:save_json(ROOT/'reports/benchmark.json',{'status':'COMPLETE','results':result})
    elif args.phase=='freeze':
        validation=json.loads((ROOT/'reports/training_validation.json').read_text())
        if validation['status']!='PASSED':raise ValueError('Validation not passed')
        evaluator=json.loads((ROOT/'reports/evaluator_validation.json').read_text())
        if evaluator['status']!='PASSED':raise ValueError('Evaluation path not passed')
        benchmark_result=json.loads((ROOT/'reports/benchmark.json').read_text())['results']
        candidates={}
        for row in benchmark_result:candidates.setdefault((row['batch'],row['microbatch']),[]).append(row)
        candidates={key:rows for key,rows in candidates.items() if {r['group'] for r in rows}=={'D0','R0','R08'}}
        costs={key:sum(32000/r['starts_per_second'] for r in rows) for key,rows in candidates.items()}
        minimum=min(costs.values())
        # Within 10% of measured best total training time, prefer smaller batch and then faster microbatch.
        chosen=min((key for key in costs if costs[key]<=minimum*1.10),key=lambda k:(k[0],costs[k]))
        batch,micro=chosen
        selected=json.loads((ROOT/'reports/selected_validation.json').read_text())
        if selected['status']!='PASSED' or (selected['batch'],selected['microbatch'])!=(batch,micro):
            raise ValueError('Final selected configuration is not validated')
        if 3200%batch or 32000%batch:raise ValueError('Frozen updates must exactly hit all evaluation nodes')
        if rank==0:
            grouping={name:[{'rank':b['rank'],'start_ids':[r['start_id'] for r in b['records']]}
                       for b in evaluation_groups(rows,2,2)] for name,rows in store.lists.items()}
            save_json(ROOT/'protocol/evaluation_grouping.json',grouping)
            code={str(p.relative_to(ROOT)):sha256(p) for directory in ('v4','physical_bridge','scripts')
                  for p in sorted((ROOT/directory).glob('*.py'))}
            code.update({str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts').glob('*.sh')})
            for rel in ('configs/correctness.json','configs/base_source_identity.json','docs/TRAINING_PLAN.md','protocol/AUTHORIZATION.json'):
                code[rel]=sha256(ROOT/rel)
            protocol={'status':'FROZEN','comparison_id':'v4_overfit100_paired_20260929',
                'comparison_dir':'/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/runs/v4_overfit100_paired_20260929',
                'groups':['R08','R0','D0'],'world_size':2,'GPU_UUIDs':os.environ['CUDA_VISIBLE_DEVICES'].split(','),
                'batch':batch,'microbatch':micro,'accumulation':batch//(2*micro),'updates_per_group':32000//batch,
                'exposures_per_group':32000,'eval_network_batch':2,'refinement_execution':'per_sample_serial',
                'refine_batch':1,'precision':'FP32_autocast_and_TF32_off','tf32':False,'autocast':False,
                'refinement':{'updates':40,'candidates':41,'lr':.01,'lambda':.01,'betas':[.9,.999],'eps':1e-8,'weight_decay':0},
                'evaluation_grouping_sha256':sha256(ROOT/'protocol/evaluation_grouping.json'),
                'seed':20260916,'eta_seed_per_rank':[91810000,91810001],
                'initialization_sha256':sha256(initial_path),'stream_sha256':sha256(ROOT/'protocol/stream.json'),
                'normalization_artifact_sha256':sha256(ROOT/config['normalization_artifact']),
                'authorization_sha256':sha256(ROOT/'protocol/AUTHORIZATION.json'),
                'list_hashes':store.manifest()['list_hashes'],'data_manifest_hashes':store.dataset.manifest_hashes,
                'code_sha256':code,'validation_report':'reports/training_validation.json',
                'validation_sha256':sha256(ROOT/'reports/training_validation.json'),
                'selected_validation_sha256':sha256(ROOT/'reports/selected_validation.json'),
                'evaluator_validation_sha256':sha256(ROOT/'reports/evaluator_validation.json'),
                'benchmark_sha256':sha256(ROOT/'reports/benchmark.json'),'measured_training_seconds_by_candidate':
                    {str(key):value for key,value in costs.items()},
                'batch_selection':'minimum measured three-group time; within 10% prefer smaller B',
                'software':{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
                'tail_policy':'all selected batches and nodes divisible; reject padding/drop/unequal shards',
                'evaluation_grouping':'global fixed chunks of 2; assign chunk_index modulo world_size',
                'evaluation_nodes':[0]+list(range(3200,32001,3200)),
                'optimizer':{'name':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4,
                             'no_decay':'bias and normalization','clip_global_norm':1.},
                'scheduler':'exposure_warmup_1600_cosine_to_32000','EMA':'0.996**(B_u/32)',
                'formal_training_authorized':True,'training_eta':'fresh per occurrence and sample; shared within path',
                'eval_eta':'SHA256 eval:20260916:start_id modulo 2**63-1',
                'determinism':'CUDA grid_sample backward remains nondeterministic; no bitwise claim'}
            import hashlib
            protocol['protocol_hash']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            save_json(ROOT/'protocol/FROZEN.json',protocol)
            save_json(ROOT/'protocol/evaluation_grouping.json',
                {name:[{'rank':b['rank'],'start_ids':[r['start_id'] for r in b['records']]}
                       for b in evaluation_groups(rows,2,2)] for name,rows in store.lists.items()})
            print(json.dumps(protocol,indent=2),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        save_json(ROOT/f'reports/preparation_failure_rank{os.environ.get("LOCAL_RANK","unknown")}.json',
                  {'status':'FAILED','traceback':traceback.format_exc()})
        raise

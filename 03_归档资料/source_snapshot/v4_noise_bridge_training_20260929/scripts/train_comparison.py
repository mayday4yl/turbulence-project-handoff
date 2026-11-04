"""Two-GPU, three-run formal comparison. Explicit protocol and validation gates required."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from physical_bridge.simulator import PiRNP2SSimulator
from v4.model import NoiseBridgeUNet
from v4.normalization import FrozenRGBStats
from v4.runtime import verify_imports,sha256,save_json
from v4.refinement import module_hash
from v4.training import (RecordStore,TrainingObjective,make_optimizer,make_ema,update_ema,
    learning_rate,backward_batch,finite_gradients,rng_state,set_rng_state,atomic_torch_save)
from v4.evaluation import evaluate

STOP=False
CONTEXT={'attempt_id':uuid.uuid4().hex}
def stop_after_update(signum,frame):
    global STOP
    STOP=True


def setup():
    if socket.gethostname()!='h3c-gpu-server01' or os.getuid()!=1036:
        raise RuntimeError('Wrong host/account')
    if int(os.environ.get('WORLD_SIZE','0'))!=2 or torch.cuda.device_count()!=2:
        raise RuntimeError('Exactly two explicitly visible GPUs required')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(20260916+rank);np.random.seed(20260916+rank);torch.manual_seed(20260916+rank)
    torch.cuda.manual_seed(20260916+rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    return rank,torch.device('cuda',rank)


def build(config,device,group,initial):
    model=NoiseBridgeUNet(FrozenRGBStats.from_artifact(ROOT/config['normalization_artifact']),**config['network']).to(device)
    model.load_state_dict(initial,strict=True)
    simulator=PiRNP2SSimulator(config['asset_dir'],**config['physics'],device=device).to(device).eval()
    if any(p.requires_grad for p in simulator.parameters()):raise RuntimeError('Simulator must be frozen')
    objective=TrainingObjective(model,simulator,group).train()
    wrapped=DDP(objective,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=False)
    return model,simulator,objective,wrapped


def save_checkpoint(path,*,model,ema,optimizer,noise,protocol,group,E,u):
    states=[None]*dist.get_world_size();dist.all_gather_object(states,rng_state(noise))
    identity=[None]
    if dist.get_rank()==0:
        payload={'purpose':'FORMAL_V4_COMPARISON','protocol_hash':protocol['protocol_hash'],
            'group':group,'E':E,'u':u,'model':model.state_dict(),'ema':ema.state_dict(),
            'optimizer':optimizer.state_dict(),'rng_per_rank':states,'sampler_position':E,
            'scheduler':{'kind':'exposure_warmup_cosine','next_update_lr':learning_rate(min(E+protocol['batch'],32000))},
            'normalization_artifact_sha256':protocol['normalization_artifact_sha256'],
            'stream_sha256':protocol['stream_sha256'],'protocol':protocol,
            'model_hash':module_hash(model),'ema_hash':module_hash(ema)}
        atomic_torch_save(payload,path)
        identity[0]=sha256(path)
        save_json(Path(path).with_suffix('.identity.json'),{'sha256':identity[0],'E':E,'u':u,
                  'protocol_hash':protocol['protocol_hash'],'model_hash':payload['model_hash'],
                  'ema_hash':payload['ema_hash']})
    dist.broadcast_object_list(identity,0);dist.barrier()
    return identity[0]


def load_checkpoint(path,model,ema,optimizer,noise,protocol,group):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['purpose']!='FORMAL_V4_COMPARISON' or payload['protocol_hash']!=protocol['protocol_hash'] or payload['group']!=group:
        raise ValueError('Resume identity mismatch')
    if payload['sampler_position']!=payload['E'] or payload['stream_sha256']!=protocol['stream_sha256']:
        raise ValueError('Resume sampler mismatch')
    model.load_state_dict(payload['model'],strict=True);ema.load_state_dict(payload['ema'],strict=True)
    optimizer.load_state_dict(payload['optimizer']);set_rng_state(payload['rng_per_rank'][dist.get_rank()],noise)
    if module_hash(model)!=payload['model_hash'] or module_hash(ema)!=payload['ema_hash']:
        raise ValueError('Checkpoint state roundtrip differs')
    return payload['E'],payload['u']


def run_evaluations(model,sim,ema,store,noise,protocol,group,E,run_dir,checkpoint_identity):
    lists=['train64','val64']
    if E in (0,16000,32000):lists+=['val800']
    if E in (16000,32000):lists+=['train1600']
    best_path=Path(run_dir)/'best_val.json'
    for weight,network in [('raw',model),('ema',ema)]:
        for name in lists:
            summary=evaluate(network,sim,store,store.lists[name],group=group,weights=weight,refinement=False,
                protocol=protocol,E=E,run_dir=run_dir,list_name=name,noise_generator=noise,
                checkpoint_identity=checkpoint_identity)
            if dist.get_rank()==0 and name=='val800' and weight=='raw':
                score=summary['overall']['psnr']
                previous=json.loads(best_path.read_text()) if best_path.exists() else None
                if previous is None or score>previous['psnr']:
                    # All complete validation nodes also retain their full checkpoint.
                    save_json(best_path,{'E':E,'psnr':score,'checkpoint':f'checkpoints/E{E:05d}.pt',
                              'checkpoint_identity':checkpoint_identity,'tie_break':'earlier exposure'})
        dist.barrier()


def train_group(group,config,protocol,store,stream,initial,device,comparison_dir):
    rank=dist.get_rank();run_dir=Path(comparison_dir)/group;run_dir.mkdir(parents=True,exist_ok=True)
    completed=run_dir/'TRAINING_OFF_COMPLETE.json'
    if completed.exists():
        if json.loads(completed.read_text())['protocol_hash']!=protocol['protocol_hash']:
            raise ValueError('Completed group protocol mismatch')
        return
    model,sim,objective,wrapped=build(config,device,group,initial)
    optimizer=make_optimizer(model);ema=make_ema(model)
    noise=torch.Generator(device=device).manual_seed(91810000+rank)
    latest=run_dir/'LATEST.json';E=u=0
    if latest.exists():
        checkpoint=run_dir/json.loads(latest.read_text())['checkpoint']
        E,u=load_checkpoint(checkpoint,model,ema,optimizer,noise,protocol,group)
        checkpoint_identity=sha256(checkpoint)
        if rank==0:
            with (run_dir/'resume_events.jsonl').open('a') as f:
                f.write(json.dumps({'attempt_id':CONTEXT['attempt_id'],'resume_E':E,'resume_u':u,
                                    'checkpoint_identity':checkpoint_identity})+'\n')
    else:
        checkpoint=run_dir/'checkpoints/E00000.pt'
        checkpoint_identity=save_checkpoint(checkpoint,model=model,ema=ema,optimizer=optimizer,
            noise=noise,protocol=protocol,group=group,E=0,u=0)
        if rank==0:save_json(latest,{'checkpoint':str(checkpoint.relative_to(run_dir)),'E':E,'u':u})
    # Complete/skipped saved-node evaluations on resume before advancing any sampler state.
    if E%3200==0:
        run_evaluations(model,sim,ema,store,noise,protocol,group,E,run_dir,checkpoint_identity)
    while E<32000:
        begin=time.monotonic();batch=protocol['batch'];next_E=E+batch
        CONTEXT.update(group=group,E_before=E,next_E=next_E,u_next=u+1,run_dir=str(run_dir))
        if next_E>32000:raise ValueError('Frozen divisor batch unexpectedly has a tail')
        records=[store.train[index] for index in stream[E:next_E]]
        if any(r['split']!='train' for r in records):raise ValueError('Nontrain update record')
        objective.train();optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(next_E)
        for pg in optimizer.param_groups:pg['lr']=lr
        torch.cuda.reset_peak_memory_stats()
        totals=backward_batch(wrapped,store,records,protocol['microbatch'],noise)
        finite_gradients(model)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite updated model; do not commit exposure')
        decay=update_ema(ema,model,batch)
        E=next_E;u+=1
        torch.cuda.synchronize();elapsed=time.monotonic()-begin
        memory=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        if rank==0:
            row={'u':u,'E':E,'epochs':E/1600,'batch':batch,'LR':lr,'EMA_decay':decay,
                 'attempt_id':CONTEXT['attempt_id'],
                 'gradient_norm_before_clip':float(norm),'seconds':elapsed,'starts_per_second':batch/elapsed,
                 'peak_GiB_per_card':float(memory),'precision':protocol['precision'],
                 'loss':float(totals[:,3].sum()/batch),'clean':float(totals[:,1].sum()/batch),
                 'reconstruction':float(totals[:,2].sum()/batch),'by_G':totals.cpu().tolist(),
                 'record_indices':stream[E-batch:E], 'protocol_hash':protocol['protocol_hash']}
            with (run_dir/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            save_json(Path(comparison_dir)/'STATUS.json',{'state':'TRAINING','group':group,'E':E,'u':u,
                      'last_update':row,'exposures_target':32000,'ordered_groups':['R08','R0','D0']})
            if u%10==0:print(json.dumps({'group':group,'E':E,'u':u,'loss':row['loss'],'seconds':elapsed}),flush=True)
        stop=torch.tensor(int(STOP),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if E%3200==0 or bool(stop):
            checkpoint=run_dir/f'checkpoints/E{E:05d}.pt'
            checkpoint_identity=save_checkpoint(checkpoint,model=model,ema=ema,optimizer=optimizer,noise=noise,
                protocol=protocol,group=group,E=E,u=u)
            if rank==0:save_json(latest,{'checkpoint':str(checkpoint.relative_to(run_dir)),'E':E,'u':u})
            if bool(stop):
                raise InterruptedError('Requested stop; committed full checkpoint after update')
            if rank==0:save_json(Path(comparison_dir)/'STATUS.json',{'state':'EVALUATING','group':group,'E':E,'u':u})
            run_evaluations(model,sim,ema,store,noise,protocol,group,E,run_dir,checkpoint_identity)
    if rank==0:
        save_json(run_dir/'last_32k.json',{'checkpoint':str(checkpoint.relative_to(run_dir)),
                  'sha256':checkpoint_identity,'E':E,'u':u})
        save_json(completed,{'status':'COMPLETE','protocol_hash':protocol['protocol_hash'],'E':E,'u':u,
                            'last_checkpoint_sha256':checkpoint_identity})
    dist.barrier()
    del wrapped,objective,model,sim,ema,optimizer
    import gc;gc.collect();torch.cuda.empty_cache()


def finish_refinement(group,config,protocol,store,initial,device):
    run_dir=Path(protocol['comparison_dir'])/group
    finished=json.loads((run_dir/'TRAINING_OFF_COMPLETE.json').read_text())
    if finished['protocol_hash']!=protocol['protocol_hash'] or finished['E']!=32000:
        raise ValueError('Final refinement requires complete matched training and off evaluation')
    if group!='D0':
        model,sim,objective,wrapped=build(config,device,group,initial)
        checkpoint=run_dir/json.loads((run_dir/'last_32k.json').read_text())['checkpoint']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True);model.eval()
        noise=torch.Generator(device=device).manual_seed(91810000+dist.get_rank())
        if dist.get_rank()==0:save_json(Path(protocol['comparison_dir'])/'STATUS.json',
            {'state':'FINAL_PAIRED_REFINEMENT','group':group,'E':32000,'refinement_execution':'per_sample_serial'})
        evaluate(model,sim,store,store.val,group=group,weights='raw',refinement=True,protocol=protocol,E=32000,
                 run_dir=run_dir,list_name='val800',noise_generator=noise,checkpoint_identity=sha256(checkpoint))
        dist.barrier()
        del wrapped,objective,model,sim,payload
        import gc;gc.collect();torch.cuda.empty_cache()
    if dist.get_rank()==0:save_json(run_dir/'COMPLETE.json',finished | {'paired_evaluation_complete':True})
    dist.barrier()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    args=parser.parse_args();rank,device=setup()
    protocol=json.loads(Path(args.protocol).read_text())
    if protocol['status']!='FROZEN' or protocol['refinement_execution']!='per_sample_serial':
        raise ValueError('No frozen supported protocol')
    config=json.loads((ROOT/'configs/correctness.json').read_text());verify_imports(ROOT,config)
    validation=json.loads((ROOT/protocol['validation_report']).read_text())
    if validation['status']!='PASSED':raise ValueError('Formal training requires passed validation')
    if sha256(ROOT/protocol['validation_report'])!=protocol['validation_sha256']:
        raise ValueError('Validation report changed after protocol freeze')
    if sha256(ROOT/config['normalization_artifact'])!=protocol['normalization_artifact_sha256']:
        raise ValueError('Normalization artifact changed after protocol freeze')
    if sha256(ROOT/'protocol/evaluation_grouping.json')!=protocol['evaluation_grouping_sha256']:
        raise ValueError('Frozen evaluation grouping changed')
    for rel,expected in protocol['code_sha256'].items():
        if sha256(ROOT/rel)!=expected:raise ValueError(f'Frozen source changed: {rel}')
    if sha256(ROOT/'protocol/common_init.pt')!=protocol['initialization_sha256']:
        raise ValueError('Common initialization mismatch')
    store=RecordStore(config)
    if store.manifest()['list_hashes']!=protocol['list_hashes']:raise ValueError('Evaluation/training population changed')
    stream=json.loads((ROOT/'protocol/stream.json').read_text())
    if sha256(ROOT/'protocol/stream.json')!=protocol['stream_sha256']:raise ValueError('Sampler stream changed')
    initial=torch.load(ROOT/'protocol/common_init.pt',map_location='cpu',weights_only=True)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_after_update)
    for group in ('R08','R0','D0'):
        train_group(group,config,protocol,store,stream,initial,device,protocol['comparison_dir'])
    # Finish all primary comparisons before the separately reported inference-only refinement checks.
    for group in ('R08','R0','D0'):
        finish_refinement(group,config,protocol,store,initial,device)
    if rank==0:save_json(Path(protocol['comparison_dir'])/'STATUS.json',{'state':'TRAINING_AND_EVALUATION_COMPLETE',
                'groups':['R08','R0','D0'],'E_per_group':32000,'protocol_hash':protocol['protocol_hash']})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print(traceback.format_exc(),flush=True)
        # No automatic protocol changes, skipped batches or extra updates on failure.
        error=ROOT/'reports'/f'formal_failure_rank{os.environ.get("LOCAL_RANK","unknown")}.json'
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'context':CONTEXT}
        save_json(error,failure)
        if 'run_dir' in CONTEXT:
            with (Path(CONTEXT['run_dir'])/f'failed_attempts_rank{os.environ.get("LOCAL_RANK","unknown")}.jsonl').open('a') as stream:
                stream.write(json.dumps(failure)+'\n')
        raise

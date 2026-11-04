"""One method/device, independent recovery, no report or cross-method barrier."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from execution.common import *
from execution.data import Store,Stream
from execution.checkpoints import save_full,load_latest,save_eval_weights
from execution.engine import step
from execution.recovery import reconcile
from npu_runtime.backend import make_scaler,synchronize,empty_cache
from execution.devices import setup
from restormer_v4.factory import build,build_optimizer
from v4.training import make_ema
STOP=False

def stop(*_):
    global STOP
    STOP=True


def binding_for(config,store,stream):
    return {'experiment':ROOT.name,'run_id':'formal_20k','scientific_spec':digest(ROOT/'protocol/SCIENTIFIC_SPEC.json'),
            'dataset_id':config['dataset_id'],'dataset_manifests':store.dataset.manifest_hashes,
            'normalization':digest(ROOT/config['normalization_artifact']),
            'common_initialization':digest(ROOT/'protocol/common_init.pt'),'stream':stream.identity}


def verify_gate(group,binding,config):
    backend=config.get('execution_backend','npu')
    prefix='TRAINING_ACCEPTANCE' if backend=='npu' else 'CUDA_TRAINING_ACCEPTANCE'
    path=ROOT/'reports'/f'{prefix}_{group}.json'
    gate=json.loads(path.read_text())
    if gate['status']!='PASS' or gate['binding']!=binding:
        raise RuntimeError('Formal launch requires matching actual training acceptance')
    auth=json.loads((ROOT/'protocol/AUTHORIZATION.json').read_text())
    if not auth['authorized'] or socket.gethostname() not in auth['formal_training_hosts']:
        raise RuntimeError('Missing formal training authorization for this host')
    if backend=='cuda' and not auth.get('A100_formal_training_authorized',False):
        raise RuntimeError('A100 is assets-only: explicit future CUDA launch authorization required')
    if backend=='npu' and socket.gethostname()!='liteserver-de36':raise RuntimeError('NPU training only on authorized de36')
    for rel,expected in gate['source_hashes'].items():
        if digest(ROOT/rel)!=expected:raise RuntimeError('Accepted executable changed: '+rel)
    return gate


def main(args):
    config=load_config(args.config);group=args.group;device_index=config.get('method_devices',GROUP_DEVICES)[group]
    run=Path(config['results_root'])/group;run.mkdir(parents=True,exist_ok=True)
    lease=(run/'worker.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=run/'STATUS.json';previous=json.loads(status_path.read_text()) if status_path.exists() else {}
    store=Store(config);stream=Stream(store);binding=binding_for(config,store,stream)
    gate=verify_gate(group,binding,config)
    runtime=json.loads((run/'RUNTIME.json').read_text()) if (run/'RUNTIME.json').exists() else gate['runtime']
    backend=setup(device_index,runtime['precision'],config.get('execution_backend','npu'));device=backend['device']
    config['precision']=runtime['precision'];config['network']['use_checkpoint']=runtime.get('activation_checkpoint',True)
    model,sim,obj=build(config,group,device,ROOT);opt=build_optimizer(model);ema=make_ema(model)
    scaler=make_scaler(device,runtime['precision']);u=load_latest(run,group,model,ema,opt,scaler,binding,stream)
    reconcile(run,u)
    if previous.get('update',0)>u:
        append_json(run/'recovery.jsonl',{'time':time.time(),'event':'volatile_updates_rolled_back',
            'previous_observed_update':previous['update'],'resume_committed_update':u,'exposures_recounted_from':u*32,
            'old_attempt_scalar_rows_above_resume_are_not_committed':True})
    attempt=f'{int(time.time())}_{os.getpid()}'
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if not (run/'LATEST.json').exists():save_full(run,group,model,ema,opt,scaler,u,binding,stream,runtime)
    atomic_json(run/'PROCESS.json',{'pid':os.getpid(),'attempt':attempt,'device':device,'host':socket.gethostname(),'start_time':time.time()})
    pending=[];step_phase={};safe_state=True;training_seconds=0.;eval_seconds=0.;recent=[]
    def evaluation_boundary():
        nonlocal eval_seconds
        if u%500==0:
            ckpt=save_eval_weights(run,group,model,u,binding)
            kinds=['monitor']
            if u%2000==0 or u in MILESTONES:kinds.append('fullval')
            for kind in kinds:
                job={'kind':kind,'method':group,'update':u,'checkpoint':str(ckpt),
                     'checkpoint_sha256':digest(ckpt),'run_dir':str(run),'config':str(Path(args.config).resolve())}
                job_path=run/'eval_queue'/f'{kind}_u{u:05d}.json';atomic_json(job_path,job)
                started=time.monotonic()
                try:
                    from evaluate_worker import run_job
                    run_job(job,config=config,store=store,objective=obj,device=device)
                    atomic_json(job_path.with_suffix('.done.json'),{'time':time.time(),'update':u})
                except Exception:
                    append_json(run/'evaluation_errors.jsonl',{'time':time.time(),'update':u,'kind':kind,'traceback':traceback.format_exc()})
                    # Retain pending immutable job; no method training shutdown on eval engineering error.
                eval_seconds+=time.monotonic()-started
            if u in MILESTONES:
                job={'kind':'milestone','method':group,'update':u,'checkpoint':str(ckpt),
                     'checkpoint_sha256':digest(ckpt),'run_dir':str(run),'config':str(Path(args.config).resolve())}
                atomic_json(run/'eval_queue'/f'milestone_u{u:05d}.json',job)
            empty_cache(device)
    try:
        if u and u%500==0:evaluation_boundary()
        while u<20000:
            control=json.loads((ROOT/'protocol/CONTROL.json').read_text())
            if STOP or control.get('action')=='stop':
                save_full(run,group,model,ema,opt,scaler,u,binding,stream,runtime)
                atomic_json(status_path,{'status':'STOPPED_AT_COMPLETE_UPDATE','update':u,'exposures':u*32,'time':time.time()});return
            model.train();obj.train();safe_state=False
            values=step(model,obj,opt,scaler,store,stream.rows(u),u+1,runtime['microbatch'],ema,step_phase)
            u+=1;safe_state=True;training_seconds+=values['update_seconds'];recent.append(values['update_seconds']);recent=recent[-100:]
            if u%20==0:
                append_json(run/'scalars.jsonl',{'update':u,'epoch':u/1000,'exposures':u*32,'attempt':attempt,
                   'time':time.time(),'microbatch':runtime['microbatch'],'precision':runtime['precision'],
                   'starts_per_second':32/values['update_seconds'],**values})
            atomic_json(status_path,{'status':'TRAINING','update':u,'epoch':u/1000,'exposures':u*32,'attempt':attempt,
                'last_update_seconds':values['update_seconds'],'mean_recent_update_seconds':sum(recent)/len(recent),
                'training_seconds_this_attempt':training_seconds,'evaluation_seconds_this_attempt':eval_seconds,
                'training_only_remaining_seconds':(20000-u)*sum(recent)/len(recent),'time':time.time(),'device':device,
                'microbatch':runtime['microbatch'],'precision':runtime['precision']})
            if u%1000==0:save_full(run,group,model,ema,opt,scaler,u,binding,stream,runtime)
            if u%500==0:evaluation_boundary()
        save_full(run,group,model,ema,opt,scaler,u,binding,stream,runtime)
        atomic_json(status_path,{'status':'TRAINING_COMPLETE_EVALUATION_PENDING','update':u,'exposures':640000,'epoch':20,'time':time.time()})
    except BaseException as exc:
        # Backward failures do not update parameters, but an exception during optimizer
        # mutation may. Never falsely commit a potentially partial optimizer state.
        safe_state=safe_state or not step_phase.get('optimizer_started',False)
        if safe_state:
            try:save_full(run,group,model,ema,opt,scaler,u,binding,stream,runtime)
            except Exception:append_json(run/'checkpoint_errors.jsonl',{'time':time.time(),'traceback':traceback.format_exc()})
        atomic_json(run/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'update':u,
            'attempt':attempt,'safe_emergency_saved':safe_state,'time':time.time(),
            'oom':any(term in str(exc).lower() for term in ('out of memory','memory allocation','malloc','507001','alloc memory failed','memoryerror')),'runtime':runtime})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=list(GROUP_DEVICES),required=True);p.add_argument('--config')
    main(p.parse_args())

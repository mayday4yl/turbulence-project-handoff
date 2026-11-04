"""Durable autonomous controller. No dependency on A100 mirror completion."""
import signal
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from execution.common import GROUP_DEVICES,atomic_json,append_json,load_config,digest

# First-use CANN compilation is outside steady-state training throughput. Allow
# calibration, two real B32 updates, disk resume and serial refinement to finish;
# completed methods still launch immediately and independently.
PREPARATION_TIMEOUT_SECONDS=3600


def launch(script,*args,label):
    log=(ROOT/'logs'/f'{label}.log').open('a')
    return subprocess.Popen([str(ROOT/'scripts/run_env.sh'),str(ROOT/'scripts'/script),*map(str,args)],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def state(group,cfg):
    path=Path(cfg['results_root'])/group/'STATUS.json'
    return json.loads(path.read_text()) if path.exists() else {}


def training_complete(group,cfg):
    run=Path(cfg['results_root'])/group
    if state(group,cfg).get('status')!='TRAINING_COMPLETE_EVALUATION_PENDING':return False
    identity=run/'checkpoints/full_U20000.identity.json'
    if not identity.exists():return False
    saved=json.loads(identity.read_text())
    return saved.get('update')==20000 and saved.get('exposures')==640000 and (run/saved['path']).exists()


def pid_alive(path):
    try:
        obj=json.loads(path.read_text());pid=int(obj['pid'])
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        return str(ROOT) in cmd and any(x in cmd for x in ('train_worker.py','evaluate_worker.py','prepare_method.py','accept_training.py'))
    except (OSError,ValueError,KeyError):return False


def preparation_stage(previous,stage_path,group):
    if pid_alive(stage_path):return json.loads(stage_path.read_text())
    # Also adopt a live stage from an older/restarted preparation wrapper; do not
    # stop useful child work simply because its parent predates stage journaling.
    try:
        parent=previous['pid']
        children=Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        boot=next(float(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
        for pid in children:
            proc=Path('/proc')/pid;args=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if str(ROOT/'scripts/accept_training.py') not in args or args[args.index('--group')+1]!=group:continue
            start_ticks=float(proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19])
            stage={'pid':int(pid),'time':boot+start_ticks/os.sysconf('SC_CLK_TCK'),
                'microbatch':int(args[args.index('--microbatch')+1]),'precision':args[args.index('--precision')+1],
                'calibrate':'--calibrate' in args,'adopted_from_live_child':True}
            atomic_json(stage_path,stage);return stage
    except (OSError,ValueError,KeyError,IndexError,StopIteration):pass
    return previous


def wait_ready(cfg):
    gate=Path(cfg['dataset_root'])/'DATA_READY.json'
    while True:
        ready=(gate.exists() and (ROOT/'reports/EXECUTION_CODE_READY.json').exists()
               and (ROOT/cfg['normalization_artifact']).exists())
        atomic_json(ROOT/'reports/WORKFLOW_STATUS.json',{'status':'READY_FOR_ACCEPTANCE' if ready else 'WAITING_DEPENDENCIES',
            'dataset_ready':gate.exists(),'execution_code_ready':(ROOT/'reports/EXECUTION_CODE_READY.json').exists(),
            'normalization_ready':(ROOT/cfg['normalization_artifact']).exists(),'A100_dependency':False,'time':time.time()})
        if ready:
            data=json.loads(gate.read_text());code=json.loads((ROOT/'reports/EXECUTION_CODE_READY.json').read_text())
            if (data.get('status')!='DATA_READY' or data.get('dataset_id')!=cfg['dataset_id'] or
                data.get('split_starts')!={'train':32000,'val':2000,'test':1600} or
                not data.get('all_array_hashes_shapes_dtype_finite_verified') or code.get('status')!='PASS'):
                raise ValueError('Readiness gate contents invalid')
            if digest(ROOT/cfg['normalization_artifact'])!=data['normalization_sha256']:
                raise ValueError('Canonical normalization bytes mismatch')
            for name,key in (('ASSET_LOCK.json','asset_lock_sha256'),('hash_manifest.json','hash_manifest_sha256')):
                if digest(Path(cfg['dataset_root'])/name)!=data[key]:raise ValueError('Frozen dataset '+name+' changed')
            for rel,expected in code['source_hashes'].items():
                if digest(ROOT/rel)!=expected:raise ValueError('Code readiness source changed: '+rel)
            return
        time.sleep(20)


def prepare(group):
    gate=ROOT/'reports'/f'TRAINING_ACCEPTANCE_{group}.json'
    if gate.exists() and json.loads(gate.read_text()).get('status')=='PASS':return True
    process=launch('prepare_method.py','--group',group,label=f'prepare_{group}')
    return process.wait()==0


def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    lease=(ROOT/'workflow.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(ROOT/'reports/CONTROLLER.json',{'pid':os.getpid(),'time':time.time(),'root':str(ROOT)})
    cfg=load_config();wait_ready(cfg)
    from restormer_v4.factory import initialize_common
    initialize_common(ROOT)
    # All four preparation processes start together. Each can enter formal training
    # independently after its gate; a hung or failed method cannot hold the others.
    accepted=[];results={g:'pending' for g in GROUP_DEVICES};prep_procs={};prep_attempts={g:0 for g in GROUP_DEVICES}
    procs={};failures={g:[] for g in GROUP_DEVICES};retry_after={g:0 for g in GROUP_DEVICES};eval_proc=None;eval_path=None;last_report=0
    while True:
        # Finalized kernel metadata is concrete cold-build progress. All methods
        # share these operator shapes/cache; retain the60min total ceiling below.
        kernel_progress=max((p.stat().st_mtime for p in (ROOT/'kernel_meta').rglob('*.json')),default=0)
        for group in GROUP_DEVICES:
            if group in accepted or results[group]=='failed3times':continue
            gate=ROOT/'reports'/f'TRAINING_ACCEPTANCE_{group}.json'
            prep_identity=ROOT/'reports'/f'PREPARATION_PROCESS_{group}.json'
            if gate.exists() and json.loads(gate.read_text()).get('status')=='PASS':
                accepted.append(group);results[group]='PASS'
                atomic_json(ROOT/'reports/FORMAL_LAUNCH.json',{'prepared':results,'accepted':accepted,'time':time.time(),
                    'A100_waited':False,'training_budget':20000})
                continue
            pp=prep_procs.get(group)
            alive=(pp is not None and pp.poll() is None) or pid_alive(prep_identity)
            previous=json.loads(prep_identity.read_text()) if prep_identity.exists() else {}
            stage_path=ROOT/'reports'/f'PREPARATION_STAGE_{group}.json'
            stage=preparation_stage(previous,stage_path,group)
            stage_start=stage.get('time',previous.get('time',time.time()))
            elapsed=time.time()-stage_start
            hint_path=ROOT/'reports'/f'PREPARATION_HINT_{group}.json'
            hint=json.loads(hint_path.read_text()) if hint_path.exists() else {'max_microbatch':16}
            completed_calibration=any(json.loads(p.read_text()).get('time',0)>previous.get('time',0)
                for p in (ROOT/'reports').glob(f'CALIBRATION_{group}_MB*.json'))
            cold_idle=time.time()-max(stage_start,kernel_progress)
            current_micro=stage.get('microbatch',hint['max_microbatch'])
            if alive and stage.get('calibrate',True) and cold_idle>1800 and not completed_calibration and current_micro>1:
                # A large CANN convolution specialization can compile indefinitely.
                # Bound this engineering choice, preserve all artifacts, and retry a
                # smaller native microbatch while keeping the effective batch at32.
                hint={'max_microbatch':current_micro//2,'reason':'30minutes without completed calibration or kernel progress',
                    'effective_batch':32,'time':time.time()}
                atomic_json(hint_path,hint)
                os.killpg(previous['pid'],signal.SIGTERM)
                prep_attempts[group]=0;previous['attempt']=0;atomic_json(prep_identity,previous)
                append_json(ROOT/'reports/PREPARATION_RETRIES.jsonl',{'group':group,**hint})
                continue
            if alive and elapsed<PREPARATION_TIMEOUT_SECONDS:
                results[group]='preparing';continue
            if alive:
                # PID/group was created by this workflow and verified by command path.
                os.killpg(previous['pid'],signal.SIGTERM)
                append_json(ROOT/'reports/PREPARATION_RETRIES.jsonl',{'group':group,'reason':'60minute current-stage timeout including cold CANN compilation','stage':stage,'time':time.time()})
                continue
            if pp is not None or previous:
                prep_attempts[group]=max(prep_attempts[group],previous.get('attempt',1))
                if prep_attempts[group]>=3:
                    results[group]='failed3times';continue
            prep_attempts[group]+=1
            process=launch('prepare_method.py','--group',group,label=f'prepare_{group}');prep_procs[group]=process
            atomic_json(prep_identity,{'pid':process.pid,'time':time.time(),'attempt':prep_attempts[group]})
            results[group]='preparing'
        for group in accepted:
            if time.time()<retry_after[group]:continue
            run=Path(cfg['results_root'])/group;s=state(group,cfg)
            p=procs.get(group)
            if p is not None and p.poll() is None:continue
            if training_complete(group,cfg):continue
            if pid_alive(run/'PROCESS.json'):continue
            if (run/'METHOD_STOPPED.json').exists():continue
            if p is not None:
                failure=run/'FAILED.json';info=json.loads(failure.read_text()) if failure.exists() else {'error':f'exit{p.returncode}'}
                signature=info.get('error','unknown');failures[group].append(signature)
                append_json(run/'supervisor_events.jsonl',{'event':'worker_exit','exit':p.returncode,'diagnostic':info,'time':time.time()})
                if info.get('oom'):
                    runtime=info.get('runtime',json.loads((ROOT/'reports'/f'TRAINING_ACCEPTANCE_{group}.json').read_text())['runtime'])
                    if runtime['microbatch']>1:
                        runtime['microbatch']//=2;atomic_json(run/'RUNTIME.json',runtime)
                        append_json(run/'supervisor_events.jsonl',{'event':'OOM_reduce_microbatch_keep_B32','runtime':runtime,'time':time.time()})
                        failures[group]=[]
                    else:
                        # MB1 already passed actual acceptance. A later allocation
                        # failure on a shared card is recoverable resource pressure.
                        retry_after[group]=time.time()+300;procs[group]=None;failures[group]=[]
                        append_json(run/'supervisor_events.jsonl',{'event':'MB1_memory_pressure_wait_then_retry',
                            'retry_after':retry_after[group],'others_continue':True,'time':time.time()})
                        continue
                if len(failures[group])>=3 and len(set(failures[group][-3:]))==1:
                    atomic_json(run/'METHOD_STOPPED.json',{'reason':'same unrecoverable failure3times','error':signature,'time':time.time()})
                    continue
            procs[group]=launch('train_worker.py','--group',group,label=f'train_{group}')
        # Prefer D0's card after it finishes; an individually failed/stopped method
        # must not prevent successful methods' evaluations on its released card.
        free_devices=[]
        for group,index in GROUP_DEVICES.items():
            run=Path(cfg['results_root'])/group;p=procs.get(group)
            finished=(training_complete(group,cfg) or results[group]=='failed3times' or (run/'METHOD_STOPPED.json').exists())
            busy=((p is not None and p.poll() is None) or pid_alive(run/'PROCESS.json') or
                  pid_alive(ROOT/'reports'/f'PREPARATION_PROCESS_{group}.json'))
            if finished and not busy:free_devices.append(index)
        if eval_proc is not None and eval_proc.poll() is not None:
            eval_proc=None;eval_path=None
        eval_identity=ROOT/'reports/HEAVY_EVALUATION_PROCESS.json'
        external_eval=pid_alive(eval_identity)
        if external_eval and eval_proc is None:
            eval_path=Path(json.loads(eval_identity.read_text())['job'])
        if free_devices and eval_proc is None and not external_eval:
            jobs=[]
            for group in accepted:
                for path in sorted((Path(cfg['results_root'])/group/'eval_queue').glob('*.json')):
                    if path.name.endswith(('.done.json','.error.json','.running.json','.deferred.json')) or path.with_suffix('.done.json').exists():continue
                    if pid_alive(path.with_suffix('.running.json')):continue
                    job=json.loads(path.read_text())
                    latest=Path(job['run_dir'])/'LATEST.json'
                    if not latest.exists() or job['update']>json.loads(latest.read_text())['update']:
                        # A volatile 500-step snapshot may be rolled back after a
                        # hard crash. Background writers wait for a durable update
                        # so normal recovery cannot quarantine their active output.
                        continue
                    lock_path=Path(job['run_dir'])/'evaluation_locks'/f'U{job["update"]:05d}.lock'
                    lock_path.parent.mkdir(parents=True,exist_ok=True)
                    with lock_path.open('a+') as eval_lease:
                        try:fcntl.flock(eval_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:continue
                    err=path.with_suffix('.error.json')
                    if err.exists() and json.loads(err.read_text()).get('attempts',0)>=3:continue
                    jobs.append(path)
            if jobs:
                eval_path=jobs[0];eval_device=f'npu:{free_devices[0]}'
                eval_proc=launch('evaluate_worker.py','--job',eval_path,'--device',eval_device,label='heavy_evaluation')
                atomic_json(eval_identity,{'pid':eval_proc.pid,'job':str(eval_path),'device':eval_device,'time':time.time()})
        # CPU analysis has its own process and never gates training/evaluation.
        if time.time()-last_report>180:
            if (ROOT/'scripts/report_async.py').exists():launch('report_async.py','--run-dir',cfg['results_root'],label='reports')
            last_report=time.time()
        snapshot={g:state(g,cfg) for g in GROUP_DEVICES}
        atomic_json(ROOT/'reports/WORKFLOW_STATUS.json',{'status':'RUNNING','methods':snapshot,
            'method_preparation':results,'heavy_eval':str(eval_path) if eval_path else None,'A100_dependency':False,'time':time.time()})
        all_done=all(training_complete(g,cfg) for g in GROUP_DEVICES)
        pending=[]
        for g in GROUP_DEVICES:
            pending += [p for p in (Path(cfg['results_root'])/g/'eval_queue').glob('*.json')
                if not p.name.endswith(('.done.json','.error.json','.running.json','.deferred.json')) and not p.with_suffix('.done.json').exists()]
        final_jobs=all((Path(cfg['results_root'])/g/'eval_queue/milestone_u20000.done.json').exists() for g in GROUP_DEVICES)
        if all_done and final_jobs and not pending and eval_proc is None and not external_eval:
            for g in GROUP_DEVICES:
                run=Path(cfg['results_root'])/g
                ident=json.loads((run/'checkpoints/full_U20000.identity.json').read_text())
                if digest(run/ident['path'])!=ident['sha256']:raise ValueError('Final checkpoint integrity failed: '+g)
            atomic_json(ROOT/'reports/NPU_COMPLETE.json',{'status':'ALL_FOUR20K_AND_EVALUATION_COMPLETE','time':time.time(),'methods':snapshot})
            return
        time.sleep(30)

if __name__=='__main__':
    try:main()
    except BaseException:
        append_json(ROOT/'reports/CONTROLLER_ERRORS.jsonl',{'time':time.time(),'traceback':traceback.format_exc()});raise

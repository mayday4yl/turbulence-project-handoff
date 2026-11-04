"""A100-only continuation supervisor, user authorized 2026-10-03.
Original NPU files remain immutable; import conversion preserves all train state,
with explicitly reseeded CUDA RNG because NPU RNG bytes are not portable.
"""
from pathlib import Path
import fcntl, json, os, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from execution.common import atomic_json, append_json, digest, seed_for
from execution.checkpoints import atomic_save, scheduler_state
GROUPS=['D0','R0','R08','L08']
CONFIG=ROOT/'configs/runtime_cuda.json'
C=json.loads(CONFIG.read_text()); RUNS=Path(C['results_root'])
REP=ROOT/'reports/migration_a100_20261003'

def read(p, default=None):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return {} if default is None else default

def alive(pid, needle):
    try:
        p=Path('/proc')/str(pid)
        return needle in (p/'cmdline').read_bytes().decode() and (p/'cwd').resolve()==ROOT
    except (OSError,ValueError):return False

def import_checkpoint(g, gate):
    run=RUNS/g
    if (run/'LATEST.json').exists():
        if not (run/'FRESH_U0_ORIGIN.json').exists():raise RuntimeError('Fresh A100 run origin missing')
        return True
    if list((run/'checkpoints').glob('*.pt')):raise RuntimeError('Uncommitted checkpoint requires explicit recovery')
    origin={'initialization':'shared random common U0','common_init_sha256':digest(ROOT/'protocol/common_init.pt'),
        'formal_start_update':0,'imported_npu_checkpoint':False,'time':time.time()}
    atomic_json(run/'FRESH_U0_ORIGIN.json',origin)
    atomic_json(run/'RUNTIME.json',gate['runtime'])
    return True

def launch(script,args,label):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3,5'
    with (ROOT/'logs'/f'{label}.log').open('ab') as log:
        return subprocess.Popen(['bash',str(ROOT/'scripts/run_cuda_env.sh'),str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def main():
    lock=(REP/'fresh_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(REP/'FRESH_CONTROLLER.json',{'pid':os.getpid(),'time':time.time(),'root':str(ROOT)})
    procs={};failures={g:[] for g in GROUPS};retry={g:0 for g in GROUPS};evalproc=None;lastreport=0
    while True:
        if read(ROOT/'protocol/CONTROL.json').get('action')=='stop':return
        snapshot={}
        for g in GROUPS:
            run=RUNS/g;run.mkdir(parents=True,exist_ok=True);gate=read(ROOT/'reports'/f'CUDA_TRAINING_ACCEPTANCE_{g}.json')
            if gate.get('status')!='PASS':snapshot[g]={'status':'WAIT_CUDA_ACCEPTANCE'};continue
            if not import_checkpoint(g,gate):snapshot[g]={'status':'WAIT_CHECKPOINT_TRANSFER'};continue
            s=read(run/'STATUS.json');snapshot[g]=s
            if s.get('update',0)>=20000 and read(run/'LATEST.json').get('update')==20000:continue
            if (run/'METHOD_STOPPED.json').exists():continue
            pid=read(run/'PROCESS.json').get('pid')
            if pid and alive(pid,'train_worker.py'):continue
            p=procs.get(g)
            if p is not None and p.poll() is None:continue
            if time.time()<retry[g]:continue
            if p is not None:
                error=read(run/'FAILED.json',{'error':f'exit{p.returncode}'})
                append_json(run/'supervisor_events.jsonl',{'event':'worker_exit','time':time.time(),'exit':p.returncode,'diagnostic':error})
                failures[g].append(error.get('error','unknown'))
                if error.get('oom'):
                    runtime=read(run/'RUNTIME.json',gate['runtime'])
                    if runtime['microbatch']>1:
                        runtime['microbatch']//=2;atomic_json(run/'RUNTIME.json',runtime);failures[g]=[]
                        append_json(run/'supervisor_events.jsonl',{'event':'OOM_microbatch_reduced','effective_batch':32,'runtime':runtime,'time':time.time()})
                    else:retry[g]=time.time()+120;procs[g]=None;continue
                if len(failures[g])>=3 and len(set(failures[g][-3:]))==1:
                    atomic_json(run/'METHOD_STOPPED.json',{'reason':'repeated error3times','error':failures[g][-1],'time':time.time()});continue
            procs[g]=launch('train_worker.py',['--config',str(CONFIG),'--group',g],f'a100_formal_{g}')
            atomic_json(REP/f'LAUNCH_{g}.json',{'pid':procs[g].pid,'resume_update':read(run/'LATEST.json').get('update',0),'physical_gpu':C['physical_cuda_devices'][C['method_devices'][g]],'time':time.time()})
        # Heavy evaluation only on a released task slot; active training never waits.
        free=[]
        for index in (0,1):
            assigned=[g for g in GROUPS if C['method_devices'][g]==index]
            if all(read(RUNS/g/'LATEST.json').get('update')==20000 or (RUNS/g/'METHOD_STOPPED.json').exists() for g in assigned):
                if not any(alive(read(RUNS/g/'PROCESS.json').get('pid',0),'train_worker.py') for g in assigned):free.append(index)
        if evalproc is not None and evalproc.poll() is not None:evalproc=None
        if free and evalproc is None:
            jobs=[p for g in GROUPS for p in sorted((RUNS/g/'eval_queue').glob('*.json')) if not p.name.endswith(('.done.json','.error.json','.running.json','.deferred.json')) and not p.with_suffix('.done.json').exists() and read(p.with_suffix('.error.json')).get('attempts',0)<3]
            if jobs:
                job=jobs[0];evalproc=launch('evaluate_worker.py',['--job',str(job),'--device',f'cuda:{free[0]}'],f'a100_eval_{job.stem}')
                atomic_json(REP/'HEAVY_EVALUATION_PROCESS.json',{'pid':evalproc.pid,'job':str(job),'time':time.time()})
        if time.time()-lastreport>300:
            launch('report_async.py',['--run-dir',str(RUNS)],'a100_report_async');lastreport=time.time()
        atomic_json(REP/'WORKFLOW_STATUS.json',{'status':'A100_FRESH_COMMON_U0_TRAINING','methods':snapshot,'time':time.time(),'NPU_restart_permitted':False})
        if all(read(RUNS/g/'LATEST.json').get('update')==20000 and (RUNS/g/'eval_queue/milestone_u20000.done.json').exists() for g in GROUPS):
            atomic_json(REP/'COMPLETE.json',{'status':'ALL_FOUR_20K_WITH_EVALUATION','time':time.time()});return
        time.sleep(10)

if __name__=='__main__':
    try:main()
    except BaseException:
        append_json(REP/'FRESH_CONTROLLER_ERRORS.jsonl',{'time':time.time(),'traceback':traceback.format_exc()});raise

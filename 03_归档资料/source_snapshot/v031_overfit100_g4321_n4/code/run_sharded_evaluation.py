"""Four independent batch-one evaluators; no training entry point is exposed."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import sharded_eval_support as support
from sharded_eval_support import BASE, ROOT, CARDS, frozen, prior
from guard_distributed import process_info


def state(**fields):
    w=frozen.read(frozen.STATUS)
    w.update(fields,updated_utc=frozen.utc(),
        controller_pid=int(os.environ.get('V031_SHARDED_CONTROLLER_PID',os.getpid())))
    frozen.write(frozen.STATUS,w)


def cpu_env():
    return dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT/'code'))


def launch(label,card,args):
    folder=BASE/'processes'/label
    folder.mkdir(parents=True,exist_ok=False)
    stream=(folder/'worker.log').open('w')
    env=dict(os.environ,V031_EVAL_CARD=str(card))
    p=subprocess.Popen(['bash','code/run_eval_sharded_env.sh']+args,cwd=ROOT,
        env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    record={'label':label,'card':card,'argv':args,'identity':process_info(p.pid),'utc':frozen.utc()}
    frozen.write(folder/'launch.json',record)
    return {'process':p,'stream':stream,'folder':folder,'record':record,'started':time.monotonic()}


def finished(job):
    code=job['process'].poll()
    if code is None:
        if time.monotonic()-job['started']>7200:
            raise TimeoutError('Evaluation shard exceeded two-hour bound')
        return False
    job['stream'].close()
    frozen.write(job['folder']/'result.json',dict(job['record'],returncode=code,
        finished_utc=frozen.utc(),elapsed_seconds=time.monotonic()-job['started']))
    if code:
        raise RuntimeError(job['record']['label']+' failed: '+
            (job['folder']/'worker.log').read_text(errors='replace')[-6000:])
    return True


def job_args(job):
    return ['code/calibrate_evaluate.py','--mode','evaluate','--config','configs/train_v031.json',
        '--checkpoint',str(ROOT/'runs/formal/checkpoint_001000.pt'),'--weights',job['weights'],
        '--manifest',job['manifest'],'--batch-size','1','--output',job['output'],
        '--rollout-modes',job['mode'],'--numeric-config','artifacts/calibration_update100/calibration.json']


def worker():
    cfg=support.verify()
    plan=support.make_plan()
    state(phase='evaluation_cross_card_preflight',status='running',active_training=None,active_evaluation=None,
          active_evaluations=[],evaluation_shard_plan=str(BASE/'PLAN.json'))
    probes=[launch(f'probe_card{card}',card,['code/check_sharded_inference.py','--output',
        str(BASE/'probes'/f'card{card}')]) for card in CARDS]
    while probes:
        if (ROOT/'STOP_REQUESTED').exists():
            raise RuntimeError('Project stop requested')
        probes=[job for job in probes if not finished(job)]
        time.sleep(1)
    subprocess.run([frozen.PYTHON,'-B','code/verify_sharded_inference.py'],cwd=ROOT,env=cpu_env(),check=True)
    queues={card:[job for job in plan['jobs'] if job['card']==card] for card in CARDS}
    active={};done=[]
    started=time.monotonic()
    while any(queues.values()) or active:
        if (ROOT/'STOP_REQUESTED').exists():
            raise RuntimeError('Project stop requested')
        for card in list(active):
            proc,job=active[card]
            if finished(proc):
                if not support.verify_job(job,plan):
                    raise RuntimeError('Successful evaluator lacks complete shard')
                done.append(job['name']);del active[card]
        for card in CARDS:
            if card not in active and queues[card]:
                job=queues[card].pop(0)
                active[card]=(launch(job['name'],card,job_args(job)),job)
        running=[dict(card=card,weights=job['weights'],mode=job['mode'],
            global_update=1000,output=job['output'],identity=proc['record']['identity'])
            for card,(proc,job) in active.items()]
        state(status='running',phase='evaluation_sharded',active_training=None,
            active_evaluation={'global_update':1000,'weights':'raw_and_ema','sharded':True,
                'output':str(BASE),'cards':list(CARDS)},active_evaluations=running,
            completed_evaluation_shards=done,formal_global_update=1000,training_complete_to_1000=True)
        frozen.write(BASE/'PROGRESS.json',{'status':'running','utc':frozen.utc(),
            'active':running,'completed_shards':done,'total_shards':len(plan['jobs']),
            'elapsed_seconds':time.monotonic()-started,'reused_rows':len(plan['reused'])})
        time.sleep(5)
    state(phase='evaluation_merge',active_evaluations=[],active_evaluation=None)
    subprocess.run([frozen.PYTHON,'-B','-c','from sharded_eval_support import merge; merge()'],
                   cwd=ROOT,env=cpu_env(),check=True)
    for step in range(500,1001,100):
        for weights in ['raw','ema']:
            if not prior.verify_evaluation(cfg,step,weights):
                raise RuntimeError('A previously required evaluation is incomplete')
    subprocess.run([sys.executable,'code/write_progress_report.py','--root',str(ROOT),'--update','1000'],
                   cwd=ROOT,check=True)
    support.verify()
    state(phase='evaluation_complete_pending_release',completed_evaluation_updates=list(range(500,1001,100)),
          reports_completed=[500,1000],evaluation_complete_at_current_update=True)
    frozen.write(BASE/'COMPLETION.json',{'status':'complete','utc':frozen.utc(),
        'global_update':1000,'training_updates_added':0,'evaluation_elapsed_seconds':time.monotonic()-started,
        'completed_shards':done,'merge_acceptance_sha256':frozen.digest(BASE/'MERGE_ACCEPTANCE.json')})


def controller():
    lock=(ROOT/'.auto_to_1000.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    support.verify()
    own=process_info(os.getpid())
    os.environ['V031_SHARDED_CONTROLLER_PID']=str(own['pid'])
    BASE.mkdir(parents=True,exist_ok=True)
    frozen.write(ROOT/'artifacts/sharded_evaluation_job.json',dict(own,
        created_utc=frozen.utc(),command=sys.argv,log=str(BASE/'controller.log')))
    guard_dir=ROOT/'runs/stage3_eval4_a01'
    state(status='running',phase='evaluation_handoff_complete',execution_mode='four_card_sharded_evaluation',
        controller_starttime_ticks=own['starttime_ticks'],active_guard_directory=str(guard_dir),
        persistent_handoff_job='artifacts/sharded_evaluation_job.json',error=None,failure=None,
        active_training=None,active_evaluation=None,active_evaluations=[])
    frozen.write(ROOT/'artifacts/PERSISTENT_HANDOFF_STATUS.json',{
        'status':'executing_guarded_four_card_evaluation','utc':frozen.utc(),**own,
        'job':'artifacts/sharded_evaluation_job.json','training_updates_allowed':0})
    try:
        result=subprocess.run([sys.executable,'code/guard_distributed.py','--run-dir',str(guard_dir),
            '--timeout-seconds','14400','--memory-limit-gib','61','--command',sys.executable,
            '-u','code/run_sharded_evaluation.py','--mode','worker'],cwd=ROOT,stdin=subprocess.DEVNULL)
        frozen.require_released(guard_dir)
        if result.returncode:
            raise RuntimeError('Four-card evaluation guard failed; inspect retained artifacts before retry')
        if frozen.read(BASE/'COMPLETION.json')['status']!='complete':
            raise RuntimeError('Missing complete evaluation evidence')
        subprocess.run([frozen.PYTHON,'-B','code/check_parallel_final_checkpoint.py','1000'],
                       cwd=ROOT,env=cpu_env(),check=True)
        support.verify()
        state(status='PAUSED_AT_1000',phase='complete',formal_global_update=1000,
            training_complete_to_1000=True,full_evaluation_complete_at_1000=True,
            active_training=None,active_evaluation=None,active_evaluations=[],active_guard_directory=None,
            allow_automatic_updates_beyond_1000=False,next_action='Await explicit user authorization for continuation')
        frozen.write(ROOT/'artifacts/PERSISTENT_HANDOFF_STATUS.json',{'status':'PAUSED_AT_1000',
            'utc':frozen.utc(),'formal_global_update':1000,'allow_automatic_updates_beyond_1000':False})
    except BaseException as exc:
        state(status='blocked',phase='evaluation_sharded_engineering_inspection',error=repr(exc))
        frozen.write(BASE/'FAILURE.json',{'utc':frozen.utc(),'error':repr(exc)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['controller','worker'],required=True)
    args=parser.parse_args()
    (controller if args.mode=='controller' else worker)()

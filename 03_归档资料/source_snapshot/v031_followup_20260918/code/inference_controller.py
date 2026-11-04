"""Finite downstream inference queue, outside training processes and RNG."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time
from guard_shared import snapshot,process_info

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data2/yanglin/v031_overfit100_g4321_n4')
CARDS=(0,1,2,3)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def group(key,jobs):
    root=ROOT/'inference_queue'/key;root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():return
    attempts=sorted(root.glob('attempt_*'))
    if len(attempts)>=3:raise RuntimeError('Bounded inference infrastructure retry limit: '+key)
    if attempts:
        previous=attempts[-1]/'FAILURE.json'
        if previous.exists():
            f=json.loads(previous.read_text())
            if not f['resource_related']:raise RuntimeError('Repair inference implementation before retry: '+str(previous))
            delay=max(0,f['time']+3600-time.time())
            if delay:time.sleep(delay)
    attempt=root/f'attempt_{len(attempts)+1:02d}';attempt.mkdir()
    deadline=time.time()+180
    while True:
        locks=[]
        try:
            for card in CARDS:
                f=(ROOT/f'.followup_inference_card{card}.lock').open('a');locks.append(f)
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time()>deadline:raise TimeoutError('Previous inference guard still owns its role lock')
            time.sleep(2)
        finally:
            for f in locks:f.close()
    snap=snapshot(CARDS);save(attempt/'resource_before.json',snap)
    processes=[]
    for card,job in zip(CARDS,jobs):
        job=dict(job,rank=card,shards=4);path=attempt/f'job_rank{card}.json';save(path,job)
        if job['kind']=='heldout_generation':
            worker=['bash','code/run_single_env.sh','code/heldout.py','generate','--rank',str(card),'--shards','4']
        else:worker=['bash','code/run_single_env.sh','code/evaluate_followup.py','--job',str(path)]
        guard=attempt/f'guard_rank{card}'
        cmd=['python3','code/guard_shared.py','--cards',str(card),'--role','inference','--output',str(guard),
             '--seconds','21600','--own-cap-gib','8','--reserve-gib','6','--',*worker]
        f=(attempt/f'launcher_rank{card}.log').open('w')
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        processes.append((p,guard));save(attempt/f'guard_pid_rank{card}.json',process_info(p.pid))
    while any(p.poll() is None for p,_ in processes):time.sleep(5)
    results=[json.loads((g/'result.json').read_text()) for _,g in processes]
    if any(r['status']!='completed' for r in results):
        resource=all(r['status']=='completed' or any(x in str(r['reason']).lower() for x in ('latency noticeably','hbm','unqualified external','memory cap')) for r in results)
        save(attempt/'FAILURE.json',{'results':results,'resource_related':resource,'time':time.time()})
        if resource:return group(key,jobs)
        raise RuntimeError('Inference worker failed: '+str(attempt))
    save(root/'COMPLETE.json',{'status':'complete','attempt':str(attempt),'time':time.time()})

def eval_group(task,update,subset,candidate=None):
    cfg=json.loads((ROOT/'shared/original_train_config.json').read_text())
    old=task in ('V031_UNREFINED','I1_HELDOUT')
    cp=OLD/'runs/formal'/f'checkpoint_{update:06d}.pt' if old else ROOT/task/'formal'/f'checkpoint_{update:06d}.pt'
    data=ROOT/'shared/heldout32' if subset=='heldout32' else Path(cfg['dataset_root'])
    manifest=cfg['evaluation']['fixed64_manifest'] if subset=='fixed64' else 'start_manifest.jsonl'
    output=(ROOT/'I1_REFINE_REPAIR/heldout32'/candidate if task=='I1_HELDOUT'
            else ROOT/task/'evaluation'/f'update_{update:06d}_{subset}')
    job={'kind':'evaluation','task':task,'update':update,'checkpoint':str(cp),'checkpoint_sha256':digest(cp),
         'dataset_root':str(data),'manifest':manifest,'subset':subset,'output':str(output)}
    if subset=='previews':
        split=json.loads((ROOT/'shared/manifests/FROZEN_SPLITS.json').read_text())
        records=[json.loads(x) for x in (data/'start_manifest.jsonl').read_text().splitlines()]
        selected=set(split['legacy_preview_start_ids'])
        for sid in split['additional_preview_source_ids']:
            tid=min(r['trajectory_id'] for r in records if r['source_id']==sid)
            selected.update(r['start_id'] for r in records if r['trajectory_id']==tid)
        job['start_ids']=sorted(selected)
        save(ROOT/'shared/manifests/fixed_preview_starts.json',{'ids':job['start_ids'],'basis':'original example + first trajectory of each of5 previously frozen source IDs'})
    if candidate:job.update(candidate=candidate,selection_sha256=digest(ROOT/'I1_REFINE_REPAIR/selection.json'))
    key=f'{task}_{update}_{subset}'+('_'+candidate if candidate else '')
    group(key,[job]*4)

def reuse_heldout_u0():
    source=ROOT/'V031_UNREFINED/evaluation/update_001000_heldout32'
    target=ROOT/'I1_REFINE_REPAIR/heldout32/U0'
    count=0
    for p in source.glob('*/result.json'):
        r=json.loads(p.read_text())
        if r['metrics'][0]['G']==1:continue
        r['reused_from']={'path':str(p),'sha256':digest(p),'basis':'same frozen raw1000, same unrefined run_path, same FP32 backend and heldout input'}
        save(target/p.parent.name/'result.json',r);count+=1
    assert count==192
    save(target/'REUSED_COMPLETE.json',{'starts':192,'selected_before_evaluation':True})

def main():
    lock=(ROOT/'.inference_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Current I1 workers own all four inference role locks. They finish their
    # finite dev/confirm stage before this queue occupies those same resources.
    while not all((ROOT/'I1_REFINE_REPAIR'/f'worker_rank{r}_complete.json').exists() for r in CARDS):time.sleep(20)
    while not (ROOT/'shared/heldout32/PREPARED.json').exists():
        if (ROOT/'shared/heldout32/FAILURE.json').exists():break
        time.sleep(20)
    heldout=(ROOT/'shared/heldout32/PREPARED.json').exists()
    if heldout:
        try:
            group('heldout_generation',[{'kind':'heldout_generation'}]*4)
            eval_group('V031_UNREFINED',1000,'heldout32')
            selected=json.loads((ROOT/'I1_REFINE_REPAIR/selection.json').read_text())['selected']
            if selected:
                reuse_heldout_u0()
                for c in ('R_legacy',selected['name']):eval_group('I1_HELDOUT',1000,'heldout32',c)
        except Exception as e:
            heldout=False
            save(ROOT/'shared/heldout32/QUEUE_FAILURE.json',{'error':repr(e),'time':time.time(),'main_training_evaluation_continues':True})
    save(ROOT/'I1_REFINE_REPAIR/COMPUTE_COMPLETE.json',{'status':'finite_declared_inference_complete',
        'heldout_available':heldout,'time':time.time(),'no_training_dependency':True})
    # Poll only checkpoint files while waiting. Resource inspection occurs at
    # a real job launch; no repeated idle accelerator qualification.
    required={(task,u) for task in ('T1_DIRECT','T2_FEEDBACK_R') for u in range(100,1001,100)}
    done=set()
    while done!=required:
        available=[(task,u) for task,u in sorted(required-done) if (ROOT/task/'formal'/f'checkpoint_{u:06d}.pt').exists()]
        if not available:time.sleep(30);continue
        for task,u in available:
            eval_group(task,u,'full1600' if u in (500,1000) else 'fixed64')
            if u==1000 and heldout:eval_group(task,u,'heldout32')
            done.add((task,u))
    # Same fixed preview starts for all three raw models at the reported checkpoints.
    for update in (500,1000):eval_group('V031_UNREFINED',update,'previews')
    save(ROOT/'reports/EVALUATION_COMPLETE.json',{'tasks':sorted(done),'heldout_available':heldout,'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        save(ROOT/'reports/INFERENCE_CONTROLLER_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

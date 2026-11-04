"""Resource handoff -> disposable acceptance -> immutable freeze -> fresh common-U0 formal run."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from v4.runtime import save_json,sha256
from freeze import source_map
GROUPS=('D0','R0','R08','L08');R=ROOT/'reports'

def inventory():
    result={};apps={}
    for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
        pid,uuid=[s.strip() for s in l.split(',')];apps.setdefault(uuid,[]).append(int(pid))
    for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        idx,uuid,free,used,util=[s.strip() for s in l.split(',')]
        result[int(idx)]={'uuid':uuid,'free_MiB':int(free),'used_MiB':int(used),'util':int(util),'pids':apps.get(uuid,[])}
    return result

def parent_files():
    parent=ROOT.parent/'v4_train500_matched_v2_perf1_eval5kfull';run=ROOT.parents[1]/'runs/v4_train500_matched_v2'
    paths=[parent/'protocol'/n for n in ('FROZEN.json','CONTROL.json')]
    for g in GROUPS:paths.extend(run/g/n for n in ('STATUS.json','LATEST.json','train_steps.jsonl'))
    return {str(p):sha256(p) for p in paths}

def main():
    lock=(R/'PIPELINE.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'protocol/FROZEN.json').exists():raise RuntimeError('Already frozen: inspect existing formal PIDs; no duplicate pipeline')
    if (R/'ACCEPTANCE.json').exists():raise RuntimeError('Existing acceptance retained: inspect, do not repeat automatically')
    if not (R/'CPU_ACCEPTANCE.json').exists() or json.loads((R/'CPU_ACCEPTANCE.json').read_text())['status']!='PASS':raise RuntimeError('CPU gates not passed')
    children={};initial=parent_files();save_json(R/'PARENT_UNCHANGED_BEFORE.json',initial)
    save_json(R/'PIPELINE_STATUS.json',{'status':'READY_WAITING_RESOURCES','pid':os.getpid(),'formal_updates':0,'deadline':'23:30 Asia/Shanghai'})
    while True:
        if json.loads((ROOT/'protocol/CONTROL.json').read_text())['stop_at_next_boundary']:return
        state=json.loads((R/'RESOURCE_WAITER.json').read_text())
        control=json.loads((R/'RESOURCE_WAITER_CONTROL.json').read_text())
        reservations=state.get('reservations',{})
        if len(reservations)==2 and all(v['status']=='RESERVED' for v in reservations.values()):
            cards=sorted(map(int,reservations));mode='exclusive';break
        if time.monotonic()>=control['server_monotonic_deadline']:
            readings=[]
            for _ in range(10):readings.append(inventory());time.sleep(1)
            if all(min(x[g]['free_MiB'] for x in readings)>=65536 and sum(x[g]['util'] for x in readings)/10<=35 for g in (5,6)):
                cards=[5,6];mode='shared_existing_jobs';break
            save_json(R/'PIPELINE_STATUS.json',{'status':'FALLBACK_WAITING_SAFE_GPU56_HEADROOM','pid':os.getpid(),'formal_updates':0,'latest':readings[-1]})
            time.sleep(20);continue
        time.sleep(1)
    # Stop only our reservation manager and its children, before allocating the selected pair.
    (R/'RELEASE_RESERVATION').write_text('Handoff to authorized disposable acceptance\n')
    manager=state['pid'];cmd=Path(f'/proc/{manager}/cmdline')
    if cmd.exists():
        if b'scripts/resource_waiter.py' not in cmd.read_bytes() or Path(f'/proc/{manager}').stat().st_uid!=os.getuid():raise RuntimeError('Reservation manager identity mismatch')
        os.kill(manager,signal.SIGTERM)
    deadline=time.monotonic()+30
    while cmd.exists() and time.monotonic()<deadline:time.sleep(.1)
    if cmd.exists():raise RuntimeError('Reservation manager did not exit; refuse third-GPU allocation')
    now=inventory()
    if mode=='exclusive' and any(now[g]['pids'] for g in cards):raise RuntimeError('Foreign process appeared during reservation handoff')
    selection={'status':'SELECTED_PENDING_RESOURCE_ACCEPTANCE','gpu_uuids':{str(g):now[g]['uuid'] for g in cards},'mapping':{str(cards[0]):['R08','D0'],str(cards[1]):['R0','L08']},'group_gpu':{'R08':cards[0],'D0':cards[0],'R0':cards[1],'L08':cards[1]},'mode':mode,'external_before':{str(g):now[g] for g in cards},'time':time.time()}
    save_json(ROOT/'protocol/RESOURCE_SELECTION.json',selection)
    accepted_sources=source_map();attempt='training_acceptance_v1';target=R/attempt;target.mkdir(exist_ok=False)
    minimum_free={str(g):now[g]['free_MiB'] for g in cards}
    try:
        for g in GROUPS:
            gpu=selection['group_gpu'][g];log=(target/(g+'.log')).open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=selection['gpu_uuids'][str(gpu)],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/accept_training.py'),'--group',g,'--attempt',attempt],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children[g]=(child,log)
        save_json(R/'PREFLIGHT_PIDS.json',{'parent':os.getpid(),'workers':{g:p.pid for g,(p,_) in children.items()},'selection':selection})
        save_json(R/'PIPELINE_STATUS.json',{'status':'FOUR_TASK_L1_AND_RESOURCE_ACCEPTANCE_RUNNING','pid':os.getpid(),'formal_updates':0,'cards':cards})
        while any(p.poll() is None for p,_ in children.values()):
            sample=inventory()
            for g in cards:minimum_free[str(g)]=min(minimum_free[str(g)],sample[g]['free_MiB'])
            if min(minimum_free.values())<12288:raise RuntimeError('Two-card four-task headroom acceptance failed')
            if any(p.poll() not in (None,0) for p,_ in children.values()):raise RuntimeError('Temporary training acceptance failed; preserve evidence')
            if mode=='exclusive':
                owned={p.pid for p,_ in children.values()}
                if any(pid not in owned for c in cards for pid in sample[c]['pids']):raise RuntimeError('Foreign process raced exclusive acceptance')
            time.sleep(1)
        if any(p.returncode!=0 for p,_ in children.values()):raise RuntimeError('Acceptance nonzero exit')
    finally:
        for p,log in children.values():
            if p.poll() is None:p.terminate()
            log.close()
    groups={g:json.loads((target/g/'acceptance.json').read_text()) for g in GROUPS}
    assert all(r['status']=='PASS' for r in groups.values())
    assert parent_files()==initial,'Old stopped experiment changed'
    assert source_map()==accepted_sources,'Source changed during acceptance'
    save_json(R/'ACCEPTANCE.json',{'status':'PASS','formal_updates':0,'temporary_optimizer_calls':sum(r['temporary_optimizer_calls'] for r in groups.values()),'groups':groups,'resource':{'actual_four_tasks_on_two_cards':True,'min_free_MiB':minimum_free,'selection':selection},'accepted_source_sha256':accepted_sources,'old_stopped_run_unchanged':True})
    subprocess.run([sys.executable,str(ROOT/'scripts/freeze.py')],cwd=ROOT,check=True)
    save_json(R/'PIPELINE_STATUS.json',{'status':'FORMAL_LAUNCH_FRESH_COMMON_U0','pid':os.getpid(),'cards':cards,'acceptance_weights_discarded':True})
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_formal.py')],cwd=ROOT,check=True)
    assert parent_files()==initial
    save_json(R/'PIPELINE_STATUS.json',{'status':'FORMAL_CONTROLLER_EXITED_REQUIRES_DELIVERY_AUDIT','pid':os.getpid(),'old_stopped_run_unchanged':True})
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save_json(R/'PIPELINE_FAILED.json',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

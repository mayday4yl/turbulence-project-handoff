"""Guarded four-worker coordinator. Stage deliveries precede barrier release."""
import fcntl,json,os,signal,subprocess,sys,time,traceback,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deps'));sys.path.insert(0,str(ROOT/'scripts'))
os.environ['CUDA_VISIBLE_DEVICES']=''
from v4.runtime import save_json,sha256
from v4.stages import GROUPS,NODES,read_control,verify_budget_scope
from train_worker import verify_frozen
from stage_report import report


def safe_json(path):
    return json.loads(path.read_text()) if path.exists() else None

def eta_report(run):
    profiles={};result={'time':time.time(),'groups':{},'estimates_seconds':{},'method':'conditional stable train rate plus per-card serialized eval and checkpoint time; stage wait uses slowest group'}
    for group in GROUPS:
        path=run/group;status=safe_json(path/'STATUS.json') or {};u=status.get('update',0)
        lines=[]
        if (path/'train_steps.jsonl').exists():
            for line in (path/'train_steps.jsonl').read_text().splitlines():
                try:lines.append(json.loads(line))
                except json.JSONDecodeError:pass
        stable=[r for r in lines if r['update']>25][-200:]
        rate=sum(r['update_seconds'] for r in stable)/len(stable) if len(stable)>=20 else None
        populations={}
        for f in sorted((path/'evaluations').glob('U*/raw/*/aggregate.json')):
            a=json.loads(f.read_text())
            if a['elapsed_seconds']>0:populations[f.parent.name]=a['elapsed_seconds']
        timings=[]
        if (path/'timings.jsonl').exists():
            timings=[json.loads(s) for s in (path/'timings.jsonl').read_text().splitlines()]
        saves=[r['seconds'] for r in timings if r['kind']=='checkpoint_save'][-20:]
        save_rate=sum(saves)/len(saves) if saves else 0.
        profiles[group]=(u,rate,populations,save_rate)
        result['groups'][group]={'update':u,'stable_samples':len(stable),'sec_per_update':rate,'status':status.get('status','INITIALIZING'),
                                  'actual_time_categories':safe_json(path/'TIMEBOOK.json')}
    if all(p[1] is not None for p in profiles.values()):
        for target in NODES:
            if all(p[0]>=target for p in profiles.values()):continue
            costs={}
            for group,(u,rate,pops,save_rate) in profiles.items():
                training=max(0,target-u)*rate;eval_seconds=0.
                first=((u+249)//250)*250
                for node in range(first,target+1,250):
                    if (run/group/'evaluations'/f'U{node:05d}'/'NODE_COMMIT.json').exists():continue
                    names=['train8000','val800'] if node in (0,5000,7500,10000) else ['train_monitor400','val800']
                    per_pass=sum(pops.get(name,pops.get('val800',60.)*(10 if name=='train8000' else .5 if name=='train_monitor400' else 1)) for name in names)
                    eval_seconds+=per_pass*(2 if node%1250==0 else 1)
                costs[group]={'training':training,'evaluation':eval_seconds,'checkpoint_save':max(0,target-u)/25*save_rate}
            card_times=[]
            for members in [('R08','D0'),('R0','L08')]:
                card_times.append(max(costs[g]['training'] for g in members)+sum(costs[g]['evaluation']+costs[g]['checkpoint_save'] for g in members))
            result['estimates_seconds'][str(target)]={'to_all_four_evaluated':max(card_times),'per_group_remaining_costs':costs,
                 'stage_report_seconds':'not yet calibrated; CPU report and packaging time additional'}
    save_json(run/'ETA.json',result)


def main():
    frozen=verify_frozen();run=Path(frozen['results_root']);run.mkdir(parents=True,exist_ok=True)
    handle=(run/'COORDINATOR.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Explicit PAUSED state blocks launch')
    gpus=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    if any(v in gpus for v in frozen['gpu_uuids'].values()):raise RuntimeError('Target GPU not idle at formal start')
    attempt=uuid.uuid4().hex;coord=run/'coordination'/attempt;workers=[]
    try:
        for group,gpu in [('R08',5),('D0',5),('R0',6),('L08',6)]:
            directory=run/group;directory.mkdir(exist_ok=True)
            log=(directory/f'worker_{attempt}.log').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_HOME=str(ROOT/'assets/torch'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/train_worker.py'),'--group',group,'--coordination',str(coord)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            workers.append((child,log,group))
        save_json(run/'PIDS.json',{'attempt':attempt,'coordinator':os.getpid(),'workers':{g:p.pid for p,_,g in workers},'coordination':str(coord)})
        last_eta=0;last_resource_check=0
        while any(p.poll() is None for p,_,_ in workers):
            if (ROOT/'reports/RUN_PAUSED.json').exists() or any(p.poll() not in (None,0) for p,_,_ in workers):
                raise RuntimeError('Worker failure or explicit numerical/correctness pause')
            if time.monotonic()-last_resource_check>5:
                owned={p.pid for p,_,_ in workers}
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
                for line in rows.splitlines():
                    pid,gpu_uuid=[v.strip() for v in line.split(',')]
                    if gpu_uuid in frozen['gpu_uuids'].values() and int(pid) not in owned:
                        raise RuntimeError('Unregistered process on a reserved target GPU; stop owned workers only')
                last_resource_check=time.monotonic()
            for node in NODES:
                stage=run/'stages'/f'U{node:05d}'
                if (stage/'RELEASE.json').exists():continue
                ready=[stage/f'{g}.READY.json' for g in GROUPS]
                if not all(f.exists() for f in ready):continue
                verify_budget_scope(ROOT)
                for f in ready:
                    record=json.loads(f.read_text());assert record['update']==node
                    assert sha256(record['checkpoint_path'])==record['checkpoint_sha256']
                    assert sha256(record['evaluation_commit'])==record['evaluation_commit_sha256']
                save_json(stage/'REPORT_STATUS.json',{'status':'BUILDING_COMPLETE_DELIVERY','time':time.time()})
                manifest=report(run,node)
                control=read_control(ROOT/'protocol/CONTROL.json')
                action='stop' if node==10000 or control['stop_at_common_update']==node or control['stop_at_next_boundary'] else 'continue'
                release={'update':node,'all_four_complete':True,'delivery_manifest':str(manifest),
                         'delivery_manifest_sha256':sha256(manifest),'action':action,'time':time.time(),
                         'reason':'USER_DECIDED_COMMON_BUDGET_TRUNCATION' if action=='stop' and node<10000 else 'DEFAULT_COMMON_PROTOCOL'}
                save_json(stage/'RELEASE.json',release)
                save_json(stage/'REPORT_STATUS.json',{'status':'COMPLETE','action':action,'time':time.time()})
                save_json(run/'LATEST_MILESTONE.json',release)
                print(json.dumps({'event':'COMMON_STAGE_DELIVERED',**release}),flush=True)
            if time.monotonic()-last_eta>60:eta_report(run);last_eta=time.monotonic()
            time.sleep(1)
        if any(p.returncode for p,_,_ in workers):raise RuntimeError('Worker failure at shutdown')
        statuses={g:json.loads((run/g/'STATUS.json').read_text()) for g in GROUPS}
        control=read_control(ROOT/'protocol/CONTROL.json')
        complete=all(r['update']==10000 and r['status']=='COMMON_STAGE_COMPLETE_STOP' for r in statuses.values())
        save_json(run/'STATUS.json',{'status':'COMPLETE_U10000_ALL_DELIVERIES' if complete else 'USER_STOPPED_PRESERVE_ACTUAL_COUNTS',
                  'groups':statuses,'control':control,'no_auto_extension':True})
        eta_report(run);print('All workers exited; no further training authorized',flush=True)
    except BaseException as exc:
        if not (ROOT/'reports/RUN_PAUSED.json').exists():
            save_json(ROOT/'reports/RUN_PAUSED.json',{'status':'PAUSED','gate':'formal_coordinator','exception':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        for p,log,g in workers:
            if p.poll() is None:p.terminate()
        deadline=time.monotonic()+120
        while any(p.poll() is None for p,_,_ in workers) and time.monotonic()<deadline:time.sleep(.5)
        for p,log,g in workers:
            if p.poll() is None:p.kill();p.wait()
            log.close()
        fcntl.flock(handle,fcntl.LOCK_UN);handle.close()

if __name__=='__main__':main()

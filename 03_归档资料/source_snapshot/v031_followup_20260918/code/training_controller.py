"""Sequential bounded training, conditional on measured real co-load qualification."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, shutil, statistics, subprocess, sys, time
from guard_shared import snapshot,policy_latency,process_info

ROOT=Path(__file__).resolve().parents[1]
CARDS=(0,1,2,3)

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def choose_candidate(viable):
    fastest=min(x['median_seconds'] for x in viable)
    tied=[x for x in viable if x['median_seconds']<=fastest*1.03]
    # Selection metadata must not create a cycle through all_candidates.
    return dict(min(tied,key=lambda x:(x['peak_reserved_bytes'],x['relative_spread'])))

def freeze_selection(task,selected):
    cfg=ROOT/'configs'/f"{task}_mb{selected['microbatch']}.json"
    proof=ROOT/task/'native_semantics.json'
    save(ROOT/task/'RUNTIME_SELECTION.json',selected)
    c=json.loads(cfg.read_text());c['gates']={'targeted_acceptance_passed':True,'training_config_frozen':True}
    paths=[ROOT/'code/train_worker.py']+sorted((ROOT/'code/v031').glob('*.py'))
    c['code_hashes']={str(p.relative_to(ROOT)):digest(p) for p in paths}
    c['identity']['runtime_selection_sha256']=digest(ROOT/task/'RUNTIME_SELECTION.json')
    c['identity']['native_semantics_sha256']=digest(proof)
    c['identity']['actual_configuration']=c['training']
    final=ROOT/'configs'/f'{task}_formal.json';save(final,c)
    snap=ROOT/task/'accepted_code';snap.mkdir(exist_ok=False)
    for p in paths:
        target=snap/p.relative_to(ROOT/'code');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    return selected,final

def saved_acceptance(task):
    final=ROOT/'configs'/f'{task}_formal.json'
    if not final.exists():return None
    c=json.loads(final.read_text());p=ROOT/task/'RUNTIME_SELECTION.json'
    if not all(c['gates'].get(k) for k in ('targeted_acceptance_passed','training_config_frozen')):
        raise RuntimeError('Saved formal configuration is not accepted')
    if digest(p)!=c['identity']['runtime_selection_sha256'] or digest(ROOT/task/'native_semantics.json')!=c['identity']['native_semantics_sha256']:
        raise RuntimeError('Saved qualification identity changed')
    for name,sha in c['code_hashes'].items():
        if digest(ROOT/name)!=sha or digest(ROOT/task/'accepted_code'/Path(name).relative_to('code'))!=sha:
            raise RuntimeError('Accepted trainer identity changed: '+name)
    return json.loads(p.read_text()),final

def run(task, label, command, *, cards='0,1,2,3',cap=32,seconds=3600):
    folder=ROOT/task/'qualification'/label
    guard=folder/'guard';folder.mkdir(parents=True,exist_ok=False)
    cmd=['python3',str(ROOT/'code/guard_shared.py'),'--cards',cards,'--role','training',
         '--output',str(guard),'--seconds',str(seconds),'--own-cap-gib',str(cap),
         '--reserve-gib','8','--',*command]
    with (folder/'launcher.log').open('w') as f:
        p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return json.loads((guard/'result.json').read_text()),folder

def command(config, output, mode, mb, extra=()):
    return ['bash','code/run_distributed_env.sh','-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node=4','code/train_worker.py','--config',str(config),'--output',str(output),
        '--mode',mode,'--microbatch',str(mb),*map(str,extra)]

def smoke(task,mode,mb,cap,ledger,tag,extra=()):
    reserved=3 if mode=='resume-check' else (2 if mode=='memory-smoke' else 4)
    if ledger['consumed']+reserved>30:raise RuntimeError('Declared total30 smoke budget would be exceeded')
    output=ROOT/task/'qualification'/tag/'worker'
    cfg=ROOT/'configs'/f'{task}_mb{mb}.json'
    if mode=='memory-smoke':extra=('--warmup-updates',1,'--measure-updates',1)
    ledger['reservations'].append({'tag':tag,'reserved':reserved,'mode':mode,'mb':mb})
    save(ROOT/task/'SMOKE_LEDGER.json',ledger)
    status,folder=run(task,tag,command(cfg,output,mode,mb,extra),cap=cap,seconds=3600)
    files=list(output.glob('results_rank*.json'))+list(output.glob('error_rank*.json'))
    attempts=max([json.loads(p.read_text()).get('optimizer_step_attempts_this_invocation',0) for p in files] or [reserved])
    # Missing output is charged conservatively; never grant free retries.
    ledger['consumed']+=attempts;ledger['reservations'][-1].update(consumed=attempts,status=status['status'])
    save(ROOT/task/'SMOKE_LEDGER.json',ledger)
    result=json.loads((output/'results.json').read_text()) if (output/'results.json').exists() else None
    return status,result,folder

def changed(old,new):
    pids=lambda s:{(p['pid'],p.get('starttime_ticks')) for c in s['cards'] for p in c['processes']}
    return bool(pids(old)-pids(new)) or min(c['hbm_free_mb'] for c in new['cards'])>=min(c['hbm_free_mb'] for c in old['cards'])+4096 or statistics.mean(c['aicore_percent'] for c in new['cards'])<=statistics.mean(c['aicore_percent'] for c in old['cards'])-20

def wait_improvement(task,baseline,reason,first_check=None):
    next_check=first_check or time.time()+3600
    save(ROOT/task/'status.json',{'status':'waiting_resources','reason':reason,'next_check_unix':next_check,'snapshot':baseline})
    while True:
        time.sleep(max(0,next_check-time.time()))
        next_check=time.time()+3600
        new=snapshot(CARDS);save(ROOT/'resources/training_latest.json',new)
        if changed(baseline,new):return
        save(ROOT/task/'status.json',{'status':'waiting_resources','reason':reason,'next_check_unix':time.time()+3600,'snapshot':new})

def resource_failure(status,folder):
    reason=str(status.get('reason','')).lower()
    log=(folder/'guard/worker.log').read_text(errors='replace') if (folder/'guard/worker.log').exists() else ''
    if any(x in reason for x in ('latency noticeably','hbm reserve','memory cap','insufficient measured','new unqualified','unhealthy')):return True
    if any(x in log.lower() for x in ('out of memory','memory allocation failed')):return True
    raise RuntimeError('Engineering/semantic failure requires repair before retry: '+str(folder)+' '+reason)

def qualify(task,attempt):
    baseline=snapshot(CARDS);save(ROOT/'resources/training_latest.json',baseline)
    cap=min(32, min(c['hbm_free_mb']/1024 for c in baseline['cards'])-10)
    if cap<8:return None,baseline,'Less than8GiB available for controlled own allocation plus reserve'
    ledger_path=ROOT/task/'SMOKE_LEDGER.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'consumed':3,'fixture_updates':3,'reservations':[]}
    # No optimizer update. A real single-card function/graph check is sufficient here;
    # the selected runtime must separately pass the actual four-rank DDP test.
    proof=ROOT/task/'native_semantics.json'
    if not proof.exists():
        status,folder=run(task,f'a{attempt}_semantics',['bash','code/run_single_env.sh','code/accept_native.py',
            '--task',task,'--output',str(proof)],cards='0',cap=cap,seconds=3600)
        if status['status']!='completed':
            resource_failure(status,folder)
            return None,baseline,'Native semantics shared resource gate: '+str(status['reason'])
    if json.loads(proof.read_text()).get('status')!='passed':raise RuntimeError('Native semantics failed')
    viable=[]
    pressure=[]
    for p in (ROOT/task/'qualification').glob('*/worker/results.json'):
        r=json.loads(p.read_text())
        if r.get('mode')=='memory-smoke' and r.get('status')=='passed':pressure.append((p,r))
    for mb in (8,4,2,1):
        reusable=[(p,r) for p,r in pressure if r['microbatch']>=mb and r['max_peak_reserved_bytes']<(cap-1)*2**30]
        if reusable:
            proof_path,memory=min(reusable,key=lambda pr:pr[1]['microbatch'])
            save(ROOT/task/'qualification'/f'a{attempt}_pressure_reuse_mb{mb}.json',{'source':str(proof_path),
                'basis':'same task FP32 all-G4 full updates at equal or larger microbatch; new candidate still executes warmup plus3 measured complete updates',
                'actual_cap_gib':cap,'microbatch':mb})
        else:
            status,memory,folder=smoke(task,'memory-smoke',mb,cap,ledger,f'a{attempt}_memory_mb{mb}')
            if status['status']!='completed':
                text=(folder/'guard/worker.log').read_text(errors='replace')
                if 'out of memory' in text.lower():continue
                resource_failure(status,folder)
                return None,baseline,'Shared resource/engineering failure during pressure check: '+str(status['reason'])
            pressure.append((folder/'worker/results.json',memory))
        status,performance,folder=smoke(task,'performance-smoke',mb,cap,ledger,f'a{attempt}_speed_mb{mb}')
        if status['status']!='completed':
            if 'out of memory' in (folder/'guard/worker.log').read_text(errors='replace').lower():continue
            resource_failure(status,folder)
            return None,baseline,'Shared resource gate during speed check: '+str(status['reason'])
        durations=performance['measured_seconds_per_update']
        # A highly erratic shared run is not evidence for a stable formal launch.
        if max(durations)>1.5*min(durations):
            return None,baseline,'Real shared throughput unstable (>1.5 max/min); defer training'
        viable.append({'microbatch':mb,'median_seconds':performance['median_seconds_per_update'],
            'peak_reserved_bytes':max(memory['max_peak_reserved_bytes'],performance['max_peak_reserved_bytes']),
            'relative_spread':(max(durations)-min(durations))/statistics.median(durations)})
    if not viable:return None,baseline,'No candidate passed bounded pressure checks'
    selected=choose_candidate(viable)
    mb=selected['microbatch'];cfg=ROOT/'configs'/f'{task}_mb{mb}.json'
    ddpout=ROOT/task/'qualification'/f'a{attempt}_ddp_mb{mb}'/'worker'
    cmd=['bash','code/run_distributed_env.sh','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
         'code/check_distributed_gradients.py','--config',str(cfg),'--output',str(ddpout),'--microbatch',str(mb)]
    status,folder=run(task,f'a{attempt}_ddp_mb{mb}',cmd,cap=cap,seconds=3600)
    if status['status']!='completed':raise RuntimeError('Selected native DDP acceptance failed; inspect exact evidence')
    status,resume,folder=smoke(task,'resume-check',mb,cap,ledger,f'a{attempt}_resume_mb{mb}')
    if status['status']!='completed':raise RuntimeError('Selected native complete-state restore acceptance failed')
    selected.update(own_cap_gib=cap,all_candidates=viable,expected_1000_update_seconds=1000*selected['median_seconds'],
                    qualification_unix=time.time(),actual_shared_snapshot=baseline)
    return freeze_selection(task,selected),baseline,None

def main():
    lock=(ROOT/'.training_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        # Ensure I1 has actually started, not merely that its launch was requested.
        while not list((ROOT/'I1_REFINE_REPAIR').glob('worker_rank*_progress.json')):time.sleep(20)
        if (ROOT/task/'COMPLETE_1000.json').exists():continue
        accepted=saved_acceptance(task)
        status_path=ROOT/task/'status.json'
        if accepted is None and status_path.exists():
            prior=json.loads(status_path.read_text())
            if prior.get('status')=='waiting_resources':
                wait_improvement(task,prior['snapshot'],prior['reason'],prior['next_check_unix'])
        attempt=1+max([int(p.name.split('_')[0][1:]) for p in (ROOT/task/'qualification').glob('a*_*')] or [0])
        while accepted is None:
            accepted,baseline,reason=qualify(task,attempt)
            if accepted is not None:break
            wait_improvement(task,baseline,reason);attempt+=1
        selected,cfg=accepted;mb=selected['microbatch']
        output=ROOT/task/'formal'
        save(ROOT/task/'status.json',{'status':'formal_launching','microbatch':mb,'effective_batch':32,'limit':1000,
             'expected_1000_update_seconds':selected['expected_1000_update_seconds']})
        status,folder=run(task,f'a{attempt}_formal',command(cfg,output,'train',mb,('--run-until-update',1000)),
                         cap=selected['own_cap_gib'],seconds=max(7200,selected['expected_1000_update_seconds']*1.6+1800))
        if status['status']!='completed':
            save(ROOT/task/'status.json',{'status':'needs_engineering_resume','guard':status,
                'checkpoint_paths':[str(p) for p in output.glob('checkpoint_*.pt')]})
            raise RuntimeError('Formal run interrupted; preserve states, heartbeat must classify/recover before continuing')
        result=json.loads((output/'results.json').read_text())
        if result['global_update']!=1000 or result['counts']['paths']!=32000:raise RuntimeError('Unexpected formal endpoint')
        cp=output/'checkpoint_001000.pt'
        subprocess.run(['bash','code/run_single_env.sh','code/verify_boundary.py','--checkpoint',str(cp),
            '--output',str(ROOT/task/'COMPLETE_1000.json')],cwd=ROOT,check=True)
    save(ROOT/'reports/TRAINING_COMPLETE.json',{'T1':1000,'T2':1000,'evaluation_pending':True})

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        save(ROOT/'reports/TRAINING_CONTROLLER_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()})
        raise

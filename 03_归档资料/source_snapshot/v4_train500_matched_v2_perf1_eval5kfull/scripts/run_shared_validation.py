"""Controller monitors only its four children and physical GPU5/6."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
work=ROOT/'reports/shared_gpu_acceptance'
workers=[];samples=[]

def paused(message):
    (ROOT/'reports/RUN_PAUSED.json').write_text(json.dumps({'status':'PAUSED','gate':'shared_resources','exception':str(message),'time':time.time()},indent=2)+'\n')

try:
    if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Existing PAUSED state')
    assert json.loads((ROOT/'reports/mutex_cpu.json').read_text())['status']=='PASS'
    for group in ('D0','R0','R08','L08'):
        assert json.loads((ROOT/f'reports/acceptance_{group}.json').read_text())['status']=='PASS'
    if work.exists():raise RuntimeError('Existing shared acceptance attempt; review before relaunch')
    work.mkdir()
    for group,gpu in [('R08',5),('D0',5),('R0',6),('L08',6)]:
        log=(work/f'{group}.log').open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/validate_shared_gpu.py'),'--group',group],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((p,log,group))
    (work/'PIDS.json').write_text(json.dumps({g:p.pid for p,_,g in workers}))
    while any(p.poll() is None for p,_,_ in workers):
        if (ROOT/'reports/RUN_PAUSED.json').exists() or any(p.poll() not in (0,None) for p,_,_ in workers):
            raise RuntimeError('Worker failure or explicit pause')
        lines=subprocess.check_output(['nvidia-smi','--id=5,6','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
        for line in lines.splitlines():
            gpu,used,total=[int(x.strip()) for x in line.split(',')]
            samples.append({'time':time.time(),'gpu':gpu,'used_MiB':used,'total_MiB':total})
            if total-used<max(4096,.1*total):raise RuntimeError(f'GPU{gpu} headroom below agreed envelope')
        time.sleep(.5)
    if any(p.returncode for p,_,_ in workers):raise RuntimeError('Child nonzero exit')
    for p,_,g in workers:assert json.loads((work/f'{g}.json').read_text())['status']=='PASS'
    # Verify no update-permitted event is inside an evaluation lease.
    for gpu in (5,6):
        owner=None;commits=[]
        for line in (work/'mutex'/f'gpu{gpu}.events.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e['event']=='grant':
                assert owner is None
                assert all(v['status'] in ('PAUSED','TERMINAL') for v in e['workers'].values())
                owner=e['owner']
            elif e['event']=='update_permitted':assert owner is None
            elif e['event']=='evaluation_committed_release':
                assert owner==e['group'];owner=None;commits.append((e['group'],e['evaluation_id']))
        assert owner is None and len(commits)==len(set(commits))==6
    (work/'memory_samples.json').write_text(json.dumps(samples)+'\n')
    (ROOT/'reports/shared_gpu.json').write_text(json.dumps({'status':'PASS','four_processes':True,'gpu_indices':[5,6],
        'microbatch':2,'effective_batch':32,'eval_peer_models_optimizer_EMA_resident':True,
        'max_used_MiB':{str(g):max(r['used_MiB'] for r in samples if r['gpu']==g) for g in [5,6]},
        'headroom_min_fraction':.1,'no_overlapping_eval_leases':True,'no_peer_update_during_eval':True,
        'extra_optimizer_steps':0},indent=2)+'\n')
    print('PASS four-process resources/mutex/full metrics/repeat validation; no formal training',flush=True)
except BaseException as exc:
    if not (ROOT/'reports/RUN_PAUSED.json').exists():paused(exc)
    traceback.print_exc();sys.exit(1)
finally:
    for p,log,g in workers:
        if p.poll() is None:p.terminate();p.wait(timeout=30)
        log.close()

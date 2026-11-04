"""Wait for parent boundary hold, then run bounded real-GPU zero-step benchmarks."""
import json,os,pathlib,subprocess,sys,time,uuid,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deps')]
from v4.runtime import save_json
PARENT=ROOT.parent/'v4_train500_matched_v2'
RUN=ROOT.parents[1]/'runs/v4_train500_matched_v2'
GPUS={'R08':5,'D0':5,'R0':6,'L08':6}

def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False

def main():
 out=ROOT/'reports/PERFORMANCE_STATUS.json';pids=json.loads((RUN/'PIDS.json').read_text())
 save_json(out,{'status':'WAITING_PARENT_U0_DELIVERY_AND_BOUNDARY_EXIT','parent_pids':pids,'time':time.time()})
 while any(alive(pid) for pid in [pids['coordinator'],*pids['workers'].values()]):
  if (PARENT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Parent numerical/correctness pause; no automatic bypass')
  time.sleep(5)
 statuses={g:json.loads((RUN/g/'STATUS.json').read_text()) for g in GPUS}
 if not all(v['status'] in ('USER_STOPPED_AT_BOUNDARY','ENGINEERING_STOPPED_AFTER_RAW_U0') for v in statuses.values()):raise RuntimeError(f'Unexpected parent exit {statuses}')
 if not all(v['update']==0 for v in statuses.values()):raise RuntimeError('Parent made formal updates; review execution transition before tuning')
 uuids=json.loads((ROOT/'protocol/FROZEN_PARENT.json').read_text())['gpu_uuids']
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if any(u in apps for u in uuids.values()):raise RuntimeError('Reserved GPU not idle')
 attempt=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());work=ROOT/'reports/performance'/attempt;work.mkdir(parents=True,exist_ok=False)
 workers=[];samples=[]
 try:
  for g,gpu in GPUS.items():
   log=(work/f'{g}.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/benchmark_performance.py'),'--group',g,'--attempt',attempt],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   workers.append((g,p,log))
  save_json(out,{'status':'BENCHMARK_RUNNING','attempt':attempt,'pids':{g:p.pid for g,p,_ in workers},'optimizer_updates':0,'time':time.time()})
  while any(p.poll() is None for _,p,_ in workers):
   if (work/'FAILURE.json').exists() or any(p.poll() not in (None,0) for _,p,_ in workers):raise RuntimeError('Benchmark failure; preserve and pause')
   sample=subprocess.check_output(['nvidia-smi','--id=5,6','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
   samples.append({'time':time.time(),'gpu':sample});time.sleep(1)
  if any(p.returncode for _,p,_ in workers):raise RuntimeError('Benchmark failed')
  reports={g:json.loads((work/f'{g}.json').read_text()) for g in GPUS}
  if not all(v['status']=='PASS' for v in reports.values()):raise RuntimeError('Incomplete benchmark')
  save_json(work/'gpu_samples.json',samples)
  # True solo GPU runs: the other benchmark process has exited and freed its model.
  solo_reports={}
  for pair in [[('R08',5),('R0',6)],[('D0',5),('L08',6)]]:
   workers=[];solo_attempt=attempt+'_solo_'+pair[0][0];solo_work=ROOT/'reports/performance'/solo_attempt;solo_work.mkdir()
   for g,gpu in pair:
    log=(solo_work/f'{g}.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/benchmark_performance.py'),'--group',g,'--attempt',solo_attempt,'--solo'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    workers.append((g,child,log))
   save_json(out,{'status':'SOLO_BENCHMARK_RUNNING','attempt':attempt,'solo_attempt':solo_attempt,'pids':{g:c.pid for g,c,_ in workers},'optimizer_updates':0})
   while any(c.poll() is None for _,c,_ in workers):
    if (solo_work/'FAILURE.json').exists() or any(c.poll() not in (None,0) for _,c,_ in workers):raise RuntimeError('Solo benchmark failure')
    time.sleep(1)
   for g,c,log in workers:
    if c.returncode:raise RuntimeError('Solo benchmark failure')
    log.close();solo_reports[g]=json.loads((solo_work/f'{g}.json').read_text())
  save_json(out,{'status':'BENCHMARK_PASS_NEEDS_SELECTION_AND_EXECUTION_ACCEPTANCE','attempt':attempt,'reports':reports,'solo_reports':solo_reports,'optimizer_updates':0,'time':time.time()})

 finally:
  save_json(work/'gpu_samples.json',samples)
  for g,p,log in workers:
   if p.poll() is None:p.terminate();p.wait(timeout=60)
   log.close()

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save_json(ROOT/'reports/PERFORMANCE_STATUS.json',{'status':'PAUSED','exception':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

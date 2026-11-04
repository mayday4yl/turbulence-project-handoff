"""Bounded experiment orchestrator. Only starts workers in this directory."""
from common import *
import subprocess,signal,fcntl,traceback,sys
children=[]

def available():
 text=subprocess.check_output(['npu-smi','info'],text=True)
 return [i for i in (2,3) if f'No running processes found in NPU {i}' in text],text

def launch(phase,arm,device):
 check_deadline(1800)
 free,snapshot=available();assert device in free,f'Physical NPU {device} occupied; will not co-load'
 log=(ROOT/f'logs/{phase}_{arm}.log').open('a')
 p=subprocess.Popen(['bash',str(ROOT/'run_env.sh'),str(ROOT/'worker.py'),phase,'--arm',arm,'--device',str(device)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 item={'pid':p.pid,'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'phase':phase,'arm':arm,'device':device,'start_epoch':time.time()};children.append((p,item,log))
 save_json(ROOT/'reports/WORKERS.json',[x[1] for x in children]);append_json(ROOT/'logs/lifecycle.jsonl',dict(item,event='start'))
 return p

def wait_group(procs):
 while any(p.poll() is None for p in procs):
  check_deadline(300)
  failed=[p for p in procs if p.poll() not in (None,0)]
  if failed:raise RuntimeError(f'Worker failed: {[(p.pid,p.returncode) for p in failed]}')
  time.sleep(15)
 assert all(p.returncode==0 for p in procs),[(p.pid,p.returncode) for p in procs]
 for p,item,log in children:
  if p in procs:
   append_json(ROOT/'logs/lifecycle.jsonl',dict(item,event='finish',end_epoch=time.time(),exit_code=p.returncode));log.close()

def stop_own():
 for p,item,log in children:
  if p.poll() is None:
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for p,item,log in children:
  if p.poll() is None:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)

def freeze(devices):
 reports={arm:json.loads((ROOT/f'reports/BENCH_{arm}.json').read_text()) for arm in ('Raw','Field')}
 for r in reports.values():assert r['status']=='PASS'
 parallel=len(devices)==2
 combine=max if parallel else sum
 update_seconds=combine([r['max_update_seconds']*1.20 for r in reports.values()])
 off_seconds=combine([r['off_input_seconds']*1.3 for r in reports.values()])
 refine_seconds=combine([r['refinement_stage_seconds']*1.35 for r in reports.values()])
 remaining=experiment_deadline()-time.time();candidates=json.loads((ROOT/'protocol/REFINEMENT_SUBSET_CANDIDATES.json').read_text())
 selection=None;estimates=[]
 for sources in (8,4):
  post=max(7200,64*off_seconds+sources*(3*refine_seconds+off_seconds)+900)
  early_eval=2*64*off_seconds
  feasible=[]
  for n in CONFIG['allowed_formal_updates']:
   required=n*update_seconds+early_eval+post+1800
   estimates.append({'N':n,'refinement_sources':sources,'predicted_remaining_seconds':required,'remaining_seconds':remaining,'fits':required<=remaining})
   if required<=remaining:feasible.append(n)
  if feasible:
   selection=(max(feasible),sources,post);break
 if selection is None:
  save_json(ROOT/'reports/BUDGET_BLOCKED.json',{'reason':'Even common N=200 with symmetric 4-source refinement cannot fit','estimates':estimates});raise RuntimeError('Budget infeasible: no formal training launched')
 n,sources,post=selection
 frozen={'N':n,'common_nodes':[0,n//2,n],'devices':devices,'parallel':parallel,'refinement_sources':sources,'refinement_trajectory_ids':candidates['ordered_8_distinct_sources'][:sources],
 'post_training_reserve_seconds':post+1800,'minimum_eval_save_reserve_seconds':7200,'safety_seconds':1800,'validation_metrics_seen_before_lock':False,'epoch':time.time(),'estimates':estimates,
 'common_warm_sha256':digest(ROOT/'runs/common_warm.pt'),'sample_manifest_sha256':digest(ROOT/'protocol/SAMPLES.json'),'scale_sha256':digest(ROOT/'protocol/FIELD_SCALES.json'),
 'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*.py')},'config':CONFIG,'training_refinement':False,'evaluation_refinement':['OFF','ON'],'no_automatic_continuation':True}
 save_json(ROOT/'protocol/FROZEN.json',frozen)
 return frozen

def main():
 lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 existing_clock=ROOT/'reports/EXPERIMENT_CLOCK.json'
 retry='--repair-preformal-environment' in sys.argv
 if existing_clock.exists():
  assert retry and not (ROOT/'protocol/FROZEN.json').exists() and not (ROOT/'runs/common_warm.pt').exists(),'No formal restart/resume'
 devices,snapshot=available();assert devices,'No free authorized NPU 2/3'
 if existing_clock.exists():started=json.loads(existing_clock.read_text())['start_epoch']
 else:
  started=time.time();save_json(existing_clock,{'start_epoch':started,'deadline_epoch':started+43200,'budget_seconds':43200,'coding_preparation_seconds':started-CONFIG['coding_started_epoch'],'starts_before_first_NPU_acceptance':True})
 save_json(ROOT/'reports/STATUS.json',{'status':'ACCEPTANCE','epoch':time.time(),'preserved_original_clock':retry})
 save_json(ROOT/'reports/CONTROLLER.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'devices':devices,'initial_npu_smi':snapshot,'status':'ACCEPTANCE','epoch':started})
 guardlog=(ROOT/'logs/deadline_guard.log').open('w');subprocess.Popen([sys.executable,str(ROOT/'deadline_guard.py')],cwd=ROOT,stdout=guardlog,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  wait_group([launch('accept','Raw',devices[0])])
  wait_group([launch('warm','Raw',devices[0])])
  if len(devices)==2:wait_group([launch('bench','Raw',devices[0]),launch('bench','Field',devices[1])])
  else:
   for arm in ('Raw','Field'):wait_group([launch('bench',arm,devices[0])])
  frozen=freeze(devices)
  save_json(ROOT/'reports/STATUS.json',{'status':'FORMAL_TRAINING','N':frozen['N'],'epoch':time.time()})
  if len(devices)==2:wait_group([launch('train','Raw',devices[0]),launch('train','Field',devices[1])])
  else:
   for arm in ('Raw','Field'):wait_group([launch('train',arm,devices[0])])
  result='COMPLETE'
 except BaseException:
  result='BLOCKED';save_json(ROOT/'reports/CONTROLLER_FAILURE.json',{'epoch':time.time(),'traceback':traceback.format_exc()});stop_own()
 finally:
  subprocess.run([sys.executable,str(ROOT/'report.py')],cwd=ROOT,timeout=240)
  save_json(ROOT/'reports/STATUS.json',{'status':result,'epoch':time.time(),'wall_seconds':time.time()-started,'no_autoresume':True})
 print(result,flush=True)
if __name__=='__main__':main()

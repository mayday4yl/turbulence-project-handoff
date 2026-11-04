"""CPU-only follow-through after both training actors finish full validation.

This does not construct a restoration network, launch an NPU job, or update an
optimizer. Visual inspection and local delivery remain explicitly pending.
"""
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'unidb_d0_20261102'
DEADLINE=1793678400

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def write(path,q):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 t=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 t.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,path)

def identity():return {'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21]}

def command(args,label,timeout):
 log=ROOT/'evidence'/f'{label}_{time.time_ns()}.log'
 env=os.environ.copy();env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='8',
  OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
 with log.open('w') as f:
  p=subprocess.run(['bash',str(OLD/'run_env.sh'),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,
   stdin=subprocess.DEVNULL,env=env,timeout=timeout)
 assert p.returncode==0,('CPU postprocess failed',label,p.returncode,str(log))
 return {'command':list(map(str,args)),'log':str(log),'log_sha256':sha(log),'exit_code':p.returncode}

def collect():
 refs=[];updates={}
 for branch in ['original','improved']:
  out=ROOT/'runs'/branch
  state=json.loads((out/'STATUS.json').read_text());assert state['status']=='COMPLETE' and state['final_full_validation_done']
  current=json.loads((out/'LATEST.json').read_text());best=json.loads((out/'BEST.json').read_text())
  updates[branch]={'latest':current['update'],'best':best['update'],'new_updates':state['new_updates']}
  for role,u in [('latest',current['update']),('best',best['update'])]:
   path=out/'validation'/f'U{u:08d}_ema_full';summary=path/'SUMMARY.json'
   q=json.loads(summary.read_text())
   assert q['status']=='PASS' and q['full'] and q['weights']=='EMA' and q['count']==5000 and q['per_g']==1000
   assert q['update']==u and q['branch']==branch and set(q['by_g'])=={'0.5','1','2','3','4'}
   checkpoint=current if role=='latest' else best['checkpoint']
   cpu_path=ROOT/'evidence'/f'STATE_AUDIT_{branch}_{"FINAL" if role=="latest" else "BEST"}.json'
   cpu=json.loads(cpu_path.read_text());assert cpu['status']=='PASS'
   bound=cpu['checkpoint']
   assert bound['path']==checkpoint['path'] and bound['sha256']==checkpoint['sha256'] and bound['update']==u
   assert sha(checkpoint['path'])==checkpoint['sha256']
   assert q['parameter_digest']==bound['EMA']['full_parameter_digest'],'Saved predictions are not bound to the actual checkpoint EMA'
   refs.append({'branch':branch,'role':role,'update':u,'validation':str(path),'summary':str(summary),'summary_sha256':sha(summary),
    'checkpoint_path':checkpoint['path'],'checkpoint_sha256':checkpoint['sha256'],'checkpoint_EMA_full_digest':bound['EMA']['full_parameter_digest'],
    'checkpoint_CPU_audit':str(cpu_path),'checkpoint_CPU_audit_sha256':sha(cpu_path)})
 return refs,updates

def main():
 lock=(ROOT/'CPU_FINALIZER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time();commands=[];status='WAITING_FOR_MODEL_VALIDATION';pid=identity()
 write(ROOT/'evidence/ACTIVE_CPU_FINALIZER.json',{**pid,'started_unix':started,'script':str(Path(__file__).resolve()),
  'source_sha256':sha(__file__),'deadline_unix':DEADLINE,'NPU_jobs_allowed':0,'optimizer_updates_allowed':0})
 try:
  while True:
   complete=ROOT/'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json';exitfile=ROOT/'evidence/CONTROLLER_EXIT.json'
   if complete.exists() and exitfile.exists():
    q=json.loads(complete.read_text());ex=json.loads(exitfile.read_text())
    if q['status']=='PASS' and ex['all_subprocesses_exited'] and ex['unix']>=q['unix']:break
   assert time.time()<DEADLINE+8*3600,'Model pipeline did not finish by CPU finalizer engineering timeout'
   write(ROOT/'evidence/CPU_FINALIZER_STATUS.json',{'status':status,'actor':pid,'unix':time.time()})
   time.sleep(20)
  refs,updates=collect();unique=sorted({r['validation'] for r in refs})
  status='AUDITING_SAVED_OUTPUTS_CPU';write(ROOT/'evidence/CPU_FINALIZER_STATUS.json',{'status':status,'actor':pid,'refs':refs,'unix':time.time()})
  args=[ROOT/'scripts/audit_saved_outputs.py']
  for val in unique:args+=['--validation',val]
  args+=['--out',ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json','--threads','8']
  commands.append(command(args,'SAVED_OUTPUTS_CPU_AUDIT',4*3600))
  audit=json.loads((ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json').read_text());assert audit['status']=='PASS'
  assert all(audit['audited_summaries'].get(r['summary'])==r['summary_sha256'] for r in refs)
  commands.append(command([ROOT/'scripts/make_campaign_report.py','--final'],'FINAL_CAMPAIGN_REPORT',1800))
  # Prove none of the NPU rank actors recorded for this campaign still exist.
  alive=[]
  for f in (ROOT/'runs').glob('*/RANK_*.json'):
   q=json.loads(f.read_text());p=Path('/proc')/str(q['pid'])/'stat'
   if p.exists() and p.read_text().split()[21]==q['start_ticks']:alive.append(q)
  assert not alive,('Own rank actors remain alive',alive)
  status='CPU_AUDIT_AND_REPORT_PASS_VISUAL_AND_LOCAL_DELIVERY_PENDING'
  write(ROOT/'evidence/CPU_POSTPROCESS_ACCEPTANCE.json',{'status':'PASS','actor':pid,'started_unix':started,'completed_unix':time.time(),
   'updates':updates,'refs':refs,'commands':commands,'restoration_model_calls':0,'new_optimizer_updates':0,'NPU_jobs':0,
   'visual_QA_pending':True,'local_sync_pending':True,'final_delivery_complete':False})
  write(ROOT/'STATUS.json',{'status':status,'unix':time.time(),'deadline_unix':DEADLINE,'updates':updates,
   'instruction':'Inspect fixed figures/curves, sync local small deliverables, final acceptance, then delete heartbeat.'})
 except BaseException as exc:
  status='CPU_POSTPROCESS_FAILURE'
  write(ROOT/'evidence'/f'CPU_FINALIZER_FAILURE_{time.time_ns()}.json',{'status':'FAIL','error':repr(exc),
   'traceback':traceback.format_exc(),'actor':pid,'unix':time.time(),'commands':commands})
  raise
 finally:
  write(ROOT/'evidence/CPU_FINALIZER_STATUS.json',{'status':status,'actor':pid,'unix':time.time()})
  write(ROOT/'evidence/CPU_FINALIZER_EXIT.json',{'actor':pid,'unix':time.time(),'status':status})

if __name__=='__main__':main()

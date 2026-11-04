"""CPU-only follow-through after both training actors finish full validation.

This does not construct a restoration network, launch an NPU job, or update an
optimizer. Visual inspection and local delivery remain explicitly pending.
"""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'unidb_d0_20261102'
DEADLINE=1793721600

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
 refs=[];updates={};checked_files={}
 authority=json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())
 assert authority['deadline_unix']==DEADLINE
 history=[Path(p).resolve() for p in authority.get('history_campaign_roots',[authority['previous_campaign_root']])]
 assert history and history[-1]==Path(authority['previous_campaign_root']).resolve() and len(history)==len(set(history))
 approved=[ROOT.resolve()]+history
 for branch in ['original','improved']:
  out=ROOT/'runs'/branch
  state=json.loads((out/'STATUS.json').read_text());assert state['status']=='COMPLETE' and state['final_full_validation_done']
  current=json.loads((out/'LATEST.json').read_text());best=json.loads((out/'BEST.json').read_text())
  base=45411 if branch=='original' else 0;extension_base=authority['extension_parents'][branch]['update']
  assert state['update']==current['update'] and state['new_updates']==current['update']-base
  assert state['extension_new_updates']==current['update']-extension_base
  assert state['extension_new_presentations']==96*state['extension_new_updates']
  updates[branch]={'latest':current['update'],'best':best['update'],'campaign_new_updates':current['update']-base,
   'campaign_presentations':96*(current['update']-base),'extension_start_update':extension_base,
   'extension_new_updates':current['update']-extension_base,'extension_new_presentations':96*(current['update']-extension_base)}
  for role,u in [('latest',current['update']),('best',best['update'])]:
   alias=out/'validation'/f'U{u:08d}_ema_full';path=alias.resolve();summary=(path/'SUMMARY.json').resolve()
   assert any(path.is_relative_to(r/'runs'/branch/'validation') for r in approved),path
   q=json.loads(summary.read_text())
   assert q['status']=='PASS' and q['full'] and q['weights']=='EMA' and q['count']==5000 and q['per_g']==1000
   assert q['update']==u and q['branch']==branch and set(q['by_g'])=={'0.5','1','2','3','4'}
   checkpoint=current if role=='latest' else best['checkpoint'];checkpoint_path=Path(checkpoint['path']).resolve()
   assert any(checkpoint_path.is_relative_to(r/'runs'/branch/'checkpoints') for r in approved),checkpoint_path
   cpu_path=ROOT/'evidence'/f'STATE_AUDIT_{branch}_{"FINAL" if role=="latest" else "BEST"}.json'
   cpu=json.loads(cpu_path.read_text());assert cpu['status']=='PASS'
   bound=cpu['checkpoint']
   assert Path(bound['path']).resolve()==checkpoint_path and bound['sha256']==checkpoint['sha256'] and bound['update']==u
   if str(checkpoint_path) not in checked_files:checked_files[str(checkpoint_path)]=sha(checkpoint_path)
   assert checked_files[str(checkpoint_path)]==checkpoint['sha256']
   assert q['parameter_digest']==bound['EMA']['full_parameter_digest'],'Saved predictions are not bound to actual checkpoint EMA'
   refs.append({'branch':branch,'role':role,'update':u,'validation':str(path),'validation_alias':str(alias),
    'campaign_stage':next(r.name for r in approved if path.is_relative_to(r)),
    'summary':str(summary),'summary_sha256':sha(summary),
    'checkpoint_path':str(checkpoint_path),'checkpoint_sha256':checkpoint['sha256'],'checkpoint_EMA_full_digest':bound['EMA']['full_parameter_digest'],
    'checkpoint_CPU_audit':str(cpu_path),'checkpoint_CPU_audit_sha256':sha(cpu_path)})
 return refs,updates


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
 if args.dry_run:
  try:
   complete=json.loads((ROOT/'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json').read_text());assert complete['status']=='PASS'
   refs,updates=collect()
  except (AssertionError,FileNotFoundError,KeyError) as exc:
   write(ROOT/'evidence/CPU_FINALIZER_DRYRUN.json',{'status':'EXPECTED_REFUSAL_INCOMPLETE','reason':repr(exc),
    'restoration_model_calls':0,'metric_model_calls':0,'optimizer_updates':0,'NPU_jobs':0});raise SystemExit(2)
  write(ROOT/'evidence/CPU_FINALIZER_DRYRUN.json',{'status':'READY_FOR_CPU_AUDIT','refs':refs,'updates':updates,'new_model_calls':0});return
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
  args=[ROOT/'scripts/audit_campaign_saved_outputs.py']
  for val in unique:args+=['--validation',val]
  args+=['--out',ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json','--threads','8']
  commands.append(command(args,'SAVED_OUTPUTS_CPU_AUDIT',4*3600))
  audit=json.loads((ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json').read_text());assert audit['status']=='PASS'
  audited={str(Path(p).resolve()):h for p,h in audit['audited_summaries'].items()}
  assert all(audited.get(str(Path(r['summary']).resolve()))==r['summary_sha256'] for r in refs)
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
   'visual_QA_pending':True,'visual_QA_requires_new_figure_SHA':True,'local_sync_pending':True,'final_delivery_complete':False,
   'source_sha256':{name:sha(ROOT/'scripts'/name) for name in ['finalize_campaign.py','make_campaign_report.py','audit_campaign_saved_outputs.py']},
   'CPU_metric_forward_batches_this_invocation':audit.get('CPU_metric_forward_batches_this_invocation'),
   'historical_completed_audits_reused':audit.get('historical_completed_audits_reused'),
   'deadline_scope':'Optimizer stop2026-11-04 00:00 Asia/Shanghai; complete final CPU metric evaluation and report afterward.'})
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

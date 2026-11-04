"""Persistent, scoped controller for the explicitly authorized 3+5 campaign.

No old run is restarted. Gates discard their updates. Deadline stopping also
exists independently in every worker. This controller never signals outsiders.
"""
import fcntl,hashlib,json,os,re,signal,subprocess,sys,threading,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'unidb_d0_20261102'
RUN_ENV=OLD/'run_env.sh'
DEADLINE=1793678400
ALLOCATION={'original':[0,1,2],'improved':[3,4,5,6,7]}
LOCK_ROOT=ROOT.parent/'v6_diffuir_l_20261027/gates'
AUDIT_ENV={'TORCH_DEVICE_BACKEND_AUTOLOAD':'0'}
LOCKS=[]
JOBS={}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def identity(pid=None):
 pid=os.getpid() if pid is None else pid
 return {'pid':pid,'start_ticks':Path(f'/proc/{pid}/stat').read_text().split()[21]}

def append(name,obj):
 with (ROOT/'evidence'/name).open('a') as f:f.write(json.dumps({'unix':time.time(),**obj},ensure_ascii=False)+'\n')

def stage(name,**extra):
 write(ROOT/'STATUS.json',{'status':name,'controller':identity(),'deadline_unix':DEADLINE,'unix':time.time(),**extra})
 print(name,json.dumps(extra,ensure_ascii=False),flush=True)

def freeze_check():
 q=json.loads((ROOT/'protocol/RUN_SOURCE.json').read_text())
 for rel,expected in q['files'].items():assert sha(ROOT/rel)==expected,('frozen source changed',rel)
 a=json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())
 assert a['deadline_unix']==DEADLINE and a['max_NPUs']==8 and a['effective_batch_both']==96
 return a

def acquire():
 for p in [ROOT/'CONTROLLER.lock']+[LOCK_ROOT/f'AB_NPU_{i}.lock' for i in range(8)]:
  f=p.open('a+');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);LOCKS.append(f)

def check_accelerators(label):
 raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
 (ROOT/'evidence'/f'NPU_{label}_{int(time.time())}.txt').write_text(raw)
 found=[]
 for line in raw.splitlines():
  m=re.match(r'^\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|',line)
  if m:found.append((int(m[1]),int(m[2])))
 assert all(card==1 and pid==2938958 for card,pid in found),('new external occupants; no process was signalled',found)
 assert sum('910B3' in x and 'OK' in x for x in raw.splitlines())==8
 write(ROOT/'evidence'/f'RESOURCE_ACCEPTANCE_{label}.json',{'status':'PASS','unix':time.time(),
  'external_contexts':found,'card1_context_authorized':True,'external_processes_signalled':0,
  'limitation':'Driver context2938958 exists; this is not an OS-process liveness claim.'})

def cpu_command(args,out,timeout=1800):
 out=Path(out)
 if out.exists():
  prior=json.loads(out.read_text())
  if prior.get('status','').startswith('PASS'):
   records=[prior['checkpoint']] if 'checkpoint' in prior else list(prior.get('audits',{}).values())
   requested={str(Path(args[i+1]).resolve()) for i,v in enumerate(args[:-1]) if str(v) in ['--path','--compare-eight','--compare-three','--parent']}
   assert records and {str(Path(q['path']).resolve()) for q in records}==requested
   assert all(sha(q['path'])==q['sha256'] for q in records)
   append('ACTOR_EVENTS.jsonl',{'event':'reuse_completed_CPU_acceptance','path':str(out),'sha256':sha(out),
    'historical_auditor_sha256':prior.get('source_sha256'),'checks_unchanged_for_previously_passed_states':True})
   return prior
 env=os.environ.copy();env.update(AUDIT_ENV)
 log=ROOT/'evidence'/(Path(out).stem+'.log')
 with log.open('w') as f:
  p=subprocess.run(['bash',str(RUN_ENV),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
 assert p.returncode==0,('CPU gate failed',str(log),p.returncode)
 q=json.loads(Path(out).read_text());assert q['status'].startswith('PASS'),q.get('error')
 return q

def gate(name,branch,out,cards,steps=2,resume=None):
 """Reuse completed immutable gates after a controller/checker repair only."""
 out=Path(out);result=out/'RESULT.json'
 if not result.exists():return launch(name,branch,'gate',out,cards,steps=steps,resume=resume)
 q=json.loads(result.read_text());assert q['status']=='PASS' and q['branch']==branch and q['phase']=='gate'
 assert len(q['ranks'])==len(cards) and all(v['completed_updates_this_process']==steps for v in q['ranks'])
 final=q['final_checkpoint'];assert sha(final['path'])==final['sha256'] and latest(out)==Path(final['path'])
 for rank in range(len(cards)):
  actor=json.loads((out/f'RANK_{rank}.json').read_text());p=Path('/proc')/str(actor['pid'])/'stat'
  assert not(p.exists() and p.read_text().split()[21]==actor['start_ticks']),('Completed gate actor still alive',actor)
  assert json.loads((out/f'EXIT_rank{rank}.json').read_text())['status']=='PASS'
 cfg=json.loads((out/'CONFIG.json').read_text());assert cfg['branch']==branch and cfg['phase']=='gate' and cfg['physical_world_size']==len(cards)
 # Both controlled repairs preserve every model/data/training-related source.
 archive=ROOT/'evidence/engineering_repair_02_EMA_checker/RUN_SOURCE.json'
 if archive.exists():
  old=json.loads(archive.read_text())['files'];now=json.loads((ROOT/'protocol/RUN_SOURCE.json').read_text())['files']
  for rel in ['training_worker.py','improved_objective.py','validation.py','protocol/USER_AUTHORITY.json','protocol/AUDIT_STATE_FROZEN_TOLERANCES.json']:
   assert old[rel]==now[rel]==sha(ROOT/rel),('Gate scientific/worker source changed',rel)
 append('ACTOR_EVENTS.jsonl',{'event':'reuse_completed_NPU_gate','name':name,'out':str(out),'checkpoint':final,
  'training_model_data_sources_unchanged':True,'new_model_calls':0})
 return {'out':out,'name':name,'branch':branch,'reused':True}

def launch(name,branch,phase,out,cards,steps=2,resume=None,eval_only=False):
 freeze_check();out=Path(out);out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(UNIDB_DEVICES=','.join(map(str,cards)),HCCL_CONNECT_TIMEOUT='600',
  HCCL_EXEC_TIMEOUT='7200',PYTHONDONTWRITEBYTECODE='1')
 args=['bash',str(RUN_ENV),'-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(cards)}',
  str(ROOT/'training_worker.py'),'--branch',branch,'--phase',phase,'--out',str(out),
  '--steps',str(steps),'--dummy-mode','parameters']
 if resume:args+=['--resume',str(resume)]
 if eval_only:args+=['--eval-only']
 stamp=str(time.time_ns());log=ROOT/'evidence'/f'{name}_{stamp}.log'
 with log.open('w') as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
 record={**identity(p.pid),'name':name,'branch':branch,'phase':phase,'out':str(out),'cards':cards,
  'command':args,'log':str(log),'started_unix':time.time(),'eval_only':eval_only,'controller':identity()}
 write(ROOT/'evidence'/f'ACTOR_{name}_{stamp}.json',record)
 job={'process':p,'record':record,'out':out,'name':name,'branch':branch,'eval_only':eval_only}
 JOBS[name]=job;append('ACTOR_EVENTS.jsonl',{'event':'launched',**record});return job

def terminate_own_group(job,why):
 p=job['process'];pgid=p.pid
 members=[]
 for path in Path('/proc').glob('[0-9]*/stat'):
  try:
   fields=path.read_text().split();pid=int(path.parent.name)
   if int(fields[4])==pgid:members.append(identity(pid))
  except (OSError,ValueError,IndexError):continue
 if members:
  append('ACTOR_EVENTS.jsonl',{'event':'terminate_own_launched_group','why':why,'job':job['record'],'members':members})
  try:os.killpg(pgid,signal.SIGTERM)
  except ProcessLookupError:pass
  end=time.time()+30
  while time.time()<end and p.poll() is None:time.sleep(1)
  # Only this controller's recorded subprocess group is eligible for escalation.
  try:os.killpg(pgid,signal.SIGKILL)
  except ProcessLookupError:pass
 try:p.wait(timeout=30)
 except subprocess.TimeoutExpired:raise RuntimeError('Own subprocess group failed to exit')

def wait_gates(jobs,timeout=1500):
 active=[j for j in jobs if not j.get('reused')]
 start=time.time()
 while True:
  for j in active:
   code=j['process'].poll()
   if code is not None and code!=0:
    for own in active:terminate_own_group(own,'peer engineering gate failed')
    raise RuntimeError(('NPU engineering gate failed',j['record'],code))
  if all(j['process'].poll()==0 for j in active):break
  if time.time()-start>timeout or time.time()>=DEADLINE:
   for j in active:terminate_own_group(j,'engineering gate timeout/deadline')
   raise RuntimeError('NPU engineering gate timeout/deadline')
  time.sleep(3)
 for j in jobs:
  q=json.loads((j['out']/'RESULT.json').read_text());assert q['status']=='PASS'
  append('ACTOR_EVENTS.jsonl',{'event':'exited','name':j['name'],'exit_code':0})

def latest(out):
 q=json.loads((Path(out)/'LATEST.json').read_text());p=Path(q['path']).resolve()
 assert p.is_relative_to(ROOT) and p.is_file() and sha(p)==q['sha256']
 return p

def audit(path,label):
 out=ROOT/'evidence'/f'STATE_AUDIT_{label}.json'
 return cpu_command([ROOT/'audit_state.py','--path',path,'--out',out],out)

def deadline_guard():
 while time.time()<DEADLINE:time.sleep(min(15,max(.1,DEADLINE-time.time())))
 write(ROOT/'STOP_REQUEST.json',{'reason':'absolute_user_deadline','deadline_unix':DEADLINE,'unix':time.time(),
  'instruction':'Finish only in-flight update, save complete state, then final validation.'})
 append('ACTOR_EVENTS.jsonl',{'event':'deadline_stop_request_written'})

def archive_attempt(branch,out,attempt,job):
 archive=ROOT/'evidence'/'recovery'/branch/f'attempt{attempt}_{time.time_ns()}'
 archive.mkdir(parents=True)
 # Snapshot mutable markers/logs; immutable checkpoints and predictions remain in place.
 for path in Path(out).glob('*.json*'):
  if path.is_file():
   import shutil
   shutil.copy2(path,archive/path.name)
 for p in (ROOT/'protocol/RUN_SOURCE.json',ROOT/'training_worker.py'):
  import shutil
  shutil.copy2(p,archive/p.name)
 write(archive/'EXIT.json',{'job':job['record'],'returncode':job['process'].returncode,'unix':time.time()})

def formal():
 check_accelerators('BEFORE_FORMAL')
 running={b:launch('formal_'+b,b,'formal',ROOT/'runs'/b,cards) for b,cards in ALLOCATION.items()}
 attempts={b:0 for b in running};done={};stage('FORMAL_RUNNING',allocation=ALLOCATION,actors={b:j['record'] for b,j in running.items()})
 while running:
  snapshot={}
  for b,j in list(running.items()):
   code=j['process'].poll();out=j['out'];snapshot[b]={'actor':j['record'],'returncode':code}
   sp=out/'STATUS.json'
   if sp.exists():
    try:snapshot[b]['worker_status']=json.loads(sp.read_text())
    except (json.JSONDecodeError,OSError):pass
   if code is None:
    status=snapshot[b].get('worker_status',{});now=time.time()
    if status.get('status')=='TRAINING' and now-status.get('last_completed_unix',now)>900:
     terminate_own_group(j,'no completed update for15min');code=j['process'].returncode
    elif status.get('status')=='EVALUATING':
     # Full evaluation may take hours, but every rank must continue producing
     # four-image groups. Use artifact progress rather than total runtime.
     paths=list((out/'validation').glob('**/PROGRESS_rank*.json'))+list((out/'validation').glob('**/groups/G*/*.json'))
     newest=max([p.stat().st_mtime for p in paths]+[status.get('evaluation_started_unix',now)])
     if now-newest>1800:
      terminate_own_group(j,'no saved validation progress for30min');code=j['process'].returncode
     elif now<=DEADLINE+4*3600:continue
     else:terminate_own_group(j,'final evaluation engineering allowance exceeded');code=j['process'].returncode
    elif now>DEADLINE+4*3600:
     terminate_own_group(j,'final evaluation exceeded four-hour engineering allowance');code=j['process'].returncode
    else:continue
   if code==0:
    q=json.loads((out/'STATUS.json').read_text());assert q['status']=='COMPLETE' and q['final_full_validation_done']
    audit(latest(out),b+'_FINAL')
    best=json.loads((out/'BEST.json').read_text());audit(best['checkpoint']['path'],b+'_BEST')
    done[b]={'status':'PASS','exit_code':0,'result':q,'latest':json.loads((out/'LATEST.json').read_text()),'best':best}
    del running[b];append('ACTOR_EVENTS.jsonl',{'event':'formal_branch_complete','branch':b,'result':q})
   else:
    terminate_own_group(j,'failed formal actor cleanup');attempts[b]+=1;archive_attempt(b,out,attempts[b],j)
    assert attempts[b]<=2,('bounded recovery exhausted',b)
    path=latest(out);audit(path,b+'_RECOVERY_'+str(attempts[b]))
    eval_only=time.time()>=DEADLINE or (out/'TRAINING_STOPPED.json').exists()
    running[b]=launch('recovery_'+b+'_'+str(attempts[b]),b,'formal',out,ALLOCATION[b],resume=path,eval_only=eval_only)
  write(ROOT/'evidence'/'LIVE_ACTORS.json',{'unix':time.time(),'controller':identity(),'running':snapshot,'done':done})
  time.sleep(10)
 write(ROOT/'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json',{'status':'PASS','branches':done,'unix':time.time(),
  'delivery_pending':'Independent saved-output CPU metric audit, visual QA, local sync and final report remain required.'})
 stage('MODEL_VALIDATION_COMPLETE_CPU_QA_PENDING',branches={b:v['result'] for b,v in done.items()})

def main():
 (ROOT/'evidence').mkdir(exist_ok=True);acquire();a=freeze_check()
 write(ROOT/'evidence'/'ACTIVE_CONTROLLER.json',{**identity(),'script':str(Path(__file__).resolve()),'source_sha256':sha(__file__),'started_unix':time.time(),'deadline_unix':DEADLINE})
 threading.Thread(target=deadline_guard,daemon=True).start()
 try:
  assert time.time()<DEADLINE,'No optimizer update is authorized after deadline'
  stage('ENGINEERING_GATES');check_accelerators('BEFORE_GATES')
  for p,status in [(ROOT/'evidence/improved_cpu_math/CPU_MATH_ACCEPTANCE.json','PASS'),
                   (ROOT/'evidence/cpu_engineering/ENGINEERING_CPU_GATE.json','PASS_REQUIRED_TOY_GATES'),
                   (ROOT/'evidence/WORKER_CPU_SELFTEST.json','PASS')]:
   q=json.loads(p.read_text());assert q['status']==status,(p,q.get('status'))
  ref=gate('gate_ref8','original',ROOT/'gates/original_ref8',list(range(8)))
  wait_gates([ref]);check_accelerators('AFTER_REF8')
  g3=gate('gate_original3','original',ROOT/'gates/original3',ALLOCATION['original'])
  g5=gate('gate_improved5','improved',ROOT/'gates/improved5',ALLOCATION['improved'])
  wait_gates([g3,g5]);stage('CHECKPOINT_AND_MIGRATION_CPU_AUDIT')
  out=ROOT/'evidence'/'EIGHT_TO_THREE_ACCEPTANCE.json'
  cpu_command([ROOT/'audit_state.py','--compare-eight',latest(ref['out']),'--compare-three',latest(g3['out']),
   '--parent',a['original_checkpoint'],'--out',out],out)
  audit(latest(g5['out']),'IMPROVED_GATE')
  r3=gate('gate_original_resume','original',ROOT/'gates/original_resume',ALLOCATION['original'],steps=1,resume=latest(g3['out']))
  r5=gate('gate_improved_resume','improved',ROOT/'gates/improved_resume',ALLOCATION['improved'],steps=1,resume=latest(g5['out']))
  wait_gates([r3,r5]);audit(latest(r3['out']),'ORIGINAL_NEW_FORMAT_RESUME');audit(latest(r5['out']),'IMPROVED_NEW_FORMAT_RESUME')
  write(ROOT/'evidence'/'ENGINEERING_ACCEPTANCE.json',{'status':'PASS','unix':time.time(),
   'actual_eight_to_three':str(out),'gate_weights_discarded':True,'formal_original_parent':a['original_checkpoint'],
   'formal_improved_parent':a['official_SR_sha256'],'allocation':ALLOCATION,'effective_batch_both':96,
   'new_format_resume_actual_NPU_updates':1,'deadline_unix':DEADLINE})
  formal()
 except BaseException as exc:
  for j in JOBS.values():
   if j['process'].poll() is None:terminate_own_group(j,'controller failure cleanup')
  write(ROOT/'evidence'/f'CONTROLLER_FAILURE_{time.time_ns()}.json',{'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc(),'unix':time.time(),'controller':identity()})
  stage('ENGINEERING_FAILURE_REQUIRES_REPAIR',error=repr(exc));raise
 finally:
  write(ROOT/'evidence'/'CONTROLLER_EXIT.json',{**identity(),'unix':time.time(),'all_subprocesses_exited':all(j['process'].poll() is not None for j in JOBS.values())})

if __name__=='__main__':main()

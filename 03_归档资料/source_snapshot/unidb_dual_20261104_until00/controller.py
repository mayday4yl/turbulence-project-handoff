"""Persistent, scoped controller for the explicitly authorized 3+5 campaign.

No old run is restarted. Gates discard their updates. Deadline stopping also
exists independently in every worker. This controller never signals outsiders.
"""
import fcntl,hashlib,json,os,re,signal,subprocess,sys,threading,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'unidb_d0_20261102'
RUN_ENV=OLD/'run_env.sh'
DEADLINE=1793721600
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
 policy=json.loads((ROOT/'protocol/SHARED_CONTEXTS.json').read_text())
 allowed={(q['card'],q['pid']):q for q in policy['permitted_contexts']};found=[];usage=[]
 for line in raw.splitlines():
  m=re.match(r'^\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|',line)
  if m:
   pair=(int(m[1]),int(m[2]));assert pair in allowed,('unrecognized external occupant; no signal sent',pair)
   item=allowed[pair]
   if item.get('start_ticks') is not None:
    st=Path('/proc')/str(pair[1])/'stat';assert st.is_file() and st.read_text().rsplit(')',1)[1].split()[19]==item['start_ticks'],('external PID identity changed',pair)
   found.append(pair)
  m=re.search(r'(\d+)\s*/\s*(65536)\s*\|$',line)
  if m:usage.append({'used_MiB':int(m[1]),'total_MiB':int(m[2]),'free_MiB':int(m[2])-int(m[1])})
 assert sum('910B3' in x and 'OK' in x for x in raw.splitlines())==8
 assert len(usage)==8 and all(x['free_MiB']>=policy['min_free_HBM_MiB_before_launch'] for x in usage),('insufficient safe launch headroom',usage)
 write(ROOT/'evidence'/f'RESOURCE_ACCEPTANCE_{label}.json',{'status':'PASS','unix':time.time(),'external_contexts':found,'HBM':usage,
  'policy_sha256':sha(ROOT/'protocol/SHARED_CONTEXTS.json'),'external_processes_signalled':0,
  'limitation':'Shared accelerator compute may affect timing. Card1 driver context is not an OS-process liveness claim.'})


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

def _proc_snapshot():
 result={}
 for path in Path('/proc').glob('[0-9]*/stat'):
  try:
   raw=path.read_text();f=raw[raw.rfind(')')+2:].split();pid=int(path.parent.name)
   result[pid]={'pid':pid,'state':f[0],'ppid':int(f[1]),'pgid':int(f[2]),'sid':int(f[3]),'start_ticks':f[19]}
  except (OSError,ValueError,IndexError):pass
 return result

def cache_owned(job):
 """Cache only authenticated roots, their descendants and isolated sessions."""
 current=_proc_snapshot();record=job['record'];anchors=[record]
 for path in Path(job['out']).glob('RANK_*.json'):
  try:
   q=json.loads(path.read_text())
   if int(q['start_ticks'])>=int(record['start_ticks']):anchors.append(q)
  except (OSError,ValueError,KeyError,json.JSONDecodeError):continue
 history=job.setdefault('_owned_history',{});sessions=job.setdefault('_owned_sessions',set())
 owned={pid for pid,q in current.items() if f"{pid}:{q['start_ticks']}" in history}
 for q in anchors:
  pid=int(q['pid']);now=current.get(pid)
  if now and now['start_ticks']==str(q['start_ticks']):owned.add(pid)
 while True:
  before=set(owned)
  owned.update(pid for pid,q in current.items() if q['ppid'] in owned)
  sessions.update(current[pid]['sid'] for pid in owned if current[pid]['sid']==pid and current[pid]['pgid']==pid)
  pinned={current[pid]['sid'] for pid in owned}&sessions
  owned.update(pid for pid,q in current.items() if q['sid'] in pinned)
  if owned==before:break
 members={pid:current[pid] for pid in owned};pgids={q['pgid'] for q in members.values()}
 unsafe=[q for pid,q in current.items() if q['pgid'] in pgids and pid not in owned]
 ambiguous=[q for pid,q in current.items() if q['sid'] in {int(a['pid']) for a in anchors} and pid not in owned]
 if 2938958 in members:unsafe.append(members[2938958])
 for pid,q in members.items():history[f"{pid}:{q['start_ticks']}"]=q
 job['_owned_current']=members;job['_owned_unsafe']=unsafe+ambiguous
 evidence={'unix':time.time(),'job':record,'members':list(members.values()),'pgids':sorted(pgids),
  'unsafe_or_unverified_members':unsafe+ambiguous,'signals_sent':0}
 target=ROOT/'evidence'/f"ACTOR_OWNED_{record['pid']}_{record['start_ticks']}.json"
 write(target,evidence)
 return evidence

def terminate_own_group(job,why):
 p=job['process'];verified=cache_owned(job)
 assert not job['_owned_unsafe'],('Refuse ambiguous/reused/external process group',job['_owned_unsafe'])
 first=dict(job['_owned_current']);groups=sorted({q['pgid'] for q in first.values()})
 assert 2938958 not in first and all(g>1 for g in groups)
 snapshot=ROOT/'evidence'/f"CLEANUP_VERIFIED_{job['record']['pid']}_{time.time_ns()}.json"
 write(snapshot,{'why':why,'job':job['record'],'first_verified_members':list(first.values()),'groups':groups,'signals_sent':0})
 append('ACTOR_EVENTS.jsonl',{'event':'verified_own_cleanup_started','why':why,'snapshot':str(snapshot),'snapshot_sha256':sha(snapshot)})
 def signal_verified(pgid,signum):
  # Never extend the first verified set after TERM; PID reuse/new members fail closed.
  now=_proc_snapshot();members=[q for q in now.values() if q['pgid']==pgid]
  for q in members:
   old=first.get(q['pid'])
   assert q['pid']!=2938958 and old and all(q[k]==old[k] for k in ['start_ticks','pgid','sid']),('Refuse changed cleanup group membership',pgid,q)
  if any(q['state']!='Z' for q in members):
   append('ACTOR_EVENTS.jsonl',{'event':'signal_verified_own_group','pgid':pgid,'signal':int(signum),'members':members})
   try:os.killpg(pgid,signum)
   except ProcessLookupError:pass
 def remaining():
  now=_proc_snapshot();return [q for pid,q in now.items() if pid in first and q['start_ticks']==first[pid]['start_ticks'] and q['state']!='Z']
 for pgid in groups:signal_verified(pgid,signal.SIGTERM)
 end=time.monotonic()+30
 while remaining() and time.monotonic()<end:p.poll();time.sleep(1)
 if remaining():
  for pgid in groups:signal_verified(pgid,signal.SIGKILL)
  end=time.monotonic()+30
  while remaining() and time.monotonic()<end:p.poll();time.sleep(1)
 assert not remaining(),'Verified own descendants remain alive after scoped cleanup'
 try:p.wait(timeout=30)
 except subprocess.TimeoutExpired:raise RuntimeError('Own launcher failed to exit after scoped cleanup')
 append('ACTOR_EVENTS.jsonl',{'event':'verified_own_cleanup_complete','why':why,'snapshot':str(snapshot)})

def wait_gates(jobs,timeout=1500):
 active=[j for j in jobs if not j.get('reused')]
 start=time.time()
 while True:
  for j in active:
   cache_owned(j)
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

def seed_history(authority):
 import shutil
 previous=Path(authority['previous_campaign_root']);refs=[]
 for branch in ALLOCATION:
  source=previous/'runs'/branch;target=ROOT/'runs'/branch
  (target/'validation').mkdir(parents=True,exist_ok=True)
  for val in (source/'validation').glob('*_ema_full'):
   sp=val/'SUMMARY.json'
   if not sp.exists():continue
   q=json.loads(sp.read_text());assert q['status']=='PASS' and q['full'] and q['count']==5000
   link=target/'validation'/val.name
   if not link.exists():link.symlink_to(val,target_is_directory=True)
   assert link.resolve()==val.resolve()
   refs.append({'branch':branch,'validation':str(val),'summary_sha256':sha(sp),'update':q['update']})
  for name in ['BEST_FULL.json','BEST.json']:
   if (source/name).exists() and not (target/name).exists():shutil.copy2(source/name,target/name)
  cache=previous/'evidence/input_metric_cache'/branch
  if cache.exists():shutil.copytree(cache,ROOT/'evidence/input_metric_cache'/branch,dirs_exist_ok=True)
 for f in (previous/'evidence').glob('FINAL_OUTPUTS_CPU_AUDIT*'):
  if f.is_file() and not (ROOT/'evidence'/f.name).exists():shutil.copy2(f,ROOT/'evidence'/f.name)
 write(ROOT/'evidence/HISTORICAL_FULL_REUSE.json',{'status':'PASS','unix':time.time(),'refs':refs,
  'completed_full_candidates_only':True,'previous_assets_read_only':True,'new_model_calls':0,
  'partial_parent_evaluations_preserved_in_previous_root':True})


def formal():
 check_accelerators('BEFORE_FORMAL')
 authority=freeze_check();seed_history(authority)
 running={b:launch('formal_'+b,b,'formal',ROOT/'runs'/b,cards,resume=authority['extension_parents'][b]['path']) for b,cards in ALLOCATION.items()}
 attempts={b:0 for b in running};done={};stage('FORMAL_RUNNING',allocation=ALLOCATION,actors={b:j['record'] for b,j in running.items()})
 while running:
  snapshot={}
  for b,j in list(running.items()):
   cache_owned(j)
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
    stopfile=ROOT/'STOP_REQUEST.json'
    explicit_stop=stopfile.exists() and json.loads(stopfile.read_text()).get('reason')=='user_request'
    if time.time()<DEADLINE and not explicit_stop:
     attempts[b]+=1;assert attempts[b]<=2,('early completion recovery exhausted',b)
     archive_attempt(b,out,attempts[b],j);path=latest(out);audit(path,b+'_EARLY_COMPLETION_'+str(attempts[b]))
     running[b]=launch('recovery_early_'+b+'_'+str(attempts[b]),b,'formal',out,ALLOCATION[b],resume=path,eval_only=False)
     continue
    audit(latest(out),b+'_FINAL')
    best=json.loads((out/'BEST.json').read_text());audit(best['checkpoint']['path'],b+'_BEST')
    done[b]={'status':'PASS','exit_code':0,'result':q,'latest':json.loads((out/'LATEST.json').read_text()),'best':best}
    del running[b];append('ACTOR_EVENTS.jsonl',{'event':'formal_branch_complete','branch':b,'result':q})
   else:
    terminate_own_group(j,'failed formal actor cleanup');attempts[b]+=1;archive_attempt(b,out,attempts[b],j)
    assert attempts[b]<=2,('bounded recovery exhausted',b)
    path=latest(out);audit(path,b+'_RECOVERY_'+str(attempts[b]))
    eval_only=time.time()>=DEADLINE
    running[b]=launch('recovery_'+b+'_'+str(attempts[b]),b,'formal',out,ALLOCATION[b],resume=path,eval_only=eval_only)
  write(ROOT/'evidence'/'LIVE_ACTORS.json',{'unix':time.time(),'controller':identity(),'running':snapshot,'done':done})
  time.sleep(10)
 write(ROOT/'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json',{'status':'PASS','branches':done,'unix':time.time(),
  'delivery_pending':'Independent saved-output CPU metric audit, visual QA, local sync and final report remain required.'})
 stage('MODEL_VALIDATION_COMPLETE_CPU_QA_PENDING',branches={b:v['result'] for b,v in done.items()})

def main():
 (ROOT/'evidence').mkdir(exist_ok=True);acquire();a=freeze_check()
 write(ROOT/'evidence/ACTIVE_CONTROLLER.json',{**identity(),'script':str(Path(__file__).resolve()),'source_sha256':sha(__file__),'started_unix':time.time(),'deadline_unix':DEADLINE})
 threading.Thread(target=deadline_guard,daemon=True).start()
 try:
  assert time.time()<DEADLINE,'No optimizer update is authorized after deadline'
  handoff=json.loads((ROOT/'evidence/PARENT_ACTOR_HANDOFF.json').read_text());assert handoff['status']=='PASS_OWN_ACTORS_STOPPED_FOR_EXTENSION'
  for b,parent in a['extension_parents'].items():
   report=json.loads((ROOT/'evidence'/('PARENT_STATE_AUDIT_'+b+'.json')).read_text());q=report['checkpoint']
   assert report['status']=='PASS' and all(q[k]==parent[k] for k in ['path','sha256','update','cursor'])
  stage('EXTENSION_ENGINEERING_GATES');check_accelerators('BEFORE_EXTENSION_GATES')
  assert json.loads((ROOT/'evidence/EXTENSION_WORKER_CPU_SELFTEST.json').read_text())['status']=='PASS'
  assert json.loads((ROOT/'evidence/EXTENSION_AUDITOR_CPU_TESTS.json').read_text())['status']=='PASS'
  previous=Path(a['previous_campaign_root']);prior=previous/'evidence/ENGINEERING_ACCEPTANCE.json'
  assert json.loads(prior.read_text())['status']=='PASS'
  write(ROOT/'evidence/REUSED_PREVIOUS_ENGINEERING_GATES.json',{'status':'PASS','path':str(prior),'sha256':sha(prior),'new_eight_rank_gate_calls':0,'scope':'Same network/loss/DDP/3+5 layouts; previously accepted actual8-to3 and mathematical gates reused.'})
  jobs=[gate('gate_extension_'+b,b,ROOT/'gates'/b,ALLOCATION[b],steps=1,resume=a['extension_parents'][b]['path']) for b in ALLOCATION]
  wait_gates(jobs)
  for b in ALLOCATION:audit(latest(ROOT/'gates'/b),b+'_EXTENSION_RESTORE_GATE')
  resumed=[gate('gate_extension_restart_'+b,b,ROOT/'gates'/(b+'_restart'),ALLOCATION[b],steps=1,resume=latest(ROOT/'gates'/b)) for b in ALLOCATION]
  wait_gates(resumed)
  for b in ALLOCATION:audit(latest(ROOT/'gates'/(b+'_restart')),b+'_EXTENSION_RESTART_GATE')
  write(ROOT/'evidence/ENGINEERING_ACCEPTANCE.json',{'status':'PASS','unix':time.time(),'gate_weights_discarded':True,'formal_parents':a['extension_parents'],'allocation':ALLOCATION,'effective_batch_both':96,'new_real_restore_and_restart_gate_updates_each_branch':2,'deadline_unix':DEADLINE,'extension_lrs':a['extension_lrs']})
  formal()
 except BaseException as exc:
  for j in JOBS.values():
   if j['process'].poll() is None:terminate_own_group(j,'controller failure cleanup')
  write(ROOT/'evidence'/f'CONTROLLER_FAILURE_{time.time_ns()}.json',{'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc(),'unix':time.time(),'controller':identity()})
  stage('ENGINEERING_FAILURE_REQUIRES_REPAIR',error=repr(exc));raise
 finally:
  write(ROOT/'evidence/CONTROLLER_EXIT.json',{**identity(),'unix':time.time(),'all_subprocesses_exited':all(j['process'].poll() is not None for j in JOBS.values())})

if __name__=='__main__':main()

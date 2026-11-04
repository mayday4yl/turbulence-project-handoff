"""Bounded, resumable NAFNet gate orchestration; all GPU states are disposable."""
from pathlib import Path
import sys,os,json,time,subprocess,fcntl,signal,statistics,copy,itertools,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from runlib.common import *
from runlib.resources import check,inventory
F=R/'reports/preflight';F.mkdir(parents=True,exist_ok=True)
class GateError(RuntimeError):pass
def alive(pid,token):
 try:return Path('/proc',str(pid),'cwd').resolve()==R and token in Path('/proc',str(pid),'cmdline').read_bytes().decode().replace('\0',' ')
 except (FileNotFoundError,PermissionError):return False
def clock():
 path=F/'TIMEBOX.json'
 if not path.exists():atomic(path,{'start':time.time(),'deadline':time.time()+2700,'minutes':45})
 return read(path)['deadline']
def deadline():return read(F/'TIMEBOX.json')['deadline'] if (F/'TIMEBOX.json').exists() else float('inf')
def budget():
 if time.time()>=deadline():raise GateError('45-minute preflight deadline reached; incomplete gates forbid formal launch')
def job(kind,g,precision,B,micro,peak=1e-4,tag=''):
 return dict(kind=kind,group=g,precision=precision,batch=B,micro=micro,peak_lr=peak,tag=tag)
def location(j):return F/'jobs'/canonical(j)[:20]
def existing(j):
 base=location(j)
 for d in sorted(base.glob('attempt_*'),reverse=True):
  if (d/'RESULT.json').exists():
   r=read(d/'RESULT.json')
   if r['status']=='PASS':return r
   if r['status'] not in ['STARTING','RUNNING','WAIT_RESOURCE'] or alive(r['pid'],str(d/'JOB.json')):return r
 return None
def launch(j,gpu,co_pids=(),barrier=None):
 base=location(j);base.mkdir(parents=True,exist_ok=True);n=len(list(base.glob('attempt_*')))+1;out=base/f'attempt_{n:03d}';out.mkdir()
 full={**j,'gpu':gpu,'output':str(out),'deadline':clock(),'co_pids':list(co_pids)}
 if barrier:full['barrier']=str(barrier)
 path=out/'JOB.json';atomic(path,full)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 with (out/'worker.log').open('ab') as log:p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/preflight_job.py'),'--job',str(path)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 atomic(out/'PROCESS.json',{'pid':p.pid,'job':str(path),'time':time.time()});return out,p.pid
def stop_owned():
 for p in F.glob('jobs/*/attempt_*/PROCESS.json'):
  q=read(p)
  if alive(q['pid'],q['job']):os.kill(q['pid'],signal.SIGTERM)
def run(jobs):
 pending=list(jobs);running={};answer={}
 while pending or running:
  budget()
  for j in list(pending):
   key=canonical(j);r=existing(j)
   if r and r['status'] not in ['STARTING','RUNNING','WAIT_RESOURCE']:
    answer[key]=r;pending.remove(j);continue
   if r and alive(r['pid'],str(Path(r['job']['output'])/'JOB.json')):
    running[key]=(j,Path(r['job']['output']),r['pid']);pending.remove(j);continue
   for gpu in [3,5]:
    try:check(gpu)
    except RuntimeError:continue
    # Include children still constructing CPU assets, not yet visible to nvidia-smi.
    if any(read(d/'JOB.json')['gpu']==gpu for _,d,_ in running.values()):continue
    out,pid=launch(j,gpu);running[key]=(j,out,pid);pending.remove(j);break
  for key,(j,out,pid) in list(running.items()):
   if alive(pid,str(out/'JOB.json')):continue
   r=read(out/'RESULT.json') if (out/'RESULT.json').exists() else {'status':'ERROR','error':'Worker exited without result'}
   if r['status']=='WAIT_RESOURCE':pending.append(j)
   elif r['status'] in ['STARTING','RUNNING']:raise GateError('Disposable worker exited without a committed result')
   else:answer[key]=r
   del running[key]
  atomic(F/'STATUS.json',{'status':'PREFLIGHT_RUNNING_OR_WAIT_RESOURCE','pending':pending,'running':{k:{'pid':p,'job':j} for k,(j,d,p) in running.items()},'deadline':None if deadline()==float('inf') else deadline(),'time':time.time()})
  if pending or running:time.sleep(3)
 return [answer[canonical(j)] for j in jobs]
def require(r):
 if r['status']!='PASS':raise GateError(json.dumps({'failed_gate':r.get('job'),'status':r['status'],'error':r.get('error')}))
def benchmark(g,precision,B,micro,tag='exclusive'):
 while True:
  j=job('benchmark',g,precision,B,micro,tag=tag);r=run([j])[0]
  if r['status']=='OOM' and micro>1:micro//=2;continue
  require(r);return r,micro
def makespan(durations):
 slots=[0.,0.];order=sorted(GROUPS,key=lambda g:durations[g],reverse=True);mapping={}
 for g in order:
  i=min(range(2),key=lambda x:slots[x]);slots[i]+=durations[g];mapping[g]=[3,5][i]
 return max(slots),order,mapping
def co_test(groups,precision,B,micros):
 # A single representative pair, synchronized after both models are constructed.
 tag='representative_pair';js=[job('benchmark',g,precision,B,max(1,micros[g]//2),tag=tag) for g in groups]
 cached=[existing(j) for j in js]
 if all(r and r['status']=='PASS' for r in cached):return cached
 if any(r for r in cached):raise GateError('Partial co-residency attempt requires diagnosis; never silently test a second combination')
 while True:
  budget();free=[]
  for gpu in [3,5]:
   try:check(gpu);free.append(gpu)
   except RuntimeError:pass
  if free:break
  time.sleep(5)
 gpu=free[0];barrier=F/'CORESIDENT_GO.json';first,pid1=launch(js[0],gpu,barrier=barrier)
 while not (first/'READY.json').exists():
  budget()
  if not alive(pid1,str(first/'JOB.json')):return [read(first/'RESULT.json')]
  time.sleep(1)
 second,pid2=launch(js[1],gpu,co_pids=[pid1],barrier=barrier)
 while not (second/'READY.json').exists():
  budget()
  if not alive(pid2,str(second/'JOB.json')):
   if alive(pid1,str(first/'JOB.json')):os.kill(pid1,signal.SIGTERM)
   return [read(second/'RESULT.json')]
  time.sleep(1)
 atomic(barrier,{'go':True,'time':time.time(),'pair':groups})
 while alive(pid1,str(first/'JOB.json')) or alive(pid2,str(second/'JOB.json')):budget();time.sleep(2)
 return [read(x/'RESULT.json') for x in [first,second]]
def main():
 lock=(F/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (R/'protocol/LAUNCH_GATE.json').exists():return
 assert read(R/'reports/CPU_PROTOCOL_ACCEPTANCE.json')['status']=='PASS'
 cpu=read(R/'reports/CPU_ENGINE_ACCEPTANCE.json');assert cpu['status']=='PASS' and cpu['source_hashes']==source_hashes()
 environment=runtime_environment()
 # Tolerances are fixed before examining preflight results, not tuned to pass.
 policy={'loss_relative_tolerance':.05,'gradient_norm_ratio':[.25,4.],'sampled_gradient_cosine_min':.9,'BF16_min_speed_ratio':1.03,'VRAM_fraction_max':.85,'B64_min_throughput_ratio':1.10,'co_makespan_min_gain':.05,'resume_rtol':1e-4,'resume_atol':1e-6}
 freeze(F/'GATE_POLICY.json',policy)
 precision_jobs=[job('precision',g,p,32,1) for g in GROUPS for p in ['fp32','bf16']];precision_results=run(precision_jobs)
 for r in precision_results:
  if r['job']['precision']=='fp32' or r['status'] not in ['PASS','NUMERICAL_FAILURE','OOM']:require(r)
 import torch
 pairs={};use_bf16=True
 for i,g in enumerate(GROUPS):
  a,b=precision_results[2*i:2*i+2]
  if b['status']!='PASS':use_bf16=False;pairs[g]={'PASS':False,'reason':b['status']};continue
  assert a['fixed_batch']==b['fixed_batch']
  ga=torch.load(Path(a['job']['output'])/'gradient_sample.pt',weights_only=True);gb=torch.load(Path(b['job']['output'])/'gradient_sample.pt',weights_only=True)
  va=torch.cat([ga[k] for k in sorted(ga)]).double();vb=torch.cat([gb[k] for k in sorted(gb)]).double();cos=float(torch.dot(va,vb)/(va.norm()*vb.norm()).clamp_min(1e-30))
  ratio=b['gradient_norm']/max(a['gradient_norm'],1e-30);err=abs(b['loss']-a['loss'])/max(abs(a['loss']),1e-30)
  ok=err<=.05 and .25<=ratio<=4 and cos>=.9;use_bf16 &= ok;pairs[g]={'loss_relative_error':err,'gradient_norm_ratio':ratio,'sampled_gradient_cosine':cos,'PASS':ok}
 speed=sum(precision_results[i]['seconds'] for i in range(0,8,2))/sum(precision_results[i]['seconds'] for i in range(1,8,2)) if all(r['status']=='PASS' for r in precision_results) else 0.;use_bf16 &= speed>=1.03;precision='bf16' if use_bf16 else 'fp32'
 atomic(F/'PRECISION_DECISION.json',{'selected':precision,'diagnostics':pairs,'BF16_speed_ratio':speed,'policy':policy})
 # Separate throughput jobs: two warmup and five timed effective updates.
 baseline_jobs=[job('benchmark',g,precision,32,4,tag='exclusive') for g in GROUPS];br=run(baseline_jobs);baseline={};micros={}
 for g,r in zip(GROUPS,br):
  if r['status']=='OOM':r,micro=benchmark(g,precision,32,2)
  else:require(r);micro=4
  if not r['safe_memory']:
   r,micro=benchmark(g,precision,32,max(1,micro//2),tag='safety_margin')
  assert r['safe_memory'];baseline[g]=r;micros[g]=micro
 heavy=max(['R0','R08','L08'],key=lambda g:baseline[g]['seconds_per_update'])
 b64=run([job('benchmark',g,precision,64,micros[g],tag='batch64') for g in ['D0',heavy]])
 choose64=all(r['status']=='PASS' and r['safe_memory'] and r['starts_per_second']>=1.1*baseline[g]['starts_per_second'] for g,r in zip(['D0',heavy],b64))
 for r in b64:
  if r['status'] not in ['PASS','OOM']:require(r)
 B=64 if choose64 else 32
 atomic(F/'BATCH_DECISION.json',{'effective_batch':B,'heaviest_recursive':heavy,'B32':baseline,'B64':b64,'rule':'B64 only when both representative cases safe and >=10% faster in starts/sec'})
 peak=1e-4
 for trial in [1e-4,5e-5]:
  peak=trial;lr_jobs=[job('lr',g,precision,B,micros[g],peak) for g in GROUPS];lr=run(lr_jobs)
  for i,r in enumerate(lr):
   while r['status']=='OOM' and micros[GROUPS[i]]>1:
    g=GROUPS[i];micros[g]//=2;r=run([job('lr',g,precision,B,micros[g],peak)])[0];lr[i]=r
  non_lr_failures=[r for r in lr if r['status'] not in ['PASS','LR_NUMERICAL_FAILURE']]
  if non_lr_failures:require(non_lr_failures[0])
  if all(r['status']=='PASS' for r in lr):break
  if trial==5e-5:raise GateError('Both authorized peak learning rates failed numerical gates')
 assert all(r['status']=='PASS' and r['effective_optimizer_updates']>=10 and r['resume_exact_state'] for r in lr)
 atomic(F/'LR_DECISION.json',{'peak_lr':peak,'results':lr,'four_method_uniform':True,'tested_actual_peak_not_warmup':True})
 # For B64, time other methods at the final scientific effective batch as well.
 if B==64:
  finalbench={g:r for g,r in zip(['D0',heavy],b64)}
  for g in GROUPS:
   if g not in finalbench:finalbench[g],micros[g]=benchmark(g,precision,B,micros[g],tag='final_batch')
 else:finalbench=baseline
 other=next(g for g in ['R08','R0','L08'] if g!=heavy);pair=[heavy,other]
 co=co_test(pair,precision,B,micros)
 durations={g:finalbench[g]['seconds_per_update']*(640000//B) for g in GROUPS};single_span,order,mapping=makespan(durations)
 mode='one_task_per_GPU';co_span=None
 if len(co)==2 and all(r['status']=='PASS' and r['safe_memory'] for r in co):
  slow=max(r['seconds_per_update']/finalbench[g]['seconds_per_update'] for g,r in zip(pair,co))
  pairs2=[['R0','D0'],['R08','L08']];co_span=max(max(durations[g]*slow for g in gs) for gs in pairs2)
  # Conservative total memory bound from both models' measured allocator peaks.
  safe=sum(r['peak_reserved_bytes'] for r in co)<.85*min(r['total_memory'] for r in co)
  if safe and co_span<=.95*single_span:
   mode='two_tasks_per_GPU';mapping={g:gpu for gpu,gs in zip([3,5],pairs2) for g in gs};micros={g:max(1,micros[g]//2) for g in GROUPS}
 elif any(r['status'] not in ['PASS','OOM'] for r in co):require(next(r for r in co if r['status'] not in ['PASS','OOM']))
 scheduling={'mode':mode,'order':order,'preferred_gpu':mapping,'exclusive_seconds':durations,'predicted_two_GPU_training_only_seconds':co_span if mode=='two_tasks_per_GPU' else single_span,'co_representative_results':co,'capacity_available_at_launch_must_be_rechecked':True,'evaluation_cost_excluded':True}
 atomic(F/'SCHEDULING_DECISION.json',scheduling)
 budget();c=read(R/'configs/PREFLIGHT_TEMPLATE.json');c.update(precision=precision,effective_batch=B,max_updates=640000//B,peak_lr=peak,microbatch=micros,scheduling=scheduling,phase='FORMAL_PHASE1',purpose='Official pretrained all-parameter NAFNet V4 Phase-1',formal_training_authorized=True,method='V4_NAFNET_SIDD116M_PHASE1')
 eval_peak=max(r['peak_reserved_bytes'] for r in precision_results if r['job']['precision']==precision)+(1<<30)
 c['evaluation_policy']={'network_batch':1,'tail':'single_sample','refinement':False,'primary_weight':'online','secondary_weight':'EMA_final_only','eta_namespace':'evaluation','co_resident_permitted':len(co)==2 and all(r['status']=='PASS' for r in co),'eval_peak_upper_bound_bytes':eval_peak,'training_peak_bytes':{g:finalbench[g]['peak_reserved_bytes'] for g in GROUPS},'TF32':False,'pending_jobs_do_not_gate_training':True}
 freeze(R/'configs/FROZEN_RUNTIME.json',c)
 gate={'status':'PASS','config_sha256':sha(R/'configs/FROZEN_RUNTIME.json'),'source_hashes':source_hashes(),'science_hashes':science_hashes(),'environment':environment,'common_init_sha256':sha(R/'protocol/common_init.pt'),'CPU_protocol_sha256':sha(R/'reports/CPU_PROTOCOL_ACCEPTANCE.json'),'CPU_engine_sha256':sha(R/'reports/CPU_ENGINE_ACCEPTANCE.json'),'required_preflight_reports':{p.name:sha(p) for p in F.glob('*DECISION.json')},'timebox':read(F/'TIMEBOX.json'),'time':time.time(),'formal_restart_from_common_U0_required':True}
 freeze(R/'protocol/LAUNCH_GATE.json',gate);atomic(F/'STATUS.json',{'status':'ALL_GATES_PASS','time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:
  stop_owned();atomic(F/'STATUS.json',{'status':'FORMAL_LAUNCH_BLOCKED','traceback':traceback.format_exc(),'time':time.time()});raise

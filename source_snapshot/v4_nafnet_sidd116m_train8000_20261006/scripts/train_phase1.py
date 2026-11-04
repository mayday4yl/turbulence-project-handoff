"""One formal method, fresh common U0 or strictly its own committed checkpoint."""
from pathlib import Path
import sys,os,argparse,time,signal,fcntl,copy,random,traceback,concurrent.futures,statistics
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from runlib.common import *
p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--micro',type=int,required=True);p.add_argument('--co-pids',default='');a=p.parse_args()
RUN=R/'runs/phase1'/a.group;RUN.mkdir(parents=True,exist_ok=True);STOP=False;u=0;attempting=0
def stop(*_):
 global STOP
 STOP=True
def status(s,**kw):atomic(RUN/'STATUS.json',{'status':s,'group':a.group,'update':u,'attempting_update':attempting,'pid':os.getpid(),'physical_gpu':a.gpu,'microbatch':a.micro,'time':time.time(),**kw})
def main():
 global u,attempting
 lock=(RUN/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from runlib.launch import verify,latest
 from runlib.resources import claim
 c,gate=verify()
 from runlib.operational_revision import microbatch_ceiling
 assert c['effective_batch']%a.micro==0 and a.micro<=microbatch_ceiling(c,a.group)
 assert not (RUN/'STOP.json').exists() and not (R/'protocol/STOP_FORMAL.json').exists()
 status('STARTING');device=claim(a.gpu,allowed_copids=[int(x) for x in a.co_pids.split(',') if x])
 import torch,numpy as np
 torch.set_num_threads(4);random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006)
 from runlib.models import build
 from runlib.data import Store,Stream
 from runlib.engine import step
 from runlib.checkpoints import save,restore,binding
 from v4.refinement import module_hash
 store=Store(c);stream=Stream(store,c['effective_batch']);identity=binding(c,store,stream)
 model,sim,obj,opt=build(c,a.group,device);ema=copy.deepcopy(model).eval().requires_grad_(False);model.train();sim.eval()
 path,info=latest(RUN,identity);health={'first_losses':[],'consecutive_high':0}
 if path:
  u,saved=restore(path,model,ema,opt,a.group,c,stream,identity);health=saved['training_health'];assert health is not None;del saved
  previous=read(RUN/'LAST_ATTEMPT.json') if (RUN/'LAST_ATTEMPT.json').exists() else {}
  append(RUN/'recovery.jsonl',{'restored_update':u,'checkpoint':str(path),'checkpoint_sha256':info['sha256'],'previous_observed_update':previous.get('last_observed_update'),'uncommitted_update_rollback':max(0,previous.get('last_observed_update',u)-u),'next_sample_ids':stream.state(u)['next_sample_ids'],'microbatch':a.micro,'time':time.time()})
 else:
  assert not list((RUN/'checkpoints').glob('*.identity.json'))
  assert module_hash(model)==read(R/'protocol/COMMON_INITIALIZATION.json')['model_hash'] and not opt.state
  freeze(RUN/'U0_ORIGIN.json',{'common_init_sha256':identity['common_init'],'model_hash':module_hash(model),'optimizer_state_count':0,'update':0,'exposures':0,'EMA_equals_common_init':module_hash(ema)==module_hash(model),'preflight_weights_imported':False,'RNG_namespace':'formal_20261006','stream_identity':stream.identity})
  save(RUN/'checkpoints/full_U00000.pt',model,ema,opt,a.group,c,stream,0,a.micro,identity,training_health=health)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 logger=concurrent.futures.ThreadPoolExecutor(max_workers=1);pending=[];last_checkpoint=u;last=None
 def persist():
  nonlocal last_checkpoint
  result=save(RUN/'checkpoints'/f'full_U{u:05d}.pt',model,ema,opt,a.group,c,stream,u,a.micro,identity,training_health=health);atomic(RUN/'LATEST.json',result);last_checkpoint=u
  if u and u%1000==0:
   kinds=['monitor']+(['fullval'] if u*c['effective_batch']%128000==0 else [])
   for kind in kinds:
    job={'group':a.group,'update':u,'exposures':u*c['effective_batch'],'kind':kind,'checkpoint':result['path'],'checkpoint_sha256':result['sha256'],'weight':'online','refinement':False,'config_sha256':gate['config_sha256']}
    freeze(R/'evaluation/queue'/f'{a.group}_U{u:05d}_{kind}_online.json',job)
   if u==c['max_updates']:
    job={'group':a.group,'update':u,'exposures':u*c['effective_batch'],'kind':'fullval','checkpoint':result['path'],'checkpoint_sha256':result['sha256'],'weight':'ema','refinement':False,'config_sha256':gate['config_sha256']}
    freeze(R/'evaluation/queue'/f'{a.group}_U{u:05d}_fullval_ema.json',job)
 while u<c['max_updates']:
  if STOP or (RUN/'STOP.json').exists() or (R/'protocol/STOP_FORMAL.json').exists():break
  attempting=u+1;status('RUNNING',exposures=u*c['effective_batch'],last_checkpoint=last_checkpoint,last=last)
  last=step(model,obj,opt,store,stream.rows(u),c,u+1,a.micro,ema,namespace='formal',audit=(u+1)%20==0);u+=1
  if len(health['first_losses'])<5:health['first_losses'].append(last['total_loss'])
  else:
   baseline=statistics.median(health['first_losses']);health['consecutive_high']=health['consecutive_high']+1 if last['total_loss']>3*baseline else 0
   if health['consecutive_high']>=5:raise FloatingPointError('Numerical loss explosion: five consecutive effective-batch losses exceed three times first-five median')
  if u%20==0:
   row={'group':a.group,'update':u,'exposures':u*c['effective_batch'],'epoch':u*c['effective_batch']/128000,'loss_population':'this_optimizer_effective_training_batch','microbatch':a.micro,'accumulation':c['effective_batch']//a.micro,'time':time.time(),**last}
   pending=[f for f in pending if not f.done()];pending.append(logger.submit(append,RUN/'scalars.jsonl',row))
  if u%1000==0:persist()
 if u!=last_checkpoint:persist()
 for f in pending:f.result()
 logger.shutdown(wait=True)
 if u==c['max_updates']:
  atomic(RUN/'COMPLETE.json',{'status':'PHASE1_TRAINING_COMPLETE','update':u,'exposures':u*c['effective_batch'],'checkpoint':read(RUN/'LATEST.json'),'evaluation_pending_is_separate':True,'time':time.time()});status('PHASE1_TRAINING_COMPLETE',exposures=u*c['effective_batch'])
 else:status('GRACEFUL_STOPPED',exposures=u*c['effective_batch'],last_checkpoint=last_checkpoint)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  msg=str(e).lower();kind='OOM' if 'out of memory' in msg else ('WAIT_RESOURCE' if 'wait_resource' in msg else ('NUMERICAL_FAILURE' if isinstance(e,FloatingPointError) or any(x in msg for x in ['nonfinite','non-finite','nan']) else ('PROTOCOL_FAILURE' if isinstance(e,(AssertionError,ValueError)) else 'INFRA_FAILURE')))
  status(kind,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());atomic(RUN/'LAST_ATTEMPT.json',{'last_observed_update':u,'attempting_update':attempting,'kind':kind,'time':time.time()});raise

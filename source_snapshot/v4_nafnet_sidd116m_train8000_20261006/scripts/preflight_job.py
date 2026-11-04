"""One disposable GPU job, with fixed peak LR distinct from throughput timing."""
from pathlib import Path
import argparse,os,sys,time,json,copy,statistics,traceback,random
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from runlib.common import *
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();j=read(a.job);D=Path(j['output']);D.mkdir(parents=True,exist_ok=True)
def status(s,**kw):atomic(D/'RESULT.json',{'status':s,'job':j,'pid':os.getpid(),'time':time.time(),**kw})
def budget():
 if time.time()>=j['deadline']:raise TimeoutError('PREFLIGHT_45_MINUTE_LIMIT')
def main():
 budget();status('STARTING')
 from runlib.resources import claim
 device=claim(j['gpu'],allowed_copids=j.get('co_pids',[]))
 import torch,numpy as np
 torch.set_num_threads(4);torch.manual_seed(20261006);np.random.seed(20261006);random.seed(20261006)
 from runlib.data import Store,Stream,eta_batch
 from runlib.models import build
 from runlib.engine import step,gradnorm
 from runlib.checkpoints import save,restore,binding
 from runlib.fixtures import sample_tensors,tree_equal,close
 from npu_runtime.backend import capture_rng
 c=read(R/'configs/PREFLIGHT_TEMPLATE.json');c.update(precision=j['precision'],effective_batch=j['batch'],max_updates=640000//j['batch'],peak_lr=j['peak_lr'],probe_constant_lr=True)
 store=Store(c);stream=Stream(store,c['effective_batch']);model,sim,obj,opt=build(c,j['group'],device);model.train();sim.eval();budget()
 if j['kind']=='precision':
  # Same eight G/domain strata, full native resolution, same eta and U0.
  rows=stream.rows(0)[:8];opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic();loss=[]
  for row in rows:
   budget();C,Y,G=store.batch([row],device);out=obj(C,Y,G,eta=eta_batch([row],device,0,'precision'))
   (out['loss_mean']/8).backward();loss.append(out['loss_mean'].detach());del out,C,Y,G
  norm=gradnorm(model.parameters());assert torch.isfinite(norm),'nonfinite precision gradient'
  assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters()),'nonfinite or missing precision gradient'
  torch.cuda.synchronize();duration=time.monotonic()-start
  grad=sample_tensors({n:q.grad for n,q in model.named_parameters()});torch.save(grad,D/'gradient_sample.pt')
  status('PASS',fixed_batch=[r['start_id'] for r in rows],loss=float(torch.stack(loss).mean()),gradient_norm=float(norm),seconds=duration,gradient_sample_sha256=sha(D/'gradient_sample.pt'),peak_reserved_bytes=torch.cuda.max_memory_reserved(),optimizer_updates=0);return
 ema=copy.deepcopy(model).eval().requires_grad_(False);history=[]
 if j.get('barrier'):
  atomic(D/'READY.json',{'pid':os.getpid(),'time':time.time()})
  while not Path(j['barrier']).exists():budget();time.sleep(.2)
 total=10 if j['kind']=='lr' else 7
 for u in range(1,total+1):
  budget();begin=time.time();m=step(model,obj,opt,store,stream.rows(u-1),c,u,j['micro'],ema,peak_probe=True,namespace='preflight_fixed',audit=j['kind']=='lr');m.update(start_time=begin,end_time=time.time())
  history.append(m);append(D/'updates.jsonl',{'update':u,**m});status('RUNNING',completed_updates=u,last=m)
 losses=[x['total_loss'] for x in history]
 if j['kind']=='lr' and all(x>3*statistics.median(losses[:5]) for x in losses[5:10]):
  status('LR_NUMERICAL_FAILURE',reason='Five consecutive losses exceed three times first-five median',updates=history);return
 if j['kind']=='lr':
  # Check exact persisted state, then compare the identical next effective batch.
  budget();identity=binding(c,store,stream);path=D/'resume_U00010.pt'
  save(path,model,ema,opt,j['group'],c,stream,10,j['micro'],identity)
  expected_rng=capture_rng(device=device)
  m1=step(model,obj,opt,store,stream.rows(10),c,11,j['micro'],ema,peak_probe=True,namespace='preflight_fixed',audit=True)
  after1=sample_tensors(model.state_dict());ema1=sample_tensors(ema.state_dict());rng1=capture_rng(device=device)
  u,saved=restore(path,model,ema,opt,j['group'],c,stream,identity)
  assert u==10 and tree_equal(model.state_dict(),saved['model']) and tree_equal(ema.state_dict(),saved['ema']) and tree_equal(opt.state_dict(),saved['optimizer']) and tree_equal(capture_rng(device=device),expected_rng)
  budget();m2=step(model,obj,opt,store,stream.rows(10),c,11,j['micro'],ema,peak_probe=True,namespace='preflight_fixed',audit=True)
  for key in ['L_C','L_R','total_loss']:assert np.isclose(m1[key],m2[key],rtol=1e-4,atol=1e-6),(key,m1[key],m2[key])
  assert close(after1,sample_tensors(model.state_dict())) and close(ema1,sample_tensors(ema.state_dict())) and tree_equal(rng1,capture_rng(device=device))
  # Exercise the exact metric implementation on a training-only fixture.
  env=read(CANDIDATE/'protocol/ENVIRONMENT.json');sys.path.insert(0,env['metrics_dependency_dir']);os.environ['TORCH_HOME']=env['TORCH_HOME']
  from nafnet_v4.metrics import Metrics
  model.eval();C,Y,G=store.batch([stream.rows(0)[0]],device)
  with torch.no_grad():out=obj(C,Y,G,eta=eta_batch([stream.rows(0)[0]],device,0,'evaluation'));metric=Metrics()(out['final_clean'],C)[0]
  status('PASS',updates=history,actual_peak_lr_tested=c['peak_lr'],effective_optimizer_updates=10,resume_exact_state=True,resume_next_batch_tolerance={'rtol':1e-4,'atol':1e-6},resume_next_sample_ids=stream.state(10)['next_sample_ids'],metrics=metric,formal_state_discarded=True)
 else:
  # Whole-state finite audit is outside timed optimizer intervals; heavy LR
  # branch diagnostics must not manufacture an artificial B64 speed advantage.
  assert all(torch.isfinite(p).all() for p in model.parameters()),'nonfinite benchmark model state'
  assert all(torch.isfinite(v).all() for st in opt.state.values() for v in st.values() if torch.is_tensor(v)),'nonfinite benchmark optimizer state'
  timed=history[2:];seconds=sum(x['update_seconds'] for x in timed);mem=max(x['peak_reserved_bytes'] for x in history);capacity=torch.cuda.get_device_properties(0).total_memory
  status('PASS',updates=history,warmup_updates=2,timed_updates=5,starts_per_second=5*j['batch']/seconds,seconds_per_update=seconds/5,peak_reserved_bytes=mem,total_memory=capacity,safe_memory=mem<.85*capacity,formal_state_discarded=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  msg=str(e);numerical=any(s in msg.lower() for s in ['nonfinite','non-finite','nan','inf gradient'])
  kind='OOM' if 'out of memory' in msg.lower() else ('TIMEBOX_EXPIRED' if isinstance(e,TimeoutError) else ('WAIT_RESOURCE' if 'WAIT_RESOURCE' in msg else (('LR_NUMERICAL_FAILURE' if j['kind']=='lr' else 'NUMERICAL_FAILURE') if numerical else 'ERROR')))
  status(kind,error_type=type(e).__name__,error=msg,traceback=traceback.format_exc());raise

"""Independent CPU/gloo toy gates for the dual-run engineering only.

No original checkpoint is loaded, no 137M network is instantiated, and no NPU
is used. Toy Adam updates test distributed weighting/checkpoint semantics.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import timedelta

HERE=Path(__file__).resolve().parent
OUT=HERE/'evidence'/'cpu_engineering'
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(mode):
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch.nn.parallel import DistributedDataParallel as DDP
 from ema_pytorch import EMA
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
 dist.init_process_group('gloo',timeout=timedelta(seconds=35))
 assert world==(5 if mode=='candidate5' else 3)
 class Net(torch.nn.Module):
  def __init__(self):
   super().__init__();self.a=torch.nn.Linear(7,32);self.b=torch.nn.Linear(32,5)
  def forward(self,x):return self.b(torch.nn.functional.gelu(self.a(x)))
 class Obj(torch.nn.Module):
  def __init__(self,net):super().__init__();self.network=net
  def forward(self,x,y,dummy=False):
   if dummy:
    return sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
   pred=self.network(x)
   return torch.nn.functional.mse_loss(pred,y) if mode=='candidate5' else torch.nn.functional.l1_loss(pred,y)
 def seedstate(lid):
  a=torch.Generator().manual_seed(90113+lid*173);b=torch.Generator().manual_seed(94277+lid*199)
  torch.randint(1,100,(lid+1,),generator=a);torch.randn((lid+3,),generator=b)
  return {'tau':a.get_state(),'eps':b.get_state()}
 def draw(lid,step,state):
  # Simulate independent CPU-timestep and accelerator-epsilon logical streams.
  a=torch.Generator();a.set_state(state['tau']);b=torch.Generator();b.set_state(state['eps'])
  tau=torch.randint(1,100,(12,1),generator=a);eps=torch.randn((12,7),generator=b)
  pos=torch.arange(step*96+lid*12,step*96+(lid+1)*12,dtype=torch.float32)[:,None]
  x=torch.sin(pos*.017+torch.arange(7)[None]*.11)+.05*eps
  y=torch.cos(pos*.023+torch.arange(5)[None]*.13)+tau.float()*.001
  return x,y,{'tau':a.get_state(),'eps':b.get_state()}
 def deterministic(step):
  q=torch.arange(step*96,(step+1)*96,dtype=torch.float32)[:,None]
  return torch.sin(q*.017+torch.arange(7)[None]*.11),5*torch.cos(q*.023+torch.arange(5)[None]*.13)
 def optimizer(net):return torch.optim.Adam(net.parameters(),lr=1e-3,betas=(.9,.99),eps=1e-8,weight_decay=0.)
 def ema_for(net):return EMA(net,beta=.995,update_every=10)
 torch.manual_seed(112233);base=Net();baseopt=optimizer(base);baseema=ema_for(base)
 # Meaningful pre-existing Adam moments, plus a mature EMA state for old branch.
 for j in range(3):
  x,y=deterministic(j);baseopt.zero_grad(set_to_none=True)
  torch.nn.functional.mse_loss(base(x),y).backward();baseopt.step();baseema.update()
 if mode!='candidate5':baseema.step.fill_(45411);baseema.initted.fill_(True)
 initial={'net':copy.deepcopy(base.state_dict()),'optimizer':copy.deepcopy(baseopt.state_dict()),'ema':copy.deepcopy(baseema.state_dict())}
 ref=Net();ref.load_state_dict(initial['net']);refopt=optimizer(ref);refopt.load_state_dict(initial['optimizer']);refema=ema_for(ref);refema.load_state_dict(initial['ema'])
 net=Net();net.load_state_dict(initial['net']);opt=optimizer(net);opt.load_state_dict(initial['optimizer']);ema=ema_for(net);ema.load_state_dict(initial['ema'])
 # Deep-copy optimizer state per arm; load_state_dict may share CPU tensor storage.
 refopt.load_state_dict(copy.deepcopy(initial['optimizer']));opt.load_state_dict(copy.deepcopy(initial['optimizer']))
 ddp=DDP(Obj(net),broadcast_buffers=False,find_unused_parameters=False,bucket_cap_mb=.001)
 lids=[r for r in range(8) if r%3==rank] if mode!='candidate5' else []
 states={lid:seedstate(lid) for lid in lids};refstates={lid:seedstate(lid) for lid in range(8)}
 tolerance={'atol':2e-6,'rtol':2e-5};max_gradient=0.;max_parameter=0.;max_optimizer=0.;max_ema=0.;dummy_rng_checks=0
 def compare(a,b,name):
  nonlocal max_optimizer,max_ema
  if isinstance(a,torch.Tensor):
   assert a.shape==b.shape and a.dtype==b.dtype,(name,a.shape,b.shape,a.dtype,b.dtype)
   if not a.is_floating_point() and not a.is_complex():
    assert torch.equal(a,b),(name,'nonfloating mismatch')
    return
   err=float((a-b).abs().max()) if a.numel() else 0.
   assert torch.allclose(a,b,**tolerance),(name,err)
   if name.startswith('optimizer'):max_optimizer=max(max_optimizer,err)
   if name.startswith('ema'):max_ema=max(max_ema,err)
  elif isinstance(a,dict):
   assert set(a)==set(b),name
   for k in a:compare(a[k],b[k],name+'.'+str(k))
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b),name
   for i,(x,y) in enumerate(zip(a,b)):compare(x,y,name+'.'+str(i))
  else:assert a==b,(name,a,b)
 for step in range(12):
  opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
  # Independently calculated serial reference for global batch96.
  if mode=='candidate5':
   x,y=deterministic(step);torch.nn.functional.mse_loss(ref(x),y).backward()
  else:
   for lid in range(8):
    x,y,refstates[lid]=draw(lid,step,refstates[lid]);(torch.nn.functional.l1_loss(ref(x),y)/8).backward()
  if mode=='candidate5':
   sizes=[20,19,19,19,19];start=sum(sizes[:rank]);splits=[10,sizes[rank]-10];x,y=deterministic(step)
   offset=start
   for slot,n in enumerate(splits):
    cm=ddp.no_sync() if slot==0 else contextlib.nullcontext()
    with cm:(ddp(x[offset:offset+n],y[offset:offset+n])*(n*5/96)).backward()
    offset+=n
  else:
   slots=len(lids) if mode=='old3_unequal' else 3
   for slot in range(slots):
    cm=ddp.no_sync() if slot<slots-1 else contextlib.nullcontext()
    with cm:
     if slot<len(lids):
      lid=lids[slot];x,y,states[lid]=draw(lid,step,states[lid]);loss=ddp(x,y)*(3/8)
     else:
      before=torch.get_rng_state().clone();loss=ddp(None,None,dummy=True)
      assert torch.equal(before,torch.get_rng_state());dummy_rng_checks+=1
     loss.backward()
  for p,q in zip(net.parameters(),ref.parameters()):
   err=float((p.grad-q.grad).abs().max());max_gradient=max(max_gradient,err)
   assert torch.allclose(p.grad,q.grad,**tolerance),('gradient',mode,rank,step,err)
  if mode=='candidate5':
   norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);refnorm=torch.nn.utils.clip_grad_norm_(ref.parameters(),1.0)
   assert torch.allclose(norm,refnorm,**tolerance)
  opt.step();ema.update();refopt.step();refema.update()
  for p,q in zip(net.parameters(),ref.parameters()):
   err=float((p-q).abs().max());max_parameter=max(max_parameter,err);assert torch.allclose(p,q,**tolerance),('parameters',mode,rank,step,err)
  compare(opt.state_dict(),refopt.state_dict(),'optimizer');compare(ema.state_dict(),refema.state_dict(),'ema')
  if mode!='candidate5':
   for lid,state in states.items():
    for key in ['tau','eps']:assert torch.equal(state[key],refstates[lid][key]),(lid,key,step)
  if step==3:
   # Real serialize/readback and a fresh DDP wrapper while the group stays live.
   path=OUT/mode/f'toy_rank{rank}.pt';path.parent.mkdir(parents=True,exist_ok=True)
   torch.save({'network':net.state_dict(),'optimizer':opt.state_dict(),'ema':ema.state_dict(),'logical_rng':states,
               'physical_rng':torch.get_rng_state(),'cursor':(step+1)*96,'update':step+1},path)
   saved=torch.load(path,map_location='cpu',weights_only=False);dist.barrier();del ddp;gc.collect()
   net=Net();net.load_state_dict(saved['network']);opt=optimizer(net);opt.load_state_dict(saved['optimizer']);ema=ema_for(net);ema.load_state_dict(saved['ema'])
   states=saved['logical_rng'];torch.set_rng_state(saved['physical_rng']);assert saved['cursor']==384 and saved['update']==4
   ddp=DDP(Obj(net),broadcast_buffers=False,find_unused_parameters=False,bucket_cap_mb=.001)
  dist.barrier()
 record={'status':'PASS','rank':rank,'mode':mode,'steps':12,'resume_after_step':4,'effective_batch':96,'max_gradient_abs_error':max_gradient,
         'max_parameter_abs_error':max_parameter,'max_optimizer_abs_error':max_optimizer,'max_EMA_abs_error':max_ema,
         'logical_ids':lids,'logical_rng_exact':True if mode!='candidate5' else None,'dummy_global_rng_checks':dummy_rng_checks,
         'tolerance':tolerance,'toy_model_only':True,'NPU_calls':0,'original_checkpoints_loaded':False}
 write(OUT/mode/f'RESULT_rank{rank}.json',record);dist.barrier();dist.destroy_process_group()
 if rank==0:print('PASS_GLOO_TOY',mode,flush=True)

def positions_and_boundary():
 import ast
 import random
 from collections import defaultdict
 import numpy as np
 import torch
 source=HERE.parent/'unidb_d0_20261102/data.py';tree=ast.parse(source.read_text())
 core=HERE.parent/'unidb_d0_20261102/core.py';d=next(n for n in ast.parse(core.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='derive')
 context={'hashlib':hashlib,'random':random,'defaultdict':defaultdict,'torch':torch,'Dataset':torch.utils.data.Dataset}
 exec(compile(ast.Module(body=[d],type_ignores=[]),str(core),'exec'),context)
 fake=np.zeros((3,256,256),dtype=np.float32)
 class NP:
  float32=np.float32
  @staticmethod
  def load(*args,**kwargs):return fake
 context['np']=NP
 selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Order','Positions']]
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),context)
 pool=[]
 for category in range(50):
  for src in range(200):
   for rep in range(16):pool.append({'category_id':category,'source_id':f'c{category}s{src}','g':rep%4+1,'observation_path':'toy','clean_path':'toy'})
 order=context['Order'](pool,843101);checks=[]
 for cursor in [0,159936,4359456]:
  for step in range(3):
   expected=[(cursor+step*96+i,order.at(cursor+step*96+i)) for i in range(96)];actual=[]
   for physical in range(3):
    for lid in range(8):
     if lid%3!=physical:continue
     ds=context['Positions'](pool,843101,cursor,36,lid,8,12)
     for j in range(12):
      _,_,row,pos,_=ds[step*12+j];actual.append((pos,row))
   assert sorted(actual)==expected and len({x[0] for x in actual})==96
   candidate=[];sizes=[20,19,19,19,19]
   for rank,n in enumerate(sizes):
    start=sum(sizes[:rank]);candidate += [(cursor+step*96+start+i,order.at(cursor+step*96+start+i)) for i in range(n)]
   assert candidate==expected;checks.append({'cursor':cursor,'step':step,'positions':96,'old8logical_exact':True,'candidate5_coverage_exact':True})
 # The deadline is a decision at a full-update boundary, never between microsteps.
 deadline=10.;now=9.5;updates=0;cursor=4359456;micros=0
 while now<deadline:
  for _ in range(3):now+=.3;micros+=1
  updates+=1;cursor+=96
 assert updates==1 and micros==3 and cursor==4359552 and now>=deadline
 no_start=(10.>=deadline);assert no_start
 write(OUT/'POSITIONS_BOUNDARY.json',{'status':'PASS_CPU_TOY','checks':checks,'deadline_inflight_full_update_completed':True,
   'deadline_no_new_update_after_cutoff':True,'original_data_source_sha256':sha(source),'fake_pool_160K_only':True,'NPU_calls':0})

def objective_schedule():
 import importlib.util
 import types
 import torch
 # Inject a tiny analytic bridge to exercise the actual new objective module.
 class Bridge:
  T=100;dt=.1
  def __init__(self):self.sigmas=torch.linspace(.01,.2,101,dtype=torch.float64)
  def set_model(self,m):self.model=m
  def generate_random_states(self,x0,mu):
   self.mu=mu;self.clean=x0;self.tau=torch.tensor([1,10,50,99]).reshape(4,1,1,1)
   self.state=.6*x0+.4*mu+.03;return self.tau,self.state
  def f_sigma(self,t):return .02+.001*t
  def noise_fn(self,state,t):return self.model(state,self.mu,t)
  def reverse_sde_step_mean(self,state,score,t):
   a=1+.002*t;b=-.002*t;return a*state+b*self.mu+self.sigmas[t].square()*self.dt*score
  def reverse_optimum_step(self,state,clean,t):return .7*clean+.2*state+.1*self.mu
 previous=sys.modules.get('core');sys.modules['core']=types.SimpleNamespace(process=lambda device:Bridge())
 spec=importlib.util.spec_from_file_location('dual_improved_objective_cpu_gate',HERE/'improved_objective.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if previous is None:sys.modules.pop('core',None)
 else:sys.modules['core']=previous
 class Tiny(torch.nn.Module):
  def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor([.2,-.1,.03],dtype=torch.float64));self.calls=0
  def forward(self,x,y,t):self.calls+=1;return self.w[0]*x+self.w[1]*y+self.w[2]
 net=Tiny();obj=mod.ImprovedObjective(net,'cpu');c=torch.arange(4*3*2*2,dtype=torch.float64).reshape(4,3,2,2)/50;y=c*.8+.01
 loss,t=obj(y,c);assert net.calls==1
 state=obj.sde.state;s=obj.sde;pred=net.w[0]*state+net.w[1]*y+net.w[2];coef=s.sigmas[t].square()*s.dt/s.f_sigma(t)
 mu0=(1+.002*t)*state-.002*t*y;target=.7*c+.2*state+.1*y
 residual=mu0-coef*pred-target;reference=(residual/coef).square().mean()
 assert torch.allclose(loss,reference,atol=1e-10,rtol=1e-10)
 grad=torch.autograd.grad(loss,net.w)[0];finite=[]
 for j in range(3):
  old=float(net.w[j]);eps=1e-6
  with torch.no_grad():net.w[j]=old+eps
  plus=float(obj(y,c)[0]);
  with torch.no_grad():net.w[j]=old-eps
  minus=float(obj(y,c)[0]);
  with torch.no_grad():net.w[j]=old
  finite.append((plus-minus)/(2*eps))
 assert torch.allclose(grad,torch.tensor(finite,dtype=torch.float64),atol=1e-5,rtol=1e-6),(grad,finite)
 f=mod.lr_for_update;start=1000.;end=10000.;warm=2000.
 assert f(1,start,end,now_unix=1000.)==2e-7
 assert abs(f(200,start,end,now_unix=2000.)-2e-6)<1e-20
 assert f(199,start,end,now_unix=1999.)<2e-6
 assert abs(f(201,start,end,warm,now_unix=warm)-2e-6)<1e-20
 assert f(201,start,end,warm,now_unix=end)==2e-7
 expected=f(401,start,end,warm,now_unix=5000.);saved=json.loads(json.dumps({'update':400,'start':start,'deadline':end,'warmup_end':warm}))
 resumed=f(saved['update']+1,saved['start'],saved['deadline'],saved['warmup_end'],now_unix=5000.)
 assert expected==resumed and resumed==f(4001,start,end,warm,now_unix=5000.)
 for args in [(0,start,end,None),(201,start,end,None),(100,start,end,warm)]:
  try:f(args[0],args[1],args[2],args[3],now_unix=3000.)
  except ValueError:pass
  else:raise AssertionError(('invalid scheduler state accepted',args))
 write(OUT/'OBJECTIVE_SCHEDULE.json',{'status':'PASS_CPU_TINY_ANALYTIC_BRIDGE','objective_source_sha256':sha(HERE/'improved_objective.py'),
  'epsilonopt_MSE_equals_mu_residual_squared_over_c_squared':True,'one_forward_for_primary_objective':True,
  'full_tiny_parameter_gradient_finite_difference_maxabs':float((grad-torch.tensor(finite,dtype=torch.float64)).abs().max()),
  'warmup1_initial_200_peak':True,'persisted_wall_anchor_resume_identical':True,'same_walltime_independent_of_postwarm_update_number':True,
  'invalid_missing_anchor_or_rewarm_state_rejected':True,'new_137M_or_NPU_calls':0,'limit':'Analytic bridge stub, actual native bridge tested separately by math agent.'})

def orchestrate():
 import torch
 OUT.mkdir(parents=True,exist_ok=True)
 if not (OUT/'POSITIONS_BOUNDARY.json').exists():positions_and_boundary()
 if not (OUT/'OBJECTIVE_SCHEDULE.json').exists():objective_schedule()
 assert json.loads((OUT/'POSITIONS_BOUNDARY.json').read_text())['status']=='PASS_CPU_TOY'
 assert json.loads((OUT/'OBJECTIVE_SCHEDULE.json').read_text())['objective_source_sha256']==sha(HERE/'improved_objective.py')
 results=[]
 for mode,world in [('old3_padded',3),('candidate5',5),('old3_unequal',3)]:
  folder=OUT/mode;folder.mkdir(exist_ok=True);log=folder/'gloo.log'
  env=os.environ.copy();env.update({'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',str(Path(__file__)),'--worker',mode]
  with log.open('w') as f:
   process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=75)
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10);code=124
  required=mode!='old3_unequal';passed=code==0 and all((folder/f'RESULT_rank{rank}.json').exists() for rank in range(world))
  results.append({'mode':mode,'world':world,'exit_code':code,'required':required,'passed':passed,'log_sha256':sha(log)})
  if required:assert passed,(mode,code)
 write(OUT/'ENGINEERING_CPU_GATE.json',{'status':'PASS_REQUIRED_TOY_GATES','torch_version':torch.__version__,'tests':results,
  'positions':'POSITIONS_BOUNDARY.json','objective_scheduler':'OBJECTIVE_SCHEDULE.json','test_source_sha256':sha(Path(__file__)),
  'NPU_calls':0,'original_checkpoints_loaded':False,'original_weights_updated':False,
  'scope':'CPU/gloo small models only. Actual original137M/NPU/RNG resume gate belongs to root.'})
 print('PASS_DUAL_CPU_GLOO_GATES',json.dumps(results),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
 if args.worker:worker(args.worker)
 else:orchestrate()

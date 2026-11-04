"""Independent CPU full-checkpoint audit. No network/optimizer construction.

Single: --path CHECKPOINT --out JSON
8-to-3 gate: --compare-eight REF --compare-three TEST --parent ORIGINAL --out JSON
Tolerance policy is frozen in protocol/AUDIT_STATE_FROZEN_TOLERANCES.json.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,sys,time,traceback
from pathlib import Path
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
POLICY_PATH=ROOT/'protocol/AUDIT_STATE_FROZEN_TOLERANCES.json'
CHUNK=1048576
PARAMETERS=137147523
TENSORS=151


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def ensure(condition,message):
 if not condition:raise AssertionError(message)

def integer(value,label):
 if isinstance(value,torch.Tensor):
  ensure(value.numel()==1,label+' not scalar');value=float(value)
 ensure(not isinstance(value,bool) and isinstance(value,(int,float,np.integer,np.floating)) and math.isfinite(float(value)) and int(value)==value,label+' not an exact finite integer')
 return int(value)

def finite_number(value,label):
 value=float(value);ensure(math.isfinite(value),label+' not finite');return value

def state_tensor(v,label,dtype=torch.float32):
 ensure(isinstance(v,torch.Tensor),label+' not Tensor');ensure(v.device.type=='cpu',label+' not CPU-loaded')
 ensure(v.dtype==dtype,label+' wrong dtype '+str(v.dtype));ensure(bool(torch.isfinite(v).all()),label+' nonfinite')

def expected_ema_initted(update,update_after_step=100,update_every=10):
 # ema-pytorch0.3.2 tests the old step, increments stored step first,
 # and leaves initted False while old_step <= update_after_step.
 first_old_step=(update_after_step//update_every+1)*update_every
 return update >= first_old_step+1

def tensor_stats(v):
 flat=v.reshape(-1);sse=0.;maxabs=0.
 for i in range(0,flat.numel(),CHUNK):
  q=flat[i:i+CHUNK].double();sse+=float(q.square().sum());maxabs=max(maxabs,float(q.abs().max()) if q.numel() else 0.)
 return {'shape':list(v.shape),'dtype':str(v.dtype),'numel':v.numel(),'finite':True,'L2':math.sqrt(sse),'maxabs':maxabs}

def network_audit(net,label):
 ensure(isinstance(net,dict) and len(net)==TENSORS,label+' must have151 tensors');ensure(sum(v.numel() for v in net.values())==PARAMETERS,label+' parameter count mismatch')
 result={};h=hashlib.sha256()
 for name,v in net.items():
  state_tensor(v,label+'.'+name);result[name]=tensor_stats(v);h.update(name.encode())
  flat=v.contiguous().reshape(-1)
  for i in range(0,flat.numel(),CHUNK):h.update(flat[i:i+CHUNK].numpy().tobytes())
 return {'tensors':len(net),'parameters':PARAMETERS,'all_FP32_finite':True,'full_parameter_digest':h.hexdigest(),'per_tensor':result}

def rng_structure(state,label):
 ensure(isinstance(state,dict) and set(state)=={'torch','npu','numpy','python'},label+' missing/extra RNG fields')
 result={}
 for kind in ['torch','npu']:
  v=state[kind];state_tensor(v,label+'.'+kind,dtype=torch.uint8);ensure(v.ndim==1 and v.numel()>0,label+'.'+kind+' invalid byte state')
  result[kind]={'bytes':v.numel(),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()}
 n=state['numpy'];ensure(isinstance(n,tuple) and len(n)==5 and n[0]=='MT19937',label+' invalid NumPy RNG tuple')
 ensure(isinstance(n[1],np.ndarray) and n[1].dtype==np.uint32 and n[1].shape==(624,),label+' invalid NumPy keys')
 ensure(0<=integer(n[2],label+' numpy position')<=624 and integer(n[3],label+' numpy has_gauss') in [0,1],label+' invalid NumPy metadata');finite_number(n[4],label+' numpy cached Gaussian')
 p=state['python'];ensure(isinstance(p,tuple) and len(p)==3 and p[0]==3 and isinstance(p[1],tuple) and len(p[1])==625,label+' invalid Python random state')
 ensure(all(isinstance(v,int) and 0<=v<=2**32-1 for v in p[1][:-1]) and 0<=p[1][-1]<=624,label+' invalid Python random integers')
 if p[2] is not None:finite_number(p[2],label+' Python cached Gaussian')
 result['numpy']={'algorithm':n[0],'keys':624,'position':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])}
 result['python']={'version':p[0],'integers':625,'position':p[1][-1],'cached_gaussian':p[2]}
 return result

def exact(a,b,label):
 if isinstance(a,torch.Tensor):ensure(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),label+' tensor differs')
 elif isinstance(a,np.ndarray):ensure(isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label+' NumPy array differs')
 elif isinstance(a,dict):
  ensure(isinstance(b,dict) and set(a)==set(b),label+' dictionary keys differ')
  for k in a:exact(a[k],b[k],label+'.'+str(k))
 elif isinstance(a,(tuple,list)):
  ensure(type(a) is type(b) and len(a)==len(b),label+' sequence differs')
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'.'+str(i))
 else:ensure(type(a) is type(b) and a==b,label+' value differs')

def count_map(value):
 ensure(isinstance(value,dict),'counts_by_g not dict');out={str(g):0 for g in range(1,5)};seen=set()
 for g,n in value.items():
  key=str(g);ensure(key in out and key not in seen,'invalid/duplicate strength count key');seen.add(key)
  out[key]=integer(n,'counts_by_g.'+key);ensure(out[key]>=0,'negative strength count')
 return out

def check_sidecar(path,digest,payload):
 checked=[];side=path.with_suffix('.json')
 if side.exists():
  q=json.loads(side.read_text());ensure(q['sha256']==digest,'Checkpoint SHA differs from '+str(side))
  for k in ['update','cursor']:
   if k in q:ensure(q[k]==payload[k],'Checkpoint '+k+' differs from sidecar')
  if 'bytes' in q:ensure(q['bytes']==path.stat().st_size,'Checkpoint byte count differs from sidecar')
  if payload['format']=='UNIDB_DUAL_FULL_STATE_V1':ensure(q.get('full_state') is True,'New checkpoint sidecar not full_state')
  checked.append({'path':str(side),'sha256':sha(side)})
 for s in [path.with_suffix('.sha256'),path.with_suffix(path.suffix+'.sha256')]:
  if s.exists():ensure(s.read_text().split()[0].lower()==digest,'Text SHA sidecar mismatch');checked.append({'path':str(s),'sha256':sha(s)})
 return checked

def checkpoint(path):
 path=Path(path).resolve();ensure(path.is_file(),'Checkpoint missing '+str(path));digest=sha(path)
 p=torch.load(path,map_location='cpu',weights_only=False);ensure(isinstance(p,dict),'Checkpoint payload not dict')
 fmt=p.get('format');ensure(fmt in ['UNIDB_DUAL_FULL_STATE_V1','UNIDB_NATIVE_FULL_STATE_V1'],'Unknown checkpoint format')
 report={'path':str(path),'sha256':digest,'bytes':path.stat().st_size,'format':fmt,'sidecars':check_sidecar(path,digest,p)}
 u=integer(p['update'],'update');cursor=integer(p['cursor'],'cursor');ensure(u>=0 and cursor>=0,'Negative update/cursor')
 raw=network_audit(p['network'],'network');ema=p['ema'];ea=network_audit(ema['network'],'ema.network')
 ensure(list(p['network'])==list(ema['network']),'RAW/EMA parameter names/order differ')
 ensure(integer(ema['step'],'EMA.step')==u,'EMA.step != global update');ensure(float(ema['beta'])==.995 and integer(ema['update_every'],'EMA.update_every')==10 and integer(ema['update_after_step'],'EMA.update_after_step')==100,'Unexpected EMA recipe')
 ensure(isinstance(ema['initted'],torch.Tensor) and ema['initted'].dtype==torch.bool and ema['initted'].numel()==1,'EMA.initted invalid')
 ensure(bool(ema['initted'])==expected_ema_initted(u,ema['update_after_step'],ema['update_every']),'EMA.initted inconsistent with installed0.3.2 old-step/update_after/update_every semantics')
 opt=p['optimizer'];ensure(isinstance(opt,dict) and set(opt)=={'state','param_groups'},'Optimizer schema mismatch')
 groups=opt['param_groups'];ensure(len(groups)==1,'Expected one Adam group');g=groups[0];ids=g['params']
 ensure(ids==list(range(TENSORS)),'Adam parameter IDs/order mismatch');ensure(tuple(g['betas'])==(.9,.99) and float(g['eps'])==1e-8 and float(g['weight_decay'])==0.,'Adam recipe mismatch')
 for flag in ['amsgrad','maximize','differentiable','decoupled_weight_decay']:ensure(not g.get(flag,False),'Unexpected Adam flag '+flag)
 lr=finite_number(g['lr'],'Adam.lr');ensure(lr>0,'Adam lr not positive');states=opt['state'];ensure(set(states)==(set(ids) if u else set()),'Missing/extra Adam states')
 astats={};names=list(p['network'])
 for ident in ids if u else []:
  v=states[ident];ensure(set(v)=={'step','exp_avg','exp_avg_sq'},'Adam state keys mismatch '+str(ident))
  ensure(integer(v['step'],'Adam.step')==u,'Adam state step mismatch '+str(ident));state_tensor(v['step'],'Adam.step')
  row={'step':u,'parameter':names[ident]}
  for field in ['exp_avg','exp_avg_sq']:
   z=v[field];state_tensor(z,'Adam.'+str(ident)+'.'+field);ensure(z.shape==p['network'][names[ident]].shape,'Adam moment shape mismatch')
   if field=='exp_avg_sq':ensure(bool((z>=0).all()),'Negative second moment')
   row[field]=tensor_stats(z)
  astats[str(ident)]=row
 cfg=p['config'];ensure(integer(cfg['effective_batch'],'effective_batch')==96 and cfg['gamma_inv']==1e-7 and cfg['T']==100,'Bridge/batch recipe mismatch')
 counts=count_map(p['counts_by_g']);ensure(sum(counts.values())==cursor,'Strength counts do not sum to cursor');ensure(p.get('scaler') is None,'Unexpected AMP scaler state')
 scheduler=p['scheduler']
 if fmt=='UNIDB_NATIVE_FULL_STATE_V1':
  ensure(cfg['precision']=='FP32' and cfg['HF32'] is False,'Legacy precision mismatch');ensure(cursor==u*96,'Legacy cursor !=96*update')
  ensure(lr==1e-5 and scheduler['type']=='constant' and scheduler['lr']==lr,'Legacy LR mismatch');ensure(cfg['seed']==843101 and cfg['world_size']==8 and cfg['microbatch']==12 and cfg['accumulation']==1,'Legacy baseline layout/seed mismatch')
  branch='original';initial=0;added=u;banks=p['rng_by_rank'];ensure(len(banks)==8,'Legacy RNG bank must have8 ranks');physical=cfg['world_size']
 else:
  branch=p['branch'];ensure(branch in ['original','improved'] and p['phase'] in ['gate','formal'],'Invalid new branch/phase')
  initial=integer(p['initial_update'],'initial_update');added=integer(p['new_updates'],'new_updates');ensure(added==u-initial and added>=0,'New update accounting mismatch')
  physical=integer(p['physical_world_size'],'physical world');ensure(cfg['branch']==branch and cfg['phase']==p['phase'] and cfg['physical_world_size']==physical and cfg['data_seed']==843101,'Config identity mismatch')
  ensure(cfg['deadline_unix']==1793678400,'Config deadline mismatch');ensure(cfg['precision']=='FP32_HF32off' and cfg['Adam_betas']==[.9,.99] and cfg['Adam_eps']==1e-8 and cfg['weight_decay']==0. and cfg['EMA_beta']==.995 and cfg['EMA_update_every']==10,'New optimization/precision recipe mismatch')
  ensure(p['data_positions']['next_global_position']==cursor and p['data_positions']['effective_batch']==96 and p['data_positions']['seed']==843101,'Data-position checkpoint inconsistent')
  ensure(integer(scheduler['completed_updates'],'scheduler.completed')==u and integer(scheduler['next_update'],'scheduler.next')==u+1,'Scheduler update counters inconsistent')
  start=finite_number(scheduler['start_unix'],'scheduler.start');deadline=finite_number(scheduler['deadline_unix'],'scheduler.deadline');ensure(0<start<deadline and deadline==1793678400,'Scheduler deadline/start invalid')
  ensure(float(scheduler['current_lr'])==lr,'Scheduler LR != optimizer group LR')
  if branch=='original':
   ensure(initial==45411 and cursor==4359456+96*added,'Original baseline/cursor mismatch');ensure(physical==3 or (physical==8 and p['phase']=='gate'),'Original physical world invalid')
   ensure(p['logical_world_size']==cfg['logical_world_size']==8 and p['rng_by_rank'] is None,'Original logical RNG metadata invalid')
   ensure(p['parent']['update']==45411 and p['parent']['sha256']==json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())['original_checkpoint_sha256'],'Original parent identity mismatch')
   banks=p['rng_by_logical_rank'];ensure(len(banks)==8,'Original requires8 logical RNG banks');ensure(lr==1e-5 and scheduler['type']=='constant' and scheduler['warmup_end_unix'] is None and cfg['gradient_clip'] is None and cfg['objective']=='native_mean_matching_L1','Original scientific recipe changed')
  else:
   ensure(initial==0 and cursor==96*added and u==added and physical==5,'Improved baseline/cursor/world mismatch')
   ensure(p['logical_world_size'] is None and cfg['logical_world_size'] is None and p['rng_by_logical_rank'] is None,'Improved RNG metadata invalid');banks=p['rng_by_rank'];ensure(len(banks)==5,'Improved requires5 RNG banks')
   ensure(p['parent']['official_SR_sha256']=='a6bb76bfabe32b8403d8b112f0ce7876114d830d20b9f4a4119aa860afef64b4' and p['parent']['fresh_optimizer'] is True,'Improved parent mismatch')
   ensure(cfg['objective']=='epsilon_opt_MSE' and cfg['gradient_clip']==1. and scheduler['type']=='warmup_updates_then_wallclock_cosine','Improved scientific recipe mismatch')
   spec=scheduler['schedule_spec'];ensure(spec['version']=='epsilon_opt_mse_wallclock_cosine_v1' and spec['initial_lr']==spec['floor_lr']==2e-7 and spec['peak_lr']==2e-6 and spec['warmup_updates']==200,'Improved schedule specification changed')
   ensure(2e-7-1e-16<=lr<=2e-6+1e-16,'Improved lr outside frozen range');anchor=scheduler['warmup_end_unix']
   if u<200:ensure(anchor is None,'Premature warmup completion timestamp')
   else:ensure(anchor is not None and math.isfinite(float(anchor)) and float(anchor)>=start,'Missing/invalid persisted warmup anchor')
   if u<=200:
    expected=2e-7 if u==0 else 2e-7+(2e-6-2e-7)*(u-1)/199
    ensure(math.isclose(lr,expected,rel_tol=1e-12,abs_tol=1e-16),'Warmup LR does not match completed update')
   if u>200:ensure(float(anchor)<deadline,'Post-warmup update has no remaining interval')
  loaders=p['loader_generator_states'];ensure(len(loaders)==physical,'Incomplete physical loader-generator bank')
  for i,v in enumerate(loaders):state_tensor(v,'loader_generator.'+str(i),torch.uint8);ensure(v.ndim==1 and v.numel()>0,'Invalid loader RNG byte state')
  if p.get('full_parameter_digest') is not None:ensure(p['full_parameter_digest']==raw['full_parameter_digest'],'Claimed full parameter digest mismatch')
 rng=[rng_structure(v,'rng.'+str(i)) for i,v in enumerate(banks)]
 report.update(status='PASS',branch=branch,update=u,initial_update=initial,new_updates=added,cursor=cursor,counts_by_g=counts,physical_world_size=physical,
  RAW=raw,EMA=ea,Adam={'states':len(states),'step_all':u,'all_FP32_finite':True,'lr':lr,'per_parameter':astats},EMA_step=u,RNG={'banks':len(banks),'structure_complete':True,'per_bank':rng},scheduler=scheduler,
  schema_scope='No model/optimizer constructed; Adam IDs mapped to frozen all-parameter state_dict order and shape-checked.')
 return p,report


def compare_tensors(reference,test,atol,rtol,parent=None):
 ensure(reference.shape==test.shape and reference.dtype==test.dtype,'Comparison tensor schema mismatch')
 ref=reference.reshape(-1);tst=test.reshape(-1);par=parent.reshape(-1) if parent is not None else None
 error2=reference2=delta2=0.;maximum=0.;allclose=True
 for i in range(0,ref.numel(),CHUNK):
  a=ref[i:i+CHUNK].double();b=tst[i:i+CHUNK].double();diff=b-a
  error2+=float(diff.square().sum());reference2+=float(a.square().sum());maximum=max(maximum,float(diff.abs().max()) if diff.numel() else 0.)
  allclose=allclose and bool((diff.abs()<=atol+rtol*a.abs()).all())
  if par is not None:delta2+=float((a-par[i:i+CHUNK].double()).square().sum())
 return {'maxabs':maximum,'error_squared_L2':error2,'reference_squared_L2':reference2,'reference_delta_squared_L2':delta2,'allclose':allclose}

def ratio(numerator2,denominator2):
 if denominator2==0:return (0.,True) if numerator2==0 else (None,False)
 return math.sqrt(numerator2/denominator2),True

def compare_states(reference,test,parent,policy,result=None):
 for field in ['update','cursor','initial_update','new_updates','counts_by_g']:exact(reference[field],test[field],field)
 ensure(reference['branch']==test['branch']=='original' and reference['physical_world_size']==8 and test['physical_world_size']==3,'Expected original8 vs original3 comparison')
 ensure(reference['initial_update']==test['initial_update']==parent['update']==45411,'Comparison parent update mismatch')
 ensure(reference['new_updates']==test['new_updates']==policy['comparison_added_updates_required']==2,'Gate comparison must cover exactly2 added updates')
 ensure(reference['cursor']==test['cursor']==parent['cursor']+192,'Two-update cursor mismatch')
 for g,n in count_map(parent['counts_by_g']).items():ensure(count_map(reference['counts_by_g'])[g]>=n,'Strength counts decreased from parent')
 if result is None:result={}
 result.update(status='PENDING',method_RNG_exact=True,counts_cursor_update_exact=True,frozen_policy=policy,RAW={},EMA={},Adam={},failures=[])
 for bank in range(8):
  for field in ['torch','npu','numpy','python']:
   try:exact(reference['rng_by_logical_rank'][bank][field],test['rng_by_logical_rank'][bank][field],'logical_RNG.'+str(bank)+'.'+field)
   except AssertionError as exc:result['method_RNG_exact']=False;result['failures'].append(str(exc))
 for label,key in [('RAW','network'),('EMA','ema')]:
  a=reference[key] if key=='network' else reference[key]['network'];b=test[key] if key=='network' else test[key]['network'];p=parent[key] if key=='network' else parent[key]['network']
  ensure(list(a)==list(b)==list(p),label+' keys differ');rows={};total={'error_squared_L2':0.,'reference_squared_L2':0.,'reference_delta_squared_L2':0.}
  tol=policy['network_and_EMA_allclose']
  for name in a:
   q=compare_tensors(a[name],b[name],**tol,parent=p[name]);rows[name]=q
   for field in total:total[field]+=q[field]
  rel,defined=ratio(total['error_squared_L2'],total['reference_delta_squared_L2']);result[label]={'per_tensor':rows,'global_error_L2':math.sqrt(total['error_squared_L2']),'reference_two_update_delta_L2':math.sqrt(total['reference_delta_squared_L2']),'error_over_reference_delta':rel,'ratio_defined_or_exact_zero':defined,'allclose':all(q['allclose'] for q in rows.values())}
  if not result[label]['allclose']:result['failures'].append(label+' per-tensor allclose failed: '+str([(k,v['maxabs']) for k,v in rows.items() if not v['allclose']]))
  if label=='RAW' and not (defined and rel<=policy['RAW_error_L2_over_reference_two_update_delta_L2_max']):result['failures'].append('RAW error exceeds1% of reference2-update delta')
 for field in ['step','initted','beta','update_every','update_after_step']:exact(reference['ema'][field],test['ema'][field],'EMA.'+field)
 exact(reference['optimizer']['param_groups'],test['optimizer']['param_groups'],'Adam.param_groups')
 combined_error2=combined_reference2=0.
 for field in ['exp_avg','exp_avg_sq']:
  rows={};e2=r2=0.
  for ident in range(TENSORS):
   exact(reference['optimizer']['state'][ident]['step'],test['optimizer']['state'][ident]['step'],'Adam.step.'+str(ident))
   q=compare_tensors(reference['optimizer']['state'][ident][field],test['optimizer']['state'][ident][field],**policy['Adam_moment_allclose']);rows[str(ident)]=q;e2+=q['error_squared_L2'];r2+=q['reference_squared_L2']
  rel,defined=ratio(e2,r2);result['Adam'][field]={'per_parameter':rows,'global_error_L2':math.sqrt(e2),'reference_moment_L2':math.sqrt(r2),'relative_error':rel,'allclose':all(q['allclose'] for q in rows.values())}
  if not result['Adam'][field]['allclose']:result['failures'].append(field+' moment per-tensor allclose failed')
  if not (defined and rel<=policy['Adam_each_and_combined_error_L2_over_reference_moment_L2_max']):result['failures'].append(field+' moment relative error exceeds1%')
  combined_error2+=e2;combined_reference2+=r2
 rel,defined=ratio(combined_error2,combined_reference2)
 if not (defined and rel<=.01):result['failures'].append('Combined moment relative error exceeds1%')
 result['Adam']['combined_relative_error']=rel;result['status']='FAIL' if result['failures'] else 'PASS';result['physical_loader_generators_scope']='Each checkpoint complete for its physical world; not expected to be identical across8 vs3.'
 return result


def save_report(path,report):
 path=Path(path).resolve();ensure(path.is_relative_to(ROOT),'Output must remain inside new experiment root');path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  archive=path.with_name(path.stem+'.previous.'+str(time.time_ns())+path.suffix);shutil.copy2(path,archive);report['previous_output_preserved']={'path':str(archive),'sha256':sha(archive)}
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--path',type=Path);ap.add_argument('--compare-eight',type=Path);ap.add_argument('--compare-three',type=Path);ap.add_argument('--parent',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);started=time.time();report={'status':'FAIL','actor':{'pid':os.getpid()},'source_sha256':sha(__file__),'policy_sha256':sha(POLICY_PATH),'CPU_threads':4,'model_objects':0,'optimizer_objects':0,'NPU_calls':0}
 code=0
 try:
  policy=json.loads(POLICY_PATH.read_text());ensure(policy['network_and_EMA_allclose']=={'atol':2e-6,'rtol':2e-5} and policy['Adam_moment_allclose']=={'atol':2e-8,'rtol':2e-4} and policy['RAW_error_L2_over_reference_two_update_delta_L2_max']==.01 and policy['Adam_each_and_combined_error_L2_over_reference_moment_L2_max']==.01,'Frozen tolerance policy changed')
  if args.path:
   ensure(not any([args.compare_eight,args.compare_three,args.parent]),'Choose single audit or comparison, not both')
   payload,one=checkpoint(args.path);report.update(mode='single',checkpoint=one,status='PASS')
  else:
   ensure(all([args.compare_eight,args.compare_three,args.parent]),'Comparison requires eight/three/parent paths')
   report.update(mode='eight_to_three',audits={})
   parent,pa=checkpoint(args.parent);report['audits']['parent']=pa
   ensure(pa['sha256']==json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())['original_checkpoint_sha256'],'Actual compare-parent SHA differs from authorized original45411')
   reference,ra=checkpoint(args.compare_eight);report['audits']['reference_eight']=ra
   test,ta=checkpoint(args.compare_three);report['audits']['test_three']=ta
   report['comparison']={};compare_states(reference,test,parent,policy,report['comparison']);report['status']=report['comparison']['status'];code=0 if report['status']=='PASS' else 1
 except BaseException as exc:
  report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());code=1
 report['elapsed_seconds']=time.time()-started;report['unix']=time.time();save_report(args.out,report)
 print(json.dumps({'status':report['status'],'out':str(args.out),'error':report.get('error'),'elapsed_seconds':report['elapsed_seconds']},indent=2));raise SystemExit(code)

if __name__=='__main__':main()

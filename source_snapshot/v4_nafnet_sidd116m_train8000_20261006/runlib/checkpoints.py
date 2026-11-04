from pathlib import Path
import torch,time,copy,random,numpy as np
from .common import *
from .operational_revision import compatible_binding
from execution.checkpoints import atomic_save,cpu_tree
from npu_runtime.backend import capture_rng,restore_rng
PURPOSE='V4_NAFNET_TRAIN8000_PHASE1_STATE_V1'
def binding(c,store,stream):
 return {'run':ROOT.name,'config_sha256':canonical(c),'sources':source_hashes(),'science_sources':science_hashes(),'environment':runtime_environment(),'dataset':store.dataset.manifest_hashes,'stats':sha(Path(c['dataset_root'])/'normalization_train8000.json'),'common_init':sha(ROOT/'protocol/common_init.pt'),'stream':stream.identity,'effective_batch':c['effective_batch'],'precision':c['precision']}
def schedule(c,u):
 return {'kind':'constant_peak_disposable_probe' if c.get('probe_constant_lr') else 'exposure_linear16000_cosine640000','completed_updates':u,'completed_exposures':u*c['effective_batch'],'next_lr':None if u==c['max_updates'] else (c['peak_lr'] if c.get('probe_constant_lr') else lr_at(u+1,c))}
def save(path,model,ema,opt,group,c,stream,u,micro,identity,scaler=None,training_health=None):
 path=Path(path);side=path.with_suffix('.identity.json');torch.cuda.synchronize()
 if path.exists() and not side.exists():
  archive=path.parent/'uncommitted_transactions';archive.mkdir(exist_ok=True)
  preserved=archive/(path.name+f'.{time.time_ns()}.uncommitted');path.replace(preserved)
  atomic(preserved.with_suffix('.record.json'),{'reason':'Atomic tensor file existed without committed identity sidecar; preserved before same-update recomputation','path':str(preserved),'sha256':sha(preserved),'time':time.time()})
 if path.exists() or side.exists():
  assert path.exists() and side.exists(),'Incomplete checkpoint transaction must be archived explicitly'
  d=read(side);assert compatible_binding(d['binding'],identity) and d['sha256']==sha(path) and d['update']==u
  return d
 payload={'purpose':PURPOSE,'model':cpu_tree(model.state_dict()),'ema':cpu_tree(ema.state_dict()),'optimizer':cpu_tree(opt.state_dict()),'scheduler':schedule(c,u),'scaler':{} if scaler is None else scaler.state_dict(),'AMP_policy':c['precision'],'rng':capture_rng(device=next(model.parameters()).device),'sampler':stream.state(u),'update':u,'exposures':u*c['effective_batch'],'epoch':u*c['effective_batch']/128000,'group':group,'binding':identity,'runtime':{'microbatch':micro,'accumulation':c['effective_batch']//micro,'effective_batch':c['effective_batch']},'time':time.time()}
 payload['training_health']=training_health
 atomic_save(payload,path);d={'sha256':sha(path),'update':u,'exposures':payload['exposures'],'binding':identity,'path':str(path),'group':group,'time':time.time()};atomic(side,d);return d

def restore(path,model,ema,opt,group,c,stream,identity,scaler=None):
 path=Path(path);d=read(path.with_suffix('.identity.json'));assert sha(path)==d['sha256'] and compatible_binding(d['binding'],identity) and d['group']==group
 p=torch.load(path,map_location='cpu',weights_only=False);assert p['purpose']==PURPOSE and p['binding']==d['binding'] and compatible_binding(p['binding'],identity) and p['group']==group
 u=p['update'];assert u==d['update'] and p['exposures']==u*c['effective_batch'] and p['sampler']==stream.state(u)
 assert p['scheduler']==schedule(c,u) and p['AMP_policy']==c['precision'] and p['scaler']=={}
 for n,v in model.stats.state_dict().items():assert torch.equal(v.cpu(),p['model']['stats.'+n])
 model.load_state_dict(p['model'],strict=True);ema.load_state_dict(p['ema'],strict=True);opt.load_state_dict(p['optimizer'])
 if u:assert opt.state and all(int(x['step'])==u for x in opt.state.values())
 if scaler is not None:scaler.load_state_dict(p['scaler'])
 restore_rng(p['rng'],device=next(model.parameters()).device)
 return u,p

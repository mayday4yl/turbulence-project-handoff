from pathlib import Path
import hashlib,json,os,time
from . import ROOT,CANDIDATE
GROUPS=('D0','R0','R08','L08');DATASET_ID='places_lsdir_p2s_canonical_20261006_p4000_l4000_v1000_t200_g4321_v1'
OFFICIAL_SHA='cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445'
CANDIDATE_U0_SHA='b5ed0a9b430618f477c7b1bc0377537365f74df384df2ae20a6874b7d8214b2b'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seed(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%(2**63-1)
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def freeze(p,x):
 p=Path(p)
 if p.exists():assert read(p)==x,'Frozen identity differs: '+str(p)
 else:atomic(p,x)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
def lr_at(update,c):
 import math
 exposure=update*c['effective_batch'];assert 1<=update<=c['max_updates']
 if exposure<=16000:return c['peak_lr']*exposure/16000
 ratio=(exposure-16000)/(640000-16000)
 return 1e-6+.5*(c['peak_lr']-1e-6)*(1+math.cos(math.pi*ratio))
def source_hashes():
 names=['cpu_prepare.py','check_cpu_engine.py','preflight_job.py','preflight_controller.py','train_phase1.py','evaluate_phase1.py','workflow_controller.py','supervise_workflow.py']
 paths=list((ROOT/'runlib').glob('*.py'))+[ROOT/'scripts'/n for n in names]
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def science_hashes():
 return {str(p.relative_to(CANDIDATE)):sha(p) for folder in ['nafnet_v4','v4','physical_bridge','npu_runtime'] for p in sorted((CANDIDATE/folder).glob('*.py'))}
def runtime_environment():
 import sys,platform,importlib.metadata,torch
 expected=read(CANDIDATE/'protocol/ENVIRONMENT.json');deps=expected['metrics_dependency_dir']
 if deps not in sys.path:sys.path.append(deps)
 packages={name:importlib.metadata.version(name) for name in expected['packages']}
 assert packages==expected['packages'],'Execution packages differ from accepted pretrained environment'
 return {'python':platform.python_version(),'packages':packages,'CUDA_runtime':torch.version.cuda,'TF32':False,'physics_dtype':'float32','optimizer_master_dtype':'float32'}
def runtime_config():return read(ROOT/'configs/FROZEN_RUNTIME.json')

"""Immutable launch authority and exact process-ownership helpers."""
from pathlib import Path
import os
from .common import *
from .operational_revision import approved_source_revision,compatible_binding
def verify():
 c=runtime_config();g=read(ROOT/'protocol/LAUNCH_GATE.json')
 assert g['status']=='PASS' and g['config_sha256']==sha(ROOT/'configs/FROZEN_RUNTIME.json')
 actual=source_hashes()
 assert (g['source_hashes']==actual or approved_source_revision(actual,g['source_hashes'])) and g['science_hashes']==science_hashes(),'Launch source identity differs'
 assert g['environment']==runtime_environment(),'Execution environment changed'
 assert g['common_init_sha256']==sha(ROOT/'protocol/common_init.pt')
 assert read(Path(c['dataset_root'])/'A100_RUNTIME_READY.json')['status']=='PASS'
 assert c['effective_batch'] in [32,64] and c['max_updates']*c['effective_batch']==640000
 assert c['peak_lr'] in [1e-4,5e-5] and c['precision'] in ['fp32','bf16'] and c['formal_training_authorized']
 return c,g
def alive(pid,token):
 try:return Path('/proc',str(pid),'cwd').resolve()==ROOT and token in Path('/proc',str(pid),'cmdline').read_bytes().decode().replace('\0',' ')
 except (FileNotFoundError,PermissionError):return False
def latest(run,identity):
 # Orphans remain untouched; only fully committed same-run transactions qualify.
 for path in sorted((run/'checkpoints').glob('full_U*.pt'),reverse=True):
  side=path.with_suffix('.identity.json')
  if not side.exists():continue
  d=read(side);assert compatible_binding(d['binding'],identity) and sha(path)==d['sha256'],'Committed checkpoint corrupted or identity differs'
  assert path.resolve().is_relative_to((ROOT/'runs/phase1').resolve())
  return path,d
 return None,None

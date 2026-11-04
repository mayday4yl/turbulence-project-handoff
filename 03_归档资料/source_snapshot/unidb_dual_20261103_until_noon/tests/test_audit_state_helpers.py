"""CPU-only helper selftest; not a full137M checkpoint acceptance."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_state import compare_tensors,exact,ratio,rng_structure,count_map,integer,expected_ema_initted

def reject(fn):
 try:fn()
 except AssertionError:return
 raise AssertionError('Expected rejection')

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 a=torch.tensor([1.,-2.,0.],dtype=torch.float32);b=a+torch.tensor([1e-6,-1e-6,1e-7]);p=a-.001
 q=compare_tensors(a,b,2e-6,2e-5,parent=p);assert q['allclose'] and q['reference_delta_squared_L2']>0
 reject(lambda:exact(a,b,'not_exact'));exact(a,a.clone(),'exact');assert ratio(0.,0.)==(0.,True) and ratio(1.,0.)==(None,False)
 qbad=compare_tensors(a,b+.01,2e-6,2e-5,parent=p);assert not qbad['allclose']
 assert count_map({'1':2,'4':3})=={'1':2,'2':0,'3':0,'4':3};reject(lambda:count_map({'1':1,1:2}));reject(lambda:integer(1.5,'not_integer'))
 # Actual local CPU RNG, synthetic nonempty accelerator byte state: no NPU API.
 import random
 state={'torch':torch.get_rng_state(),'npu':torch.tensor([1,2,3,4],dtype=torch.uint8),'numpy':np.random.get_state(),'python':random.getstate()}
 rng_structure(state,'test');exact(state,{'torch':state['torch'].clone(),'npu':state['npu'].clone(),'numpy':np.random.get_state(),'python':random.getstate()},'same_rng')
 changed=dict(state);changed['npu']=state['npu'].clone();changed['npu'][0]=0;reject(lambda:exact(state,changed,'different_rng'))
 from importlib.metadata import version
 from ema_pytorch import EMA
 assert version('ema-pytorch')=='0.3.2'
 class TinyEMA(torch.nn.Module):
  def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.ones(1))
 tiny=TinyEMA();ema=EMA(tiny,beta=.995,update_every=10);boundaries=[];wanted={0,2,100,101,110,111,200}
 for u in range(201):
  if u:
   with torch.no_grad():tiny.value.fill_(u/1000.)
   ema.update()
  if u in wanted:
   expected=expected_ema_initted(u,ema.update_after_step,ema.update_every)
   assert int(ema.step)==u and bool(ema.initted)==expected
   boundaries.append({'update':u,'EMA_step':int(ema.step),'initted':bool(ema.initted),'expected':expected,'EMA_value':float(ema.ema_model.value.detach())})
 assert [r['initted'] for r in boundaries]==[False,False,False,False,False,True,True]
 out=ROOT/'evidence/audit_state_helpers';out.mkdir(parents=True,exist_ok=True)
 report={'status':'PASS','scope':'Small helper tests only; no fullcheckpoint quality or acceptance claimed','real_restoration_model_objects':0,'tiny_online_EMA_model_objects':1,'tiny_copied_EMA_model_objects':1,'optimizer_objects':0,'NPU_calls':0,
  'EMA_version':version('ema-pytorch'),'EMA_boundaries':boundaries,'toy_EMA_network_forward_calls':0,'source_sha256':hashlib.sha256((ROOT/'audit_state.py').read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256((ROOT/'protocol/AUDIT_STATE_FROZEN_TOLERANCES.json').read_bytes()).hexdigest(),'comparison':q,'deliberately_different_comparison':qbad,'unix':time.time()}
 (out/'CPU_HELPER_SELFTEST.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

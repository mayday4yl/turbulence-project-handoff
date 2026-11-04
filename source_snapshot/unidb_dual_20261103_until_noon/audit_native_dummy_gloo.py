"""Additional gloo test of actual SyncObjective native-dummy branch.
Reuses the independently checked 8logical-to3 toy reference harness. No NPU.
"""
import ast
import os
from pathlib import Path
import sys
import torch
from torch.nn.parallel import DistributedDataParallel as RealDDP

ROOT=Path(__file__).resolve().parent
node=next(n for n in ast.parse((ROOT/'training_worker.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SyncObjective')
scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_SyncObjective','exec'),scope)
ActualSync=scope['SyncObjective']

class Inner(torch.nn.Module):
 def __init__(self,network):super().__init__();self.network=network
 def forward(self,x,y):
  return torch.nn.functional.l1_loss(self.network(x),y),torch.ones(len(x),1,1,1,dtype=torch.long)

class Adapter(torch.nn.Module):
 def __init__(self,network):super().__init__();self.actual=ActualSync(Inner(network),dummy_mode='native')
 def forward(self,x,y,dummy=False):
  if not dummy:return self.actual(x,y,dummy=False)[0]
  # Same engineering pattern as the NPU worker: isolate bridge RNG, keep all
  # actual network/loss/backward paths exercised, zero its optimization weight.
  before=torch.get_rng_state().clone()
  x=torch.randn(12,7);y=torch.randn(12,5)
  loss,tau=self.actual(x,y,dummy=True)
  torch.set_rng_state(before)
  assert float(loss)==0 and tau.numel()==12
  return loss

class DDP(RealDDP):
 def __init__(self,objective,*args,**kwargs):super().__init__(Adapter(objective.network),*args,**kwargs)

# The harness imports DDP inside worker(), so this substitutes only its wrapper;
# serial reference, scaling, model/Adam/EMA comparison and RNG checks are intact.
torch.nn.parallel.DistributedDataParallel=DDP
import audit_engineering
audit_engineering.worker('old3_native_dummy')

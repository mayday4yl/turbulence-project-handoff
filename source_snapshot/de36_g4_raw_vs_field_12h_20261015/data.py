from common import *
import numpy as np
import torch

class Store:
 def __init__(self):
  self.rows=json.loads((ROOT/'protocol/SAMPLES.json').read_text());self.cache={};self.checked=set()
 def array(self,r,key):
  assert r['split'] in ('train','val')
  a=r['arrays'][key];p=Path(CONFIG['dataset_root'])/a['ref']
  assert '/test_' not in str(p)
  if str(p) not in self.checked:
   assert digest(p)==a['sha256'],str(p);self.checked.add(str(p))
  x=np.load(p,allow_pickle=False);assert list(x.shape)==a['shape'] and str(x.dtype)=='float32' and np.isfinite(x).all()
  return torch.from_numpy(x.copy()).unsqueeze(0)
 def batch(self,r,device):
  key=r['trajectory_id']
  if key not in self.cache:self.cache[key]=(self.array(r,'clean'),self.array(r,'X_4'))
  return tuple(x.to(device) for x in self.cache[key])
 def training_rows(self,update,phase='formal'):
  n=len(self.rows['train']);indices=[]
  for k in range(4):
   pos=update*4+k;epoch,offset=divmod(pos,n)
   generator=torch.Generator().manual_seed(seed_for(f"{CONFIG['seed']}|{phase}|epoch{epoch}"))
   indices.append(int(torch.randperm(n,generator=generator)[offset]))
  return [self.rows['train'][i] for i in indices]

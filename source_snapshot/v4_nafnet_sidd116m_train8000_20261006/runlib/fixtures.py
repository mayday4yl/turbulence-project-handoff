"""Independent numerical diagnostics; no probe state enters formal training."""
import torch,numpy as np
def tree_equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
 return a==b
def sample_tensors(state):
 out={}
 for n,v in state.items():
  if torch.is_tensor(v):
   flat=v.detach().float().reshape(-1);stride=max(1,len(flat)//128);out[n]=flat[::stride][:128].cpu().clone()
 return out
def close(a,b):
 assert a.keys()==b.keys()
 return all(torch.allclose(a[k],b[k],rtol=1e-4,atol=1e-6) for k in a)

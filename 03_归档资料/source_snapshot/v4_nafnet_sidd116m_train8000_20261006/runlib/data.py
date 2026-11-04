from pathlib import Path
from collections import OrderedDict,Counter
import json,random,torch
from .common import *
from v4.data import PhysicalDataset,read_array
from v4.normalization import load_artifact
class Store:
 def __init__(self,c):
  self.root=Path(c['dataset_root']);self.dataset=PhysicalDataset(self.root,'train',self.root/'ASSET_LOCK.json');assert self.dataset.dataset_id==DATASET_ID
  self.sources=self.dataset.source_by_id;stats=load_artifact(self.root/'normalization_train8000.json')
  assert stats['population']=='train' and stats['state_count']==128000 and stats['dataset_id']==DATASET_ID
  assert stats['train_manifest_hash']==self.dataset.train_manifest_hash and stats['manifest_sha256']==self.dataset.manifest_hashes
  self.train=self.records(self.dataset.selected);self.val=self.records([r for r in self.dataset.trajectories if r['split']=='val'])
  assert len(self.train)==128000 and len(self.val)==8000
  assert not {x['source_id'] for x in self.train}&{x['source_id'] for x in self.val}
  for domain in ['Places365','LSDIR']:
   assert len({x['source_id'] for x in self.train if x['domain']==domain})==4000
   assert len({x['source_id'] for x in self.val if x['domain']==domain})==500
  self.cache=OrderedDict();self.cache_bytes=0;self.cache_limit=int(c.get('cache_gib',1)*(1<<30));self.by_id={x['start_id']:x for x in self.train+self.val}
 def records(self,rr):
  out=[]
  for r in sorted(rr,key=lambda r:r['trajectory_id']):
   src=self.sources[r['source_id']];assert src['split']==r['split'] and src['clean']==r['arrays']['clean']
   for g in [1,2,3,4]:out.append({'start_id':r['trajectory_id']+'::G'+str(g),'trajectory_id':r['trajectory_id'],'source_id':r['source_id'],'split':r['split'],'domain':src['dataset_source'],'G':g,'clean':r['arrays']['clean'],'state':r['arrays'][f'X_{g}']})
  return out
 def array(self,a):
  k=(a['ref'],a['sha256'])
  if k in self.cache:self.cache.move_to_end(k);return self.cache[k]
  v=torch.from_numpy(read_array(self.root,a));size=v.numel()*v.element_size()
  while self.cache and self.cache_bytes+size>self.cache_limit:
   _,old=self.cache.popitem(last=False);self.cache_bytes-=old.numel()*old.element_size()
  self.cache[k]=v;self.cache_bytes+=size;return v
 def batch(self,rr,device):
  assert rr and all(r['split'] in ['train','val'] for r in rr),'Test forbidden'
  return torch.stack([self.array(r['clean']) for r in rr]).to(device),torch.stack([self.array(r['state']) for r in rr]).to(device),torch.tensor([r['G'] for r in rr],dtype=torch.float32,device=device)
class Stream:
 def __init__(self,store,B):
  assert B in [32,64];self.store=store;self.B=B;self.per_epoch=128000//B;self.epoch=None;self.orders=None
  # A single universal ordered exposure stream made from balanced blocks of 8.
  # B64 concatenates two B32 batches, not a different permutation.
  self.identity=canonical({'population':[r['start_id'] for r in store.train],'seed':20261006,'algorithm':'balanced_8_strata_universal_order_v1','epochs':5,'microbatch_independent_eta':True})
 def rows(self,u):
  assert 0<=u<640000//self.B;epoch=u//self.per_epoch
  if self.epoch!=epoch:
   self.orders={(g,d):[r for r in self.store.train if r['G']==g and r['domain']==d] for g in [1,2,3,4] for d in ['Places365','LSDIR']}
   for k,rr in self.orders.items():random.Random(seed(f'train8000|20261006|epoch{epoch}|{k}')).shuffle(rr)
   self.epoch=epoch
  start=(u%self.per_epoch)*(self.B//8)
  rr=[self.orders[k][i] for i in range(start,start+self.B//8) for k in self.orders]
  assert Counter((r['G'],r['domain']) for r in rr)=={k:self.B//8 for k in self.orders}
  return rr
 def state(self,u):
  return {'completed_updates':u,'completed_exposures':u*self.B,'stream_identity':self.identity,'effective_batch':self.B,'next_sample_ids':[] if u*self.B==640000 else [r['start_id'] for r in self.rows(u)]}
def eta_batch(rr,device,epoch,namespace='formal'):
 return torch.stack([torch.randn((3,256,256),generator=torch.Generator().manual_seed(seed(f'nafnet8000|{namespace}|20261006|epoch{epoch}|{r["start_id"]}')),dtype=torch.float32) for r in rr]).to(device)
def monitor(store):
 out={}
 for split,pop in [('train',store.train),('val',store.val)]:
  selected=[]
  for domain in ['Places365','LSDIR']:
   sids=sorted({r['source_id'] for r in pop if r['domain']==domain},key=lambda s:seed('monitor|'+s))[:25]
   for sid in sids:
    rr=[r for r in pop if r['source_id']==sid];trajectory=min(r['trajectory_id'] for r in rr);selected.extend(r['start_id'] for r in rr if r['trajectory_id']==trajectory)
  assert len(selected)==200;out[split+'_monitor']=selected
 assert not {store.by_id[s]['source_id'] for s in out['train_monitor']}&{store.by_id[s]['source_id'] for s in out['val_monitor']}
 return out

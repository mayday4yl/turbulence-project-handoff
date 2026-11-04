"""Frozen full V6 pool plus canonical CPU B1 raw-xi replay; no new data IDs."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from method_impl.training_io import BalancedOrder,load_training_pool
from scripts.runtime_utils import ROOT,sha
from scripts.generate_g4 import raw_noise


def full_pool():
    pool,identity=load_training_pool(require_v6=True)
    path=ROOT/'datasets/manifests/V6_TRAJECTORY40000.jsonl'
    fp=json.loads((ROOT/'datasets/manifests/V6_DATA_FINGERPRINT.json').read_text())
    assert sha(path)==fp['manifest_sha256'][path.name]
    trajectories={p['sample_id']:p for p in map(json.loads,path.read_text().splitlines())}
    assert len(trajectories)==len(pool)==40000
    merged=[]
    for p in pool:
        q=trajectories[p['sample_id']]
        assert all(q[k]==p[k] for k in ['source_id','clean_sha256','observation_sha256'])
        merged.append({**q,**p})
    return merged,{**identity,'trajectory_manifest_sha256':sha(path)}


def read_pair(row):
    arrays=[]
    for field in ['observation','clean']:
        path=row[field+'_path'];assert sha(path)==row[field+'_sha256']
        a=np.load(path,allow_pickle=False)
        assert a.shape==(3,256,256) and a.dtype==np.float32 and np.isfinite(a).all()
        arrays.append(torch.from_numpy(a))
    raw,hashes=raw_noise(row)
    assert hashes==row['raw_sha256'],'Raw physical realization identity changed'
    return (*arrays,raw[0][0],raw[1][0])


class ABPositionDataset(Dataset):
    def __init__(self,pool,seed,cursor,count,rank,world,microbatch):
        self.pool=pool;self.order=BalancedOrder(pool,seed);self.cursor=cursor;self.count=count
        self.rank=rank;self.world=world;self.microbatch=microbatch
    def __len__(self):return self.count
    def __getitem__(self,k):
        pos=self.cursor+(k//self.microbatch)*self.world*self.microbatch+self.rank*self.microbatch+k%self.microbatch
        idx=self.order.at(pos)
        return (*read_pair(self.pool[idx]),idx,pos)

"""Deterministic source/class balanced traversal of the accepted 160K pool."""
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from core import ROOT,derive,sha


def load_pool():
    gate=json.loads((ROOT/'evidence/DATA_GATE.json').read_text())
    assert gate['status']=='PASS'
    assert sha(gate['train_manifest'])==gate['train_manifest_sha256']
    pool=[json.loads(s) for s in open(gate['train_manifest'])]
    return pool,gate


class Order:
    def __init__(self,pool,seed):
        self.pool=pool;self.seed=seed;self.cached_epoch=None
        self.groups=defaultdict(lambda:defaultdict(list))
        for i,row in enumerate(pool):self.groups[row['category_id']][row['source_id']].append(i)
        self.groups={c:dict(v) for c,v in self.groups.items()}
        assert len(self.groups)==50 and all(len(v)==200 for v in self.groups.values())
        assert all(len(ids)==16 for group in self.groups.values() for ids in group.values())

    def cycle(self,epoch):
        if epoch==self.cached_epoch:return self.cached_order
        rng=random.Random(derive(self.seed,'balanced_cycle',epoch));cats=sorted(self.groups);byclass={}
        for category in cats:
            names=sorted(self.groups[category]);rng.shuffle(names);byclass[category]=[]
            for name in names:
                ids=list(self.groups[category][name]);rng.shuffle(ids);byclass[category].append(ids)
        result=[]
        for realization_pass in range(16):
            for source_position in range(200):
                categories=list(cats);rng.shuffle(categories)
                result.extend(byclass[c][source_position][realization_pass] for c in categories)
        assert len(result)==160000 and len(set(result))==160000
        self.cached_epoch=epoch;self.cached_order=result
        return result

    def at(self,position):
        epoch,offset=divmod(position,len(self.pool));return self.cycle(epoch)[offset]


class Positions(Dataset):
    def __init__(self,pool,seed,cursor,count,rank,world,microbatch):
        self.pool=pool;self.order=Order(pool,seed);self.cursor=cursor;self.count=count
        self.rank=rank;self.world=world;self.microbatch=microbatch

    def __len__(self):return self.count

    def __getitem__(self,index):
        local_batch,k=divmod(index,self.microbatch)
        position=self.cursor+local_batch*self.world*self.microbatch+self.rank*self.microbatch+k
        selected=self.order.at(position);row=self.pool[selected]
        y=np.load(row['observation_path'],allow_pickle=False)
        c=np.load(row['clean_path'],allow_pickle=False)
        assert y.shape==c.shape==(3,256,256) and y.dtype==c.dtype==np.float32
        return torch.from_numpy(y),torch.from_numpy(c),selected,position,row['g']


def validation_groups(full):
    gate=json.loads((ROOT/'evidence/DATA_GATE.json').read_text())
    assert sha(gate['val_manifest'])==gate['val_manifest_sha256']
    rows=[json.loads(s) for s in open(gate['val_manifest'])]
    by_source=defaultdict(list)
    for row in rows:by_source[row['source_id']].append(row)
    monitor=set(json.loads((ROOT/'protocol/MONITOR_SELECTION.json').read_text())['sources'])
    groups=[]
    for g in [.5,1,2,3,4]:
        for source in sorted(by_source):
            if not full and source not in monitor:continue
            group=sorted(by_source[source],key=lambda r:r['sample_id'])
            assert len(group)==4
            groups.append((g,source,group))
    return groups

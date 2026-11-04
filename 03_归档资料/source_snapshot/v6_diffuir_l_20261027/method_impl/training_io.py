"""Deterministic no-padding traversal and atomic full-state saves for V6 D0."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math,os,random
import numpy as np
import torch
from torch.utils.data import Dataset
from method_impl.d0 import derive_seed
from scripts.prepare_sources import ROOT,OLD,sha,rows

def load_training_pool(require_v6=True):
    ready=ROOT/'datasets/DATA_READY.json'
    if require_v6:
        d=json.loads(ready.read_text());assert d['status']=='PASS' and d['train_pairs']==40000
        fp=ROOT/'datasets/manifests/V6_DATA_FINGERPRINT.json';assert sha(fp)==d['fingerprint_sha256']
        identity=json.loads(fp.read_text());p=fp.parent/'V6_G4_PAIRS40000.jsonl'
        assert sha(p)==identity['manifest_sha256'][p.name]
        return rows(p),{'population':'V6_Train10000_40000_pairs','fingerprint_sha256':sha(fp),'pair_manifest_sha256':sha(p)}
    source={x['source_id']:x for x in rows(OLD/'source_manifest.jsonl')}
    pool=[]
    for r in rows(OLD/'sample_manifest.jsonl'):
        p=r['sample_plan_row']
        if p['split']!='train':continue
        pool.append({**p,'category_id':source[p['source_id']]['category_id'],'origin':'original_v5',
            'clean_path':str(OLD/p['clean_ref']),'observation_path':str(OLD/p['observation_ref']),'observation_sha256':r['observation_sha256']})
    return pool,{'population':'original_Train2000_8000_pairs_ENGINEERING_ONLY','pair_manifest_sha256':sha(OLD/'sample_manifest.jsonl')}

class BalancedOrder:
    def __init__(self,pool,seed):
        self.pool=pool;self.seed=seed;self._cycle=None;self._order=None
        self.groups=defaultdict(lambda:defaultdict(list))
        for index,row in enumerate(pool):self.groups[row['category_id']][row['source_id']].append(index)
        self.groups={c:dict(sources) for c,sources in self.groups.items()}
        assert len(self.groups)==50 and len({len(s) for s in self.groups.values()})==1
        assert all(len(ids)==4 for group in self.groups.values() for ids in group.values())
    def cycle(self,cycle):
        if cycle==self._cycle:return self._order
        rng=random.Random(derive_seed(self.seed,'balanced_cycle',cycle));cats=sorted(self.groups)
        byclass={}
        for c in cats:
            names=sorted(self.groups[c]);rng.shuffle(names);byclass[c]=[]
            for name in names:
                indices=list(self.groups[c][name]);rng.shuffle(indices);byclass[c].append(indices)
        result=[];count=len(next(iter(byclass.values())))
        for realization_pass in range(4):
            for source_position in range(count):
                order=list(cats);rng.shuffle(order)
                result.extend(byclass[c][source_position][realization_pass] for c in order)
        assert len(result)==len(self.pool) and len(set(result))==len(result)
        self._cycle=cycle;self._order=result;return result
    def at(self,position):
        cycle,offset=divmod(position,len(self.pool));return self.cycle(cycle)[offset]

class PositionDataset(Dataset):
    def __init__(self,pool,seed,cursor,count,rank,world,microbatch):
        self.pool=pool;self.order=BalancedOrder(pool,seed);self.cursor=cursor;self.count=count
        self.rank=rank;self.world=world;self.microbatch=microbatch;self.verified=set()
    def __len__(self):return self.count
    def __getitem__(self,k):
        position=self.cursor+(k//self.microbatch)*self.world*self.microbatch+self.rank*self.microbatch+k%self.microbatch
        index=self.order.at(position);row=self.pool[index];values=[]
        for field in ['observation','clean']:
            path=row[field+'_path']
            if path not in self.verified:
                assert sha(path)==row[field+'_sha256'];self.verified.add(path)
            a=np.load(path,allow_pickle=False)
            assert a.shape==(3,256,256) and a.dtype==np.float32 and np.isfinite(a).all()
            if field=='clean':assert a.min()>=0 and a.max()<=1
            values.append(torch.from_numpy(a))
        return values[0],values[1],index,position

def learning_rate(update,total,peak=2e-5):
    assert 1<=update<=total
    warmup=max(1,int(.02*total))
    if update<=warmup:return peak*update/warmup
    ratio=(update-warmup)/(total-warmup)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*ratio)))

def atomic_torch_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return sha(path)

def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def new_snapshot(model,opt,ema,*,update,cursor,rng,config,identity):
    return {'schema':'V6_D0_FULL_v1','network':cpu_tree(model.network.state_dict()),'ema':cpu_tree(ema.state_dict()),
        'optimizer':cpu_tree(opt.state_dict()),'scheduler':{'last_update':update,'total_updates':config['updates'],'peak':2e-5,'warmup_fraction':.02,'end_ratio':.1},
        'update':update,'cursor':cursor,'images_seen':cursor,'rng_by_rank':rng,'config':config,'identity':identity,
        'native_buffers':{n:t.detach().cpu().clone() for n,t in model.native.named_buffers()},
        'sampling_buffers':{n:t.detach().cpu().clone() for n,t in model.sampling.named_buffers()},'scaler':None}

def restore_snapshot(payload,model,opt,ema):
    assert payload['schema']=='V6_D0_FULL_v1'
    model.network.load_state_dict(payload['network'],strict=True)
    for branch in ['native','sampling']:
        actual=dict(getattr(model,branch).named_buffers());saved=payload[branch+'_buffers']
        assert set(actual)==set(saved) and all(torch.equal(actual[n].cpu(),t) for n,t in saved.items()),'Diffusion schedule drift'
    opt.load_state_dict(payload['optimizer']);ema.model.load_state_dict(payload['ema']['model'],strict=True)
    assert payload['ema']['beta']==.995;ema.updates=payload['ema']['updates']
    assert ema.updates==payload['update']

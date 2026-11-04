"""CPU-only immutable subset/config/source preparation, no old writes."""
from lib import *
import numpy as np

def main():
    parent=read(PARENT/'configs/FROZEN_RUNTIME.json');data=Path(parent['dataset_root'])
    assert read(data/'A100_RUNTIME_READY.json')['status']=='PASS'
    c={'purpose':PURPOSE,'allowed_gpus':[4,5],'seed':20261013,'max_updates':1000,'effective_batch':32,'peak_lr':5e-5,'min_lr':1e-6,'warmup_updates':100,'network':parent['network'],'physics':parent['physics'],'asset_dir':parent['asset_dir'],'asset_sha256':parent['asset_sha256'],'dataset_root':str(data),'precision':'fp32','cache_bytes':1<<30,'pilot_disk_limit_bytes':20<<30,'account_quota_bytes':2_000_000_000_000,'account_safety_bytes':100<<30,'checkpoint_interval':100,'milestones':[250,500,750,1000],'primary_eval_K':4,'weight':'online','refinement':False,'Test':False,'training_grid':'K=2/3/4 equal probability; internal equal-grid jitter +/-0.2 interval; full endpoints','loss':'path mean MAE(clean)+0.2 MAE(F(pred,original G4),Y4)','state_update':'Q+(F(pred,next_g)-F(pred,current_g))','minimum_grid_interval':1/32}
    c['peak_lr']=5e-6
    c.update({'pilot_storage_roots':[str(ROOT.parent/'v4_followup_pilot_train15000_20261013'),str(ROOT)],
              'diagnostic_interval':20,'timebox_policy':{'training_seconds':18000,'gpu_seconds':20400,'total_seconds':21600},
              'numerical_stop':{'physics_L1':10.,'preclip_gradient_norm':1e6,'consecutive_updates':3},
              'retention':'latest+previous full states; highest common online milestone plus faster-arm future milestones'})
    freeze(ROOT/'CONFIG.json',c)
    ss=rows(data/'source_manifest.jsonl');tt=rows(data/'trajectory_manifest.jsonl');byid={s['source_id']:s for s in ss};selected={}
    for split,n in [('train',500),('val',100)]:
        selected[split]=set()
        for d in ['Places365','LSDIR']:
            ids=sorted([s['source_id'] for s in ss if s['split']==split and s['dataset_source']==d],key=lambda s:seed(f'pilot-source|20261013|{split}|{d}|{s}'))[:n]
            assert len(ids)==n;selected[split].update(ids)
    sub={'parent_dataset':data.name,'source_selection_seed':20261013,'train':[],'val':[]}
    for t in sorted(tt,key=lambda x:x['trajectory_id']):
        if t['split'] not in selected or t['source_id'] not in selected[t['split']]:continue
        s=byid[t['source_id']];assert s['split']==t['split'] and s['clean']==t['arrays']['clean']
        sub[t['split']].append({'start_id':t['trajectory_id']+'::G4','trajectory_id':t['trajectory_id'],'source_id':t['source_id'],'split':t['split'],'domain':s['dataset_source'],'G':4,'clean':t['arrays']['clean'],'state':t['arrays']['X_4']})
    assert len(sub['train'])==4000 and len(sub['val'])==400
    for split in ['train','val']:
        assert len(set(x['start_id'] for x in sub[split]))==len(sub[split])
    freeze(ROOT/'SUBSET.json',sub)
    assert sha(ROOT/'SUBSET.json')==sha(ROOT.parent/'v4_followup_pilot_train15000_20261013/SUBSET.json'),'Retry population must exactly match the first attempt'
    # Full subset content check before launch: parent arrays are never modified.
    from v4.data import read_array
    verified=set()
    for split in ['train','val']:
        for r in sub[split]:
            for field in ['clean','state']:
                a=r[field];key=(a['ref'],a['sha256'])
                if key in verified:continue
                read_array(data,a);verified.add(key)
    identity={'purpose':PURPOSE,'config_hash':canonical(c),'subset_sha256':sha(ROOT/'SUBSET.json'),'common_init_sha256':sha(PARENT/'protocol/common_init.pt'),'parent_source_hashes':{str(p.relative_to(CANDIDATE)):sha(p) for directory in ['nafnet_v4','v4','physical_bridge'] for p in sorted((CANDIDATE/directory).glob('*.py'))},'parent_dataset_hashes':{n:sha(data/n) for n in ['ASSET_LOCK.json','BUILD_IDENTITY.json','source_manifest.jsonl','trajectory_manifest.jsonl','normalization_train8000.json']},'asset_sha256':c['asset_sha256'],'parent_U20000_not_used_for_pilot':True}
    assert identity['common_init_sha256']=='17aa7aca8cbe5fcff6d5af9631cb2c92cf68fc1c13e8e37f9ee9d5d53b0b9d8b'
    for n,h in c['asset_sha256'].items():assert sha(Path(c['asset_dir'])/n)==h
    freeze(ROOT/'IDENTITY.json',identity);freeze(ROOT/'SOURCE_LOCK.json',{'files':sources()})
    atomic(ROOT/'CPU_PREPARE_COMPLETE.json',{'status':'PASS','verified_distinct_arrays':len(verified),'train_inputs':4000,'val_inputs':400,'train_sources':1000,'val_sources':200,'identity_sha256':sha(ROOT/'IDENTITY.json'),'source_lock_sha256':sha(ROOT/'SOURCE_LOCK.json'),'time':time.time()})

if __name__=='__main__':main()

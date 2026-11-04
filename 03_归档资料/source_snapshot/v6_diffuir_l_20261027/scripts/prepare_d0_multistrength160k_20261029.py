"""Freeze the user-approved full Cartesian 10k Clean x 4 xi x 4 g cache."""
import collections,json,time
from pathlib import Path
from scripts.runtime_utils import ROOT,sha,write_json
from scripts.prepare_d0_multistrength_20261029 import rows,write_rows,OLD,PARENT,PARENT_SHA
EX=ROOT/'execution_20261029_d0_multistrength160k'
PREVIOUS=ROOT/'execution_20261029_d0_multistrength'
def main():
    EX.mkdir(parents=True,exist_ok=True);assert not (EX/'DATA_PLAN.json').exists()
    assert sha(PARENT)==PARENT_SHA
    original=ROOT/'datasets/manifests/V6_G4_PAIRS40000.jsonl';fp=json.load(open(original.parent/'V6_DATA_FINGERPRINT.json'))
    assert sha(original)==fp['manifest_sha256'][original.name]
    pool=rows(original);plans={r['sample_id']:r for r in rows(original.parent/'TRAJECTORY_PLAN40000.jsonl')}
    assert len(pool)==40000 and len({r['source_id'] for r in pool})==10000
    records=[]
    for row in pool:
        p=plans[row['sample_id']]
        for k in ['source_id','realization_index','clean_sha256']:assert p[k]==row[k]
        for g in [1,2,3,4]:
            records.append({**p,**row,'g':g,'pair_id':row['sample_id']+f'_G{g}',
                'old_g4_path':row['observation_path'],'old_g4_sha256':row['observation_sha256'],
                'observation_path':str(EX/'data/train'/f'G{g}'/(row['sample_id']+'.npy'))})
    validation=rows(OLD/'GENERATED_INPUTS.jsonl');assert len(validation)==1000
    assert not {r['source_id'] for r in records}&{r['source_id'] for r in validation}
    for r in validation:r['states']['0.5']={'path':str(EX/'data/val/G0.5'/(r['sample_id']+'.npy'))}
    write_rows(EX/'TRAIN_PLAN.jsonl',records);write_rows(EX/'VAL_PLAN.jsonl',validation)
    write_json(EX/'DATA_PLAN.json',{'status':'PLANNED','training_counts':{g:40000 for g in [1,2,3,4]},
        'clean_sources':10000,'raw_realizations':40000,'realizations_per_source':4,'observations_per_source':16,
        'training_g':[1,2,3,4],'validation_g':[.5,1,2,3,4],'same_realizations_at_every_g':True,
        'train_plan_sha256':sha(EX/'TRAIN_PLAN.jsonl'),'val_plan_sha256':sha(EX/'VAL_PLAN.jsonl'),
        'G0_5_training_count':0,'test_used':False})
    legacy=ROOT/'execution_20261028_d0_microopt/STOP_REQUEST.json'
    write_json(EX/'PRE_EXECUTION_SNAPSHOT.json',{'unix':time.time(),'parent':str(PARENT),'parent_sha256':PARENT_SHA,
        'legacy_stop_path':str(legacy),'legacy_stop_sha256':sha(legacy),
        'previous_policy':json.load(open(ROOT/'gates/EXECUTION_POLICY_CURRENT.json')),
        'old_data_fingerprint_sha256':sha(original.parent/'V6_DATA_FINGERPRINT.json'),
        'old_train_manifest_sha256':sha(original),'old_val_manifest_sha256':sha(OLD/'GENERATED_INPUTS.jsonl')})
    print('Full matched 160K plan frozen',flush=True)
if __name__=='__main__':main()

"""Read-only data/source audit; writes only the independent V0.31 artifacts."""
from collections import Counter
import json
import os
import platform
from pathlib import Path
import numpy as np
import torch
from v031.runtime import sha256, save_json, verify_source

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root/'configs/stage1_acceptance.json').read_text())
data = Path(cfg['dataset_root'])
def rows(name):
    return [json.loads(line) for line in (data/name).read_text().splitlines() if line]
assert sha256(data/'hash_manifest.json') == cfg['data_hash_manifest_sha256']
hashes = json.loads((data/'hash_manifest.json').read_text())['files']
names = ['source_manifest.jsonl','trajectory_manifest.jsonl','start_manifest.jsonl',
         'dataset_summary.json','integrity_report.json','integrity_report.md',
         'subsets/subset_summary.json','subsets/overfit_16traj_starts.jsonl']
metadata = {n:sha256(data/n) for n in names}
assert all(metadata[n] == hashes[n] for n in names)
sources, trajectories, starts = [rows(n) for n in names[:3]]
assert (len(sources),len(trajectories),len(starts)) == (100,400,1600)
assert len({s['source_id'] for s in sources}) == 100
assert len({r['trajectory_id'] for r in trajectories}) == 400
assert len({r['start_id'] for r in starts}) == 1600
assert set(Counter(t['source_id'] for t in trajectories).values()) == {4}
by_traj = {t['trajectory_id']:t for t in trajectories}
assert all(t['severity_values']==[4,3,2,1] for t in trajectories)
for s in starts:
    g = s['initial_G']; t = by_traj[s['trajectory_id']]
    assert s['severity_path']==list(range(g,0,-1)) and s['num_predictions']==g
    assert s['initial_state_ref']==t[f'x{g}_ref'] and s['clean_ref']==t['clean_ref']
    assert s['source_id']==t['source_id']
for t in trajectories:
    for file, expected in t['simulator_identity'].items():
        if file not in globals().get('pinned_assets',{}):
            globals().setdefault('pinned_assets',{})[file] = sha256(file)
        assert pinned_assets[file] == expected
subset=rows('subsets/overfit_16traj_starts.jsonl')
assert len(subset)==64
lookup={s['start_id']:s for s in starts}
assert all(lookup[s['start_id']]==s for s in subset)
sample_ids=[s['trajectory_id'] for s in subset if s['initial_G']==4][:2]
sampled={}
for tid in sample_ids:
    t=by_traj[tid]
    for key in ['clean_ref','x4_ref','x3_ref','x2_ref','x1_ref','epsilon_zernike_ref','epsilon_tilt_ref']:
        name=t[key]; p=(data/name).resolve()
        assert p.is_relative_to(data.resolve())
        actual=sha256(p); assert actual==hashes[name]
        a=np.load(p,allow_pickle=False)
        expected=(36,512,512) if key=='epsilon_zernike_ref' else ((512,512,2) if key=='epsilon_tilt_ref' else (3,256,256))
        assert a.shape==expected and a.dtype==np.float32 and np.isfinite(a).all()
        sampled[name]={'sha256':actual,'shape':list(a.shape),'dtype':str(a.dtype)}
summary=json.loads((data/'dataset_summary.json').read_text())
assert summary['source_overlap']=={'test':[],'val':[]}
accept_path=data.parents[1]/'audits/independent_20260916T115840Z/independent_acceptance.json'
accept=json.loads(accept_path.read_text())
assert accept['status']=='independent_dataset_acceptance_passed'
assert accept['replay']['max_abs']==0 and accept['replay']['rendered_states']==1600
manifest={
 'method_version':'V0.31','dataset_id':cfg['dataset_id'],'dataset_root':str(data),
 'project_root':str(root),'stage':1,'status':'data_source_verified',
 'counts':{'sources':100,'trajectories':400,'states':1600,'starts':1600},
 'metadata_hashes':metadata,'hash_manifest_sha256':sha256(data/'hash_manifest.json'),
 'sample_trajectory_ids':sample_ids,'sampled_arrays':sampled,
 'base_hashes':verify_source(cfg),'simulator_assets':pinned_assets,
 'archived_independent_acceptance':{'path':str(accept_path),'sha256':sha256(accept_path),'status':accept['status'],'replay_max_abs':0,'replay_states':1600},
 'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
 'formal_training_started':False,'restoration_optimizer_updates':0}
save_json(root/os.environ.get('V031_INSPECTION_OUTPUT','artifacts/run_manifest.json'),manifest)
print(json.dumps({k:manifest[k] for k in ['status','counts','sample_trajectory_ids','versions']}))

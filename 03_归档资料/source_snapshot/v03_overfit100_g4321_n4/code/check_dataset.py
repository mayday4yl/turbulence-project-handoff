#!/usr/bin/env python3
"""Independent CPU-only delivery check; does not initialize or import an NPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from dataset import OverfitDataset, rows


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve()
    manifest=json.loads((root/'hash_manifest.json').read_text())
    require(set(manifest['files'])=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='hash_manifest.json'},'Hash coverage mismatch')
    for ref,wanted in manifest['files'].items():
        require(sha(root/ref)==wanted,'Hash mismatch: '+ref)
    reader=OverfitDataset(root)
    require((len(reader.sources),len(reader.trajectories),len(reader))==(100,400,1600),'Counts mismatch')
    source_counts=Counter(r['source_id'] for r in reader.trajectories.values())
    require(set(source_counts.values())=={4} and len(source_counts)==100,'Per-source count mismatch')
    for i,start in enumerate(reader.starts):
        item=reader[i]
        g=start['initial_G']
        require(g in (1,2,3,4),'Unknown actual severity')
        require(start['severity_path']==list(range(g,0,-1)) and start['num_predictions']==g,'Path semantics mismatch')
        tr=reader.trajectories[start['trajectory_id']]
        require(start['initial_state_ref']==tr[f'x{g}_ref'],'Starting observation mismatch')
        require(start['source_id']==tr['source_id'] and start['clean_ref']==tr['clean_ref'],'Clean/source reference mismatch')
        require(item['C'].dtype==item['Y'].dtype==np.float32,'Dtype mismatch')
    for trajectory in reader.trajectories:
        reader.diagnostic_xi(trajectory)
    replay=rows(root/'replay_metrics.jsonl')
    require(len(replay)==1600 and len({(r['trajectory_id'],r['g']) for r in replay})==1600,'Replay coverage mismatch')
    require(all(r['pass'] and r['max_abs']<=1e-6 and r['MSE']<=1e-12 for r in replay),'Replay failed')
    require({(r['trajectory_id'],r['g']) for r in replay}=={(t,g) for t in reader.trajectories for g in (1,2,3,4)},'Missing trajectory/severity replay')
    seeds=rows(root/'seed_replay_metrics.jsonl')
    require(len(seeds)==400 and all(r['passed'] for r in seeds),'Seed replay failed')
    actual_seeds=[r[k] for r in reader.trajectories.values() for k in ('seed_zernike','seed_tilt')]
    require(len(set(actual_seeds))==800,'Repeated seed')
    previous=set()
    subsets={}
    for count in (1,16,400):
        ts=rows(root/f'subsets/overfit_{count}traj_trajectories.jsonl')
        ss=rows(root/f'subsets/overfit_{count}traj_starts.jsonl')
        ids={r['trajectory_id'] for r in ts}
        require(len(ids)==len(ts)==count and len(ss)==4*count,'Subset count mismatch')
        require(previous<=ids<=set(reader.trajectories),'Subset not nested')
        require({r['trajectory_id'] for r in ss}==ids,'Subset start identities mismatch')
        previous=ids
        subsets[str(count)]={'trajectories':count,'starts':len(ss),'sources':len({r['source_id'] for r in ts})}
    report={'status':'passed','checked_hash_files':len(manifest['files']),
            'hash_manifest_sha256':sha(root/'hash_manifest.json'),'sources':100,'trajectories':400,'starts':1600,
            'all_authoritative_arrays_decoded':2500,'replay_states':len(replay),'seed_replayed_realizations':len(seeds),
            'max_abs':max(r['max_abs'] for r in replay),'max_MSE':max(r['MSE'] for r in replay),'subsets':subsets,
            'scope':'Independent CPU verification of files and archived full NPU replay results; no new NPU inference'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()

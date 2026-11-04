"""Quarantine volatile future outputs when optimizer state rolls back."""
import json
from pathlib import Path
import re
import time
from .common import atomic_json,append_json


def reconcile(run,update):
    run=Path(run);stamp=f'{time.time_ns()}_resume_U{update:05d}'
    archive=run/'recovery_archive'/stamp;moved=[]
    for folder,pattern in [('eval_checkpoints',r'update_(\d+)'),('eval_queue',r'_u(\d+)'),('evaluations',r'^U(\d+)')]:
        for path in list((run/folder).glob('*')):
            match=re.search(pattern,path.name)
            if match and int(match[1])>update:
                dst=archive/folder/path.name;dst.parent.mkdir(parents=True,exist_ok=True);path.rename(dst)
                moved.append(str(path.relative_to(run)))
    best=run/'BEST_VAL.json'
    if best.exists() and json.loads(best.read_text()).get('update',0)>update:
        archive.mkdir(parents=True,exist_ok=True);best.rename(archive/'BEST_VAL.json');moved.append('BEST_VAL.json')
    if not best.exists():
        candidates=[]
        for folder in (run/'evaluations').glob('U*'):
            try:
                u=int(folder.name[1:]);agg=folder/'raw/val2000/aggregate.json'
                if u>update or u not in set(range(2000,20001,2000))|{5000}:continue
                if not (folder/'raw/val2000/COMMIT.json').exists():continue
                value=json.loads(agg.read_text());identity=json.loads((folder/'raw/val2000/IDENTITY.json').read_text())
                checkpoint=run/'eval_checkpoints'/f'update_{u:05d}.pt'
                if not checkpoint.exists():continue
                candidates.append({'criterion':'RAW_OFF_full_val2000_mean_PSNR','psnr':value['overall']['psnr'],
                    'update':u,'checkpoint':str(checkpoint),'checkpoint_sha256':value['checkpoint_sha256'],
                    'tie_tolerance_dB':1e-6,'tie_choose':'earlier','fixed_grid':sorted(set(range(2000,20001,2000))|{5000}),
                    'refinement_used':False,'test_used':False,'evaluation_execution':identity.get('execution'),
                    'reconciled_after_resume':True})
            except (OSError,KeyError,ValueError):continue
        if candidates:
            peak=max(x['psnr'] for x in candidates)
            atomic_json(best,min((x for x in candidates if peak-x['psnr']<=1e-6),key=lambda x:x['update']))
    if moved:
        append_json(run/'recovery.jsonl',{'time':time.time(),'event':'quarantine_future_outputs_after_rollback',
            'resume_update':update,'archive':str(archive),'moved':moved})
    return moved

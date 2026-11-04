"""Independent CPU audit of saved full predictions; no neural models."""
import concurrent.futures,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
from core import ROOT,derive,sha,write

def check(row):
    assert row['nfe']==100 and row['group_batch']==4
    assert row['group_seed']==derive(843001,row['source_id'],'four_realizations_native_SDE')
    path=Path(row['prediction']);assert sha(path)==row['prediction_sha256']
    assert sha(row['input_path'])==row['input_sha256']
    p=np.load(path,allow_pickle=False).astype(np.float64)
    c=np.load(row['clean_path'],allow_pickle=False).astype(np.float64)
    assert p.shape==c.shape==(3,256,256) and np.isfinite(p).all()
    actual={'psnr':-10*math.log10(max(float(np.mean((p-c)**2)),1e-30)),
        'mae':float(np.abs(p-c).mean()),
        'ssim':float(structural_similarity(np.moveaxis(c,0,2),np.moveaxis(p,0,2),
            data_range=1.,channel_axis=2,gaussian_weights=True,win_size=11,sigma=1.5,
            use_sample_covariance=False,K1=.01,K2=.03))}
    errors={key:abs(actual[key]-row['metrics'][key]) for key in actual}
    assert all(value<=1e-8 for value in errors.values()),(row['sample_id'],errors)
    assert all(math.isfinite(v) for v in row['metrics'].values())
    return errors

def validate(path):
    summary=json.loads((path/'SUMMARY.json').read_text());assert summary['full'] and summary['count']==5000
    records=[]
    for file in sorted(path.glob('rank*.jsonl')):records.extend(json.loads(s) for s in file.read_text().splitlines())
    assert len(records)==5000 and len({(r['sample_id'],r['g']) for r in records})==5000
    assert Counter(r['g'] for r in records)=={.5:1000,1:1000,2:1000,3:1000,4:1000}
    errors={'psnr':0.,'mae':0.,'ssim':0.}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for result in pool.map(check,records):
            for key,value in result.items():errors[key]=max(errors[key],value)
    mean_errors={}
    for g in [.5,1,2,3,4]:
        for key in ['psnr','ssim','lpips','mae']:
            value=float(np.mean([r['metrics'][key] for r in records if r['g']==g]))
            error=abs(value-summary['by_g'][f'{g:g}'][key]);assert error<=1e-10
            mean_errors[key]=max(mean_errors.get(key,0.),error)
    return {'update':summary['update'],'count':5000,'prediction_and_input_SHA_all_checked':True,
        'recomputed_per_image_maxabs':errors,'all_mean_maxabs':mean_errors,
        'LPIPS_scope':'all records finite; complete means reaggregated; no new LPIPS network calls'}

if __name__=='__main__':
    run=ROOT/'runs/formal_until_20261103_noon'
    done=json.loads((run/'TRAINING_AND_VALIDATION_COMPLETE.json').read_text());assert done['status']=='PASS'
    best=json.loads((run/'BEST.json').read_text());last=json.loads((run/'LATEST.json').read_text())
    results=[validate(run/'validation'/f'U{update:08d}_full') for update in sorted({best['update'],last['update']})]
    cfg=json.loads((run/'CONFIG.json').read_text());rows=[json.loads(s) for s in (run/'UPDATES_rank0.jsonl').read_text().splitlines()]
    late=[r['update'] for r in rows if r['started_unix']>=cfg['deadline_unix']]
    write(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',{'status':'PASS' if not late else 'REVIEW_DEADLINE_BOUNDARY',
        'full_outputs':results,'training_updates':len(rows),'post_deadline_started_updates':late,
        'restoration_network_calls':0,'new_metric_network_calls':0,'auditor_sha256':sha(Path(__file__))})
    print('INDEPENDENT_FINAL_OUTPUTS_AUDIT_COMPLETE',flush=True)

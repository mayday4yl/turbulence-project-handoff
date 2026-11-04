"""Fixed-ID, fixed-noise EMA evaluation on the unchanged V5 Val1000 only."""
from collections import defaultdict
import importlib.util,json,math,time,os
import numpy as np
import torch
from method_impl.d0 import derive_seed
from scripts.prepare_sources import ROOT,OLD,sha,rows,write_rows
from scripts.runtime_utils import write_json

BASE_SEED=2026102707

def validation_rows():
    sources={r['source_id']:r for r in rows(OLD/'source_manifest.jsonl')}
    selected=[]
    for row in rows(OLD/'sample_manifest.jsonl'):
        p=row['sample_plan_row']
        if p['split']=='val':selected.append({**p,'category_id':sources[p['source_id']]['category_id'],'observation_sha256':row['observation_sha256']})
    selected.sort(key=lambda p:p['sample_id']);assert len(selected)==1000 and len({x['source_id'] for x in selected})==250
    return selected

def observation_subset(pool):
    byclass=defaultdict(list)
    for p in pool:byclass[p['category_id']].append(p)
    small=[]
    for c in sorted(byclass):
        ranking=sorted(byclass[c],key=lambda p:derive_seed(BASE_SEED,'val100_observation',p['sample_id']))
        small.extend(ranking[:2])
    assert len(small)==100;return sorted(small,key=lambda p:p['sample_id'])

def evaluate_rank(model,ema,device,out,*,rank,world,full=True):
    out.mkdir(parents=True,exist_ok=True)
    metric_path=ROOT.parent/'v5_concertormer_10k_20261021/training/image_metrics.py'
    spec=importlib.util.spec_from_file_location('v6_readonly_image_metrics',metric_path);metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    pool=validation_rows();visual_ids={p['sample_id'] for p in observation_subset(pool)}
    pool=pool if full else observation_subset(pool)
    original=model.sampling.model;model.sampling.model=ema.model
    results=[];started=time.time()
    try:
        with torch.no_grad():
            for p in pool[rank::world]:
                assert sha(OLD/p['clean_ref'])==p['clean_sha256'] and sha(OLD/p['observation_ref'])==p['observation_sha256']
                c=torch.from_numpy(np.load(OLD/p['clean_ref'],allow_pickle=False)[None]).to(device)
                x=torch.from_numpy(np.load(OLD/p['observation_ref'],allow_pickle=False)[None]).to(device)
                seed=derive_seed(BASE_SEED,p['source_id'],p['sample_id'],'outer',0)
                y=model.sample_clean(x,seed=seed,differentiable=False)
                assert bool(torch.isfinite(y).all()) and y.shape==c.shape
                yp=y.cpu().numpy().astype(np.float64);cp=c.cpu().numpy().astype(np.float64);xp=x.cpu().numpy().astype(np.float64)
                mse=float(np.mean((yp-cp)**2));input_mse=float(np.mean((xp-cp)**2))
                ssim,lpips=metrics.image_metrics(y,c)
                result={'sample_id':p['sample_id'],'source_id':p['source_id'],'category_id':p['category_id'],'seed':seed,
                    'psnr':-10*math.log10(mse),'input_psnr':-10*math.log10(input_mse),'mae':float(np.abs(yp-cp).mean()),
                    'ssim':ssim[0],'lpips':lpips[0],'output_min':float(y.min()),'output_max':float(y.max()),'nfe':3}
                assert all(math.isfinite(result[k]) for k in ['psnr','mae','ssim','lpips'])
                if full and p['sample_id'] in visual_ids:
                    from PIL import Image
                    visual=out/'fixed_visual100';visual.mkdir(exist_ok=True)
                    filename=f'{seed:016x}'
                    array=visual/(filename+'.npy');temp=array.with_suffix('.npy.tmp')
                    with temp.open('wb') as f:np.save(f,y.cpu().numpy()[0],allow_pickle=False)
                    os.replace(temp,array)
                    # Clamping applies only to PNG display; metrics and NPY keep actual outputs.
                    tiles=[np.rint(np.clip(a[0].transpose(1,2,0),0,1)*255).astype(np.uint8) for a in [xp,yp,cp]]
                    Image.fromarray(np.concatenate(tiles,axis=1)).save(visual/(filename+'.png'))
                    result['visual']={'raw_npy':str(array.relative_to(out)),'raw_npy_sha256':sha(array),
                        'png':str((visual/(filename+'.png')).relative_to(out)),'png_order':['input_Y4','EMA_S3','clean_GT']}
                results.append(result)
    finally:model.sampling.model=original
    write_rows(out/f'rank{rank}.jsonl',results)
    write_json(out/f'rank{rank}_status.json',{'status':'PASS','count':len(results),'seconds':time.time()-started,'metrics_source_sha':sha(metric_path)})

def summarize_validation(out,world,full):
    data=[]
    for rank in range(world):
        assert json.loads((out/f'rank{rank}_status.json').read_text())['status']=='PASS'
        data+=rows(out/f'rank{rank}.jsonl')
    expected=validation_rows();expected=expected if full else observation_subset(expected)
    assert len(data)==len(expected) and {x['sample_id'] for x in data}=={x['sample_id'] for x in expected}
    data.sort(key=lambda x:x['sample_id']);keys=['psnr','mae','ssim','lpips','input_psnr']
    mean=lambda population:{k:float(np.mean([x[k] for x in population])) for k in keys}
    summary={'status':'PASS','count':len(data),'full_val1000':full,'means':mean(data),
        'per_category':{str(c):mean([x for x in data if x['category_id']==c]) for c in sorted({x['category_id'] for x in data})},
        'selection_eligible':full,'selection_scope':'D0 EMA actual S3 Clean','validation_base_seed':BASE_SEED,
        'noise_protocol':'sha256(base_seed, source_id, sample_id, outer, 0), identical per sample across checkpoints',
        'test_used':False,'input_identity_sha':sha(OLD/'sample_manifest.jsonl')}
    write_rows(out/'PER_IMAGE.jsonl',data);write_json(out/'SUMMARY.json',summary);return summary

def choose_candidate(candidate,best):
    if best is None:return True,'first complete Val1000'
    difference=candidate['means']['psnr']-best['means']['psnr']
    if abs(difference)>=.02:return difference>0,'PSNR difference at least 0.02 dB'
    if candidate['means']['lpips']<best['means']['lpips']:return True,'PSNR within 0.02 dB tie; lower LPIPS'
    return False,'PSNR within 0.02 dB tie; LPIPS not lower, retain earlier checkpoint'

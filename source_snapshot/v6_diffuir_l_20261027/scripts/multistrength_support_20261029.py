"""New data/evaluation bookkeeping around unchanged native D0 mathematics."""
import collections,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from scripts.runtime_utils import ROOT,sha,write_json
from scripts.prepare_d0_multistrength160k_20261029 import EX,OLD,PARENT,PARENT_SHA,rows,write_rows
from method_impl.d0 import derive_seed
from method_impl.evaluation import BASE_SEED

BASE=95544
FORMAL='V6_D0_U95544_multistrength160k_formal01'
G_VALUES=[.5,1,2,3,4]
METRICS=['psnr','ssim','lpips','mae','input_psnr','input_ssim','input_lpips','input_mae']
def freeze():
    for n,h in json.load(open(EX/'SOURCE_MANIFEST.json'))['files'].items():assert sha(ROOT/n)==h,('Source drift',n)
    assert sha(EX/'CONFIG.json')==json.load(open(EX/'CONFIG_MANIFEST.json'))['config_sha256']
def load_pool():
    gate=json.load(open(EX/'DATA_GATE.json'));assert gate['status']=='PASS'
    assert sha(EX/'TRAIN_MANIFEST.jsonl')==gate['train_manifest_sha256']
    pool=rows(EX/'TRAIN_MANIFEST.jsonl')
    assert len(pool)==160000 and collections.Counter(r['g'] for r in pool)=={g:40000 for g in [1,2,3,4]}
    return pool,{'population':'V6_Train10000_xi40000_G1234_equal_160000_pairs','pair_manifest_sha256':gate['train_manifest_sha256'],
        'fingerprint_sha256':sha(EX/'DATA_GATE.json'),'source_manifest_sha256':sha(EX/'SOURCE_MANIFEST.json'),
        'physical_devices':[4,5,6,7],'parent_checkpoint_sha256':PARENT_SHA}
def val_rows(full=True):
    gate=json.load(open(EX/'DATA_GATE.json'));assert sha(EX/'VAL_MANIFEST.jsonl')==gate['val_manifest_sha256']
    allrows=rows(EX/'VAL_MANIFEST.jsonl')
    if full:return allrows
    selected=set(json.load(open(EX/'VAL100_SELECTION.json'))['sample_ids'])
    return [r for r in allrows if r['sample_id'] in selected]
def metric_module():
    p=ROOT.parent/'v5_concertormer_10k_20261021/training/image_metrics.py'
    spec=importlib.util.spec_from_file_location('v6_multig_unchanged_metrics',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def measure(value,clean,module):
    a=value.cpu().numpy().astype(np.float64);b=clean.cpu().numpy().astype(np.float64)
    ss,lp=module.image_metrics(value,clean)
    return dict(psnr=-10*math.log10(float(np.mean((a-b)**2))),ssim=ss[0],lpips=lp[0],mae=float(np.abs(a-b).mean()))
def evaluate_rank(model,ema,device,out,rank,world,full=True,g_values=None,check_parent=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);g_values=G_VALUES if g_values is None else g_values
    pool=val_rows(full);vis=set(json.load(open(EX/'VISUAL_SELECTION.json'))['sample_ids'])
    metric=metric_module();initial_native=model.native.model;initial_sampling=model.sampling.model
    model.native.model=ema.model;model.sampling.model=ema.model
    calls=[];hook=ema.model.register_forward_hook(lambda m,a,v:calls.append(1));records=[];started=time.time()
    reference={(str(r['g']),r['sample_id']):r for r in rows(OLD/'PER_IMAGE_ALL_G.jsonl')} if check_parent else {}
    try:
        with torch.no_grad():
            for index,row in enumerate(pool[rank::world]):
                assert sha(row['clean_path'])==row['clean_sha256'];c=torch.from_numpy(np.load(row['clean_path'],allow_pickle=False)[None]).to(device)
                seed=derive_seed(BASE_SEED,row['source_id'],row['sample_id'],'outer',0)
                # One existing G4 trajectory per rank verifies the unchanged S3 path.
                if check_parent and index==0:
                    entry=row['states']['4'];assert sha(entry['path'])==entry['sha256']
                    x=torch.from_numpy(np.load(entry['path'],allow_pickle=False)[None]).to(device)
                    calls.clear();y=model.sample_clean(x,seed=seed,differentiable=False);assert len(calls)==3
                    old=reference[('4',row['sample_id'])];actual=measure(y,c,metric)
                    error=max(abs(actual[k]-old['restored_metrics'][k]) for k in actual);assert error<=1e-6,(rank,error)
                    write_json(out/f'PARENT_S3_REPLAY_rank{rank}.json',{'status':'PASS','sample_id':row['sample_id'],'nfe':3,'metric_maxabs':error})
                for g in g_values:
                    key=f'{g:g}';entry=row['states'][key];assert sha(entry['path'])==entry['sha256']
                    x=torch.from_numpy(np.load(entry['path'],allow_pickle=False)[None]).to(device)
                    torch.npu.synchronize();begin=time.perf_counter();calls.clear()
                    y=model.sample_clean(x,seed=seed,differentiable=False);torch.npu.synchronize()
                    seconds=time.perf_counter()-begin;assert len(calls)==3 and bool(torch.isfinite(y).all())
                    pred=measure(y,c,metric)
                    old=reference.get((key,row['sample_id']))
                    inp=old['input_metrics'] if old else measure(x,c,metric)
                    result={'sample_id':row['sample_id'],'source_id':row['source_id'],'category_id':row['category_id'],
                        'g':g,'seed':seed,'nfe':3,'inference_seconds':seconds,**pred,**{'input_'+k:v for k,v in inp.items()},
                        'clean_path':row['clean_path'],'input_path':entry['path'],'input_sha256':entry['sha256']}
                    assert all(math.isfinite(result[k]) for k in METRICS)
                    if not full:
                        calls.clear();loss,tau=model.residual_loss(x,c,seed=derive_seed(BASE_SEED,'fixed_val_residual',key,row['sample_id']))
                        assert len(calls)==1 and bool(torch.isfinite(loss))
                        result.update(residual_loss=float(loss),residual_loss_nfe=1,residual_tau=int(tau[0]))
                    if full and row['sample_id'] in vis:
                        p=out/'visuals'/('G'+key)/(row['sample_id']+'.npy');p.parent.mkdir(parents=True,exist_ok=True)
                        np.save(p,y[0].cpu().numpy(),allow_pickle=False)
                        result.update(prediction_npy=str(p),prediction_sha256=sha(p))
                    records.append(result)
                    del x,y
                if index%50==0:print('VAL',rank,index,len(records),flush=True)
    finally:
        hook.remove();model.native.model=initial_native;model.sampling.model=initial_sampling
    write_rows(out/f'rank{rank}.jsonl',records)
    write_json(out/f'rank{rank}_status.json',{'status':'PASS','count':len(records),'seconds':time.time()-started,
        'g_values':g_values,'full':full,'metrics_sha256':sha(ROOT.parent/'v5_concertormer_10k_20261021/training/image_metrics.py')})
def summarize(out,world,full,update,g_values=None):
    g_values=G_VALUES if g_values is None else g_values;data=[]
    for r in range(world):
        assert json.load(open(out/f'rank{r}_status.json'))['status']=='PASS';data+=rows(out/f'rank{r}.jsonl')
    expected={(r['sample_id'],g) for r in val_rows(full) for g in g_values}
    assert len(data)==len(expected) and {(r['sample_id'],r['g']) for r in data}==expected
    data.sort(key=lambda r:(r['g'],r['sample_id']))
    by={f'{g:g}':{k:float(np.mean([r[k] for r in data if r['g']==g])) for k in METRICS} for g in g_values}
    summary={'status':'PASS','update':update,'full_val1000_per_g':full,'count':len(data),'g_values':g_values,'by_g':by,
        'means':{k:float(np.mean([v[k] for v in by.values()])) for k in METRICS},
        'seconds':max(json.load(open(out/f'rank{r}_status.json'))['seconds'] for r in range(world)),
        'test_used':False,'source_manifest_sha256':sha(EX/'SOURCE_MANIFEST.json')}
    if not full:
        summary['residual_loss_by_g']={f'{g:g}':float(np.mean([r['residual_loss'] for r in data if r['g']==g])) for g in g_values}
    write_rows(out/'PER_IMAGE.jsonl',data);write_json(out/'SUMMARY.json',summary);return summary
def select_best(candidates):
    eligible=[r for r in candidates if r['validation']['by_g']['4']['psnr']>=21.16514236838766-.20]
    assert eligible,'Parent should remain eligible'
    maximum=max(r['validation']['means']['psnr'] for r in eligible)
    ties=[r for r in eligible if r['validation']['means']['psnr']>=maximum-.02]
    selected=min(ties,key=lambda r:(r['validation']['means']['lpips'],-r['validation']['means']['ssim'],r['update']))
    return {**selected,'selection':{'G4_floor':21.16514236838766-.20,'eligible_count':len(eligible),
        'candidate_count':len(candidates),'Smax':maximum,'PSNR_tie':.02,'criterion':'G4 retention then five-g mean PSNR, LPIPS, SSIM, earlier update'}}

"""Fixed Val1000 four-step EMA selection; fixed Val100 TF grid is auxiliary."""
from contextlib import contextmanager
import importlib.util,json,math,time,os
import numpy as np
import torch
from method_impl.evaluation import BASE_SEED,validation_rows,observation_subset
from method_impl.d0 import derive_seed
from scripts.generate_g4 import raw_noise
from scripts.prepare_sources import ROOT,OLD,sha,rows,write_rows
from scripts.runtime_utils import write_json


@contextmanager
def using_core(model,core):
    original=model.core;native=model.restorer.native.model;sampling=model.restorer.sampling.model
    model.core=core;model.restorer.native.model=core['network'];model.restorer.sampling.model=core['network']
    try:yield
    finally:model.core=original;model.restorer.native.model=native;model.restorer.sampling.model=sampling


def evaluate_rank(model,ema,device,out,*,rank,world,full=True,stage):
    out.mkdir(parents=True,exist_ok=True)
    metric_path=ROOT.parent/'v5_concertormer_10k_20261021/training/image_metrics.py'
    spec=importlib.util.spec_from_file_location('v6_ab_metrics',metric_path);metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    pool=validation_rows();visual_ids={p['sample_id'] for p in observation_subset(pool)}
    pool=pool if full else observation_subset(pool);results=[];tf_results=[];started=time.time()
    def measure(y,c):
        y64=y.detach().cpu().numpy().astype(np.float64);c64=c.detach().cpu().numpy().astype(np.float64)
        ssim,lpips=metrics.image_metrics(y,c)
        return {'psnr':-10*math.log10(float(np.mean((y64-c64)**2))),'mae':float(np.abs(y64-c64).mean()),
                'ssim':ssim[0],'lpips':lpips[0],'output_min':float(y.min()),'output_max':float(y.max())}
    with using_core(model,ema.model),torch.no_grad():
        for p in pool[rank::world]:
            assert sha(OLD/p['clean_ref'])==p['clean_sha256'] and sha(OLD/p['observation_ref'])==p['observation_sha256']
            c=torch.from_numpy(np.load(OLD/p['clean_ref'],allow_pickle=False)[None]).to(device)
            original=torch.from_numpy(np.load(OLD/p['observation_ref'],allow_pickle=False)[None]).to(device)
            x=original;predictions=[];step_metrics=[];seeds=[]
            for i in range(4):
                seed=derive_seed(BASE_SEED,p['source_id'],p['sample_id'],'outer',i);seeds.append(seed)
                t=torch.tensor([i/4],device=device,dtype=torch.float32)
                clean,raw=model.predict(x,t,seed=seed,differentiable=False)
                assert bool(torch.isfinite(clean).all()) and bool(torch.isfinite(raw).all())
                reconstruction=model.render(clean,raw,4.)
                stat=measure(clean,c);stat['original_G4_physical_mae']=float((reconstruction-original).abs().mean())
                stat['raw_min']=float(raw.min());stat['raw_max']=float(raw.max())
                assert all(math.isfinite(v) for v in stat.values())
                step_metrics.append(stat);predictions.append(clean)
                if i<3:
                    x=model.render(clean,raw,3.-i);assert bool(torch.isfinite(x).all())
            input_mse=float(((original.double()-c.double())**2).mean())
            row={'sample_id':p['sample_id'],'source_id':p['source_id'],'category_id':p['category_id'],
                 'seeds':seeds,'input_psnr':-10*math.log10(input_mse),'steps':step_metrics,'nfe':12,
                 'step4_minus_step1_psnr':step_metrics[-1]['psnr']-step_metrics[0]['psnr']}
            if full and p['sample_id'] in visual_ids:
                from PIL import Image
                visual=out/'fixed_visual100';visual.mkdir(exist_ok=True);stem=f'{seeds[0]:016x}'
                path=visual/(stem+'.npy');tmp=path.with_suffix('.npy.tmp')
                with tmp.open('wb') as f:np.save(f,torch.stack(predictions).cpu().numpy()[:,0],allow_pickle=False)
                os.replace(tmp,path)
                tiles=[np.rint(np.clip(v[0].cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8) for v in [original,*predictions,c]]
                Image.fromarray(np.concatenate(tiles,axis=1)).save(visual/(stem+'.png'))
                row['visual']={'raw_npy':str(path.relative_to(out)),'sha256':sha(path),'order':['Y4','Step1','Step2','Step3','Step4','C']}
                # True-g diagnostic uses only the fixed 100-trajectory subset, never selection.
                raw_gt,hashes=raw_noise(p);z,tilt=[v.to(device) for v in raw_gt]
                replay=model.render_gt(c,z,tilt,4.)
                assert float((replay-original).abs().max())<=2e-5
                row['TF_raw_sha256']=hashes;del replay
                for g in [3.,2.,1.]:
                    true_x=model.render_gt(c,z,tilt,g);t=torch.tensor([1-g/4],device=device)
                    seed=derive_seed(BASE_SEED,p['source_id'],p['sample_id'],'TF_grid',g)
                    restored,raw_prediction=model.predict(true_x,t,seed=seed,differentiable=False)
                    stat=measure(restored,c);stat['physical_mae_current_g']=float((model.render(restored,raw_prediction,g)-true_x).abs().mean())
                    tf_results.append({'sample_id':p['sample_id'],'source_id':p['source_id'],'category_id':p['category_id'],'g':g,'seed':seed,'metrics':stat})
                del raw_gt,z,tilt
            results.append(row)
            if len(results)%10==0:write_json(out/f'rank{rank}_progress.json',{'status':'RUNNING','completed':len(results),'seconds':time.time()-started})
    write_rows(out/f'rank{rank}.jsonl',results);write_rows(out/f'rank{rank}_TF_grid.jsonl',tf_results)
    write_json(out/f'rank{rank}_status.json',{'status':'PASS','count':len(results),'seconds':time.time()-started,
               'TF_grid_rows':len(tf_results),'metrics_source_sha256':sha(metric_path)})


def summarize(out,world,full,stage):
    data=[];tf=[]
    for r in range(world):
        assert json.loads((out/f'rank{r}_status.json').read_text())['status']=='PASS'
        data+=rows(out/f'rank{r}.jsonl');tf+=rows(out/f'rank{r}_TF_grid.jsonl')
    expected=validation_rows();expected=expected if full else observation_subset(expected)
    assert len(data)==len(expected) and {x['sample_id'] for x in data}=={x['sample_id'] for x in expected}
    assert len({x['source_id'] for x in data})==(250 if full else len({x['source_id'] for x in expected}))
    data.sort(key=lambda x:x['sample_id']);keys=['psnr','mae','ssim','lpips','original_G4_physical_mae']
    step_means=[{k:float(np.mean([x['steps'][i][k] for x in data])) for k in keys} for i in range(4)]
    summary={'status':'PASS','stage':stage,'count':len(data),'unique_clean':len({x['source_id'] for x in data}),
        'full_val1000':full,'means':step_means[3],'step_means':step_means,'selection_scope':'fixed Val1000 Step4 final PSNR',
        'input_psnr':float(np.mean([x['input_psnr'] for x in data])),
        'step4_minus_step1_psnr':step_means[3]['psnr']-step_means[0]['psnr'],
        'physical_path':[4,3,2,1,0],'sampling_nfe':12,'test_used':False,'validation_base_seed':BASE_SEED,
        'input_identity_sha256':sha(OLD/'sample_manifest.jsonl'),
        'per_category':{str(c):[{k:float(np.mean([x['steps'][i][k] for x in data if x['category_id']==c])) for k in keys} for i in range(4)] for c in sorted({x['category_id'] for x in data})},
        'TF_grid':{'selection_eligible':False,'count_per_g':100 if full else 0,
            'means':{str(g):{k:float(np.mean([x['metrics'][k] for x in tf if x['g']==g])) for k in ['psnr','ssim','lpips','mae','physical_mae_current_g']} for g in [3.,2.,1.]} if full else {}}}
    if full:assert len(tf)==300 and all(sum(x['g']==g for x in tf)==100 for g in [3.,2.,1.])
    write_rows(out/'PER_IMAGE.jsonl',data);write_rows(out/'TF_GRID_VAL100.jsonl',tf);write_json(out/'SUMMARY.json',summary)
    return summary

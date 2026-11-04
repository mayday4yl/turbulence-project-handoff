"""Frozen native SDE evaluation for the two new training routes; RNG-isolated, saved outputs."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from core import ROOT,derive,metrics_module,process,parameter_digest,rng_capture,rng_restore,sha,write
from data import validation_groups

HERE=Path(__file__).resolve().parent

def evaluate(net,device,out,rank,world,full,update,weights='EMA',branch='unknown'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    previous_rng=rng_capture(device);was_training=net.training;net.eval()
    groups=validation_groups(full)
    if not full:groups=[q for q in groups if q[0] in [1,4]]
    state_digest=parameter_digest(net,full=True)
    if (out/'SUMMARY.json').exists():
        done=json.loads((out/'SUMMARY.json').read_text());assert done['parameter_digest']==state_digest and done['update']==update and done['weights']==weights and done['branch']==branch
        assert done['status']=='PASS' and done['full']==full and done['count']==(5000 if full else 200)
        for ix in range(rank,len(groups),world):
            g,source,rows=groups[ix];key=f'{g:g}'
            group=json.loads((out/'groups'/f'G{key}'/(hashlib.sha256(source.encode()).hexdigest()+'.json')).read_text())
            assert group['parameter_digest']==state_digest and group['update']==update and group['weights']==weights and group['branch']==branch
            by_id={r['sample_id']:r for r in group['records']};assert set(by_id)=={r['sample_id'] for r in rows}
            for row in rows:
                r=by_id[row['sample_id']]
                assert r['input_sha256']==row['states'][key]['sha256'] and r['clean_sha256']==row['clean_sha256']
                assert sha(r['prediction'])==r['prediction_sha256'] and sha(row['states'][key]['path'])==r['input_sha256'] and sha(row['clean_path'])==r['clean_sha256']
        dist.barrier()
        net.train(was_training);rng_restore(previous_rng,device);return done
    metric=metrics_module();sde=process(device);sde.set_model(net);records=[];started=time.time();reused_groups=0
    try:
      with torch.no_grad():
       for ix in range(rank,len(groups),world):
        g,source,rows=groups[ix];key=f'{g:g}'
        group_path=out/'groups'/f'G{key}'/(hashlib.sha256(source.encode()).hexdigest()+'.json')
        if group_path.exists():
            saved=json.loads(group_path.read_text())
            assert saved['parameter_digest']==state_digest and saved['update']==update and saved['weights']==weights and saved['branch']==branch
            rr=saved['records'];assert len(rr)==4 and {r['sample_id'] for r in rr}=={r['sample_id'] for r in rows}
            expected_rows={r['sample_id']:r for r in rows}
            for r in rr:
                expected=expected_rows[r['sample_id']]
                assert r['input_sha256']==expected['states'][key]['sha256'] and r['clean_sha256']==expected['clean_sha256']
                assert sha(r['prediction'])==r['prediction_sha256'] and sha(r['input_path'])==r['input_sha256'] and sha(r['clean_path'])==r['clean_sha256']
            records.extend(rr);reused_groups+=1;continue
        for row in rows:
            assert sha(row['clean_path'])==row['clean_sha256']
            assert sha(row['states'][key]['path'])==row['states'][key]['sha256']
        y=torch.from_numpy(np.stack([np.load(r['states'][key]['path'],allow_pickle=False) for r in rows])).to(device)
        clean=torch.from_numpy(np.stack([np.load(r['clean_path'],allow_pickle=False) for r in rows])).to(device)
        seed=derive(843001,source,'four_realizations_native_SDE')
        torch.manual_seed(seed);torch.npu.manual_seed(seed)
        calls=[0];hook=net.register_forward_hook(lambda m,a,v:calls.__setitem__(0,calls[0]+1))
        sde.set_mu(y);torch.npu.synchronize();begin=time.perf_counter()
        try:pred=sde.reverse_sde(y,save_states=False)
        finally:hook.remove()
        torch.npu.synchronize();seconds=time.perf_counter()-begin
        assert calls[0]==100 and bool(torch.isfinite(pred).all())
        p=pred.detach().float().cpu();c=clean.detach().float().cpu()
        ss,lp=metric.image_metrics(p,c)
        pa=p.numpy().astype(np.float64);ca=c.numpy().astype(np.float64)
        mse=np.mean((pa-ca)**2,axis=(1,2,3));mae=np.mean(np.abs(pa-ca),axis=(1,2,3))
        cache=HERE/'evidence'/'input_metric_cache'/branch/f'G{key}'/(hashlib.sha256(source.encode()).hexdigest()+'.json')
        input_sha=[r['states'][key]['sha256'] for r in rows]
        if cache.exists():
            cached=json.loads(cache.read_text());assert cached['input_sha256']==input_sha;im=cached['metrics']
        else:
            yp=y.detach().float().cpu();iss,ilp=metric.image_metrics(yp,c);ya=yp.numpy().astype(np.float64)
            imse=np.mean((ya-ca)**2,axis=(1,2,3));imae=np.mean(np.abs(ya-ca),axis=(1,2,3))
            im=[{'psnr':-10*math.log10(max(float(e),1e-30)),'ssim':float(s),'lpips':float(l),'mae':float(a)} for e,s,l,a in zip(imse,iss,ilp,imae)]
            write(cache,{'input_sha256':input_sha,'metrics':im,'protocol':'rawfloat RGB, same original metric module, no crop/clip/resize'})
        for j,row in enumerate(rows):
            dest=out/'predictions'/f'G{key}'/(row['sample_id']+'.npy');dest.parent.mkdir(parents=True,exist_ok=True)
            np.save(dest,p[j].numpy(),allow_pickle=False)
            m={'psnr':-10*math.log10(max(float(mse[j]),1e-30)),'ssim':float(ss[j]),'lpips':float(lp[j]),'mae':float(mae[j])}
            assert all(math.isfinite(z) for q in [m,im[j]] for z in q.values())
            records.append({'branch':branch,'weights':weights,'update':update,'sample_id':row['sample_id'],'source_id':source,'g':g,
                'group_seed':seed,'group_batch':4,'group_position':j,'nfe':100,'metrics':m,'input_metrics':im[j],
                'input_path':row['states'][key]['path'],'input_sha256':input_sha[j],'clean_path':row['clean_path'],
                'clean_sha256':row['clean_sha256'],'prediction':str(dest),'prediction_sha256':sha(dest),'group_inference_seconds':seconds})
        write(group_path,{'parameter_digest':state_digest,'update':update,'weights':weights,'branch':branch,'records':records[-4:]})
        if len(records)%40==0:write(out/f'PROGRESS_rank{rank}.json',{'records':len(records),'seconds':time.time()-started})
        del y,clean,pred,p,c
    finally:
        net.train(was_training);rng_restore(previous_rng,device)
    path=out/f'rank{rank}.jsonl'
    with path.open('w') as f:
        for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
    write(out/f'DONE_rank{rank}.json',{'status':'PASS','count':len(records),'seconds':time.time()-started,'sha256':sha(path),'training_RNG_restored':True,'reused_completed_groups':reused_groups})
    dist.barrier();summary=None
    if rank==0:
        data=[]
        for k in range(world):
            done=json.loads((out/f'DONE_rank{k}.json').read_text());assert done['status']=='PASS'
            assert sha(out/f'rank{k}.jsonl')==done['sha256'];data.extend(json.loads(s) for s in (out/f'rank{k}.jsonl').read_text().splitlines())
        expected={(r['sample_id'],g) for g,_,rr in groups for r in rr}
        assert len(data)==len(expected) and {(r['sample_id'],r['g']) for r in data}==expected
        gs=sorted(set(r['g'] for r in data));by={}
        for g in gs:
            subset=[r for r in data if r['g']==g]
            by[f'{g:g}']={q:float(np.mean([r['metrics'][q] for r in subset])) for q in ['psnr','ssim','lpips','mae']}
            by[f'{g:g}']['input_psnr']=float(np.mean([r['input_metrics']['psnr'] for r in subset]))
        summary={'status':'PASS','branch':branch,'update':update,'weights':weights,'full':full,'count':len(data),
            'sources_per_g':250 if full else len(groups)//len(gs),'per_g':1000 if full else len(data)//len(gs),
            'by_g':by,'nfe_per_image':100,'sampler':'native_reverse_sde','precision':'FP32/HF32off','image_clip_crop_resize':False,
            'seconds':max(json.loads((out/f'DONE_rank{k}.json').read_text())['seconds'] for k in range(world)),
            'selection_role':'full_validation' if full else 'monitor_only','training_RNG_restored':True,'test_used':False,'parameter_digest':state_digest}
        write(out/'SUMMARY.json',summary)
        with (HERE/'evidence'/f'VALIDATION_EVENTS_{branch}.jsonl').open('a') as f:f.write(json.dumps({'out':str(out),**summary})+'\n')
    answer=[summary];dist.broadcast_object_list(answer,src=0);return answer[0]

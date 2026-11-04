"""Four real resident workers, concurrent gradients and serialized fullval repeats.
No optimizer steps. Existing temporary optimizer/EMA states remain resident.
"""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deps'));sys.path.insert(0,str(ROOT/'scripts'))
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import numpy as np
import torch
from v4.matched import Store,build,eta_batch,backward32,GROUP_GPU
from v4.runtime import configure_cuda,verify_imports,save_json,sha256
from v4.training import make_optimizer,make_ema,finite_gradients
from v4.refinement import module_hash
from v4.mutex import EvaluationMutex
from v4.evaluate_matched import Evaluator,output_digest
from train_worker import capture_rng,restore_rng
from metrics import Metrics,SPEC


def main(args):
    if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Global PAUSED')
    configure_cuda(20260929)
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());verify_imports(ROOT,cfg)
    for g in GROUP_GPU:
        assert json.loads((ROOT/f'reports/acceptance_{g}.json').read_text())['status']=='PASS'
    store=Store(cfg);model,sim,obj=build(cfg,args.group);opt=make_optimizer(model);ema=make_ema(model)
    saved=torch.load(ROOT/f'reports/temporary_{args.group}_step1.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);ema.load_state_dict(saved['ema'],strict=True);opt.load_state_dict(saved['optimizer']);del saved
    report={'status':'RUNNING','group':args.group,'formal_optimizer_steps':0,'additional_temporary_optimizer_steps':0,'pid':os.getpid()}
    work=ROOT/'reports/shared_gpu_acceptance';work.mkdir(exist_ok=True)
    def persist(stage):
        report['stage']=stage;save_json(work/f'{args.group}.json',report);print(args.group,stage,flush=True)
    persist('resident_model_optimizer_EMA')
    begin=time.monotonic()
    while not all((work/f'{g}.json').exists() for g in GROUP_GPU):
        if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Peer failure')
        if time.monotonic()-begin>300:raise TimeoutError('Four-worker startup timed out')
        time.sleep(.1)
    model_hash=module_hash(model);ema_hash=module_hash(ema);sim_hash=module_hash(sim)
    # Worst-case two G4 paths plus actual balanced B32 accumulation. No Adam step.
    obj.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    rows=[store.train[3],store.train[7]];C,Y,G=store.batch(rows,'cuda')
    out=obj(C,Y,G,eta=eta_batch(rows,'cuda',1));out['loss_mean'].backward();finite_gradients(model)
    del out,C,Y,G;obj.zero_grad(set_to_none=True)
    stream=json.loads((ROOT/'protocol/stream.json').read_text());batch=[store.by_id[s] for s in stream['updates'][0]]
    started=time.monotonic();backward32(obj,store,batch,2,1);finite_gradients(model);torch.cuda.synchronize()
    report['concurrent_B32_backward_seconds']=time.monotonic()-started
    report['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
    obj.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();persist('shared_backward_complete')
    members=['R08','D0'] if GROUP_GPU[args.group]==5 else ['R0','L08']
    mutex=EvaluationMutex(work/'mutex',args.group,GROUP_GPU[args.group],members,torch.cuda.synchronize)
    evaluation=Evaluator(store,'TEMPORARY_SHARED_ACCEPTANCE','current_registered_environment')
    repeats=[]
    for repeat in range(3):
        assert mutex.boundary(0,f'repeat{repeat}')
        rng=capture_rng();obj.eval()
        try:
            if repeat==0:
                metrics=Metrics('cuda')
                fixture=store.val[:8]
                evidence=evaluation.evaluate(obj,fixture,work/args.group/'metric_fixture',update=0,
                         weights='raw',checkpoint_hash=model_hash,metrics=metrics)
                # Exact saved float bytes and corresponding full-precision metrics.
                C,Y,G=store.batch(fixture[:2],'cuda')
                with torch.no_grad():out=obj(C,Y,G,eta=eta_batch(fixture[:2],'cuda'))
                original=metrics(out['final_clean'],C)
                rows=[json.loads(s) for s in (work/args.group/'metric_fixture/per_start.jsonl').read_text().splitlines()]
                for i in range(2):
                    for key in ('psnr','ssim','lpips','mse'):
                        assert abs(rows[i][key]-original[i][key])<1e-8
                    assert rows[i]['output_hash_final']==output_digest(out['final_clean'][i])[0]
                report['metric_fixture']=evidence['overall'];del out,C,Y,G
                full_evaluation=[]
                for label,network in [('raw',model),('EMA',ema)]:
                    obj.network=network
                    for population in ['train_monitor400','val800']:
                        value=evaluation.evaluate(obj,store.lists[population],work/args.group/'full_evaluation'/label/population,
                            update=0,weights=label,checkpoint_hash=model_hash,metrics=metrics)
                        full_evaluation.append({'weights':label,'population':population,'overall':value['overall'],
                                                'seconds':value['elapsed_seconds']})
                obj.network=model;report['full_evaluation']=full_evaluation;del metrics
            values=[]
            with torch.no_grad():
                for offset in range(0,800,2):
                    chunk=store.val[offset:offset+2];C,Y,G=store.batch(chunk,'cuda')
                    out=obj(C,Y,G,eta=eta_batch(chunk,'cuda'))
                    errors=(out['final_clean']-C).square().flatten(1).mean(1).cpu().numpy()
                    if not np.isfinite(errors).all() or (errors<=0).any():raise FloatingPointError('Invalid repeat MSE')
                    values.extend((-10*np.log10(errors.astype(np.float64))).tolist())
                    del out,C,Y,G
            repeats.append(values)
            save_json(work/args.group/f'repeat{repeat}.json',{'start_ids':[r['start_id'] for r in store.val],'psnr':values,'mean_psnr':float(np.mean(values))})
        finally:
            obj.train();gc.collect();torch.cuda.empty_cache();restore_rng(rng)
        # No network, simulator, EMA or global RNG state is altered by evaluation.
        assert module_hash(model)==model_hash and module_hash(sim)==sim_hash and module_hash(ema)==ema_hash
        assert torch.equal(capture_rng()['torch'],rng['torch']) and torch.equal(capture_rng()['cuda'],rng['cuda'])
        mutex.complete(f'repeat{repeat}');persist(f'fullval_repeat{repeat}_complete')
    means=np.mean(repeats,axis=1);spread=float(np.ptp(means))
    report['repeat_fullval800']={'means':means.tolist(),'range_db':spread,'limit_db':.005,
                               'max_per_start_range_db':float(np.ptp(repeats,axis=0).max())}
    assert spread<=.005, f'Fullval repeated PSNR range exceeds frozen tolerance: {spread}'
    mutex.terminal(0)
    report['status']='PASS';report['model_sim_EMA_unchanged']=True;persist('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=list(GROUP_GPU),required=True);args=p.parse_args()
    try:main(args)
    except BaseException as exc:
        save_json(ROOT/'reports/RUN_PAUSED.json',{'status':'PAUSED','gate':'four_process_GPU_mutex_eval','group':args.group,'exception':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise

"""Bounded paired throughput and gradient tests; ZERO optimizer updates."""
import argparse,contextlib,fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
import numpy as np
import torch
from v4.matched import Store,build,backward32 as reference_backward,GROUP_GPU
from v4.performance import backward32
from v4.runtime import configure_cuda,verify_imports,save_json
from v4.training import make_optimizer,make_ema,finite_gradients
from v4.refinement import module_hash


def wait_for(paths,work,timeout=1800):
    start=time.monotonic()
    while not all(p.exists() for p in paths):
        if (work/'FAILURE.json').exists():raise RuntimeError('Peer benchmark failure')
        if time.monotonic()-start>timeout:raise TimeoutError('Benchmark barrier')
        time.sleep(.1)


def flat_grads(model):
    return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}


def grad_diff(model,ref):
    num=den=0.
    for n,p in model.named_parameters():
        if (p.grad is None)!=(n not in ref):raise AssertionError('Gradient None mismatch '+n)
        if p.grad is not None:
            a=p.grad.detach().cpu().double();b=ref[n].double()
            num+=float((a-b).square().sum());den+=float(b.square().sum())
    value=(num/max(den,1e-300))**.5
    if value>1e-4:raise AssertionError(f'Gradient mismatch {value}')
    return value


def main(a):
    configure_cuda(20260929)
    work=ROOT/'reports/performance'/a.attempt;work.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());verify_imports(ROOT,cfg)
    store=Store(cfg);store.cache_limit=8*1024**3
    model,sim,obj=build(cfg,a.group);optimizer=make_optimizer(model);ema=make_ema(model)
    # Retain real, populated AdamW state without performing an optimizer update.
    parent=ROOT.parent/'v4_train500_matched_v2'
    saved=torch.load(parent/f'reports/temporary_{a.group}_step1.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(saved['optimizer']);del saved
    original=module_hash(model);simhash=module_hash(sim);emahash=module_hash(ema)
    stream=json.loads((ROOT/'protocol/stream.json').read_text())
    batches=[[store.by_id[s] for s in ids] for ids in stream['updates'][:4]]
    # Stage fixed actual images in pageable host RAM; no new data or changed stream.
    for batch in batches:
        for row in batch:store.array(row['clean']);store.array(row['state'])
    report={'group':a.group,'pid':os.getpid(),'gpu':GROUP_GPU[a.group],
            'status':'RUNNING','optimizer_updates':0,'cases':[],
            'effective_batch':32,'evaluation_unchanged':True,'start_time':time.time()}
    save_json(work/f'{a.group}.json',report)
    groups=['D0','R08'] if GROUP_GPU[a.group]==5 else ['R0','L08']
    wait_for([work/f'{g}.json' for g in ([a.group] if a.solo else ['D0','R0','R08','L08'])],work)
    # Single active process per GPU for baseline and candidate gradient qualification.
    with (work/f'gpu{GROUP_GPU[a.group]}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        obj.zero_grad(set_to_none=True)
        ref_values=reference_backward(obj,store,batches[0],2,1)
        finite_gradients(model);reference=flat_grads(model)
        obj.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        for micro in [2]:
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            values=backward32(obj,store,batches[0],micro,1);torch.cuda.synchronize()
            finite_gradients(model);delta=grad_diff(model,reference)
            np.testing.assert_allclose(values,ref_values,rtol=2e-4,atol=2e-5)
            report['cases'].append({'kind':'gradient_parity','microbatch':micro,'relative_gradient_L2':delta,
                                   'seconds':time.monotonic()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
            obj.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            save_json(work/f'{a.group}.json',report)
        del reference
        fcntl.flock(lock,fcntl.LOCK_UN)
    save_json(work/f'{a.group}.qualified',{'time':time.time()})
    wait_for([work/f'{g}.qualified' for g in ([a.group] if a.solo else ['D0','R0','R08','L08'])],work)
    # Compare current shared execution and serialized higher microbatch.
    cases=[('solo_m2','solo',2)] if a.solo else [('concurrent_m2','concurrent',2),('serial_m2','serial',2)]
    if a.solo:groups=[a.group]
    for case,mode,micro in cases:
        save_json(work/f'{a.group}.{case}.ready',{'time':time.time()})
        wait_for([work/f'{g}.{case}.ready' for g in groups],work)
        timings=[];waiting=0.;torch.cuda.reset_peak_memory_stats();wall=time.monotonic()
        for repeat in range(4):
            with (work/f'gpu{GROUP_GPU[a.group]}.lock').open('a+') as lock:
                before=time.monotonic()
                if mode=='serial':fcntl.flock(lock,fcntl.LOCK_EX)
                waiting+=time.monotonic()-before
                start=time.monotonic();obj.zero_grad(set_to_none=True)
                backward32(obj,store,batches[repeat],micro,1)
                finite_gradients(model);torch.cuda.synchronize()
                timings.append(time.monotonic()-start)
                obj.zero_grad(set_to_none=True)
                if mode=='serial':fcntl.flock(lock,fcntl.LOCK_UN)
        torch.cuda.synchronize()
        entry={'kind':'throughput','name':case,'microbatch':micro,'mode':mode,'seconds':timings,
               'measured_wall_seconds':time.monotonic()-wall,'training_lock_wait_seconds':waiting,
               'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
        report['cases'].append(entry);save_json(work/f'{a.group}.json',report)
        save_json(work/f'{a.group}.{case}.done',{'time':time.time()})
        wait_for([work/f'{g}.{case}.done' for g in groups],work)
        gc.collect();torch.cuda.empty_cache()
    assert module_hash(model)==original and module_hash(sim)==simhash and module_hash(ema)==emahash
    report.update(status='PASS',elapsed_seconds=time.time()-report['start_time'],weights_unchanged=True)
    save_json(work/f'{a.group}.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--attempt',required=True);p.add_argument('--solo',action='store_true');a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        save_json(ROOT/'reports/performance'/a.attempt/'FAILURE.json',{'group':a.group,'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise

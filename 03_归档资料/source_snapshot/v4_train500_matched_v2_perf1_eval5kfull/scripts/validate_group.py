"""Real FP32 CUDA acceptance, bounded two temporary optimizer updates per arm."""
import argparse,copy,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from v4.core import mse,noise_schedule
from v4.matched import Store,build,eta_batch,backward32,lr_at
from v4.runtime import configure_cuda,verify_imports,save_json,sha256
from v4.refinement import module_hash
from v4.training import make_optimizer,make_ema,update_ema,finite_gradients,atomic_torch_save

ATOL=2e-5;RTOL=2e-4;GRAD_LIMIT=1e-4

def check_close(a,b):
    torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)
    return float((a-b).abs().max())

def tree_equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):tree_equal(x,y)
    else:assert a==b

def gradient_difference(model, reference):
    err=norm=0.
    for name,p in model.named_parameters():
        a=p.grad.detach().cpu().double();b=reference[name].double()
        err+=float((a-b).square().sum());norm+=float(b.square().sum())
    relative=(err/max(norm,1e-300))**.5
    assert relative<=GRAD_LIMIT, f'Gradient relative L2 {relative} > {GRAD_LIMIT}'
    return relative

def main(args):
    configure_cuda(20260929)
    config=json.loads((ROOT/'configs/runtime.json').read_text())
    imports=verify_imports(ROOT,config);store=Store(config)
    model,sim,obj=build(config,args.group)
    report={'status':'RUNNING','group':args.group,'microbatch':args.microbatch,
            'temporary_optimizer_steps':0,'formal_optimizer_steps':0,'checks':{},'imports':imports,
            'start_time':time.time(),'pid':os.getpid(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}
    def persist(label):
        report['last_completed_check']=label
        save_json(ROOT/f'reports/acceptance_{args.group}.json',report)
        print(args.group,label,flush=True)
    persist('loaded_real_model_and_simulator')
    sim_hash=module_hash(sim);initial=copy.deepcopy(model.state_dict())
    rows=[store.train[i*16+g-1] for i,g in enumerate([1,2,3,4])]
    C,Y,G=store.batch(rows,'cuda');eta=eta_batch(rows,'cuda',1)
    singles=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            out=obj(C[i:i+1],Y[i:i+1],G[i:i+1],eta=eta[i:i+1],retain_intermediates=True)
            expected=1 if args.group=='D0' else r['G']
            assert out['prediction_count']==expected and out['transition_count']==expected-1
            assert out['observation_render_count']==expected and out['training_refinement_count']==0
            if args.group!='D0':
                for k,tr in enumerate(out['trace']):
                    assert float(tr['g'])==r['G']-k and torch.equal(tr['Y'],Y[i:i+1])
                    assert torch.equal(tr['eta'],eta[i:i+1])
                    assert torch.equal(tr['t'],noise_schedule(G[i:i+1],k,obj.T0))
                    if args.group=='L08' and k+1<expected:
                        a=(r['G']-k-2)/(r['G']-1)
                        check_close(tr['next_state'],(1-a)*tr['clean']+a*Y[i:i+1])
                    if k==expected-1:
                        assert float(tr['g'])==1 and float(tr['t'])==0
                        assert torch.equal(tr['Z'],tr['state']) and tr['next_state'] is None
                lc=torch.stack([tr['lc'] for tr in out['trace']]).mean(0)
                lr=torch.stack([tr['lr'] for tr in out['trace']]).mean(0)
                check_close(out['path_losses'],lc+.2*lr)
            singles.append({k:out[k].clone() for k in ['final_clean','first_clean','path_losses','path_clean','path_reconstruction']})
            del out
        mixed=obj(C,Y,G,eta=eta,retain_intermediates=True)
        errors={k:check_close(mixed[k],torch.cat([r[k] for r in singles])) for k in singles[0]}
        if args.group!='D0':
            for k,tr in enumerate(mixed['trace']):
                active=torch.nonzero(G>k).flatten()
                assert torch.equal(tr['active'],active)
                assert torch.equal(tr['Y'],Y[active]) and torch.equal(tr['G'],G[active])
                assert torch.equal(tr['eta'],eta[active])
            assert mixed['prediction_count']==10 and mixed['transition_count']==6
        # Target clean must never enter prediction or interpolation.
        changed=obj(C.flip(-1),Y,G,eta=eta)
        assert torch.equal(changed['final_clean'],mixed['final_clean'])
        report['checks']['forward']={'G':[1,2,3,4],'mixed_vs_single_max_abs':errors,'target_not_input':True,
              'path_first_loss_reduction':True,'active_identity':True}
        del mixed,changed,singles
    persist('forward_all_G_mixed_and_loss')
    if args.group!='D0':
        answers=[]
        for detach in (False,True):
            obj.zero_grad(set_to_none=True);counter=[0]
            def cut(module,inputs):
                counter[0]+=1
                if counter[0]==2:return (inputs[0].detach(),inputs[1])
            hook=model.register_forward_pre_hook(cut) if detach else None
            try:out=obj(C[3:4],Y[3:4],G[3:4],eta=eta[3:4],retain_intermediates=True)
            finally:
                if hook:hook.remove()
            mse(out['final_clean'],C[3:4]).mean().backward()
            stages=[]
            for i,tr in enumerate(out['trace'][:-1]):
                entry={}
                for key in ('clean','epsilon_zernike','epsilon_tilt'):
                    grad=tr[key].grad
                    entry[key]={'none':grad is None,'finite':None if grad is None else bool(torch.isfinite(grad).all()),
                                'norm':None if grad is None else float(grad.double().norm())}
                    expected_absent=(detach and i==0) or (args.group=='L08' and key!='clean')
                    if expected_absent:assert grad is None, (args.group,i,key,'expected no endpoint dependency')
                    else:assert grad is not None and torch.isfinite(grad).all() and grad.double().norm()>0,(args.group,i,key,entry[key])
                stages.append(entry)
            answers.append({'detach_after_first_stage':detach,'stages':stages})
            # Preserve the actual values even if a subsequent assertion fails.
            report['checks']['terminal_only_BPTT']=answers;persist('terminal_only_BPTT_'+str(detach))
            del out
    obj.zero_grad(set_to_none=True);del C,Y,G,eta;gc.collect();torch.cuda.empty_cache()
    stream=json.loads((ROOT/'protocol/stream.json').read_text())
    batch=[store.by_id[sid] for sid in stream['updates'][0]]
    # Compare same 32 observations/noise and per-path mean with singleton reference.
    backward32(obj,store,batch,1,1);finite_gradients(model)
    reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
    obj.zero_grad(set_to_none=True)
    totals=backward32(obj,store,batch,args.microbatch,1);finite_gradients(model)
    report['checks']['B32_accumulation']={'relative_gradient_L2':gradient_difference(model,reference),
          'limit':GRAD_LIMIT,'effective_batch':32,'G_counts':[8]*4,'totals':totals}
    del reference;obj.zero_grad(set_to_none=True);persist('B32_gradient_accumulation')
    optimizer=make_optimizer(model);ema=make_ema(model)
    # Exactly two updates: one save, one after verified resume. No third replay update.
    backward32(obj,store,batch,args.microbatch,1);finite_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for pg in optimizer.param_groups:pg['lr']=lr_at(1)
    oldhash=module_hash(model);optimizer.step();update_ema(ema,model,32)
    report['temporary_optimizer_steps']=1;persist('temporary_step1')
    assert module_hash(model)!=oldhash
    path=ROOT/f'reports/temporary_{args.group}_step1.pt'
    payload={'purpose':'TEMPORARY_ACCEPTANCE_NOT_FORMAL','model':model.state_dict(),'ema':ema.state_dict(),
             'optimizer':optimizer.state_dict(),'u':1,'E':32,'next_stream_index':1,
             'stats_artifact_hash':model.stats.artifact_hash,'stream_hash':stream['content_hash']}
    atomic_torch_save(payload,path);saved=torch.load(path,map_location='cpu',weights_only=False)
    tree_equal(saved['model'],model.state_dict());tree_equal(saved['optimizer'],optimizer.state_dict())
    tree_equal(saved['ema'],ema.state_dict())
    batch2=[store.by_id[sid] for sid in stream['updates'][1]]
    obj.zero_grad(set_to_none=True);backward32(obj,store,batch2,args.microbatch,1);finite_gradients(model)
    reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
    model.load_state_dict(saved['model'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
    optimizer.load_state_dict(saved['optimizer']);obj.zero_grad(set_to_none=True)
    backward32(obj,store,batch2,args.microbatch,1);finite_gradients(model)
    diff=gradient_difference(model,reference);del reference
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for pg in optimizer.param_groups:pg['lr']=lr_at(2)
    optimizer.step();update_ema(ema,model,32);report['temporary_optimizer_steps']=2
    assert module_hash(sim)==sim_hash
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report['checks']['checkpoint']={'state_exact':True,'next_gradient_relative_L2':diff,
             'saved_sha256':sha256(path),'stats_hash':model.stats.artifact_hash,'simulator_unchanged':True}
    persist('temporary_step2_and_checkpoint_roundtrip')
    model.load_state_dict(initial,strict=True)
    report.update(status='PASS',elapsed_seconds=time.time()-report['start_time'],
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,temporary_weights_discarded=True)
    persist('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['D0','R0','R08','L08'],required=True)
    p.add_argument('--microbatch',type=int,choices=[1,2],default=2);args=p.parse_args()
    existing=ROOT/f'reports/acceptance_{args.group}.json'
    if existing.exists():raise RuntimeError('Existing attempt; review evidence before another acceptance attempt')
    try:main(args)
    except BaseException as exc:
        fail={'status':'PAUSED','gate':'real_CUDA_correctness','group':args.group,'exception':repr(exc),
              'traceback':traceback.format_exc(),'automatic_training_authorized_now':False,'time':time.time()}
        save_json(ROOT/'reports/RUN_PAUSED.json',fail)
        save_json(ROOT/f'reports/acceptance_{args.group}_failure.json',fail)
        raise

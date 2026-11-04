"""L1/B32/trained-AdamW resume acceptance, using disposable network state only."""
import argparse,fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deps')]
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import numpy as np
import torch
from v4.core import mae
from v4.matched import Store,build,eta_batch,lr_at
from v4.performance import backward32
from v4.runtime import configure_cuda,save_json,sha256,verify_imports
from v4.training import make_optimizer,make_ema,update_ema,finite_gradients
from v4.refinement import module_hash
from v4.evaluate_matched import Evaluator
from train_worker import checkpoint,restore_full
PURPOSE='TEMPORARY_L1_ACCEPTANCE_NOT_FORMAL'

def relative_error(named,reference):
    numerator=denominator=0.;maximum=0.
    for name,value in named:
        a=value.detach().cpu().double();b=reference[name].double();difference=a-b
        numerator+=float(difference.square().sum());denominator+=float(b.square().sum())
        maximum=max(maximum,float(difference.abs().max()))
    return {'relative_L2':(numerator/max(denominator,1e-300))**.5,'max_abs':maximum}

def main(a):
    selection=json.loads((ROOT/'protocol/RESOURCE_SELECTION.json').read_text());gpu=selection['group_gpu'][a.group]
    assert os.environ['CUDA_VISIBLE_DEVICES']==selection['gpu_uuids'][str(gpu)]
    configure_cuda(20260929);torch.cuda.set_per_process_memory_fraction(.35)
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());imports=verify_imports(ROOT,cfg)
    directory=ROOT/'reports'/a.attempt/a.group;directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();report={'status':'RUNNING','group':a.group,'pid':os.getpid(),'physical_gpu':gpu,'purpose':PURPOSE,'formal_updates':0,'temporary_optimizer_calls':0,'checks':{},'imports':imports}
    def write(label):
        report['last_check']=label;report['elapsed_seconds']=time.monotonic()-started
        save_json(directory/'acceptance.json',report);print(a.group,label,flush=True)
    store=Store(cfg);model,sim,obj=build(cfg,a.group);opt=make_optimizer(model);ema=make_ema(model)
    original=module_hash(model);sim_hash=module_hash(sim);common=sha256(ROOT/'protocol/common_init.pt')
    stream=json.loads((ROOT/'protocol/stream.json').read_text());batches=[[store.by_id[s] for s in stream['updates'][u]] for u in (0,1)]
    frozen={'protocol_hash':sha256(ROOT/'protocol/SCIENTIFIC_SPEC.json'),'execution_hash':'DISPOSABLE_ACCEPTANCE_'+a.attempt,'stream_sha256':sha256(ROOT/'protocol/stream.json')}
    lock=(directory.parent/f'compute_gpu{gpu}.lock').open('a+')
    def barrier(name):
        save_json(directory/(name+'.json'),{'pid':os.getpid(),'time':time.time()})
        limit=time.monotonic()+1200
        while not all((directory.parent/g/(name+'.json')).exists() for g in ('D0','R0','R08','L08')):
            if list(directory.parent.glob('*_FAILED.json')):raise RuntimeError('Peer acceptance failed')
            if time.monotonic()>limit:raise TimeoutError('Four-task acceptance residency barrier')
            time.sleep(.2)
    write('LOADED_COMMON_U0');barrier('ALL_LOADED')
    fcntl.flock(lock,fcntl.LOCK_EX)
    with torch.no_grad():
        rows=[store.train[i*16+g-1] for i,g in enumerate((1,2,3,4))]
        C,Y,G=store.batch(rows,'cuda');eta=eta_batch(rows,'cuda',1)
        result=obj(C,Y,G,eta=eta,retain_intermediates=True)
        flipped=obj(C.flip(-1),Y,G,eta=eta)
        torch.testing.assert_close(result['final_clean'],flipped['final_clean'],atol=0,rtol=0)
        if a.group=='D0':
            pred=model(torch.cat((Y,Y),1),torch.stack((G,torch.zeros_like(G)),1))
            lc=mae(pred.clean,C);lr=mae(sim.render(pred.clean,pred.realization,G).output,Y)
            torch.testing.assert_close(result['path_clean'],lc)
            torch.testing.assert_close(result['path_reconstruction'],lr)
        else:
            for tr in result['trace']:
                torch.testing.assert_close(tr['lc'],mae(tr['clean'],C[tr['active']]))
            assert result['prediction_count']==10 and result['transition_count']==6
        torch.testing.assert_close(result['path_losses'],result['path_clean']+.2*result['path_reconstruction'])
    del C,Y,G,eta,result,flipped
    obj.zero_grad(set_to_none=True);values1=backward32(obj,store,batches[0],1,1);finite_gradients(model)
    grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
    obj.zero_grad(set_to_none=True);values2=backward32(obj,store,batches[0],2,1);finite_gradients(model)
    difference=relative_error(((n,p.grad) for n,p in model.named_parameters()),grads)
    assert difference['relative_L2']<=1e-4,difference
    np.testing.assert_allclose(values1,values2,atol=2e-5,rtol=2e-4)
    assert np.isclose(values2[2],values2[0]+.2*values2[1])
    report['checks']['L1_B32_accumulation']={'microbatches':[1,2],'batch':32,'gradient':difference,'values':values2,'no_GT_leak':True,'formal_initial_hash':original}
    del grads
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for pg in opt.param_groups:pg['lr']=lr_at(1)
    opt.step();update_ema(ema,model,32);report['temporary_optimizer_calls']+=1
    assert module_hash(model)!=original
    assert any(v['exp_avg'].abs().sum()>0 for v in opt.state.values()) and all(int(v['step'])==1 for v in opt.state.values())
    path=directory/'temporary_U1.pt';identity=checkpoint(path,model,ema,opt,1,frozen,a.group,purpose=PURPOSE)
    fcntl.flock(lock,fcntl.LOCK_UN);write('L1_UPDATE_AND_NONZERO_MOMENTS');barrier('ALL_NONZERO_MOMENTS_RESIDENT')
    def advance2():
        opt.zero_grad(set_to_none=True);begin=time.monotonic();values=backward32(obj,store,batches[1],2,1);finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        for pg in opt.param_groups:pg['lr']=lr_at(2)
        opt.step();update_ema(ema,model,32);torch.cuda.synchronize();report['temporary_optimizer_calls']+=1
        return values,time.monotonic()-begin
    fcntl.flock(lock,fcntl.LOCK_EX)
    values,seconds=advance2();reference={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    ema_reference={n:p.detach().cpu().clone() for n,p in ema.named_parameters()}
    moments={f'{i}:{k}':v[k].detach().cpu().clone() for i,v in opt.state_dict()['state'].items() for k in ('exp_avg','exp_avg_sq','step')}
    assert restore_full(path,identity,model,ema,opt,frozen,a.group,purpose=PURPOSE)==1
    replay,seconds_replay=advance2();diff=relative_error(model.named_parameters(),reference);ediff=relative_error(ema.named_parameters(),ema_reference)
    momentdiff=relative_error(((f'{i}:{k}',v[k]) for i,v in opt.state_dict()['state'].items() for k in ('exp_avg','exp_avg_sq','step')),moments)
    assert diff['max_abs']<=2e-5 and diff['relative_L2']<=1e-4,(diff,ediff)
    assert ediff['max_abs']<=2e-5 and momentdiff['relative_L2']<=1e-4,(ediff,momentdiff)
    np.testing.assert_allclose(values,replay,atol=2e-5,rtol=2e-4)
    report['checks']['nonzero_moments_resume']={'model_difference':diff,'EMA_difference':ediff,'optimizer_difference':momentdiff,'scheduler_next_lr':lr_at(2),'temporary_U1_sha256':identity['sha256'],'optimizer_calls_total':3,'update2_seconds':seconds,'replayed_update2_seconds':seconds_replay}
    del reference,ema_reference,moments
    evaluator=Evaluator(store,frozen['protocol_hash'],'ENGINEERING_FIXTURE_NOT_SCIENTIFIC_RESULTS')
    fixture_path=directory/'temporary_U2.pt';fixture_identity=checkpoint(fixture_path,model,ema,opt,2,frozen,a.group,purpose=PURPOSE)
    aggregate=evaluator.evaluate(obj,rows,directory/'metric_fixture',update=2,weights='raw',checkpoint_hash=fixture_identity['sha256'])
    assert all(k in aggregate['overall'] for k in ('psnr','ssim','lpips','path_clean_MAE','path_weighted_reconstruction_MAE','path_total'))
    report['checks']['metric_fixture']={'count':aggregate['start_count'],'COMMIT_sha256':sha256(directory/'metric_fixture/COMMIT.json')}
    if a.group in ('R0','R08'):
        from pirn_v4.inference import infer_group
        from v4.matched import seed_for,SEED
        sample=rows[-1];c,y,g=store.batch([sample],'cuda');model.eval()
        before=module_hash(model)
        answer=infer_group(model,sim,y,g,group=a.group,noise_seeds=[seed_for(f'eval|{SEED}|{sample["start_id"]}')],refinement=True,retain_trace=True,audit_refinement=True)
        assert answer['refinement_sample_count']==3 and all(x.optimizer['state_steps_at_end']==[40,40] for x in answer['refinements'])
        assert module_hash(model)==before
        report['checks']['refinement']={'G':4,'serial_40_step':True,'seconds_per_stage_sample':answer['refinement_seconds']/3,'basis':'temporary_trained_G4_engineering_fixture_extrapolation_only'}
        del answer,c,y,g
    fcntl.flock(lock,fcntl.LOCK_UN);write('RESUME_AND_METRICS_PASS');barrier('ALL_COMPLETED')
    assert module_hash(sim)==sim_hash and sha256(ROOT/'protocol/common_init.pt')==common
    report.update(status='PASS',peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,common_U0_unchanged=True,simulator_unchanged=True,formal_updates=0)
    write('PASS_ALL_L1_TRAINING_GATES')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True,choices=['D0','R0','R08','L08']);p.add_argument('--attempt',required=True);a=p.parse_args()
    try:main(a)
    except BaseException as e:
        save_json(ROOT/'reports'/a.attempt/(a.group+'_FAILED.json'),{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'time':time.time(),'formal_updates':0});raise

"""One isolated GPU worker: disposable gate, training, or immutable evaluation."""
from lib import *
import argparse, traceback, signal

ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['gate','train','eval'],required=True);ap.add_argument('--method',choices=GROUPS,required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--update',type=int);a=ap.parse_args()
RUN=ROOT/('gates' if a.mode=='gate' else 'runs')/a.method
RUN.mkdir(parents=True,exist_ok=True)
STOP=False
def stopped(*_):
    global STOP
    STOP=True
def status(s,**kw):atomic(RUN/('EVAL_STATUS.json' if a.mode=='eval' else 'STATUS.json'),{'status':s,'mode':a.mode,'method':a.method,'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'gpu':a.gpu,'time':time.time(),**kw})
def gate(c,identity,store,device):
    import torch,numpy as np
    attempts=[]
    for micro in [8,4,2,1]:
        model=sim=opt=ema=None
        try:
            torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
            model,sim,opt=build(c,a.method,device);model.train();ema=copy.deepcopy(model)
            first=update(model,sim,opt,ema,store,store.batch_rows(0),c,1,micro,a.method,force_K=4,peak=True)
            second=update(model,sim,opt,ema,store,store.batch_rows(1),c,2,micro,a.method,force_K=4,peak=True)
            for values in [first,second]:
                assert math.isfinite(values['loss']) and values['L_R']<c['numerical_stop']['physics_L1'] and values['gradient_norm']<c['numerical_stop']['preclip_gradient_norm'],'NUMERICAL_GATE_UNSAFE'
            assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
            cap=torch.cuda.get_device_properties(0).total_memory
            assert max(first['peak_reserved_bytes'],second['peak_reserved_bytes'])+(1<<30)<.85*cap,'Unsafe training headroom'
            row=store.batch_rows(2)[0];C,Y=store.batch([row],device);gg=[[1.]] if a.method=='D0' else [grid_for(row['start_id'],3,c,4)]
            with torch.no_grad():
                expected=objective(model,sim,C,Y,gg)['final'].cpu().clone()
                pred=model(torch.cat((Y,Y),1),torch.tensor([[4.,0.]],device=device))
                prep=sim.prepare_image(pred.clean);render=lambda g:sim.render_prepared(prep,pred.realization,g).output
                f4=render(4.);zero=Y+(render(4.)-f4);zero_error=float((zero-Y).abs().max())
                walk=Y.clone();u=[1.,.78,.51,.19,0.]
                for x,y in zip(u,u[1:]):walk=walk+(render(1+3*y)-render(1+3*x))
                direct=Y+(render(1.)-f4);telescope=float((walk-direct).abs().max())
                assert zero_error<=2e-6 and telescope<=5e-6,(zero_error,telescope)
            receipt=save_checkpoint(RUN,model,ema,opt,store,2,micro,a.method,identity)
            # Verify complete restore, including optimizer step, RNG, sampler and
            # the deterministic next-grid prediction in a newly constructed model.
            del model,sim,opt,ema;torch.cuda.empty_cache()
            model,sim,opt=build(c,a.method,device);model.train();ema=copy.deepcopy(model)
            u,m=restore_checkpoint(receipt,model,ema,opt,store,identity,a.method);assert (u,m)==(2,micro)
            with torch.no_grad():actual=objective(model,sim,C,Y,gg)['final'].cpu()
            resume_error=float((actual-expected).abs().max());assert resume_error<=1e-6,resume_error
            if a.method=='D0':
                from v4.training import TrainingObjective
                with torch.no_grad():
                    old=TrainingObjective(model,sim,'D0')(C,Y,torch.tensor([4.],device=device),eta=torch.zeros_like(Y))
                    new=objective(model,sim,C,Y,[[1.]])
                    assert torch.equal(old['final_clean'],new['final'])
                    assert abs(float(old['loss_mean']-new['loss']))<=1e-7
            d={'status':'PASS','method':a.method,'microbatch':micro,'effective_batch':32,'disposable_updates':2,'peak_LR':c['peak_lr'],'first':first,'second':second,'zero_step_max_error':zero_error,'fixed_prediction_telescope_max_error':telescope,'save_resume_next_prediction_max_error':resume_error,'identity':identity,'attempts':attempts,'time':time.time()}
            atomic(RUN/'PASS.json',d);status('PASS',microbatch=micro)
            # Probe states are disposable; do not preserve an extra optimizer.
            for p in (RUN/'checkpoints').glob('full_U*.pt'):p.unlink();p.with_suffix('.identity.json').unlink()
            return
        except torch.cuda.OutOfMemoryError as e:
            attempts.append({'microbatch':micro,'error':'OOM','time':time.time()});atomic(RUN/'OOM_ATTEMPTS.json',attempts)
            del model,sim,opt,ema;torch.cuda.empty_cache()
    raise RuntimeError('GATE_OOM_AT_MICRO1')

def train(c,identity,store,device):
    import torch,numpy as np
    gate=read(ROOT/'gates'/a.method/'PASS.json');assert gate['status']=='PASS' and gate['identity']==identity
    torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    model,sim,opt=build(c,a.method,device);model.train();ema=copy.deepcopy(model);micro=gate['microbatch'];u=0
    if (RUN/'LATEST.json').exists():
        u,micro=restore_checkpoint(read(RUN/'LATEST.json'),model,ema,opt,store,identity,a.method)
        append(RUN/'recovery.jsonl',{'event':'same_run_resume','restored_update':u,'time':time.time()})
    else:save_checkpoint(RUN,model,ema,opt,store,0,micro,a.method,identity)
    timebox=read(ROOT/'TIMEBOX.json');status('RUNNING',update=u,microbatch=micro)
    durations=[];bad_updates=0
    while u<c['max_updates'] and not STOP and time.time()<timebox['training_deadline_server']:
        rr=store.batch_rows(u)
        try:r=update(model,sim,opt,ema,store,rr,c,u+1,micro,a.method)
        except torch.cuda.OutOfMemoryError:
            if micro==1:raise
            attempted=u+1;smaller=micro//2;torch.cuda.empty_cache()
            restored,_=restore_checkpoint(read(RUN/'LATEST.json'),model,ema,opt,store,identity,a.method)
            append(RUN/'recovery.jsonl',{'event':'OOM_COMPLETE_STATE_ROLLBACK','attempted_update':attempted,'restored_update':restored,'rollback_updates':u-restored,'old_micro':micro,'new_micro':smaller,'effective_batch':32,'time':time.time()})
            u=restored;micro=smaller;continue
        except (RuntimeError,AssertionError,ValueError) as error:
            text=str(error).lower()
            if not any(word in text for word in ['nonfinite','non-finite','non finite']):raise
            latest=read(RUN/'LATEST.json')
            failure={'status':'NUMERICAL_FAILURE_LAST_VALID_CHECKPOINT','attempted_update':u+1,'last_committed_update':latest['update'],'error_type':type(error).__name__,'error':str(error),'latest':latest,'time':time.time()}
            atomic(RUN/'NUMERICAL_FAILURE.json',failure)
            atomic(RUN/'TRAIN_FINISHED.json',{'status':failure['status'],'update':latest['update'],'exposures':latest['exposures'],'latest':latest,'identity':identity,'time':time.time()})
            status(failure['status'],update=latest['update'],failure_receipt='NUMERICAL_FAILURE.json');return
        u+=1;durations.append(r['seconds']);durations=durations[-40:]
        finite=all(math.isfinite(r[key]) for key in ['L_C','L_R','loss','gradient_norm'])
        if not finite:raise RuntimeError('Nonfinite scalar after completed update; latest committed checkpoint preserved')
        limits=c['numerical_stop'];bad_updates=bad_updates+1 if r['L_R']>limits['physics_L1'] and r['gradient_norm']>limits['preclip_gradient_norm'] else 0
        if u%20==0:
            append(RUN/'scalars.jsonl',r);status('RUNNING',update=u,exposures=u*32,microbatch=micro,last=r,mean_update_seconds=sum(durations)/len(durations),estimated_training_seconds_remaining=(c['max_updates']-u)*sum(durations)/len(durations))
        if u%c['checkpoint_interval']==0 or u==c['max_updates']:
            save_checkpoint(RUN,model,ema,opt,store,u,micro,a.method,identity)
        if u in c['milestones']:save_milestone(RUN,model,u,a.method,identity)
        if bad_updates>=limits['consecutive_updates']:
            latest=save_checkpoint(RUN,model,ema,opt,store,u,micro,a.method,identity)
            failure={'status':'LOSS_GRADIENT_EXPLOSION','update':u,'consecutive_updates':bad_updates,'last':r,'policy':limits,'action':'complete state saved at update boundary; no automatic scientific retry','time':time.time()}
            atomic(RUN/'NUMERICAL_FAILURE.json',failure)
            atomic(RUN/'TRAIN_FINISHED.json',{'status':'NUMERICAL_FAILURE_SAVED','update':u,'exposures':u*32,'latest':latest,'identity':identity,'failure_receipt':'NUMERICAL_FAILURE.json','time':time.time()})
            status('NUMERICAL_FAILURE_SAVED',update=u,exposures=u*32,failure_receipt='NUMERICAL_FAILURE.json');return
    if u and (not (RUN/'LATEST.json').exists() or read(RUN/'LATEST.json')['update']!=u):save_checkpoint(RUN,model,ema,opt,store,u,micro,a.method,identity)
    d={'status':'PASS' if u==c['max_updates'] else 'TIMEBOX_STOPPED','update':u,'exposures':u*32,'latest':read(RUN/'LATEST.json') if u else None,'identity':identity,'time':time.time()}
    atomic(RUN/'TRAIN_FINISHED.json',d);status(d['status'],update=u,exposures=u*32,microbatch=micro)

def evaluate(c,identity,store,device):
    import torch,csv
    from PIL import Image
    import numpy as np
    from nafnet_v4.metrics import Metrics,SPEC
    u=a.update;assert u in c['milestones'];dest=ROOT/'evaluation'/f'U{u:05d}'/a.method;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    cp=RUN/'milestones'/f'online_U{u:05d}.pt';d=read(cp.with_suffix('.identity.json'));assert d['identity']==identity and sha(cp)==d['sha256']
    env=read(CANDIDATE/'protocol/ENVIRONMENT.json');sys.path.insert(0,env['metrics_dependency_dir']);os.environ['TORCH_HOME']=env['TORCH_HOME']
    model,sim,_=build(c,a.method,device,with_optimizer=False);p=torch.load(cp,map_location='cpu',weights_only=False);assert p['purpose']==PURPOSE+'_ONLINE' and p['identity']==identity and p['update']==u and p['method']==a.method;model.load_state_dict(p['model'],strict=True);del p;model.eval();metric=Metrics()
    contract={'method':a.method,'update':u,'checkpoint_sha256':d['sha256'],'identity':identity,'population':[r['start_id'] for r in store.val],'metric_spec':SPEC,'K':1 if a.method=='D0' else 4,'dtype':'FP32','TF32':False,'weight':'online','refinement':False,'Test':False}
    freeze(dest/'IDENTITY.json',contract);allrows=[];latencies=[]
    chosen={}
    for domain in ['Places365','LSDIR']:
        chosen[domain]=sorted({r['source_id'] for r in store.val if r['domain']==domain},key=lambda s:seed('preview|'+s))[:2]
    for begin in range(0,len(store.val),16):
        if STOP or time.time()>=read(ROOT/'TIMEBOX.json')['gpu_deadline_server']:raise RuntimeError('EVALUATION_DEADLINE_PARTS_PRESERVED')
        target=dest/'parts'/f'{begin:04d}.json'
        if target.exists():
            part=read(target);assert [x['start_id'] for x in part]==[x['start_id'] for x in store.val[begin:begin+16]];allrows.extend(part);continue
        part=[]
        for row in store.val[begin:begin+16]:
            C,Y=store.batch([row],device);grid=[1.] if a.method=='D0' else [1-j/3 for j in range(4)]
            with torch.no_grad():
                torch.cuda.synchronize();t=time.monotonic();r=objective(model,sim,C,Y,[grid],trace=True);torch.cuda.synchronize();dt=time.monotonic()-t
                mm=[metric(x,C)[0] for x in r['trace']]
            values={'start_id':row['start_id'],'source_id':row['source_id'],'domain':row['domain'],'steps':mm,'seconds':dt,'state_ranges':r['ranges']};part.append(values)
            if row['source_id'] in chosen[row['domain']] and row['start_id'].endswith('r00::G4'):
                def png(x,name):
                    arr=(x.detach().cpu()[0].clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8);out=dest/'previews'/row['source_id'];out.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(out/(name+'.png'))
                png(Y,'input');png(C,'clean')
                for j,x in enumerate(r['trace']):png(x,f'C{j+1}')
            del C,Y,r
        atomic(target,part);allrows.extend(part);status('EVALUATING',update=u,completed=len(allrows),total=400)
    assert len(allrows)==400 and len({r['source_id'] for r in allrows})==200
    K=contract['K'];means=[]
    for j in range(K):means.append({k:math.fsum(x['steps'][j][k] for x in allrows)/len(allrows) for k in ['psnr','ssim','lpips']})
    result={'method':a.method,'update':u,'inputs':400,'sources':200,'step_means':means,'final':means[-1],'mean_seconds_per_input':math.fsum(r['seconds'] for r in allrows)/len(allrows),'identity_sha256':sha(dest/'IDENTITY.json'),'parts_sha256':{p.name:sha(p) for p in sorted((dest/'parts').glob('*.json'))},'time':time.time()}
    atomic(dest/'SUMMARY.json',result)
    with (dest/'per_input.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['start_id','source_id','domain','step','psnr','ssim','lpips','seconds']);w.writeheader()
        for r in allrows:
            for j,m in enumerate(r['steps']):w.writerow({**{k:r[k] for k in ['start_id','source_id','domain','seconds']},'step':j+1,**{k:m[k] for k in ['psnr','ssim','lpips']}})
    atomic(dest/'COMPLETE.json',{'status':'PASS','summary_sha256':sha(dest/'SUMMARY.json'),'per_input_sha256':sha(dest/'per_input.csv'),'time':time.time()});status('EVALUATION_COMPLETE',update=u)

def main():
    lock=(RUN/('evaluation.lock' if a.mode=='eval' else 'worker.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c,identity=verify();device=claim(a.gpu)
    import torch
    torch.set_num_threads(4);store=Store(c)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    if a.mode=='gate':gate(c,identity,store,device)
    elif a.mode=='train':train(c,identity,store,device)
    else:evaluate(c,identity,store,device)
if __name__=='__main__':
    try:main()
    except Exception as e:status('ERROR',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise

"""One independent GPU worker; guarded by complete frozen acceptance evidence."""
import argparse,gc,json,os,random,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deps'))
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import numpy as np
import torch
from v4.matched import Store,build,lr_at,GROUP_GPU
from v4.performance import backward32
import fcntl
from v4.runtime import configure_cuda,verify_imports,save_json,sha256
from v4.training import make_optimizer,make_ema,update_ema,finite_gradients,atomic_torch_save
from v4.refinement import module_hash
from v4.mutex import EvaluationMutex
from v4.evaluate_matched import Evaluator
from metrics import Metrics
from v4.stages import StageBarrier,read_control,late_stop,verify_budget_scope,NODES

STOP=False
RELOCATE=False
def relocate(signum,frame):
    global RELOCATE
    RELOCATE=True

def stop(signum,frame):
    global STOP
    STOP=True

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state()}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state(state['cuda'].cpu())

def verify_frozen():
    verify_budget_scope(ROOT)
    frozen=json.loads((ROOT/'protocol/FROZEN.json').read_text())
    if frozen['status']!='PASS_ALL_GATES' or not frozen['formal_training_authorized']:
        raise RuntimeError('All gates and explicit training authorization required')
    if frozen['updates']!=10000 or frozen['batch']!=32:raise ValueError('Budget changed')
    for name,digest in frozen['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    for name,digest in frozen['evidence_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError(f'Frozen gate changed: {name}')
    return frozen

def checkpoint(path,model,ema,opt,u,frozen,group,purpose='FORMAL_TRAIN500_MATCHED_V2'):
    if path.exists():
        identity=json.loads(path.with_suffix('.identity.json').read_text())
        if sha256(path)!=identity['sha256']:raise ValueError('Existing checkpoint identity corrupt')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if (saved['purpose']!=purpose or saved['update']!=u or saved['group']!=group or saved['protocol_hash']!=frozen['protocol_hash'] or
            saved['model_hash']!=module_hash(model) or saved['ema_hash']!=module_hash(ema)):
            raise ValueError('Refusing to overwrite existing immutable/recovery checkpoint')
        return identity
    payload={'purpose':purpose,'group':group,'update':u,'exposures':u*32,'epoch':u/250,
             'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'rng':capture_rng(),
             'protocol_hash':frozen['protocol_hash'],'stream_position':u,'stream_sha256':frozen['stream_sha256'],
             'scheduler':{'kind':'original_10k_warmup_cosine','completed_updates':u,'next_lr':lr_at(u+1) if u<10000 else None},
             'execution_hash':frozen.get('execution_hash',frozen['protocol_hash']),
             'stats_artifact_hash':model.stats.artifact_hash,'model_hash':module_hash(model),'ema_hash':module_hash(ema)}
    atomic_torch_save(payload,path)
    identity={'update':u,'exposures':u*32,'sha256':sha256(path),'protocol_hash':frozen['protocol_hash']}
    save_json(path.with_suffix('.identity.json'),identity);return identity

def main(args):
    global MUTEX
    frozen=verify_frozen()
    if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Run is explicitly PAUSED')
    if args.gpu not in (5,6) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise RuntimeError('Wrong physical GPU')
    configure_cuda(20260929);config=json.loads((ROOT/'configs/runtime.json').read_text());verify_imports(ROOT,config)
    result=Path(frozen['results_root'])/args.group;result.mkdir(parents=True,exist_ok=True)
    ckpt=result/'checkpoints';ckpt.mkdir(exist_ok=True)
    store=Store(config);store.cache_limit=8*1024**3;stream=json.loads((ROOT/'protocol/stream.json').read_text())
    model,sim,obj=build(config,args.group);opt=make_optimizer(model);ema=make_ema(model)
    u=0
    latest=result/'LATEST.json'
    if latest.exists():
        identity=json.loads(latest.read_text());path=result/identity['path']
        if sha256(path)!=identity['sha256']:raise ValueError('Resume checkpoint corrupt')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['purpose']!='FORMAL_TRAIN500_MATCHED_V2' or saved['protocol_hash']!=frozen['protocol_hash'] or saved['group']!=args.group:raise ValueError('Wrong resume')
        if saved['stream_sha256']!=frozen['stream_sha256'] or saved['stream_position']!=saved['update']:raise ValueError('Wrong stream cursor')
        model.load_state_dict(saved['model'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
        opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);u=saved['update'];del saved
    elif any(ckpt.glob('*.pt')):raise RuntimeError('Orphan checkpoint requires reviewed recovery')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGUSR1,relocate)
    members=args.members.split(',')
    if args.group not in members:raise ValueError('Missing self from GPU cohort')
    MUTEX=EvaluationMutex(Path(args.coordination),args.group,args.gpu,members,torch.cuda.synchronize)
    save_json(result/'EXECUTION_READY.json',{'pid':os.getpid(),'gpu':args.gpu,'execution_hash':frozen['execution_hash']})
    evaluator=Evaluator(store,frozen['protocol_hash'],frozen['environment_identity'])
    timebook={'training':0.,'evaluation':0.,'checkpoint_save':0.,'same_card_mutex_wait':0.,'stage_wait':0.,'same_card_compute_wait':0.}
    if (result/'TIMEBOOK.json').exists():timebook.update(json.loads((result/'TIMEBOOK.json').read_text()))
    stage_root=Path(frozen['results_root'])/'stages'
    stage=StageBarrier(stage_root,args.group,ROOT/'protocol/CONTROL.json',ROOT/'reports/RUN_PAUSED.json')
    def timing(kind,seconds):
        timebook[kind]+=seconds
        with (result/'timings.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,'seconds':seconds,'update':u,'time':time.time()})+'\n')
        save_json(result/'TIMEBOOK.json',timebook)
    def save_full():
        started=time.monotonic()
        path=ckpt/f'full_U{u:05d}.pt';identity=checkpoint(path,model,ema,opt,u,frozen,args.group)
        save_json(latest,{**identity,'path':str(path.relative_to(result))})
        # Preserve every1250 full state plus latest two rolling recovery states.
        rolling=sorted(p for p in ckpt.glob('full_U*.pt') if int(p.stem.split('U')[1])%1250!=0)
        for old in rolling[:-2]:old.unlink();old.with_suffix('.identity.json').unlink()
        timing('checkpoint_save',time.monotonic()-started)
    def weights_file():
        path=ckpt/f'weights_U{u:05d}.pt'
        if not path.exists():
            atomic_torch_save({'model':model.state_dict(),'ema':ema.state_dict(),'update':u,
                               'stats_artifact_hash':model.stats.artifact_hash,'protocol_hash':frozen['protocol_hash']},path)
            save_json(path.with_suffix('.identity.json'),{'sha256':sha256(path),'update':u,'protocol_hash':frozen['protocol_hash']})
        identity=json.loads(path.with_suffix('.identity.json').read_text())
        if sha256(path)!=identity['sha256']:raise ValueError('Weights checkpoint hash mismatch')
        return path,identity['sha256']
    def eval_node():
        epoch_dir=result/'evaluations'/f'U{u:05d}'
        if (epoch_dir/'NODE_COMMIT.json').exists():return
        started=time.monotonic();_,identity=weights_file();timing('checkpoint_save',time.monotonic()-started)
        started=time.monotonic();lease=MUTEX.boundary(u,f'U{u:05d}');timing('same_card_mutex_wait',time.monotonic()-started)
        eval_started=time.monotonic()
        if not lease:raise RuntimeError('Evaluation ticket returned without lease')
        before=capture_rng();metrics=Metrics('cuda')
        populations=['train8000','val800'] if u in (0,5000,10000) else ['train_monitor400','val800']
        try:
            for label,network in [('raw',model)]:
                obj.network=network
                for population in populations:
                    evaluator.evaluate(obj,store.lists[population],epoch_dir/label/population,
                        update=u,weights=label,checkpoint_hash=identity,metrics=metrics)
                if 'train8000' in populations:
                    evaluator.derive_subset(epoch_dir/label/'train8000',store.lists['train_monitor400'],
                                            epoch_dir/label/'train_monitor400')
            save_json(epoch_dir/'NODE_COMMIT.json',{'update':u,'checkpoint_hash':identity,'weights':['raw'],'evaluation_schedule_revision':'v3_RAW_fulltrain_at5k_and10k'})
        finally:
            obj.network=model;obj.train();del metrics;gc.collect();torch.cuda.empty_cache();restore_rng(before)
        MUTEX.complete(f'U{u:05d}')
        timing('evaluation',time.monotonic()-eval_started)
    if not latest.exists():save_full()
    while True:
        control=read_control(ROOT/'protocol/CONTROL.json')
        if late_stop(control,stage_root):
            save_full();MUTEX.terminal(u)
            save_json(result/'STATUS.json',{'status':'USER_STOPPED_AT_BOUNDARY','update':u,'exposures':32*u,
                  'requested_common_endpoint':control['stop_at_common_update'],'excess_updates_preserved':True})
            return
        if u%250==0:eval_node()
        if u in NODES:
            node=stage_root/f'U{u:05d}';released=node/'RELEASE.json'
            if not released.exists():
                MUTEX.park(u)
                save_json(node/f'{args.group}.READY.json',{'group':args.group,'update':u,'exposures':32*u,
                    'checkpoint_path':str(ckpt/f'full_U{u:05d}.pt'),'checkpoint_sha256':sha256(ckpt/f'full_U{u:05d}.pt'),
                    'evaluation_commit':str(result/'evaluations'/f'U{u:05d}'/'NODE_COMMIT.json'),
                    'evaluation_commit_sha256':sha256(result/'evaluations'/f'U{u:05d}'/'NODE_COMMIT.json'),
                    'ready_at':time.time()})
                MUTEX.terminal(u)
                save_json(result/'STATUS.json',{'status':'STAGE_READY_RELEASE_GPU','update':u,'exposures':32*u,
                    'gpu':args.gpu,'execution_hash':frozen['execution_hash']})
                return
            release=json.loads(released.read_text())
            if sha256(release['delivery_manifest'])!=release['delivery_manifest_sha256']:
                raise ValueError('Stage delivery changed')
            if release['action']!='continue':
                MUTEX.terminal(u)
                save_json(result/'STATUS.json',{'status':'COMMON_STAGE_COMPLETE_STOP','update':u,'exposures':32*u})
                return
        if u==10000:
            MUTEX.terminal(u)
            save_json(result/'STATUS.json',{'status':'COMPLETE_U10000','update':u,'exposures':u*32,'all_evaluation_nodes_complete':True})
            return
        started=time.monotonic();MUTEX.boundary(u);timing('same_card_mutex_wait',time.monotonic()-started)
        if RELOCATE:
            save_full();MUTEX.terminal(u)
            save_json(result/'STATUS.json',{'status':'ENGINEERING_RELOCATION_BOUNDARY','update':u,'exposures':32*u,
                 'gpu':args.gpu,'execution_hash':frozen['execution_hash']})
            return
        if STOP:
            save_full();MUTEX.fail('External stop at complete optimizer boundary; reviewed resume required')
            save_json(result/'STATUS.json',{'status':'PAUSED_AT_BOUNDARY','update':u});return
        if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Global pause requested')
        if late_stop(read_control(ROOT/'protocol/CONTROL.json'),stage_root):continue
        lock_path=Path(args.coordination)/f'compute_gpu{args.gpu}.lock'
        waiting=time.monotonic()
        with lock_path.open('a+') as compute_lock:
            if frozen['compute_scheduling']=='serialized_updates':fcntl.flock(compute_lock,fcntl.LOCK_EX)
            timing('same_card_compute_wait',time.monotonic()-waiting)
            start=time.monotonic();next_u=u+1;epoch=u//250+1
            rows=[store.by_id[s] for s in stream['updates'][u]]
            opt.zero_grad(set_to_none=True);values=backward32(obj,store,rows,frozen['microbatch'],epoch)
            finite_gradients(model);gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            for pg in opt.param_groups:pg['lr']=lr_at(next_u)
            opt.step();update_ema(ema,model,32)
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite updated weights')
            torch.cuda.synchronize();u=next_u
            update_seconds=time.monotonic()-start
            if frozen['compute_scheduling']=='serialized_updates':fcntl.flock(compute_lock,fcntl.LOCK_UN)
        log={'update':u,'epoch':u/250,'exposures':32*u,'lr':lr_at(u),'path_clean_MSE':values[0],
             'path_reconstruction_MSE':values[1],'path_total':values[2],'gradient_norm_before_clip':gradnorm,
             'update_seconds':update_seconds,'gpu':args.gpu,'execution_hash':frozen['execution_hash'],'time':time.time(),'start_ids':stream['updates'][u-1],
             'protocol_hash':frozen['protocol_hash']}
        timing('training',log['update_seconds'])
        with (result/'train_steps.jsonl').open('a') as f:f.write(json.dumps(log,allow_nan=False)+'\n')
        save_json(result/'STATUS.json',{'status':'TRAINING','update':u,'epoch':u/250,'exposures':32*u,'last_update_seconds':log['update_seconds']})
        if u%25==0:save_full()
        if u%250==0:weights_file()

MUTEX=None
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=list(GROUP_GPU),required=True);p.add_argument('--coordination',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--members',required=True);args=p.parse_args()
    try:main(args)
    except BaseException as exc:
        if MUTEX is not None:MUTEX.fail(repr(exc))
        save_json(ROOT/'reports/RUN_PAUSED.json',{'status':'PAUSED','group':args.group,'exception':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise

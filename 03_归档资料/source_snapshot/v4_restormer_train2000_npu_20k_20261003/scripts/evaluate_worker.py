"""Idempotent evaluation jobs; no stage barrier and no training optimizer calls.

run_job(job, config=..., store=..., model=..., simulator=..., objective=...) may
reuse the live model at an update boundary. A separate daemon consumes immutable
checkpoint jobs. Test is forbidden. Each completed sample is resumable.
"""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import argparse,fcntl,hashlib,importlib.metadata,json,math,os,random,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ.setdefault('TORCH_HOME',str(ROOT/'assets/torch'))
import numpy as np
import torch
from evaluate_metrics import Metrics,SPEC
from v4.data import PhysicalDataset,canonical_hash,read_array
from v4.normalization import load_artifact
from v4.runtime import sha256

IDENTITY_KEYS=('start_id','trajectory_id','source_id','split','G')
METRIC_KEYS=('mse','psnr','ssim','lpips','path_clean_MAE','path_reconstruction_MAE','path_total','psnr_first','delta_prog_PSNR')
SEED=20261003

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temp.replace(path)

def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def eta_seed(start_id):return int.from_bytes(hashlib.sha256(f'eval|{SEED}|{start_id}'.encode()).digest()[:8],'little')%(2**63-1)
def eta_batch(rows,device):
    return torch.stack([torch.randn((3,256,256),generator=torch.Generator(device='cpu').manual_seed(eta_seed(r['start_id']))) for r in rows]).to(device)

def aggregate(rows):
    def means(items):return {k:float(np.mean([r[k] for r in items],dtype=np.float64)) for k in METRIC_KEYS if items and all(k in r for r in items)}
    sources=defaultdict(list);groups=defaultdict(list)
    for r in rows:sources[r['source_id']].append(r);groups[str(r['G'])].append(r)
    return {'overall':means(rows),'per_source':{k:means(v) for k,v in sorted(sources.items())},'per_G':{k:means(v) for k,v in sorted(groups.items())},'start_count':len(rows),'source_count':len(sources)}

class EvalStore:
    def __init__(self,config):
        self.root=Path(config['dataset_root'])
        if not (self.root/'DATA_READY.json').exists():raise ValueError('Canonical dataset is not frozen DATA_READY')
        dataset=PhysicalDataset(self.root,'train',config.get('asset_lock') or self.root/'ASSET_LOCK.json')
        stats=load_artifact(ROOT/config['normalization_artifact'])
        if stats['population']!='train' or stats['state_count']!=32000 or stats['train_manifest_hash']!=dataset.train_manifest_hash or stats['manifest_sha256']!=dataset.manifest_hashes:
            raise ValueError('Canonical train normalization/manifest identity mismatch')
        source=read_jsonl(self.root/'source_manifest.jsonl');self.source_by_id={r['source_id']:r for r in source}
        trajectories=read_jsonl(self.root/'trajectory_manifest.jsonl')
        def records(split):
            return [{'start_id':f'{r["trajectory_id"]}::G{g}','trajectory_id':r['trajectory_id'],'source_id':r['source_id'],'split':split,'G':g,'clean':r['arrays']['clean'],'state':r['arrays'][f'X_{g}']} for r in sorted(trajectories,key=lambda x:x['trajectory_id']) if r['split']==split for g in (1,2,3,4)]
        self.train=records('train');self.val=records('val')
        if len(self.train)!=32000 or len(self.val)!=2000:raise ValueError('Wrong canonical evaluation population')
        if {r['source_id'] for r in self.train}&{r['source_id'] for r in self.val}:raise ValueError('Train/val source overlap')
        self.cache={}
    def batch(self,rows,device):
        if not rows or any(r['split'] not in ('train','val') for r in rows):raise ValueError('Test evaluation is locked')
        def array(record):return torch.from_numpy(read_array(self.root,record))
        return (torch.stack([array(r['clean']) for r in rows]).to(device),torch.stack([array(r['state']) for r in rows]).to(device),torch.tensor([r['G'] for r in rows],dtype=torch.float32,device=device))

def populations(store,config):
    """Freeze 50 sources (all categories where metadata is available), xi0 and all G."""
    root=Path(config['dataset_root']);sources={r['source_id']:r for r in read_jsonl(root/'source_manifest.jsonl')}
    def monitor(rows):
        groups=defaultdict(list)
        for sid in sorted({r['source_id'] for r in rows}):
            s=sources[sid];category=s.get('category',s.get('category_name',s.get('scene_category','unknown')))
            if isinstance(category,dict):category=canonical_hash(category)
            groups[str(category)].append(sid)
        rng=random.Random(SEED);selected=[]
        if len(groups)==50:
            for _,ids in sorted(groups.items()):selected.append(rng.choice(ids))
        else:
            ids=sorted({r['source_id'] for r in rows});rng.shuffle(ids);selected=ids[:50]
        first={s:min(r['trajectory_id'] for r in rows if r['source_id']==s) for s in selected}
        return sorted([r for r in rows if r['source_id'] in first and r['trajectory_id']==first[r['source_id']]],key=lambda x:x['start_id'])
    return {'train_monitor':monitor(store.train),'val_monitor':monitor(store.val),'val2000':store.val,'train32000':store.train}

@contextmanager
def preserve_execution(model,objective):
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    device=next(model.parameters()).device
    backend=getattr(torch,device.type,None)
    accel_rng=backend.get_rng_state(device) if device.type in ('npu','cuda') else None
    modes={id(m):(m,m.training) for m in objective.modules()}
    amp=getattr(model,'amp_dtype',None)
    try:
        objective.eval()
        # Model adapters may select accelerator mixed precision for training.
        # Evaluation explicitly records and uses their configured precision.
        yield
    finally:
        for m,mode in modes.values():m.training=mode
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        if accel_rng is not None:backend.set_rng_state(accel_rng,device)


def evaluate_population(*,objective,store,rows,destination,job,metrics,refinement=False):
    from restormer_v4.inference import infer_group
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    expected={'checkpoint_sha256':job['checkpoint_sha256'],'method':job['method'],'update':job['update'],'population_hash':canonical_hash([{k:r[k] for k in IDENTITY_KEYS} for r in rows]),'refinement':refinement,'metrics_spec':SPEC,'network_batch':1,'tail':'none','seed':SEED,'eta_generator':'CPU_FP32','test_used':False,'execution':job['evaluation_execution']}
    identity=destination/'IDENTITY.json'
    if identity.exists() and json.loads(identity.read_text())!=expected:raise ValueError('Cannot mix evaluation identities')
    save_json(identity,expected)
    committed=destination/'COMMIT.json'
    if committed.exists():
        commit=json.loads(committed.read_text())
        for rel,digest in commit['sha256'].items():
            if sha256(destination/rel)!=digest:raise ValueError('Committed evaluation corruption')
        return json.loads((destination/'aggregate.json').read_text())
    device=next(objective.network.parameters()).device
    parts=destination/'parts';parts.mkdir(exist_ok=True)
    all_rows=[];baseline=[];started=time.monotonic();refinement_seconds=0.
    visual_ids={r['start_id'] for r in rows if r['G']==4} # select by frozen record order, never quality
    visual_ids=set(sorted(visual_ids)[:8])
    for i,r in enumerate(rows):
        path=parts/f'{i:06d}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['row']['start_id']!=r['start_id']:raise ValueError('Partial row order mismatch')
            all_rows.append(old['row']);baseline.append(old['baseline']);refinement_seconds+=old.get('refinement_seconds',0.);continue
        C,Y,G=store.batch([r],device)
        with torch.no_grad():
            if refinement:
                result=infer_group(objective.network,objective.simulator,Y,G,group=job['method'],noise_seeds=[eta_seed(r['start_id'])],refinement=True,retain_trace=True)
                output=result['clean'];extra={'refinement_sample_count':result['refinement_sample_count'],'refinement_state_steps':[item.optimizer['state_steps_at_end'] for item in result['refinements']]}
            else:
                result=objective(C,Y,G,eta=eta_batch([r],device))
                output=result['final_clean'];first_error=float((result['first_clean'].float().cpu()-C.cpu()).square().mean())
                extra={'path_clean_MAE':float(result['path_clean'][0]),'path_reconstruction_MAE':float(result['path_reconstruction'][0]),'path_total':float(result['path_losses'][0]),'psnr_first':-10*math.log10(first_error)}
            row={**{k:r[k] for k in IDENTITY_KEYS},**metrics(output,C)[0],**extra,'method':job['method'],'update':job['update'],'exposures':32*job['update'],'epoch':job['update']/1000,'eta_seed':eta_seed(r['start_id']),'refinement':'ON_serial_40' if refinement else 'OFF'}
            if not refinement:row['delta_prog_PSNR']=row['psnr']-row['psnr_first']
            base={**{k:r[k] for k in IDENTITY_KEYS},**metrics(Y,C)[0]}
            if r['start_id'] in visual_ids:
                visual=destination/'fixed_fp32';visual.mkdir(exist_ok=True);stem=r['start_id'].replace('::','_').replace('/','_')
                for label,tensor in [('output',output[0]),('input',Y[0]),('clean',C[0])]:np.save(visual/f'{stem}_{label}.npy',tensor.detach().float().cpu().numpy(),allow_pickle=False)
            seconds=result.get('refinement_seconds',0.)
            save_json(path,{'row':row,'baseline':base,'refinement_seconds':seconds})
            all_rows.append(row);baseline.append(base);refinement_seconds+=seconds
        if (i+1)%20==0:save_json(destination/'PROGRESS.json',{'complete_starts':i+1,'total_starts':len(rows),'wall_seconds_this_attempt':time.monotonic()-started,'time':time.time()})
        del C,Y,G,result,output
    (destination/'per_start.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in all_rows))
    result={**aggregate(all_rows),'baseline':aggregate(baseline),'method':job['method'],'update':job['update'],'refinement':expected['refinement'],'checkpoint_sha256':job['checkpoint_sha256'],'elapsed_seconds_this_attempt':time.monotonic()-started,'refinement_seconds':refinement_seconds}
    save_json(destination/'aggregate.json',result)
    save_json(committed,{'identity':expected,'sha256':{str(p.relative_to(destination)):sha256(p) for p in destination.rglob('*') if p.is_file() and p.name not in ('COMMIT.json','PROGRESS.json')},'time':time.time()})
    return result


class EvaluationBusy(RuntimeError):
    """An inline or standalone evaluator owns this method/update already."""

def evaluation_lock_path(job):
    return Path(job['run_dir'])/'evaluation_locks'/f'U{int(job["update"]):05d}.lock'

def record_best_validation(run_dir,candidate):
    """Protect the tiny cross-update selection transaction, without inference."""
    run_dir=Path(run_dir);folder=run_dir/'evaluation_locks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'BEST_VAL.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        path=run_dir/'BEST_VAL.json';previous=json.loads(path.read_text()) if path.exists() else None
        score=candidate['psnr'];tolerance=candidate['tie_tolerance_dB']
        if previous is None or score>previous['psnr']+tolerance or (abs(score-previous['psnr'])<=tolerance and candidate['update']<previous['update']):save_json(path,candidate)

def run_job(job,*,config=None,store=None,model=None,simulator=None,objective=None,device='cpu',queue_path=None):
    """Serialize every kind sharing a method/update, before accelerator setup.

    The persistent lock file must never be unlinked: all callers need the same
    inode. Busy calls defer immediately and do not allocate a model or device.
    """
    path=evaluation_lock_path(job);path.parent.mkdir(parents=True,exist_ok=True)
    marker=Path(queue_path).with_suffix('.running.json') if queue_path is not None else None
    with path.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise EvaluationBusy(f'Evaluation already owns {path}') from exc
        owner={'pid':os.getpid(),'time':time.time(),'device':str(device),'method':job['method'],'update':job['update'],'kind':job['kind'],'lock':str(path)}
        try:
            lock.seek(0);lock.truncate();lock.write(json.dumps(owner,sort_keys=True)+'\n');lock.flush()
            if marker is not None:save_json(marker,owner)
            return _run_job(job,config=config,store=store,model=model,simulator=simulator,objective=objective,device=device)
        finally:
            if marker is not None and marker.exists():
                # A duplicate that failed to acquire the lock never touches this
                # owner's marker. Be conservative if external recovery replaced it.
                current=json.loads(marker.read_text())
                if current.get('pid')==owner['pid'] and current.get('time')==owner['time']:marker.unlink()
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _run_job(job,*,config=None,store=None,model=None,simulator=None,objective=None,device='cpu'):
    job=dict(job)
    if job.get('kind') not in ('monitor','fullval','milestone'):raise ValueError('Unknown evaluation job')
    if job['method'] not in ('D0','R0','R08','L08'):raise ValueError('Unknown method')
    config=dict(config or json.loads(Path(job.get('config',ROOT/'configs/runtime.json')).read_text()))
    store=store or EvalStore(config)
    checkpoint=Path(job['checkpoint']);actual=sha256(checkpoint)
    if job.get('checkpoint_sha256',actual)!=actual:raise ValueError('Evaluation checkpoint corrupt')
    job['checkpoint_sha256']=actual
    if objective is None:
        runtime_path=Path(job['run_dir'])/'RUNTIME.json'
        if not runtime_path.exists():
            prefix='CUDA_TRAINING_ACCEPTANCE' if str(device).startswith('cuda') else 'TRAINING_ACCEPTANCE'
            runtime_path=ROOT/'reports'/f'{prefix}_{job["method"]}.json'
        if runtime_path.exists():
            accepted=json.loads(runtime_path.read_text());accepted=accepted.get('runtime',accepted)
            config['precision']=accepted['precision'];config['model_precision']=accepted['precision']
            config['network']=dict(config['network']);config['network']['use_checkpoint']=accepted.get('activation_checkpoint',True)
        if str(device).startswith('npu'):
            from npu_runtime.backend import configure
            configure(int(str(device).split(':')[-1]),config['precision'])
        elif str(device).startswith('cuda'):
            from execution.devices import setup
            setup(int(str(device).split(':')[-1]),config['precision'],'cuda')
        elif str(device)=='cpu' and job.get('formal',True):
            raise RuntimeError('Formal heavy inference waits for an authorized released NPU; CPU formal inference would mix evaluation execution identities')
        from restormer_v4.factory import build
        model,simulator,objective=build(config,job['method'],device,ROOT)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['update']!=job['update'] or saved.get('method',saved.get('group'))!=job['method']:raise ValueError('Wrong checkpoint update/method')
        model.load_state_dict(saved['model'],strict=True)
    else:model=objective.network;simulator=objective.simulator
    actual_device=next(model.parameters()).device
    execution={'device_type':actual_device.type,'precision':getattr(model,'precision','fp32'),'torch':str(torch.__version__),'metrics_device':'cpu','physics_dtype':'float32','tf32':False,'hf32':False,'jit_compile':False if actual_device.type=='npu' else None,'metric_packages':{name:importlib.metadata.version(name) for name in ('lpips','torchvision','scikit-image','numpy')}}
    if actual_device.type=='npu':
        import torch_npu
        # torch-npu 2.7 exposes a setter only on npu.config; read the backend option.
        internal_format = torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
        if internal_format not in (b'enable', b'disable'):
            raise RuntimeError(f'Unknown NPU internal format option: {internal_format!r}')
        execution.update(torch_npu=str(torch_npu.__version__),internal_format=internal_format == b'enable')
    job['evaluation_execution']=execution
    job['evaluation_physical_device']=str(actual_device)
    available=populations(store,config)
    run_dir=Path(job['run_dir']);dest=run_dir/'evaluations'/f'U{job["update"]:05d}'
    plan={k:[{x:r[x] for x in IDENTITY_KEYS} for r in v] for k,v in available.items() if 'monitor' in k}
    plan_path=run_dir/'EVALUATION_POPULATIONS.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen monitor populations changed')
    save_json(plan_path,plan)
    names={'monitor':['train_monitor','val_monitor'],'fullval':['val2000'],'milestone':['val2000','train32000']}[job['kind']]
    with preserve_execution(model,objective):
        metrics=Metrics();results={}
        for name in names:results[name]=evaluate_population(objective=objective,store=store,rows=available[name],destination=dest/'raw'/name,job=job,metrics=metrics)
        if job['kind']=='milestone' and job['method'] in ('R0','R08'):
            results['val2000_refinement']=evaluate_population(objective=objective,store=store,rows=available['val2000'],destination=dest/'refinement'/ 'val2000',job=job,metrics=metrics,refinement=True)
        save_json(dest/f'{job["kind"]}_DONE.json',{'job':job,'populations':list(results),'time':time.time()})
        if 'val2000' in results:
            record_best_validation(run_dir,{'criterion':'RAW_OFF_full_val2000_mean_PSNR','psnr':results['val2000']['overall']['psnr'],'update':job['update'],'checkpoint':str(checkpoint),'checkpoint_sha256':actual,'tie_tolerance_dB':1e-6,'tie_choose':'earlier','fixed_grid':sorted(set(range(2000,20001,2000))|{5000}),'refinement_used':False,'test_used':False,'evaluation_execution':execution})
    if job['update'] in (5000,10000,20000):
        log=run_dir.parent/'reports'/f'report_u{job["update"]:05d}.log';log.parent.mkdir(exist_ok=True,parents=True)
        with log.open('ab') as output:subprocess.Popen([sys.executable,str(ROOT/'scripts/report_async.py'),'--run-dir',str(run_dir.parent),'--update',str(job['update'])],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path);p.add_argument('--job',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--once',action='store_true');p.add_argument('--wait-for-training-done',type=Path);p.add_argument('--poll-seconds',type=float,default=20)
    args=p.parse_args();torch.set_num_threads(4)
    while args.wait_for_training_done and not args.wait_for_training_done.exists():time.sleep(args.poll_seconds)
    while True:
        jobs=[args.job] if args.job else sorted(args.queue.glob('*.json'))
        deferred=False;failed=False
        for path in jobs:
            if path.name.endswith(('.done.json','.error.json','.running.json','.deferred.json')):continue
            done=path.with_suffix('.done.json')
            if done.exists():continue
            failure=path.with_suffix('.error.json')
            if failure.exists() and json.loads(failure.read_text()).get('attempts',0)>=3:continue
            job=json.loads(path.read_text())
            try:
                result=run_job(job,device=args.device,queue_path=path)
                save_json(done,{'completed':time.time(),'populations':list(result)})
                path.with_suffix('.deferred.json').unlink(missing_ok=True)
            except EvaluationBusy as exc:
                deferred=True
                save_json(path.with_suffix('.deferred.json'),{'reason':str(exc),'time':time.time(),'pid':os.getpid(),'lock':str(evaluation_lock_path(job)),'attempts_consumed':0,'training_affected':False})
            except Exception as exc:
                failed=True
                old=json.loads(failure.read_text()) if failure.exists() else {}
                save_json(failure,{'error':repr(exc),'traceback':traceback.format_exc(),'attempts':old.get('attempts',0)+1,'time':time.time(),'training_affected':False})
        if args.once or args.job:return 1 if failed else (75 if deferred else 0)
        time.sleep(args.poll_seconds)

if __name__=='__main__':sys.exit(main())

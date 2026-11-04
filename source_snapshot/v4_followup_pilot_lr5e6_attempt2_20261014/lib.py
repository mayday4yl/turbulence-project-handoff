"""Isolated, bounded two-arm pilot. Old experiment files are read only."""
from pathlib import Path
from collections import OrderedDict, Counter
import os, sys, json, hashlib, random, math, time, fcntl, subprocess, pwd, copy

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'v4_nafnet_sidd116m_train8000_20261006'
CANDIDATE = ROOT.parent / 'v4_nafnet_sidd116m_candidate_20261005'
sys.path.insert(0, str(CANDIDATE))
GROUPS = ('D0', 'CONT_DELTA_R0')
PURPOSE = 'V4_G4_CONTINUOUS_DELTA_PILOT_LR5E6_ATTEMPT2_20261014'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def canonical(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seed(s): return int(hashlib.sha256(s.encode()).hexdigest()[:16],16) % (2**63-1)
def atomic(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with q.open('w') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)
def freeze(p,x):
    if Path(p).exists(): assert read(p)==x, f'Frozen identity differs: {p}'
    else: atomic(p,x)
def append(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f: f.write(json.dumps(x,allow_nan=False)+'\n'); f.flush()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def ticks(pid):
    try:
        a=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
        return None if a[0]=='Z' else a[19]
    except FileNotFoundError: return None
def config(): return read(ROOT/'CONFIG.json')
def sources(): return {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
def verify():
    c=config(); ident=read(ROOT/'IDENTITY.json')
    assert canonical(c)==ident['config_hash']
    assert sources()==read(ROOT/'SOURCE_LOCK.json')['files'], 'New pilot source identity changed'
    assert sha(ROOT/'SUBSET.json')==ident['subset_sha256']
    assert sha(PARENT/'protocol/common_init.pt')==ident['common_init_sha256']
    return c,ident
def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus={}
    for line in raw.strip().splitlines():
        i,u,t,m=[x.strip() for x in line.split(',')];gpus[int(i)]={'uuid':u,'total_MiB':int(t),'used_MiB':int(m)}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    pp=[]
    for line in apps.strip().splitlines():
        if not line.strip(): continue
        u,p,m=[x.strip() for x in line.split(',')];pid=int(p)
        try: uid=Path('/proc',p).stat().st_uid
        except FileNotFoundError: uid=None
        pp.append({'gpu':next(i for i,x in gpus.items() if x['uuid']==u),'pid':pid,'uid':uid,'MiB':m})
    return {'gpus':gpus,'processes':pp,'account_uid':os.getuid(),'time':time.time()}
def resource_check(gpu):
    c=config();assert gpu in c['allowed_gpus'] and len(c['allowed_gpus'])==2
    inv=inventory();mine={x['gpu'] for x in inv['processes'] if x['uid']==os.getuid()}
    unknown={x['gpu'] for x in inv['processes'] if x['uid'] is None}
    if len(mine|unknown|{gpu})>2: raise RuntimeError('WAIT_RESOURCE_ACCOUNT_OR_UNKNOWN_TWO_GPU_LIMIT')
    if any(x['gpu']==gpu and x['pid']!=os.getpid() for x in inv['processes']): raise RuntimeError('WAIT_RESOURCE_OCCUPIED')
    if inv['gpus'][gpu]['used_MiB']>1024: raise RuntimeError('WAIT_RESOURCE_UNEXPLAINED_MEMORY')
    return inv
def claim(gpu):
    with (ROOT/'account_launch.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);inv=resource_check(gpu)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
        import torch
        torch.cuda.set_device(0);torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        atomic(ROOT/'claims'/f'{os.getpid()}.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'gpu':gpu,'inventory':inv})
    return 'cuda:0'
def tree_cpu(x):
    import torch
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:tree_cpu(v) for k,v in x.items()}
    if isinstance(x,list): return [tree_cpu(v) for v in x]
    if isinstance(x,tuple): return tuple(tree_cpu(v) for v in x)
    return copy.deepcopy(x)
def rng_state():
    import torch,numpy as np
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def rng_restore(r):
    import torch,numpy as np
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def tensor_save(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with q.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def storage_check(additional=0):
    c=config();base=read(ROOT/'STORAGE_BASELINE.json')
    usage=sum(int(subprocess.check_output(['du','-sx','-B1',p],text=True).split()[0]) for p in c['pilot_storage_roots'])
    if usage+additional>c['pilot_disk_limit_bytes']:raise RuntimeError('STORAGE_PILOT_LIMIT')
    # Baseline accounts for the entire account. Other projects may grow; fresh
    # account scan is performed by controller before formal launch and each hour.
    if base['account_allocated_bytes']+usage+additional+c['account_safety_bytes']>c['account_quota_bytes']:
        raise RuntimeError('STORAGE_ACCOUNT_HEADROOM')
    return usage

class Store:
    def __init__(self,c):
        self.c=c;self.root=Path(c['dataset_root']);self.subset=read(ROOT/'SUBSET.json')
        self.train=self.subset['train'];self.val=self.subset['val'];self.cache=OrderedDict();self.cache_bytes=0
        assert len(self.train)==4000 and len(self.val)==400
        assert not {x['source_id'] for x in self.train}&{x['source_id'] for x in self.val}
    def array(self,r):
        import torch
        from v4.data import read_array
        k=(r['ref'],r['sha256'])
        if k in self.cache:self.cache.move_to_end(k);return self.cache[k]
        a=torch.from_numpy(read_array(self.root,r));n=a.numel()*a.element_size()
        while self.cache and self.cache_bytes+n>self.c['cache_bytes']:
            _,old=self.cache.popitem(last=False);self.cache_bytes-=old.numel()*old.element_size()
        self.cache[k]=a;self.cache_bytes+=n;return a
    def batch(self,rr,device):
        import torch
        assert all(r['split'] in ['train','val'] and r['G']==4 for r in rr)
        return torch.stack([self.array(r['clean']) for r in rr]).to(device),torch.stack([self.array(r['state']) for r in rr]).to(device)
    def batch_rows(self,u):
        assert 0<=u<self.c['max_updates'];B=self.c['effective_batch'];epoch=(u*B)//len(self.train)
        pools={d:[r for r in self.train if r['domain']==d] for d in ['Places365','LSDIR']}
        for d,p in pools.items():random.Random(seed(f'pilot|{self.c["seed"]}|{epoch}|{d}')).shuffle(p)
        start=(u*B%len(self.train))//2
        return [pools[d][j] for j in range(start,start+B//2) for d in pools]
    def stream_state(self,u):
        return {'completed_updates':u,'completed_exposures':u*self.c['effective_batch'],'next_ids':[] if u==self.c['max_updates'] else [r['start_id'] for r in self.batch_rows(u)]}

def grid_for(start_id,update,c,K=None):
    r=random.Random(seed(f'grid|{c["seed"]}|{update}|{start_id}'))
    K=r.choice([2,3,4]) if K is None else K
    if K==1:return [1.]
    u=[1.]+[1-j/(K-1)+(r.random()-.5)*.4/(K-1) for j in range(1,K-1)]+[0.]
    assert all(u[j]-u[j+1]>=1/32 for j in range(len(u)-1));return u

def build(c,method,device,with_optimizer=True):
    import torch
    from nafnet_v4.model import NAFNetNoiseBridge
    from nafnet_v4.pretrained import _mark_pretrained
    from nafnet_v4.factory import build_optimizer
    from v4.normalization import FrozenRGBStats
    from physical_bridge.simulator import PiRNP2SSimulator
    class ContinuousNAF(NAFNetNoiseBridge):
        def _validate_inputs(self,image,condition):
            if image.ndim!=4 or tuple(image.shape[1:])!=(6,self.image_size,self.image_size) or len(image)<1:raise ValueError('Invalid RGB concat')
            if image.dtype!=torch.float32 or condition.dtype!=torch.float32 or condition.shape!=(len(image),2) or condition.device!=image.device:raise ValueError('Invalid continuous condition')
            if not torch.isfinite(image).all() or not torch.isfinite(condition).all():raise ValueError('Nonfinite inputs')
            g,t=condition[:,0],condition[:,1]
            if not ((g>=1)&(g<=4)).all() or not ((t>=0)&(t<=1)).all():raise ValueError('g/t out of range')
    cls=ContinuousNAF if method=='CONT_DELTA_R0' else NAFNetNoiseBridge
    stats=Path(c['dataset_root'])/'normalization_train8000.json'
    model=_mark_pretrained(cls(FrozenRGBStats.from_artifact(stats),**c['network']))
    p=torch.load(PARENT/'protocol/common_init.pt',map_location='cpu',weights_only=True)
    assert p['V4_optimizer_updates']==0;model.load_state_dict(p['state_dict'],strict=True);del p
    assert sum(p.numel() for p in model.parameters())==117767209
    model.precision='fp32';model=model.to(device)
    sim=PiRNP2SSimulator(c['asset_dir'],**c['physics'],device=device).to(device).eval();sim.requires_grad_(False)
    opt=build_optimizer(model) if with_optimizer else None
    if opt is not None:
        for g in opt.param_groups:g['lr']=c['peak_lr']
    return model,sim,opt

def objective(model,sim,C,Y,grids,trace=False):
    """A microbatch has one K, but each input has its own continuous grid."""
    import torch
    from v4.core import mae
    assert len(grids)==len(Y) and len({len(g) for g in grids})==1
    K=len(grids[0]);Q=Y;lc=Y.new_zeros(len(Y));lr=Y.new_zeros(len(Y));out=[];ranges=[]
    for j in range(K):
        g=torch.tensor([1+3*u[j] for u in grids],device=Y.device,dtype=torch.float32)
        pred=model(torch.cat((Q,Y),1),torch.stack((g,torch.zeros_like(g)),1))
        prep=sim.prepare_image(pred.clean);F4=sim.render_prepared(prep,pred.realization,torch.full_like(g,4.)).output
        lc=lc+mae(pred.clean,C);lr=lr+mae(F4,Y)
        if trace:
            out.append(pred.clean)
            limits=lambda x:[float(x.detach().min()),float(x.detach().max())]
            xi=pred.realization
            xi_limits=({'zernike':limits(xi.epsilon_zernike),'tilt':limits(xi.epsilon_tilt)} if hasattr(xi,'epsilon_zernike') else limits(xi))
            ranges.append({'step':j+1,'g':limits(g),'Q':limits(Q),'clean':limits(pred.clean),'xi':xi_limits,'F4':limits(F4)})
        if j<K-1:
            gn=torch.tensor([1+3*u[j+1] for u in grids],device=Y.device,dtype=torch.float32)
            current=F4 if j==0 else sim.render_prepared(prep,pred.realization,g).output
            nxt=sim.render_prepared(prep,pred.realization,gn).output
            Q=Q+(nxt-current)
            if trace:ranges[-1].update({'current_render':limits(current),'next_render':limits(nxt),'next_Q':limits(Q)})
    lc=lc/K;lr=lr/K
    return {'loss':(lc+.2*lr).mean(),'lc':lc,'lr':lr,'final':pred.clean,'trace':out,'ranges':ranges,'K':K}
def lr_at(u,c):
    if u<=c['warmup_updates']:return c['peak_lr']*u/c['warmup_updates']
    x=(u-c['warmup_updates'])/(c['max_updates']-c['warmup_updates'])
    return c['min_lr']+.5*(c['peak_lr']-c['min_lr'])*(1+math.cos(math.pi*x))
def update(model,sim,opt,ema,store,rr,c,u,micro,method,force_K=None,peak=False):
    import torch
    torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    buckets={}
    for row in rr:
        grid=[1.] if method=='D0' else grid_for(row['start_id'],u,c,force_K)
        buckets.setdefault(len(grid),[]).append((row,grid))
    total_c=total_r=0.;nfes=0;diagnostics=[]
    for K,items in sorted(buckets.items()):
        for begin in range(0,len(items),micro):
            part=items[begin:begin+micro];C,Y=store.batch([x[0] for x in part],next(model.parameters()).device)
            trace=(u%c['diagnostic_interval']==0)
            r=objective(model,sim,C,Y,[x[1] for x in part],trace=trace);(r['loss']*(len(part)/len(rr))).backward()
            if trace:diagnostics.append({'K':K,'start_ids':[x[0]['start_id'] for x in part],'ranges':r['ranges']})
            total_c+=float(r['lc'].detach().sum());total_r+=float(r['lr'].detach().sum());nfes+=K*len(part)
            del r,C,Y
    assert all(p.grad is not None for p in model.parameters()),'Missing parameter gradient'
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
    rate=c['peak_lr'] if peak else lr_at(u,c)
    for g in opt.param_groups:g['lr']=rate
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite parameter'
    if ema is not None:
        with torch.no_grad():
            for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.004)
            for p,q in zip(ema.buffers(),model.buffers()):p.copy_(q)
    torch.cuda.synchronize();dt=time.monotonic()-start
    return {'update':u,'exposures':u*c['effective_batch'],'L_C':total_c/len(rr),'L_R':total_r/len(rr),'loss':(total_c+.2*total_r)/len(rr),'gradient_norm':float(norm),'LR':rate,'seconds':dt,'starts_per_second':len(rr)/dt,'NFE':nfes,'microbatch':micro,'accumulation':math.ceil(len(rr)/micro),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'nonfinite':0,'diagnostics':diagnostics}

def save_checkpoint(run,model,ema,opt,store,u,micro,method,identity):
    with (ROOT/'storage_commit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _save_checkpoint_locked(run,model,ema,opt,store,u,micro,method,identity)

def _save_checkpoint_locked(run,model,ema,opt,store,u,micro,method,identity):
    import torch
    run=Path(run);storage_check(1_950_000_000);torch.cuda.synchronize()
    path=run/'checkpoints'/f'full_U{u:05d}.pt';payload={'purpose':PURPOSE,'method':method,'update':u,'exposures':u*store.c['effective_batch'],'model':tree_cpu(model.state_dict()),'ema':tree_cpu(ema.state_dict()),'optimizer':tree_cpu(opt.state_dict()),'rng':rng_state(),'sampler':store.stream_state(u),'microbatch':micro,'identity':identity,'next_lr':None if u==store.c['max_updates'] else lr_at(u+1,store.c)}
    tensor_save(path,payload);del payload
    receipt={'path':str(path),'sha256':sha(path),'update':u,'exposures':u*store.c['effective_batch'],'identity':identity,'method':method}
    atomic(path.with_suffix('.identity.json'),receipt);atomic(run/'LATEST.json',receipt)
    # Delete only committed rolling states generated by this new run after the
    # replacement has a durable tensor and verified sidecar. Parent is protected.
    old=sorted((run/'checkpoints').glob('full_U*.pt'))[:-2]
    for p in old:
        side=p.with_suffix('.identity.json');d=read(side)
        assert d['identity']==identity and d['method']==method and p.parent.resolve()==(run/'checkpoints').resolve()
        append(run/'retention.jsonl',{'deleted':str(p),'sha256':d['sha256'],'replaced_by':receipt,'time':time.time()});p.unlink();side.unlink()
    return receipt
def restore_checkpoint(receipt,model,ema,opt,store,identity,method):
    import torch
    path=Path(receipt['path']);assert sha(path)==receipt['sha256']
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['purpose']==PURPOSE and p['identity']==identity and p['method']==method
    u=p['update'];assert p['sampler']==store.stream_state(u) and p['exposures']==u*store.c['effective_batch']
    model.load_state_dict(p['model'],strict=True);ema.load_state_dict(p['ema'],strict=True);opt.load_state_dict(p['optimizer'])
    assert (not opt.state if u==0 else opt.state and all(int(s['step'])==u for s in opt.state.values()))
    rng_restore(p['rng']);return u,p['microbatch']
def save_milestone(run,model,u,method,identity):
    with (ROOT/'storage_commit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _save_milestone_locked(run,model,u,method,identity)

def _save_milestone_locked(run,model,u,method,identity):
    p=Path(run)/'milestones'/f'online_U{u:05d}.pt';storage_check(512<<20)
    if not p.exists():tensor_save(p,{'purpose':PURPOSE+'_ONLINE','update':u,'method':method,'identity':identity,'model':tree_cpu(model.state_dict())})
    d={'path':str(p),'sha256':sha(p),'update':u,'method':method,'identity':identity};atomic(p.with_suffix('.identity.json'),d)
    common=set(store_u for store_u in config()['milestones'])
    for group in GROUPS:
        common&={int(q.stem.split('_U')[1]) for q in (ROOT/'runs'/group/'milestones').glob('online_U*.pt')}
    if common:
        latest_common=max(common)
        for group in GROUPS:
            directory=ROOT/'runs'/group/'milestones'
            for q in directory.glob('online_U*.pt'):
                if int(q.stem.split('_U')[1])>=latest_common:continue
                side=q.with_suffix('.identity.json');old=read(side)
                assert old['identity']==identity and old['method']==group and q.parent.resolve()==directory.resolve()
                append(ROOT/'milestone_retention.jsonl',{'deleted':str(q),'sha256':old['sha256'],'replaced_by_common_update':latest_common,'time':time.time()})
                q.unlink();side.unlink()
    return d

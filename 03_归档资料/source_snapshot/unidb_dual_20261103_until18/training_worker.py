"""Independent 3+5 UniDB extension workers. Parent campaigns are read-only.

Original: eight logical B12 streams on three physical ranks; equal DDP slots,
with an RNG-isolated zero-loss dummy on the rank owning only two streams.
Improved: physical local B=[20,19,19,19,19], sample-weighted to global B96.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import timedelta
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT.parent / 'unidb_d0_20261102'
for key, rel in [('XDG_CACHE_HOME','cache'),('ASCEND_CACHE_PATH','cache/ascend'),('PIP_CACHE_DIR','cache/pip')]:
    os.environ[key] = str(ROOT / rel)
sys.path.insert(0, str(OLD_ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from core import (NativeObjective, configure_npu, cpu_tree, derive, ema_restore,
    ema_save, initialize, make_ema, parameter_digest, rng_capture, rng_restore,
    sha, sources_ok, write)
from data import Order, load_pool
sys.path.insert(0, str(ROOT))

FORMAT = 'UNIDB_DUAL_FULL_STATE_V1'
BATCH = 96
SEED = 843101
SIGNAL_STOP = False
TRAINING_DEADLINE = 1793700000
SCHEDULE_DEADLINE = 1793678400
EXTENSION_KEYS = ('extension_parent', 'extension_parent_cpu_audit',
    'extension_start_update', 'extension_start_cursor', 'extension_start_counts_by_g',
    'extension_start_unix', 'prior_scheduler', 'previous_monitor_done')


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def identity():
    return {'pid':os.getpid(), 'start_ticks':Path('/proc/self/stat').read_text().split()[21]}


def broadcast(value, rank):
    box = [value if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def source_checks():
    sources_ok()
    authority = ROOT/'protocol/USER_AUTHORITY.json'
    assert authority.is_file()
    manifests = [p for p in [ROOT/'protocol/RUN_SOURCE.json', ROOT/'protocol/SOURCE_MANIFEST.json'] if p.is_file()]
    assert manifests, 'New run source manifest must be frozen before gate/formal.'
    covered = set()
    result = {}
    for manifest in manifests:
        files = json.loads(manifest.read_text())['files']
        for name, record in files.items():
            expected = record.get('sha256') if isinstance(record,dict) else record
            path = (ROOT/name).resolve()
            assert path.is_relative_to(ROOT), name
            assert sha(path) == expected, ('new source changed', name)
            covered.add(str(path.relative_to(ROOT)))
        result[str(manifest.relative_to(ROOT))] = sha(manifest)
    assert 'training_worker.py' in covered, 'training_worker.py absent from source manifest'
    return {'original_sources_checked':True, 'new_manifests':result, 'authority_sha256':sha(authority)}


def rng_digest(state):
    h = hashlib.sha256()
    for key in ['torch','npu']:
        h.update(key.encode()); h.update(state[key].cpu().contiguous().numpy().tobytes())
    n = state['numpy']
    h.update(str(n[0]).encode()); h.update(n[1].tobytes()); h.update(repr(n[2:]).encode())
    h.update(repr(state['python']).encode())
    return h.hexdigest()


def capture(device):
    return copy.deepcopy(rng_capture(device))


def layout_for_rank(branch, rank, world):
    if branch == 'original':
        lids = list(range(rank,8,world))
        return [{'lid':lid, 'offset':lid*12, 'n':12} for lid in lids]
    assert world == 5
    sizes = [20,19,19,19,19]
    offset = sum(sizes[:rank])
    return [{'lid':None,'offset':offset,'n':10},
            {'lid':None,'offset':offset+10,'n':sizes[rank]-10}]


def positions_for_layout(cursor, branch, rank, world):
    return [[cursor+x['offset']+k for k in range(x['n'])]
            for x in layout_for_rank(branch,rank,world)]


class GlobalPositions(Dataset):
    def __init__(self, pool):
        self.pool=pool; self.order=Order(pool, SEED)
    def __len__(self):return 10**12
    def __getitem__(self, position):
        index=self.order.at(int(position)); row=self.pool[index]
        y=np.load(row['observation_path'],allow_pickle=False)
        c=np.load(row['clean_path'],allow_pickle=False)
        assert y.shape==c.shape==(3,256,256) and y.dtype==c.dtype==np.float32
        return torch.from_numpy(y),torch.from_numpy(c),index,int(position),int(row['g'])


class PositionBatches:
    def __init__(self,cursor,branch,rank,world):
        self.cursor=cursor;self.branch=branch;self.rank=rank;self.world=world
    def __iter__(self):
        cursor=self.cursor
        while True:
            yield from positions_for_layout(cursor,self.branch,self.rank,self.world)
            cursor += BATCH


class SyncObjective(torch.nn.Module):
    """Dummy is an engineering synchronization operation, never a data update."""
    def __init__(self,inner,dummy_mode='parameters'):
        super().__init__();self.inner=inner;self.dummy_mode=dummy_mode
    def forward(self,degraded=None,clean=None,dummy=False):
        if dummy and self.dummy_mode=='parameters':
            zero=sum(p.reshape(-1)[0]*0. for p in self.inner.network.parameters())
            return zero,torch.empty(0,dtype=torch.long,device=zero.device)
        loss,tau=self.inner(degraded,clean)
        return (loss*0. if dummy else loss),tau


def full_consensus(net, rank, label, out):
    digest=parameter_digest(net,full=True);all_digests=[None]*dist.get_world_size()
    dist.all_gather_object(all_digests,digest)
    assert len(set(all_digests))==1, ('full parameter divergence',label,all_digests)
    if rank==0:
        target=out/f'FULL_SHA_{label}.json'
        if target.exists():target=out/f'FULL_SHA_{label}_{int(time.time())}_P{os.getpid()}.json'
        write(target,{'status':'PASS','parameter_digest_full':digest,
            'all_rank_full_digests':all_digests,'sampled':False,'verified_unix':time.time()})
    return digest


def install_gate_state_trace(objective, phase):
    if phase!='gate':return None
    original=objective.sde.generate_random_states;trace={}
    def measured(x0,mu):
        tau,state=original(x0,mu)
        trace.clear();trace.update(tau=tau.reshape(-1).tolist(),
            generated_state_sha256=hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        return tau,state
    objective.sde.generate_random_states=measured
    return trace


def extension_progress(state):
    return {'extension_start_update':state['extension_start_update'],
            'extension_new_updates':state['update']-state['extension_start_update'],
            'extension_start_cursor':state['extension_start_cursor'],
            'extension_new_presentations':state['cursor']-state['extension_start_cursor']}


def validate_payload_structure(payload, branch, extension=True, authority=None):
    assert payload['format']==FORMAT and payload['branch']==branch
    assert payload['cursor']==payload['data_positions']['next_global_position']
    assert payload['scheduler']['deadline_unix']==(TRAINING_DEADLINE if extension else SCHEDULE_DEADLINE)
    assert payload['scaler'] is None
    assert len(payload['rng_by_logical_rank'] if branch=='original' else payload['rng_by_rank'])==(8 if branch=='original' else 5)
    steps={int(float(x['step'])) for x in payload['optimizer']['state'].values()}
    expected=payload['update']
    assert steps==({expected} if expected else set()), (steps,expected)
    assert int(payload['ema']['step'])==expected
    assert payload['initial_update']==(45411 if branch=='original' else 0)
    assert payload['new_updates']==expected-payload['initial_update']
    assert len(payload['loader_generator_states'])==payload['physical_world_size']
    assert all(x is not None for x in payload['loader_generator_states'])
    if extension:
        scheduler=payload['scheduler']
        assert scheduler['training_stop_deadline_unix']==TRAINING_DEADLINE
        assert scheduler['schedule_deadline_unix']==SCHEDULE_DEADLINE
        assert scheduler['extension_lr']==(1e-5 if branch=='original' else 2e-7)
        assert scheduler['type']==('constant' if branch=='original' else 'extension_constant_floor')
        for key in EXTENSION_KEYS:assert key in payload, key
        parent=payload['extension_parent']
        assert payload['extension_start_update']==parent['update']
        assert payload['extension_start_cursor']==parent['cursor']
        assert payload['extension_new_updates']==expected-parent['update']>=0
        assert payload['extension_new_presentations']==payload['cursor']-parent['cursor']==96*payload['extension_new_updates']
        prior=payload['prior_scheduler']
        assert prior['deadline_unix']==SCHEDULE_DEADLINE
        for key in ['start_unix','warmup_end_unix']:assert scheduler[key]==prior[key], key
        assert math.isfinite(float(payload['extension_start_unix']))
        for g,n in payload['extension_start_counts_by_g'].items():
            assert payload['counts_by_g'].get(g,0)>=n
        assert sum(payload['counts_by_g'].values())-sum(payload['extension_start_counts_by_g'].values())==payload['extension_new_presentations']
        if payload['extension_new_updates']:
            assert all(float(g['lr'])==scheduler['extension_lr'] for g in payload['optimizer']['param_groups'])
        if authority is not None:assert parent==authority['extension_parents'][branch]
    return True


def approved_extension_parent(path, actual, payload, branch, authority):
    """Only the explicitly approved, independently CPU-read-back noon state."""
    approved=authority['extension_parents'][branch]
    assert path==Path(approved['path']).resolve() and actual==approved['sha256']
    assert path.is_relative_to(Path(authority['previous_campaign_root']).resolve())
    assert payload['update']==approved['update'] and payload['cursor']==approved['cursor']
    assert payload['phase']=='formal', 'Extension cannot start from a discarded gate.'
    audit_path=Path(approved.get('cpu_audit_path',str(ROOT/'evidence'/f'PARENT_STATE_AUDIT_{branch}.json'))).resolve()
    assert audit_path.is_relative_to(ROOT) and audit_path.is_file(), audit_path
    audit=json.loads(audit_path.read_text());assert audit['status']=='PASS'
    checked=audit['checkpoint']
    assert Path(checked['path']).resolve()==path and checked['sha256']==actual
    assert checked['update']==payload['update'] and checked['cursor']==payload['cursor']
    return copy.deepcopy(approved), {'path':str(audit_path),'sha256':sha(audit_path)}


def save_checkpoint(out,state,net,optimizer,ema,device,rank,world,cfg,banks,loader_generator,
                    label=None,check_full=False):
    source_id=source_checks();before=capture(device)
    local={'rng':copy.deepcopy(banks) if cfg['branch']=='original' else before,
           'loader_generator':loader_generator.get_state().clone()}
    gathered=[None]*world;dist.all_gather_object(gathered,local)
    full=full_consensus(net,rank,label or f"U{state['update']:08d}",out) if check_full else None
    result=None
    if rank==0:
        folder=out/'checkpoints';folder.mkdir(exist_ok=True)
        path=folder/f"U{state['update']:08d}.pt"
        logical=None
        if cfg['branch']=='original':
            bank={int(k):v for r in gathered for k,v in r['rng'].items()}
            assert sorted(bank)==list(range(8))
            logical=[bank[j] for j in range(8)]
        scheduler={k:state[k] for k in ['start_unix','deadline_unix','warmup_end_unix','schedule_deadline_unix']}
        scheduler.update(training_stop_deadline_unix=state['deadline_unix'],
                         extension_lr=1e-5 if cfg['branch']=='original' else 2e-7)
        scheduler.update(type='constant' if cfg['branch']=='original' else 'extension_constant_floor',
                         completed_updates=state['update'],next_update=state['update']+1,
                         current_lr=float(optimizer.param_groups[0]['lr']))
        if cfg['branch']=='improved':
            from improved_objective import SCHEDULE_SPEC
            scheduler['schedule_spec']=SCHEDULE_SPEC
        payload={'format':FORMAT,'branch':cfg['branch'],'phase':cfg['phase'],'update':state['update'],
            'initial_update':state['initial_update'],'new_updates':state['update']-state['initial_update'],
            'cursor':state['cursor'],'counts_by_g':dict(state['counts']),
            'network':cpu_tree(net.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'ema':ema_save(ema),
            'rng_by_logical_rank':logical,'rng_by_rank':[r['rng'] for r in gathered] if logical is None else None,
            'logical_world_size':8 if logical is not None else None,'physical_world_size':world,
            'loader_generator_states':[r['loader_generator'] for r in gathered],
            'data_positions':{'next_global_position':state['cursor'],'seed':SEED,'effective_batch':96,
                'scheme':'logical lid*12' if logical is not None else 'local20_19_19_19_19',
                'prefetch':'Explicit positions; unconsumed prefetched deterministic arrays are reconstructed from cursor.'},
            'scheduler':scheduler,'scaler':None,'config':cfg,'source_identity':source_id,
            'parent':state['parent'],'monitor_done':sorted(state['monitor_done']),
            'full_parameter_digest':full,'last_completed_unix':state.get('last_completed_unix'),
            **{k:copy.deepcopy(state[k]) for k in EXTENSION_KEYS},**extension_progress(state)}
        validate_payload_structure(payload,cfg['branch'])
        if not path.exists():
            temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
            result={'status':'PASS','path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
                'update':state['update'],'cursor':state['cursor'],'branch':cfg['branch'],
                'phase':cfg['phase'],'full_state':True,'full_parameter_digest':full}
            write(path.with_suffix('.json'),result)
        else:
            result=json.loads(path.with_suffix('.json').read_text())
            assert result['update']==state['update'] and result['cursor']==state['cursor']
        write(out/'LATEST.json',result)
    result=broadcast(result,rank);dist.barrier();rng_restore(before,device)
    return result


def load_resume(path,net,optimizer,ema,device,rank,world,cfg,authority):
    path=Path(path).resolve();assert path.is_file()
    branch=cfg['branch'];inside=path.is_relative_to(ROOT)
    if not inside:
        assert path==Path(authority['extension_parents'][branch]['path']).resolve(), 'Unapproved external state.'
    actual=sha(path)
    if not inside:assert actual==authority['extension_parents'][branch]['sha256']
    payload=torch.load(path,map_location='cpu',weights_only=False)
    validate_payload_structure(payload,branch,extension=inside,authority=authority if inside else None)
    assert payload['physical_world_size']==world and payload['config']['physical_world_size']==world
    assert not (cfg['phase']=='formal' and payload['phase']=='gate'), 'Formal must not inherit gate updates.'
    side=json.loads(path.with_suffix('.json').read_text())
    assert actual==side['sha256'] and side['full_state'] is True
    for k in ['branch','effective_batch','data_seed','precision','objective','gamma_inv','T']:
        assert payload['config'][k]==cfg[k],(k,payload['config'][k],cfg[k])
    banks={}
    if branch=='original':
        banks={lid:copy.deepcopy(payload['rng_by_logical_rank'][lid]) for lid in range(rank,8,world)}
        # Restore ambient method state too; every real micro still uses its exact logical bank.
        rng_restore(banks[min(banks)],device)
    else:rng_restore(payload['rng_by_rank'][rank],device)
    loader_state=payload['loader_generator_states'][rank]
    state={'update':payload['update'],'initial_update':payload['initial_update'],'cursor':payload['cursor'],
        'counts':Counter(payload['counts_by_g']),'parent':copy.deepcopy(payload['parent']),
        'last_completed_unix':payload.get('last_completed_unix'),
        'start_unix':payload['scheduler']['start_unix'],
        'warmup_end_unix':payload['scheduler']['warmup_end_unix'],
        'deadline_unix':authority['deadline_unix'],'schedule_deadline_unix':authority['previous_deadline_unix']}
    if inside:
        # Own recovery retains the original extension base/history; never re-anchor it.
        state.update({k:copy.deepcopy(payload[k]) for k in EXTENSION_KEYS})
        state['monitor_done']=set(payload.get('monitor_done',[]))
    else:
        parent,audit=approved_extension_parent(path,actual,payload,branch,authority)
        state.update(extension_parent=parent,extension_parent_cpu_audit=audit,
            extension_start_update=payload['update'],extension_start_cursor=payload['cursor'],
            extension_start_counts_by_g=copy.deepcopy(payload['counts_by_g']),extension_start_unix=None,
            prior_scheduler=copy.deepcopy(payload['scheduler']),
            previous_monitor_done=copy.deepcopy(payload.get('monitor_done',[])),monitor_done=set())
    net.load_state_dict(payload['network'],strict=True);optimizer.load_state_dict(payload['optimizer']);ema_restore(ema,payload['ema'])
    assert all(float(g['lr'])==1e-5 for g in optimizer.param_groups) if branch=='original' else True
    assert all(math.isfinite(float(g['lr'])) and float(g['lr'])>0 for g in optimizer.param_groups)
    steps={int(float(x['step'])) for x in optimizer.state.values()}
    assert steps==({state['update']} if state['update'] else set())
    assert int(ema.step)==state['update']
    del payload
    return state,banks,loader_state


def evaluate_isolated(net,device,out,rank,world,full,update,weights,branch):
    before=capture(device);was_training=net.training
    try:
        target=out/'validation'/f'U{update:08d}_{weights.lower()}_{"full" if full else "monitor"}'
        # Delegate all completed/partial reuse to validation's current full-model
        # digest and input/clean identity checks; no wrapper SUMMARY shortcut.
        from validation import evaluate
        return evaluate(net,device,target,rank,world,full,update,weights=weights,branch=branch)
    finally:
        rng_restore(before,device);net.train(was_training)
        assert rng_digest(capture(device))==rng_digest(before), 'Evaluation consumed training RNG.'


def run_evals(state,cfg,net,ema,device,out,rank,world,full=False,raw=False,terminal=False):
    summary=None
    for weights,model in [('EMA',ema.ema_model)]+([('RAW',net)] if raw else []):
        if not terminal and broadcast(time.time()>=state['deadline_unix'] if rank==0 else None,rank):break
        use_full=full if weights=='EMA' else False
        key=f"{state['update']}:{weights}:{int(use_full)}"
        if key in state['monitor_done']:continue
        if rank==0:
            write(out/'STATUS.json',{'status':'EVALUATING','branch':cfg['branch'],'update':state['update'],
                'new_updates':state['update']-state['initial_update'],'weights':weights,'full':use_full,
                'evaluation_started_unix':time.time(),'deadline_unix':state['deadline_unix'],
                'terminal_evaluation':terminal})
        summary=evaluate_isolated(model,device,out,rank,world,use_full,state['update'],weights,cfg['branch'])
        state['monitor_done'].add(key)
        if rank==0:
            append(out/'VALIDATIONS.jsonl',{'update':state['update'],'new_updates':state['update']-state['initial_update'],**extension_progress(state),
                'weights':weights,'full':use_full,'summary':summary,'completed_unix':time.time()})
            write(out/'EVALUATION_STATE.json',{'done':sorted(state['monitor_done'])})
            if weights=='EMA':
                side=out/'checkpoints'/f"U{state['update']:08d}.json"
                assert side.is_file(), 'Every selectable validation must bind a complete checkpoint.'
                identity=json.loads(side.read_text());metrics=summary['by_g']['4']
                score=(float(metrics['psnr']),-float(metrics['lpips']),-state['update'])
                target=out/('BEST_FULL.json' if use_full else 'BEST_MONITOR.json')
                prior=json.loads(target.read_text()) if target.exists() else None
                if prior is None or score>tuple(prior['selection_score']):
                    best={'status':'PASS','checkpoint':identity,'update':state['update'],
                          'weights':'EMA','selection_score':list(score),'full':use_full,
                          'selection_role':'full_validation' if use_full else 'monitor_only',
                          'summary':summary}
                    write(target,best)
                    if use_full:write(out/'BEST.json',best)
    return summary


def stop_decision(state,cfg,device,rank,gate_completed,steps):
    flag=torch.tensor(int(SIGNAL_STOP),device=device,dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    reason=None
    if rank==0:
        if time.time()>=state['deadline_unix']:reason='deadline'
        elif (ROOT/'STOP_REQUEST.json').exists():reason='stop_request'
        elif int(flag.item()):reason='signal_request'
        elif cfg['phase']=='gate' and gate_completed>=steps:reason='gate_steps'
    return broadcast(reason,rank)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['original','improved'],required=True)
    parser.add_argument('--phase',choices=['gate','formal'],required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--steps',type=int,default=2);parser.add_argument('--resume');parser.add_argument('--monitor-every',type=int,default=1000)
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--dummy-mode',choices=['native','parameters'],default='parameters');args=parser.parse_args()
    assert not args.eval_only or args.resume, '--eval-only requires explicit --resume and executes zero updates.'
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT) and out!=ROOT
    out.mkdir(parents=True,exist_ok=True);write(out/f'RANK_{rank}.json',{**identity(),'rank':rank,'local_rank':local,'world':world,'branch':args.branch,'phase':args.phase})
    loader=None;iterator=None;initialized=False;exit_status='FAILED';started=time.time()
    try:
        authority=json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text());assert authority['deadline_unix']==TRAINING_DEADLINE
        assert authority['previous_deadline_unix']==SCHEDULE_DEADLINE
        assert authority['extension_lrs']=={'original':1e-5,'improved':2e-7}
        assert world==(3 if args.branch=='original' else 5)
        assert args.steps>=1 and args.monitor_every>=1
        source_id=source_checks();device=configure_npu(local);dist.init_process_group('hccl',timeout=timedelta(hours=2));initialized=True
        def requested(signum,frame):
            global SIGNAL_STOP
            SIGNAL_STOP=True
        signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
        net,strict=initialize(device)
        objective=NativeObjective(net,device) if args.branch=='original' else __import__('improved_objective').ImprovedObjective(net,device)
        objective.train();optimizer=torch.optim.Adam(net.parameters(),lr=1e-5 if args.branch=='original' else 2e-7,betas=(.9,.99),eps=1e-8,weight_decay=0.)
        ema=make_ema(net,device)
        cfg={'branch':args.branch,'phase':args.phase,'physical_world_size':world,'logical_world_size':8 if args.branch=='original' else None,
             'effective_batch':96,'data_seed':SEED,'rng_seed':SEED,'precision':'FP32_HF32off',
             'objective':'native_mean_matching_L1' if args.branch=='original' else 'epsilon_opt_MSE',
             'gamma_inv':1e-7,'T':100,'dummy_mode':args.dummy_mode if args.branch=='original' else None,
             'physical_micro_layout':[layout_for_rank(args.branch,r,world) for r in range(world)],
             'gradient_clip':None if args.branch=='original' else 1.,'Adam_betas':[.9,.99],'Adam_eps':1e-8,'weight_decay':0.,
             'EMA_beta':.995,'EMA_update_every':10,'deadline_unix':authority['deadline_unix'],
             'schedule_deadline_unix':authority['previous_deadline_unix'],
             'extension_lr':authority['extension_lrs'][args.branch],'startup_monitor':False,'source_identity':source_id}
        # Both routes are complete-state continuations: no fresh seed/optimizer path.
        resume=args.resume or authority['extension_parents'][args.branch]['path']
        state,banks,loader_state=load_resume(resume,net,optimizer,ema,device,rank,world,cfg,authority)
        assert state['start_unix'] is not None
        state['extension_start_unix']=broadcast(state['extension_start_unix'] or time.time() if rank==0 else None,rank)
        # Gates never inherit unfinished log records; formal recovery keeps original records as evidence.
        log=out/f'UPDATES_rank{rank}.jsonl'
        if log.exists():
            rows=[json.loads(s) for s in log.read_text().splitlines()];tail=[r for r in rows if r['update']>state['update']]
            if tail:
                write(out/f'UNCOMMITTED_RECOVERY_rank{rank}_{int(time.time())}.json',{'records':tail})
                log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['update']<=state['update']))
        wrapper=SyncObjective(objective,args.dummy_mode)
        ddp=DDP(wrapper,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
        full_consensus(net,rank,'INITIAL',out)
        if args.eval_only:
            assert args.phase=='formal', '--eval-only is final-validation recovery, not a gate.'
            if rank==0:
                write(out/'CONFIG_EVAL_ONLY.json',cfg)
                write(out/'EVAL_ONLY_STARTED.json',{'resume':str(Path(args.resume).resolve()),
                    'update':state['update'],'cursor':state['cursor'],'optimizer_steps_allowed':0,'unix':time.time()})
            run_evals(state,cfg,net,ema,device,out,rank,world,full=True,raw=True,terminal=True)
            final=json.loads(Path(args.resume).with_suffix('.json').read_text())
            result={'status':'PASS','branch':args.branch,'phase':'eval_only','rank':rank,'world':world,
                'global_update':state['update'],'cursor':state['cursor'],'completed_updates_this_process':0,
                'new_updates_total':state['update']-state['initial_update'],**extension_progress(state),'final_checkpoint':final,
                'new_optimizer_updates':0,'seconds':time.time()-started,'stop_reason':'evaluation_only'}
            write(out/f'EVAL_ONLY_RESULT_rank{rank}.json',result);dist.barrier()
            if rank==0:
                write(out/'EVAL_ONLY_RESULT.json',{'status':'PASS','ranks':[
                    json.loads((out/f'EVAL_ONLY_RESULT_rank{r}.json').read_text()) for r in range(world)],
                    'new_optimizer_updates':0,'final_checkpoint':final})
                write(out/'STATUS.json',{'status':'COMPLETE','update':state['update'],
                    'new_updates':state['update']-state['initial_update'],'new_optimizer_updates_this_process':0,
                    'final_full_validation_done':True,'evaluation_only':True,'completed_unix':time.time()})
            exit_status='PASS'
            return
        pool,data_gate=load_pool();dataset=GlobalPositions(pool)
        generator=torch.Generator()
        assert loader_state is not None, 'Full continuation requires the saved loader generator.'
        generator.set_state(loader_state)
        loader=DataLoader(dataset,batch_sampler=PositionBatches(state['cursor'],args.branch,rank,world),
            num_workers=2,prefetch_factor=2,persistent_workers=True,multiprocessing_context='spawn',generator=generator)
        before=capture(device);iterator=iter(loader);rng_restore(before,device)
        trace=install_gate_state_trace(objective,args.phase)
        if rank==0:
            write(out/'CONFIG.json',cfg);write(out/'STRICT_LOAD.json',strict)
            write(out/'RESUME_IDENTITY.json',{'resume':str(resume) if resume else None,'state_update':state['update'],'cursor':state['cursor'],'parent':state['parent'],'extension_parent':state['extension_parent'],
                'extension_parent_cpu_audit':state['extension_parent_cpu_audit'],**extension_progress(state)})
        torch.npu.reset_peak_memory_stats();dist.barrier();initial_update=state['update'];gate_completed=0
        if args.phase=='formal':
            save_checkpoint(out,state,net,optimizer,ema,device,rank,world,cfg,banks,generator,label='START',check_full=True)
            # The existing parent monitors are retained as history; no startup monitor.
        reason=None
        while True:
            reason=stop_decision(state,cfg,device,rank,gate_completed,args.steps)
            if reason is not None:break
            begin=time.time();next_update=state['update']+1
            # Extension policy is constant LR; the old noon cosine anchors remain metadata.
            lr=float(broadcast(authority['extension_lrs'][args.branch] if rank==0 else None,rank))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=device);gcounts=torch.zeros(4,dtype=torch.int64,device=device)
            layout=layout_for_rank(args.branch,rank,world);slots=math.ceil(8/world) if args.branch=='original' else 2
            records=[];positions=[];indices=[];dummy_batch=None;real_calls=0;dummy_calls=0
            calls=[0];hook=net.register_forward_hook(lambda m,a,v:calls.__setitem__(0,calls[0]+1))
            for slot in range(slots):
                real=slot<len(layout)
                if real:
                    item=layout[slot];y,c,index,position,g=next(iterator);dummy_batch=(y,c)
                    expected=[state['cursor']+item['offset']+k for k in range(item['n'])];assert position.tolist()==expected
                    positions+=expected;indices+=index.tolist();local_g=Counter(g.tolist())
                    for value,n in local_g.items():gcounts[int(value)-1]+=n
                    if args.branch=='original':rng_restore(banks[item['lid']],device)
                    before_rng=capture(device) if args.branch=='original' or args.phase=='gate' else None
                    y=y.to(device);c=c.to(device)
                    weight=world/8 if args.branch=='original' else item['n']*world/96
                else:
                    assert args.branch=='original' and dummy_batch is not None
                    before_rng=capture(device);y,c=dummy_batch;y=y.to(device);c=c.to(device);weight=1.
                context=ddp.no_sync() if slot+1<slots else contextlib.nullcontext()
                with context:
                    loss,tau=ddp(y,c,dummy=not real);assert bool(torch.isfinite(loss))
                    (loss*weight).backward()
                if real:
                    real_calls+=1;loss_sum+=loss.detach()*weight
                    tt=tau.detach().cpu().reshape(-1).tolist();assert len(tt)==item['n'] and all(1<=t<=99 for t in tt)
                    entry={'logical_rank':item['lid'],'n':item['n'],'positions':expected,'indices':index.tolist(),'tau':tt,'loss':float(loss.detach()),'gradient_scale_before_DDP_mean':weight}
                    if args.branch=='original':
                        torch.npu.synchronize();banks[item['lid']]=capture(device)
                        entry.update(rng_before_sha256=rng_digest(before_rng),rng_after_sha256=rng_digest(banks[item['lid']]))
                    if trace is not None:entry.update(generated_state_sha256=trace['generated_state_sha256'])
                    records.append(entry)
                else:
                    dummy_calls+=1;assert float(loss.detach())==0.
                    torch.npu.synchronize();rng_restore(before_rng,device)
                    assert rng_digest(capture(device))==rng_digest(before_rng),'Dummy consumed method RNG.'
                del y,c,loss,tau
            hook.remove();expected_calls=real_calls+(dummy_calls if args.dummy_mode=='native' else 0)
            assert calls[0]==expected_calls,(calls[0],expected_calls)
            gradients=[p.grad for p in net.parameters()];assert all(g is not None for g in gradients)
            pre=torch.nn.utils.get_total_norm(gradients,error_if_nonfinite=True);assert bool(pre>0)
            if args.branch=='improved':
                torch.nn.utils.clip_grad_norm_(list(net.parameters()),1.,error_if_nonfinite=True)
                post=torch.nn.utils.get_total_norm([p.grad for p in net.parameters()],error_if_nonfinite=True)
                assert float(post)<=1.00001
            else:post=pre
            optimizer.step();ema.update();state['update']+=1;state['cursor']+=96;gate_completed+=1
            dist.all_reduce(loss_sum);loss_sum/=world;dist.all_reduce(gcounts)
            for g,n in enumerate(gcounts.cpu().tolist(),1):state['counts'][str(g)]+=n
            assert int(gcounts.sum())==96
            torch.npu.synchronize();state['last_completed_unix']=broadcast(time.time() if rank==0 else None,rank)
            new=state['update']-state['initial_update']
            entry={'update':state['update'],'new_updates':new,**extension_progress(state),'cursor':state['cursor'],'branch':args.branch,'rank':rank,
                'started_unix':begin,'completed_unix':state['last_completed_unix'],'seconds':time.time()-begin,
                'loss':float(loss_sum),'lr':lr,'grad_norm':float(pre),'grad_norm_after_clip':float(post),
                'positions':positions,'indices':indices,'microbatches':records,'real_microbatches':real_calls,'dummy_microbatches':dummy_calls,
                'real_presentations_local':len(positions),'nfe_local_including_sync_dummy':calls[0],
                'counts_by_g':dict(state['counts']),'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved()}
            if args.branch=='improved':
                diag=getattr(objective,'last_diagnostics',{})
                entry['objective_diagnostics']={k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in diag.items()}
            append(log,entry)
            if rank==0:
                write(out/'STATUS.json',{'status':'TRAINING' if args.phase=='formal' else 'GATE','update':state['update'],
                    'new_updates':new,**extension_progress(state),'presentations_total':state['cursor'],'new_presentations':new*96,
                    'last_completed_unix':state['last_completed_unix'],'lr':lr,'loss':float(loss_sum),
                    'counts_by_g':dict(state['counts']),'deadline_unix':state['deadline_unix']})
                if new<=10 or new%10==0:print(args.branch,args.phase,'U',state['update'],'new',new,'loss',float(loss_sum),'seconds',entry['seconds'],flush=True)
            if args.phase=='formal':
                full=new>0 and new%10000==0
                monitor=new in [50,100,250,500] or (new>=args.monitor_every and new%args.monitor_every==0)
                if new%250==0 or monitor or full:
                    save_checkpoint(out,state,net,optimizer,ema,device,rank,world,cfg,banks,generator)
                # Recheck wall-clock/STOP after every update and before optional validation.
                if (full or monitor) and stop_decision(state,cfg,device,rank,gate_completed,args.steps) is None:
                    run_evals(state,cfg,net,ema,device,out,rank,world,full=full,raw=new in [500,2000])
        stopped={'update':state['update'],'new_updates':state['update']-state['initial_update'],**extension_progress(state),'cursor':state['cursor'],
                 'reason':reason,'detected_unix':time.time(),'deadline_unix':state['deadline_unix']}
        if rank==0:write(out/'TRAINING_STOPPED.json',stopped)
        final=save_checkpoint(out,state,net,optimizer,ema,device,rank,world,cfg,banks,generator,label='FINAL',check_full=True)
        if args.phase=='formal':run_evals(state,cfg,net,ema,device,out,rank,world,full=True,raw=True,terminal=True)
        result={'status':'PASS','branch':args.branch,'phase':args.phase,'rank':rank,'world':world,
                'start_update_this_process':initial_update,'global_update':state['update'],'completed_updates_this_process':gate_completed,
                'new_updates_total':state['update']-state['initial_update'],**extension_progress(state),'cursor':state['cursor'],'final_checkpoint':final,
                'stop_reason':reason,'counts_by_g':dict(state['counts']),'seconds':time.time()-started,
                'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved()}
        if args.phase=='gate':assert gate_completed==args.steps,('Gate stopped before authorized test updates',reason,gate_completed)
        write(out/f'RESULT_rank{rank}.json',result);dist.barrier()
        if rank==0:
            all_results=[json.loads((out/f'RESULT_rank{r}.json').read_text()) for r in range(world)]
            write(out/'RESULT.json',{'status':'PASS','ranks':all_results,'branch':args.branch,'phase':args.phase,'final_checkpoint':final})
            write(out/'STATUS.json',{'status':'COMPLETE','update':state['update'],'new_updates':state['update']-state['initial_update'],**extension_progress(state),
                'stop_reason':reason,'final_full_validation_done':args.phase=='formal','completed_unix':time.time()})
        exit_status='PASS'
    except BaseException as exc:
        write(out/f'FAILURE_rank{rank}.json',{'status':'FAIL','actor':identity(),'error':repr(exc),'traceback':traceback.format_exc(),'unix':time.time()})
        raise
    finally:
        if iterator is not None:
            shutdown=getattr(iterator,'_shutdown_workers',None)
            if shutdown is not None:
                try:shutdown()
                except Exception:pass
        if initialized:
            try:dist.destroy_process_group()
            except Exception:pass
        write(out/f'EXIT_rank{rank}.json',{'status':exit_status,'actor':identity(),'unix':time.time(),
            'loader_workers_shutdown_requested':iterator is not None,'original_assets_written':False})


if __name__=='__main__':main()

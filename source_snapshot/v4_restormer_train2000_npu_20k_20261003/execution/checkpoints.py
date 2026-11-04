"""Transactional complete-update checkpoints with identity-checked fallback."""
import json
import os
from pathlib import Path
import time
import torch
from .common import atomic_json, append_json, digest, lr_at, MILESTONES
from npu_runtime.backend import capture_rng, restore_rng, synchronize


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(payload, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        fd=os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()


def scheduler_state(update):
    return {'kind':'warmup500_cosine20k','completed_updates':update,
            'next_lr':lr_at(update+1) if update<20000 else None}


def save_full(run, group, model, ema, optimizer, scaler, update, binding, stream, runtime, purpose='V4_RESTORMER_TRAIN2000_FORMAL_20K'):
    run=Path(run); folder=run/'checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'full_U{update:05d}.pt'
    identity_path=path.with_suffix('.identity.json')
    synchronize(next(model.parameters()).device)
    payload={'purpose':purpose,'group':group,
             'model':cpu_tree(model.state_dict()),'ema':cpu_tree(ema.state_dict()),
             'optimizer':cpu_tree(optimizer.state_dict()),'scheduler':scheduler_state(update),
             'scaler':scaler.state_dict(),'rng':capture_rng(device=next(model.parameters()).device),
             'update':update,'exposures':update*32,'epoch':update/1000,
             'sampler':{'completed_updates':update,'stream_identity':stream.identity},
             'binding':binding,'runtime':runtime,'saved_at':time.time()}
    if path.exists() and identity_path.exists():
        identity=json.loads(identity_path.read_text())
        if digest(path)!=identity['sha256']: raise ValueError('Existing checkpoint corrupt')
        if identity['update']!=update or identity['binding']!=binding: raise ValueError('Checkpoint identity conflict')
    else:
        atomic_save(payload,path)
        identity={'path':str(path.relative_to(run)),'sha256':digest(path),'update':update,
                  'exposures':update*32,'binding':binding,'saved_at':time.time()}
        atomic_json(identity_path,identity)
    atomic_json(run/'LATEST.json',identity)
    # Pending evaluation uses independent immutable RAW weights, never this rolling file.
    rolling=sorted(p for p in folder.glob('full_U*.pt') if int(p.stem[6:]) not in MILESTONES)
    for old in rolling[:-2]:
        old.unlink(); old.with_suffix('.identity.json').unlink(missing_ok=True)
    return identity


def load_latest(run, group, model, ema, optimizer, scaler, binding, stream, purpose='V4_RESTORMER_TRAIN2000_FORMAL_20K'):
    run=Path(run); candidates=sorted((run/'checkpoints').glob('full_U*.identity.json'),reverse=True)
    errors=[]
    for sidecar in candidates:
        try:
            identity=json.loads(sidecar.read_text());path=run/identity['path']
            if digest(path)!=identity['sha256']: raise ValueError('SHA256 mismatch')
            p=torch.load(path,map_location='cpu',weights_only=False)
            u=p['update']
            if (p['purpose']!=purpose or p['group']!=group or
                not 0<=u<=20000 or p['exposures']!=32*u or p['binding']!=binding or
                p['sampler']!={'completed_updates':u,'stream_identity':stream.identity} or
                p['scheduler']!=scheduler_state(u)):
                raise ValueError('Checkpoint scientific identity/counters mismatch')
            for name,value in model.stats.state_dict().items():
                if not torch.equal(p['model']['stats.'+name],value.cpu()): raise ValueError('Stats changed')
            model.load_state_dict(p['model'],strict=True);ema.load_state_dict(p['ema'],strict=True)
            optimizer.load_state_dict(p['optimizer']);scaler.load_state_dict(p['scaler'])
            if u and (not optimizer.state or any(int(v['step'].item())!=u for v in optimizer.state.values())):
                raise ValueError('Optimizer moment step mismatch')
            restore_rng(p['rng'],device=next(model.parameters()).device)
            atomic_json(run/'LATEST.json',identity)
            if errors:
                archive=run/'recovery_archive'/f'{time.time_ns()}_rejected_checkpoints';archive.mkdir(parents=True)
                for error in errors:
                    rejected=Path(error['file'])
                    # Preserve rejected files and sidecars; do not collide with the replayed update.
                    tensor=rejected.with_name(rejected.name.replace('.identity.json','.pt'))
                    for old in (rejected,tensor):
                        if old.exists():old.rename(archive/old.name)
                append_json(run/'recovery.jsonl',{'time':time.time(),'recovered_update':u,'rejected':errors,'archive':str(archive)})
            return u
        except (OSError,ValueError,RuntimeError,KeyError,EOFError) as e:
            errors.append({'file':str(sidecar),'error':repr(e)})
    tensors=list((run/'checkpoints').glob('*.pt'))
    if not candidates and tensors and all(p.name=='full_U00000.pt' for p in tensors):
        # The first U0 tensor can be durable before its commit sidecar exists.
        # Returning zero is safe only if no evidence of any training is present;
        # the caller has just rebuilt the pinned common initialization.
        status=run/'STATUS.json'
        observed=json.loads(status.read_text()) if status.exists() else {}
        scalars=run/'scalars.jsonl'
        prior=(run/'LATEST.json').exists() or observed.get('update',0)>0 or (scalars.exists() and scalars.stat().st_size>0)
        prior=prior or any((run/'eval_queue').glob('*')) or any((run/'eval_checkpoints').glob('*'))
        if not prior:
            archive=run/'recovery_archive'/f'{time.time_ns()}_uncommitted_U0';archive.mkdir(parents=True)
            for orphan in tensors:orphan.rename(archive/orphan.name)
            append_json(run/'recovery.jsonl',{'time':time.time(),'event':'uncommitted_initial_U0_archived',
                'update':0,'archive':str(archive)})
            return 0
    if candidates or tensors:
        raise RuntimeError('No complete compatible checkpoint remains: '+repr(errors))
    return 0


def save_eval_weights(run, group, model, update, binding):
    path=Path(run)/'eval_checkpoints'/f'update_{update:05d}.pt'
    sidecar=path.with_suffix('.identity.json')
    if path.exists()!=sidecar.exists():
        # A crash can occur after the atomic tensor rename but before its commit
        # sidecar. The caller holds the exact model for this update; preserve the
        # incomplete transaction and regenerate it instead of repeatedly failing.
        archive=Path(run)/'recovery_archive'/f'{time.time_ns()}_incomplete_eval_U{update:05d}'
        archive.mkdir(parents=True)
        for orphan in (path,sidecar):
            if orphan.exists():orphan.rename(archive/orphan.name)
        append_json(Path(run)/'recovery.jsonl',{'time':time.time(),'event':'incomplete_eval_checkpoint_rebuilt',
            'update':update,'archive':str(archive)})
    if path.exists():
        identity=json.loads(sidecar.read_text())
        if digest(path)!=identity['sha256'] or identity['binding']!=binding:raise ValueError('Existing eval checkpoint corrupt/incompatible')
    else:
        atomic_save({'model':cpu_tree(model.state_dict()),'update':update,'group':group,
                     'binding':binding,'weights':'RAW'},path)
        atomic_json(sidecar,{'sha256':digest(path),'update':update,'binding':binding})
    return path

"""Identity-bound complete state; schedule/sampler are explicit, never inferred.

This candidate's acceptance script labels its tiny optimizer test as acceptance,
not as formal B32 updates. Formal run authorization/launch is intentionally absent.
"""
from pathlib import Path
import torch
from execution.checkpoints import atomic_save, cpu_tree
from npu_runtime.backend import capture_rng, restore_rng
from v4.runtime import sha256, save_json

PURPOSE='V4_NAFNET_SIDD116M_STATE_V1'


def save(path, *, model, optimizer, scaler, group, binding, update, exposures,
         scheduler, sampler, noise_generator=None):
    path=Path(path)
    payload={'purpose':PURPOSE,'group':group,'binding':binding,'update':update,'exposures':exposures,
             'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
             'scheduler':scheduler,'scaler':scaler.state_dict(),'sampler':sampler,
             'rng':capture_rng(noise_generator,device=next(model.parameters()).device)}
    atomic_save(payload,path)
    identity={'sha256':sha256(path),'group':group,'binding':binding,'update':update,'exposures':exposures}
    save_json(path.with_suffix('.identity.json'),identity)
    return identity


def restore(path, *, model, optimizer, scaler, group, binding, noise_generator=None):
    import json
    path=Path(path);identity=json.loads(path.with_suffix('.identity.json').read_text())
    if identity['sha256']!=sha256(path) or identity['group']!=group or identity['binding']!=binding:
        raise ValueError('NAFNet checkpoint integrity or binding mismatch')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['purpose']!=PURPOSE or p['group']!=group or p['binding']!=binding:
        raise ValueError('Cross-run/backbone checkpoint forbidden')
    if p['update']!=identity['update'] or p['exposures']!=identity['exposures']:
        raise ValueError('Checkpoint counter mismatch')
    for name,value in model.stats.state_dict().items():
        if not torch.equal(p['model']['stats.'+name],value.cpu()):raise ValueError('Statistics changed')
    model.load_state_dict(p['model'],strict=True);optimizer.load_state_dict(p['optimizer']);scaler.load_state_dict(p['scaler'])
    restore_rng(p['rng'],noise_generator,device=next(model.parameters()).device)
    return {k:p[k] for k in ('update','exposures','scheduler','sampler')}

"""Bookkeeping-only six-to-five-rank migration; mathematical model stays frozen."""
import copy
import torch
from grad_helpers import state_digest
from scripts.audit_ddp_resume import equal_tree
from scripts.runtime_utils import sha

def admit_parent(payload, config, identity, checkpoint):
    if payload['identity']==identity and payload['config']==config:
        return 'same_five_rank_resume'
    assert config['stage']=='A'
    assert sha(checkpoint)==config['migration_parent_checkpoint_sha256']
    assert payload['update']==2410 and payload['cursor']==115680
    assert payload['config']['world_size']==6 and config['world_size']==5
    assert payload['identity']['physical_devices']==[2,3,4,5,6,7]
    assert identity['physical_devices']==[3,4,5,6,7]
    allowed={'mode','updates','world_size','effective_batch','stop_request'}
    for key,value in payload['config'].items():
        if key not in allowed:assert config[key]==value,('Changed non-topology config',key)
    assert config['data_cursor_offset']==115680-2410*config['effective_batch']
    assert payload['scheduler']['last_update']==2410 and payload['scheduler']['total_updates']==6000
    for key in ['fingerprint_sha256','pair_manifest_sha256','trajectory_manifest_sha256','population','task_queue','runtime','torch','torch_npu']:
        assert payload['identity'][key]==identity[key],('Changed scientific/runtime identity',key)
    for key,value in payload['identity']['code_sha256'].items():
        # The new runner adds topology/cursor admission, not mathematical operators.
        if key!='scripts/run_ab_authorized.py':assert identity['code_sha256'][key]==value,key
    return 'authorized_six_to_five_migration'

def rng_for_physical_card(payload,physical,new_devices):
    old_devices=payload['identity']['physical_devices']
    selected=copy.deepcopy(payload['rng_by_rank'][old_devices.index(physical)])
    if 'device' in selected:
        assert len(selected['device'])==len(old_devices)
        selected['device']=[selected['device'][old_devices.index(c)] for c in new_devices]
    return selected

def inheritance_proof(payload,model,opt,ema):
    core=state_digest(model.core.state_dict());ema_digest=state_digest(ema.model.state_dict())
    assert core==state_digest(payload['core'])
    assert ema_digest==state_digest(payload['ema']['model'])
    assert equal_tree(opt.state_dict(),payload['optimizer'])
    assert ema.updates==payload['update']
    assert all(int(v['step'])==payload['update'] for v in opt.state_dict()['state'].values())
    return {'core_exact':True,'EMA_exact':True,'Adam_exact':True,'EMA_updates':ema.updates,
        'loaded_update':payload['update'],'loaded_cursor':payload['cursor'],'core_digest':core,'EMA_digest':ema_digest}

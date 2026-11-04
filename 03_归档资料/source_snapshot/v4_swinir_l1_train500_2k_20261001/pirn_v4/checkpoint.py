"""Strict RAW/EMA/AdamW/RNG/data-cursor round-trip for preparation at U0.

No trained optimizer moments or nonzero-update resume is claimed by this gate.
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from .factory import ROOT,GROUPS,configuration,initialization_contract
from v4.data import canonical_hash
from v4.refinement import module_hash
from v4.runtime import save_json,sha256
from v4.training import atomic_torch_save

PURPOSE='V4_PIRN_MATCHED4_PREPARATION_FULL_STATE_ZERO_UPDATES'

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'].cpu())

def validate_payload(payload,model,group,root=ROOT):
    config=configuration(root); stream=json.loads((Path(root)/'protocol/stream.json').read_text())
    common=json.loads((Path(root)/'protocol/COMMON_INITIALIZATION.json').read_text())
    expected={'purpose':PURPOSE,'group':group,'update':0,'exposures':0,'stream_position':0,
              'stream_content_hash':stream['content_hash'],'common_initialization_sha256':common['sha256'],
              'architecture':model.config,'stats_artifact_hash':model.stats.artifact_hash,
              'initialization_contract_hash':canonical_hash(initialization_contract(config,root))}
    if group not in GROUPS or any(payload.get(k)!=v for k,v in expected.items()):
        raise ValueError('Checkpoint architecture/group/data/normalization/purpose identity mismatch')
    if payload['optimizer']['state']: raise ValueError('Preparation cannot contain trained optimizer moments')
    for label in ('model','ema'):
        if set(payload[label])!=set(model.state_dict()): raise ValueError('Checkpoint state keys mismatch')
        for key,value in model.stats.state_dict().items():
            if not torch.equal(payload[label]['stats.'+key].cpu(),value.cpu()):
                raise ValueError('Checkpoint normalization buffer mismatch')

def save_checkpoint(path,model,ema,optimizer,group,root=ROOT):
    path,root=Path(path),Path(root)
    if path.exists(): raise FileExistsError('Preserve existing acceptance checkpoint')
    stream=json.loads((root/'protocol/stream.json').read_text())
    common=json.loads((root/'protocol/COMMON_INITIALIZATION.json').read_text())
    payload={'purpose':PURPOSE,'group':group,'update':0,'exposures':0,'stream_position':0,
             'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'rng':capture_rng(),
             'architecture':model.config,'initialization_contract_hash':canonical_hash(initialization_contract(configuration(root),root)),
             'common_initialization_sha256':common['sha256'],'stream_content_hash':stream['content_hash'],
             'stats_artifact_hash':model.stats.artifact_hash,'model_hash':module_hash(model),'ema_hash':module_hash(ema),
             'scheduler':{'completed_updates':0,'kind':'inherited_template_not_launched'}}
    validate_payload(payload,model,group,root);atomic_torch_save(payload,path)
    identity={'purpose':PURPOSE,'group':group,'update':0,'sha256':sha256(path)}
    save_json(path.with_suffix('.identity.json'),identity);return identity

def load_checkpoint(path,model,ema,optimizer,group,root=ROOT):
    path=Path(path);identity=json.loads(path.with_suffix('.identity.json').read_text())
    if sha256(path)!=identity['sha256']: raise ValueError('Checkpoint file hash mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    validate_payload(payload,model,group,root)
    model.load_state_dict(payload['model'],strict=True);ema.load_state_dict(payload['ema'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    if module_hash(model)!=payload['model_hash'] or module_hash(ema)!=payload['ema_hash']:
        raise ValueError('Checkpoint tensor hash mismatch')
    restore_rng(payload['rng']);return payload

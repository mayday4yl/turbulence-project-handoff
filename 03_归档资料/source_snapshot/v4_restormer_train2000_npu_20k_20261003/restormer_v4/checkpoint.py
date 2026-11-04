"""Candidate U0-only checkpoint roundtrip; never a formal resume checkpoint."""
import random
from pathlib import Path
import json
import hashlib

import numpy as np
import torch
from v4.data import canonical_hash
from v4.refinement import module_hash, tensor_hash
from v4.runtime import save_json, sha256
from v4.training import atomic_torch_save
from .factory import ROOT, GROUPS, configuration, initialization_contract, build_optimizer

PURPOSE = 'V4_RESTORMER_CANDIDATE_FULL_STATE_ZERO_UPDATES'


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() and torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def validate_targets(model, ema, optimizer):
    """Reject incompatible restore containers before changing any live state."""
    if not isinstance(ema, torch.nn.Module) or ema is model:
        raise ValueError('EMA must be an independent model module')
    current, ema_state = model.state_dict(), ema.state_dict()
    if set(ema_state) != set(current):
        raise ValueError('Target EMA state keys mismatch')
    for name, value in current.items():
        target = ema_state[name]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError('Target EMA tensor shape/dtype mismatch')

    # Distinct Parameter objects can still share storage via detach/views.
    # Reject those aliases too, including an EMA buffer aliasing a model weight.
    model_parameter_ids = {id(p) for p in model.parameters()}
    model_parameter_storage = {
        (p.device, p.untyped_storage().data_ptr())
        for p in model.parameters() if p.numel()
    }
    if any(id(p) in model_parameter_ids for p in ema.parameters()) or any(
        value.numel() and (value.device, value.untyped_storage().data_ptr())
        in model_parameter_storage for value in ema_state.values()
    ):
        raise ValueError('Target EMA must not alias model parameter storage')

    expected_optimizer = build_optimizer(model)
    if type(optimizer) is not type(expected_optimizer):
        raise ValueError('Target optimizer type mismatch')
    if len(optimizer.param_groups) != len(expected_optimizer.param_groups):
        raise ValueError('Target optimizer parameter group count mismatch')
    for actual, expected in zip(optimizer.param_groups, expected_optimizer.param_groups):
        if tuple(map(id, actual['params'])) != tuple(map(id, expected['params'])):
            raise ValueError('Target optimizer model parameter identity/order mismatch')


def validate_payload(payload, model, group, root=ROOT):
    root = Path(root)
    common = json.loads((root / 'protocol/COMMON_INITIALIZATION.json').read_text())
    expected = {'purpose': PURPOSE, 'group': group, 'update': 0, 'exposures': 0, 'stream_position': 0,
                'common_initialization_sha256': common['sha256'], 'architecture': model.config,
                'stats_artifact_hash': model.stats.artifact_hash,
                'initialization_contract_hash': canonical_hash(initialization_contract(configuration(root), root))}
    if group not in GROUPS or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError('Candidate checkpoint identity mismatch')
    if payload['optimizer']['state']:
        raise ValueError('Candidate U0 cannot contain trained optimizer moments')
    if payload['optimizer'] != build_optimizer(model).state_dict():
        raise ValueError('Candidate optimizer configuration/parameter mapping mismatch')
    if payload.get('scheduler') != {'completed_updates': 0, 'kind': 'NOT_SELECTED_NO_TRAINING_AUTHORITY'}:
        raise ValueError('Candidate scheduler must remain unselected at U0')
    rng = payload.get('rng')
    if not isinstance(rng, dict) or set(rng) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('Candidate RNG state keys mismatch')
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch'].cpu())
        if rng['cuda'] is not None:
            if not torch.cuda.is_available():
                raise ValueError('CUDA RNG state requires CUDA runtime')
            torch.Generator(device='cuda').set_state(rng['cuda'].cpu())
    except (ValueError, TypeError, RuntimeError, AttributeError) as error:
        raise ValueError('Invalid candidate RNG state before restoration') from error
    if payload['model_hash'] != common['model_hash'] or payload['ema_hash'] != common['model_hash']:
        raise ValueError('Candidate checkpoint must contain untouched common U0')
    current = model.state_dict()
    for label in ('model', 'ema'):
        if set(payload[label]) != set(current):
            raise ValueError('Checkpoint state keys mismatch')
        for name, value in current.items():
            saved = payload[label][name]
            if saved.shape != value.shape or saved.dtype != value.dtype or not torch.isfinite(saved).all():
                raise ValueError('Checkpoint tensor shape/dtype/finite mismatch')
        for name, value in model.stats.state_dict().items():
            if not torch.equal(payload[label]['stats.' + name].cpu(), value.cpu()):
                raise ValueError('Checkpoint normalization mismatch')
        digest = hashlib.sha256()
        for name, value in sorted(payload[label].items()):
            digest.update(name.encode()); digest.update(tensor_hash(value).encode())
        if digest.hexdigest() != common['model_hash']:
            raise ValueError('Checkpoint tensor content mismatch before model mutation')


def save_checkpoint(path, model, ema, optimizer, group, root=ROOT):
    path, root = Path(path), Path(root)
    if path.exists():
        raise FileExistsError('Preserve existing candidate checkpoint')
    validate_targets(model, ema, optimizer)
    common = json.loads((root / 'protocol/COMMON_INITIALIZATION.json').read_text())
    payload = {'purpose': PURPOSE, 'group': group, 'update': 0, 'exposures': 0, 'stream_position': 0,
               'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
               'rng': capture_rng(), 'architecture': model.config,
               'initialization_contract_hash': canonical_hash(initialization_contract(configuration(root), root)),
               'common_initialization_sha256': common['sha256'], 'stats_artifact_hash': model.stats.artifact_hash,
               'model_hash': module_hash(model), 'ema_hash': module_hash(ema),
               'scheduler': {'completed_updates': 0, 'kind': 'NOT_SELECTED_NO_TRAINING_AUTHORITY'}}
    validate_payload(payload, model, group, root)
    atomic_torch_save(payload, path)
    identity = {'purpose': PURPOSE, 'group': group, 'update': 0, 'sha256': sha256(path)}
    save_json(path.with_suffix('.identity.json'), identity)
    return identity


def load_checkpoint(path, model, ema, optimizer, group, root=ROOT):
    path = Path(path)
    identity = json.loads(path.with_suffix('.identity.json').read_text())
    if identity.get('purpose') != PURPOSE or identity.get('group') != group or sha256(path) != identity['sha256']:
        raise ValueError('Checkpoint file hash/identity mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_payload(payload, model, group, root)
    validate_targets(model, ema, optimizer)
    model.load_state_dict(payload['model'], strict=True); ema.load_state_dict(payload['ema'], strict=True)
    if module_hash(model) != payload['model_hash'] or module_hash(ema) != payload['ema_hash']:
        raise ValueError('Checkpoint tensor hash mismatch')
    optimizer.load_state_dict(payload['optimizer']); restore_rng(payload['rng'])
    return payload

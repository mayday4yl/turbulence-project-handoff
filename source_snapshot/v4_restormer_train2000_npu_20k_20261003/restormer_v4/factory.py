"""Restormer Train2000 identity binding; shared fresh U0 and frozen V4 methods."""
import json
from pathlib import Path

import torch
from physical_bridge.simulator import PiRNP2SSimulator
from v4.data import canonical_hash
from v4.normalization import FrozenRGBStats, load_artifact
from v4.refinement import module_hash
from v4.runtime import sha256, save_json
from v4.training import TrainingObjective, atomic_torch_save
from .model import RestormerNoiseBridge

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('D0', 'R0', 'R08', 'L08')
PURPOSE = 'V4_RESTORMER_TRAIN2000_COMMON_INITIALIZATION_U0'
CANDIDATE_U0_SHA256 = '5cb7587bbe0fc3089d281ea1af7f5ec89e17ac20a0690268fcf2a6e2a55231bb'
GROUP_RULES = {'D0': (0., 'direct', '1', 'N/A'),
               'R0': (0., 'physical', 'G', 'optional_serial'),
               'R08': (.8, 'physical', 'G', 'optional_serial'),
               'L08': (.8, 'linear', 'G', 'N/A')}


def configuration(root=ROOT):
    return json.loads((Path(root) / 'configs/runtime.json').read_text())


def group_spec(group, root=ROOT):
    if group not in GROUPS:
        raise ValueError('Expected D0/R0/R08/L08')
    spec = json.loads((Path(root) / 'configs/groups' / f'{group}.json').read_text())
    if spec['group'] != group or tuple(spec[k] for k in ('T0', 'path', 'predictions', 'refinement')) != GROUP_RULES[group]:
        raise ValueError('Four-method semantics changed')
    if spec['learned_parameters'] != 'shared+clean+realization' or not spec['full_BPTT']:
        raise ValueError('All methods require both heads and full BPTT')
    return spec


def verify_inherited(root=ROOT):
    root = Path(root)
    identity = json.loads((root / 'protocol/INHERITED_IDENTITY.json').read_text())
    for rel, expected in identity['files'].items():
        if sha256(root / rel) != expected:
            raise ValueError(f'Inherited scientific source changed: {rel}')
    return identity['files']


def stats_for(config, root=ROOT):
    root = Path(root)
    artifact = root / config['normalization_artifact']
    info = load_artifact(artifact)
    if info['population'] != 'train' or info['state_count'] != 32000:
        raise ValueError('Requires frozen canonical Train2000/32000 statistics')
    if info['dataset_id'] != config['dataset_id']:
        raise ValueError('Canonical dataset and normalization identity mismatch')
    expected = config.get('normalization_sha256')
    if expected is not None and sha256(artifact) != expected:
        raise ValueError('Train2000 normalization file identity mismatch')
    return FrozenRGBStats.from_artifact(artifact)


def initialization_contract(config, root=ROOT):
    root = Path(root)
    vendor_hash = sha256(root / 'vendor/restormer/d2turb/code/models/restormer.py')
    if vendor_hash != '70ba46b5f28b3d1e4780725bd742e7826642be0e488aac6e906891cdcb13331a':
        raise ValueError('Pinned D2Turb backbone source identity mismatch')
    return {'purpose': PURPOSE, 'network': {k:v for k,v in config['network'].items() if k != 'use_checkpoint'}, 'seed': config['seed'],
            'normalization_file_sha256': sha256(root / config['normalization_artifact']),
            'pretrained_weights': None, 'population': 'train2000',
            'candidate_u0_sha256': CANDIDATE_U0_SHA256, 'dataset_id': config['dataset_id'],
            'model_source_sha256': sha256(root / 'restormer_v4/model.py'),
            'vendor_source_sha256': vendor_hash,
            'inherited_scientific_files': verify_inherited(root)}


def initialize_common(root=ROOT):
    root = Path(root)
    config = configuration(root)
    contract = initialization_contract(config, root)
    path, descriptor = root / 'protocol/common_init.pt', root / 'protocol/COMMON_INITIALIZATION.json'
    if path.exists() or descriptor.exists():
        if not path.exists() or not descriptor.exists():
            raise RuntimeError('Incomplete U0; preserve evidence and review')
        info = json.loads(descriptor.read_text())
        if info['contract'] != contract or info['sha256'] != sha256(path):
            raise ValueError('Existing common initialization identity mismatch')
        return info
    with torch.random.fork_rng(devices=[]):
        # Seed the same CPU generator as manual_seed without touching CUDA RNG.
        torch.default_generator.manual_seed(config['seed'])
        model = RestormerNoiseBridge(stats_for(config, root), **config['network'])
    candidate = root / config.get('candidate_initialization', 'protocol/candidate_common_init.pt')
    if sha256(candidate) != CANDIDATE_U0_SHA256:
        raise ValueError('Pinned candidate U0 hash mismatch')
    old = torch.load(candidate, map_location='cpu', weights_only=True)
    if old.get('network_optimizer_steps') != 0:
        raise ValueError('Only candidate random U0 is allowed')
    state = dict(old['state_dict'])
    old_stats = {k:state[k].clone() for k in state if k.startswith('stats.')}
    for name, value in model.stats.state_dict().items():
        state['stats.' + name] = value
    model.load_state_dict(state, strict=True)
    # All learned tensors are retained bit-for-bit from random candidate U0.
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter, old['state_dict'][name]):
            raise ValueError('Candidate learned initialization changed')
    atomic_torch_save({'purpose': PURPOSE, 'contract': contract, 'architecture': model.config,
                       'network_optimizer_steps': 0, 'state_dict': model.state_dict()}, path)
    info = {'contract': contract, 'contract_hash': canonical_hash(contract), 'architecture': model.config,
            'sha256': sha256(path), 'model_hash': module_hash(model),
            'stats_artifact_hash': model.stats.artifact_hash,
            'parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'common_to_groups': list(GROUPS), 'network_optimizer_steps': 0,
            'formal_training_authorized': True,
            'candidate_learned_tensors_preserved': True,
            'statistics_replaced_train500_to_train2000': True,
            'candidate_statistics': {k:v.tolist() for k,v in old_stats.items()}}
    save_json(descriptor, info)
    return info


def build_model(config, group, device='cpu', root=ROOT):
    root = Path(root)
    group_spec(group, root)
    path = root / 'protocol/common_init.pt'
    info = json.loads((root / 'protocol/COMMON_INITIALIZATION.json').read_text())
    contract = initialization_contract(config, root)
    if info['contract'] != contract or info['sha256'] != sha256(path):
        raise ValueError('Common Restormer initialization binding failed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('purpose') != PURPOSE or payload.get('network_optimizer_steps') != 0 or payload.get('contract') != contract:
        raise ValueError('Unexpected initialization; trained/pretrained migration forbidden')
    with torch.random.fork_rng(devices=[]):
        model = RestormerNoiseBridge(stats_for(config, root), **config['network'])
    if {k:v for k,v in payload['architecture'].items() if k != 'use_checkpoint'} != {k:v for k,v in model.config.items() if k != 'use_checkpoint'}:
        raise ValueError('Restormer architecture mismatch')
    for name, value in model.stats.state_dict().items():
        if not torch.equal(payload['state_dict']['stats.' + name], value):
            raise ValueError('Checkpoint frozen statistics mismatch')
    model.load_state_dict(payload['state_dict'], strict=True)
    if module_hash(model) != info['model_hash']:
        raise ValueError('Common model tensor hash mismatch')
    model.precision = config.get("model_precision", config.get("precision", "fp32")).lower()
    if model.precision not in ("bf16", "fp16", "fp32"):
        model.precision = "fp32"
    return model.to(device)


def build(config, group, device='cpu', root=ROOT):
    model = build_model(config, group, device, root)
    for name, expected in config['asset_sha256'].items():
        if sha256(Path(config['asset_dir']) / name) != expected:
            raise ValueError(f'Simulator asset mismatch: {name}')
    sim = PiRNP2SSimulator(config['asset_dir'], **config['physics'], device=device).to(device).eval()
    objective = TrainingObjective(model, sim, group)
    spec = group_spec(group, root)
    if objective.T0 != spec['T0'] or (group != 'D0' and objective.path != spec['path']):
        raise ValueError('Objective dispatch changed method semantics')
    return model, sim, objective


def build_optimizer(model):
    """Scientific AdamW, identical parameter groups and hyperparameters on each backend."""
    norm_types = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d)
    excluded = {id(p) for m in model.modules() if isinstance(m, norm_types)
                for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if name.endswith('bias') or id(p) in excluded else decay).append(p)
    parameters = decay + no_decay
    if len({id(p) for p in parameters}) != len(parameters):
        raise RuntimeError('Duplicate optimizer parameters')
    if {id(p) for p in parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Incomplete optimizer coverage')
    return torch.optim.AdamW([{'params': decay, 'weight_decay': 1e-4},
                             {'params': no_decay, 'weight_decay': 0.}],
                            lr=1e-4, betas=(.9, .999), eps=1e-8, foreach=False)


def require_training_authorization(root=ROOT):
    info = json.loads((Path(root) / 'protocol/AUTHORIZATION.json').read_text())
    authorized = info.get('formal_training_authorized', info.get('npu_formal_training_authorized', False))
    authorized = authorized or (info.get('authorized') is True and
                                info.get('formal_training_hosts') == ['liteserver-de36'] and
                                info.get('automatic_launch_after_correctness') is True)
    if not authorized:
        raise RuntimeError('New run formal NPU authorization is required')
    return info

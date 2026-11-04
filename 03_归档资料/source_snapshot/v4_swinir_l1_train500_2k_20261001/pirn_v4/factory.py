"""PiRN binding to unchanged Train500 methods; no implicit weight migration."""
import json
from pathlib import Path
import torch
from physical_bridge.simulator import PiRNP2SSimulator
from .model import PiRNNoiseBridge
from v4.data import canonical_hash
from v4.normalization import FrozenRGBStats, load_artifact
from v4.refinement import module_hash
from v4.runtime import sha256, save_json
from v4.training import TrainingObjective, make_optimizer, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('D0', 'R0', 'R08', 'L08')
PURPOSE = 'V4_PIRN_MATCHED4_TRAIN500_COMMON_INITIALIZATION'
GROUP_RULES = {'D0':(0.,'direct','1','N/A'), 'R0':(0.,'physical','G','optional_serial'),
               'R08':(.8,'physical','G','optional_serial'), 'L08':(.8,'linear','G','N/A')}

def configuration(root=ROOT):
    return json.loads((Path(root)/'configs/runtime.json').read_text())

def group_spec(group, root=ROOT):
    if group not in GROUPS: raise ValueError('Expected D0/R0/R08/L08')
    spec=json.loads((Path(root)/'configs/groups'/f'{group}.json').read_text())
    if spec['group']!=group or tuple(spec[k] for k in ('T0','path','predictions','refinement'))!=GROUP_RULES[group]:
        raise ValueError('Four-method semantics changed')
    if spec['learned_parameters']!='shared+clean+realization' or not spec['full_BPTT']:
        raise ValueError('All four methods retain both heads and full BPTT')
    return spec

def stats_for(config, root=ROOT):
    artifact=Path(root)/config['normalization_artifact']; info=load_artifact(artifact)
    if info['population']!='train' or info['state_count']!=8000:
        raise ValueError('Requires frozen Train500/8000 statistics')
    identity=json.loads((Path(root)/'protocol/ADAPTATION_SPEC.json').read_text())
    if sha256(artifact)!=identity['normalization_file_sha256']:
        raise ValueError('Train500 normalization identity mismatch')
    return FrozenRGBStats.from_artifact(artifact)

def initialization_contract(config, root=ROOT):
    return {'purpose':PURPOSE,'network':config['network'],
            'normalization_file_sha256':sha256(Path(root)/config['normalization_artifact']),
            'population':'train500','seed':config['seed'],'pretrained_weights':None,
            'model_source_sha256':sha256(Path(root)/'pirn_v4/model.py'),
            'vendor_source_sha256':sha256(Path(root)/'vendor/pirn/swinir.py')}

def initialize_common(root=ROOT):
    root=Path(root); config=configuration(root); contract=initialization_contract(config,root)
    path,descriptor=root/'protocol/common_init.pt',root/'protocol/COMMON_INITIALIZATION.json'
    if path.exists() or descriptor.exists():
        if not path.exists() or not descriptor.exists():
            raise RuntimeError('Incomplete common initialization; preserve evidence and review')
        existing=json.loads(descriptor.read_text())
        if existing['contract']!=contract or existing['sha256']!=sha256(path):
            raise ValueError('Existing common initialization identity mismatch')
        return existing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        model=PiRNNoiseBridge(stats_for(config,root),**config['network'])
    atomic_torch_save({'purpose':PURPOSE,'contract':contract,'architecture':model.config,
                       'network_optimizer_steps':0,'state_dict':model.state_dict()},path)
    info={'contract':contract,'contract_hash':canonical_hash(contract),'architecture':model.config,
          'sha256':sha256(path),'model_hash':module_hash(model),'stats_artifact_hash':model.stats.artifact_hash,
          'parameter_count':sum(p.numel() for p in model.parameters()),
          'common_to_groups':list(GROUPS),'network_optimizer_steps':0}
    save_json(descriptor,info); return info

def build_model(config, group, device='cpu', root=ROOT):
    root=Path(root); group_spec(group,root); path=root/'protocol/common_init.pt'
    info=json.loads((root/'protocol/COMMON_INITIALIZATION.json').read_text())
    contract=initialization_contract(config,root)
    if info['contract']!=contract or info['sha256']!=sha256(path):
        raise ValueError('Common PiRN initialization binding failed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('purpose')!=PURPOSE or payload.get('network_optimizer_steps')!=0 or payload.get('contract')!=contract:
        raise ValueError('Unexpected initialization checkpoint; no migration allowed')
    model=PiRNNoiseBridge(stats_for(config,root),**config['network'])
    if payload['architecture']!=model.config: raise ValueError('PiRN architecture mismatch')
    for name,value in model.stats.state_dict().items():
        if not torch.equal(payload['state_dict']['stats.'+name],value):
            raise ValueError('Checkpoint Train500 statistics mismatch')
    model.load_state_dict(payload['state_dict'],strict=True)
    if module_hash(model)!=info['model_hash']: raise ValueError('Common model tensor hash mismatch')
    return model.to(device)

def build(config, group, device='cuda', root=ROOT):
    model=build_model(config,group,device,root)
    for name,expected in config['asset_sha256'].items():
        if sha256(Path(config['asset_dir'])/name)!=expected: raise ValueError('Simulator asset mismatch')
    sim=PiRNP2SSimulator(config['asset_dir'],**config['physics'],device=device).to(device).eval()
    if any(p.requires_grad for p in sim.parameters()): raise ValueError('Simulator must remain frozen')
    objective=TrainingObjective(model,sim,group); spec=group_spec(group,root)
    if objective.T0!=spec['T0'] or (group!='D0' and objective.path!=spec['path']):
        raise ValueError('Objective dispatch changed method semantics')
    return model,sim,objective

def build_optimizer(model):
    optimizer=make_optimizer(model)
    expected={id(p) for p in model.parameters() if p.requires_grad}
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise RuntimeError('Optimizer parameter coverage is not exact')
    return optimizer

def require_training_authorization(root=ROOT):
    root=Path(root);config=configuration(root);control=json.loads((root/'protocol/CONTROL.json').read_text())
    if not config.get('formal_training_authorized') or not control.get('formal_training_authorized') or control.get('stop_at_next_boundary',True):
        raise RuntimeError('Formal training not authorized or user stop requested')
    frozen=json.loads((root/'protocol/FROZEN.json').read_text())
    if frozen.get('status')!='PASS_ALL_GATES' or frozen.get('updates')!=2000:
        raise RuntimeError('Independent 2k runtime acceptance and execution freeze required')
    return frozen

"""Official SIDD pretrained body -> one identity-bound V4 initialization.

All shared NAFNet feature layers are transferred. The 3-channel intro is
expanded as [W/2,W/2], exactly preserving the original input mapping for Z=Y
up to floating-point reduction. Encoder/middle/decoder FiLM starts at identity.
The original ending predicts an additive RGB residual. V4 predicts absolute
sigmoid RGB, so the ending is NOT mislabeled as a compatible absolute-image
head: keep the original checkpoint intact and initialize the V4 heads freshly.
"""
import json
from pathlib import Path
import torch
from .factory import ROOT, GROUPS, group_spec, stats_for, verify_inherited, build_optimizer
from .model import NAFNetNoiseBridge
from .upstream import NAFNet, verify_upstream
from v4.runtime import sha256, save_json
from v4.data import canonical_hash
from v4.refinement import module_hash
from v4.training import atomic_torch_save, TrainingObjective
from physical_bridge.simulator import PiRNP2SSimulator

PURPOSE='V4_NAFNET_SIDD116M_OFFICIAL_PRETRAINED_BODY_COMMON_V4_U0'
DESCRIPTOR='protocol/PRETRAINED_COMMON_INITIALIZATION.json'
WEIGHTS='protocol/pretrained_common_init.pt'


def _mark_pretrained(model):
    model.config.update(initialization='official_SIDD_pretrained_body',
                        official_beta_gamma_initialization='loaded_from_official_trained_checkpoint',
                        condition_projection_init='stage_FiLM_zero_identity; new_xi_block_FiLM_trunc_normal_std_.02',
                        clean_head_initialization='fresh_absolute_sigmoid_head; original_residual_head_kept_in_official_file')
    return model


def configuration(root=ROOT):
    return json.loads((Path(root)/'configs/runtime_pretrained.json').read_text())


def official_model(root=ROOT):
    root=Path(root);verify_upstream()
    identity=json.loads((root/'protocol/PRETRAINED_SOURCE.json').read_text())
    path=root/identity['path']
    if sha256(path)!=identity['sha256']:raise ValueError('Official pretrained file identity mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    state=payload[identity['state_key']] if identity['state_key'] else payload
    model=NAFNet(img_channel=3,width=64,enc_blk_nums=[2,2,4,8],middle_blk_num=12,dec_blk_nums=[2,2,2,2])
    model.load_state_dict(state,strict=True)
    return model,identity


def contract_for(config, root=ROOT):
    root=Path(root)
    return {'purpose':PURPOSE,'network':{k:v for k,v in config['network'].items() if k!='use_checkpoint'},
            'new_layer_seed':config['seed'],'dataset_id':config['dataset_id'],
            'statistics_sha256':sha256(root/config['normalization_artifact']),
            'source_identity':json.loads((root/'protocol/PRETRAINED_SOURCE.json').read_text()),
            'model_source_sha256':sha256(root/'nafnet_v4/model.py'),
            'transfer_source_sha256':sha256(root/'nafnet_v4/pretrained.py'),
            'loader_source_sha256':sha256(root/'nafnet_v4/upstream.py'),
            'upstream':verify_upstream(),'inherited_science':verify_inherited(root),
            'intro_mapping':'[W/2,W/2], original bias','stage_film_initialization':'zero scale/shift, identity',
            'official_ending':'excluded; residual head differs from V4 sigmoid absolute RGB',
            'V4_heads':'fresh clean head; fresh xi decoder with zero raw output heads',
            'training_updates_on_V4':0,'formal_training_authorized':False}


def initialize_common(root=ROOT):
    root=Path(root);config=configuration(root);contract=contract_for(config,root)
    path,descriptor=root/WEIGHTS,root/DESCRIPTOR
    if path.exists() or descriptor.exists():
        if not path.exists() or not descriptor.exists():raise ValueError('Incomplete pretrained U0; preserve and diagnose')
        info=json.loads(descriptor.read_text())
        if info['contract']!=contract or sha256(path)!=info['sha256']:raise ValueError('Pretrained V4 U0 changed')
        return info
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config['seed'])
        model=_mark_pretrained(NAFNetNoiseBridge(stats_for(config,root),**config['network']))
    official,source=official_model(root)
    with torch.no_grad():
        for name in model._OFFICIAL_SHARED_MODULES:
            if name!='intro':getattr(model,name).load_state_dict(getattr(official,name).state_dict(),strict=True)
        model.intro.weight.copy_(torch.cat([official.intro.weight/2]*2,dim=1))
        model.intro.bias.copy_(official.intro.bias)
        for film in model.condition_film.values():film.weight.zero_();film.bias.zero_()
    original=dict(official.named_parameters());unchanged=[];mapping=[]
    for name,param in model.named_parameters():
        if name.split('.')[0] not in model._OFFICIAL_SHARED_MODULES:continue
        if name=='intro.weight':
            assert torch.equal(param,torch.cat([original[name]/2]*2,dim=1))
            mapping.append(name)
        else:
            assert torch.equal(param,original[name]);unchanged.append(name)
    transfer={'exactly_transferred_tensors':unchanged,'exactly_transferred_parameter_count':sum(original[n].numel() for n in unchanged),
              'mapped_tensors':mapping,'official_unused_tensors':['ending.weight','ending.bias'],
              'official_unused_parameter_count':sum(original[n].numel() for n in ['ending.weight','ending.bias']),
              'fresh_or_added_tensors':[n for n,p in model.named_parameters() if n.split('.')[0] not in model._OFFICIAL_SHARED_MODULES],
              'all_shared_layers_accounted_for':True}
    atomic_torch_save({'purpose':PURPOSE,'contract':contract,'architecture':model.config,
                       'V4_optimizer_updates':0,'state_dict':model.state_dict()},path)
    info={'purpose':PURPOSE,'contract':contract,'contract_hash':canonical_hash(contract),'architecture':model.config,
          'sha256':sha256(path),'model_hash':module_hash(model),'parameter_count':sum(p.numel() for p in model.parameters()),
          'common_to_groups':list(GROUPS),'V4_optimizer_updates':0,'initialization':'official_SIDD_pretrained_body',
          'official_source_sha256':source['sha256'],'transfer':transfer,'formal_training_authorized':False}
    save_json(descriptor,info);return info


def build_model(config,group,device='cpu',root=ROOT):
    root=Path(root);group_spec(group,root)
    info=json.loads((root/DESCRIPTOR).read_text());contract=contract_for(config,root)
    if info['contract']!=contract or sha256(root/WEIGHTS)!=info['sha256']:raise ValueError('Pretrained common initialization mismatch')
    payload=torch.load(root/WEIGHTS,map_location='cpu',weights_only=True)
    if payload['purpose']!=PURPOSE or payload['V4_optimizer_updates']!=0 or payload['contract']!=contract:
        raise ValueError('Wrong initialization; no Restormer/NPU/trained-V4 checkpoint migration')
    with torch.random.fork_rng(devices=[]):model=_mark_pretrained(NAFNetNoiseBridge(stats_for(config,root),**config['network']))
    if {k:v for k,v in model.config.items() if k!='use_checkpoint'}!={k:v for k,v in payload['architecture'].items() if k!='use_checkpoint'}:
        raise ValueError('Architecture changed')
    for k,v in model.stats.state_dict().items():
        if not torch.equal(v,payload['state_dict']['stats.'+k]):raise ValueError('Train statistics changed')
    model.load_state_dict(payload['state_dict'],strict=True)
    if module_hash(model)!=info['model_hash']:raise ValueError('Pretrained tensor binding mismatch')
    model.precision=config['precision']
    if model.precision not in ('bf16','fp16','fp32'):raise ValueError('Unknown precision')
    return model.to(device)


def build(config,group,device='cpu',root=ROOT):
    model=build_model(config,group,device,root)
    for name,expected in config['asset_sha256'].items():
        if sha256(Path(config['asset_dir'])/name)!=expected:raise ValueError('Simulator asset changed '+name)
    sim=PiRNP2SSimulator(config['asset_dir'],**config['physics'],device=device).to(device).eval()
    objective=TrainingObjective(model,sim,group);spec=group_spec(group,root)
    if objective.T0!=spec['T0'] or (group!='D0' and objective.path!=spec['path']):raise ValueError('Method semantics changed')
    return model,sim,objective

import torch,copy
from pathlib import Path
from .common import *
from nafnet_v4.model import NAFNetNoiseBridge
from nafnet_v4.pretrained import _mark_pretrained
from nafnet_v4.factory import build_optimizer,verify_inherited
from nafnet_v4.upstream import verify_upstream
from v4.normalization import FrozenRGBStats,load_artifact
from v4.refinement import module_hash
from v4.training import TrainingObjective
from physical_bridge.simulator import PiRNP2SSimulator
from execution.checkpoints import atomic_save

def prepare_common(c):
 assert sha(CANDIDATE/'assets/pretrained/NAFNet-SIDD-width64.pth')==OFFICIAL_SHA
 assert sha(CANDIDATE/'protocol/pretrained_common_init.pt')==CANDIDATE_U0_SHA
 acceptance=read(CANDIDATE/'reports/pretrained/ACCEPTANCE_SUMMARY.json');assert acceptance['status']=='PASS_PRETRAINED_ASSETS_AND_BOUNDED_ACCEPTANCE'
 verify_inherited(CANDIDATE);verify_upstream()
 p=torch.load(CANDIDATE/'protocol/pretrained_common_init.pt',map_location='cpu',weights_only=True);assert p['V4_optimizer_updates']==0
 stats_path=Path(c['dataset_root'])/'normalization_train8000.json';stats=load_artifact(stats_path);assert stats['state_count']==128000 and stats['population']=='train'
 model=_mark_pretrained(NAFNetNoiseBridge(FrozenRGBStats.from_artifact(stats_path),**c['network']))
 state=dict(p['state_dict']);changed=[]
 for n,v in model.stats.state_dict().items():state['stats.'+n]=v;changed.append('stats.'+n)
 model.load_state_dict(state,strict=True);assert all(x.requires_grad for x in model.parameters())
 assert sum(x.numel() for x in model.parameters())==117767209
 assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
 param_names=set(dict(model.named_parameters()));assert all(torch.equal(v,p['state_dict'][n]) for n,v in model.state_dict().items() if n in param_names)
 contract={'purpose':'NAFNET_TRAIN8000_OFFICIAL_PRETRAINED_COMMON_U0','candidate_U0_sha256':CANDIDATE_U0_SHA,'official_sha256':OFFICIAL_SHA,'dataset_id':DATASET_ID,'stats_sha256':sha(stats_path),'adapter_sha256':sha(CANDIDATE/'nafnet_v4/model.py'),'all_trainable_tensors_exactly_candidate_U0':True,'only_rebound_buffers':changed,'parameter_count':117767209,'all_parameters_trainable':True,'has_batchnorm':False,'acceptance_sha256':sha(CANDIDATE/'reports/pretrained/ACCEPTANCE_SUMMARY.json'),'common_to_groups':list(GROUPS)}
 path=ROOT/'protocol/common_init.pt';desc=ROOT/'protocol/COMMON_INITIALIZATION.json'
 if path.exists() or desc.exists():
  d=read(desc);assert d['contract']==contract and sha(path)==d['sha256'];return d
 atomic_save({'purpose':contract['purpose'],'contract':contract,'state_dict':model.state_dict(),'V4_optimizer_updates':0},path)
 d={'contract':contract,'sha256':sha(path),'model_hash':module_hash(model)};freeze(desc,d);return d

def build(c,g,device):
 assert g in GROUPS;desc=read(ROOT/'protocol/COMMON_INITIALIZATION.json');path=ROOT/'protocol/common_init.pt';assert sha(path)==desc['sha256']
 st=Path(c['dataset_root'])/'normalization_train8000.json';assert sha(st)==desc['contract']['stats_sha256']
 p=torch.load(path,map_location='cpu',weights_only=True);assert p['contract']==desc['contract'] and p['V4_optimizer_updates']==0
 model=_mark_pretrained(NAFNetNoiseBridge(FrozenRGBStats.from_artifact(st),**c['network']));model.load_state_dict(p['state_dict'],strict=True);assert module_hash(model)==desc['model_hash'];model.precision=c['precision'];model=model.to(device)
 for name,h in c['asset_sha256'].items():assert sha(Path(c['asset_dir'])/name)==h
 sim=PiRNP2SSimulator(c['asset_dir'],**c['physics'],device=device).to(device).eval();sim.requires_grad_(False)
 obj=TrainingObjective(model,sim,g);opt=build_optimizer(model)
 assert all(x.requires_grad for x in model.parameters()) and not any(x.requires_grad for x in sim.parameters())
 assert {id(x) for q in opt.param_groups for x in q['params']}=={id(x) for x in model.parameters()}
 return model,sim,obj,opt

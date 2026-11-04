"""CPU-only identity checks, common initial state, and pre-result sample freeze."""
from pathlib import Path
import hashlib, json, os, random, shutil
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch
from physical_bridge.model import DualHeadUNet
from v031.runtime import sha256, state_hash, save_json, verify_source
from v031.followup import CleanOnly
from v031.training import atomic_torch_save, parameter_groups

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data2/yanglin/v031_overfit100_g4321_n4')

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def main():
    torch.set_num_threads(4)
    cfg=json.loads((OLD/'configs/train_v031.json').read_text())
    verify_source(cfg)
    data=Path(cfg['dataset_root'])
    for name,value in cfg['data_manifest_hashes'].items():
        assert sha256(data/name)==value, name
    for name,value in cfg['simulator_asset_hashes'].items():
        assert sha256(Path(cfg['asset_dir'])/name)==value, name
    assert sha256(data/'hash_manifest.json')==cfg['data_hash_manifest_sha256']
    ckpt=OLD/'runs/formal/checkpoint_001000.pt'
    assert sha256(ckpt)=='6eb3de6a064a0215a9a5b02935145528b23209d2f633b6dd50b60a999bad2454'
    for name,item in json.loads((ROOT/'shared/authority/FILE_HASHES.json').read_text()).items():
        assert sha256(ROOT/'shared/authority'/name)==item['sha256']
    for name,value in cfg['code_hashes'].items():
        assert sha256(OLD/name)==value, name
    for d in ('artifacts','configs','shared/manifests','T1_DIRECT','T2_FEEDBACK_R','I1_REFINE_REPAIR'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OLD/'configs/train_v031.json',ROOT/'shared/original_train_config.json')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    full=DualHeadUNet(**cfg['network'])
    full_hash=state_hash(full)
    assert full_hash==cfg['expected_initial_network_sha256'], (full_hash,cfg['expected_initial_network_sha256'])
    initial=ROOT/'shared/initial_full_network.pt'
    if not initial.exists(): atomic_torch_save(initial,full.state_dict())
    full.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
    assert state_hash(full)==full_hash
    clean=CleanOnly(full)
    clean_hash=state_hash(clean)
    parameter_counts={task:parameter_groups(net)[1]['total_trainable_parameters']
                      for task,net in [('T1_DIRECT',clean),('T2_FEEDBACK_R',full)]}
    sources=rows(data/'source_manifest.jsonl'); starts=rows(data/'start_manifest.jsonl')
    ordered=sorted((x['source_id'] for x in sources),
        key=lambda x:hashlib.sha256(('v031-refine-split|20260918|'+x).encode()).hexdigest())
    split={'refine_dev8':ordered[:8],'refine_confirm16':ordered[8:24]}
    manifests={}
    for label,ids in split.items():
        trajectories={s:min(r['trajectory_id'] for r in starts if r['source_id']==s) for s in ids}
        chosen=[next(r for r in starts if r['trajectory_id']==trajectories[s] and r['initial_G']==g)
                for s in ids for g in (4,3,2)]
        p=ROOT/'shared/manifests'/f'{label}.jsonl'
        content=''.join(json.dumps(r,sort_keys=True)+'\n' for r in chosen)
        if p.exists() and p.read_text()!=content: raise ValueError('Frozen IDs changed')
        p.write_text(content)
        manifests[label]={'source_ids':ids,'start_ids':[r['start_id'] for r in chosen],'sha256':sha256(p)}
    preview_source_ids=sorted(x['source_id'] for x in sources)[:5]
    save_json(ROOT/'shared/manifests/FROZEN_SPLITS.json',{
        'selection_before_new_quality':True,'splits':manifests,
        'additional_preview_source_ids':preview_source_ids,
        'legacy_preview_start_ids':cfg['evaluation']['fixed_preview_start_ids'],
        'diagnostic_arrays_dev_sources':ordered[:2],
        'diagnostic_arrays_confirm_sources':ordered[8:10],
        'full_xi_saved_sources':ordered[:2]})
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        for mb in (8,4,2,1):
            c=json.loads(json.dumps(cfg))
            c.update(task=task,project_root=str(ROOT),mode='FOLLOWUP_TO_1000',
                lambda_R=0.0 if task=='T1_DIRECT' else 0.2,
                initial_state_path=str(initial),expected_full_initial_network_sha256=full_hash,
                expected_initial_network_sha256=clean_hash if task=='T1_DIRECT' else full_hash)
            if task=='T1_DIRECT': c['expected_simulator_sha256']=hashlib.sha256(b'').hexdigest()
            c['training'].pop('ema_decay');c['training'].pop('ema_buffer_rule')
            c['training'].update(ema_enabled=False,microbatch=mb,accumulation=8//mb)
            c['evaluation'].update(weights=['raw'],modes=['unrefined'])
            c['identity']={'task':task,'plan_sha256':{n:i['sha256'] for n,i in
                json.loads((ROOT/'shared/authority/FILE_HASHES.json').read_text()).items()},
                'common_initial_network_sha256':full_hash,'formal_fresh_after_smoke':True}
            c['gates']={'targeted_acceptance_passed':False,'training_config_frozen':False}
            c.pop('code_hashes',None);c.pop('authority_hashes',None)
            save_json(ROOT/'configs'/f'{task}_mb{mb}.json',c)
    save_json(ROOT/'artifacts/run_manifest.json',{'method_version':'V0.31',
        'followup_version':'20260918','dataset_id':cfg['dataset_id'],
        'old_checkpoint_sha256':sha256(ckpt),'common_initial_network_sha256':full_hash,
        'clean_initial_network_sha256':clean_hash,'actual_parameter_counts':parameter_counts,
        'old_assets_read_only':True,'old_source_hashes':cfg['base_hashes'],
        'npu_acceptance_passed':False,'formal_training_started':False})
    print(json.dumps({'status':'cpu_preparation_passed','parameters':parameter_counts,
                      'splits':{k:len(v['start_ids']) for k,v in manifests.items()}}))

if __name__=='__main__':main()

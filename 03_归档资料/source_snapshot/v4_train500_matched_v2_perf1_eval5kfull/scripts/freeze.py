"""Freeze only after all independently recorded gates pass; never change budget."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,platform,socket,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deps'))
from v4.runtime import save_json,sha256,verify_imports
from v4.stages import verify_budget_scope,GROUPS
from v4.data import canonical_hash
from v4.normalization import load_artifact
import torch,lpips,skimage,torchvision


def main():
    expected=verify_budget_scope(ROOT)
    if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Explicit pause prevents freeze')
    if (ROOT/'protocol/FROZEN.json').exists():raise RuntimeError('Existing freeze retained; not overwritten')
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());imports=verify_imports(ROOT,cfg)
    spec=json.loads((ROOT/'protocol/TRAINING_PLAN_SPEC.json').read_text())
    auth=json.loads((ROOT/'protocol/AUTHORIZATION.json').read_text())
    assert auth['user_authorized'] and auth['automatic_formal_start_only_after_all_gates_PASS']
    assert not auth['scientific_changes_authorized'] and not auth['extension_authorized']
    for name,digest in auth['source_plan_hashes'].items():
        path=ROOT/('docs' if name.endswith('.md') else 'protocol')/name
        assert sha256(path)==digest, f'Approved source plan changed: {name}'
    gates=['reports/PREPARE_STATUS.json','reports/mutex_cpu.json','reports/execution_cpu.json',
           'reports/shared_gpu.json','reports/final_runtime.json','reports/mutex_with_stage_park.json']
    gates.extend(f'reports/acceptance_{g}.json' for g in GROUPS)
    gates.extend(f'reports/shared_gpu_acceptance/{g}.json' for g in GROUPS)
    evidence={}
    for name in gates:
        report=json.loads((ROOT/name).read_text())
        assert report['status']=='PASS', f'Gate not passed: {name}'
        evidence[name]=sha256(ROOT/name)
    steps=sum(json.loads((ROOT/f'reports/acceptance_{g}.json').read_text())['temporary_optimizer_steps'] for g in GROUPS)
    assert steps==8 and steps<=spec['preflight']['temporary_optimizer_steps_all_groups_max']
    for group in GROUPS:
        accepted=json.loads((ROOT/f'reports/acceptance_{group}.json').read_text())
        for name,item in accepted['imports'].items():assert imports[name]['sha256']==item['sha256']
        shared=json.loads((ROOT/f'reports/shared_gpu_acceptance/{group}.json').read_text())
        assert shared['repeat_fullval800']['range_db']<=.005
    stats=load_artifact(ROOT/cfg['normalization_artifact']);assert stats['state_count']==8000
    assert skimage.__version__=='0.25.2' and importlib.metadata.version('lpips')=='0.1.4'
    metric_assets=spec['evaluation']['metric_asset_sha256']
    for name,digest in metric_assets.items():assert sha256(ROOT/name)==digest
    sources={}
    for folder in ['v4','physical_bridge','scripts','tests','configs','docs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md','.sh'):sources[str(p.relative_to(ROOT))]=sha256(p)
    for p in (ROOT/'protocol').iterdir():
        if p.is_file() and p.name not in ('FROZEN.json','CONTROL.json') and p.suffix in ('.json','.pt','.py'):
            sources[str(p.relative_to(ROOT))]=sha256(p)
    environment={'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,
                 'torch_import':torch.__file__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':importlib.metadata.version('numpy'),
                 'skimage':skimage.__version__,'lpips':importlib.metadata.version('lpips'),'torchvision':torchvision.__version__,
                 'lpips_import':lpips.__file__,'skimage_import':skimage.__file__,'host':socket.gethostname(),
                 'driver':subprocess.check_output(['nvidia-smi','--id=5,6','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines(),
                 'precision':'FP32, TF32/autocast off, cudnn benchmark false'}
    payload={'status':'PASS_ALL_GATES','formal_training_authorized':True,'plan_id':spec['plan_id'],
             'plan_revision':spec['revision_id'],'created_time':time.time(),'updates':10000,'updates_per_group':expected,
             'batch':32,'microbatch':2,'accumulation':16,'world_size':1,
             'gpu_uuids':{'5':'GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673','6':'GPU-8079b40c-338b-f5a5-2be6-db5255033862'},
             'mapping':{'5':['R08','D0'],'6':['R0','L08']},'source_sha256':sources,'evidence_sha256':evidence,
             'environment':environment,'environment_identity':canonical_hash(environment),
             'metric_assets':metric_assets,'stats_artifact_hash':stats['artifact_hash'],
             'stream_sha256':sha256(ROOT/'protocol/stream.json'),'temporary_optimizer_updates':steps,
             'formal_updates_before_freeze':0,'refinement':False,'test_arrays_used':0,
             'milestones':[5000,7500,10000],'automatic_extension':False,
             'results_root':str(ROOT.parents[1]/'runs/v4_train500_matched_v2')}
    payload['protocol_hash']=canonical_hash(payload)
    save_json(ROOT/'protocol/FROZEN.json',payload)
    verify_budget_scope(ROOT)
    print(json.dumps({k:payload[k] for k in ['status','protocol_hash','updates_per_group','batch','microbatch','milestones']}),flush=True)

if __name__=='__main__':main()

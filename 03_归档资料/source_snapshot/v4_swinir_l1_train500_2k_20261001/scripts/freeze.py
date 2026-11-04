"""Immutable new-experiment execution freeze, only after recorded independent L1 gates."""
import importlib.metadata,json,os,platform,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
from v4.runtime import save_json,sha256,verify_imports
from v4.data import canonical_hash
from v4.stages import verify_budget_scope,GROUPS
import torch

def source_map():
    result={}
    for folder in ('v4','pirn_v4','physical_bridge','vendor','scripts','configs','docs','tests'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md','.sh') and '__pycache__' not in p.parts:result[str(p.relative_to(ROOT))]=sha256(p)
    for p in (ROOT/'protocol').iterdir():
        if p.is_file() and p.suffix in ('.json','.pt','.py') and p.name not in ('CONTROL.json','FROZEN.json','RESOURCE_WAIT.json'):result[str(p.relative_to(ROOT))]=sha256(p)
    return result

def main():
    if (ROOT/'protocol/FROZEN.json').exists():raise RuntimeError('Existing freeze retained, no overwrite')
    report=json.loads((ROOT/'reports/ACCEPTANCE.json').read_text());assert report['status']=='PASS' and report['formal_updates']==0
    assert report['temporary_optimizer_calls']==12
    assert source_map()==report['accepted_source_sha256'],'Source changed after acceptance began'
    cpu=json.loads((ROOT/'reports/CPU_ACCEPTANCE.json').read_text());assert cpu['status']=='PASS'
    selection=json.loads((ROOT/'protocol/RESOURCE_SELECTION.json').read_text());assert len(selection['gpu_uuids'])==2
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());verify_imports(ROOT,cfg)
    auth=json.loads((ROOT/'protocol/AUTHORIZATION.json').read_text());assert auth['user_authorized'] and auth['updates_per_method']==2000
    common=json.loads((ROOT/'protocol/COMMON_INITIALIZATION.json').read_text());assert sha256(ROOT/'protocol/common_init.pt')==common['sha256']
    for g in GROUPS:
        r=report['groups'][g];assert r['status']=='PASS' and r['common_U0_unchanged'] and r['simulator_unchanged']
        assert r['checks']['L1_B32_accumulation']['formal_initial_hash']==common['model_hash']
        assert r['peak_allocated_GiB']<28
    assets=json.loads((ROOT/'protocol/TRAINING_PLAN_SPEC.json').read_text())['evaluation']['metric_asset_sha256']
    for rel,digest in assets.items():assert sha256(ROOT/rel)==digest
    environment={'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_import':torch.__file__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'host':socket.gethostname(),'precision':'FP32 TF32/autocast off','numpy':importlib.metadata.version('numpy'),'lpips':importlib.metadata.version('lpips'),'skimage':importlib.metadata.version('scikit-image')}
    evidence={'reports/ACCEPTANCE.json':sha256(ROOT/'reports/ACCEPTANCE.json'),'reports/CPU_ACCEPTANCE.json':sha256(ROOT/'reports/CPU_ACCEPTANCE.json')}
    payload={'status':'PASS_ALL_GATES','formal_training_authorized':True,'updates':2000,'updates_per_group':{g:2000 for g in GROUPS},'batch':32,'microbatch':2,'accumulation':16,'world_size':1,'source_sha256':source_map(),'evidence_sha256':evidence,'environment':environment,'environment_identity':canonical_hash(environment),'stream_sha256':sha256(ROOT/'protocol/stream.json'),'common_initialization_sha256':common['sha256'],'common_initial_model_hash':common['model_hash'],'gpu_uuids':selection['gpu_uuids'],'mapping':selection['mapping'],'resource_mode':selection['mode'],'minimum_free_MiB':12288,'allocator_cap_fraction':.35,'compute_scheduling':'serialized_updates','milestones':[1000,2000],'test_arrays_used':0,'temporary_optimizer_updates':12,'formal_updates_before_freeze':0,'refinement_spec_sha256':sha256(ROOT/'protocol/REFINEMENT_ABLATION.json'),'results_root':str(ROOT.parents[1]/'runs'/ROOT.name),'automatic_extension':False,'created_at_server_time':time.time()}
    payload['protocol_hash']=canonical_hash({'science':json.loads((ROOT/'protocol/SCIENTIFIC_SPEC.json').read_text()),'normalization':sha256(ROOT/'protocol/normalization_train500.json'),'stream':payload['stream_sha256'],'common_init':common['sha256']})
    payload['execution_hash']=canonical_hash(payload);save_json(ROOT/'protocol/FROZEN.json',payload);verify_budget_scope(ROOT)
    print('PASS_ALL_GATES',payload['execution_hash'],flush=True)
if __name__=='__main__':main()

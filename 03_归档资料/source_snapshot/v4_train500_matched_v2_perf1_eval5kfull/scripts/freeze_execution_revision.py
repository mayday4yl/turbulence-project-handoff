"""Append a separately identified execution freeze; retain parent identity and bytes."""
import json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps')]
from v4.runtime import sha256,save_json,verify_imports
from v4.data import canonical_hash
from v4.stages import verify_budget_scope,GROUPS

def main():
    assert not (ROOT/'protocol/FROZEN.json').exists()
    assert not (ROOT/'reports/RUN_PAUSED.json').exists()
    parent=ROOT.parent/'v4_train500_matched_v2';old=json.loads((ROOT/'protocol/FROZEN_PARENT.json').read_text())
    assert sha256(parent/'protocol/FROZEN.json')==sha256(ROOT/'protocol/FROZEN_PARENT.json')
    verify_budget_scope(ROOT);verify_imports(ROOT,json.loads((ROOT/'configs/runtime.json').read_text()))
    for name in ['v4/core.py','v4/model.py','v4/training.py','v4/normalization.py','physical_bridge/model.py','physical_bridge/simulator.py',
                 'protocol/stream.json','protocol/common_init.pt','protocol/normalization_train500.json','protocol/PLANNED_MONITOR_LISTS.json',
                 'protocol/TRAINING_PLAN_SPEC.json','scripts/metrics.py']:
        assert sha256(ROOT/name)==old['source_sha256'][name],f'Scientific source/stream changed: {name}'
    required=['reports/PARENT_RAW_HOLD.json','reports/PERFORMANCE_STATUS.json','reports/EXECUTION_REVISION_ACCEPTANCE.json',
              'reports/RESOURCE_CPU_ACCEPTANCE.json','reports/REPORT_REVISION_ACCEPTANCE.json']
    evidence={}
    for name in required:
        data=json.loads((ROOT/name).read_text());assert data['status'].startswith(('PASS','BENCHMARK_PASS')),(name,data['status'])
        evidence[name]=sha256(ROOT/name)
    schedule=json.loads((ROOT/'protocol/RESOURCE_SCHEDULE.json').read_text())
    assert schedule['microbatch']==2 and schedule['effective_batch']==32
    run=pathlib.Path(old['results_root'])
    for g in GROUPS:
        p=run/g/'train_steps.jsonl';assert not p.exists() or not p.read_text().strip()
        latest=json.loads((run/g/'LATEST.json').read_text());assert latest['update']==0
        assert sha256(run/g/latest['path'])==latest['sha256']
    source={}
    for folder in ['v4','scripts','physical_bridge','tests','configs','docs','protocol']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md','.pt') and p.name not in ['CONTROL.json','FROZEN.json']:
                source[str(p.relative_to(ROOT))]=sha256(p)
    new={**old,'execution_revision':'perf1_RAW_schedule_v2','created_time':time.time(),'source_sha256':source,'evidence_sha256':evidence,
         'parent_runtime_freeze_file_sha256':sha256(ROOT/'protocol/FROZEN_PARENT.json'),
         'scientific_protocol_identity':old['protocol_hash'],'protocol_hash':old['protocol_hash'],
         'identity_note':'Parent scientific/checkpoint identity retained for exact U0 reuse; execution_hash uniquely identifies revised source/scheduling/evaluation schedule',
         'compute_scheduling':schedule['selected']['compute_scheduling'],'dynamic_single_GPU_relocation':schedule['selected']['dynamic_relocation'],
         'microbatch':2,'accumulation':16,'world_size':1,'additional_optimizer_steps':0,'formal_updates_before_execution_revision':0,
         'evaluation_schedule_revision':'v2_RAW_only','EMA_updates_maintained':True,'default_EMA_evaluation':False,
         'intermediate_populations':['train_monitor400','val800'],'final_populations':['train8000','val800'],
         'refinement_auxiliary_spec_sha256':sha256(ROOT/'protocol/REFINEMENT_ABLATION.json')}
    new['execution_hash']=canonical_hash(new)
    save_json(ROOT/'protocol/FROZEN.json',new)
    save_json(ROOT/'protocol/CONTROL.json',{'stop_at_common_update':None,'stop_at_next_boundary':False,'authorized_default_endpoint':10000,'reason':None})
    print(json.dumps({k:new[k] for k in ['protocol_hash','execution_hash','compute_scheduling','microbatch','updates_per_group']}))
if __name__=='__main__':main()

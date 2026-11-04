"""Append evaluation-only freeze and resume exact current states; no training step."""
import ast,difflib,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'v4_train500_matched_v2_perf1'
sys.path[:0]=[str(ROOT),str(ROOT/'deps')]


def main():
    assert not (ROOT/'protocol/FROZEN.json').exists()
    assert not (PARENT/'reports/RUN_PAUSED.json').exists()
    for folder in ['deps','assets']:
        if not (ROOT/folder).exists():(ROOT/folder).symlink_to(PARENT/folder,target_is_directory=True)
    old=json.loads((PARENT/'protocol/FROZEN.json').read_text())
    run=Path(old['results_root'])
    pids=json.loads((run/'PIDS.json').read_text());assert not pids['workers']
    try:os.kill(pids['coordinator'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Parent coordinator still running')
    for name in old['source_sha256']:
        dest=ROOT/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            if (PARENT/name).is_symlink():dest.symlink_to((PARENT/name).resolve())
            else:shutil.copy2(PARENT/name,dest)
    import torch
    from v4.runtime import save_json,sha256,verify_imports
    from v4.data import canonical_hash
    from v4.stages import verify_budget_scope
    from v4.matched import lr_at
    changed=[name for name,digest in old['source_sha256'].items() if sha256(ROOT/name)!=digest]
    expected={'scripts/train_worker.py','scripts/stage_report.py','v4/eta.py'}
    assert set(changed)==expected,changed
    diff=''
    for name in sorted(changed):
        a=(PARENT/name).read_text();b=(ROOT/name).read_text();ast.parse(b)
        diff+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='parent/'+name,tofile='eval5k/'+name))
    (ROOT/'reports/EVAL5K_SOURCE_DIFF.patch').write_text(diff)
    tree=ast.parse((ROOT/'scripts/train_worker.py').read_text())
    expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='populations' for t in n.targets))
    code=compile(ast.Expression(expression),'<evaluation-populations>','eval')
    for u,want in [(0,'train8000'),(250,'train_monitor400'),(5000,'train8000'),(7500,'train_monitor400'),(10000,'train8000')]:
        assert eval(code,{'u':u})==[want,'val800']
    verify_budget_scope(ROOT);verify_imports(ROOT,json.loads((ROOT/'configs/runtime.json').read_text()))
    states={}
    for g in ('D0','R0','R08','L08'):
        latest=json.loads((run/g/'LATEST.json').read_text());path=run/g/latest['path']
        assert sha256(path)==latest['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False);u=latest['update']
        assert state['update']==state['stream_position']==u and state['exposures']==32*u
        assert state['scheduler']['next_lr']==lr_at(u+1)
        assert state['stream_sha256']==old['stream_sha256'] and state['protocol_hash']==old['protocol_hash']
        assert state['model'] and state['ema'] and state['optimizer']['state'] and state['rng']
        lines=[json.loads(s) for s in (run/g/'train_steps.jsonl').read_text().splitlines()]
        assert [r['update'] for r in lines]==list(range(1,u+1))
        states[g]={'checkpoint':str(path),'sha256':latest['sha256'],'update':u,'next_update':u+1,
                   'RAW_EMA_optimizer_RNG_LR_stream_preserved':True}
        del state
    evidence={}
    for name,digest in old['evidence_sha256'].items():
        dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PARENT/name,dest)
        assert sha256(dest)==digest;evidence[name]=digest
    archive=ROOT/'protocol/FROZEN_EXECUTION_PARENT_PERF1.json';shutil.copy2(PARENT/'protocol/FROZEN.json',archive)
    check={'status':'PASS','scope':'Evaluation schedule routing/static source identity and actual full-checkpoint CPU verification; prior unchanged CUDA gates reused',
           'changed_source_files':changed,'unchanged_science_and_training_sources':True,'schedule_nodes_checked':[0,250,5000,7500,10000],
           'source_diff_sha256':sha256(ROOT/'reports/EVAL5K_SOURCE_DIFF.patch'),'resume_states':states,
           'optimizer_steps_during_revision':0,'parent_runtime_freeze_sha256':sha256(archive),'time':time.time()}
    save_json(ROOT/'reports/EVAL5K_REVISION_ACCEPTANCE.json',check)
    evidence['reports/EVAL5K_REVISION_ACCEPTANCE.json']=sha256(ROOT/'reports/EVAL5K_REVISION_ACCEPTANCE.json')
    for name in ['REFINEMENT_RUNTIME_R0.json','REFINEMENT_RUNTIME_R08.json']:
        shutil.copy2(PARENT/'reports'/name,ROOT/'reports'/name)
    source={}
    for folder in ['v4','scripts','physical_bridge','tests','configs','docs','protocol']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.md','.pt') and path.name not in ['CONTROL.json','FROZEN.json']:
                source[str(path.relative_to(ROOT))]=sha256(path)
    new={**old,'execution_revision':'perf1_eval5k_full_RAW_v3','created_time':time.time(),
         'source_sha256':source,'evidence_sha256':evidence,'parent_execution_hash':old['execution_hash'],
         'parent_runtime_freeze_file_sha256':sha256(archive),'evaluation_schedule_revision':'v3_RAW_fulltrain_at5k_and10k',
         'intermediate_populations':{'5000':['train8000','val800'],'7500':['train_monitor400','val800']},
         'formal_updates_before_execution_revision':{g:s['update'] for g,s in states.items()},
         'evaluation_only_revision':True,'scientific_changes':False}
    new.pop('execution_hash');new['execution_hash']=canonical_hash(new)
    save_json(ROOT/'protocol/FROZEN.json',new)
    save_json(ROOT/'protocol/CONTROL.json',{'stop_at_common_update':None,'stop_at_next_boundary':False,'authorized_default_endpoint':10000,'reason':None})
    print(json.dumps({'execution_hash':new['execution_hash'],'resume_states':states}),flush=True)


if __name__=='__main__':main()

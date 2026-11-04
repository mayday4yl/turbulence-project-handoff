"""Build train500 statistics and the fixed stream; no optimizer or GPU allocation."""
from pathlib import Path
import json, os, sys, shutil, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4.normalization import build_artifact, load_artifact
from v4.runtime import save_json,sha256,verify_imports
from v4.matched import Store,make_stream,INITIAL_SHA

def main():
    config=json.loads((ROOT/'configs/runtime.json').read_text())
    report={'stage':'PREPARE','status':'RUNNING','pid':os.getpid(),'time':time.time(),'formal_optimizer_steps':0}
    save_json(ROOT/'reports/PREPARE_STATUS.json',report)
    old=ROOT.parent/'v4_train_20260929/protocol/common_init.pt'
    if sha256(old)!=INITIAL_SHA:raise ValueError('Initial checkpoint identity mismatch')
    target=ROOT/'protocol/common_init.pt'
    if not target.exists():shutil.copy2(old,target)
    assert sha256(target)==INITIAL_SHA
    imports=verify_imports(ROOT,config)
    artifact_path=ROOT/config['normalization_artifact']
    if artifact_path.exists():artifact=load_artifact(artifact_path)
    else:artifact=build_artifact(config['dataset_root'],artifact_path,'train',asset_lock=config['asset_lock'])
    if artifact['state_count']!=8000 or any(v['unique_state_count']!=2000 for v in artifact['groups'].values()):
        raise ValueError('Wrong train500 stats population')
    store=Store(config)
    # Array verification restricted to required train/validation, never test.
    refs={v['ref']:v for r in store.train+store.val for v in [r['clean'],r['state']]}
    from v4.data import read_array
    for index,record in enumerate(refs.values()):
        read_array(store.root,record)
        if index%1000==0:print('verified train/val arrays',index,'/',len(refs),flush=True)
    stream=make_stream(store)
    path=ROOT/'protocol/stream.json'
    if path.exists() and json.loads(path.read_text())!=stream:raise ValueError('Existing stream differs')
    if not path.exists():save_json(path,stream)
    save_json(ROOT/'protocol/evaluation_lists.json', {k:[{f:r[f] for f in ('start_id','trajectory_id','source_id','split','G')} for r in v] for k,v in store.lists.items()})
    report.update(status='PASS',imports=imports,stats_artifact_hash=artifact['artifact_hash'],
                  stats_sha256=sha256(artifact_path),mu=artifact['mu'],std=artifact['std'],
                  train_states=8000,verified_train_val_arrays=len(refs),test_arrays_read=0,
                  stream_sha256=sha256(path),elapsed_seconds=time.time()-report['time'])
    save_json(ROOT/'reports/PREPARE_STATUS.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        save_json(ROOT/'reports/PREPARE_FAILURE.json',{'status':'PAUSED','exception':repr(e),'traceback':traceback.format_exc()})
        raise

"""Metadata-only recovery entrypoint; the accepted D0 trainer is unchanged."""
import json,os,sys,time
from pathlib import Path
from scripts.runtime_utils import ROOT,sha,write_json

REPAIR=ROOT/'execution_20261029_d0_recovery'
RUN='V6_D0_U14927_residual_recovery_20261029_01'

def main():
    for name,digest in json.loads((REPAIR/'RECOVERY_SOURCE_MANIFEST.json').read_text())['files'].items():
        assert sha(ROOT/name)==digest,('Recovery source drift',name)
    assert json.loads((REPAIR/'PARENT_U030500_CPU_AUDIT.json').read_text())['status']=='PASS'
    assert sys.argv[sys.argv.index('--run')+1]==RUN
    parent=json.loads((REPAIR/'PRE_RECOVERY_SNAPSHOT.json').read_text())['parent']
    assert sys.argv[sys.argv.index('--resume')+1]==parent['path']
    assert sha(parent['path'])==parent['sha256']
    best=json.loads((REPAIR/'INHERITED_BEST.json').read_text())
    assert best['update']<=parent['update'] and sha(best['path'])==best['sha256']
    out=ROOT/'distributed'/RUN
    if int(os.environ['RANK'])==0:
        assert not (out/'BEST.json').exists()
        write_json(out/'BEST.json',best)
        write_json(out/'RECOVERY_PROVENANCE.json',{
            'parent':parent,'source_run':'V6_D0_U14927_official_residual_microopt_01',
            'accepted_training_entrypoint':'scripts/run_d0_residual_microopt.py',
            'accepted_entrypoint_sha256':sha(ROOT/'scripts/run_d0_residual_microopt.py'),
            'wrapper_sha256':sha(__file__),'retained_old_logs_unchanged':True,
            'uncheckpointed_updates_replayed':420,'uncheckpointed_presentations_replayed':25200,
            'training_method_config_and_graph_changed':False})
    started=time.time()
    while not (out/'BEST.json').exists():
        assert time.time()-started<60,'Rank0 best inheritance did not finish'
        time.sleep(.1)
    assert json.loads((out/'BEST.json').read_text())==best
    # No model, random, optimizer, data, sampling or loss implementation here.
    from scripts.run_d0_residual_microopt import main as accepted_main
    accepted_main()

if __name__=='__main__':main()

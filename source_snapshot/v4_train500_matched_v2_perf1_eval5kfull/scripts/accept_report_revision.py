"""Render revised RAW deliverables from committed U0 only; no inference."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from v4.runtime import save_json,sha256
from v4.stages import GROUPS,verify_budget_scope
from stage_report import load_eval,plots,qualitative,paired_table,progression_table,write_table


def main():
    verify_budget_scope(ROOT)
    run=Path(json.loads((ROOT/'protocol/FROZEN_PARENT.json').read_text())['results_root'])
    out=ROOT/'reports/RAW_REPORT_ACCEPTANCE_ONLY';out.mkdir(exist_ok=False)
    curves=[];by={};identities={}
    for g in GROUPS:
        for pop in ['train_monitor400','val800']:
            a,path=load_eval(run,g,0,'raw',pop)
            assert a['start_count']==(400 if pop=='train_monitor400' else 800)
            identities[str(path/'COMMIT.json')]=sha256(path/'COMMIT.json')
            curves.append({'method':g,'update':0,'epoch':0,'exposures':0,'weights':'raw','population':pop,
                           **a['overall'],**{f'input_{k}':v for k,v in a['baseline']['overall'].items()}})
            if pop=='val800':by[g]=[json.loads(s) for s in (path/'per_start.jsonl').read_text().splitlines()]
    paired,_=paired_table(by);progression=progression_table(by)
    write_table(out,'paired_ablation_table',paired);write_table(out,'progression_by_G_table',progression)
    plots(out,curves,progression,0);qualitative(out,run,0,train_population='train_monitor400')
    assert len(list((out/'qualitative').glob('*.png')))==80
    for name in ['train_val_psnr_curve','train_val_loss_curve','progression_by_G']:
        assert (out/(name+'.png')).stat().st_size>1000
    for path,h in identities.items():assert sha256(Path(path))==h
    save_json(ROOT/'reports/REPORT_REVISION_ACCEPTANCE.json',{
        'status':'PASS','scope':'RAW report functions and fixed80 horizontal/vertical renders using existing U0 metrics; CPU only, zero inference/training',
        'input_commits_unchanged':True,'qualitative_count':80,'main_figures':3,
        'intermediate_train_population':'train_monitor400','source_hash':sha256(ROOT/'scripts/stage_report.py'),
        'time':time.time()})
    print('RAW report revision PASS',flush=True)


if __name__=='__main__':main()

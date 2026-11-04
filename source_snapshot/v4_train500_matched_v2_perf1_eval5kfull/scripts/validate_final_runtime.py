"""Actual production checkpoint API, committed subset reuse and report rendering.
Temporary acceptance only, zero optimizer updates, no formal result claims.
"""
from pathlib import Path
import json,os,shutil,sys,traceback,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'deps'))
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import torch
from v4.matched import Store,build
from v4.runtime import configure_cuda,save_json,sha256
from v4.refinement import module_hash
from v4.training import make_optimizer,make_ema
from v4.evaluate_matched import Evaluator
from v4.stages import GROUPS,verify_budget_scope
from train_worker import checkpoint,capture_rng,restore_rng
from stage_report import plots,qualitative,paired_table,progression_table,write_table
from validate_group import tree_equal


def main():
    assert json.loads((ROOT/'reports/shared_gpu.json').read_text())['status']=='PASS'
    verify_budget_scope(ROOT);configure_cuda(20260929)
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());store=Store(cfg)
    work=ROOT/'reports/final_runtime_acceptance';work.mkdir(exist_ok=False)
    model,sim,obj=build(cfg,'D0');opt=make_optimizer(model);ema=make_ema(model)
    initial=torch.load(ROOT/'reports/temporary_D0_step1.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(initial['model']);ema.load_state_dict(initial['ema']);opt.load_state_dict(initial['optimizer']);del initial
    frozen={'protocol_hash':'TEMPORARY_RUNTIME_ACCEPTANCE','stream_sha256':sha256(ROOT/'protocol/stream.json')}
    before=capture_rng();path=work/'TEMPORARY_NOT_FORMAL.pt'
    identity=checkpoint(path,model,ema,opt,1,frozen,'D0',purpose='TEMPORARY_RUNTIME_ACCEPTANCE_NOT_FORMAL')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    tree_equal(saved['model'],model.state_dict());tree_equal(saved['ema'],ema.state_dict());tree_equal(saved['optimizer'],opt.state_dict())
    restore_rng(saved['rng']);tree_equal(before,capture_rng())
    assert saved['stream_position']==1 and saved['exposures']==32 and saved['scheduler']['completed_updates']==1
    # Same existing state must not rewrite the checkpoint bytes.
    stat=path.stat().st_mtime_ns
    repeated=checkpoint(path,model,ema,opt,1,frozen,'D0',purpose='TEMPORARY_RUNTIME_ACCEPTANCE_NOT_FORMAL')
    assert identity==repeated and path.stat().st_mtime_ns==stat
    del saved,model,sim,obj,opt,ema;torch.cuda.empty_cache()
    # Reuse only registered batch2 pairs, never infer just to make reports.
    shared=ROOT/'reports/shared_gpu_acceptance';curves=[];by_group={};fake_run=work/'TEMPORARY_EVALUATION_LAYOUT'
    for group in GROUPS:
        directory=fake_run/group/'evaluations/U00000';directory.mkdir(parents=True)
        for label in ['raw','EMA']:
            (directory/label).symlink_to(shared/group/'full_evaluation'/label,target_is_directory=True)
            for pop in ['train_monitor400','val800']:
                a=json.loads((shared/group/'full_evaluation'/label/pop/'aggregate.json').read_text())
                curves.append({'method':group,'update':0,'epoch':0,'exposures':0,'weights':label,'population':pop,
                               **a['overall'],**{f'input_{k}':v for k,v in a['baseline']['overall'].items()}})
        source=shared/group/'full_evaluation/raw/val800'
        evaluator=Evaluator(store,'TEMPORARY_SHARED_ACCEPTANCE','current_registered_environment')
        subset=evaluator.derive_subset(source,store.val[:4],work/group/'subset4')
        assert subset['start_count']==4 and subset['inference_reused']
        originals=[json.loads(s) for s in (source/'per_start.jsonl').read_text().splitlines()]
        derived=[json.loads(s) for s in (work/group/'subset4/per_start.jsonl').read_text().splitlines()]
        assert derived==originals[:4]
        by_group[group]=originals
    paired,matrix=paired_table(by_group);prog=progression_table(by_group)
    render=work/'TEMPORARY_RENDER_TEST';render.mkdir()
    write_table(render,'paired_ablation_table',paired);write_table(render,'progression_by_G_table',prog)
    plots(render,curves,prog,0)
    qualitative(render,fake_run,0,train_population='train_monitor400')
    assert len(list((render/'qualitative').glob('*.png')))==160
    report={'status':'PASS','purpose':'TEMPORARY_ACCEPTANCE_ONLY_NOT_TRAINING_RESULTS','optimizer_steps':0,
            'real_production_checkpoint_roundtrip':True,'LR_rng_stream_counters_verified':True,
            'existing_checkpoint_not_rewritten':True,'subset_exact_reuse_without_inference':True,
            'three_main_figures_rendered':True,'qualitative_horizontal_vertical_raw_EMA_count':160,
            'budget_scope_consistent':True,'report_root':str(render)}
    save_json(ROOT/'reports/final_runtime.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save_json(ROOT/'reports/RUN_PAUSED.json',{'status':'PAUSED','gate':'final_runtime_checkpoint_reports','exception':repr(exc),'traceback':traceback.format_exc()})
        raise

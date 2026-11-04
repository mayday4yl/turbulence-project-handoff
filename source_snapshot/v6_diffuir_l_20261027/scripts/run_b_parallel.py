"""Unchanged B training runner with verified, explicitly labelled U0 evaluation reuse."""
import os
os.environ['TASK_QUEUE_ENABLE']='1'
import json,sys,shutil
from pathlib import Path
import torch.distributed as dist
from grad_helpers import state_digest
from scripts.runtime_utils import ROOT,sha,write_json
from scripts import run_ab_five_npu as runner
EX=ROOT/'execution_20261027_five_npu'

def can_reuse(out,full,stage):
    gate=EX/'B_INITIAL_VALIDATION_REUSE_GATE.json'
    return stage=='B' and full and Path(out).name=='U000000_1000' and gate.exists() and json.loads(gate.read_text())['status']=='PASS'

def main():
    cfg=json.load(open(sys.argv[sys.argv.index('--config')+1]));assert cfg['stage']=='B'
    assert cfg['validation_wrapper_sha256']==sha(__file__)
    original_evaluate=runner.evaluate_rank;original_summarize=runner.summarize
    def evaluate(model,ema,device,out,*,rank,world,full=True,stage):
        if not can_reuse(out,full,stage):return original_evaluate(model,ema,device,out,rank=rank,world=world,full=full,stage=stage)
        g=json.loads((EX/'B_INITIAL_VALIDATION_REUSE_GATE.json').read_text())
        assert g['A_checkpoint_sha256']==cfg['initial_checkpoint_sha256']
        assert state_digest(ema.model.state_dict())==g['initial_core_digest']
        assert sha(ROOT/'method_impl/ab_evaluation.py')==g['evaluation_source_sha256']
        assert sha(ROOT.parent/'v5_concertormer_10k_20261021/training/image_metrics.py')==g['metrics_source_sha256']
        for branch in ['native','sampling']:
            assert state_digest(dict(getattr(model.restorer,branch).named_buffers()))==g[branch+'_buffers_digest']
        if rank==0:
            source=Path(g['source_validation']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
            assert sha(source/'SUMMARY.json')==g['source_summary_sha256']
            assert sha(source/'PER_IMAGE.jsonl')==g['source_per_image_sha256']
            assert sha(source/'TF_GRID_VAL100.jsonl')==g['source_TF_grid_sha256']
            for name in ['PER_IMAGE.jsonl','TF_GRID_VAL100.jsonl']:shutil.copy2(source/name,out/name)
            if (source/'fixed_visual100').exists():shutil.copytree(source/'fixed_visual100',out/'fixed_visual100',dirs_exist_ok=False)
            summary=json.loads((source/'SUMMARY.json').read_text())
            summary.update(stage='B',update=0,status='REUSED_VERIFIED_EQUIVALENT',execution='REUSED_A_BEST_IDENTICAL_CORE_AND_PROTOCOL',
                fresh_B_Val1000_inference_executed=False,reuse_source=str(source),reuse_gate_sha256=sha(EX/'B_INITIAL_VALIDATION_REUSE_GATE.json'))
            write_json(out/'SUMMARY.json',summary);write_json(out/'REUSE_PROVENANCE.json',g)
        dist.barrier()
    def summarize(out,world,full,stage):
        if can_reuse(out,full,stage) and (Path(out)/'REUSE_PROVENANCE.json').exists():return json.loads((Path(out)/'SUMMARY.json').read_text())
        return original_summarize(out,world,full,stage)
    runner.evaluate_rank=evaluate;runner.summarize=summarize;runner.main()
if __name__=='__main__':main()

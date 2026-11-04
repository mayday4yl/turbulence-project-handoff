"""Static/runtime CPU contract checks; this alone never authorizes GPU training."""
import ast,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deps')];os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from v4.runtime import save_json,sha256
from v4.core import mae,mse
from v4.matched import lr_at
from v4.stages import NODES,read_control,verify_budget_scope
from pirn_v4.factory import configuration,initialize_common

def main():
    for p in ROOT.rglob('*.py'):
        if 'deps' not in p.parts and 'assets' not in p.parts and '__pycache__' not in p.parts:ast.parse(p.read_text())
    verify_budget_scope(ROOT);assert NODES==(1000,2000);read_control(ROOT/'protocol/CONTROL.json')
    assert math.isclose(lr_at(1),1e-6) and math.isclose(lr_at(100),1e-4) and math.isclose(lr_at(2000),1e-5)
    assert all(lr_at(u+1)<=lr_at(u) for u in range(100,2000))
    for u in (0,2001):
        try:lr_at(u)
        except ValueError:pass
        else:raise AssertionError('LR accepted unauthorized update')
    a=torch.zeros(2,3,4,4);b=torch.ones_like(a)*.25
    assert torch.equal(mae(a,b),torch.tensor([.25,.25])) and torch.equal(mse(a,b),torch.tensor([.0625,.0625]))
    common=initialize_common();assert common['network_optimizer_steps']==0
    cfg=configuration();assert cfg['network']['embed_dim']==180 and cfg['network']['depths']==[6]*8 and cfg['network']['num_heads']==[6]*8
    parent=ROOT.parent/'v4_pirn_backbone_matched4_20261001'
    unchanged=['v4/refinement.py','v4/inference.py','pirn_v4/inference.py','physical_bridge/simulator.py','pirn_v4/model.py','protocol/common_init.pt','protocol/stream.json']
    for rel in unchanged:assert sha256(ROOT/rel)==sha256(parent/rel),rel
    for mod in ('train_worker','run_formal','stage_report','refinement_ablation','evaluate_endpoint_train','report_supplement','freeze'):
        __import__(mod)
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mutex.py','-v'],cwd=ROOT,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'deps')))
    (ROOT/'reports/CPU_MUTEX.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:raise RuntimeError('CPU mutex tests failed; see CPU_MUTEX.log')
    save_json(ROOT/'reports/CPU_ACCEPTANCE.json',{'status':'PASS','formal_updates':0,'temporary_optimizer_calls':0,'LR_formula_endpoints_and_budget':True,'MAE_distinct_from_metric_MSE':True,'common_U0_hash':common['sha256'],'unchanged_parent_files':{rel:sha256(ROOT/rel) for rel in unchanged},'four_mutex_negative_cases_PASS':True,'GPU_gates_still_required':True,'time':time.time()})
    print('CPU_ACCEPTANCE_PASS_GPU_GATES_STILL_REQUIRED',flush=True)
if __name__=='__main__':main()

"""Real CUDA revision gates: exact state reload, migration, fixed inference and refinement.
Uses independent loaded copies. Does not call optimizer.step or alter formal weights.
"""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import numpy as np
import torch
from v4.matched import Store,build,eta_batch,seed_for,SEED,lr_at
from v4.runtime import configure_cuda,save_json,sha256
from v4.training import make_optimizer,make_ema,finite_gradients
from v4.performance import backward32
from v4.refinement import module_hash
from v4.inference import infer
from train_worker import checkpoint,capture_rng,restore_rng
from validate_group import tree_equal
from metrics import Metrics


def main(a):
    configure_cuda(20260929);cfg=json.loads((ROOT/'configs/runtime.json').read_text());store=Store(cfg)
    work=ROOT/'reports/execution_revision'/a.group;work.mkdir(parents=True,exist_ok=True)
    report={'group':a.group,'gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'optimizer_updates':0,'status':'RUNNING'}
    parent=ROOT.parent/'v4_train500_matched_v2';run=ROOT.parents[1]/'runs/v4_train500_matched_v2'
    model,sim,obj=build(cfg,a.group);opt=make_optimizer(model);ema=make_ema(model)
    # Actual complete U0 formal checkpoint identity and exact full state reload.
    path=run/a.group/'checkpoints/full_U00000.pt';identity=json.loads(path.with_suffix('.identity.json').read_text());assert sha256(path)==identity['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['update']==saved['stream_position']==0 and saved['scheduler']['next_lr']==lr_at(1)
    model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    tree_equal(saved['model'],model.state_dict());tree_equal(saved['ema'],ema.state_dict());tree_equal(saved['optimizer'],opt.state_dict());tree_equal(saved['rng'],capture_rng())
    report['formal_U0_state_exact']=True;del saved
    # Actual populated historical temporary optimizer state, with no additional updates.
    temporary=torch.load(parent/f'reports/temporary_{a.group}_step1.pt',map_location='cpu',weights_only=False)
    opt.load_state_dict(temporary['optimizer']);del temporary
    before=capture_rng();frozen={'protocol_hash':'EXECUTION_REVISION_ACCEPTANCE_ONLY','stream_sha256':sha256(ROOT/'protocol/stream.json'),'execution_hash':'temporary'}
    statefile=work/f'TEMPORARY_gpu{a.gpu}_state.pt'
    identity=checkpoint(statefile,model,ema,opt,1,frozen,a.group,purpose='TEMPORARY_EXECUTION_REVISION_NOT_FORMAL')
    loaded=torch.load(statefile,map_location='cpu',weights_only=False)
    tree_equal(loaded['optimizer'],opt.state_dict());restore_rng(loaded['rng']);tree_equal(before,capture_rng())
    report['populated_optimizer_RNG_roundtrip']=True;del loaded
    stream=json.loads((ROOT/'protocol/stream.json').read_text());batch=[store.by_id[s] for s in stream['updates'][1]]
    opt.zero_grad(set_to_none=True);backward32(obj,store,batch,2,1);finite_gradients(model)
    gradients=torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None]).numpy()
    gradpath=work/f'gradient_gpu{a.gpu}.npy';np.save(gradpath,gradients,allow_pickle=False)
    opt.zero_grad(set_to_none=True);model.eval();obj.eval()
    values=[];off_max=0.;seeds_all=[];raw_hashes=[];metrics=Metrics('cuda')
    fixed=store.val[:4];by={r['start_id']:r for r in [json.loads(s) for s in (run/a.group/'evaluations/U00000/raw/val800/per_start.jsonl').read_text().splitlines()]}
    all_g=sorted(r['G'] for r in fixed);assert all_g==[1,2,3,4]
    for offset in range(0,4,2):
        chunk=fixed[offset:offset+2];C,Y,G=store.batch(chunk,'cuda');seeds=[seed_for(f'eval|{SEED}|{r["start_id"]}') for r in chunk]
        with torch.no_grad():
            direct=obj(C,Y,G,eta=eta_batch(chunk,'cuda'))
            # D0 is one prediction; this inference function belongs only to recursive methods.
            if a.group in ['R0','R08']:
                off=infer(model,sim,Y,G,noise_seeds=seeds,T0=0. if a.group=='R0' else .8,refinement=False)
                if a.group in ['R0','R08']:
                    torch.testing.assert_close(off['clean'],direct['final_clean'],rtol=2e-4,atol=2e-5)
                    off_max=max(off_max,float((off['clean']-direct['final_clean']).abs().max()))
            v=metrics(direct['final_clean'],C)
        for idx,row in enumerate(chunk):
            for key in ['psnr','ssim','lpips']:assert abs(v[idx][key]-by[row['start_id']][key])<1e-5
        values.append(direct['final_clean'].detach().cpu().numpy());del C,Y,G,direct
    np.save(work/f'output_gpu{a.gpu}.npy',np.concatenate(values),allow_pickle=False)
    report.update(evaluation_matches_original_U0=True,off_inference_parity_max_abs=off_max)
    # Only run the prescribed serial refinement once per arm, on its primary diagnostic GPU.
    if a.refine and a.group in ['R0','R08']:
        initial=(module_hash(model),module_hash(sim));times=[];stage_count=0
        for offset in range(0,4,2):
            chunk=fixed[offset:offset+2];C,Y,G=store.batch(chunk,'cuda');seeds=[seed_for(f'eval|{SEED}|{r["start_id"]}') for r in chunk]
            result=infer(model,sim,Y,G,noise_seeds=seeds,T0=0. if a.group=='R0' else .8,refinement=True,retain_trace=True)
            assert result['refinement_sample_count']==sum(r['G']-1 for r in chunk)
            for tr in result['trace']:
                assert all(tr['g'][j]>1 for j,idx in enumerate(tr['active']) if idx in tr['refined_original_indices'])
            for ref in result['refinements']:
                assert len(ref.candidates)==41 and len(ref.trace)==40 and all(ref.invariance['unchanged'].values())
                totals=np.array([r['total'] for r in ref.candidates]);assert np.allclose(totals.min(0),ref.best_objective,atol=0,rtol=0)
                assert np.array_equal(totals.argmin(0),ref.best_iterations)
                assert ref.optimizer['execution']=='per_sample_serial'
                assert all(v['state_entries_at_start']==0 and v['state_steps_at_end']==[40,40] for v in ref.optimizer['per_sample'])
            times.append(result['refinement_seconds']);stage_count+=result['refinement_sample_count'];del C,Y,G,result
        assert (module_hash(model),module_hash(sim))==initial
        report['refinement']={'status':'PASS','G':[1,2,3,4],'native_stage_samples':stage_count,
          'solver_seconds':sum(times),'seconds_per_native_stage_sample':sum(times)/stage_count,
          'estimated_val800_solver_seconds':sum(times)/stage_count*1200,'estimate_not_trained_checkpoint_measurement':True,
          '41_candidates_total_J_best':True,'per_sample_serial':True,'fresh_optimizers':True,'no_terminal_refinement':True,
          'fixed_network_sim_clean_center_verified':True}
    report['status']='PASS';save_json(work/f'gpu{a.gpu}.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--refine',action='store_true');a=p.parse_args()
    try:main(a)
    except BaseException as e:
        save_json(ROOT/'reports/RUN_PAUSED.json',{'gate':'execution_revision_correctness','error':repr(e),'group':a.group,'traceback':traceback.format_exc(),'time':time.time()});raise

"""Auxiliary RAW val800 inference-only refinement. Main OFF files are read-only."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import numpy as np
import torch
from v4.matched import Store,build,seed_for,SEED
from v4.inference import infer
from v4.refinement import module_hash
from v4.runtime import configure_cuda,save_json,sha256
from v4.evaluate_matched import output_digest,aggregates
from metrics import Metrics
from train_worker import verify_frozen


def paired_rows(on,off):
    index={r['start_id']:r for r in off}
    assert len(on)==len(off)==800 and len(index)==800
    assert {r['start_id'] for r in on}==set(index)
    sources=sorted({r['source_id'] for r in off});assert len(sources)==100
    matrix=np.random.Generator(np.random.PCG64(20260929)).integers(0,100,(10000,100))
    rows=[]
    for metric in ['psnr','ssim','lpips']:
        by=defaultdict(list)
        for r in on:
            b=index[r['start_id']]
            assert all(r[k]==b[k] for k in ['source_id','trajectory_id','G','split'])
            by[r['source_id']].append(r[metric]-b[metric])
        assert all(len(by[s])==8 for s in sources)
        values=np.array([np.mean(by[s]) for s in sources]);bs=values[matrix].mean(1)
        lo,hi=np.percentile(bs,[2.5,97.5],method='linear')
        wins=-values if metric=='lpips' else values;tol=1e-6 if metric=='psnr' else 0.
        rows.append({'metric':metric,'contrast':'refinement_ON-minus-OFF','mean_paired_delta':float(values.mean()),
                     'source_win_rate':float(np.mean(wins>tol)),'CI95_low':float(lo),'CI95_high':float(hi),
                     'win_direction':'lower' if metric=='lpips' else 'higher','source_count':100,
                     'start_count':800,'bootstrap_resamples':10000,'bootstrap_seed':20260929,
                     'CI_scope':'source variability, not training seed or stochastic refinement variability'})
    return rows,matrix


def main(a):
    if a.group not in ['R0','R08']:raise ValueError('D0/L08 refinement is N/A')
    frozen=verify_frozen();configure_cuda(20260929)
    run=Path(frozen['results_root']);cfg=json.loads((ROOT/'configs/runtime.json').read_text())
    store=Store(cfg);rows=store.val;offdir=run/a.group/'evaluations'/f'U{a.update:05d}'/'raw/val800'
    offcommit=json.loads((offdir/'COMMIT.json').read_text())
    for rel,h in offcommit['file_sha256'].items():assert sha256(offdir/rel)==h
    off=[json.loads(s) for s in (offdir/'per_start.jsonl').read_text().splitlines()]
    weights=run/a.group/'checkpoints'/f'weights_U{a.update:05d}.pt';assert sha256(weights)==offcommit['checkpoint_hash']
    model,sim,obj=build(cfg,a.group);saved=torch.load(weights,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);del saved,obj
    model.eval().requires_grad_(False);sim.eval();before=(module_hash(model),module_hash(sim))
    out=run/'refinement_ablation'/f'U{a.update:05d}'/a.group
    if (out/'COMMIT.json').exists():
        commit=json.loads((out/'COMMIT.json').read_text())
        assert commit['checkpoint_hash']==offcommit['checkpoint_hash']
        for rel,h in commit['file_sha256'].items():assert sha256(out/rel)==h
        return
    out.mkdir(parents=True,exist_ok=True);(out/'batches').mkdir(exist_ok=True)
    metrics=Metrics('cuda');all_rows=[];inference_seconds=0.;metric_seconds=0.;refinement_seconds=0.;begin=time.monotonic()
    for offset in range(0,800,2):
        batchfile=out/'batches'/f'{offset:04d}.json'
        batch=rows[offset:offset+2]
        if batchfile.exists():
            old=json.loads(batchfile.read_text());assert old['start_ids']==[r['start_id'] for r in batch]
            assert old['checkpoint_hash']==offcommit['checkpoint_hash']
            for rel,h in old['fixed_sha256'].items():assert sha256(out/rel)==h
            all_rows.extend(old['rows']);inference_seconds+=old['inference_seconds'];metric_seconds+=old['metric_seconds'];refinement_seconds+=old['refinement_seconds'];continue
        C,Y,G=store.batch(batch,'cuda');seeds=[seed_for(f'eval|{SEED}|{r["start_id"]}') for r in batch]
        torch.cuda.synchronize();start=time.monotonic()
        result=infer(model,sim,Y,G,noise_seeds=seeds,T0=0. if a.group=='R0' else .8,
                     refinement=True,retain_trace=True,audit_refinement=False)
        torch.cuda.synchronize();seconds=time.monotonic()-start;start=time.monotonic()
        with torch.no_grad():values=metrics(result['clean'],C)
        metric_elapsed=time.monotonic()-start
        entries=[];fixed={}
        for j,r in enumerate(batch):
            digest,array=output_digest(result['clean'][j]);entry={**{k:r[k] for k in ['start_id','source_id','trajectory_id','split','G']},
                **values[j],'output_hash':digest,'checkpoint_hash':offcommit['checkpoint_hash'],'update':a.update,
                'method':a.group,'weights':'raw','refinement':'ON_per_sample_serial','eta_seed':seeds[j]}
            entries.append(entry)
            if r['start_id'] in store.visual_ids:
                d=out/'fixed_fp32';d.mkdir(exist_ok=True);file=d/(r['start_id'].replace('::','_')+'_output.npy')
                np.save(file,array,allow_pickle=False);fixed[str(file.relative_to(out))]=sha256(file)
        diagnostic=[{'initial_objective':v.initial_objective,'best_objective':v.best_objective,
                     'best_iterations':v.best_iterations,'candidates':v.candidates,'optimizer':v.optimizer,
                     'unchanged':v.invariance['unchanged']} for v in result['refinements']]
        assert len(result['trace'])==int(G.max())
        assert result['refinement_sample_count']==sum(r['G']-1 for r in batch)
        save_json(batchfile,{'start_ids':[r['start_id'] for r in batch],'checkpoint_hash':offcommit['checkpoint_hash'],
                  'rows':entries,'inference_seconds':seconds,'refinement_seconds':result['refinement_seconds'],'metric_seconds':metric_elapsed,'fixed_sha256':fixed,
                  'stage_trace':result['trace'],'refinement_audit':diagnostic})
        all_rows.extend(entries);inference_seconds+=seconds;metric_seconds+=metric_elapsed;refinement_seconds+=result['refinement_seconds']
        save_json(out/'PROGRESS.json',{'completed_starts':len(all_rows),'inference_seconds':inference_seconds,
                  'metric_seconds':metric_seconds,'eta_remaining_seconds':(800-len(all_rows))*(inference_seconds+metric_seconds)/len(all_rows)})
        del C,Y,G,result,values
    assert (module_hash(model),module_hash(sim))==before
    comparison,matrix=paired_rows(all_rows,off);np.save(out/'bootstrap_source_indices.npy',matrix,allow_pickle=False)
    (out/'per_start.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in all_rows))
    offagg=json.loads((offdir/'aggregate.json').read_text())
    save_json(out/'SUMMARY.json',{'group':a.group,'update':a.update,'weights':'raw','OFF':{k:offagg['overall'][k] for k in ['psnr','ssim','lpips']},
              'ON':aggregates(all_rows),'paired':comparison,'off_reused_commit_hash':sha256(offdir/'COMMIT.json'),
              'runtime':{'refinement_solver_seconds':refinement_seconds,'refined_stage_sample_count':1200,'seconds_per_refined_stage_sample':refinement_seconds/1200,'ON_inference_seconds':inference_seconds,'ON_metric_seconds':metric_seconds,
                 'OFF_original_evaluation_seconds':offagg['elapsed_seconds'],'total_this_invocation_seconds':time.monotonic()-begin,
                 'note':'ON inference includes network and serial refinement; OFF timing also includes original metrics/I-O, so difference is not a pure solver microbenchmark'},
              'main_tables_unchanged':True,'best_val_selection_unchanged':True,'network_simulator_unchanged':True})
    save_json(out/'COMMIT.json',{'checkpoint_hash':offcommit['checkpoint_hash'],'protocol_hash':frozen['protocol_hash'],
              'execution_hash':frozen['execution_hash'],'refinement_spec_sha256':sha256(ROOT/'protocol/REFINEMENT_ABLATION.json'),
              'file_sha256':{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='COMMIT.json' and p.suffix!='.log'}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--update',type=int,choices=[5000,7500,10000],required=True);a=p.parse_args()
    try:main(a)
    except BaseException as e:
        save_json(ROOT/'reports/RUN_PAUSED.json',{'gate':'inference_only_refinement','group':a.group,'update':a.update,
                  'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

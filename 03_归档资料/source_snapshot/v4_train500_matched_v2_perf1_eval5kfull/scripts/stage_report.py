"""Read committed evaluations only. No inference, training, or checkpoint selection beyond protocol."""
import argparse,csv,hashlib,importlib.util,json,os,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4.runtime import save_json,sha256
from v4.stages import GROUPS,NODES,verify_budget_scope
_spec=importlib.util.spec_from_file_location('plan_rules',ROOT/'protocol/check_training_plan.py')
rules=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(rules)


def write_table(directory,name,rows):
    save_json(directory/f'{name}.json',rows)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (directory/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,columns);w.writeheader();w.writerows(rows)
    def display(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
    (directory/f'{name}.md').write_text('| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(display(row.get(k,'')) for k in columns)+' |\n' for row in rows))


def load_eval(run,group,u,weights,pop):
    path=run/group/'evaluations'/f'U{u:05d}'/weights/pop
    commit=json.loads((path/'COMMIT.json').read_text())
    for name in ['aggregate.json','per_start.jsonl','baseline_per_start.json']:
        if sha256(path/name)!=commit['file_sha256'][name]:raise ValueError('Evaluation identity mismatch')
    return json.loads((path/'aggregate.json').read_text()),path


def paired_table(rows_by_group,seed=20260929):
    pairs=[('R0','D0'),('R08','R0'),('R08','D0'),('L08','D0'),('R08','L08')]
    index={g:{r['start_id']:r for r in rows} for g,rows in rows_by_group.items()}
    ids=sorted(index['D0']);assert len(ids)==800
    assert all(set(v)==set(ids) for v in index.values())
    sources=sorted({index['D0'][i]['source_id'] for i in ids});assert len(sources)==100
    matrix=np.random.Generator(np.random.PCG64(seed)).integers(0,100,size=(10000,100))
    matrix_hash=hashlib.sha256(np.ascontiguousarray(matrix,dtype='<i8').tobytes()).hexdigest()
    table=[]
    for a,b in pairs:
        by_source=defaultdict(list)
        for sid in ids:
            x,y=index[a][sid],index[b][sid]
            assert all(x[k]==y[k] for k in ['source_id','trajectory_id','G','split'])
            by_source[x['source_id']].append(x['psnr']-y['psnr'])
        assert all(len(by_source[s])==8 for s in sources)
        values=np.array([np.mean(by_source[s]) for s in sources],dtype=np.float64)
        bootstrap=values[matrix].mean(1);low,high=np.percentile(bootstrap,[2.5,97.5],method='linear')
        table.append({'contrast':f'{a}-{b}','mean_paired_delta_PSNR':float(values.mean()),
                      'source_win_rate':float(np.mean(values>1e-6)),'wins':int(np.sum(values>1e-6)),
                      'ties':int(np.sum(np.abs(values)<=1e-6)),'losses':int(np.sum(values<-1e-6)),
                      'CI95_low':float(low),'CI95_high':float(high),'source_count':100,'start_count':800,
                      'bootstrap_index_hash':matrix_hash})
    return table,matrix


def progression_table(rows_by_group):
    table=[]
    for method in ['R0','R08','L08']:
        parts=[]
        for g in [2,3,4]:
            rows=[r for r in rows_by_group[method] if r['G']==g]
            sources=defaultdict(list)
            for r in rows:sources[r['source_id']].append(r)
            assert len(rows)==200 and len(sources)==100 and all(len(v)==2 for v in sources.values())
            first=float(np.mean([np.mean([r['psnr_first'] for r in v]) for v in sources.values()]))
            final=float(np.mean([np.mean([r['psnr'] for r in v]) for v in sources.values()]))
            delta=float(np.mean([np.mean([r['delta_prog_PSNR'] for r in v]) for v in sources.values()]))
            assert abs(delta-(final-first))<1e-10
            entry={'method':method,'G':g,'first_PSNR':first,'final_PSNR':final,'delta_PSNR':delta,
                   'source_count':100,'start_count':200}
            parts.append(entry);table.append(entry)
        table.append({'method':method,'G':'Overall_G2to4',
                      **{k:float(np.mean([r[k] for r in parts])) for k in ['first_PSNR','final_PSNR','delta_PSNR']},
                      'source_count':100,'start_count':600})
    return table


def plots(out,curves,prog,update):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    colors={'D0':'#555555','R0':'#0072B2','R08':'#D55E00','L08':'#009E73'}
    def save(fig,name):
        fig.savefig(out/f'{name}.png',bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    def axes_label(ax):
        ax.set_xlim(0,max(update,250))
        ax.set_xlabel('Optimizer updates (epoch=U/250; exposures=32U)');ax.grid(alpha=.2)
        sec=ax.secondary_xaxis('top',functions=(lambda u:u/250,lambda e:e*250));sec.set_xlabel('Epoch')
    for weights in ['raw']:
        data=[r for r in curves if r['weights']==weights]
        for metric in ['psnr','ssim','lpips']:
            fig,axs=plt.subplots(1,2,figsize=(12,4.8))
            for ax,pop in zip(axs,['train_monitor400','val800']):
                for group in GROUPS:
                    rows=[r for r in data if r['population']==pop and r['method']==group]
                    ax.plot([r['update'] for r in rows],[r[metric] for r in rows],label=group,color=colors[group],marker='.',markersize=3)
                base=next(r[f'input_{metric}'] for r in data if r['population']==pop)
                ax.axhline(base,color='gray',ls='--',label='Input Y');ax.set_title(pop);ax.set_ylabel(metric.upper()+(' (dB)' if metric=='psnr' else ''));axes_label(ax);ax.legend()
            fig.suptitle(f'V4 {weights.upper()} | refinement OFF | common U={update}')
            fig.tight_layout();name='train_val_psnr_curve' if metric=='psnr' and weights=='raw' else f'train_val_{metric}_{weights}_curve';save(fig,name)
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        for ax,(pop,metric) in zip(axs.flat,[(p,m) for p in ['train_monitor400','val800'] for m in ['path_total','path_clean_MSE','path_weighted_reconstruction_MSE']]):
            for group in GROUPS:
                rows=[r for r in data if r['population']==pop and r['method']==group]
                ax.plot([r['update'] for r in rows],[r[metric] for r in rows],label=group,color=colors[group])
            ax.set_title(pop+' | '+{'path_total':'L_total','path_clean_MSE':'L_C','path_weighted_reconstruction_MSE':'0.2 L_R'}[metric]);axes_label(ax);ax.set_yscale('log');ax.legend()
        fig.suptitle(f'V4 {weights.upper()} | same-checkpoint evaluation losses | U={update}');fig.tight_layout();save(fig,'train_val_loss_curve' if weights=='raw' else 'train_val_loss_EMA_curve')
    fig,ax=plt.subplots(figsize=(8,5))
    for group in ['R0','R08','L08']:
        rows=[r for r in prog if r['method']==group and r['G']!='Overall_G2to4']
        ax.plot([r['G'] for r in rows],[r['delta_PSNR'] for r in rows],marker='o',label=group,color=colors[group])
    ax.axhline(0,color='gray',ls='--');ax.set_xticks([2,3,4]);ax.set_xlabel('Original severity G');ax.set_ylabel('PSNR(final) - PSNR(first), dB');ax.set_title(f'Val800 | RAW | U={update} | refinement OFF');ax.legend();ax.grid(alpha=.2);save(fig,'progression_by_G')
    write_table(out,'curve_plot_data',curves)


def qualitative(out,run,update,train_population='train8000'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    planned=json.loads((ROOT/'protocol/PLANNED_MONITOR_LISTS.json').read_text())
    folder=out/'qualitative';folder.mkdir()
    for weights in ['raw']:
        for row in planned['visual_train20']+planned['visual_val20']:
            pop=train_population if row['split']=='train' else 'val800';stem=row['start_id'].replace('::','_')
            roots={g:run/g/'evaluations'/f'U{update:05d}'/weights/pop/'fixed_fp32' for g in GROUPS}
            arrays=[np.load(roots['D0']/f'{stem}_clean.npy'),np.load(roots['D0']/f'{stem}_input.npy')]
            arrays.extend(np.load(roots[g]/f'{stem}_output.npy') for g in GROUPS)
            title=f'{row["source_id"]} | {row["split"]} | G={row["G"]} | U={update} | {weights.upper()}'
            for layout in ['horizontal','vertical']:
                fig,axs=plt.subplots(1,6,figsize=(18,3.8)) if layout=='horizontal' else plt.subplots(6,1,figsize=(4.2,20))
                for ax,array,label in zip(np.ravel(axs),arrays,['Clean','Input Y',*GROUPS]):
                    ax.imshow(np.moveaxis(array,0,2).clip(0,1));ax.set_title(label);ax.axis('off')
                fig.suptitle(title,fontsize=9);fig.tight_layout();fig.savefig(folder/f'{stem}_{weights}_{layout}.png',dpi=130,bbox_inches='tight');plt.close(fig)


def report(run,update):
    verify_budget_scope(ROOT);assert update in NODES
    out=run/'stage_deliveries'/f'U{update:05d}'
    if (out/'DELIVERY_MANIFEST.json').exists():return out/'DELIVERY_MANIFEST.json'
    if out.exists():raise RuntimeError('Partial stage delivery exists; review before replay')
    out.mkdir(parents=True)
    spec=json.loads((ROOT/'protocol/TRAINING_PLAN_SPEC.json').read_text())
    nodes=list(range(0,update+1,250));curves=[];main=[];best=[];rows_by_group={};inputs=[]
    for group in GROUPS:
        full=run/group/'checkpoints'/f'full_U{update:05d}.pt'
        identity=json.loads(full.with_suffix('.identity.json').read_text());assert sha256(full)==identity['sha256']
        for u in nodes:
            for weights in ['raw']:
                for pop in ['train_monitor400','val800']:
                    aggregate,path=load_eval(run,group,u,weights,pop)
                    curves.append({'method':group,'update':u,'epoch':u/250,'exposures':32*u,'weights':weights,'population':pop,
                        **aggregate['overall'],**{f'input_{k}':v for k,v in aggregate['baseline']['overall'].items()}})
                    inputs.append({'path':str(path/'COMMIT.json'),'sha256':sha256(path/'COMMIT.json')})
        raw,path=load_eval(run,group,update,'raw','val800')
        main.append({'panel':'A_matched_budget','method':group,'update':update,'epoch':update/250,
                     **{k:raw['overall'][k] for k in ['psnr','ssim','lpips']},
                     'delta_PSNR_vs_Input':raw['overall']['psnr']-raw['baseline']['overall']['psnr'],'checkpoint_hash':raw['checkpoint_hash']})
        rows_by_group[group]=[json.loads(s) for s in (path/'per_start.jsonl').read_text().splitlines()]
        scores={r['update']:r['psnr'] for r in curves if r['method']==group and r['weights']=='raw' and r['population']=='val800'}
        chosen=rules.best_update(scores,nodes,1e-6);a,_=load_eval(run,group,chosen,'raw','val800')
        best.append({'panel':'B_best_validation_auxiliary','method':group,'best_update':chosen,'best_epoch':chosen/250,
                     **{k:a['overall'][k] for k in ['psnr','ssim','lpips']},
                     'delta_PSNR_vs_Input':a['overall']['psnr']-a['baseline']['overall']['psnr'],
                     'checkpoint_hash':a['checkpoint_hash'],'candidate_interval':f'[0,{update}]'})
        # Self-contained stage data snapshot; hardlink immutable large checkpoint, copy committed metrics/fixed floats.
        evidence=out/'evidence'/group;evidence.mkdir(parents=True)
        os.link(full,evidence/full.name);shutil.copy2(full.with_suffix('.identity.json'),evidence/full.with_suffix('.identity.json').name)
        shutil.copytree(run/group/'evaluations'/f'U{update:05d}',evidence/'evaluation')
        for name in ['train_steps.jsonl','timings.jsonl','TIMEBOOK.json','STATUS.json']:
            if (run/group/name).exists():shutil.copy2(run/group/name,evidence/name)
        for weights in ['raw']:
            for population in (['train8000','val800'] if update in (5000,10000) else ['train_monitor400','val800']):
                a,_=load_eval(run,group,update,weights,population)
                assert a['start_count']=={'train8000':8000,'train_monitor400':400,'val800':800}[population]
    write_table(out,'main_performance_table',main+best);write_table(out,'best_validation_capability_table',best)
    paired,matrix=paired_table(rows_by_group);write_table(out,'paired_ablation_table',paired);np.save(out/'bootstrap_source_indices.npy',matrix)
    prog=progression_table(rows_by_group);write_table(out,'progression_by_G_table',prog)
    plots(out,curves,prog,update);qualitative(out,run,update,train_population="train8000" if update==10000 else "train_monitor400")
    if update==10000:
        series={g:{r['update']//250:r['psnr'] for r in curves if r['method']==g and r['weights']=='raw' and r['population']=='val800'} for g in GROUPS}
        trigger=rules.continuation_review(series,GROUPS,spec['evaluation']['sufficiency_review'])
    else:trigger={'status':'NOT_APPLICABLE_REQUIRES_FULL_10K_CURVE','update':update,'window_not_shortened':True}
    save_json(out/'sufficiency_trigger.json',trigger)
    protocol=out/'protocol';protocol.mkdir()
    for path in (ROOT/'protocol').iterdir():
        if path.suffix=='.json':shutil.copy2(path,protocol/path.name)
    shutil.copytree(ROOT/'docs',out/'docs');shutil.copytree(ROOT/'configs',out/'configs')
    save_json(out/'INPUT_IDENTITIES.json',inputs)
    (out/'README.md').write_text(f'# Common update {update} stage delivery\n\nAll four methods at U={update}, epoch={update/250:g}, exposures={32*update} per method. RAW/refinement-OFF is primary; EMA weights are maintained in checkpoints, with no new default EMA evaluation. Intermediate training curves and samples use train-monitor400; unmeasured full-train8000 values are absent. No inference was rerun to make this package. All source/start metrics and fixed FP32 samples are retained in evidence. Display PNGs alone are clipped to [0,1], never used for metrics.\n\nBest-validation capability selects within the shared [0,U] interval and is subject to validation selection bias; it never replaces the common-U primary table. Full train/val evaluation curves are distinct from online optimization logs. Source-cluster intervals describe source variation, not training-seed uncertainty. This stage is not a convergence claim. Training remains on the original10k LR schedule.\n\nThe release decision is recorded separately by the coordinator. A user-requested early endpoint is USER_DECIDED_COMMON_BUDGET_TRUNCATION; the10k adequacy trigger is not applicable before10k. No automatic extension is authorized.\n')
    # Auxiliary inference results are separate and cannot enter main/best-val selection.
    fulltrain=[]
    for group in GROUPS:
        fullpath=run/group/'evaluations'/f'U{update:05d}'/'raw/train8000/aggregate.json'
        if fullpath.exists():
            row=json.loads(fullpath.read_text());fulltrain.append({'method':group,'update':update,'population':'full_train8000',**row['overall']})
    save_json(out/'full_train8000_measurements.json',{'measured':fulltrain,'missing':[g for g in GROUPS if g not in {r['method'] for r in fulltrain}],'missing_values_not_filled':True})
    auxiliary=[]
    for group in ['R0','R08']:
        source=run/'refinement_ablation'/f'U{update:05d}'/group
        commit=json.loads((source/'COMMIT.json').read_text())
        for name,digest in commit['file_sha256'].items():
            assert sha256(source/name)==digest
        summary=json.loads((source/'SUMMARY.json').read_text())
        auxiliary.append(summary)
        shutil.copytree(source,out/'auxiliary_refinement'/group)
    save_json(out/'auxiliary_refinement_table.json',{'main_comparison_unchanged':True,'RAW_ONLY':True,
              'results':auxiliary,'not_applicable':{'D0':'No intermediate xi path','L08':'Linear state independent of xi'}})
    write_table(out,'auxiliary_refinement_paired_metrics',[
        {'method':s['group'],'update':update,**row,
         'OFF':s['OFF'][row['metric']], 'ON':s['ON']['overall'][row['metric']],
         'refinement_solver_seconds':s['runtime']['refinement_solver_seconds']}
        for s in auxiliary for row in s['paired']])
    with (out/'README.md').open('a') as file:
        file.write('\nAuxiliary RAW inference-only refinement ON/OFF for R0/R08 is in auxiliary_refinement/. OFF metrics are reused; these results do not enter main comparison, best-val selection or training. D0/L08 refinement is N/A. Solver timings are reported separately.\n')
    files={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file()}
    save_json(out/'DELIVERY_MANIFEST.json',{'status':'COMPLETE_COMMON_STAGE','update':update,'epoch':update/250,
              'exposures_per_group':32*update,'groups':list(GROUPS),'file_sha256':files,'main_context':'RAW_refinement_OFF',
              'max_training_budget_unchanged':10000,'sufficiency_status':trigger['status'],'created_time':time.time()})
    return out/'DELIVERY_MANIFEST.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--update',type=int,choices=NODES,required=True);a=p.parse_args()
    print(report(a.run,a.update),flush=True)

"""CPU-only matched old/new comparisons and fixed refinement visuals from committed data."""
import json
from pathlib import Path
import numpy as np
from v4.runtime import sha256,save_json
ROOT=Path(__file__).resolve().parents[1]
GROUPS=('D0','R0','R08','L08');METRICS=('psnr','ssim','lpips')

def verified(path):
    commit=json.loads((path/'COMMIT.json').read_text())
    for name in ('aggregate.json','per_start.jsonl','baseline_per_start.json'):
        if name in commit['file_sha256']:assert sha256(path/name)==commit['file_sha256'][name]
    return json.loads((path/'aggregate.json').read_text())

def supplement(out,run,update,curves,best):
    from stage_report import write_table,rules
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    oldrun=run.parent/'v4_train500_matched_v2'
    diagnosis=ROOT.parent/'GENERALIZATION_DIAGNOSIS_U1000_VS_U5000'
    data=[];gap=[];bestdata=[]
    def get(root,group,u,pop,isold=False):
        path=root/group/'evaluations'/f'U{u:05d}'/'raw'/pop
        if (path/'COMMIT.json').exists():return verified(path),str(path)
        if isold:
            path=diagnosis/'evidence'/f'U{u:05d}'/group/pop
            if (path/'COMMIT.json').exists():return verified(path),str(path)
        return None,None
    for u in (1000,2000):
        if u>update:continue
        for label,base in [('Old U-Net + MSE + old LR',oldrun),('New SwinIR + L1 + 2k LR',run)]:
            for group in GROUPS:
                val,vpath=get(base,group,u,'val800',label.startswith('Old'))
                train,tpath=get(base,group,u,'train8000',label.startswith('Old'))
                if val is None:raise ValueError(f'Missing required {label} {group} U{u} val800')
                entry={'configuration':label,'method':group,'update':u,**{m:val['overall'][m] for m in METRICS},'delta_PSNR_vs_Input':val['overall']['psnr']-val['baseline']['overall']['psnr'],'source':vpath,'checkpoint_hash':val['checkpoint_hash']}
                data.append(entry)
                gap.append({'configuration':label,'method':group,'update':u,'train_population':'full_train8000',**{f'full_train_{m}':train['overall'][m] if train else None for m in METRICS},**{f'val_{m}':val['overall'][m] for m in METRICS},**{f'train_minus_val_{m}':train['overall'][m]-val['overall'][m] if train else None for m in METRICS},'train_source':tpath,'missing_fulltrain':'N/A; no monitor proxy' if train is None else ''})
        for label in sorted({r['configuration'] for r in data}):
            d0=next(r for r in data if r['configuration']==label and r['update']==u and r['method']=='D0')
            for row in data:
                if row['configuration']==label and row['update']==u:row['delta_PSNR_vs_D0']=row['psnr']-d0['psnr']
    for label,base in [('Old U-Net + MSE + old LR',oldrun),('New SwinIR + L1 + 2k LR',run)]:
        for group in GROUPS:
            nodes=list(range(0,update+1,250));scores={u:get(base,group,u,'val800',label.startswith('Old'))[0]['overall']['psnr'] for u in nodes}
            chosen=rules.best_update(scores,nodes,1e-6);val,src=get(base,group,chosen,'val800',label.startswith('Old'));train,tsrc=get(base,group,chosen,'train8000',label.startswith('Old'))
            bestdata.append({'configuration':label,'method':group,'selection_interval':f'[0,{update}]','best_update':chosen,**{m:val['overall'][m] for m in METRICS},'full_train_PSNR':train['overall']['psnr'] if train else None,'train_minus_val_PSNR':train['overall']['psnr']-val['overall']['psnr'] if train else None,'val_source':src,'train_source':tsrc,'selection':'same RAW/OFF PSNR checkpoint for all metrics; post-selection auxiliary'})
    write_table(out,'old_vs_new_matched_updates',data);write_table(out,'train_val_generalization_gap',gap);write_table(out,'old_vs_new_common_grid_best',bestdata)
    # Compact presentation tables retain CSV/Markdown and add PNG without changing evidence.
    for name in ('main_performance_table','paired_ablation_table','auxiliary_refinement_paired_metrics'):
        rows=json.loads((out/(name+'.json')).read_text())
        columns={'main_performance_table':['panel','method','update','best_update','psnr','ssim','lpips','delta_PSNR_vs_Input'],'paired_ablation_table':['contrast','mean_paired_delta_PSNR','source_win_rate','CI95_low','CI95_high'],'auxiliary_refinement_paired_metrics':['method','update','metric','OFF','ON','mean_paired_delta','refinement_solver_seconds']}[name]
        fig,ax=plt.subplots(figsize=(15,max(2,.45*len(rows)+1.3)));ax.axis('off')
        def fmt(x):return '—' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
        table=ax.table(cellText=[[fmt(r.get(c)) for c in columns] for r in rows],colLabels=columns,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.6);ax.set_title(name.replace('_',' ')+' | RAW | U='+str(update));fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    planned=json.loads((ROOT/'protocol/PLANNED_MONITOR_LISTS.json').read_text())['visual_val20']
    fixed=next(r for r in planned if r['G']==4);stem=fixed['start_id'].replace('::','_')
    fig,axs=plt.subplots(1,6,figsize=(18,3.6));d0=run/'D0/evaluations'/f'U{update:05d}'/'raw/val800/fixed_fp32'
    paths=[d0/f'{stem}_clean.npy',d0/f'{stem}_input.npy']
    for g in ('R0','R08'):
        paths.extend([run/g/'evaluations'/f'U{update:05d}'/'raw/val800/fixed_fp32'/f'{stem}_output.npy',run/'refinement_ablation'/f'U{update:05d}'/g/'fixed_fp32'/f'{stem}_output.npy'])
    for ax,path,label in zip(axs,paths,['Clean','Input Y','R0 OFF','R0 ON','R08 OFF','R08 ON']):
        ax.imshow(np.load(path).transpose(1,2,0).clip(0,1));ax.set_title(label);ax.axis('off')
    fig.suptitle(f'Frozen val sample {fixed["start_id"]} | G4 | U{update}');fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'refinement_OFF_ON_qualitative.png',dpi=150,bbox_inches='tight');plt.close(fig)
    save_json(out/'refinement_visual_identity.json',{'frozen_row':fixed,'inputs':{str(p):sha256(p) for p in paths}})
    (out/'OLD_VS_NEW.md').write_text('# Matched configuration comparison\n\nOld: U-Net + MSE + old LR. New: SwinIR + L1 + independent 2k LR. Only the joint configuration can be attributed; architecture/loss/schedule are confounded. MSE and MAE loss magnitudes are not directly comparable. All main comparisons are RAW/refinement OFF, matched U1000/U2000 and a shared checkpoint-selection grid [0,U]. The U1000 stage reports [0,1000]; final U2000 reports [0,2000]. U5000 is historical background only. Best validation performance is subject to selection bias; the checkpoint is selected once by PSNR and its SSIM/LPIPS are reported together. Missing historical full-train values stay N/A; monitor400 is never substituted. Table gaps use signed Train minus Val and must be read with the metric direction (LPIPS lower is better). A 2k endpoint is not a convergence/sufficiency claim.\n')

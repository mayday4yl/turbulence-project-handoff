"""Source-paired reports from immutable per-start evidence, never selects training."""
from pathlib import Path
from collections import defaultdict
import argparse,json,math,statistics,time,zipfile
import numpy as np
from v031.runtime import save_json,sha256

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data2/yanglin/v031_overfit100_g4321_n4')
REPORTS=ROOT/'reports'

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save_rows(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in x))
def stats(values):
    good=[x for x in values if x is not None and math.isfinite(x)]
    return {'n':len(values),'finite_n':len(good),'mean':statistics.mean(good) if good else None,
        'median':statistics.median(good) if good else None,'wins':sum(x>1e-6 for x in good),
        'ties':sum(abs(x)<=1e-6 for x in good),'losses':sum(x< -1e-6 for x in good)}
def deltas(a,b):
    assert set(a)==set(b),'Paired populations differ'
    pairs=[]
    for sid,r in a.items():
        if sid not in b:raise RuntimeError('Missing paired baseline '+sid)
        assert r['source_id']==b[sid]['source_id'] and r['metrics'][0]['G']==b[sid]['metrics'][0]['G']
        if 'trajectory_id' in r and 'trajectory_id' in b[sid]:assert r['trajectory_id']==b[sid]['trajectory_id']
        v=r['metrics'][-1]['PSNR'];u=b[sid]['metrics'][-1]['PSNR']
        pairs.append({'start_id':sid,'source_id':r['source_id'],'G':r['metrics'][0]['G'],
            'PSNR_delta':None if v is None or u is None else v-u,'MSE_delta':r['metrics'][-1]['MSE']-b[sid]['metrics'][-1]['MSE']})
    grouped={str(g):stats([r['PSNR_delta'] for r in pairs if r['G']==g]) for g in (4,3,2,1) if any(r['G']==g for r in pairs)}
    source=defaultdict(list)
    for r in pairs:source[(r['source_id'],r['G'])].append(r['PSNR_delta'])
    per_source=[{'source_id':s,'G':g,'PSNR_delta':statistics.mean(v) if all(x is not None for x in v) else None} for (s,g),v in source.items()]
    return {'per_G':grouped,'source_per_G':{str(g):stats([r['PSNR_delta'] for r in per_source if r['G']==g]) for g in (4,3,2,1) if any(r['G']==g for r in pairs)},
        'source_rows':per_source,'start_rows':pairs}
def load_results(folder):
    values=[read(p) for p in Path(folder).glob('*/result.json')]
    if any(r.get('status')!='complete' for r in values):raise RuntimeError('Failed result requires explicit missing-data report')
    return {r['start_id']:r for r in values}

def first_output(values):
    return {s:{**r,'metrics':[r['metrics'][0]]} for s,r in values.items()}

def input_output(values):
    return {s:{**r,'metrics':[{**r['input'],'G':r['metrics'][0]['G'],'k':0}]} for s,r in values.items()}

def metric_curves(values):
    curves=[]
    for g in (4,3,2,1):
        group=[r for r in values.values() if r['metrics'][0]['G']==g]
        if not group:continue
        for k in range(len(group[0]['metrics'])):
            curves.append({'G':g,'k':k+1,'n':len(group),
                'MSE_mean':statistics.mean(r['metrics'][k]['MSE'] for r in group),
                'PSNR':stats([r['metrics'][k]['PSNR'] for r in group]),
                'zero_MSE_n':sum(r['metrics'][k]['zero_MSE'] for r in group),
                'input_MSE_mean':statistics.mean(r['input']['MSE'] for r in group),
                'input_PSNR':stats([r['input']['PSNR'] for r in group]),
                'input_zero_MSE_n':sum(r['input']['zero_MSE'] for r in group),
                'LR_mean':statistics.mean(r['metrics'][k]['LR'] for r in group) if all(r['metrics'][k].get('LR') is not None for r in group) else None})
    return curves

def inference_cost(values):
    return {'paths':len(values),'path_seconds_sum':sum(r['seconds'] for r in values.values()),
        'path_seconds':stats([r['seconds'] for r in values.values()]),
        'calls':{k:sum(r['calls'][k] for r in values.values()) for k in ('network','F','VJP')},
        'timing_scope':'sum of recorded per-path execution; excludes worker setup and is not parallel wall time; legacy timing not directly comparable'}

def comparisons(values,old,t1=None):
    result={'vs_old':deltas(values,old),'vs_old_K1':deltas(values,first_output(old)),
        'vs_own_K1':deltas(values,first_output(values)),'vs_input':deltas(values,input_output(values))}
    if t1 is not None:result['vs_T1_K1']=deltas(values,first_output(t1))
    return result

def i1_report():
    folder=ROOT/'I1_REFINE_REPAIR';choice=read(folder/'selection.json')
    data={c:load_results(folder/'dev'/c) for c in ['U0','R_legacy']+[r['candidate']['name'] for r in choice['ranking']]}
    assert len(data)==14 and all(len(v)==24 for v in data.values())
    base=data['U0'];summaries={};allrows=[]
    for name,items in data.items():
        for r in items.values():allrows.append(r)
        transitions=[t for r in items.values() for t in r['transitions']]
        comparison=deltas(items,base)
        summaries[name]={'paired_vs_U0':comparison,'seconds':sum(r['seconds'] for r in items.values()),
            'calls':{k:sum(r['calls'][k] for r in items.values()) for k in ('network','F','VJP')},
            'transitions':len(transitions),'fallback':sum(bool(t['fallback']) for t in transitions),
            'zero_update':sum(bool(t['zero_update']) for t in transitions),
            'accepted_proposals':sum(t['accepted_proposals'] for t in transitions)}
    save_rows(REPORTS/'I1_all_dev_paths.jsonl',allrows)
    step_pairs=[]
    for name,items in data.items():
        for sid,r in items.items():
            assert len(r['metrics'])==len(base[sid]['metrics'])
            for v,u in zip(r['metrics'],base[sid]['metrics']):
                assert v['k']==u['k'] and v['G']==u['G']
                step_pairs.append({'candidate':name,'start_id':sid,'source_id':r['source_id'],'G':v['G'],'k':v['k'],
                    'MSE':v['MSE'],'PSNR':v['PSNR'],'MSE_delta_vs_U0':v['MSE']-u['MSE'],
                    'PSNR_delta_vs_U0':None if v['PSNR'] is None or u['PSNR'] is None else v['PSNR']-u['PSNR']})
    assert len(step_pairs)==1008
    save_rows(REPORTS/'I1_each_K_paired_vs_U0.jsonl',step_pairs)
    save_json(REPORTS/'I1_paired_summary.json',summaries)
    diagnostics=[read(p) for p in folder.glob('diagnostics/diagnostic/*/result.json')]
    assert len(diagnostics)==8
    save_rows(REPORTS/'I1_firsthop_diagnostics.jsonl',diagnostics)
    directions=[{'source_id':d['source_id'],**d['direction']} for d in diagnostics if 'direction' in d]
    assert len(directions)==4
    save_json(REPORTS/'I1_gradient_directions.json',directions)
    firsthop=[]
    for label in diagnostics[0]['runs']:
        nmax=diagnostics[0]['runs'][label]['trace'][-1]['n']
        for n in (0,1,5,10,20,40):
            if n>nmax:continue
            group=[next(t for t in d['runs'][label]['trace'] if t['n']==n) for d in diagnostics]
            gains=[g['best']['U_vs_GT']['PSNR']-d['runs'][label]['trace'][0]['best']['U_vs_GT']['PSNR'] for g,d in zip(group,diagnostics)]
            firsthop.append({'run':label,'n':n,'J_best_mean':statistics.mean(x['J_best'] for x in group),'U_best_vs_initial_PSNR':stats(gains)})
    save_json(REPORTS/'I1_firsthop_summary.json',firsthop)
    lines=['# I1 有限修复实验结果','',f"状态：`{choice['status']}`。开发集 8 个 source、24 个起点，14 个配置、336 条完整路径均有结果。",'',
        '所有结果固定旧 V0.31 的 raw1000；网络和 simulator 参数未更新。每个候选使用统一配置，GT 仅用于独立诊断和开发集统一选择。','',
        '| 候选 | ΔG4 dB | ΔG3 dB | ΔG2 dB | A dB | 通过门槛 |','|---|---:|---:|---:|---:|---|']
    for x in choice['ranking']:
        lines.append('| '+x['candidate']['name']+' | '+' | '.join(f"{x['Delta_G'][str(g)]:.6f}" for g in (4,3,2))+f" | {x['A']:.6f} | {'是' if x['eligible'] else '否'} |")
    legacy=summaries['R_legacy']['paired_vs_U0']['per_G']
    lines+=['', '旧 R_legacy 相对 U0 的最终配对平均增益：'+ '；'.join(f"G{g} {legacy[str(g)]['mean']:.6f} dB" for g in (4,3,2))+'。']
    if not choice['selected']:
        lines+=['','12 个新候选全部未达到预声明门槛，没有冻结胜出候选。确认集和候选 heldout 评估按规则不启动；不增加候选或重新选样本。',
            '最接近 U0 的候选仍为负增益。输出保护减轻损害不等于修复已有效；J/H 下降本身也不构成恢复质量提升。']
    else:lines+=['',f"唯一冻结候选：`{choice['selected']['name']}`。确认和 heldout 结果另列；不会据此重选。"]
    lines+=['','## 首跳诊断','',
        '| source | cos(grad J, grad Q) | grad Q · 首次 Adam proposal | 实际 ΔQ |','|---|---:|---:|---:|']
    for d in directions:lines.append(f"| {d['source_id']} | {d['all']['cosine']:.6f} | {d['all']['grad_Q_dot_first_Adam_proposal']:.6g} | {d['actual_Q_change']:.6g} |")
    lines+=['','四个预设方向诊断点的梯度余弦均为负，首次 Adam proposal 的 grad(Q)·d 与实际 ΔQ 均为正，说明这些初始点降低观测目标 J 的更新方向与下一次 clean 恢复目标冲突。范围仅限这些点，不能当作所有样本的唯一因果解释。']
    lines+=['','方向诊断只提供这四个已冻结开发 source 初始点的一阶证据；分块轨迹和全部 n=0/1/5/10/20/40 数值保存在原始诊断文件，不能据此加入块选择算法。','',
        '## 计算与资源','',
        '同一 start 的各候选分配到同一张 de36 卡，单次完整路径时间与 network/F/VJP 调用逐项保存。运行期间有外部任务和短 T1 qualification，共载延迟会影响 wall time。没有候选过质量门槛，因此本轮结果不依赖接近候选的时间排序。',
        'T1 qualification 触发外部任务延迟保护后已停止。I1 guard 的历史窗口也捕获了训练时段，受影响分片从已完成结果继续；未完成路径的尝试保留在 guard 日志中。外部进程从未被发信号、暂停或修改。','',
        '开发与确认 source 都曾被旧模型训练；此次结论是有限候选在预设开发集不获支持，不能外推为所有 refinement 都无效。']
    (REPORTS/'I1_REPORT.md').write_text('\n'.join(lines)+'\n')
    figure_dir=REPORTS/'figures';figure_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[r['candidate']['name'] for r in choice['ranking']];scores=[r['A'] for r in choice['ranking']]
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(names[::-1],scores[::-1],color='#457b9d');ax.axvline(0,color='black',linewidth=.8)
    ax.axvline(.05,color='#b33a3a',linestyle='--',label='Predeclared eligibility threshold')
    ax.set_xlabel('Mean paired final PSNR gain vs U0 (dB)');ax.set_title('I1: finite development candidates, raw checkpoint 1000')
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(figure_dir/'I1_dev_gain.png',dpi=160);fig.savefig(figure_dir/'I1_dev_gain.pdf');plt.close(fig)
    # Fixed dev source side-by-side images. Metrics always use the raw arrays.
    from PIL import Image,ImageDraw
    sid=read(ROOT/'shared/manifests/FROZEN_SPLITS.json')['splits']['refine_dev8']['start_ids'][0]
    panels=[]
    for name in ('U0','R_legacy',names[0]):
        p=folder/'dev'/name/sid/'arrays/clean_k4.npy';a=np.load(p,allow_pickle=False)[0]
        im=Image.fromarray(np.uint8(np.clip(a.transpose(1,2,0),0,1)*255))
        panel=Image.new('RGB',(256,285),'white');panel.paste(im,(0,25));ImageDraw.Draw(panel).text((5,5),name,fill='black');panels.append(panel)
    grid=Image.new('RGB',(768,285),'white')
    for i,p in enumerate(panels):grid.paste(p,(256*i,0))
    grid.save(figure_dir/'I1_fixed_dev_preview.png')
    save_json(folder/'REPORT_COMPLETE.json',{'status':'reported','selection_status':choice['status'],'dev_paths':336,
        'report':str(REPORTS/'I1_REPORT.md'),'time':time.time(),'further_candidate_search':False})

def old_results(update):
    root=OLD/'artifacts/evaluation'/f'update_{update:06d}'/'raw';summary=read(root/'summary.json')
    assert summary['status']=='complete' and summary['network_unchanged'] and summary['simulator_unchanged']
    assert summary['identity']['weights']=='raw' and summary['identity']['global_update']==update
    assert summary['identity']['checkpoint_sha256']==sha256(OLD/'runs/formal'/f'checkpoint_{update:06d}.pt')
    cfg=read(ROOT/'shared/original_train_config.json');sources={r['start_id']:r['source_id'] for r in rows(Path(cfg['dataset_root'])/'start_manifest.jsonl')}
    result={}
    for r in rows(root/'unrefined/per_start.jsonl'):
        assert r['mode']=='unrefined' and r['weights']=='raw' and r['global_update']==update
        result[r['start_id']]={'start_id':r['start_id'],'trajectory_id':r['trajectory_id'],'source_id':sources[r['start_id']],
            'metrics':[{'G':r['initial_G'],'k':p['K'],'current_g':p['current_g'],'MSE':p['clean_mse'],'PSNR':p['clean_psnr_db'],
                'zero_MSE':p['zero_mse'],'LR':p['network_redegradation_mse']} for p in r['predictions']],
            'reused_from':{'path':str(root/'unrefined/per_start.jsonl'),'summary_sha256':sha256(root/'summary.json'),
                'checkpoint_sha256':summary['identity']['checkpoint_sha256']}}
    assert len(result)==(1600 if update in (500,1000) else 64)
    save_rows(REPORTS/'old_raw_reused'/f'update_{update:06d}.jsonl',list(result.values()))
    return result

def training_report():
    allcomparisons={};missing=[];table=[];runtime={}
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        result_path=ROOT/task/'formal/results.json'
        if result_path.exists():
            result=read(result_path)
            updates=[r for r in rows(ROOT/task/'formal/updates_rank0.jsonl') if r.get('phase')=='formal']
            runtime[task]={'wall_seconds':result['wall_seconds'],'update_seconds_sum':sum(r['seconds'] for r in updates),
                'microbatch':result['microbatch'],'accumulation':result['accumulation'],'counts':result['counts'],
                'trainable_parameters':result['parameter_groups']['total_trainable_parameters'],
                'peak_reserved_bytes':result['max_peak_reserved_bytes'],'checkpoint':read(ROOT/task/'COMPLETE_1000.json')}
        for u in range(100,1001,100):
            subset='full1600' if u in (500,1000) else 'fixed64';out=ROOT/task/'evaluation'/f'update_{u:06d}_{subset}'
            if not all((out/f'complete_rank{r}.json').exists() for r in range(4)):
                missing.append(f'{task}@{u}');continue
            values=load_results(out);assert len(values)==(1600 if u in (500,1000) else 64)
            old=old_results(u)
            t1=load_results(ROOT/'T1_DIRECT/evaluation'/f'update_{u:06d}_{subset}') if task=='T2_FEEDBACK_R' else None
            paired=comparisons(values,old,t1);comparison=paired['vs_old'];own=paired['vs_own_K1']
            curves=metric_curves(values)
            save_rows(out/'per_start.jsonl',list(values.values()))
            result={**paired,'curves':curves,'inference_cost':inference_cost(values),'same_updates_not_equal_compute':True}
            save_json(out/'summary.json',result);allcomparisons[f'{task}@{u}']=result
            if u in (500,1000):
                for g in (4,3,2,1):table.append(f"| {task} | {u} | {g} | {comparison['per_G'][str(g)]['mean']:.6f} | {own['per_G'][str(g)]['mean']:.6f} |")
    save_json(REPORTS/'TRAINING_PAIRED_SUMMARY.json',{'results':allcomparisons,'missing':missing,'runtime':runtime})
    lines=['# V0.31 后续训练对照','',
        '每个任务从共同初始化新训，固定 effective batch=32、原始学习率计划、1000 updates。相同 updates 仅表示相同起点曝光，不能称为等算力。','',
        '| 任务 | update | G | 最终相对旧 raw ΔPSNR | 最终相对自身 K1 ΔPSNR |','|---|---:|---:|---:|---:|',*table,'',
        '缺项：'+(', '.join(missing) if missing else '无主要训练评估缺项。'),'',
        'source 级统计先在同 source、同 G 内平均 realization，原始逐起点及每个合法 K 的指标另存。没有独立训练 seed，不能声称多 seed 稳健性。']
    (REPORTS/'TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n')
    return missing

def heldout_report():
    ready=ROOT/'shared/heldout32/READY.json'
    if not ready.exists():
        save_json(REPORTS/'HELDOUT_SUMMARY.json',{'status':'missing_materials','scope':'only heldout blocked'})
        return
    base_folder=ROOT/'V031_UNREFINED/evaluation/update_001000_heldout32'
    base=load_results(base_folder);assert len(base)==256
    output={'dataset':read(ready),'interpretation':'same-distribution unseen sources, not an external benchmark','comparisons':{},'all_comparisons':{},
        'curves':{'V031_UNREFINED':metric_curves(base)},'inference_cost':{'V031_UNREFINED':inference_cost(base)}}
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        folder=ROOT/task/'evaluation/update_001000_heldout32'
        assert all((folder/f'complete_rank{r}.json').exists() for r in range(4))
        values=load_results(folder);assert len(values)==256
        output['comparisons'][task]=deltas(values,base)
        t1=load_results(ROOT/'T1_DIRECT/evaluation/update_001000_heldout32') if task=='T2_FEEDBACK_R' else None
        output['all_comparisons'][task]=comparisons(values,base,t1)
        output['curves'][task]=metric_curves(values);output['inference_cost'][task]=inference_cost(values)
        save_rows(folder/'per_start.jsonl',list(values.values()))
    save_rows(base_folder/'per_start.jsonl',list(base.values()))
    choice=read(ROOT/'I1_REFINE_REPAIR/selection.json')
    output['I1_status']='not_run_no_eligible_dev_candidate' if not choice['selected'] else 'see_I1_heldout32'
    save_json(REPORTS/'HELDOUT_SUMMARY.json',output)

def overview_and_figures():
    report=read(REPORTS/'TRAINING_PAIRED_SUMMARY.json');heldout=read(REPORTS/'HELDOUT_SUMMARY.json')
    lines=['# V0.31 后续实验交付','',
        'T1_DIRECT 与 T2_FEEDBACK_R 各完成1000 updates并暂停，保留完整续训状态。I1完成预声明有限候选，开发集不支持任何新修复候选，按规则停止。','',
        '| raw1000任务 | G4 ΔPSNR | G3 ΔPSNR | G2 ΔPSNR | G1 ΔPSNR |','|---|---:|---:|---:|---:|']
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        comp=report['results'][task+'@1000']['vs_old']['per_G']
        lines.append('| '+task+' | '+' | '.join(f"{comp[str(g)]['mean']:.6f}" for g in (4,3,2,1))+' |')
    lines+=['','上表为原训练集1600起点相对旧 raw1000 无 refinement 的最终配对平均PSNR增益，单位dB。逐步曲线、source级中位数/胜负和原始FP32指标另存，不从PNG算指标。','',
        '## 成本与检查点','',
        '| 任务 | microbatch/accumulation | 完整update秒数合计 | 训练墙钟秒数 |','|---|---|---:|---:|']
    for task,r in report['runtime'].items():
        lines.append(f"| {task} | {r['microbatch']}/{r['accumulation']} | {r['update_seconds_sum']:.2f} | {r['wall_seconds']:.2f} |")
    for task,r in report['runtime'].items():
        cp=r['checkpoint'];lines += ['',f"{task}：`{cp['checkpoint']}`；SHA256 `{cp['sha256']}`。",'']
    lines+=['同1000 updates代表每个起点20次曝光，不能称为等算力。网络/模拟器计数、实际配置、资源保护记录和每个checkpoint身份随包保存。没有独立训练seed，不能宣称多seed稳定性。','',
        '## Heldout32','']
    if heldout.get('comparisons'):
        lines+=['固定32个未见train-pool source、2组xi/source、共256起点，仅在raw1000评估。它是同分布新场景筛查，不是外部benchmark。','',
            '| 任务 | G4 ΔPSNR | G3 ΔPSNR | G2 ΔPSNR | G1 ΔPSNR |','|---|---:|---:|---:|---:|']
        for task,r in heldout['comparisons'].items():lines.append('| '+task+' | '+' | '.join(f"{r['per_G'][str(g)]['mean']:.6f}" for g in (4,3,2,1))+' |')
    else:lines.append('Heldout材料缺项，详见HELDOUT_SUMMARY.json；主训练与原数据评估独立完成。')
    lines+=['','## 文件与边界','',
        'I1_REPORT.md报告所有新候选与首跳诊断。TRAINING_REPORT.md、TRAINING_PAIRED_SUMMARY.json和HELDOUT_SUMMARY.json保存训练对照。固定示例沿用旧样本并加入预先固定的5个source。大checkpoint和FP32数组保留在de36新实验目录；数组路径、大小和hash另存。旧V0.31未续训，未覆盖旧数据、源码或结果。','',
        'I1确认集与候选heldout没有运行，因为没有开发候选过门槛；这是预声明停止规则，不能把最接近U0的负增益候选称为成功修复。']
    (REPORTS/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    figure_dir=REPORTS/'figures';figure_dir.mkdir(exist_ok=True)
    for update in (500,1000):
        old=old_results(update)
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for g,ax in zip((4,3,2,1),axes.flat):
            oldgroup=[r for r in old.values() if r['metrics'][0]['G']==g]
            ax.plot(range(1,g+1),[statistics.mean(r['metrics'][k]['PSNR'] for r in oldgroup) for k in range(g)],'o-',label='Old V0.31 raw')
            for task in ('T1_DIRECT','T2_FEEDBACK_R'):
                curves=[r for r in report['results'][f'{task}@{update}']['curves'] if r['G']==g]
                ax.plot([r['k'] for r in curves],[r['PSNR']['mean'] for r in curves],'o-',label=task)
            ax.set_title(f'G={g}');ax.set_xlabel('Restoration step K');ax.set_ylabel('Mean image PSNR (dB)');ax.set_xticks(range(1,g+1));ax.grid(alpha=.2)
        axes[0,0].legend();fig.suptitle(f'Fixed raw checkpoint {update}, all1600 starts');fig.tight_layout()
        fig.savefig(figure_dir/f'training_raw{update}_curves.png',dpi=160);fig.savefig(figure_dir/f'training_raw{update}_curves.pdf');plt.close(fig)
        ids=read(ROOT/'shared/manifests/fixed_preview_starts.json')['ids']
        preview_dir=figure_dir/f'fixed_raw{update}';preview_dir.mkdir(exist_ok=True)
        for sid in ids:
            g=int(sid[-1]);base=ROOT/'V031_UNREFINED/evaluation'/f'update_{update:06d}_previews'/sid/'arrays'
            sources=[('GT',base/'GT.npy'),('Y',base/'Y.npy'),('Old V0.31',base/f'clean_k{g}.npy')]
            for task,k in (('T1_DIRECT',1),('T2_FEEDBACK_R',g)):
                sources.append((task,ROOT/task/'evaluation'/f'update_{update:06d}_full1600'/sid/'arrays'/f'clean_k{k}.npy'))
            grid=Image.new('RGB',(5*256,285),'white');draw=ImageDraw.Draw(grid)
            for col,(label,path) in enumerate(sources):
                a=np.load(path,allow_pickle=False);a=a[0] if a.ndim==4 else a
                im=Image.fromarray(np.uint8(np.clip(a.transpose(1,2,0),0,1)*255))
                grid.paste(im,(col*256,25));draw.text((col*256+5,5),label,fill='black')
            grid.save(preview_dir/f'{sid}.png')

def complete_delivery_details():
    """CPU-only presentation of already frozen records; no model execution or selection."""
    import difflib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    report=read(REPORTS/'TRAINING_PAIRED_SUMMARY.json');heldout=read(REPORTS/'HELDOUT_SUMMARY.json')
    cfg=read(ROOT/'shared/original_train_config.json');figdir=REPORTS/'figures'
    lines=['# 完整配对对照与计算成本','',
        '比较方向均为新任务最终输出减指定基线，单位dB；均值是逐图PSNR差的平均，tie=1e-6。source统计先在同source、同G内平均realization。G1不排除。零MSE另列于JSON，不把无穷值当有限均值。','']
    for label,groups in [('训练集 raw500',{t:report['results'][t+'@500'] for t in ('T1_DIRECT','T2_FEEDBACK_R')}),
                         ('训练集 raw1000',{t:report['results'][t+'@1000'] for t in ('T1_DIRECT','T2_FEEDBACK_R')}),
                         ('heldout32 raw1000',heldout['all_comparisons'])]:
        lines+=['## '+label,'','|任务|基线|G|起点数|均值|中位数|胜/平/负|source数|source均值|source中位数|source胜/平/负|',
            '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|']
        for task,group in groups.items():
            for name in ('vs_input','vs_old_K1','vs_old','vs_own_K1','vs_T1_K1'):
                if name not in group:continue
                for g in (4,3,2,1):
                    a=group[name]['per_G'][str(g)];s=group[name]['source_per_G'][str(g)]
                    lines.append(f"|{task}|{name}|{g}|{a['n']}|{a['mean']:.6f}|{a['median']:.6f}|{a['wins']}/{a['ties']}/{a['losses']}|{s['n']}|{s['mean']:.6f}|{s['median']:.6f}|{s['wins']}/{s['ties']}/{s['losses']}|")
        lines.append('')
    lines+=['## 训练成本','','|任务|参数量|路径数|预测次数|LR渲染|转换渲染|训练秒数|峰值reserved GiB|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,r in report['runtime'].items():
        c=r['counts'];lines.append(f"|{task}|{r['trainable_parameters']}|{c['paths']}|{c['predictions']}|{c['LR_simulator_renders']}|{c['transition_simulator_renders']}|{r['wall_seconds']:.3f}|{r['peak_reserved_bytes']/2**30:.3f}|")
    lines+=['','同1000 updates仅表示相同起点曝光。两任务的网络结构、网络/F调用和训练时间不同，不能叫等算力。','',
        '## 推理成本','','|集合|方法|路径数|路径耗时合计秒|每路径均值秒|网络调用|F调用|VJP调用|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    costs={'train_raw500':{t:report['results'][t+'@500']['inference_cost'] for t in report['runtime']},
           'train_raw1000':{t:report['results'][t+'@1000']['inference_cost'] for t in report['runtime']},
           'heldout32_raw1000':heldout['inference_cost']}
    for subset,methods in costs.items():
        for task,c in methods.items():
            lines.append(f"|{subset}|{task}|{c['paths']}|{c['path_seconds_sum']:.3f}|{c['path_seconds']['mean']:.6f}|{c['calls']['network']}|{c['calls']['F']}|{c['calls']['VJP']}|")
    lines+=['','路径耗时是已有记录的合计，不是四卡并行墙钟，不含worker启动。不同时间/共载环境的耗时不作为硬件加速比证据。完整记录含每路径耗时、调用数和身份。','',
        '## 结论边界','',
        'T1只说明去除xi分支和LR后的直接结构拟合能力，不能单独归因于某一移除项。T2与旧模型都包含LR，因此T2胜出不等于物理无用；旧模型胜出也只支持本次F状态转换对照。训练集是overfit100，heldout32只是同分布未见source筛查；没有多seed、外部分布或真实湍流证据。']
    (REPORTS/'ALL_PAIRED_COMPARISONS.md').write_text('\n'.join(lines)+'\n')
    save_json(REPORTS/'INFERENCE_COSTS.json',costs)

    # Unsmoothed optimization traces plus comparable fixed64 checkpoint curves.
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        logs=[r for r in rows(ROOT/task/'formal/updates_rank0.jsonl') if r.get('phase')=='formal']
        assert [r['global_update'] for r in logs]==list(range(1,1001))
        for ax,key in zip(axes,('loss','LC','LR')):
            ax.plot([r['global_update'] for r in logs],[r[key] for r in logs],linewidth=.65,alpha=.8,label=task)
            ax.set_ylabel(key);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Optimizer update');fig.suptitle('Recorded training traces, no smoothing')
    fig.tight_layout();fig.savefig(figdir/'training_update_losses.png',dpi=160);fig.savefig(figdir/'training_update_losses.pdf');plt.close(fig)
    fixed={r['start_id'] for r in rows(Path(cfg['dataset_root'])/cfg['diagnostic_subset'])}
    assert len(fixed)==64
    fixed_curves=[];fig,axes=plt.subplots(2,2,figsize=(10,7))
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        byupdate={}
        for u in range(100,1001,100):
            subset='full1600' if u in (500,1000) else 'fixed64'
            values=load_results(ROOT/task/'evaluation'/f'update_{u:06d}_{subset}')
            values={s:r for s,r in values.items() if s in fixed};assert set(values)==fixed
            byupdate[u]={g:statistics.mean(r['metrics'][-1]['PSNR'] for r in values.values() if r['metrics'][0]['G']==g) for g in (4,3,2,1)}
            fixed_curves.extend({'task':task,'update':u,'G':g,'n':16,'mean_final_PSNR':v} for g,v in byupdate[u].items())
        for g,ax in zip((4,3,2,1),axes.flat):
            ax.plot(list(byupdate),[a[g] for a in byupdate.values()],'o-',label=task)
            ax.set_title(f'G={g}, same16 starts');ax.set_xlabel('Optimizer update');ax.set_ylabel('Mean image PSNR (dB)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Same fixed64 population at every checkpoint; raw final outputs')
    fig.tight_layout();fig.savefig(figdir/'training_update_fixed64_PSNR.png',dpi=160);fig.savefig(figdir/'training_update_fixed64_PSNR.pdf');plt.close(fig)
    save_rows(REPORTS/'training_update_fixed64_curves.jsonl',fixed_curves)

    # Show every legal clean prediction for the same pre-frozen24 starts.
    for u in (500,1000):
        out=figdir/f'all_steps_raw{u}';out.mkdir(exist_ok=True)
        for sid in read(ROOT/'shared/manifests/fixed_preview_starts.json')['ids']:
            g=int(sid[-1]);grid=Image.new('RGB',((g+2)*256,3*285),'#eeeeee');draw=ImageDraw.Draw(grid)
            for row,task in enumerate(('V031_UNREFINED','T1_DIRECT','T2_FEEDBACK_R')):
                subset='previews' if task=='V031_UNREFINED' else 'full1600'
                base=ROOT/task/'evaluation'/f'update_{u:06d}_{subset}'/sid/'arrays'
                paths=[('GT',base/'GT.npy'),('Y',base/'Y.npy')]+[(f'K{k}',base/f'clean_k{k}.npy') for k in range(1,2 if task=='T1_DIRECT' else g+1)]
                for col,(label,path) in enumerate(paths):
                    a=np.load(path,allow_pickle=False);a=a[0] if a.ndim==4 else a
                    assert a.dtype==np.float32 and np.isfinite(a).all()
                    grid.paste(Image.fromarray(np.uint8(np.clip(a.transpose(1,2,0),0,1)*255)),(col*256,row*285+25))
                    draw.text((col*256+5,row*285+5),task+' '+label,fill='black')
            grid.save(out/f'{sid}.png')

    # Verifiable scientific source diff, including unchanged-file hash inventory.
    code_index={};patches=[]
    for p in [ROOT/'code/train_worker.py',*sorted((ROOT/'code/v031').glob('*.py'))]:
        rel=str(p.relative_to(ROOT));old=OLD/rel
        if old.exists():
            assert sha256(old)==cfg['code_hashes'][rel]
            before=old.read_text();oldhash=sha256(old)
        else:before='';oldhash=None
        code_index[rel]={'old_sha256':oldhash,'new_sha256':sha256(p),'changed':before!=p.read_text()}
        patches.extend(difflib.unified_diff(before.splitlines(True),p.read_text().splitlines(True),fromfile='old/'+rel if old.exists() else '/dev/null',tofile='followup/'+rel))
    (REPORTS/'SOURCE_CHANGES.patch').write_text(''.join(patches));save_json(REPORTS/'SOURCE_CHANGE_INDEX.json',code_index)
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        actual=read(ROOT/'configs'/f'{task}_formal.json')
        patch=''.join(difflib.unified_diff((json.dumps(cfg,indent=2,sort_keys=True)+'\n').splitlines(True),(json.dumps(actual,indent=2,sort_keys=True)+'\n').splitlines(True),fromfile='old/train_v031.json',tofile=f'followup/{task}_formal.json'))
        (REPORTS/f'{task}_CONFIG_CHANGES.patch').write_text(patch)
    resume=['# 完整状态与恢复入口','',
        '两项正式训练已停在1000；本次没有启动1000以后的训练。完整状态在de36，各自只有一个formal目录。','',
        'CPU验收入口：`bash code/run_single_env.sh code/verify_boundary.py --checkpoint TASK/formal/checkpoint_001000.pt --output TASK/COMPLETE_1000.json`（将TASK替换为对应任务）。','',
        '恢复入口为 `code/train_worker.py --mode train --config configs/TASK_formal.json --output TASK/formal --microbatch MB --resume TASK/formal/checkpoint_NNNNNN.pt --run-until-update 1000`，由 `code/training_controller.py:command` 的四rank torch.distributed.run和现有资源guard启动。T1 MB8，T2 MB1。意外中断时使用同目录最新完整兼容checkpoint，不从初始化重训；当前两项已经到终点，无待恢复工作。','',
        '若未来明确授权超过1000，需要带有新授权上限的续训配置和 `--authorize-boundary-extension`，保留原始计划中1000以后floor LR规则以及全部身份校验；不能只改命令行步数。本次未生成或执行该扩展。','']
    for task,r in report['runtime'].items():
        cp=r['checkpoint'];resume.extend([f"- {task}: `{cp['checkpoint']}`；SHA256 `{cp['sha256']}`。完整raw FP32、AdamW、scheduler、4rank RNG、sampler/counters验收通过。",''])
    (REPORTS/'CHECKPOINT_AND_RESUME.md').write_text('\n'.join(resume)+'\n')
    save_json(REPORTS/'MISSINGNESS.json',{'required_training_evaluations_missing':report['missing'],
        'heldout32_complete':True,'I1_confirmation':'not_run_predeclared_no_eligible_dev_candidate',
        'I1_candidate_heldout':'not_run_predeclared_no_eligible_dev_candidate',
        'confirm_source_arrays':'not_applicable_confirmation_not_opened',
        'checkpoint_and_FP32_arrays':'retained remotely; paths and hashes supplied',
        'local_package_and_visual_QA':'performed after remote packaging; see DELIVERY_VALIDATION.json beside ZIP'})
    with (REPORTS/'README.md').open('a') as f:
        f.write('\n完整辅助对照、输入/旧K1/自身K1/T1配对及source统计见 [ALL_PAIRED_COMPARISONS.md](ALL_PAIRED_COMPARISONS.md)。[CHECKPOINT_AND_RESUME.md](CHECKPOINT_AND_RESUME.md) 提供完整状态位置与恢复入口；SOURCE_CHANGES.patch、各CONFIG_CHANGES.patch保留源码/配置差异；MISSINGNESS.json列明未运行分支。训练曲线在figures/training_update_*，全部合法K的固定图像在figures/all_steps_raw500与all_steps_raw1000。\n')
        f.write('\n## 未见source的绝对质量与结论\n\n| heldout32 raw1000 | G4 PSNR | G3 PSNR | G2 PSNR | G1 PSNR |\n|---|---:|---:|---:|---:|\n')
        def curve(task,g,first=False):
            group=[r for r in heldout['curves'][task] if r['G']==g]
            return group[0] if first else group[-1]
        f.write('| 输入Y | '+' | '.join(f"{curve('V031_UNREFINED',g)['input_PSNR']['mean']:.6f}" for g in (4,3,2,1))+' |\n')
        for task,first,label in [('V031_UNREFINED',True,'旧模型 K1'),('V031_UNREFINED',False,'旧模型 FINAL'),
                                 ('T1_DIRECT',False,'T1 K1'),('T2_FEEDBACK_R',True,'T2 K1'),('T2_FEEDBACK_R',False,'T2 FINAL')]:
            f.write('| '+label+' | '+' | '.join(f"{curve(task,g,first)['PSNR']['mean']:.6f}" for g in (4,3,2,1))+' |\n')
        worse_than_input=all(curve(task,g)['PSNR']['mean']<curve(task,g)['input_PSNR']['mean'] for task in heldout['curves'] for g in (4,3,2,1))
        if worse_than_input:
            f.write('\n在这个预冻结heldout32上，三种模型各G组的最终平均PSNR都低于输入Y；相对旧模型的正增益不能等同于相对输入的有效恢复，也不能把训练集提升称为泛化成功。此结论限于本次32个同分布未见source。\n')
        t1_better=all(report['results']['T1_DIRECT@1000']['vs_old']['per_G'][str(g)]['mean']>0 for g in (4,3,2,1))
        if t1_better:f.write('\nT1在原训练集raw1000的四个G组均优于旧模型末步；它支持本次直接结构的拟合能力，但本对照同时改变xi分支和LR，不能把收益单独归因于某一项。\n')
        f.write('\nT2与旧模型的逐G胜负以配对表为准，不把局部改善称为稳定优势；两者都含LR，不能据此断言物理约束整体无用。I1没有通过开发门槛，按有限候选规则终止。\n')

def package(final=False):
    target=ROOT/('V031_FOLLOWUP_RESULTS.zip' if final else 'I1_REFINE_REPAIR_RESULTS.zip')
    selected=[]
    for base in ('reports','configs','code','shared/authority','shared/manifests','shared/source_snapshots','resources','acceptance'):
        selected.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    selected=[p for p in selected if p.name!=target.stem+'_MANIFEST.json']
    selected.extend(p for p in (ROOT/'I1_REFINE_REPAIR').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
    selected.extend(p for p in (ROOT/'shared/heldout32').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
    selected.extend(p for p in (ROOT/'V031_UNREFINED/evaluation').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
    selected.extend(p for p in (ROOT/'shared').glob('*.json') if p.is_file())
    selected.append(ROOT/'WORKLOG.md')
    arrays=list((ROOT/'I1_REFINE_REPAIR').rglob('*.npy'))+list((ROOT/'I1_REFINE_REPAIR').rglob('*.npz'))
    asset_file=REPORTS/'I1_FP32_ARRAY_LOCATIONS.json'
    if not asset_file.exists():
        save_json(asset_file,{'root':str(ROOT),'location':'retained on de36; not in the small report ZIP',
            'files':{str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in arrays}})
    selected.append(asset_file)
    if final:
        locations={}
        for task in ('T1_DIRECT','T2_FEEDBACK_R','V031_UNREFINED'):
            for p in (ROOT/task/'evaluation').rglob('*.npy'):
                locations[str(p.relative_to(ROOT))]={'sha256':sha256(p),'bytes':p.stat().st_size}
        asset_file=REPORTS/'TRAINING_FP32_ARRAY_LOCATIONS.json'
        save_json(asset_file,{'root':str(ROOT),'location':'retained on de36; not in the small report ZIP','files':locations})
        selected.append(asset_file)
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        selected.extend(p for p in (ROOT/task).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log'))
    tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(set(selected)):z.write(p,str(p.relative_to(ROOT)))
    tmp.replace(target)
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    save_json(REPORTS/(target.stem+'_MANIFEST.json'),{'path':str(target),'sha256':sha256(target),'files':len(set(selected)),
        'large_checkpoints_and_npy_remain_on_server':True,'final':final,'time':time.time()})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('i1','all'),required=True);a=ap.parse_args()
    build_code_sha256=sha256(__file__)
    REPORTS.mkdir(exist_ok=True)
    i1_report()
    if a.mode=='all':
        missing=training_report()
        if missing:raise RuntimeError('Final reports incomplete: '+str(missing))
        heldout_report()
        overview_and_figures()
        complete_delivery_details()
    assert sha256(__file__)==build_code_sha256,'Report source changed during build'
    save_json(REPORTS/'REPORT_BUILD_PROVENANCE.json',{'report_code_sha256':build_code_sha256,
        'mode':a.mode,'status':'reports_generated_from_existing_records','model_execution':False,'time':time.time()})
    package(a.mode=='all')

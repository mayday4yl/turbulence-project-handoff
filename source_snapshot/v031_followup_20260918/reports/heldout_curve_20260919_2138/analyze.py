"""Paired-source analysis of frozen checkpoints; no model execution."""
from pathlib import Path
from collections import defaultdict
import json, math, csv, hashlib, statistics
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT if (ROOT/'T1_DIRECT/evaluation/update_001000_heldout32').exists() else ROOT/'V031_FOLLOWUP_RESULTS'
UPDATES = (100, 500, 800, 900, 1000)
TASKS = ('T1_DIRECT', 'T2_FEEDBACK_R')
PLAN = json.loads((HERE/'PLAN.json').read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(folder):
    out = {}
    for p in sorted(folder.glob('*/result.json')):
        r = json.loads(p.read_text())
        assert r['start_id'] not in out and r['status'] == 'complete'
        assert r['weights'] == 'raw'
        assert all(math.isfinite(m['PSNR']) and math.isfinite(m['MSE']) for m in r['metrics'])
        for m in [r['input'], *r['metrics']]:
            assert abs(m['PSNR'] + 10*math.log10(m['MSE'])) < 1e-9
        out[r['start_id']] = r
    return out

def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260919)
    indices = rng.integers(0, len(values), (10000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return float(values.mean()), [float(x) for x in np.quantile(bootstrap, [.025, .975])]

def by_source(rows, field, g):
    grouped = defaultdict(list)
    for r in rows.values():
        if g != 'all' and r['metrics'][0]['G'] != g:
            continue
        v = r['input']['PSNR'] if field == 'input' else r['metrics'][0 if field == 'first' else -1]['PSNR']
        grouped[r['source_id']].append(v)
    return {s: statistics.mean(v) for s, v in sorted(grouped.items())}

def write_csv(name, records):
    with (HERE/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)

def main():
    complete = json.loads((HERE/'COMPLETE.json').read_text())
    assert complete['new_paths'] == 2048 and complete['training_updates'] == 0
    datasets = {}; evidence = []; flat = []; aggregates = []; delta_rows = []
    fixed_ids = set(load(BASE/'T1_DIRECT/evaluation/update_000100_fixed64'))
    assert len(fixed_ids) == 64
    shared_heldout = None
    for task in TASKS:
        for update in UPDATES:
            for split in ('train_fixed64', 'heldout32'):
                suffix = ('full1600' if update in (500, 1000) else 'fixed64') if split == 'train_fixed64' else 'heldout32'
                folder = ((HERE/'evaluation'/task) if split == 'heldout32' and update != 1000 else BASE/task/'evaluation')/f'update_{update:06d}_{suffix}'
                rows = load(folder)
                if split == 'train_fixed64':
                    assert fixed_ids <= set(rows)
                    rows = {s: rows[s] for s in sorted(fixed_ids)}
                else:
                    if shared_heldout is None:
                        shared_heldout = rows
                    assert set(rows) == set(shared_heldout)
                    for sid, r in rows.items():
                        ref = shared_heldout[sid]
                        assert (r['source_id'], r['trajectory_id'], r['input']) == (ref['source_id'], ref['trajectory_id'], ref['input'])
                assert len(rows) == (64 if split == 'train_fixed64' else 256)
                sources = {r['source_id'] for r in rows.values()}
                assert len(sources) == (16 if split == 'train_fixed64' else 32)
                identities = {json.dumps(r['identity'], sort_keys=True) for r in rows.values()}
                assert len(identities) == 1
                identity = json.loads(next(iter(identities)))
                assert identity['task'] == task and identity['update'] == update
                if split == 'heldout32':
                    assert identity['worker'] == PLAN['worker_sha256']
                    assert identity['manifest'] == '50e506c0d57ddcdbab61751425e8b3f2b3ec8e7a4abb83b954f950aa9c0adc60'
                datasets[task, update, split] = rows
                evidence.append({'task': task, 'update': update, 'split': split, 'rows': len(rows), 'sources': len(sources), 'path': str(folder), 'identity': identity})
                for sid, r in rows.items():
                    assert len(r['metrics']) == (1 if task == 'T1_DIRECT' else r['metrics'][0]['G'])
                    for m in r['metrics']:
                        flat.append({'task': task, 'update': update, 'split': split, 'start_id': sid, 'source_id': r['source_id'], 'G': m['G'], 'k': m['k'], 'PSNR': m['PSNR'], 'MSE': m['MSE'], 'input_PSNR': r['input']['PSNR']})
                for g in ('all', 4, 3, 2, 1):
                    for field in ('input', 'first', 'final'):
                        source_values = by_source(rows, field, g)
                        mean, ci = mean_ci(list(source_values.values()))
                        aggregates.append({'task': task, 'update': update, 'split': split, 'G': g, 'output': field, 'sources': len(source_values), 'starts': len(rows) if g == 'all' else len(rows)//4, 'PSNR_mean': mean, 'source_bootstrap_low': ci[0], 'source_bootstrap_high': ci[1]})
    assert {r['source_id'] for r in datasets[TASKS[0],100,'train_fixed64'].values()}.isdisjoint({r['source_id'] for r in shared_heldout.values()})
    for task in TASKS:
        for split in ('train_fixed64', 'heldout32'):
            for g in ('all', 4, 3, 2, 1):
                for field in ('first', 'final'):
                    for start, end in ((100,500),(500,800),(800,900),(900,1000),(100,1000),(500,1000),(800,1000)):
                        a = by_source(datasets[task,start,split],field,g)
                        b = by_source(datasets[task,end,split],field,g)
                        assert set(a) == set(b)
                        v = [b[s]-a[s] for s in a]
                        mean, ci = mean_ci(v)
                        delta_rows.append({'task':task,'split':split,'G':g,'output':field,'from_update':start,'to_update':end,'sources':len(v),'delta_PSNR':mean,'source_bootstrap_low':ci[0],'source_bootstrap_high':ci[1],'source_wins':sum(x>1e-6 for x in v),'source_ties':sum(abs(x)<=1e-6 for x in v),'source_losses':sum(x< -1e-6 for x in v)})
    write_csv('PER_START_METRICS.csv',flat)
    write_csv('AGGREGATES.csv',aggregates)
    write_csv('PAIRED_SOURCE_DELTAS.csv',delta_rows)
    (HERE/'EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    summary = {'plan':PLAN,'aggregates':aggregates,'paired_deltas':delta_rows,'new_paths':2048,'cached_paths_used':1152,'bootstrap':'10000 resamples of independent source IDs, seed20260919; within-source trajectories and G retained together; pointwise descriptive 95% CI, no multiplicity correction','flat_metric_count':len(flat),'interpretation_boundary':'No source-count causal test, no checkpoint selection, no new training; supplementary repeated use of heldout is diagnostic, not a new untouched test set.'}
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    plot(aggregates)
    report(aggregates,delta_rows)
    validation = {'status':'passed','complete_groups':20,'new_heldout_groups':8,'new_paths':2048,'raw_record_identity_and_PSNR_from_MSE_checked':True,'same_heldout_inputs_all_checkpoints':True,'fixed_training_start_ids_every_checkpoint':True,'source_counts':{'train':16,'heldout':32},'checkpoints_unchanged':True,'frozen_model_and_simulator_unchanged':all(json.loads(p.read_text())['frozen_network_simulator_unchanged'] for p in (HERE/'evaluation').glob('*/*/complete_rank0.json')),'training_updates':0,'files':{p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.csv','.png','.pdf','.md')},'plot_visual_QA':'pending'}
    (HERE/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({'macro': [r for r in aggregates if r['G']=='all' and r['output']=='final'], 'recent_deltas':[r for r in delta_rows if r['G']=='all' and r['output']=='final' and r['from_update']==900]},indent=2))

def get(rows,task,update,split,g,field='final'):
    return next(r for r in rows if (r['task'],r['update'],r['split'],r['G'],r['output'])==(task,update,split,g,field))['PSNR_mean']

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,task in zip(axes,TASKS):
        for split,label,color in [('train_fixed64','Train: same 64 starts / 16 sources','#2367b2'),('heldout32','Heldout: same 256 starts / 32 sources','#cc4b37')]:
            ax.plot(UPDATES,[get(rows,task,u,split,'all') for u in UPDATES],'-o',color=color,label=label)
        if task=='T2_FEEDBACK_R':
            ax.plot(UPDATES,[get(rows,task,u,'heldout32','all','first') for u in UPDATES],'--s',color='#d79522',label='Heldout: first prediction')
        ax.axhline(get(rows,task,100,'heldout32','all','input'),color='#444',ls=':',label='Heldout degraded input')
        ax.set_title(task);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
        ax.set_xticks(UPDATES);ax.tick_params(axis='x',labelsize=9)
    axes[0].set_ylabel('Mean per-image PSNR (dB), equal G weighting')
    fig.suptitle('Frozen-checkpoint diagnostic: training and unseen-source curves',fontsize=13)
    fig.tight_layout();fig.savefig(HERE/'checkpoint_curves.png',dpi=180);fig.savefig(HERE/'checkpoint_curves.pdf');plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(15,7),sharey=True,sharex=True)
    for i,task in enumerate(TASKS):
        for j,g in enumerate((4,3,2,1)):
            ax=axes[i,j]
            for split,label,color in [('train_fixed64','Train final','#2367b2'),('heldout32','Heldout final','#cc4b37')]:
                ax.plot(UPDATES,[get(rows,task,u,split,g) for u in UPDATES],'-o',color=color,label=label)
            if task=='T2_FEEDBACK_R':
                ax.plot(UPDATES,[get(rows,task,u,'heldout32',g,'first') for u in UPDATES],'--s',color='#d79522',label='Heldout first')
            ax.axhline(get(rows,task,100,'heldout32',g,'input'),color='#444',ls=':',label='Heldout input')
            ax.set_title(f'{task} / G={g}',fontsize=10);ax.grid(alpha=.2);ax.set_xticks(UPDATES);ax.tick_params(axis='x',labelsize=8,rotation=35)
            if j==0:ax.set_ylabel('Mean PSNR (dB)')
            if i==1:ax.set_xlabel('Optimizer updates')
            if j==0:ax.legend(fontsize=8)
    fig.suptitle('All legal severity groups; fixed populations at every checkpoint',fontsize=14)
    fig.tight_layout();fig.savefig(HERE/'checkpoint_curves_by_G.png',dpi=180);fig.savefig(HERE/'checkpoint_curves_by_G.pdf');plt.close(fig)

def report(rows,deltas):
    lines=['# 已有 checkpoint 的泛化曲线诊断','', '仅补充推理：T1/T2 的 100、500、800、900 步各 256 条 heldout 路径；1000 步及全部训练评估复用。训练更新数为 0，旧实验、checkpoint、数据及冻结 evaluator 均不修改。','',
           '训练曲线固定同一 64 starts / 16 clean sources；500/1000 从全 1600 结果中筛出这 64 条。heldout 固定 32 sources × 2 realizations × 4 G = 256 starts。两侧曲线覆盖不同 source，不能把 gap 本身当作数据量的因果证明。','',
           '## 全 G 等权平均（逐图 PSNR 的均值，dB）','', '| Update | T1 train64 | T1 heldout32 | T2 train64 final | T2 heldout32 final | T2 heldout first |','|---:|---:|---:|---:|---:|---:|']
    for u in UPDATES:
        values=[get(rows,'T1_DIRECT',u,'train_fixed64','all'),get(rows,'T1_DIRECT',u,'heldout32','all'),get(rows,'T2_FEEDBACK_R',u,'train_fixed64','all'),get(rows,'T2_FEEDBACK_R',u,'heldout32','all'),get(rows,'T2_FEEDBACK_R',u,'heldout32','all','first')]
        lines.append('| '+str(u)+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', f"Heldout 输入基线：{get(rows,TASKS[0],100,'heldout32','all','input'):.6f} dB。",'', '## 按 source 配对的变化','', '| Model | Split | Interval | Mean delta dB | Pointwise 95% source-bootstrap CI | Source wins / ties / losses |','|---|---|---|---:|---|---|']
    for task in TASKS:
        for split in ('train_fixed64','heldout32'):
            for a in (100,500,900):
                r=next(r for r in deltas if r['task']==task and r['split']==split and r['G']=='all' and r['output']=='final' and r['from_update']==a and r['to_update']==1000)
                lines.append(f"| {task} | {split} | {a}→1000 | {r['delta_PSNR']:+.6f} | [{r['source_bootstrap_low']:+.6f}, {r['source_bootstrap_high']:+.6f}] | {r['source_wins']} / {r['source_ties']} / {r['source_losses']} |")
    lines += ['', '10,000 次 source-cluster bootstrap，seed=20260919。先在同一 source 内平均 realization 和 G，再成组重采样；同一图的多个退化版本不是独立样本。区间是逐项描述性区间，未作多重比较校正，不是多训练 seed 证据。','', '## 全部 G 的 heldout final','', '| Model | G | 100 | 500 | 800 | 900 | 1000 | Input |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for g in (4,3,2,1):
            vals=[get(rows,task,u,'heldout32',g) for u in UPDATES]+[get(rows,task,100,'heldout32',g,'input')]
            lines.append(f'| {task} | {g} | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', '## 解释边界','',
              '- 判断“训练继续变好而 heldout 变差”须看各模型自己的时间曲线；不同模型可以呈现不同趋势。',
              '- 如果曲线下降，可确认该采样区间的泛化随训练恶化；不能仅靠本实验把原因唯一归到 100 个 source，也不能排除架构、优化或内容分布因素。',
              '- 如果两侧上升，只说明截至这些 checkpoint 仍在改善；不能保证继续到 2000 会超过输入，更不能宣称收敛。',
              '- 只有 5 个采样 checkpoint，未定位未采样区间的最佳点；本诊断不据 heldout 选择新的 checkpoint 或训练设置。',
              '- 增加 source 的效益仍需控制训练预算的独立 source-scaling 实验，本次未执行。',
              '- 此 heldout32 已被重复查看，用于事后诊断；后续调参后的最终泛化主张需要新的独立测试集。','',
              '## 文件','', '`PER_START_METRICS.csv` 保存固定 train64 与 heldout256 的每轮真实指标；`AGGREGATES.csv` 保存全 G/逐 G 汇总；`PAIRED_SOURCE_DELTAS.csv` 保存配对变化；`EVIDENCE.json` 保存实际 checkpoint/evaluator/manifest 身份与原始结果目录。PNG/PDF 给出宏平均和全部 G 曲线。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()

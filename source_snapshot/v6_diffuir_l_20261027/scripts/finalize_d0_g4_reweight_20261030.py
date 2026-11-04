"""CPU-only matched-budget report. Never chooses or launches A/B."""
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from method_impl.d0_g4_reweight_20261030 import EX, BASE, config, freeze
from scripts.audit_d0_g4_reweight_20261030 import run
from scripts.runtime_utils import ROOT, sha, write_json, is_same_process


def main():
    freeze()
    cfg = config()
    parent = torch.load(cfg['parent_path'], map_location='cpu', weights_only=False)
    baseline = EX / 'baseline_u0'
    summaries = {}
    data = {}
    checkpoints = {}
    for branch in ['Control', 'G4_biased']:
        out = run(branch, 'formal')
        complete = json.loads((out / 'PROCESS_COMPLETE.json').read_text())
        assert complete['exit_code'] == 0 and complete['supervisor_stop'] is None
        assert not is_same_process(json.loads((out / 'RANK_0.json').read_text()))
        result = json.loads((out / 'RESULT.json').read_text())
        target = 5000 if branch == 'Control' else (10000 if json.loads((EX / 'EXTENSION_DECISION.json').read_text())['extend_G4_biased'] else 5000)
        assert result['completed_branch_updates'] == target
        last = json.loads((out / 'LAST.json').read_text())
        assert sha(last['path']) == last['sha256']
        payload = torch.load(last['path'], map_location='cpu', weights_only=False)
        assert payload['update'] == BASE + target and payload['cursor'] == cfg['parent_cursor'] + target * 60
        assert payload['ema']['updates'] == BASE + target and payload['ema']['beta'] == .995
        assert len(payload['rng_by_rank']) == 4 and all(len(r['device']) == 4 for r in payload['rng_by_rank'])
        assert payload['scaler'] is None and len(payload['optimizer']['state']) == 276
        assert all(int(v['step']) == BASE + target for v in payload['optimizer']['state'].values())
        assert all(g['lr'] == 2e-6 for g in payload['optimizer']['param_groups'])
        for name in ['native_buffers', 'sampling_buffers']:
            assert set(payload[name]) == set(parent[name])
            assert all(torch.equal(v, parent[name][k]) for k, v in payload[name].items())
        counts = Counter()
        updates = [json.loads(line) for line in (out / 'UPDATES.jsonl').read_text().splitlines()]
        assert len(updates) == target
        for u, row in enumerate(updates, 1):
            assert row['branch_update'] == u and row['update'] == BASE + u
            assert row['finite'] and row['training_nfe_per_sample'] == 1 and row['sampling_training_calls'] == 0
            assert row['actual_global_positions'] == list(range(cfg['parent_cursor'] + (u - 1) * 60, cfg['parent_cursor'] + u * 60))
            counts.update(row['g_values'])
        assert dict(counts) == {int(k): v for k, v in payload['branch_strength_presentations'].items()}
        assert sum(counts.values()) == target * 60
        checkpoints[branch] = dict(last, actual_strength_counts=dict(counts), formal_updates=target,
                                   all_complete_checkpoint_files_retained=len(list((out / 'checkpoints').glob('U*.pt'))))
        nodes = [0, 1000, 3000, 5000] + ([6000, 8000, 10000] if target == 10000 else [])
        for u in nodes:
            location = baseline if u == 0 else out / 'validation' / f'u{u:06d}_five_g_1000'
            summary = json.loads((location / 'SUMMARY.json').read_text())
            rows = [json.loads(line) for line in (location / 'PER_IMAGE.jsonl').read_text().splitlines()]
            assert summary['count'] == len(rows) == 5000 and all(r['nfe'] == 3 for r in rows)
            assert Counter(r['g'] for r in rows) == {.5: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000}
            summaries[(branch, u)] = summary
            data[(branch, u)] = {(r['sample_id'], r['g']): r for r in rows}
    metrics = ['psnr', 'ssim', 'lpips', 'mae']
    with (EX / 'VALIDATION_CURVES.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['branch', 'local_update', 'global_update', 'g', *metrics])
        writer.writeheader()
        for (branch, u), summary in summaries.items():
            for g, values in summary['by_g'].items():
                writer.writerow(dict(branch=branch, local_update=u, global_update=BASE + u, g=g,
                                     **{m: values[m] for m in metrics}))
    for u in [1000, 3000, 5000]:
        with (EX / f'PAIRED_u{u:06d}.csv').open('w', newline='') as f:
            columns = ['sample_id', 'source_id', 'g', 'seed']
            columns += [label + '_' + m for label in ['parent', 'Control', 'G4_biased', 'biased_minus_Control'] for m in metrics]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for key in sorted(data[('Control', u)], key=lambda k: (k[1], k[0])):
                c, b, p = data[('Control', u)][key], data[('G4_biased', u)][key], data[('Control', 0)][key]
                assert c['source_id'] == b['source_id'] == p['source_id'] and c['seed'] == b['seed'] == p['seed']
                assert c['input_sha256'] == b['input_sha256'] == p['input_sha256']
                row = {k: c[k] for k in ['sample_id', 'source_id', 'g', 'seed']}
                for label, r in [('parent', p), ('Control', c), ('G4_biased', b)]:
                    row.update({label + '_' + m: r[m] for m in metrics})
                row.update({'biased_minus_Control_' + m: b[m] - c[m] for m in metrics})
                writer.writerow(row)
    # Scientific figures use ordinary plotting; no AI-generated image content.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, metric in enumerate(['psnr', 'ssim', 'lpips']):
        for branch, style in [('Control', '-o'), ('G4_biased', '-s')]:
            nodes = sorted(u for b, u in summaries if b == branch)
            axes[0, col].plot(nodes, [summaries[(branch, u)]['by_g']['4'][metric] for u in nodes], style, label=branch)
            for g in ['1', '2', '3']:
                axes[1, col].plot(nodes, [summaries[(branch, u)]['by_g'][g][metric] for u in nodes],
                                  '--' if branch == 'Control' else '-', label=f'{branch} G{g}')
        axes[0, col].set_title('G4 ' + metric)
        axes[1, col].set_title('G1-G3 ' + metric)
        for ax in axes[:, col]:
            ax.set_xlabel('New optimizer updates')
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
    fig.savefig(EX / 'G4_AND_WEAK_VALIDATION_CURVES.png', dpi=160)
    plt.close(fig)
    # Fixed-source gallery is determined by the preserved parent visual IDs.
    from PIL import Image, ImageDraw
    visual_keys = [k for k, r in data[('Control', 0)].items() if 'prediction_npy' in r]
    gallery = EX / 'gallery'
    gallery.mkdir(exist_ok=True)
    figures = []
    for key in sorted(visual_keys, key=lambda k: (k[1], k[0])):
        p, c, b = [data[k][key] for k in [('Control', 0), ('Control', 5000), ('G4_biased', 5000)]]
        for r in [p, c, b]:
            assert sha(r['prediction_npy']) == r['prediction_sha256']
        arrays = [np.load(p['clean_path'], allow_pickle=False), np.load(p['input_path'], allow_pickle=False)]
        arrays += [np.load(r['prediction_npy'], allow_pickle=False) for r in [p, c, b]]
        canvas = Image.new('RGB', (256 * 5, 280), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (arr, title) in enumerate(zip(arrays, ['GT', 'Input', 'Mixed U157505', 'Control +5000', 'G4 biased +5000'])):
            im = np.rint(np.clip(arr.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            draw.text((256 * i + 4, 4), title, fill='black')
            canvas.paste(Image.fromarray(im), (256 * i, 24))
        name = f'G{key[1]:g}_{key[0]}.jpg'
        canvas.save(gallery / name, quality=95)
        figures.append('<figure><figcaption>' + name + '</figcaption><img loading="lazy" src="gallery/' + name + '"></figure>')
    (EX / 'PAIRED_GALLERY.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:system-ui;margin:24px}img{max-width:100%}</style><h1>D0 G4重加权：同预算5000更新配对</h1><p>固定原32来源×五档；仅显示clip，指标来自原浮点输出。训练网络/Loss/扩散不变。</p>' + ''.join(figures))
    report = ['# D0-Mixed G4重加权有限实验', '',
              '共同父U157505完整raw/EMA/Adam/scheduler/scaler/四逻辑rank RNG继承。Control原MixedOrder，G4-biased只改变现有同Clean/ξ缓存的强度选择；native残差L1系数1，LR2e-6，micro5×acc12=effective60。训练1NFE、验证原3NFE，网络和Loss未改，A/B未启动。', '',
              '物理四卡改为两组各单卡并行，保留原4逻辑rank×3micro的数据和种子；12micro均值与4组三micro均值做过梯度数值验收，未声称真实HCCL逐位重现。', '',
              '|强度|Mixed父PSNR|Control 5k PSNR|G4-biased 5k PSNR|biased-Control|', '|---|---:|---:|---:|---:|']
    for g in ['0.5', '1', '2', '3', '4']:
        p, c, b = [summaries[k]['by_g'][g]['psnr'] for k in [('Control', 0), ('Control', 5000), ('G4_biased', 5000)]]
        report.append(f'|G{g}|{p:.8f}|{c:.8f}|{b:.8f}|{b-c:+.8f}|')
    delta = summaries[('G4_biased', 5000)]['by_g']['4']['psnr'] - summaries[('Control', 5000)]['by_g']['4']['psnr']
    extension_decision = json.loads((EX / 'EXTENSION_DECISION.json').read_text())
    recommendation = ('同预算证据支持G4重加权改善且弱档代价处于预先允许范围；建议结束本轮D0参数搜索，'
                      '将通过质量检查的候选提交用户决定是否进入A/B。A/B本身能否增益仍需其正式实验，未自动启动。'
                      if extension_decision['extend_G4_biased'] else
                      '本轮未满足预先冻结的G4增益与弱档保留条件，不支持继续搜索采样比例；建议结束本轮D0追加。'
                      '当前证据不足以无条件建议转入A/B，应先由用户确认已有候选的G4/弱档取舍；A/B保持暂停。')
    report += ['', f'同预算5k的G4 PSNR差为{delta:+.8f} dB。SSIM/LPIPS/MAE完整逐图与曲线见CSV；不能只凭PSNR或训练Loss宣布整体成功。', '',
               '原U95544/U109544/U157505及旧best均未覆盖。追加Gate按预冻结规则执行，普通质量下降未导致初始5k提前结束。', '',
               '追加决定：`' + json.dumps(extension_decision, ensure_ascii=False) + '`', '',
               '若biased扩展到10k，该终点对Control 5k属于不匹配预算，只能作额外续训观察；数据比例的主因果对照仍是双方5k。', '',
               '当前仅报告此单一训练随机序列、固定250源×4realizations验证；源级配对bootstrap不替代训练seed方差。u0明确复用已验证完整父结果，未冒称重新执行两次5000图。', '',
               '研究判断：' + recommendation, '',
               '最后完整checkpoint：`' + json.dumps(checkpoints, ensure_ascii=False) + '`']
    (EX / 'D0_G4_REWEIGHT_FINAL_REPORT_ZH.md').write_text('\n'.join(report) + '\n')
    write_json(EX / 'FINAL_ACCEPTANCE.json', dict(status='PASS_FINITE_BUDGETS_COMPLETE_PENDING_VISUAL_QA_LOCAL_SYNC',
        checkpoints=checkpoints, all_train_workers_exited=True, matching_5000_nodes_five_g_complete=True,
        report_sha256=sha(EX / 'D0_G4_REWEIGHT_FINAL_REPORT_ZH.md'), test_used=False, A_B_started=False,
        quality_success_not_implied_by_engineering_PASS=True))


if __name__ == '__main__':
    main()

"""Summarize completed native UniDB experiment; never invent missing results."""
import csv
import json
from pathlib import Path

import numpy as np
from core import ROOT,sha,write


def main():
    run=ROOT/'runs/formal_until_20261103_noon'
    complete=json.loads((run/'TRAINING_AND_VALIDATION_COMPLETE.json').read_text());assert complete['status']=='PASS'
    audit=json.loads((ROOT/'evidence/FINAL_CHECKPOINT_AUDIT.json').read_text());assert audit['status']=='PASS'
    final=json.loads(Path(complete['final_val']).read_text());assert final['full'] and final['count']==5000
    candidates=json.loads((run/'FULL_CANDIDATES.json').read_text());best=json.loads((run/'BEST.json').read_text())
    selected=max(candidates,key=lambda c:(c['validation']['by_g']['4']['psnr'],-c['validation']['by_g']['4']['lpips'],-c['update']))
    assert selected['update']==best['update']
    last=json.loads((run/'LATEST.json').read_text());assert sha(Path(last['path']))==last['sha256']
    cfg=json.loads((run/'CONFIG.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    rows=[]
    for candidate in candidates:
        for g,metrics in candidate['validation']['by_g'].items():
            rows.append({'update':candidate['update'],'g':g,**metrics})
    with (ROOT/'FULL_VALIDATION.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['update','g','psnr','ssim','lpips','mae']);writer.writeheader();writer.writerows(rows)
    training=[json.loads(s) for s in (run/'UPDATES_rank0.jsonl').read_text().splitlines()]
    assert [r['update'] for r in training]==list(range(1,result['global_update']+1))
    assert result['presentations']==result['global_update']*cfg['effective_batch']
    with (ROOT/'TRAINING_CURVE.csv').open('w') as f:
        names=['update','cursor','loss','grad_norm','seconds','lr']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        writer.writerows({k:r[k] for k in names} for r in training)
    lines=['# UniDB SR初始化湍流D0：最终结果','',
        '本轮为官方UniDB SR gamma1e7完整网络全参数微调。网络、桥过程和原生一步匹配L1保持；无task prompt、Adapter、raw-ξ头或A/B物理递归。','',
        f"正式完成{result['global_update']}更新、{result['presentations']}次样本呈现，有效batch{cfg['effective_batch']}；停止原因{complete['stop_reason']}。",
        '截止时间为2026-11-03 12:00 Asia/Shanghai，截止后只有安全保存与最终验证，不新增训练更新。','',
        '训练使用原Train10000×4ξ×G1–G4完整160K组合。每档固定Val1000，250来源×4实例，EMA、原生100步SDE、RGB浮点评价，无额外clamp/crop/resize；Test未参与。','',
        f"按完整G4 Val1000 PSNR选择的best为U{best['update']}；并列按G4 LPIPS及更早update。小监控子集未混入完整选模。",'',
        '|强度|best PSNR|best SSIM|best LPIPS|latest PSNR|latest SSIM|latest LPIPS|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in ['0.5','1','2','3','4']:
        b=best['validation']['by_g'][g];v=final['by_g'][g]
        lines.append(f"|G{g}|{b['psnr']:.6f}|{b['ssim']:.6f}|{b['lpips']:.6f}|{v['psnr']:.6f}|{v['ssim']:.6f}|{v['lpips']:.6f}|")
    lines+=['',f"旧DiffUIR G4 U95544参考21.165142dB；本轮best差值{best['validation']['by_g']['4']['psnr']-21.16514236838766:+.6f}dB。两者训练量和方法不同，不能据此把变化独立归因于骨干。",'',
        '指标与固定图需要一起判断。工程验收通过不等于强湍流恢复或真实泛化成功；本轮未新增外部测试评价。','',
        f"latest checkpoint：`{last['path']}`，SHA256 `{last['sha256']}`。",'',
        f"完整状态CPU独立读回：{audit['tensors_finite_checked']}个张量，{audit['Adam_states']}项Adam状态，{audit['RNG_slots']}个rank RNG槽。",'',
        '全部临时测速、恢复测试和Pilot状态与正式训练隔离；正式从官方SR权重重新初始化。旧V5/V6资产、历史STOP和外部服务保持。','']
    (ROOT/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines))
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(1,2,figsize=(13,4.5))
        x=np.array([r['update'] for r in training]);y=np.array([r['loss'] for r in training])
        width=min(200,len(y));smooth=np.convolve(y,np.ones(width)/width,mode='valid') if width else y
        axes[0].plot(x[width-1:],smooth);axes[0].set(xlabel='Optimizer update',ylabel='Native matching L1',title=f'Train loss ({width}-update moving mean)')
        for g in ['0.5','1','2','3','4']:
            values=[r for r in rows if r['g']==g];axes[1].plot([r['update'] for r in values],[r['psnr'] for r in values],marker='o',label='G'+g)
        axes[1].set(xlabel='Optimizer update',ylabel='PSNR (dB)',title='Full Val1000 per strength');axes[1].legend()
        figure.tight_layout();figure.savefig(ROOT/'FINAL_CURVES.png',dpi=180);plt.close(figure)
    except ImportError as error:write(ROOT/'PLOT_UNAVAILABLE.json',{'reason':str(error)})
    write(ROOT/'FINAL_DELIVERY_SHA256.json',{'files':{p.name:sha(p) for p in [ROOT/'FINAL_REPORT_ZH.md',ROOT/'FULL_VALIDATION.csv',ROOT/'TRAINING_CURVE.csv']},
        'best_update':best['update'],'latest_update':result['global_update'],'all_five_strength_final_complete':True})
    print('FINAL_REPORT_READY',flush=True)


if __name__=='__main__':main()

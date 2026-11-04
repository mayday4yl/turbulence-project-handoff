"""Audit immutable inputs and summarize completed finite speed experiments."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from training.state import atomic_json

def read(name):return json.loads((HERE/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

workflow=read('FOUR_CARD_WORKFLOW.json')
if workflow['status']!='BOUNDED_BENCHMARKS_COMPLETE':raise ValueError('Benchmarks incomplete')
auth=json.loads((ROOT/'protocol/FORMAL_TRAINING_AUTHORIZATION.json').read_text())
if auth['formal_training_approved']:raise ValueError('Unexpected formal authorization')
manifest=json.loads((ROOT/'protocol/PROPOSED_CODE_MANIFEST.json').read_text())
mismatches=[name for name,expected in manifest['files'].items() if sha(ROOT/name)!=expected]
if mismatches:raise ValueError('Original frozen source changed: '+repr(mismatches))
if list((ROOT/'runs').glob('**/STATUS.json')):raise ValueError('Unexpected formal run status files')
u0sha=sha(ROOT/'weights/adapted_U0.pt')
if u0sha!='27974bbd20ed952ad392e9da83413b149bbf7f15f35340f54ac99e6f50e543a1':raise ValueError('U0 changed')

results={};rows=[];records={}
for method in ['D0','R0']:
    for backend in ['native','adjoint']:
        name=f'benchmark_{method}_{backend}_w4_b2.json';r=read(name)
        if r['status']!='PASS' or r['checkpoint_compatibility']['status']!='PASS':raise ValueError(name)
        if r['initial_checkpoint_sha256']!=u0sha:raise ValueError('Wrong benchmark U0')
        times=[v['max_rank_elapsed_seconds'] for v in r['measurements'] if v['phase']=='timed']
        if len(times)!=2:raise ValueError('Unexpected timing count')
        results[(method,backend)]={'seconds':statistics.mean(times),'measurements_seconds':times,
            'peak_allocated_gib':r['timed_summary']['max_peak_allocated_bytes']/2**30,
            'peak_reserved_gib':r['timed_summary']['max_peak_reserved_bytes']/2**30,
            'checkpoint_compatibility':'PASS'}
        records[name]=r
    a,b=results[(method,'native')],results[(method,'adjoint')]
    if records[f'benchmark_{method}_native_w4_b2.json']['data_identity']!=records[f'benchmark_{method}_adjoint_w4_b2.json']['data_identity']:
        raise ValueError('Different measured pairs')
    rows.append({'method':method,'native':a,'adjoint':b,'throughput_speedup':a['seconds']/b['seconds'],
        'time_reduction_fraction':1-b['seconds']/a['seconds'],
        'max_mean_loss_difference_over_three_updates':max(abs(x['mean_loss_across_ranks']-y['mean_loss_across_ranks'])
            for x,y in zip(records[f'benchmark_{method}_native_w4_b2.json']['measurements'],records[f'benchmark_{method}_adjoint_w4_b2.json']['measurements']))})
native_days=(results[('D0','native')]['seconds']+2*results[('R0','native')]['seconds'])*10000/86400
new_days=(results[('D0','adjoint')]['seconds']+2*results[('R0','adjoint')]['seconds'])*10000/86400
report={'status':'BOUNDED_OPTIMIZATION_VERIFIED_NOT_FORMAL_TRAINING','formal_optimizer_updates':0,
    'completed_unix':time.time(),'comparisons':rows,'projected_three_group_pure_training_days':{'native':native_days,'adjoint':new_days},
    'projection_scope':'D0 + R0 + estimated same-cost R0-D0Init, 10k each; excludes full validation, disk I/O, saving and final refinement; short timing only',
    'unchanged_frozen_files':len(manifest['files']),'frozen_source_mismatches':mismatches,'u0_sha256':u0sha,
    'formal_training_authorization':auth,'production_default_changed':False,
    'source_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
    'numerical_evidence':{n:sha(HERE/n) for n in ['primitive_cpu64.json','primitive_npu_fp32.json','simulator_parity.json','full_bptt_parity.json']},
    'benchmark_sha256':{name:sha(HERE/name) for name in records}}
library=Path('/usr/local/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/conv2d_backprop_input_d.py')
report['backend_source_evidence']={'path':str(library),'sha256':sha(library),
    'entry':'conv2d_backprop_input_d','observed_delegation':'conv2d_transpose_static_impl followed by tbe.dsl.auto_schedule',
    'entry_line':219,'internal_tiling_root_cause':'NOT_ESTABLISHED',
    'actual_device_kernel_path_confirmed_by':'primitive_profile_summary.json'}
atomic_json(report,HERE/'RESULTS.json')
table='\n'.join(f"| {r['method']} | {r['native']['seconds']:.2f} | {r['adjoint']['seconds']:.2f} | {r['throughput_speedup']:.2f} 倍 | {r['time_reduction_fraction']:.1%} | {r['adjoint']['peak_allocated_gib']:.2f} |" for r in rows)
full=read('full_bptt_parity.json')
md=f'''# V5 NPU FP32 大卷积反向优化结果

保持方法定义和 Simulator FP32 的等价计算优化已通过本轮有界检查。正式训练为 0 updates，原默认实现和待确认设置保持原样。

## 实测收益

同一 U0、同一组 32 个真实训练样本、4 张 Ascend 910B3、每卡 microbatch2 × 累积4；网络 BF16、Simulator/loss FP32、HF32 OFF。R0 仍只重算 7 个块，D0 不重算。每项 1 次预热 + 2 次计时更新，全部丢弃。以下为两次计时均值，完整 checkpoint 写读检查在计时外执行且全部通过。

| 方法 | 原实现 秒/update | 优化后 秒/update | 吞吐倍率 | 耗时减少 | 优化后峰值 allocated GiB/卡 |
|---|---:|---:|---:|---:|---:|
{table}

各组跑满 10k 的三组合计纯训练外推：约 {native_days:.2f} 天 → {new_days:.2f} 天。R0-D0Init 尚未运行，按同链路估计。此数值不含完整 Val1000、正常读盘、保存、最终 refinement 或中断，不是最终墙钟承诺。两次计时也不构成长时间稳定性认证。

## 慢在哪里，改了什么

真实 B2 的 100 个 33×33 PSF 核输入梯度，原实现稳态约 2.600 秒，等价实现约 0.080 秒；均值核约 0.376 秒 → 0.014 秒。设备采样中，原 basis 反向核两次合计 5.183 秒，等价普通卷积核两次 0.143 秒；格式转换合计仅约毫秒级。因此这里的主要问题已经定位到当前 CANN `Conv2DBackpropInput` 的设备执行路径，不是仅凭显存、模型大小或 host 等待作推断。

已安装 CANN 8.5.2 的该输入梯度入口使用通用转置卷积实现；对于当前 FP32、33×33、大输出通道/单输入通道形状，这条路径实测低效。具体哪项内部 tiling 或指令安排导致低效仍未由硬件计数器确认，不宣称已经定位到厂商源码的某一缺陷。

`adjoint.py` 保留原前向 `conv2d`，仅重写固定核的输入梯度：对梯度做普通 `conv2d`，权重交换输入输出通道并在两个空间轴翻转，padding=32。反射 padding 的原反向继续由 PyTorch 处理。没有低秩近似、核截断、改变物理参数、关闭梯度或修改 FFT。只支持这里冻结的 stride1、groups1、33×33 核。

## 数值与完整链验证

- CPU64 真实核与反射边界：最大梯度 relative L2 为 6.83e-15；有限差分方向误差均小于 1e-7。
- NPU FP32 真实 B2 原语：这组输入的前向及输入梯度逐值相同。
- 独立完整 Simulator：g=0、1e-4、2.5、4 的输出和 clean/raw 梯度逐值相同；g=0 raw 梯度为零，clean identity VJP 通过。
- 真实 Concertormer 四步：各步 clean、观测匹配输出和 loss 逐值相同；末步到首步 clean/raw 的梯度有限非零且逐值相同，主动断开状态的负对照确实失去连接。
- 全部 1173 个可训练参数有有限梯度，整体梯度 relative L2={full['parameter_gradient_relative_l2']:.3g}，最大绝对差={full['maximum_parameter_gradient_absolute_error']:.3g}。网络仍是既定 BF16，未要求完整多次优化轨迹逐值相同。

检查标准预先写入 [PLAN.md](PLAN.md)，原始数值全部保留，没有通过放宽阈值获得 PASS。首次采样的 stop-in-RECORD 提示保留；两种路径所期望的两次核调用都存在，但 profiler 时间仅作定位，不用于训练吞吐外推。

## 文件与执行边界

- [数值汇总](RESULTS.json)、[完整四步对照](full_bptt_parity.json)、[独立 Simulator 对照](simulator_parity.json)。
- [原语对照](primitive_npu_fp32.json)、[CPU64 参考](primitive_cpu64.json)、[设备采样摘要](primitive_profile_summary.json)。
- 四卡各组 `benchmark_*_w4_b2.json`、`.rank*.json`、`.log` 和 `_implementation.json` 保留。`FOUR_CARD_WORKFLOW.json` 记录命令和时间边界。
- 原冻结清单 {len(manifest['files'])} 个文件 SHA 全部不变，共同 U0 不变；所有正式启动入口仍等待用户确认。

建议把此 FP32 优化候选纳入下一版待确认训练设置。当前只是独立 opt-in 实现和整链实测，未静默替换正式默认、未启动训练；A100 的同链路性能仍待另一窗口按资源授权实测。
'''
(HERE/'REPORT_ZH.md').write_text(md)
print(json.dumps(report,ensure_ascii=False),flush=True)

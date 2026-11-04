# V5 DiffUIR-L Batch、显存与吞吐 Benchmark

日期：2026-10-26，北京时间。仅工程Benchmark，全部执行已结束。远端独立根：`/data2/yanglin/v5_diffuir_l_benchmark_20261026`，主机de36。

**推荐原生D0使用 microbatch=16，gradient accumulation先取1；需要有效batch32时可选择累积2。** 在本次已测档位中，16既是原生训练最大通过档，也是吞吐最高档：NPU2串行复测 **0.7095秒/update、22.550张/秒**，比batch8快9.69%。该推荐是工程吞吐结论，不是恢复效果或学习率/有效batch的质量验收；没有自动开始正式D0。

三步采样反传的最大已测通过batch为 **8**，同卡复测 **1.0407秒/update、7.687张/秒**。它是单次Clean Restoration三步图的诊断，不是V5四步物理递归，不作为D0训练目标。

## 1. 配置、数据与验收口径

- 直接只读复用Stage1已验收的官方DiffUIR实现，commit `c71437ce86ad1b098aa9be4678dc82e8b99649df`。Large64-1248、36,261,187参数、276参数tensor，FP32、RGB256×256，HF32关闭；没有AMP、降分辨率或gradient checkpointing。
- 每档在独立进程内从官方 `model-300.pt` **raw**权重严格初始化，文件SHA `b9d07ab6b32d37d895be65c50e99bbe6b2ae6e32823e7304ec2167bc5f18843f`；fresh Adam lr8e−5、betas(.9,.99)、clip1，fresh EMA beta.995/update_every10并保留默认warmup100。模型、Adam与EMA显存均计入；更新后副本随进程退出丢弃，**没有保存新训练checkpoint**。
- 原生模式保留官方Condition、随机0–999扩散索引、q_sample与Residual L1；每样本NFE1。所有batch采用accumulation1，每次计时都包含真实forward、backward、Adam.step和EMA.update。
- sampling模式复用Stage1可微包装，只开放原sample/ddim_sample的no_grad边界；保留官方init、999→665→332三步、clamp和公式。最终Clean与GT的mean L1仅用于诊断，每次也执行真实Adam/EMA更新。开启Y4输入梯度，并retain三个UNet输出梯度；原生模式不额外求输入梯度。
- 训练人口严格核验为 **2000独立Clean × 4 realizations =8000对、50类、固定g=4**。从完整Train8000按seed20261026冻结2048候选对，覆盖1397source；核验3445个唯一训练数组的SHA、FP32 CHW256与有限性。没有读取Test图像，也没有使用Val4/Val50做本次Benchmark。
- 全部成功进程合计 **144次测试性更新、986次样本呈现**；各档从同一冻结顺序前缀开始，合并实际只覆盖 **240个不同pair、229source、50类**。不能把8000候选人口或986次重复呈现称为已遍历整个训练集。
- 首轮每档3次warmup+8次测量；NPU2最终复测3次warmup+12次测量。warmup仍执行真实更新，排除于吞吐统计但纳入显存峰值。EMA默认在U11复制一次，真实开销保留。

## 2. 显存容量与安全策略

设备实测为Ascend910B3，`npu-smi`物理容量 **64GiB**，启动前系统HBM约3.33GiB、可用约60.66GiB；runtime `mem_get_info.total`为65,452,113,920字节，约60.96GiB。这是不同统计层级，不能混作同一总容量。

先核实2–7号卡无进程后六卡并行小档位，后续三卡并行大档位，最后在NPU2依次复测native8/native16/sampling8。0/1上的其他作业未干预。每档进程退出后检查其PID不再占用NPU。

默认保留 **启动前可用HBM的20%**，并从allocator预算额外扣1GiB用于driver/runtime；硬allocator配额约47.53GiB。sampling8的实测外推超过20%目标预算，按用户允许的15%–20%范围，另行冻结 **15%可用HBM硬底线+1GiB guard**，allocator配额约50.57GiB。该边界设置不回改先前档位的20%标签；见[边界协议](<../../NOT_INCLUDED.md#ref-2014a9404da267525bac>)。

sampling8 NPU2实测HBM采样峰值53.280GiB，最少余量10.720GiB，为最初可用空间的17.67%（约占物理容量16.75%），高于冻结15%底线。native16仍余22.382GiB，显存余量更充足。

PyTorch allocated/reserved为框架high-water峰值。设备HBM是外部npu-smi**实际采样最大值**，不是无间隔瞬时峰值；轮询目标等待0.5秒，含命令耗时后的实际间隔中位1.649秒、最大2.048秒。三者分别报告，未把allocator当作整卡显存。

## 3. 原生D0 Batch—显存—吞吐

主表每档优先采用最终NPU2串行复测；1/2/4为并行筛查。所有已测行均完整通过warmup与计时更新。

| microbatch | NPU / 运行 | allocated峰值 GiB | reserved峰值 GiB | 设备HBM采样峰值 GiB | 秒/update均值 | images/sec | 状态 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 2 / 并行筛查 | 2.757 | 3.016 | 6.479 | 0.1835 | 5.449 | 实测通过 |
| 2 | 3 / 并行筛查 | 4.700 | 5.418 | 8.888 | 0.2040 | 9.806 | 实测通过 |
| 4 | 4 / 并行筛查 | 8.578 | 10.127 | 13.590 | 0.2446 | 16.353 | 实测通过 |
| 8 | 2 / 串行复测 | 16.366 | 19.520 | 22.987 | 0.3891 | 20.558 | 实测通过 |
| 16 | 2 / 串行复测 | 31.949 | 38.150 | 41.618 | 0.7095 | 22.550 | 实测通过 |
| 32及更大 | 未启动 | — | — | — | — | — | 安全余量停止，未测试 |

在已测序列1/2/4/8/16/32中，**最大稳定通过档16，吞吐最佳档16**。32未启动；由8/16的reserved增长线性估算约75.41GiB，已经超出物理容量及安全配额。这个估算只用于决定停止，不填作batch32实测显存，也不能据此宣称找到了所有整数batch的硬件极限。未进一步测20/24等中间值或64/128。

## 4. 完整三步采样反传

| microbatch | NPU / 运行 | allocated峰值 GiB | reserved峰值 GiB | 设备HBM采样峰值 GiB | 秒/update均值 | images/sec | 状态 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 5 / 并行筛查 | 6.924 | 7.340 | 10.802 | 0.4364 | 2.291 | 实测通过 |
| 2 | 6 / 并行筛查 | 12.535 | 13.453 | 16.917 | 0.4593 | 4.354 | 实测通过 |
| 4 | 7 / 并行筛查 | 23.755 | 25.609 | 29.072 | 0.6018 | 6.647 | 实测通过 |
| 8 | 2 / 串行复测 | 46.272 | 49.811 | 53.280 | 1.0407 | 7.687 | 实测通过 |
| 16及更大 | 未启动 | — | — | — | — | — | 安全余量停止，未测试 |

已测最大稳定档为8；16未启动，依据4/8的reserved增量预测约98.21GiB，超出安全预算。8属于接近安全边界的通过档，4是更宽裕的开发档。未来若引入更多图分支、不同损失或四次物理递归，不能直接复用batch8作为已验收设置。

每次sampling更新均实测NFE3；三个UNet输出的梯度分别存在、有限且非零，Y4输入梯度也有限非零。没有通过detach、STE或改写公式来获得通过。所有已测模式每步276参数tensor梯度存在且有限，更新后参数全有限；结束时276参数tensor相对官方初始化均改变，Adam状态全部达到对应11或15更新。详细逐步范数在`runs/*/updates.jsonl`和[RESULTS.json](<RESULTS.json>)。

同NPU2、同microbatch8比较：

| 指标 | 原生随机时刻训练 | 三步采样反传 | 后者/前者 |
|---|---:|---:|---:|
| 含审查更新秒数 | 0.3891 | 1.0407 | 2.67× |
| allocated峰值GiB | 16.366 | 46.272 | 2.83× |
| reserved峰值GiB | 19.520 | 49.811 | 2.55× |

sampling额外求输入梯度、保留阶段输出梯度，并使用不同诊断Loss；上述比值是这两条实际执行链的成本差异，不能全部归因于NFE从1变3，也不代表四步外层B的成本。

## 5. 全部成功运行的计时与稳定性

| 模式 | B | 卡 | 运行 | warmup+计时更新 | 均值秒 | median秒 | p90秒 | CV | images/sec |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| native | 1 | 2 | 并行筛查 | 3+8 | 0.1835 | 0.1816 | 0.1876 | 3.27% | 5.449 |
| native | 2 | 3 | 并行筛查 | 3+8 | 0.2040 | 0.1974 | 0.2198 | 8.28% | 9.806 |
| native | 4 | 4 | 并行筛查 | 3+8 | 0.2446 | 0.2420 | 0.2488 | 2.81% | 16.353 |
| sampling | 1 | 5 | 并行筛查 | 3+8 | 0.4364 | 0.4335 | 0.4463 | 1.35% | 2.291 |
| sampling | 2 | 6 | 并行筛查 | 3+8 | 0.4593 | 0.4550 | 0.4723 | 2.30% | 4.354 |
| sampling | 4 | 7 | 并行筛查 | 3+8 | 0.6018 | 0.5989 | 0.6087 | 1.26% | 6.647 |
| native | 8 | 2 | 并行筛查 | 3+8 | 0.3934 | 0.3917 | 0.4000 | 1.43% | 20.334 |
| native | 16 | 3 | 并行筛查 | 3+8 | 0.7127 | 0.7111 | 0.7194 | 1.03% | 22.450 |
| sampling | 8 | 5 | 并行筛查 | 3+8 | 1.0421 | 1.0398 | 1.0480 | 0.65% | 7.677 |
| native | 8 | 2 | confirm串行 | 3+12 | 0.3891 | 0.3874 | 0.3896 | 1.30% | 20.558 |
| native | 16 | 2 | confirm串行 | 3+12 | 0.7095 | 0.7076 | 0.7131 | 1.01% | 22.550 |
| sampling | 8 | 2 | confirm串行 | 3+12 | 1.0407 | 1.0364 | 1.0569 | 0.96% | 7.687 |

吞吐公式为 `B × timed_updates / sum(update_seconds)`。主更新计时包括CPU stack/H2D、播种、zero_grad、前后向、所有有限性检查、clip、Adam、EMA及同步；不包含一次性NPY读取/哈希、模型加载、JSON写盘及最终CPU逐参数比对。因此它是预加载训练数据上的完整更新Benchmark，不是含磁盘DataLoader/验证/保存的正式训练全流程吞吐。

机器记录还给出排除梯度/参数诊断的forward+backward+clip/Adam/EMA计算区间。NPU2 native8/native16/sampling8分别为22.867/24.009/8.025张/秒；推荐使用主表较保守的含审查计时，不用计算区间冒充完整update。

## 6. 失败、未测与保护

共18个实际启动进程：**12个PASS，6个环境启动FAIL，0个OOM**；另有两个明确的下一档SKIPPED_SAFETY记录，所有更大档位均未启动。

| 失败组 | 对应卡与batch | 实际更新数 | 原因及处理 |
|---|---|---:|---|
| 首次native | NPU2/B1、NPU3/B2、NPU4/B4 | 各0 | 新launcher覆盖CANN PYTHONPATH，tbe导入失败；保存失败目录和日志 |
| 首次sampling | NPU5/B1、NPU6/B2、NPU7/B4 | 各0 | 同一环境原因；不是模型反向算子失败或OOM |
| 修复后的六档 | 独立`*_envfix`目录 | 各11 | 仅让新launcher保留set_env.sh设置的PYTHONPATH，全部通过 |

完整异常保存在`logs/`和各失败`runs/*/FAILURE_TRACEBACK.txt`；旧launcher副本、修复原因及前后SHA保留于[evidence/LAUNCHER_ENV_FIX.json](<../../NOT_INCLUDED.md#ref-c59f1b3226b11159e2cb>)。没有覆盖失败运行，没有借此更改模型、Loss或旧环境。

[最终保护对比](<../../NOT_INCLUDED.md#ref-8a34295eb84db0f7e50d>)PASS：9,552项受保护文件/资产记录，3,543项内容SHA，6,009项size/mtime/inode等元信息，无新增、缺失或变化。包含Stage1/Pilot、旧V5 D0/A/B及ResShift的相应资产；大checkpoint并非全部重新哈希，官方DiffUIR权重另由每档strict load完整核对SHA。保护范围排除环境/cache/.git/bytecode，细节见前后快照。测试结束后的NPU快照确认2–7无运行进程。

## 7. D0建议与停止边界

1. **推荐microbatch16。** NPU2实测吞吐最佳，约22.55张/秒；设备HBM约41.62GiB，仍余22.38GiB。batch8速度低约8.84%，但显存更省，可作为以后增加额外监控/模块时的保守退路；本轮没有增加这些模块。
2. **推荐accumulation范围1–2。** 先以1得到有效batch16；若方法设计明确需要有效batch32，可采用2。累积4对应有效batch64，可作为将来的备选配置，但本次没有实测accum2/4吞吐、稳定性或恢复质量，也不建议仅为了用满显存而增加累积。累积不能降低固定microbatch的激活峰值。
3. **采样级Clean Loss目前仅有工程可行性证据。** batch8可执行，batch4留更多空间；不把这项Loss加入D0，不推进A/B或物理递归。

全部测试副本已丢弃、相关NPU进程已退出。未开始正式D0、A或B，未新增训练checkpoint，等待下一步方法设计决定。

## 8. 配置与运行命令

每次启动均先通过`run_wave.py`的实时空闲和内存检查，完整逐档argv、PID、实际cap和设备快照位于`waves/*/LAUNCH.json`。以下是已执行命令的记录，不表示应覆盖重跑；输出目录存在时脚本拒绝覆盖。

```bash
cd /data2/yanglin/v5_diffuir_l_benchmark_20261026
/data2/yanglin/v5_diffuir_l_stage1_20261026/env/bin/python -B scripts/prepare_benchmark.py prepare
/data2/yanglin/v5_diffuir_l_stage1_20261026/env/bin/python -B scripts/prepare_benchmark.py snapshot --name PROTECTED_BEFORE.json
python3 -B scripts/run_wave.py --plan protocol/WAVE01.json   # 保留的环境失败波次
python3 -B scripts/run_wave.py --plan protocol/WAVE01B.json  # 修复后六卡并行
python3 -B scripts/run_wave.py --plan protocol/WAVE02.json   # 大batch三卡并行
python3 -B scripts/run_wave.py --plan protocol/WAVE03.json   # NPU2 native8
python3 -B scripts/run_wave.py --plan protocol/WAVE04.json   # NPU2 native16
python3 -B scripts/run_wave.py --plan protocol/WAVE05.json   # NPU2 sampling8
/data2/yanglin/v5_diffuir_l_stage1_20261026/env/bin/python -B scripts/prepare_benchmark.py snapshot --name PROTECTED_AFTER.json
/data2/yanglin/v5_diffuir_l_stage1_20261026/env/bin/python -B scripts/prepare_benchmark.py compare
python3 -B scripts/summarize_results.py
python3 -B scripts/write_benchmark_report.py
```

环境复用已验收venv（torch2.7.1+cpu / torch_npu2.7.1.post2）；新launcher禁写bytecode，CANN/Torch缓存落新Benchmark目录。没有安装或改动旧依赖。

- [机器汇总JSON](<RESULTS.json>)、[完整CSV](<../../NOT_INCLUDED.md#ref-40ccf6ba42bd3a887dc6>)：包括全部成功、失败及安全未测记录。
- [初始协议](<../../NOT_INCLUDED.md#ref-6b5f508a21a1e077a23b>)、[15%边界补充](<../../NOT_INCLUDED.md#ref-2014a9404da267525bac>)、[安全停止依据](<../../NOT_INCLUDED.md#ref-81868eef7cab85a5b548>)：解释配置版本与未测档位。
- [完整Train8000身份](<../../NOT_INCLUDED.md#ref-b417e849958cf772690d>)、[Benchmark候选选择](<../../NOT_INCLUDED.md#ref-4d3d949f3ff1a22a1210>)：候选人口与实际呈现分开。
- [独立方法复核](<../../NOT_INCLUDED.md#ref-dcb8cdb5ed4778524e21>)、[独立结果复核](<../../NOT_INCLUDED.md#ref-a7f98b42e8e4d80dae8c>)。

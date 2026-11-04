# V0.31 Stage 1 配置修复后实现与验收报告

日期：2026-09-17（Asia/Shanghai）。本报告由最终证据生成，并经独立复核确认。

## 结论与授权

**Stage 1 以 FP32 通过已定义范围的实现验收。** 原生产 FP32 有限差分失败保留；局部 tilt 导数通过同点高精度条件参考、算子 VJP 以及预声明双尺度差分得到核验，不能把旧失败改写为通过。BF16 数值对照不合格，按补充协议排除。

机器状态：`stage1_status=v031_implementation_acceptance_passed`；正式 updates=0、smoke updates=0。

真实完整反向、9 项终端梯度、3 处断图负对照、mixed-G 聚合、FP32 MB1/2 前反向预检、33 项 CPU 接口测试均通过。网络、simulator 数学/资产、数据、loss、rollout 和 full-BPTT 未改变；backend root cause 未宣称彻底定位。

用户最新显式指令已经激活 `AUTO_TO_1000`，覆盖统一补充中的 PREPARE_ONLY、旧文档的阶段间人工停点及 500 暂停点。只有 Stage 1 最终 PASS 才进入 Stage 2；Stage 2 冻结后从头正式训练，update 100 校准一次 inference，500 全量评估后继续，1000 完成全量评估并保存完整状态后暂停。当前 formal updates=0，smoke updates=0，Stage 2/3 尚未执行。

权威顺序：最新明确用户指令 > `docs/V031_AUTONOMY_PREPARATION_AND_HANDOFF_ADDENDUM.md` > `docs/V031_TRAINING_PROTOCOL_ADDENDUM.md` > 三阶段文档。授权及各文档 SHA-256 见 [AUTO_TO_1000_AUTHORIZATION.json](<../../../NOT_INCLUDED.md#ref-3b9a85ee48734e9c4e5d>)。

## 位置、数据与源代码身份

```text
V031_ROOT=/data2/yanglin/v031_overfit100_g4321_n4
SSH=liteserver-de36
DATASET=/data2/yanglin/v03_overfit100_g4321_n4/data/places_p2s_bridge_v03_overfit100_g4321_n4
SOURCE=/data0/yanglin/physical_bridge
ASSETS=/data0/yanglin/vendor/PiRN/assets/utils
PYTHON=/data0/yanglin/envs/physical_bridge/bin/python
LOCAL=/Users/lucas/Desktop/turbulence IR/v031_overfit100_g4321_n4
```

100 sources / 400 trajectories / 1600 FP32 states / 1600 starts。只读加载两条实际 trajectory、所有四起点及 native xi，核对清单、数组哈希、既有完整性/replay 报告；既有 1600 状态精确 replay 最大误差为 0，本阶段未重新生成或全量 replay。`run_manifest.json` 与 `final_inspection_after_fix.json` 完整 JSON 相同。

| 身份 | SHA-256 |
|---|---|
| physical_bridge/model.py | `d306b0a82d5f96ca442e77577cffecf977e02a73d91c968e76c5e88d3831c705` |
| physical_bridge/simulator.py | `937322fbd213d1674f315176daa47887c9d313d5dec517a1a2dbfbb91a4a042c` |
| 初始 network state | `9e377787f59f7d60106c989b8ed8e058b6b02d29456cb1b68e4aeb539b942087` |
| simulator state | `d2a29de8a38c8c26996c176d83f1393b3d3ac85d3a4e0577f717dfdf4bf319b7` |
| 数据 hash_manifest | `b896f9060a8ff012cafe476ec6c0bb1075e6291bfdec1eab7ff791dc8936b03b` |

Python 3.10.12 / torch 2.7.1+cpu / torch_npu 2.7.1.post2 / CANN 8.5.2 / Ascend910B3。原始网络 32,101,673 参数，原构造初始化与零 xi heads，未加载 checkpoint；使用 GroupNorm，无 BatchNorm。各真实验收均验证模型/simulator state 未变。当前仅使用物理 NPU0，每次 guard 先验空闲健康、监测 58 GiB 内存及时限并核对自身 PID/starttime；完成后本任务 NPU 进程释放。未终止其他用户进程。

## 方法实现与修复配置

新实现位于 `code/v031/{core,data,inference,scheduler,checkpoint,runtime}.py`，原 source 只读。完整差异表见 [LEGACY_COMPATIBILITY_AUDIT.md](<../../../NOT_INCLUDED.md#ref-bb0ddaa58baf64df9927>) 与 [LEGACY_INFERENCE_AUDIT.md](<../../../NOT_INCLUDED.md#ref-6ea39eae6305138e14f9>)。四档完整尾序列替换旧六档/curriculum；K 为预测次数；每条样本只读原 clean/Y；mixed-G collator 不 padding。

每次网络只接收当前图像/current g，输出 clean `[B,3,256,256]`、z `[B,36,512,512]`、tilt `[B,512,512,2]`。下一图像由当前预测 clean/xi 在 next g 生成，不 detach、不 teacher forcing、不做 training refinement。每一步包括末步都有全图 RGB mean-MSE LC、对原始 Y/G 的 LR；`mean_paths(mean_steps(LC+0.2*LR))`。不加额外 loss 或 g=0 预测。训练 simulator 权重冻结但输入可微。

BF16 候选仅网络 forward autocast；输出 cast FP32 保图，simulator/状态/loss/reductions 显式关闭 autocast；backward 在其外。正式候选已排除 BF16，保留严格 FP32。

实际生效修复：NPU 初始化后 global JIT=false、仅 `Conv2DBackpropInput` 走 JIT，`must_keep_origin_dtype`，internal format=true，conv/matmul/aclnn HF32=false；新进程导入前 `TASK_QUEUE_ENABLE=0`、`ASCEND_LAUNCH_BLOCKING=0`。保留设备异步执行。详细身份见 [RUNTIME_REPAIR_IDENTITY.json](<../../../NOT_INCLUDED.md#ref-8339b76e752823262e04>)，后端调查见 [NPU_BACKWARD_DIAGNOSIS.md](<../../../NOT_INCLUDED.md#ref-c7290460925d10bd0086>)。这是验证过的兼容性修复，不是完整 backend 根因结论。

修复前后的所有失败保留在对应 `artifacts/<attempt>`、`runs/<attempt>`：原 L1 tiling、global JIT complex dtype 不支持、单算子 JIT 与 host queue 下 E19999 等。没有改 simulator 数学或用布局/自定义梯度补丁绕过。`correctness_05` 和最终入口 `correctness_06` 两次通过；最终 core/runtime/config 还用于 parity、performance 和局部梯度诊断。

## 真实四起点 trace 与 loss

证据：[correctness_06/results.json](<../../../NOT_INCLUDED.md#ref-07ce4dca204aea127290>)。首步为该起点真实 X_G，后续为预测 transition；每步自动断言 current g、next g、原始 G/Y、输入身份、dtype 和 finite。训练 refinement=0。

| G | step | current g | 原始 G | next g | LC | LR |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 4 | 4 | 3 | 0.101941302 | 0.088062480 |
| 4 | 2 | 3 | 4 | 2 | 0.121375121 | 0.106437549 |
| 4 | 3 | 2 | 4 | 1 | 0.121811733 | 0.106226429 |
| 4 | 4 | 1 | 4 | — | 0.120928682 | 0.104611218 |
| 3 | 1 | 3 | 3 | 2 | 0.103422955 | 0.091231525 |
| 3 | 2 | 2 | 3 | 1 | 0.122651719 | 0.109950408 |
| 3 | 3 | 1 | 3 | — | 0.122355103 | 0.108743235 |
| 2 | 1 | 2 | 2 | 1 | 0.099722117 | 0.089964561 |
| 2 | 2 | 1 | 2 | — | 0.120253429 | 0.109402008 |
| 1 | 1 | 1 | 1 | — | 0.098549694 | 0.091157489 |

| G | predictions | logical renders | path loss |
|---:|---:|---:|---:|
| 4 | 4 | 7 | 0.136781096 |
| 3 | 3 | 5 | 0.136804938 |
| 2 | 2 | 3 | 0.129924431 |
| 1 | 1 | 1 | 0.116781190 |

这些是实现数值，不能解释为训练质量。

## 完整跨步反向、负对照与聚合

仅用末步 clean loss=0.120928682387，未混入早步自身 loss 或末步 LR。测量真实前向实际 native 张量，9 项均 connected、finite、非零、FP32。

| 早步 | 分支 | gradient norm | max abs |
|---:|---|---:|---:|
| 1 | clean | 0.01116371993 | 0.0005375914043 |
| 1 | epsilon_zernike | 0.002214465523 | 8.006916323e-05 |
| 1 | epsilon_tilt | 0.005285335705 | 0.00503884349 |
| 2 | clean | 0.004675850738 | 0.0001963173709 |
| 2 | epsilon_zernike | 0.0006934013218 | 1.174190038e-05 |
| 2 | epsilon_tilt | 0.002125669969 | 0.002090240363 |
| 3 | clean | 0.002071186667 | 6.611582648e-05 |
| 3 | epsilon_zernike | 0.0002287256357 | 3.252915121e-06 |
| 3 | epsilon_tilt | 0.0004980600788 | 0.0004538152716 |

三处独立负对照分别只在 prediction 1/2/3 后测试性 detach 下一输入；前向值不变，断点之前目标梯度断开，之后仍非零。该 hook 只存在验收 harness，生产 core 没有 detach 开关。

- boundary 1: passed=True, forward max_abs=0.0, terminal value equal=True。
- boundary 2: passed=True, forward max_abs=0.0, terminal value equal=True。
- boundary 3: passed=True, forward max_abs=0.0, terminal value equal=True。

真实 mixed G=[4,1] 共 5 次预测、8 次逻辑 render；两路径 loss=[0.13678109645843506, 0.11678119003772736]，手算与实现 mean 均为 0.126781135798，差=0.0。混合 batch 与逐路径累积梯度 cosine=0.999999999999779、relative L2=6.64509125e-07，同一连接参数集。

## 局部导数：保留原 FAIL，分解确认数值限制

原 [derivatives_01/results.json](<../../../NOT_INCLUDED.md#ref-c29e10ef5b4d622ace60>) 使用固定 GT 诊断 xi、seed+71 随机输出投影、RMS=1 的两方向/分支、FP32 simulator、CPU64 标量求和；eps=1e-3/3e-3/1e-2，atol=2e-5、rtol=0.05。clean 两方向的六项均通过；z 方向0及 tilt 未通过。这些 FAIL 不覆盖。

随后在结果出现前固定 [尺度复查方案](<../../../NOT_INCLUDED.md#ref-c565c39ea39443d20c31>)，同投影、同六方向 RNG 次序、同容差，固定 common window=3e-4/1e-4。z 两方向在两尺度都通过；tilt 两方向仍未通过，不能靠挑更小 eps 接受。实际 grid 跨格随尺度减少，而 FP32 后段误差开始影响小差分；记录见 [fd_scales_01/results.json](<../../../NOT_INCLUDED.md#ref-0fa858eb2762f3f632a6>) 和 [固定窗口判定](<../../../NOT_INCLUDED.md#ref-fb9da8915410cb7afb29>)。

最后按预先落盘的 [同点条件参考方案](<../../../NOT_INCLUDED.md#ref-eaf211d5e0f0c28c816e>)，固定真实 clean/z 与实际 blur、flow，把 literal-real tilt FFT 的实际 FP32 系数提升为 CPU64，保持原 irfft2 dimensions/crop、2/255 和 align_corners=False。比较真实 NPU VJP、同 cotangent 的独立 FFT VJP、同点 grid VJP、组合导数及中心差分。参考只用于诊断，没有替换生产 renderer。实际 base flow 完全相同；本例 CPU 算术推导的 q32/q64 cell/border 差异为 0；并非对 CANN 内部坐标的直接探测。

| 同点 VJP 对照 | relative L2 | cosine |
|---|---:|---:|
| grid | 1.881333032e-06 | 0.999999999998225 |
| tilt FFT 同 cotangent | 1.978263911e-07 | 0.999999999999981 |
| 组合 tilt | 1.495259128e-05 | 0.999999999888558 |

三项均满足预声明 relative L2≤0.001、cosine≥0.99999。两条原方向共同使用预声明 eps=3e-6/1e-6，全部满足原 scalar 容差；不是逐方向选 epsilon。

| tilt direction | NPU AD | CPU64 AD | eps | CPU64 FD | 对 NPU AD 相对误差 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0110874476167 | 0.0110878157145 | 3e-06 | 0.0111029419177 | 0.139551% |
| 0 | 0.0110874476167 | 0.0110878157145 | 1e-06 | 0.0110847374057 | 0.024444% |
| 1 | 0.00312355744343 | 0.00312393082874 | 3e-06 | 0.00315532683315 | 1.006849% |
| 1 | 0.00312355744343 | 0.00312393082874 | 1e-06 | 0.0031256525701 | 0.067030% |

完整结果、双侧斜率、原可表示 FP32 扰动回放和 fixture 见 [tilt_reference_01/results.json](<../../../NOT_INCLUDED.md#ref-230074b5cf97ef077ed8>)。较大尺度的 CPU64 差分同样会因跨插值分区偏离，缩至共同窗口后收敛；输入表示误差不能单独解释旧失败；非局部效应与后续数值误差共同限制原 FP32 差分。原 FP32 在更小尺度无法稳定分辨。算子 VJP 与同点参考的一致性支持这两条真实方向的 tilt 反向正确，不能外推到所有样本、训练稳定性或质量。

## BF16 决定与 FP32 性能

[parity_01/results.json](<../../../NOT_INCLUDED.md#ref-4ee0eec7fda34a38203a>)：真实 BF16 autocast 可运行，所有状态/loss/梯度 finite、跨步连接存在、权重不变；但 global gradient cosine=0.870548246<0.99，因此排除。total loss 相对差=0.031157%，最大逐步 LC/LR 相对差=0.228037%，都不能替代梯度门禁。FP32 terminal probe 与 correctness_06 相同。没有改 FP16、放宽门槛或继续为 BF16 搜参数。

[performance_fp32_01/results.json](<../../../NOT_INCLUDED.md#ref-f20acbec73a60724cedd>) 明确绑定同版 parity/config/model/sim/data/core/runtime 身份，执行 FP32-only fallback。先 warmup，再每档测两次真实完整 G4/K4 forward+backward。

| MB | mean forward s | mean backward s | mean s/path | peak allocated GiB | peak reserved GiB | 32-path F/B estimate s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3081 | 6.4440 | 6.7521 | 10.174 | 11.246 | 216.067 |
| 2 | 0.4485 | 12.8132 | 6.6309 | 19.881 | 22.025 | 212.187 |

设备 torch 可用总量=60.957 GiB，npu-smi HBM 总量=65536 MB。该测试未分配 optimizer states/EMA，未执行 optimizer.step；32-path 数字仅单设备 F/B 估计，不含数据/通信/clipping/optimizer/EMA/checkpoint。没有 BF16 speedup 或 memory reduction 结论。

**尚未冻结最终 microbatch/accumulation。** 新统一补充要求 Stage 2 在四卡、实际 AdamW/EMA、最长路径内存和相同 mixed-G 流上比较 MB8/4/2/1（accum1/2/4/8），每个可行候选 warmup 后≥3 updates，总 smoke≤60。上述 MB1/2 不能替代这个排序或实际 update 耗时；最终训练、校准和评估用时估计留待该阶段实测。

## CPU 接口、scheduler 与恢复

33 CPU tests 全通过，runner 防止 AdamW.step；覆盖路径/损失缩放、只读 loader/mixed collator、12 项 inference selection、scheduler 和完整状态容器。详细结果见 [cpu_acceptance.json](<../../../NOT_INCLUDED.md#ref-b66fbab4c465b5ffa8ee>)。

| update | LR |
|---:|---:|
| 1 | 2e-6 |
| 50 | 1e-4 |
| 500 | 5.8716070546254966e-5 |
| 1000 | 1e-5 |
| 1001 | 1e-5 |
| 1500 | 1e-5 |

完成 n 次后保存 last_used_lr=f(n)，param groups 已准备 f(n+1)。从虚拟 completed=500/1000 的 synthetic checkpoint 续接至2000，LR 与不中断轨迹 bit-exact。涵盖非空 synthetic AdamW moments、network/EMA、sampler、RNG、计数器、身份和错误 rollback；fixture 显式标记，普通恢复拒绝。这不是实际四卡训练恢复测试。独立 NPU RNG capture/restore 已通过；Stage 2 仍须覆盖真实更新状态续接。见 [SCHEDULER_RESUME.md](<../../../NOT_INCLUDED.md#ref-8612246dd79af956f10e>)。

Inference selection 的12项测试覆盖当前-clean重评分、整组xi逐样本选择、平局继承、首次network、只refine选中候选一次、末步/one-shot不refine。此前 injected refiner 只是接口fixture；本阶段未证明数值求解质量。Stage 2 将按统一补充冻结 update100 一次性校准规则（3 LR×3预算、24固定probes、纯观测残差选参），真实 LR/步数在该预定时点选出，不能提前宣称已完成校准。

## 复用边界与后续关卡

可复用最终同版 correctness_06、parity_01 的 FP32 reference、performance_fp32_01、局部导数新证据及不受执行配置影响的33 CPU接口测试。任何后续影响数值/梯度/分布式聚合的代码变化，都重新做针对性验收；无需重建数据或重复无关证据。

Stage 2 尚需实际四卡分批选择、全状态smoke/resume、参数分组、训练/评估/校准/续训入口及固定校准规则；不把它们记为 Stage 1 已验证。正式状态初始化必须重新使用 seed=20260916，不能继承 smoke 实例。到1000停止与 scheduler 在1000后保持1e-5分别实现，之后延长需新授权。

所有尝试与原失败报告均保留；旧报告见 [V031_IMPLEMENTATION_REPORT_blocked_before_fix.md](<../../../NOT_INCLUDED.md#ref-27440984e595321605df>)。完整执行命令在各 runs/*/launch.json，资源证据在各 guard_result.json/preflight/postflight/telemetry。复跑须使用 fresh output 并经 guard。

```bash
cd /data2/yanglin/v031_overfit100_g4321_n4
/usr/bin/python3 code/guard.py --run-dir runs/correctness_recheck --timeout-seconds 3600 --memory-limit-gib 58 --command bash code/run_env.sh code/acceptance.py --phase correctness --output artifacts/correctness_recheck
# 同点高精度 fixture CPU-only 重放
bash code/run_env.sh code/diagnose_tilt_reference.py --fixture artifacts/tilt_reference_01/tilt_reference_fixture.pt --output artifacts/tilt_reference_cpu_replay_new
```

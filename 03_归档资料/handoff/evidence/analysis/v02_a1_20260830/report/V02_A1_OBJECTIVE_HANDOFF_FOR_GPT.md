# Known-g Physical Turbulence Bridge V0.2 A1 客观证据报告

> 用途：供另一 GPT 或研究人员独立分析本次 V0.2 A1 实验结果。  
> 数据截止：正式 run 2026-08-30；追加可微性审计 2026-09-01（Asia/Shanghai）。  
> 报告性质：冻结实验的事后审计与 inference-only 诊断，不是新的方法设计或正式 replacement evaluation。  
> 可分享性：**可在明确 caveats 的前提下用于机制分析；不可直接作为论文最终实验表格。**

## 1. 本报告回答的问题

本报告只整理以下事实：

1. A1 是否按照冻结协议完成，结果是否满足 A1 gate；
2. 正式 evaluator 是否存在会改变数值口径的实现偏差；
3. 在纠正验证 seed 和 V0.1 comparator 后，STOP 是否仍成立；
4. 已有诊断能够把渐进恢复问题定位到哪一环；
5. 哪些结论已经验证，哪些仍只是待证假设。
6. Simulator 的真实一阶计算图、梯度数值和高严重度条件是否可能解释失败。

本报告不执行以下事项：

- 不修改 V0.2 方法定义；
- 不修改或覆盖 V0.1；
- 不启动新训练、Stage B、full validation 或 test；
- 不把 teacher-forced 或 oracle route 当作可部署方法；
- 不基于当前数据直接选择下一版方法。

## 2. 实验与协议状态

### 2.1 正式运行

| 项目 | 值 |
|---|---|
| 服务器 | `liteserver-40b3` |
| 正式 run | `/data2/yanglin/physical_bridge/runs/v02_new_protocol_20260829` |
| A1 更新数 | 3,000 |
| effective batch | 16 |
| 训练样本调用数 | 48,000 |
| 训练 trajectory 数 | 2,000 |
| terminal-G buckets | 1.5、2.25、3.0、3.75、4.5；各 400 条 trajectory |
| fast-val 规模 | 100 条；每个 terminal G 20 条 |
| 最终 checkpoint | `update-003000.pt` |
| checkpoint SHA-256 | `a48a7557a577d0affd745f1e06f32902801b96c2b97d1df53f68d59176ff033a` |
| 正式 marker | `A0_PASS`、`TRAINING_COMPLETE`、`A1_STOP`、`SCIENTIFIC_STOP_A1` |
| Stage B | 未启动 |

### 2.2 A1 冻结判据

A1 必须同时满足：

1. one-step next-state mean MSE 相对 comparator 至少下降 25%；
2. `K=1` no-refine PSNR 至少提升 0.1 dB；
3. `K=0` PSNR 恶化不超过 0.1 dB，即 proposed-minus-comparator 不低于 -0.1 dB；
4. 数值与训练稳定性 gate 通过。

任一科学 gate 不满足即为 A1 STOP，不允许进入 Stage B，也不通过增加 seeds 寻找偶然成功。

### 2.3 本报告中的 route 定义

- `K0`：从输入直接输出 clean prediction，不执行物理 bridge transition。
- `K1 no-refine`：执行一次相邻较弱状态 transition，不做 refinement，再从预测中间状态恢复 clean。
- `teacher-forced K1`：第二次预测的输入改为真实较弱状态，仅用于判断第二次预测本身是否有效。
- `predicted clean + true ξ`：使用预测 clean 与真实 realization 渲染中间状态，仅用于定位 clean 误差。
- `true clean + predicted ξ`：使用真实 clean 与预测 realization 渲染中间状态，仅用于定位 realization 误差。

## 3. 数据完整性与可复算性

已核验事实：

- 正式 evaluation 共 100 行、100 个唯一 trajectory key，无重复；
- 每个 terminal G 恰好 20 行；
- `K1` 只在 G=4.5 的 20 条 fast-val trajectory 上执行；
- evaluation 所有数值均有限；
- training metrics 恰好包含连续的 updates 1–3000；
- 无 non-finite training row；
- 训练 manifest 包含 3,000 个 batch、48,000 个 sample call；
- 2,000 条训练 trajectory 每条平均被使用约 24 次；
- terminal G、相邻 h、轨迹状态与 source hashes 均通过一致性检查；
- 正式报告中已经列出的聚合数值可以从 rank JSONL 精确复算。

重要聚合限制：正式总体 `K0` 均值覆盖 100 条，而正式 `K1` 均值只覆盖 G=4.5 的 20 条。两者不能直接相减来估计渐进收益。所有 `K1-K0` 结论必须在相同 trajectory 集合上配对计算。

## 4. 正式 evaluator 的两处协议执行偏差

### 4.1 验证 seed 不一致

冻结训练计划规定 fast-val seed 为 `2026082897`。

实际 `v02_data.py:200-205` 使用 `2026082801` 生成 A1 val subset。正式 100 条子集与 `2026082801` 精确一致，与 `2026082897` 不一致。

两个 seed 生成的子集：

- 各包含 100 个唯一 trajectory；
- 仅重合 19 条；
- 重合率为 19%。

### 4.2 V0.1 comparator checkpoint state 不一致

- V0.1 冻结推理路径 `evaluate_rollouts.py:269-271` 加载 `checkpoint["ema"]`；
- A1 evaluator `v02_evaluate_pilot.py:70-76` 对 V0.1 加载 `checkpoint["model"]`，对 V0.2 加载 `checkpoint["ema"]`。

V0.1 raw→EMA 参数差异并非完全可忽略：

| 模块 | raw→EMA relative L2 |
|---|---:|
| shared | 1.668% |
| clean | 1.590% |
| realization | 0.864% |

### 4.3 对正式结果身份的影响

上述偏差使正式 evaluator execution 不符合冻结协议，应该标记为 `invalidated execution`。该判断针对 evaluation execution，不等同于声称 3,000-update training artifact 损坏。

## 5. Seed × comparator 四格敏感性分析

所有数值均使用同一冻结 V0.2 3k EMA checkpoint；没有参数更新。

| 验证子集与 comparator | K0 delta (dB) | K0 gate | G=4.5 K1 gain (dB) | K1 gate | G=4.5 MSE reduction | MSE gate | 总体 |
|---|---:|:---:|---:|:---:|---:|:---:|:---:|
| 实际 seed `2026082801` + V0.1 raw | -0.1238 | FAIL | +0.0698 | FAIL | 3.59% | FAIL | STOP |
| 协议 seed `2026082897` + V0.1 raw | -0.1265 | FAIL | +0.1065 | PASS | 3.50% | FAIL | STOP |
| 实际 seed `2026082801` + V0.1 EMA | -0.1424 | FAIL | -0.0202 | FAIL | 0.71% | FAIL | STOP |
| 协议 seed `2026082897` + V0.1 EMA | -0.1482 | FAIL | -0.0019 | FAIL | 1.35% | FAIL | STOP |

客观结果：四种口径全部为 STOP。只有“协议 seed + V0.1 raw”的 K1 单项达到 +0.1 dB，但同一情形的 K0 与 MSE gate 仍失败。

## 6. 协议口径下的纠偏结果

本节使用：

- fast-val seed `2026082897`；
- V0.1 EMA 作为 frozen inference comparator；
- V0.2 3k EMA；
- 每个 terminal G 20 条 trajectory。

### 6.1 主要 gate 结果

| 指标 | 估计值 | 95% bootstrap CI | 冻结要求 | 结果 |
|---|---:|---:|---:|:---:|
| Overall K0 delta | -0.1482 dB | [-0.1680, -0.1280] | ≥ -0.1 dB | FAIL |
| G=4.5 K1 gain vs V0.1 EMA | -0.0019 dB | [-0.0649, +0.0543] | ≥ +0.1 dB | FAIL |
| G=4.5 next-state MSE reduction | 1.35% | [0.22%, 2.47%] | ≥ 25% | FAIL |

### 6.2 分 terminal-G 结果

| G | K0 delta vs V0.1 EMA (dB) | K1 delta vs V0.1 EMA (dB) | V0.2 K1-K0 (dB) | MSE reduction vs V0.1 EMA |
|---:|---:|---:|---:|---:|
| 1.50 | -0.1269 | -0.0092 | -0.2846 | +1.84% |
| 2.25 | -0.1399 | +0.0223 | -0.3855 | -0.27% |
| 3.00 | -0.1280 | +0.0565 | -0.3134 | +2.03% |
| 3.75 | -0.1941 | +0.0274 | -0.3195 | +2.29% |
| 4.50 | -0.1520 | -0.0019 | -0.3217 | +1.35% |

可直接观察到：

- K0 在五个 terminal-G bucket 中全部低于 V0.1 EMA；
- K1 delta 有正有负，但所有 bucket 都低于 +0.1 dB；
- V0.2 自身的 K1-K0 在五个 bucket 中全部为负；
- MSE 改善幅度远低于 25%，G=2.25 甚至轻微变差。

## 7. 渐进恢复的加法机制分解

定义：

\[
\text{teacher gain}
= \operatorname{PSNR}(K1_{\text{true state}})-\operatorname{PSNR}(K0)
\]

\[
\text{rollout penalty}
= \operatorname{PSNR}(K1_{\text{predicted state}})-\operatorname{PSNR}(K1_{\text{true state}})
\]

因此逐 bucket 严格满足：

\[
\text{teacher gain}+\text{rollout penalty}=K1-K0.
\]

| G | Teacher true-state gain (dB) | Predicted-state rollout penalty (dB) | Net K1-K0 (dB) | 加法残差 |
|---:|---:|---:|---:|---:|
| 1.50 | +2.9498 | -3.2344 | -0.2846 | 0 |
| 2.25 | +1.3472 | -1.7327 | -0.3855 | 0 |
| 3.00 | +0.9781 | -1.2915 | -0.3134 | 0 |
| 3.75 | +0.8469 | -1.1664 | -0.3195 | 0 |
| 4.50 | +0.7056 | -1.0273 | -0.3217 | 0 |

描述性结论：真实较弱状态上的第二次 clean prediction 在所有 G 上均为正收益；使用预测中间状态后产生的负向 penalty 更大，使净 K1-K0 为负。

这能够定位“性能损失出现于 predicted-state rollout”，但单凭该分解不能确定是 clean、ξ、simulator sensitivity、状态分布偏移或多者交互中的哪一个最终因果机制。

## 8. Controlled-render 定位结果

下表均为相对 V0.2 K0 的 PSNR change。

| G | `true clean + predicted ξ` (dB) | `predicted clean + true ξ` (dB) |
|---:|---:|---:|
| 1.50 | +5.2996 | -1.4840 |
| 2.25 | +2.8562 | -1.6352 |
| 3.00 | +1.1579 | -1.3726 |
| 3.75 | +1.0155 | -1.3730 |
| 4.50 | +0.7277 | -1.1551 |

观察：

- 使用真实 clean、保留 predicted ξ 时，五个 G 均为正收益；
- 使用 predicted clean、替换为 true ξ 时，五个 G 均为负收益。

允许的解释强度：该控制证据支持“predicted clean 作为下一步 renderer 底图是当前主要 failure locus”。

不允许的过度解释：它不能单独证明 clean head 是唯一根因，也不能证明 ξ head 已经充分准确；clean 与 ξ 的误差可能存在补偿或耦合。

## 9. G=4.5 checkpoint 演化

所有结果使用协议 seed 的同一 20 条 G=4.5 trajectory。

| Checkpoint | K0 PSNR | K1 PSNR | K1-K0 | Next-state MSE | Teacher-forced K1 PSNR |
|---|---:|---:|---:|---:|---:|
| V0.1 EMA / update 0 | 20.7239 | 20.2521 | -0.4718 | 0.00051682 | 21.4255 |
| V0.2 1k EMA | 20.6634 | 20.2009 | -0.4625 | 0.00050843 | 21.3724 |
| V0.2 2k EMA | 20.6070 | 20.1924 | -0.4147 | 0.00050786 | 21.3149 |
| V0.2 3k EMA | 20.5719 | 20.2502 | -0.3217 | 0.00050985 | 21.2775 |

观察：

- K1-K0 gap 随训练收窄；
- K0 同时持续下降；
- K1 到 3k 时基本回到 V0.1 EMA K1 水平，但没有超过；
- next-state MSE 在 2k 最低，3k 略有回升；
- teacher-forced K1 同期下降。

因此，“K1-K0 gap 收窄”不能单独解释为渐进恢复取得正向成功。

## 10. 训练健康与参数变化

### 10.1 数值与优化状态

| 项目 | 结果 |
|---|---:|
| 连续 updates | 3,000 / 3,000 |
| non-finite rows | 0 |
| gradient-clipped steps | 0 |
| median step time | 18.527 s |
| peak NPU memory | 12.412 GB |

首 200 updates 与末 200 updates 的 loss median ratio：

| Loss | Last / first median |
|---|---:|
| Clean | 0.8754 |
| Trajectory estimator | 0.8947 |
| Bridge | 0.9128 |
| Full trajectory diagnostic | 0.9037 |
| Total | 0.8783 |

描述性结论：优化器确实更新了模型，所有损失总体下降；没有证据支持“训练未运行”“NaN 导致失败”或“低显存导致失败”。

### 10.2 V0.2 3k EMA 相对 V0.1 raw 的参数漂移

| 模块 | Relative L2 drift |
|---|---:|
| shared | 2.2247% |
| clean | 2.2397% |
| realization | 1.6081% |

该结果说明 clean/shared 模块发生了可测变化，但参数漂移大小本身不等价于性能因果解释。

## 11. A1 边际 severity 暴露

V0.2 A1 先均衡 terminal G，再在当前轨迹允许的 `g≤G` 中采样。因此 terminal-G 平衡不等于边际 current-g 平衡。

| Current g | A1 count | A1 share | V0.1 uniform-g expected share | A1 / V0.1 expected exposure |
|---:|---:|---:|---:|---:|
| 0.75 | 13,999 | 29.16% | 16.67% | 1.750 |
| 1.50 | 13,924 | 29.01% | 16.67% | 1.741 |
| 2.25 | 9,045 | 18.84% | 16.67% | 1.131 |
| 3.00 | 5,986 | 12.47% | 16.67% | 0.748 |
| 3.75 | 3,419 | 7.12% | 16.67% | 0.427 |
| 4.50 | 1,627 | 3.39% | 16.67% | 0.203 |

已验证事实：V0.2 A1 中高 severity 的边际暴露显著少于 V0.1 sampler 的均匀-g 期望。

未验证因果：当前没有 otherwise-identical sampler control，因此不能断言该分布变化导致 K0 下降。它只能列为候选解释。

## 12. 梯度方向诊断

诊断设置：在四个冻结 A1 batch 上分别对 V0.1 raw 与 V0.2 3k EMA 计算 no-update gradient cosine；样本量为 4，不是总体统计。

| Checkpoint | Shared: clean vs weighted physical | Shared: clean vs trajectory | Shared: trajectory vs bridge |
|---|---:|---:|---:|
| V0.1 raw | 0.4517 | 0.2391 | 0.1228 |
| V0.2 3k EMA | 0.1331 | 0.0808 | 0.0654 |

观察：中位 cosine 仍为正，但训练末期明显更接近正交。

允许的解释：不同目标在 shared trunk 上的局部方向一致性变弱。

不允许的过度解释：四个 batch 不足以证明存在系统性 gradient conflict，也不足以证明该变化导致最终 PSNR 下降。

## 13. Simulator 可微性独立审计

本节不采用方法文档中的“可微”声明，而是核对实际执行源码、真实 NPU backward 记录与冻结样本中心有限差分。

### 13.1 真实计算图与 NPU 分支指纹

- `simulator.render` 未使用 `torch.no_grad()`；输入 clean、Zernike ξ、tilt ξ 经过 FFT/IFFT、卷积、P2S 线性层/ReLU 与 bilinear `grid_sample`；
- `self.requires_grad_(False)` 只冻结 simulator/P2S 参数，不会 detach 输入；
- 网络 BF16 forward 后的 `.float()` cast 保留 autograd 图，simulator 与 loss 在 FP32 中执行；
- A0 的 128 个真实 NPU batch 中：clean loss 对 realization branch 始终为 0，trajectory loss 对 clean branch 始终为 0，bridge loss 对 shared/clean/realization 始终非零；
- A1 的 3,000 次更新中，shared、clean、realization 三条分支梯度范数全部有限且非零。

该分支指纹与损失定义精确一致。如果 simulator 输出被 detach，trajectory/bridge loss 不可能在 realization/shared 参数上留下这种结构。

### 13.2 中心有限差分

设置：冻结样本 `val_000260`、V0.2 3k EMA、G=4.5 输入、h=3.75 bridge loss；CPU float32 执行真实冻结 simulator，不更新参数。

| 输入方向 | Autograd directional derivative | Central difference | 最佳相对步长 | 相对误差 | 符号 |
|---|---:|---:|---:|---:|:---:|
| predicted clean | 3.6800815e-5 | 3.6801012e-5 | 1e-2 | 0.000535% | PASS |
| Zernike ξ | 2.0467935e-5 | 2.0517055e-5 | 3e-4 | 0.239415% | PASS |
| tilt ξ | 2.1147263e-5 | 2.1145093e-5 | 3e-3 | 0.010264% | PASS |

三条方向符号全部一致，最大最佳步长相对误差为 0.239%。这排除了“有 grad 字段但 autograd 数值与函数变化不一致”的 CPU 实现故障。

### 13.3 G=4.5 条件诊断（10 条冻结 trajectory）

| 量 | Min | Median | Max |
|---|---:|---:|---:|
| predicted/true effective Zernike RMS | 3.769 | 4.647 | 5.548 |
| predicted/true displacement RMS | 0.549 | 0.868 | 1.051 |
| P2S 第三次 ReLU 激活率 | 14.12% | 14.40% | 14.66% |
| Zernike gradient L2 | 2.85e-5 | 9.95e-5 | 1.60e-4 |
| tilt gradient L2 | 8.51e-5 | 5.28e-4 | 1.52e-3 |
| true clean + true ξ replay MSE | 2.19e-9 | 8.25e-9 | 1.76e-8 |

观察：forward replay 与存档状态近乎数值一致，且所有物理梯度有限非零；不存在粗大 forward mismatch 或高严重度整体梯度坍缩。真正异常是 predicted effective Zernike 场系统性高于真实尺度，同时 P2S 后段 ReLU 激活较稀疏。这支持“Zernike/P2S 梯度条件不良或语义弱”作为**次要机制线索**，不支持“simulator 不可微”作为主因。

### 13.4 尚未闭合的 NPU 精度边界

A0/A1 每个 NPU rank 首次 backward 均出现一次输出形状 resize warning（`[16,17,32]` → `[16,32,32]`）。该形状类似 N=32 实数 FFT 半谱，但日志没有定位具体算子。现有梯度均有限、分支结构正确，故该 warning 不是错误梯度的证明；但它是兼容性风险。2026-09-01 审计时，授权物理 NPU 0/2/4/6 均被无关任务占用，因此没有抢卡做同样的 NPU 中心有限差分。

另一个上游实现细节是 `(N-1)` 坐标归一化与 `align_corners=False` 同时使用；零 tilt 相对 blur-only 产生 4.34e-6 MSE，而匹配的 identity grid 为 0。该约定同时存在于数据生成与当前 renderer，不能解释 V0.2 相对 V0.1 的差异，但应列为工程技术债。

客观判断：**Simulator 一阶不可微不是当前失败原因；P2S Zernike 路径的条件问题可能有次要贡献。现有更强控制证据仍把主要 failure locus 指向 predicted clean 中间底图。**

## 14. 证据等级

### 14.1 已验证事实

1. 正式 A1 evaluator 使用了错误 val seed 和非冻结推理口径的 V0.1 raw comparator。
2. 正式 evaluation execution 应标记为 invalidated。
3. 使用协议 seed 与 V0.1 EMA 纠偏后，A1 三个主要科学 gate 全部失败。
4. Seed × comparator 四种组合的总体判据均为 STOP。
5. 第二次 clean prediction 在真实较弱状态上有正收益。
6. predicted-state rollout penalty 大于上述正收益，使 K1-K0 为负。
7. 训练过程数值健康，模型参数发生了实际变化。
8. Simulator 的一阶图真实连通，CPU 中心有限差分与 autograd 一致；不可微或 detach 不是原因。

### 14.2 有较强控制证据、但仍不是完整因果证明

1. predicted clean 作为 renderer 底图是主要 failure locus；
2. predicted ξ 单独不是 controlled-render 中的主要负向来源；
3. pixel next-state MSE 与最终 clean PSNR 之间存在目标不充分对齐的迹象。

### 14.3 尚未验证的候选解释

1. high-g 边际暴露不足导致 K0 侵蚀；
2. shared trunk 的多目标弱对齐导致 clean/physics 折衷；
3. bridge renderer 对 clean prediction 的特定频率或结构误差具有放大作用；
4. 2,000 条 trajectory、每条平均复用约 24 次造成过拟合或分布记忆；
5. pixel-space next-state MSE 不是适合最终 restoration quality 的充分 surrogate。
6. P2S Zernike 有效态尺度偏大与低后段激活导致物理梯度条件不良。

## 15. 当前可以与不可以做出的决定

### 可以做出的决定

- 维持 A1 scientific STOP；
- 不进入 Stage B；
- 不通过额外 seeds 复验寻找偶然成功；
- 若需要正式归档数字，应使用同一冻结 3k checkpoint 修正 evaluator，并生成明确标记的 `invalidated-run replacement` report；该操作不需要重训。
- 授权 NPU 空闲后，可先做同一冻结样本的 NPU 中心有限差分与 FFT/grid-sample 分算子 VJP；这只核验实现，不改变方法。

### 当前证据不足以决定

- 下一版应该修改 sampler、loss、network branch、shared trunk 还是 state metric；
- 是否应冻结 clean branch 或完全分离 clean/realization backbone；
- 是否应该用 downstream-aware metric 替代或补充 pixel MSE；
- 哪一项候选机制是主要根因。

## 16. 限制与外推边界

1. 纠偏诊断是 inference-only replay，不替换服务器上的正式 marker。
2. 纠偏 headline 使用 fast-val 100 条 trajectory；每个 G 只有 20 条。
3. 未运行 full val 400、test 或多 seed 确认。
4. Bootstrap CI 描述固定 fast-val 抽样下的不确定性，不代表跨数据生成过程的全部不确定性。
5. Teacher-forced、true-clean 和 true-ξ routes 使用 oracle 信息，只能诊断，不能作为公平可部署 comparator。
6. Gradient cosine 只有四个 batch，统计强度有限。
7. 本报告没有 otherwise-identical causal ablation，因此候选根因不能写成确定因果结论。
8. 可微性条件 cohort 只有 10 条 G=4.5 trajectory；CPU 已闭合梯度精度，真实 NPU 仅闭合梯度存在性，尚未直接闭合 NPU 有限差分精度。

## 17. 建议另一 GPT 独立回答的问题

请基于以上证据独立分析，不假设本报告中的候选解释一定正确：

1. 现有证据能把 failure locus 缩小到哪一级：predicted state、predicted clean、具体 clean error type，还是 shared optimization？
2. 哪些替代解释仍与所有观察结果相容？
3. 要区分 sampler exposure、objective interference、renderer sensitivity 与 surrogate mismatch，最少需要哪些 otherwise-identical causal controls？
4. 哪些动作只是修复 evaluator，哪些动作会改变 V0.2 科学定义并必须重新冻结协议？
5. 如果只允许一个小规模探索阶段，怎样设计才能最大化机制信息而不做结果导向的超参数搜索？
6. 是否存在比“继续训练”更合理、且能被当前 checkpoint 直接检验的反事实诊断？
7. P2S effective-Zernike 尺度异常是可识别性问题、输出参数化问题、代理 loss 问题，还是 predicted-clean 控制结果下的次要补偿现象？

## 18. 可复核材料

本地分析根目录：

`/Users/lucas/Desktop/turbulence IR/analysis/v02_a1_20260830`

主要文件：

- `derived/analysis_summary.json`：正式 artifacts 独立复算；
- `derived/diagnostic_analysis.json`：纠偏结果、四格敏感性、机制分解；
- `derived/gate_sensitivity.csv`：四种 seed/comparator 口径；
- `derived/corrected_per_g.csv`：纠偏分 G 指标；
- `derived/mechanism_decomposition.csv`：teacher gain 与 rollout penalty；
- `derived/checkpoint_progression_g450.csv`：G=4.5 checkpoint 演化；
- `derived/sampling_exposure.csv`：边际 g 暴露；
- `derived/gradient_cosine_medians.csv`：梯度方向诊断；
- `derived/differentiability_summary.json`：可微性结论、证据边界与 NPU 未闭合项；
- `derived/differentiability_finite_difference.csv`：三条输入方向中心有限差分；
- `derived/differentiability_cohort_g450.csv`：G=4.5 十样本条件诊断；
- `data/differentiability/`：CPU 原始 JSON、上游源码副本和 A0/A1 NPU 日志；
- `report/qa_summary.json`：已有报告与 notebook 的独立 QA；
- `/Users/lucas/Desktop/turbulence IR/output/jupyter-notebook/v02-a1-failure-analysis.ipynb`：可顺序执行的复算 notebook。

## 19. 最终客观状态

- 正式 training artifact：完整、数值健康；
- 正式 evaluator execution：因 seed 与 comparator 口径偏差而 invalidated；
- 纠偏 scientific decision：STOP；
- Stage B：不应启动；
- 当前主要 failure locus：predicted-state rollout，controlled-render 证据进一步指向 predicted clean 底图；
- Simulator 一阶不可微：已排除；P2S Zernike 条件异常：次要、待因果证伪；
- 完整根因：尚未由因果实验确定。

# Known-\(g\) Physical Turbulence Bridge V0.2
## Codex 训练、验证与冻结执行协议

**状态：协议闭合冻结版。**

**目标：以最少的完整训练次数判断 V0.2 是否修复 V0.1 的跨-severity transport failure，同时保持 one-shot 能力，并验证受 trajectory 保护的测试时 \(\xi\) 优化是否带来独立增益。**

本计划只实现方法文档中已冻结的 V0.2，不重新设计网络、latent、数据集、loss 或 inference mechanism。当前新 V0.2 只以以下方法文档为算法权威：

- “KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_METHOD.md”；
- schedule authority：SA-V02-ADJ-1。

V0.1 仅作为数据、源码、checkpoint、结果和固定 protocol 的基线资产。任何此前停止的旧 V0.2 方法或训练协议均无权威性。

---

## 0. 权威顺序与执行门

发生不一致时，按以下顺序处理：

1. 方法文档中的方程、两个核心机制和方法边界；
2. 本文冻结的采样、公平比较、selection、统计与 test-isolation 协议；
3. V0.1 实际运行源码和配置中被本文明确要求“继承”的工程语义；
4. 最简单、可复现、不给 proposed method 增加额外自由度的工程实现。

若两个合理解释会改变方法方程、主要比较、test exposure 或科学结论，必须 STOP 并报告；不得静默选择有利于 V0.2 的解释。

在用户指定服务器前，只允许修订和审阅本文档，不连接服务器、不建立 run、不处理数据、不训练。

---

## 1. 执行前只读绑定 V0.1 身份

进入服务器后，修改代码前必须从实际运行源码、checkpoint 与 manifest 核验：

1. V0.1 真正的 forward 是否为：

\[
S_\theta(X_g,g)\rightarrow(\hat C,\hat\xi),
\]

而不是 image-only \(S_\theta(X_g)\)；
2. \(g\) 是否实际进入 network computation，而不只用于 dataloader、simulator 或 schedule；
3. V0.1 20k checkpoint、run config、model architecture、simulator hash 和 dataset manifest；
4. simulator-native full-resolution \(\xi\) 的实际 shape、layout、dtype 和 renderer call site；
5. simulator 对预测 \(\xi\) 的 forward/backward 路径；
6. V0.1 实际使用的 image range、clamp、padding、valid mask/crop、PSNR、SSIM、LPIPS backbone/version 与 aggregation。

已知冻结身份：

- Run ID：bridge_v01_seed20260826；
- V0.1 checkpoint SHA-256：34dc4dce03bd13ba4079100d4b195fcc0e8c3b07699e06ae12ca91e0aba48f47；
- V0.1 训练更新：20,000；
- 数据 trajectory：9,400；
- split：8,000 / 400 / 1,000。

输出 “v02_identity_audit.md” 和 machine-readable JSON，包含 source call sites、resolved paths、config 与 hashes。若实际源码或资产与上述身份、方法接口或 frozen report 不一致，立即输出 “implementation-report mismatch” 并 STOP；不得凭报告文字改写代码行为。

---

## 2. 冻结数据、trajectory 与模型边界

### 2.1 数据身份

每条 trajectory 固定：

\[
\boxed{(C_i,\xi_i,G_i)}
\]

其中：

\[
G_i\in\{1.50,2.25,3.00,3.75,4.50\}
\]

是 manifest 中的固定 terminal-severity metadata。不得在训练过程中为同一 trajectory 动态重分配 \(G_i\)。

冻结 split：

| Split | 总 trajectory | 每个 terminal-\(G\) bucket |
|---|---:|---:|
| Train | 8,000 | 1,600 |
| Val | 400 | 80 |
| Test | 1,000 | 200 |

进入服务器后必须核验实际 manifest 的 ID 唯一性、split 隔离、固定 \(G_i\)、bucket 计数、六档同轨迹状态及 oracle-\(\xi\) 所需真实 \(\xi_i\) 或确定性 replay 信息。若计数或语义不符，报告 mismatch 并 STOP；不得重新分配 ID、改变 split 或用重新生成数据掩盖问题。

severity grid：

\[
\boxed{\mathcal G=\{0.75,1.50,2.25,3.00,3.75,4.50\}}.
\]

不重新生成约 370.61 GiB 数据，不改变 augmentation，不把同一 source 的不同 augmentation 当作独立 trajectory 对照。

### 2.2 模型

- B0、B1 均从同一 V0.1 20k raw model weights 初始化；
- optimizer 重新初始化；
- 网络结构、显式 \(g\) conditioning、\(\xi\) output semantics/shape 和 simulator 保持不变；
- PiRN/P2S 固定参数保持冻结，但保留 renderer 到预测 \(\xi\) 的梯度；
- 不新增 global consensus、cycle、re-ID、rollout network、额外 head 或 loss。

---

## 3. 唯一 schedule authority：SA-V02-ADJ-1

\(K\in\{0,1,3,5\}\) 是“请求执行的 physical transition 数”。每次 transition 只移动到离散 grid 上的相邻较弱 severity。

\[
\boxed{
K_{\mathrm{eff}}(G,K)=
\min\{K,K_{\max}(G)\}
}
\]

其中 \(K_{\max}(G)\) 是从 \(G\) 沿相邻 grid 到 0.75 的最大 transition 数。达到 0.75 后停止，不重复状态、不补虚假 transition。\(K=0\) 始终为 one-shot。

所有正式 validation、comparator 和 test 对 trajectory \(i\) 都从其固定 terminal state \((X_{i,G_i},G_i)\) 开始。训练时随机采样的当前 \(g\) 不改变评价起点。

| 起始 \(G\) | 请求 \(K=0\) | 请求 \(K=1\) | 请求 \(K=3\) | 请求 \(K=5\) |
|---:|---|---|---|---|
| 0.75（边界输入，非 terminal bucket） | 0.75（\(K_{\mathrm{eff}}=0\)） | 0.75（0） | 0.75（0） | 0.75（0） |
| 1.50 | 1.50（0） | 1.50→0.75（1） | 1.50→0.75（1） | 1.50→0.75（1） |
| 2.25 | 2.25（0） | 2.25→1.50（1） | 2.25→1.50→0.75（2） | 2.25→1.50→0.75（2） |
| 3.00 | 3.00（0） | 3.00→2.25（1） | 3.00→2.25→1.50→0.75（3） | 3.00→2.25→1.50→0.75（3） |
| 3.75 | 3.75（0） | 3.75→3.00（1） | 3.75→3.00→2.25→1.50（3） | 3.75→3.00→2.25→1.50→0.75（4） |
| 4.50 | 4.50（0） | 4.50→3.75（1） | 4.50→3.75→3.00→2.25（3） | 4.50→3.75→3.00→2.25→1.50→0.75（5） |

Training diagnostic、validation、所有 comparator、beta/K selection 与 frozen test 均引用本表，不得另建 schedule。实际最终 clean output 是 \(g^{(K_{\mathrm{eff}})}\) 上最后一次网络 forward 的 clean branch；终点可以高于 0.75。

---

## 4. 冻结训练目标与 batch reduction

\[
\boxed{
\mathcal L=
\mathcal L_C+
\lambda_T\widehat{\mathcal L}_T+
\lambda_B\mathcal L_B
}
\]

### 4.1 图像距离

\[
\boxed{
d_I(A,B)=\operatorname{mean}[(A-B)^2]
}
\]

image range、boundary convention、mask、precision、RGB/channel、clamp 与 padding 原样继承经 §1 核验的 V0.1 实际代码。若 V0.1 没有 mask，不得新增。

### 4.2 Clean loss

\[
\mathcal L_C=d_I(\hat C_{i,g},C_i).
\]

batch reduction 对全部样本取 mean。

### 4.3 Trajectory realization loss

每个样本定义：

\[
\mathcal S_i=
\begin{cases}
\operatorname{unique}\{G_i,g,h\}, & g>0.75,\\
\operatorname{unique}\{G_i,g\}, & g=0.75.
\end{cases}
\]

完整目标：

\[
\mathcal L_T^{\mathrm{full}}
=
\frac{1}{|\mathcal S_i|}
\sum_{s\in\mathcal S_i}
d_I(F(C_i,\hat\xi_{i,g},s),X_{i,s}).
\]

正式训练使用无偏单-severity estimator：

\[
s_T\sim U(\mathcal S_i),
\]

\[
\boxed{
\widehat{\mathcal L}_T=
d_I(F(C_i,\hat\xi_{i,g},s_T),X_{i,s_T})
}
\]

\[
\boxed{
\mathbb E_{s_T\sim U(\mathcal S_i)}
[\widehat{\mathcal L}_T]
=\mathcal L_T^{\mathrm{full}}.
}
\]

batch reduction 对全部样本取 mean。Validation、checkpoint selection 与正式报告必须计算每个样本的完整 \(\mathcal L_T^{\mathrm{full}}\)，不得用随机 estimator 代替。

### 4.4 Bridge execution loss

合法 transition 子集：

\[
\mathcal B_B=\{i:g_i>0.75\}.
\]

对 \(i\in\mathcal B_B\)：

\[
\mathcal L_{B,i}
=
d_I(F(\hat C_{i,g},\hat\xi_{i,g},h),X_{i,h}).
\]

\[
\boxed{
\mathcal L_B=
\frac{1}{|\mathcal B_B|}
\sum_{i\in\mathcal B_B}\mathcal L_{B,i}
}
\]

若 \(|\mathcal B_B|=0\)，该 optimizer step 跳过 \(\lambda_B\mathcal L_B\) 项。不得对无合法 \(h\) 的样本 zero-fill 后除以整个 batch。

---

## 5. Stage A0：固定 128-batch gradient audit

### 5.1 候选与测量

从 V0.1 20k checkpoint 出发，不更新参数。在固定 128 个 effective training batch 上，使用完全相同的数据 ID、\(g/h\) 和 \(s_T\) draws，分别只反向：

\[
\mathcal L_C,\qquad
\widehat{\mathcal L}_T,\qquad
\mathcal L_B.
\]

A0 manifest 在查看任何 gradient 前按 §6 的固定 seed 生成，并确定性保证每个 effective batch 至少包含一个合法 \(g>0.75\) transition，使 \(\mathcal L_B\) norm 对全部 128 batches 均有定义。

所有 AMP 梯度必须 unscale 后再测量。对每个 batch 分别记录 shared encoder、clean head、realization head 的未加权 L2 gradient norm，再对 128 个 batch 取中位数：

\[
G_C,\quad G_T,\quad G_B.
\]

同时记录 clean head：

\[
H_C=\operatorname{median}\|\nabla_{\theta_C}\mathcal L_C\|,
\qquad
H_B=\operatorname{median}\|\nabla_{\theta_C}\mathcal L_B\|.
\]

候选集合：

\[
\boxed{
\lambda_T\in\{0.2,0.4,0.8\},\qquad
\lambda_B\in\{0.05,0.1,0.2\}.
}
\]

这些组合只基于同一份未加权梯度进行代数评估，不运行训练 grid。

### 5.2 健康条件

\[
r_T=\frac{\lambda_TG_T}{G_C},\qquad
r_B=\frac{\lambda_BG_B}{G_C},\qquad
r_{\mathrm{phys}}=r_T+r_B,\qquad
r_{B,C}=\frac{\lambda_BH_B}{H_C}.
\]

组合必须同时满足：

\[
\boxed{0.20\le r_T\le0.40}
\]

\[
\boxed{0.05\le r_B\le0.20}
\]

\[
\boxed{0.25\le r_{\mathrm{phys}}\le0.50}
\]

\[
\boxed{r_{B,C}\le0.25}.
\]

若任一分母为零、norm 非有限或 simulator backward 失败，则 A0 STOP。

### 5.3 唯一选择规则

若多个组合健康，按以下固定顺序选择：

1. 最小化相对原建议值 \((0.4,0.1)\) 的离散距离

\[
D=
\left|\log_2\frac{\lambda_T}{0.4}\right|
+
\left|\log_2\frac{\lambda_B}{0.1}\right|;
\]

2. \(D\) 相同时选择更小的 \(r_{\mathrm{phys}}\)；
3. 再相同时选择更小的 \(\lambda_B\)；
4. 再相同时选择更小的 \(\lambda_T\)。

若九个组合均不健康，输出 “v02_gradient_audit.md/JSON”，A0 STOP；不得扩大搜索或进入 pilot。选中后冻结 \(\lambda_T,\lambda_B\)，后续 seed 不再调整。

---

## 6. 训练采样与 RNG 冻结

### 6.1 每个 batch

1. 从五个固定 terminal-\(G_i\) bucket 平衡抽取 trajectory，再读取其固定 \(G_i\)。effective batch 为 16 时，同一 batch 内五桶计数差不超过 1，并轮换额外样本，使每连续五个 effective batches 的五桶总计数完全相同；
2. 从 \(\mathcal G\cap(-\infty,G_i]\) 中均匀采样当前 \(g\)；
3. 若 \(g>0.75\)，\(h\) 固定为离散 grid 上与 \(g\) 相邻的较弱 severity；不训练跳级 bridge transition；

4. 若 \(g=0.75\)，不定义 \(h\)，只计算 \(\mathcal L_C+\lambda_T\widehat{\mathcal L}_T\)；
5. 从当前样本的 \(\mathcal S_i\) 中均匀采样 \(s_T\)。

### 6.2 RNG namespace

各随机过程使用独立 generator，不得因 B1 多消费 \(g/h\) 或 \(s_T\) RNG 而改变 B0/B1 的 data order。

| 用途 | Seed 1 | Seed 2 |
|---|---:|---:|
| data/order RNG | 2026082801 | 2026082901 |
| \(g/h\) sampling RNG | 2026082802 | 2026082902 |
| \(\mathcal L_T\) Monte-Carlo RNG | 2026082803 | 2026082903 |
| refinement RNG（如实现为确定性则记录 unused） | 2026082804 | 2026082904 |
| model/optimizer stochastic RNG | 2026082805 | 2026082905 |

其他固定 seed：

| 用途 | Seed |
|---|---:|
| A0 128-batch manifest | 2026082800 |
| fast-val subset selection | 2026082897 |
| qualitative ID selection | 2026082898 |
| paired bootstrap/statistics | 2026082899 |

所有 generator algorithm、device semantics、initial state 与 seed 写入 manifest。B1 manifest 逐样本冻结 \((\text{trajectory ID},G_i,g,h,s_T)\)；B0 使用完全相同的 \((\text{trajectory ID},G_i,g)\) 与顺序，并忽略其旧目标不需要的 \(h,s_T\)。B0/B1 在同一 training seed 下共享 data/order RNG 和所有适用的 model/optimizer stochastic seed。

---

## 7. Stage A1：单次短 pilot

### 7.1 数据

- 2,000 train trajectories：每个固定 terminal-\(G\) bucket 400；
- 固定 100 条 val trajectories：每 bucket 20；
- ID、\(G_i\)、sample order 与全部 RNG draws 在运行前写入 manifest。

### 7.2 训练设置

| 项目 | 设置 |
|---|---:|
| 初始化 | V0.1 20k raw checkpoint |
| Loss weights | Stage A0 冻结值 |
| Updates | 3,000 |
| Effective batch | 16 |
| Optimizer | AdamW |
| Learning rate | \(2\times10^{-5}\) |
| Betas | 继承 V0.1 |
| Weight decay | 继承 V0.1 |
| Warmup | 200 updates |
| LR schedule | cosine |
| Gradient clipping | 1.0 |
| EMA | 0.999 |
| Network precision | 继承 V0.1（如 BF16） |
| Simulator/loss | FP32，继承 V0.1 boundary semantics |

Pilot 不选择 \(\beta\)、不运行 refinement、不跑 \(K=3/5\)、不改变 LR 或网络结构。

### 7.3 评价与继续门

固定使用 3,000-update EMA checkpoint，比较 V0.1 original checkpoint 与 V0.2 pilot 的 no-refine diagnostics。重点使用 \(G=4.5\)、SA-V02-ADJ-1 的 \(K=1:4.5\rightarrow3.75\)。

必须同时满足：

1. V0.2 one-step next-state MSE 相比 V0.1 original/no-refine diagnostic 至少下降 25%；
2. V0.2 \(K=1\) no-refine PSNR 至少提高 0.1 dB；
3. V0.2 \(K=0\) PSNR 相比 V0.1 original 下降不超过 0.1 dB；
4. 完整 \(\mathcal L_T^{\mathrm{full}}\) 与 \(\mathcal L_B\) 的最后 200 updates median 均不高于最初 200 updates median 的 95%；
5. 无 NaN/Inf 或 simulator backward failure；在固定 pilot val 上，Zernike 与 tilt 各自的 median per-sample \(\|\hat\xi\|_2\) 和 component std 均保持在 V0.1 original 对应统计的 \([0.1,10]\) 倍内，超出即判定明显 \(\xi\) collapse/explosion。

A0 已完成权重冻结，因此 A1 只允许一次科学 pilot。若未通过，输出 “v02_pilot_report.md/JSON” 并 STOP full fine-tuning，不得利用 pilot 结果重新选择权重。只有经证据确认的纯实现错误可以在相同配置、相同 manifest 下修复后重跑；该重跑必须标记 invalidated-run replacement，不计作新的科学配置。

---

## 8. Stage B：update/data-matched fine-tuning

Pilot 通过后运行一对 matched runs。

### 8.1 B0 continued control

从 V0.1 20k raw checkpoint 出发，使用 V0.1 原训练目标与经 §1 核验的冻结设置继续训练。

### 8.2 B1 V0.2

从同一 checkpoint 出发，使用：

\[
\mathcal L_C+
\lambda_T\widehat{\mathcal L}_T+
\lambda_B\mathcal L_B.
\]

### 8.3 公平性

B0 与 B1 必须匹配：

- 初始化权重；
- train/val manifest；
- data sample order；
- effective batch；
- optimizer family、betas、weight decay 与 LR schedule；
- extra-update 数；
- training seed；
- hardware allocation 与 network precision；
- checkpoint/validation index。

两者是 **update/data-matched**，不是 compute-matched。允许 simulator calls、wall time、peak memory 与 NPU-hours 不同，但必须使用相同的计数口径完整记录。不得向 B0 添加无意义 simulator calls。

### 8.4 完整训练设置

| 项目 | 设置 |
|---|---:|
| Train trajectories | 8,000 |
| Val trajectories | 400 |
| Initial extra updates | 10,000 |
| Maximum extra updates | 15,000 |
| Effective batch | 16 |
| Optimizer | AdamW |
| Learning rate | \(2\times10^{-5}\) |
| Warmup | 500 updates |
| LR schedule | cosine |
| Gradient clipping | 1.0 |
| EMA | 0.999 |
| Checkpoint interval | 1,000 updates |
| Fast validation | 每 500 updates |
| Full validation | 每 2,000 updates，并强制在 10k/15k 运行 |

Fast validation 使用预先固定的 100 条 val 子集（每 terminal bucket 20，seed 2026082897），计算 K=0、K=1 no-refine、完整 \(\mathcal L_T\) 与 one-step next-state MSE；25-ID montage 由独立 seed 2026082898 固定。

Full validation 在全部 400 条 val 上计算 K=0、K=1 no-refine、PSNR/SSIM/LPIPS/L1、完整 \(\mathcal L_T\)、one-step next-state MSE、per-\(G\) 与 paired delta。

### 8.5 唯一延长规则

在 B1 到达 10,000 updates 后，比较其 8,000 与 10,000 update EMA checkpoint 的 full-val \(K=1\) no-refine PSNR：

\[
\Delta_{\mathrm{trend}}=
\operatorname{PSNR}_{10k}-\operatorname{PSNR}_{8k}.
\]

- 若 \(\Delta_{\mathrm{trend}}>0.05\) dB，B0 与 B1 同步延长到 15,000 updates；
- 否则两者均在 10,000 updates 停止；
- 延长后在 12k、14k、15k 做 full validation；
- 不得只延长 B1，亦不得因结果不利重启不同超参数。

---

## 9. Checkpoint selection

### 9.1 B1 eligibility

V0.1 reference 明确定义为原始冻结 20k checkpoint 的 \(K=0\) one-shot：

\[
\boxed{
\operatorname{PSNR}_{K=0}^{B1}
\ge
\operatorname{PSNR}_{K=0}^{\mathrm{V0.1\ original}}
-0.1\ \mathrm{dB}.
}
\]

只有经过 full validation 的 EMA checkpoints 可参与。若没有 eligible checkpoint，STOP。

### 9.2 B1 唯一选择规则

在 eligible checkpoints 中：

1. 最大化 \(K=1\) no-refine overall PSNR；
2. 与最大值相差不超过 0.05 dB 的候选中，选择 one-step next-state MSE 最低者；
3. 若 next-state MSE 的相对差不超过 \(10^{-6}\)，选择更早的 extra-update index。

冻结 selected EMA checkpoint，并记录 raw/EMA、config、source 与 manifest hashes。

### 9.3 B0 对齐规则

B0 不独立使用 validation 选择 checkpoint。B0 comparator 强制取与 B1 selected checkpoint 相同的 extra-update index。若 B1 延长，B0 同步延长并产生对应 checkpoint。

若触发 seed2，seed2 的 B1 使用同一 eligibility/selection rule，但训练总长度直接复制 seed1 的 10k 或 15k 决定；不再用 seed2 重做延长决策。seed2 的 B0 仍强制对齐 seed2-B1 selected extra-update index。

---

## 10. Stage C：冻结 \(\beta>0\)

训练 checkpoint 冻结后才选择：

\[
\mathcal J_k=
\mathcal J_{\mathrm{obs},k}
+
\beta\mathcal J_{\mathrm{keep},k}.
\]

### 10.1 固定 refinement 配置

optimizer 类型、Adam LR、betas、iteration budget、early-stop、gradient clipping、mask/boundary 和初始化 \(\xi=\hat\xi_k\) 原样继承经 §1 核验的 V0.1 refinement 配置。第一轮不得与 \(\beta\) 联合调 optimizer。

### 10.2 候选与选择集

\[
\beta\in\{0,0.1,0.3,1,3\},
\]

另设 no-refine。

- \(\beta=0\) 只定义为 V0.2-old-refine comparator；
- V0.2-full 的 selection set 仅为 \(\{0.1,0.3,1,3\}\)；
- no-refine 与 \(\beta=0\) 均不能被命名为 V0.2-full。

使用 seed1-B1 selected checkpoint、全部 80 条 \(G=4.5\) val trajectories 和 SA-V02-ADJ-1 的 \(K=1:4.5\rightarrow3.75\)。

### 10.3 Gate

对每个 \(\beta>0\) 保存 refinement 前后 observation residual、next-state MSE、PSNR/SSIM/LPIPS、\(\|\xi^*-\hat\xi\|\)、Zernike/tilt update、iterations、simulator calls、runtime 与 peak memory。

候选必须同时满足：

\[
\boxed{
\frac{\operatorname{mean}J_{\mathrm{obs,after}}}
{\operatorname{mean}J_{\mathrm{obs,before}}}
\le0.80
}
\]

\[
\boxed{
\operatorname{meanMSE}_{\mathrm{next,refined}}
\le
1.05\,
\operatorname{meanMSE}_{\mathrm{next,no-refine}}.
}
\]

均值以 trajectory 为单位在同一 80 条样本上计算。

### 10.4 唯一选择规则

在通过 gate 的 \(\beta>0\) 中：

1. 最大化 \(K=1\) PSNR；
2. 与最大值相差不超过 0.05 dB 的候选中，选择 mean next-state MSE 最低者；
3. 若相对差不超过 \(10^{-6}\)，选择 trajectory-level next-state MSE ratio 的 95th percentile 更低者；
4. 再相同时选择更小的 \(\beta\)。

若没有任何 \(\beta>0\) 通过，输出 “v02_beta_selection.md/JSON”，判定 trajectory-preserving refinement FAIL/STOP。可以如实报告 trajectory training 的 val 结果，但不得建立 V0.2-full、进入 full-route K selection 或用 \(\beta=0\) 偷换定义。

---

## 11. Stage D：K progression 与选择

使用 seed1-B1 selected checkpoint 与冻结的 \(\beta>0\)，在全部 400 条 val 上按 SA-V02-ADJ-1 顺序运行。

定义 overall PSNR 为 400 条 trajectory 的 per-image PSNR mean；五个 terminal buckets 固定等量，因此同时报告 macro-by-bucket mean 作为核对。

1. 始终先计算 \(K=0\) 与 \(K=1\)；
2. 只有当

\[
\operatorname{PSNR}_{K=1}
\ge
\operatorname{PSNR}_{K=0}-0.05\ \mathrm{dB}
\]

时才运行 \(K=3\)；
3. 只有当

\[
\operatorname{PSNR}_{K=3}
\ge
\max(\operatorname{PSNR}_{K=0},\operatorname{PSNR}_{K=1})
-0.05\ \mathrm{dB}
\]

时才运行 \(K=5\)。

最终选择：

1. 找到已运行 K 中最大 overall PSNR；
2. 将距最大值不超过 0.05 dB 的 K 作为 quality-tied set；
3. 在 tied set 中选择 mean LPIPS 最低者；
4. 若 LPIPS 与最低值差不超过 \(10^{-4}\)，选择更小 K。

若 selected \(K=0\)，判定 progressive bridge 主 claim FAIL/STOP；不得将 one-shot 包装为 V0.2 bridge 成功。

---

## 12. Seed2 触发与冻结顺序

Seed2 必须在 frozen test 前完成，且不重新选择 \(\lambda_T,\lambda_B,\beta,K\)、refinement optimizer 或任何 selection rule。

只有 seed1 val 同时满足以下确认信号才触发 seed2：

1. selected \(K>0\)；
2. V0.2-full selected K 相对同 checkpoint \(K=0\) 的 paired mean PSNR gain 至少 0.20 dB；
3. 该 PSNR paired 95% bootstrap CI lower bound \(>0\)；
4. V0.2 no-refine \(K=1\) one-step next-state MSE 相对 seed1-B0 continued/no-refine matched diagnostic 至少下降 25%，且 paired MSE delta 的 95% CI upper bound \(<0\)；同时报告相对 V0.1 original diagnostic 的结果；
5. \(\beta>0\) refinement gate 已通过。

若不满足，不通过增加 seeds 寻找偶然成功；只按已完成的 seed1 证据和其他 stage gate 决定 STOP 或进入单-seed frozen test。

若触发 seed2：

- seed2-B1 使用完全冻结的训练协议；
- seed2-B0 使用同 seed、同数据顺序、同 extra-update budget；
- 训练长度复制 seed1 的 10k/15k 决定；
- \(\beta\) 与 K 直接复用 seed1 冻结值；
- B0 checkpoint 继续对齐 seed2-B1 selected extra-update index；
- 不以两个 seed 中表现更好者作为唯一最终模型。

Seed2 是确认性运行。若 seed2 val 在冻结的 \(\beta,K\) 下不能同时满足 selected \(K>0\)、full-vs-K0 paired PSNR CI lower bound \(>0\) 和 one-shot preservation，则在 test 前 STOP，并报告 seed1 signal 未被第二 seed 确认；不得丢弃 seed2 或只测试 seed1。

---

## 13. Comparator 执行路径

所有需要 transition 的 comparator 使用 SA-V02-ADJ-1、同一输入、同一 metric code 和同一 frozen K。

| 名称 | Checkpoint / training | 实际执行路径 |
|---|---|---|
| V0.1 original | 原始冻结 20k | 原冻结 \(K=0\) one-shot clean output；其 no-refine \(K=1\) 只作 next-state diagnostic，不冒充原 selected method |
| B0 continued \(K=0\) | 旧 loss，checkpoint index 对齐 B1 | one-shot clean output，用于控制“旧目标多训练相同步数” |
| B0 continued no-refine | 同上 | 每步预测 \((\hat C,\hat\xi)\)，直接 render 下一状态；仅用于 matched next-state/rollout diagnostic |
| V0.2 \(K=0\) | B1 selected | 同一 V0.2 checkpoint one-shot clean output，无 transition |
| V0.2 no-refine | B1 selected | 每步预测 \((\hat C,\hat\xi)\)，直接用 \(\hat\xi\) render next state，不优化 \(\xi\) |
| V0.2 old-refine | B1 selected | 每步执行 \(\beta=0\) terminal-only \(\xi\) refinement |
| V0.2 full | B1 selected | 每步执行 validation 冻结的 \(\beta>0\) trajectory-preserving refinement |
| oracle-\(\xi\) | B1 selected | 每步 clean 仍由网络预测；renderer 以真实 \(\xi_i\) 和当前预测 clean 生成下一状态；不替换 true clean、不做 \(\xi\) optimization |
| teacher-forced | B1 selected | 每一步网络输入 dataset 中真实 \(X_{i,g_k}\)；记录预测 next-state error，但下一步继续喂 true state；最终输出终点网络 clean branch |

Oracle-\(\xi\) 隔离 realization error；teacher-forced 隔离 rollout state-distribution drift。两者均不是可部署方法，也不得用于选择 checkpoint、\(\beta\) 或 K。

---

## 14. Metric 与 paired statistics

### 14.1 Metric authority

V0.2 原样复用 §1 只读核验出的 V0.1 image range、clamp、padding、valid mask/crop、PSNR、SSIM、LPIPS backbone/version、L1 与 aggregation。Metric audit 与代码 hash 必须在第一次 validation 前冻结；若 source 与 frozen report 不一致，先报告 mismatch，不得静默“修正”。

指标层级：

1. primary image-quality metric：PSNR；
2. important secondary：LPIPS；
3. auxiliary：SSIM、L1；
4. mechanism metrics：observation residual、next-state MSE。

### 14.2 Paired bootstrap

独立单位是 trajectory，不是 transition、patch 或重复 renderer call。

- 使用 10,000 次 paired bootstrap；
- 固定 seed 2026082899；
- 在五个 terminal-\(G\) bucket 内分别有放回采样与原 bucket 相同数量的 trajectory，再合并计算 overall statistic；
- 95% CI 使用 bootstrap percentile \([2.5\%,97.5\%]\)；
- 同时报告 overall、per-\(G\)、paired wins 与完整 paired delta distribution；
- paired win 定义为该 trajectory 的 primary-metric delta \(>0\)。

Paired delta 方向统一为 proposed minus comparator：PSNR/SSIM 的正值更好，LPIPS/L1/residual/next-state MSE 的负值更好。因此文中的 quality CI lower bound \(>0\) 与 error CI upper bound \(<0\) 具有唯一含义。

对 \(K>1\) 的 observation residual、next-state MSE、runtime 和 simulator calls，先在每条 trajectory 内对其实际 \(K_{\mathrm{eff}}\) 个 transition 求 mean 或 sum（按指标语义预先固定），再进行跨-trajectory aggregation；不得把 transition 当成独立样本。具体地，residual 与 next-state MSE 使用 trajectory 内 mean，runtime 与 calls 使用 trajectory 内 sum。

若运行两个 training seeds，同一 test trajectory 先对两个 seed 的 paired delta 取平均，再以 trajectory 为独立单位 bootstrap；同时单独报告每个 seed，禁止挑选较好 seed。

---

## 15. Frozen-test eligibility、执行与科学判据

### 15.1 进入 test 前必须冻结

- seed1 以及若触发的 seed2 model/checkpoint hashes；
- \(\lambda_T,\lambda_B\)；
- refinement optimizer；
- \(\beta>0\)；
- selected \(K>0\) 与 SA-V02-ADJ-1；
- 所有 comparator 路径；
- data/sample manifest；
- metric code、bootstrap seed 和 qualitative IDs。

进入 test 的最低门：

1. A0、A1、checkpoint、\(\beta\) gate 均通过；
2. selected \(K>0\)；
3. seed1 val 上 V0.2-full selected K 相对同 checkpoint K=0 的 paired PSNR mean \(>0\)，且 95% CI lower bound \(>0\)；
4. seed1 val 上满足

\[
\operatorname{PSNR}_{K=0}^{B1}
\ge
\operatorname{PSNR}_{K=0}^{B0\ \mathrm{continued}}-0.1\ \mathrm{dB},
\]

且 simulator integrity 检查通过。

若已触发 seed2，seed2 还必须通过 §12 的确认门；否则不得进入 test。

不满足则在 val 阶段 STOP，不提前查看 test。

### 15.2 一次性 test

在冻结的 1,000 test trajectories 上一次性运行：

- V0.1 original；
- B0 continued \(K=0\) 与 B0 no-refine diagnostic；
- V0.2 \(K=0\)；
- V0.2 no-refine；
- V0.2 old-refine；
- V0.2 full；
- oracle-\(\xi\)；
- teacher-forced。

若有两个 seeds，对每个 seed 的 B0/B1 系列都运行，不挑 seed。Test 期间禁止重新选择 checkpoint、\(\beta\)、K、optimizer、loss weight、mask、metric 或 schedule。

报告 PSNR/SSIM/LPIPS/L1、paired wins/deltas/CI、next-state MSE、observation residual before/after、runtime、simulator calls、peak memory 与 NPU-hours。

### 15.3 三个 claim 分开判定

**Claim T：trajectory training 成立**

主要比较 seed-matched：

\[
\text{V0.2 no-refine }K=1
\quad\text{vs}\quad
\text{B0 continued no-refine }K=1.
\]

PASS 要求 one-step next-state MSE paired mean 更低且 95% CI upper bound \(<0\)。同时报告相对 V0.1 original diagnostic 的结果。

**Claim R：trajectory-preserving refinement 成立**

主要比较：

\[
\text{V0.2-full selected }K
\quad\text{vs}\quad
\text{V0.2 no-refine same }K,
\]

并与 old-refine 对照。PASS 要求：

1. observation residual mean 至少下降 20%；
2. next-state MSE 不超过 no-refine 的 1.05 倍；
3. PSNR paired mean gain \(>0\)，且 95% CI lower bound \(>0\)。

若只降低 residual、没有最终质量增益，报告 mechanism partial，不得判为 refinement 全成功。

**Claim P：progressive physical bridge 成立**

primary comparison：

\[
\boxed{
\text{同一 V0.2 checkpoint：full selected }K>0
\quad\text{vs}\quad
K=0\text{ one-shot}.
}
\]

PASS 要求 PSNR paired mean gain \(>0\)，且 95% CI lower bound \(>0\)。LPIPS 为重要 secondary，不替代 primary PSNR 判据。

**One-shot preservation**

V0.2 \(K=0\) 相对 matched B0 continued \(K=0\) 的 mean PSNR 下降不得超过 0.1 dB。

只有 Claim T、Claim R、Claim P 和 one-shot preservation 同时通过，才可称“V0.2 full protocol success”。若仅部分成立，必须分别报告，不得合并成整体成功。

---

## 16. 训练次数上限

### Seed1

1. 一次 A0 gradient audit；九个权重组合只做代数筛选，不训练；
2. 一个 3,000-update A1 pilot；
3. 一对 B0/B1 full fine-tunes。

### 可选 Seed2

只有 §12 gate 触发时，再运行一对 seed2 B0/B1；不重新做 A0、A1、\(\beta\) 或 K search。

不运行 \(\lambda_T\times\lambda_B\) full-training grid，不因结果不利增加未声明 runs。经证据确认的实现失败重跑必须保留 invalidated run 记录、原因与相同 manifest/config hashes。

---

## 17. 过程产物

### 17.1 Training

保存：

- total / \(\mathcal L_C\) / \(\widehat{\mathcal L}_T\) / \(\mathcal L_T^{\mathrm{full}}\) / \(\mathcal L_B\)；
- shared、clean、realization head gradient norms；
- learning rate；
- per-\(g\) clean PSNR/LPIPS；
- full trajectory residual；
- one-step next-state MSE；
- \(\hat\xi\) mean/std/norm；
- fixed qualitative montage；
- RNG state；
- checkpoint/config/source/dataset hashes；
- simulator calls、runtime、memory 与 NPU-hours。

### 17.2 Refinement

每一步保存：

\[
X_{g_k},\quad
\hat C_k,\quad
\hat\xi_k,\quad
\xi_k^*,\quad
\tilde X_{g_{k+1}},\quad
X_{g_{k+1}}.
\]

以及 \(J_{\mathrm{obs}}\)、\(J_{\mathrm{keep}}\)、next-state MSE、clean metrics、Zernike/tilt update norm、optimizer trace、calls、runtime 与 memory。

### 17.3 自动报告

至少生成：

- v02_identity_audit.md；
- v02_gradient_audit.md；
- v02_pilot_report.md；
- v02_full_training_report.md；
- v02_beta_selection.md；
- v02_k_selection.md；
- v02_frozen_test_report.md；
- machine-readable JSON/CSV；
- fixed-ID qualitative trajectories。

---

## 18. 科学 STOP 条件

### Identity STOP

V0.1 接口、checkpoint、simulator、metric、data manifest 或 fixed-\(G_i\) 身份不一致。

### A0 STOP

无健康 \((\lambda_T,\lambda_B)\) 候选、gradient 非有限、分母为零或 simulator backward 失败。

### Training-modification STOP

A1 未同时达到 next-state MSE 至少下降 25%、\(K=1\) no-refine 至少提升 0.1 dB、\(K=0\) 恶化不超过 0.1 dB及稳定性门。

### Refinement STOP

没有任何 \(\beta>0\) 同时通过 observation-residual 与 next-state gate。此时 \(\beta=0\) 只能作为 old-refine control。

### Progressive STOP

若 \(\operatorname{PSNR}_{K=1}<\operatorname{PSNR}_{K=0}-0.05\) dB，则不运行 K=3；若 \(\operatorname{PSNR}_{K=3}<\max(\operatorname{PSNR}_{K=0},\operatorname{PSNR}_{K=1})-0.05\) dB，则不运行 K=5；最终 selected \(K=0\) 时 progressive claim FAIL。

### Test-isolation STOP

进入 test 的预冻结门未满足，或任何人请求在 test 结果后更改 checkpoint、\(\beta\)、K、metric、optimizer、loss weight 或新增 seed。

---

## 19. Codex 执行边界

Codex 可自行处理：

- 独立 v0.2 代码/config/run 组织；
- source interface adapter；
- batch/microbatch、gradient accumulation；
- 数据加载、cache、recompute 与 memory optimization；
- 经 scalar-render 数值等价验证的 vectorization；
- checkpoint/resume；
- metric aggregation、统计、自动报告与图表；
- 不改变数值语义的 NPU adaptation。

Codex 不得静默：

- 改成 image-only \(S(X)\) 或删除 \(g\) conditioning；
- 删除每步 \(\xi\) optimization；
- 改变三项训练 loss 或 refinement 方程；
- 新增 loss、head、module、latent 或 continuous-\(g\)；
- 改变 \(\xi\) parameterization/resolution；
- 重分配固定 \(G_i\)、更换 split 或重生成数据；
- 联合调 \(\beta\)、training LR 和 refinement optimizer；
- 在 test 上调参；
- 因结果不利增加 runs；
- 覆盖、修改或删除 V0.1。

一句话执行目标：

\[
\boxed{
\text{以冻结的 update/data-matched 与 paired-test protocol 判断：}
\text{跨-severity训练是否修复 }\hat\xi,\quad
\text{以及 }\beta>0\text{ refinement 和 physical bridge 是否各自带来可归因增益。}
}
\]

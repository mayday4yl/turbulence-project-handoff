# V0.31｜阶段一：方法定义、代码适配与验收

日期：2026-09-16  
交给 Codex 的执行文档，第一份。

> 当前只执行阶段一：理解并实现 V0.31，完成针对性验收后停止。不启动正式训练，不执行恢复网络的 `optimizer.step()`。即使同目录或附件中同时出现阶段二、三文档，也不构成执行后续阶段的授权。

## 0. 任务、版本与数据位置

此实验正式命名为 **V0.31 Overfit100**。旧 V0.3 仅作为代码来源，不再作为本轮算法定义；本文件是用户最新手写图及 A/B/C 确认的可执行文字版本。

现有数据已经由用户构造，直接只读使用：

```text
DATASET_ROOT=/data2/yanglin/v03_overfit100_g4321_n4/data/places_p2s_bridge_v03_overfit100_g4321_n4/
DATASET_ID=places_p2s_bridge_v03_overfit100_g4321_n4
METHOD_VERSION=V0.31
```

**方法改名不等于数据改名。不要移动、重命名、复制或重建该数据集，不改已有 manifests/hash。**

建议为新代码、配置和结果建立独立根目录：

```text
V031_ROOT=/data2/yanglin/v031_overfit100_g4321_n4/
  code/
  docs/
  configs/
  artifacts/
  runs/
```

这是新实验的建议工作目录，不是已核验存在的路径。先检查归属；已有内容时核对身份，不覆盖其他实验。在阶段一报告中记录最终 `V031_ROOT`，后续阶段从该报告读取，不另猜路径。

执行位置为 de36；历史 SSH alias `liteserver-de36`、源码线索 `/data0/yanglin/physical_bridge`、assets 线索 `/data0/yanglin/vendor/PiRN/assets/utils` 仅用于定位，须现场只读确认。仅使用 de36 物理 NPU 0–3 中空闲、健康的设备，不终止或干预其他任务。

### 权威边界

- 本文件定义本轮方法、接口、梯度和验收要求；阶段二确定具体运行参数。
- 数据构造依据是《V0.3 新配置：100-source / g4321 / n4 小样本数据集构建 Prompt（修订版 2）》。它约束数据身份与构造，不要求本阶段再次生成数据。
- 旧 V0.3/half 的运行配置、旧报告及上一份混合 handoff 仅作历史参考，不能覆盖本文件。
- 复用既有恢复网络结构和已用于该数据集的 simulator 数学实现；不要更换 backbone、引入新 latent、添加新训练目标或更改物理模型。

## 1. 先读现场，再适配代码

先读取现有数据根目录中的：

```text
source_manifest.jsonl
trajectory_manifest.jsonl
start_manifest.jsonl
dataset_summary.json
integrity_report.json / integrity_report.md
hash_manifest.json
subsets/subset_summary.json
subsets/overfit_16traj_starts.jsonl
```

读取既有完整性/replay 报告，检查数量、引用及四起点元数据，少量实际加载并核对 simulator 身份。数据构建规范不是实际通过报告；有具体缺项就报告，不以重新运行全量生成来代替核查。

然后只读审计旧 V0.3 入口、loader、severity helper、rollout、loss、梯度处理和 inference 求解器。把不兼容位置列成短表：

```text
文件/函数 → 当前实际行为 → V0.31 要求 → 修改或保留
```

找到冲突后直接按本文适配并继续验收，不需要为普通工程细节逐项等待确认。只在必须改变科学定义、网络结构或 simulator 数学行为时停止询问。

新实验使用独立入口和配置。不要为了支持新四档而改动旧实验依赖的全局六档常量、共享 schedule 或历史配置。允许复用经核对兼容的底层函数。

## 2. 数据和索引的精确定义

### 2.1 数据对象

本轮数据为 100 个固定 clean、每 clean 4 个 simulator-native realization，总计：

| 对象 | 数量 |
|---|---:|
| source / 固定 clean | 100 |
| 物理 trajectory，即固定 `(C, xi)` | 400 |
| 真实退化状态 | 1600 |
| 起点记录，即 `(trajectory_id, initial_G)` | 1600 |

数据内每条物理 trajectory 固定同一 clean 和同一完整 xi，只改变 severity，直接渲染四个状态：

\[
X_g=F(C,\xi,g),\qquad g\in\{4,3,2,1\}.
\]

不是把 `X_4` 再退化得到 `X_3`。

权威 clean/state 使用原有 FP32 数组，单图 `[3,256,256]`。不做额外 crop、flip、rotation、噪声或图像量化，不给 simulator 输出额外加 clamp。

原生 xi 的两块输入（不含 batch 维）应与数据及现场实现一致：

```text
epsilon_zernike: [36,512,512]
epsilon_tilt:    [512,512,2]
```

核对网络原始输出、已有解码/参数化以及实际进入 simulator 的两块张量，记录各自 shape。不能用 GT xi、seed、trajectory ID 或低维替代量偷偷代替网络预测。若旧网络不满足这一接口，先报告具体差异，不另行设计网络。

### 2.2 四起点与预测次数

\[
(g_1,g_2,g_3,g_4)=(4,3,2,1).
\]

`g_i` 的下标是序列位置；文件名 `X_4` 的下标是实际强度。实现优先使用 `initial_G`、`current_g`、`next_g`、`num_predictions`，避免两种下标混用。

| 原始起点 G | 唯一真实起点 Y | severity_path | 完整 K | transition 次数 |
|---:|---|---|---:|---:|
| 4 | X_4 | [4,3,2,1] | 4 | 3 |
| 3 | X_3 | [3,2,1] | 3 | 2 |
| 2 | X_2 | [2,1] | 2 | 1 |
| 1 | X_1 | [1] | 1 | 0 |

**K 是网络总预测次数，one-shot 为 K=1。** 从任一 G 都可评估 one-shot；不是只有 `G=1` 才叫 one-shot。

训练使用该起点完整尾序列，不使用旧 rollout curriculum，不随 update 缩短或增长路径。最后一次网络预测直接给出 clean，不生成 `g=0`，不额外调用网络。

## 3. Training：前向、损失与反向

### 3.1 单路径前向

一个起点记录提供 `(C, Y, G, severity_path)`。令 `K=len(severity_path)`、`s_k=severity_path[k-1]`，初始化：

\[
Z_1=Y=X_G.
\]

第 k 次调用同一个共享参数网络：

\[
(\hat C_k,\hat\xi_k)=S_\theta(Z_k,s_k).
\]

网络输入只有当前图像和当前强度；不要添加 GT xi、GT clean、样本 ID、原始观测旁路或新强度估计器。

仅当 `k<K` 时生成下一输入：

\[
Z_{k+1}=F(\hat C_k,\hat\xi_k,s_{k+1}).
\]

第一步之后，不读取真实较弱状态替换生成状态。真实 `X_3/X_2/X_1` 可以是其他路径的真实起点，但不能是本路径的 teacher-forcing 输入。

### 3.2 每次预测的 loss

两个 loss 都使用单张 RGB 全图 mean MSE，先平均 `C,H,W`，保留样本维度：

\[
\operatorname{MSE}(A,B)=\frac1{3HW}\sum_{c,h,w}(A_{c,h,w}-B_{c,h,w})^2.
\]

每次预测，包括最后一次，都计算：

\[
L_C^{(k)}=\operatorname{MSE}(\hat C_k,C),
\]

\[
L_R^{(k)}=\operatorname{MSE}\bigl(F(\hat C_k,\hat\xi_k,G),Y\bigr).
\]

**整条路径中重退化始终使用原始 G、原始 Y。** 它不拟合当前生成输入 `Z_k`，也不使用当前较弱强度 `s_k`。预测侧同时使用预测 clean 和预测 xi，不能改成 `F(C, xi_hat, G)`。target 直接使用保存的 Y，不必重复渲染 GT。

路径内平均，再对路径/batch 等权平均：

\[
L_b=\frac1{K_b}\sum_{k=1}^{K_b}\left(L_C^{(b,k)}+\lambda_R L_R^{(b,k)}\right),
\qquad
L=\frac1B\sum_{b=1}^{B}L_b.
\]

`lambda_R` 是显式配置项，阶段二使用本轮建议值 0.2；本阶段不搜索它。不能用全 batch 的总预测次数 `sum(K_b)` 作统一分母，也不另行按当前 g 重加权。

训练只包含上述两项 loss。没有 raw-xi regression、next-state GT loss、xi 跨步一致性正则、旧 prox/prior 项或其他辅助目标。GT xi 与真实较弱中间状态仅供独立复现/diagnostics。

### 3.3 梯度与更新边界

**Training 不执行 xi refinement。** 下一状态直接使用当前网络预测出的两路结果。

正式路径必须保留：

\[
(\hat C_k,\hat\xi_k)\rightarrow F\rightarrow Z_{k+1}\rightarrow S_\theta\rightarrow L_{k+1:K}.
\]

实现要求：

- 预测 clean、预测 xi、其原生两块张量和生成状态都不 `detach`；不通过 `.data`、NumPy、重新构造 leaf 等方式切断路径。
- 不用 `no_grad` 或 `inference_mode` 包住训练中的 simulator。冻结 simulator 参数，但必须允许梯度传回其输入。
- 完整展开一条路径并汇总逐步 loss 后再 backward；不能在路径内部更新参数。
- 可以按完整路径组成的 microbatch 累积梯度；同一有效 batch 完成后才更新参数。不要求同时保存全部有效 batch 的计算图。
- 日志可保存脱离计算图的数值副本，但副本不再进入训练前向。

这是普通完整跨步反传，不要求二阶梯度或对 refinement 展开求导。

### 3.4 便于核对的算法骨架

以下为语义伪代码，不规定模块名或分布式封装方式：

```text
path_losses = []
for record in microbatch:
    C, Y, G, path = load_start_record(record)
    Z = Y
    step_losses = []
    for k, current_g in enumerate(path):
        C_hat, xi_hat = S_theta(Z, current_g)
        Y_hat = F(C_hat, xi_hat, G)
        step_losses.append(mean_mse(C_hat, C) + lambda_R * mean_mse(Y_hat, Y))
        if k + 1 < len(path):
            Z = F(C_hat, xi_hat, path[k + 1])   # 保持计算图
    path_losses.append(mean(step_losses))
microbatch_loss = mean(path_losses)
# 按有效 batch/分布式归约规则缩放，再 backward；累计完成后才 step。
```

## 4. Inference：固定原始观测，择优初始化

本轮 inference 接口为 `(Y,G) → C_hat`，G 已知。网络及 simulator 权重不更新；GT clean/xi 不能进入求解器或候选选择，clean GT 只在外部计算评估指标。

第 k 次预测后固定当前 `C_hat_k`，定义：

\[
J_k(\xi)=\|F(\hat C_k,\xi,G)-Y\|_2^2.
\]

只优化 xi，不优化 clean 或 G，不使用旧 prox/prior 正则。

仅当 `k<K` 时选择初始化：

\[
\xi_k^{\mathrm{init}}=
\begin{cases}
\hat\xi_1,& k=1,\\
\xi_{k-1}^{*},& k>1,\ J_k(\xi_{k-1}^{*})\le J_k(\hat\xi_k),\\
\hat\xi_k,& \text{其他情况}.
\end{cases}
\]

两个候选必须用**当前同一个 clean 预测**重新计算 objective，不能把上一步保存的 `J_(k-1)` 拿来比较。以整组 native xi 为单位选择，不能把两块分别择优拼接，也不平均两个候选。相等时继承上一步 refined xi。

只对被选中的初值运行一次 refinement：

\[
\xi_k^*=\operatorname{Refine}(J_k;\xi_k^{\mathrm{init}}),
\qquad
Z_{k+1}=F(\hat C_k,\xi_k^*,s_{k+1}).
\]

不是分别优化两个候选后选最优。最后一次预测直接输出 `C_hat_K`，不做末次无用 refinement；one-shot 没有 refinement/transition。

推理数值求解器的 optimizer、学习率、内循环次数、reduction、停止/返回规则及 optimizer state 处理方式尚未完整指定。本阶段只读列出旧实现的实际设置，把这些项做成显式接口；不静默继承旧正则、梯度归一化、投影或回退策略。阶段二给出并确认一组具体数值配置后再作正式质量评估。

训练的 mean MSE 不自动决定 inference 的 reduction。候选排序和内循环必须使用一致的已声明 objective；sum/mean 的差异要在求解器配置中明示。

**物理叙述边界：** GT 数据严格固定 C、xi，只改 g；学习 rollout 中 clean 和同一未知 realization 的估计会更新。不要把生成路径宣称为已经恢复了真实 xi，或严格来自一组始终不变的预测 `(C,xi)`，也不要为实现这种更强说法添加新约束。

## 5. 实现验收：只做能证明关键定义的检查

使用少量已冻结样本和真实网络/simulator，不做学习率搜索、训练收敛试验或完整质量评估。验收可计算 forward/backward，但恢复网络参数更新次数必须为 0。

### 5.1 四起点、loss 与路径计数

覆盖 G=4/3/2/1，记录每步输入来源、网络条件、重退化 G/Y 身份、transition 强度、两个 loss 和预测次数。确认最后一步也有两个 loss、training refinement 次数为 0。

不含诊断/重计算时，每条训练路径逻辑上有 K 次观测重退化和 K−1 次 transition，即 `2K−1` 次图像渲染：

```text
G=4: 4 次网络预测 / 7 次逻辑渲染
G=3: 3 次网络预测 / 5 次逻辑渲染
G=2: 2 次网络预测 / 3 次逻辑渲染
G=1: 1 次网络预测 / 1 次逻辑渲染
```

分清 per-sample 逻辑渲染和批处理函数调用数。对一个 mixed-length batch 手工核对路径内平均、batch 平均和梯度累积缩放；padding 步不得进入 loss 分母。

### 5.2 终端 loss 到早期预测：主证据

在真实 G=4 完整 forward 中保留各次实际送入 simulator 的张量。仅使用：

\[
L_{\mathrm{probe}}=\operatorname{MSE}(\hat C_4,C).
\]

**不加入任何早步 clean/re-degradation loss，也不加入末步 re-degradation loss。** 直接查询终端 probe 对 k=1,2,3 的以下张量的梯度：

```text
C_hat_k
epsilon_zernike_hat_k
epsilon_tilt_hat_k
```

可用 `autograd.grad`。逐项报告连接/None、finite、gradient norm、max_abs。对象必须是真实前向使用的张量，不是事后 detach 再 `requires_grad_(True)` 的副本。

只看共享网络参数有梯度不能证明跨步连接；看包含早步自身 loss 的总 loss 对早期输出有梯度，也不能证明跨步连接。

### 5.3 人为断图负对照

固定同一权重、输入、随机状态和相关 buffer，另跑独立测试副本，仅在某一个 transition 后将下一状态 stop-gradient。三处边界分别检查。

前向状态、最终预测和 probe 值应与正常版一致到声明的数值容差；终端 probe 对断点之前相应预测张量的反向连接应消失。共享网络参数仍可能有梯度，这是预期现象。

测试用 detach 不能进入正式 trainer。正常版梯度全零时，区分图未连接、初始化退化、输入不敏感和局部导数为零，不能把“有 grad 字段”写成通过。必要的非退化诊断 fixture 与正式初始化分开标识，不改正式网络定义以换取 PASS。

### 5.4 Simulator 各输入分支的局部数值检查

对 clean、epsilon_zernike、epsilon_tilt 分别做少量方向导数检查：比较输出随机标量投影的 autograd 导数和中心有限差分。固定其他输入、方向和随机状态，事先声明精度、扰动尺度和容差，输出实际误差。

不对高维 xi 穷举 Jacobian，不改 simulator 数学行为。局部检查不能替代真实 rollout 的终端 probe 和负对照；非光滑点或数值分辨率不足应如实说明并定位。

### 5.5 运行可行性与 inference 选择函数

在授权空闲设备上，用真实 G=4 完整路径测 FP32 forward+backward 峰值内存和耗时。先测试 microbatch=1，可容纳再测目标值 2；这只是阶段二确定 batch 的依据，不是训练授权。记录设备总容量、是否包含 optimizer state 内存；尚未分配的状态内存不能当作已经覆盖。

检查 inference 选择函数的当前-clean 重评估、整组 xi 选择、平局规则、只 refinement 一次和末步边界。未确定内循环数值时只测选择逻辑/接口，明确标记 `refinement_numeric_config_pending`，不谎报完整推理求解器已通过质量验证。

## 6. 交付、状态与停止点

将本文件保存到新实验 `docs/`，交付独立实现、必要测试及一份：

```text
artifacts/V031_IMPLEMENTATION_REPORT.md
```

报告集中给出：实际 V031_ROOT；代码/网络/数据/simulator 身份；旧实现差异表；四起点 trace；loss 聚合结果；终端梯度、负对照及局部导数数值；内存/耗时；可复用的执行命令；待确定的 inference 数值项。

核心 trainer 和已定义 inference 选择逻辑通过时，报告：

```text
stage1_status: v031_implementation_acceptance_passed
formal_training_started: false
restoration_optimizer_updates: 0
refinement_numeric_config: pending_stage2
```

这个 PASS 只表示已定义范围的实现验收，不表示训练收敛、完整 inference 数值协议已冻结或方法有效。任何核心验收未通过则明确报告 `blocked` 和具体原因。

普通代码错误可修复后重测，保留简短记录。不要擅改 scientific definition、换 simulator、截断路径、增加 loss 或通过 detach 解决内存问题。数据身份不符、梯度无法验证、必须改变方法或无授权空闲设备时停止报告。

**交付后停止，等待阶段二。不要启动正式训练，也不要自动安排 1→16→400 三轮过拟合。**

---

资料依据：用户 2026-09-16 最新手写图、A/B/C 确认及 V0.31 命名/数据路径说明；数据构造 Prompt 修订版 2，原上传名 `99a32b45-23b8-491c-b5e9-de1d224c6c61.md`，SHA-256 `a1b432c113cf63f6ef1d8e0a6452c7af6c56c1a7ac0a450459df86976bd07d92`。前一份 `V03_OVERFIT100_TRAINER_HANDOFF.md` 已拆分为本三阶段文件，不作为另一份并行 authority。

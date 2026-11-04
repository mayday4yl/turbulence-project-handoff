# DiffUIR 原生扩散机制：理论与固定源码审计

日期：2026-10-28；官方 commit：`c71437ce86ad1b098aa9be4678dc82e8b99649df`。

本报告解释算法，不修改算法。行为实验、样本数与统计结论见同目录 `V6_DIFFUSION_MECHANISM_DIAGNOSIS_ZH.md`。源码事实、论文陈述、下面自行展开的代数推导分别标注。

## 1. 先把三种图像和两种时间分开

外部图像 `C,Y` 用 RGB 单位，shape `[B,3,256,256]`。进入 DiffUIR 时记

\[
c=2C-1,\quad y=2Y-1,\quad r=y-c=2(Y-C).
\]

- **Condition `y`**：当前恢复调用的退化观测，三次内部去噪中保持同一张图。
- **扩散状态 `x_τ`**：随机 Forward 构造或 Reverse 更新的另一张 RGB 张量；绝不能把它和 Condition 当成同一变量。
- **Clean 估计 `ĉ_τ`**：网络预测残差后计算的 `clamp(y−r̂_τ,−1,1)`，它也不等于中间扩散状态。
- **内部索引 `τ∈{0,…,999}`**：决定扩散 schedule 与原生时间编码。
- **V6 物理时间 `t=1−g/4`**：决定湍流强度和外层递归，仅 A/B 新增；D0 无物理时间输入。

归一化是线性换算；若物理观测Y原本越界，不会自动clamp回[-1,1]。

模型输入是 `cat(x_τ,y)`，shape `[B,6,256,256]`。没有 VQ、latent codec、文字 prompt、ControlNet 或额外任务 token。官方 `task='blur'` 参数在当前网络计算分支未用于生成任务嵌入；五任务由图像及共享模型处理。

## 2. DDPM → RDDM → DiffUIR：Forward 的区别

### 2.1 标准 DDPM

为避免与 DiffUIR 的残差系数混名，用 `β_j^VP` 表示 DDPM 每步方差，`A_τ=∏_{j=0}^τ(1−β_j^VP)`：

\[
q(x_τ|x_{τ-1})=\mathcal N(\sqrt{1-β_τ^{VP}}x_{τ-1},β_τ^{VP}I),
\quad x_τ=\sqrt{A_τ}c+\sqrt{1-A_τ}\epsilon.
\]

标准条件 DDPM 可以把 Y 交给去噪网络，而不改变这条 Forward。有限终点仍有微量 Clean 系数，理想终点趋向标准高斯。常见训练目标是噪声 MSE，但预测 x0、score 或其他等价参数化也可以属于 diffusion；预测目标名字不能单独决定算法类别。[DDPM 原论文](https://arxiv.org/abs/2006.11239)。

### 2.2 RDDM

定义配对图像残差 `r=y−c`，每步加定向残差漂移和独立噪声：

\[
x_j=x_{j-1}+\alpha_j r+\beta_j\epsilon_j,
\quad a_τ=\sum_{j\le τ}\alpha_j,
\quad b_τ^2=\sum_{j\le τ}\beta_j^2.
\]

相加并合并独立高斯，得到

\[
x_τ=c+a_τr+b_τ\epsilon,
\quad q(x_τ|c,y)=\mathcal N(c+a_τ(y-c),b_τ^2I).
\]

当 `a_T→1`，均值趋向 Y，而非零。故 Forward 本身已有退化图的定向信息。这是 RDDM 的残差扩散与噪声扩散，不是简单地对 Y 做一次残差回归。[RDDM 原论文](https://arxiv.org/abs/2308.13712)。

### 2.3 DiffUIR 的 SDT

DiffUIR 在漂移里进一步减去 `δ_j y`。论文正文式（4）的展开，用本报告符号为

\[
x_j=x_{j-1}+\alpha_jr+\beta_j\epsilon_j-\delta_jy,
\qquad d_τ=\sum_{j\le τ}\delta_j,
\]
\[
\boxed{x_τ=c+a_τ(y-c)+b_τ\epsilon-d_τy
=(1-a_τ)c+(a_τ-d_τ)y+b_τ\epsilon.}
\]

推导只用了确定漂移相加、独立高斯方差相加；这里高斯分布的**方差是 `b_τ²`，标准差是 `b_τ`**。不要因为论文某处用 `β̄` 写在分布第二参数位置，就把标准差和方差混用。

`SDT` 是 Shared Distribution Term：单步为 `δ_j y`，累计为 `d_τ y`。它不是学习得到的新 Head，也不是额外训练 Loss。

### 2.4 为什么仍叫残差条件扩散，而不能直接改称 Flow Matching

这里的 `r=Y−C` 是**同一配对样本的图像差**，不是扩散 Gaussian Noise，也不是物理 ξ，更不是自动等于连续时间速度。

V6 仍用上述随机 q、随机 τ、原生残差监督与官方逆采样。若把同一个重参数化路径连续化，其路径导数应含

\[
\dot x_τ=\dot a_τr+\dot b_τ\epsilon-\dot d_τy.
\]

训练直接回归 r，并未直接监督这个完整速度。Flow Matching 的对象是指定概率路径上的向量场及相应匹配目标；两类方法可有数学联系，但“残差预测”或“采样确定”都不够建立这种等价。FM 也可以使用高斯路径，所以“有噪声”同样不是二者的充分分类标准。

另一个有用的自行推导是：固定配对样本的 Forward Gaussian score 为 `∇x log q(x|c,y)=−ε/b_τ`。由残差预测反推 ε̂，就能构造 `ŝ=−ε̂/b_τ`；所以 residual 是一种可转为噪声/score 的参数化，并非“预测的是图像残差，所以扩散没了”。但对边缘条件分布的准确 score 通常涉及条件均值；原生 L1 的最优点是逐元素条件中位数，新增 Clean Loss 又改变目标。因此该参数化关系**不保证当前网络已学到校准的真实后验 score**。[Flow Matching 原论文](https://arxiv.org/abs/2210.02747)。

## 3. 代码真实使用的三个系数

固定版本 `src/model.py:661–693` 先造 VP schedule，然后重参数化为残差 schedule：

\[
β_j^{VP}=\operatorname{linspace}(10^{-4},0.02,1000)_j,
\quad A_τ=\prod_{j=0}^{τ}(1-β_j^{VP}),
\]
\[
\boxed{a_τ=1-\sqrt{A_τ},\qquad b_τ=\sqrt{1-A_τ}.}
\]

SDT 使用

\[
δ_j=10^{-6}+\frac{j}{999}(0.0018-10^{-6}),
\quad d_τ=\operatorname{clip}\left(\sum_{j=0}^{τ}δ_j,0,1\right).
\]

本配置累计最大 0.9005，未触发上限 1。源码 `ResidualDiffusion` 的**构造器默认** `delta_end=0.0015`，但官方 `train.py/test.py` 和 V6 验收配置明确传入 **0.0018**；不能用默认值代替实际配置。`gen_coefficients()` 虽在文件中存在，本分支的 schedule 不调用它。

`alphas_cumsum` 是这里的 a，既不是 DDPM 的 `A_τ`，也不是每步 α；`betas_cumsum` 是 b，`betas2_cumsum` 是 b²；`delta_cumsum` 是 d。变量名很容易造成误读。 代码τ=0已经是第一个噪声级（b0约0.01），不是精确Clean端点；逆采样以−1作为直接输出Clean的约定。

### 3.1 实测终端分布，全部为归一化 RGB 单位

官方 EMA、D0 U10000 EMA、D0 U13334 EMA 的实际终端 buffer 一致给出（中间索引的 native/sampling 最末位差异另列，不声称两张完整表逐元素全同）：

| 量 | 值 |
|---|---:|
| `a999` | 0.9936471581459045 |
| `b999` | 0.9999797940254211 |
| `d999` | 0.9004999995231628 |
| `betas2_cumsum[999]` | 0.9999596476554871 |
| `1−a999` | 0.006352841854095459 |
| `a999−d999` | 0.0931471586227417 |
| `1−d999` | 0.09950000047683716 |

因此

\[
q(x_{999}|c,y)\approx\mathcal N(0.006352842c+0.093147159y,\;0.99995965I).
\]

相反，官方当前 DDIM 初始化（961行）是

\[
\boxed{x_{\rm init}=0.0995000005y+0.1\epsilon_s,
\quad p_{\rm init}(x|y)=\mathcal N(0.0995000005y,0.01I).}
\]

两者均值相差 `μ_q−μ_init=−(1−a999)r`；方差约相差 100 倍，标准差约相差 10 倍。RGB `[0,1]` 还原时方差均再乘 1/4，倍率不变。

**`sum_scale=0.01` 没有缩放训练 q 的这条线性噪声 schedule。** 它在该分支控制推理初始化噪声（及其他未走的采样路线）；不能把它读成“训练终端噪声方差也是 0.01”。`model.py:962` 还保留了使用 `betas_cumsum[-1]` 初始化的注释替代行，但当前没有执行。

### 3.2 如何判断这个差异

- **论文定义**：正文 Algorithm 2 写初始噪声系数为 `b_T`，并在 `a_T=1` 的理想化下省去终端 Clean 残留。
- **官方可执行配置**：使用固定 `sqrt(sum_scale)=0.1`，与该训练 buffer 不严格匹配；这是官方已有配置差异，仓库未在此给出选择 0.01 的完整理论证明。
- **V6**：沿用同一初始化，没有新增这个差异。
- **审计判断**：登记为“官方训练/推理分布不完全一致及待解释的经验设置”，不能未经证据改判为 V6 Bug；更不能擅自修成 b999 后继续正式训练。

数值重放在 `results/*/NUMERICAL_SCHEDULE.json`，对固定 Val 样本及 128 次 32×32 小块噪声抽样核对 q 的构造与均值/方差；它不作为模型恢复测试。FP32 的 `b` 平方与保存的 b² 最末位可差约 6e−8，报告两种精度口径。

## 4. Shared Distribution 与 Impure Gaussian 到底是什么意思

论文在 `a_T≈1` 时把终点写成 `(1−d_T)y+b_Tε≈0.1y+b_Tε`。固定某张 y 时，这是**非零均值的普通条件高斯**；“Impure Gaussian”不是一种新的概率分布族。

跨样本、跨任务看的是边缘分布

\[
p_T(x)=\int\mathcal N\bigl(x;(1-d_T)y,b_T^2I\bigr)p(y)dy,
\]

有限 a 时还应对 `(c,y)` 的联合分布积分。这通常是图像分布与高斯的卷积/混合，而非单一标准高斯。SDT 压小退化域的均值偏移，使不同任务的状态分布更接近；**它没有严格证明所有任务的终点分布完全相同**。图像信息仍残留，Concat 通道又一直保留完整 y，网络仍可选择性利用条件。 换言之，所讨论的共享主要是状态 x 的边缘分布；网络实际接收的联合输入 `[x,y]` 从未失去完整 y 的信息，不能把两种分布混称为“完全共享”。

在 V6 单任务湍流里，没有正在联合训练雨/雪/雾/低光/模糊五类任务，因此原论文的跨任务共享收益不能自动转移成我们的实验证据。SDT 当前的实际身份是继承的状态路径和初始化定义；删除它会改变预训练网络接收的分布，未经重训的下降不能证明 SDT 的因果必要性。

## 5. 两条 Condition 路径

| 路径 | 表达 | 代码及含义 |
|---|---|---|
| 显式 | `c+a(y−c)−dy+bε`；逆过程中的 `+Δd·y`、最终 `y−r̂` | y 直接进入扩散公式，即便没有 Concat 也在状态路径中存在 |
| 网络 Concat | `[x_τ,y]`，6通道 | `p_losses:1099` / `model_predictions:839`；网络直接看到完整退化图 |

“Condition 很强”是官方设计，不是 V6 偷加旁路；但强条件设计**不保证训练后一定有效利用状态**，也不禁止网络学到很强的条件回归成分。后者只能测效应量，并用正式控制训练进一步判断。 固定 C、τ、ε 改变 Y 时，合法 Forward 的 x 也随之改变；其局部总响应含 `J_Y+(a_τ−d_τ)J_x`。因此单独 Concat 通道偏导、同Clean另一真实realization的联合响应，是不同诊断，报告不可互换。

A 中这里的 Y 是真实 `Y_g`；B 第 i 次这里的 Y 是当前 `Z_i`。原始 Y4 只作 B 物理 Loss 锚点，不通过额外网络旁路重复提供。

## 6. Large U-Net 的实际结构与时间进入方式

官方 `64-1248` 指 `dim=64, dim_mults=(1,2,4,8)`；参数总数 **36,261,187**，1个 `UnetRes.unet0`，不是两个噪声/残差网络。

| 部分 | 空间分辨率 | 本实现的实际通道流 |
|---|---|---|
| 输入/Stem | 256² | Concat6 → 7×7 Conv →64；保存长跳连 r |
| Down0 | 256²→128² | 两个64→64 ResnetBlock + LinearAttention；4×4 stride2 Conv64→64 |
| Down1 | 128²→64² | 两个64→64 Block + LinearAttention；stride2 Conv64→128 |
| Down2 | 64²→32² | 两个128→128 Block + LinearAttention；stride2 Conv128→256 |
| Down3 | 32² | 两个256→256 Block + LinearAttention；3×3 Conv256→512，无再下采样 |
| Middle | 32² | Block512→512 + 完整 Attention + Block512→512 |
| Up0 | 32²→64² | 两次拼接256通道 skip；各768→512 Block；LinearAttention；nearest×2+Conv512→256 |
| Up1 | 64²→128² | 两次拼接128 skip；各384→256；Attention；上采样256→128 |
| Up2 | 128²→256² | 两次拼接64 skip；各192→128；Attention；上采样128→64 |
| Up3 | 256² | 两次拼接64 skip；各128→64；Attention；3×3 Conv64→64 |
| 输出 | 256² | 与 Stem64 拼接→128；最终Block128→64；1×1 Conv64→3 raw residual |

共有19个 ResnetBlock、8个 LinearAttention、1个完整 Attention。每个 ResnetBlock 含两个 `WeightStandardizedConv2d→GroupNorm(8组)→SiLU`，另有 identity/1×1 shortcut；第一个 GroupNorm 后、SiLU 前施加时间 scale/shift：`h·(1+scale)+shift`。注意 Down0 的通道没有立刻变128；不能把架构简称直接画成每次降采样都翻倍的另一种 U-Net。

Attention 为4头，每头32维；LinearAttention 对 Q 的通道与 K 的空间分别 softmax，Middle 是真正空间 token 二次注意力。网络没有额外 VQ 编码器。空间大小按16的倍数做 reflect padding，最后裁回原 H/W；256无需补边。

### 6.1 原生扩散时间不是直接送入整数 τ

`p_losses/model_predictions` 构造 `[1000a_τ,1000b_τ]`，`UnetRes.forward` 在 `objective='pred_res'` 时只选第一个。因此实际

\[
e_{diff}=\operatorname{Linear}_{256\to256}\circ GELU\circ
\operatorname{Linear}_{64\to256}\circ SinEmbed_{64}(1000a_τ).
\]

每个 Block 再经 `SiLU→Linear256→2C` 产生 scale/shift。`1000a_τ` 是 τ 的单调非线性重参数化；不是 V6 物理 t，也不是直接喂入 `τ=999`。

### 6.2 V6 物理时间如何加入

原生 DiffUIR 没有这项。V6 A/B 新增

\[
e_{phys}=t\cdot W_2\,SiLU(W_1SinEmbed_{64}(1000t)),
\quad e_{total}=e_{diff}+e_{phys}.
\]

W1为64→256，W2为256→256并零初始化。`t=0` 时增量始终为0。两种时间的**来源、随机协议、输入分支分离**，编码后相加并共同调制特征，不能说特征层面“互不作用”。同次 S3 的 t 固定，τ按各步变化。

ξ Head 则只读 `(当前观测,预测Clean,t)`，输入6通道，自己使用已批准的物理时间 MLP64→128→128及各Block scale/shift；没有直接接收 τ。它可通过 Clean 间接受内部采样影响。

## 7. 原生训练、V6 训练与采样必须分开

### 7.1 官方训练到底监督什么

`forward:1151` 随机均匀抽 τ；`p_losses` 归一化 C/Y、计算 `r=y−c`、抽 Gaussian ε、调用 q。网络预测 raw residual，监督

\[
L_{official}=E_{τ,ε,c,y}\operatorname{MAE}(\hat r_θ(x_τ,y,1000a_τ),r).
\]

这条训练没有对网络 raw residual 做采样 clamp。论文从反向高斯 KL 推到均值误差，再作简化残差 L1；**L1及删去随时间变化的系数是简化训练目标，不应说与原始 ELBO 数值严格相同**。

按完整噪声反推式，自行推导有 `ε̂−ε=(1−a_τ)(r̂−r)/b_τ`；代入反向均值可得 `μ_θ−μ=−[α_τ+β_τ²(1−a_τ)/b_τ²](r̂−r)`。这里 α/β 是残差扩散每步增量，不是原 VP 的 β。高斯 KL 还带后验方差的权重。因此残差确实可作反向均值的参数化；但扔掉随 τ 变化的权重并改用 L1，是训练 surrogate 的选择。这里使用完整公式自行推导，不能把附录 C.2 的简写逐项当作数值等式。

V6 的 `L_res=0.5L_official` 仅把归一化残差误差换回 RGB 单位，目标仍是 r，不是 ξ、不是真值 Gaussian ε。

### 7.2 逆采样中残差、噪声和 Clean 估计

对当前 x/y/τ，官方 `model_predictions`：

\[
\hat r=clamp(f_θ([x,y],1000a_τ),-1,1),\qquad
\hat c=clamp(y-\hat r,-1,1),
\]
\[
\boxed{\hat ε=\frac{x-(1-d_τ)y-(a_τ-1)\hat r}{b_τ}.}
\]

噪声由残差和当前状态**代数反推**，不是第二个网络预测。论文附录 C.2 式（15/16）右侧化简式省略了共同的 `x−y` 项；逐项计算 ε 必须保留它。该共同项在 `ε̂−ε` 中可抵消，故不要据此擅改原生 residual loss；源码上述函数保留了完整项。

### 7.3 重要：官方实际采用的是论文自己的简化 Reverse

固定两时刻 τ>s，若严格沿同一 ε 的 q 重参数化，代数关系应为

\[
x_s=x_τ-(a_τ-a_s)\hat r+(d_τ-d_s)y-(b_τ-b_s)\hat ε.
\]

官方补充材料 C.1 式（13）也写出最后这个噪声差分项；随后以差分很小为理由省去，得到式（14），对应正文式（8）。当前代码 `type='use_pred_noise'` 实际执行：

\[
\boxed{x_s=x_τ-(a_τ-a_s)\hat r+(d_τ-d_s)y,\quad\eta=0.}
\]

**不能只凭分支名 `use_pred_noise` 就声称更新使用了 ε̂**：当前分支计算 ε̂，却没有将它用于这条 η=0 更新。它也不是未经近似的通用标准 DDIM 公式。3步网格上 `b999−b665≈0.005562`、`b665−b332≈0.170273`，第二个差分并非数值机器精度附近的零；所省项对实际轨迹的量级另在诊断报告计算。这是论文公开采用的近似，不能归罪于 V6 丢项，也不能把论文的“可忽略”直接当作当前湍流任务已验证的事实。

最后一个目标索引为 −1 时，官方不继续上述近似积分，而是直接返回 `x=ĉ`。其反归一化 `(ĉ+1)/2` 才是最终 RGB 恢复。

### 7.4 η=0 并不消灭初始随机性

三步开始时仍抽 `ε_s~N(0,I)`。η=0只让**中间步骤额外噪声**系数为0。给定同一 y、权重与初始 ε，后续确定；更换 ε 仍可能更换输出。输出变异很小可以是条件强、L1恢复偏好稳定解、采样器收缩等现象，不能直接推出算法不存在或后验生成能力已证明。[DDIM 原论文](https://arxiv.org/abs/2010.02502)。

### 7.5 原生三步和合法其他 NFE

代码用 `linspace(-1,999,NFE+1).int()` 后倒序，得到：

| 实际去噪器调用数 | 完整网格（−1为最终Clean输出，不再调用网络） |
|---:|---|
| 1 | 999→−1 |
| 2 | 999→499→−1 |
| 3 | 999→665→332→−1 |
| 4 | 999→749→499→249→−1 |
| 6 | 999→832→665→499→332→165→−1 |

合法1NFE从正确初始化直接预测并返回 Clean；在同噪声下，其结果等于 S3 第一调用的 Clean 估计，这是该采样器的数学性质，不是把 S3 的中间扩散状态提前当作最终结果。比较中每个 NFE 都重新执行自身完整网格。

## 8. V6 到底新增了什么

| 阶段 | 本次审计确认的真实目标 | 去噪器调用 |
|---|---|---:|
| 官方 | 原生随机τ residual L1 | 训练1；默认推理3 |
| V6 D0 | `L_res+L_C(S3(Y4;ε_s),C)`，两个独立随机流，共享当前网络 | 训练4，推理3 |
| V6 A | `L_res+0.9L_C+0.1L_P(g)`；新增物理时间/ξHead | 训练4；四步验证12 |
| V6 B | `0.9 mean4 L_C+0.1 mean4 L_P(G4)+0.1 L_res_TF`；四阶段独立采样噪声，完整物理链 | 训练13；四步验证12 |

D0 的 Clean Loss 来自**从 Y 初始化的完整采样**，不读 GT q_state 来代替推理。反向时仅开放官方 `sample/ddim_sample` 的两个 `no_grad` 装饰器，原运算与真实 clamp 导数保留；无 STE。两条 Loss 对同一276个参数张量累计梯度，随后正式训练代码只做一次 optimizer update。本轮诊断则完全不做 update。

B 的 TF 状态来自真实 `(C,ξGT,gTF)`，在四个阶段输入时间节点等概率选一个；不是把生成 Z_i 当 GT。外层三个非零状态转移保持 Clean/ξ/Simulator 梯度。记录日志的 `.detach()` 不等于训练分支 detach；实际 Loss 的 backward 在记录前已执行。现有全图/终端单独 Loss 验收另有原始证据，本次不重跑占用五卡的 B 训练。

### 8.1 schedule 生命周期

V6 保存两个独立 diffusion schedule 对象：`native` 继承官方 checkpoint EMA buffers；`sampling` 在 CPU 加载后执行一次官方 `init()`，再让两者 `.model` 指向**同一网络**。因此采样重建 buffer 不会覆盖随机τ训练对象。

`__init__` 与 `init()` 的 `alphas[0]/betas2[0]` 赋值确有差别；native/sampling 还有 FP32累计系数末位差异。 官方还在734/800行强制`one_minus_alphas_cumsum[-1]=1e−6`，它不等于实测的`1−a999≈0.006353`；当前pred_res→Y−r̂路线不读取这个buffer，不能误用它计算本轮q终端均值，也不能外推未选pred_noise路线的正确性。活跃 q 和 S3 按各自官方生命周期使用累计表；未把本轮诊断的 NFE设置或 `init()` 写回正式对象。未走的 DDPM/pred_noise 分支不能用来替代当前 `pred_res/S3` 的结论。

## 9. 官方逐项源码映射

以下行号固定于指定 commit；可查 `sources/OFFICIAL_SYMBOL_LINES.json`、完整只读快照及 `SOURCE_IDENTITY.json`。官方在线源码：[固定版本 model.py](https://github.com/iSEE-Laboratory/DiffUIR/blob/c71437ce86ad1b098aa9be4678dc82e8b99649df/src/model.py)。

| 内容 | 函数/类与行号 | 审计重点 |
|---|---|---|
| Sinusoidal embedding | `SinusoidalPosEmb` 224–236 | 64维sin/cos |
| 时间调制 | `Block`260–276、`ResnetBlock`279–304 | GN后scale/shift；第二Block不再调制 |
| 注意力 | `LinearAttention`307–337、`Attention`340–363 | 8个线性/1个完整 |
| 完整网络 | `Unet.__init__`367–455、`forward`464–503 | stem/双skip/时间MLP/输出 |
| 残差网络包装 | `UnetRes`506–552 | pred_res选time[0]；仅unet0 |
| schedule构造 | `ResidualDiffusion.__init__`627–734 | VP→a/b；SDT；buffer注册 |
| 推理生命周期 | `init`736–800 | 重建CPU buffer；不能覆盖训练对象 |
| 从残差推噪声 | `predict_noise_from_res`802–806 | 完整状态/条件/SDT项 |
| residual/clean/noise输出 | `model_predictions`835–887 | concat6、aτ×1000、两次真实clamp |
| 当前推理 | `ddim_sample`946–1046 | 961初始噪声；953网格；984终点；1012更新 |
| 对外sample | `sample`1048–1060 | `[0,1]→[-1,1]`及当前DDIM选择 |
| Forward | `q_sample`1062–1069 | `c+a r+bε−dy` |
| 原生监督 | `p_losses`1080–1143 | 1088残差GT；1093真实q；1123pred_res；1140L1 |
| 随机时刻 | `forward`1145–1153 | 1151 randint(0,1000) |

V6 本地关键实现：

- `method_impl/d0.py:30–51`：严格EMA加载、双schedule/共享模型；56–62随机τ及0.5尺度；64–80可微S3及联合目标。
- Stage1 `scripts/common.py:49–58`：Large与0.0018/0.01/3步配置；`grad_helpers.sampling_gradients_enabled`：仅开放两个装饰器。
- `method_impl/ab.py:17–29`：独立物理时间；`115–125`：同一Clean恢复器；`142–172`：A/B Loss与Full-BPTT；`175–188`：物理节点和TF选择。
- `method_impl/continuation_factory.py`：承接已完成D0续训/A best的加载许可，不改变数学训练函数。
- `scripts/run_ab_five_npu.py:168` / `scripts/run_d0.py:162`：实际联合Loss backward；日志detach在其后。

## 10. 理论审计能够与不能够保证的事情

已确认：V6 保留的是官方**带SDT的残差条件扩散训练及其简化逆采样实现**；没有把“新增三步监督”伪装成原论文方法。新增监督可能改变网络选择使用各输入的方式，是否实际弱化要看行为数据。

未得到保证：官方训练/推理终端严格一致、简化Reverse精确保持q边缘、单任务SDT收益、真实后验样本质量、比同计算量直接恢复器更强，以及新增Clean Loss的因果优劣。它们不能由“有Gaussian Noise”“梯度非零”“3步PSNR更高”任一单项自动推出。

主要原始来源：[DiffUIR 正文](https://arxiv.org/abs/2403.11157)、[CVPR 补充材料 C.1/C.2](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zheng_Selective_Hourglass_Mapping_CVPR_2024_supplemental.pdf)、上述固定源码、用户 Method Spec §3–9及Q1原件。下载原件、渲染核对图与SHA均保存在 `sources/`。

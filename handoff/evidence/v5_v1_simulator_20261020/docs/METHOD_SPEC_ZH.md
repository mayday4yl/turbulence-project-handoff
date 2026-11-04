# V5-v1 方法规格

日期：2026-10-20；起始强度、时间接口及 Concertormer 实验决定按 2026-10-21 用户追加指令更新（北京时间）。

**状态：当前 Simulator 已按保留局限的研究候选接收，不再阻塞 V5 下一阶段。** 见 [2026-10-21 后续接收决定](<../acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md>)。[原冻结 Gate](<../../../NOT_INCLUDED.md#ref-af70d59e92e0ef04a83c>) 未全通过的记录保持不变。用户随后已授权新数据与首轮训练，执行入口为 [Concertormer 10k 实验](<../../../NOT_INCLUDED.md#ref-1dfa96e23a02618db626>)；启动/完成状态以该实验日志为准，不等于恢复效果已验证。

依据为用户提供的 [V5-v1 交接文档](<../../../NOT_INCLUDED.md#ref-353124d6680a073ac0ef>)、2026-10-20 对推理路径 / refinement / 验收程序的三项补充及随后指定的 de36 NPU 执行要求，以及 [V4 实际接口审计](<../../../NOT_INCLUDED.md#ref-586f7a9c1da666f16208>)。用户最新补充优先于交接文档中尚未确定的内容。

## 1. 版本关系与执行范围

V5 是 V4 Noise Bridge 中 R0 物理递归方案的演进。它继承 clean/raw-ξ 双输出、预测状态递归、Full-BPTT 训练和观测重退化一致性；修改输入接口、时间路径、Simulator 零端点、损失权重与推理 refinement 执行范围。完整对照见 [V4→V5 变更表](<../../../NOT_INCLUDED.md#ref-9a8915ed1e3bf4bd56a1>)。

初始范围为方法整理、核对 V4 源码与资产、实现 Smooth / Tilt-Warp / FFT 并进行局部检查。2026-10-21 用户明确保留候选，暂定采用旧 P2S 完整复数 IFFT，保留 raw-ξ 接口、采样分布与物理参数；接受将非平稳协方差记录为局限，授权按已提出阈值执行最终 Gate。当前 Gate 已完成但有必检失败，未修改阈值或整体物理模型。Refinement 超参数未定不阻塞这些 Simulator 检查。

**当前目标设备为 de36 的 NPU。** 用户已授权使用 de36 实时空闲的卡，最多同时 4 张；本任务不沿用历史 NPU 0–3 编号限制。使用前核实主机、实时占用及进程归属、运行环境与算子能力，再在可用资源上进行真实 NPU 检查。CPU 结果只作为参考，不能代替 NPU 兼容性或数值验收。历史 A100-only 的任务边界不覆盖本次用户明确指定的 de36 工作；不得终止他人作业。某次探针使用 NPU 1 仅是运行快照，不是之后的固定占用或专属分配。

2026-10-21 后续范围已扩展为 Concertormer full 官方 GoPro 初始化、完整训练链、新 V5 数据与 D0/R0/R0-D0Init 首轮实验。必要自动检查通过后直接训练，每组最多 10k updates，无须再次启动审批。详细预算和验证协议见 [用户决定](<../../../NOT_INCLUDED.md#ref-1a15f3d2002f13501572>)。V4 历史资产保留原状；此前数值检查样例不冒充正式数据。

## 2. 符号与网络接口

| 符号 | 定义 |
|---|---|
| \(C\) | 训练使用的真实 clean RGB |
| \(G_{\max}\) | Simulator 最大强度，固定为 4，亦为全局时间和 Smooth 的归一化分母 |
| \(G\) | 路径起始强度；本版实际训练固定为 4，可配置采样接口不等于已启用随机 G |
| \(Y_4\) | 原始最大强度观测；未来应由验收后的 \(F_{\mathrm{V5}}(C,\xi,4)\) 生成 |
| \(Y_G\) | 一般起点的原始观测，必须实际对应 \(F_{\mathrm{V5}}(C,\xi,G)\)；不能把 Y4 重标为别的强度 |
| \(Z_i\) | 第 \(i\) 次预测前的当前 RGB 状态，\(Z_0=Y_G\)；当前即 Y4 |
| \(t_i\) | 全局恢复时间，\(t_i=1-g_i/4\)，起点 \(t_0=1-G/4\) |
| \(g_i\) | 退化强度，\(g_i=4(1-t_i)\) |
| \(\widehat C_i,\widehat\xi_i\) | 本次网络预测的 clean 与 Simulator 原生 raw realization |
| \(\xi_i^*\) | 推理时围绕本次预测 \(\widehat\xi_i\) 优化得到的 raw realization |
| \(F_{\mathrm{V5}}\) | V5 Simulator；内部保持 blur→tilt warp 顺序 |

网络接口已经明确为：

\[
(\widehat C_i,\widehat\xi_i)=N_\theta(Z_i,t_i).
\]

图像输入只有当前 RGB 三通道；连续 t 通过独立时间条件分支接入，不增加图像通道。网络不另接收 g、G、原始观测或 NFE。g 由外层按全局时间计算并传给 Simulator，原始观测由外层保存用于重退化损失和 observation matching。2026-10-21 本接口明确取代原交接中 `N(Z,g,t)`；原交接文件作为历史来源不改写。V4 中 t 的 RGB 噪声混合语义不再沿用。

Raw ξ 保持 Zernike/Tilt 两块原生输入的接口约定，不直接表示 PSF、Zernike 最终场或位移场。V4 的 `rf_range=512` 接口为：

```text
epsilon_zernike : [B,36,512,512]
epsilon_tilt   : [B,512,512,2]
```

最新 clean 参数化采用 Concertormer 官方 RGB 残差恢复形式 `clean = Z + restored_residual`，复用官方最终 RGB 输出层，替代旧 V4 sigmoid 绝对 clean head。只对最终 clean 输出使用 V5 loss，不加入官方多尺度辅助损失；原辅助输出层加载保留但不参与主损失。新增连续时间条件与原生 raw-ξ head。Simulator、raw-ξ 维度、采样分布和统计含义不变。

## 3. 起点采样、四步训练和通用推理调度

本版启用 `fixed(G=4)`，训练 NFE 固定 4，调用编号 i=0,1,2,3。独立起点采样接口支持 `fixed` 和显式指定边界的 `uniform`，合法起点为 \(0<G\le4\)。uniform 仅供后续接入，不是当前生效的训练策略；G=0 已为 clean 终点，无法满足三个内部点的训练要求。

训练在 \((t_0,1)\) 独立均匀采样三点并排序：

\[
t_0=1-G/4<t_1<t_2<t_3<t_4=1,\qquad g_i=4(1-t_i).
\]

当前 G=4 与原定义完全一致。新入口见 [method_interface_20261021](<../../../NOT_INCLUDED.md#ref-9f8205cc245cc0a32a54>) 及其默认配置；原冻结 `time_paths.py` 保留，用作历史 Gate 和默认路径回归。采样使用 CPU float64 seed RNG，在最终 dtype 下检查 t/g 严格单调及内部点。端点、重复或舍入退化使该组三点整体重采样，合法行保留；不做 clamp、jitter 或间距修补。默认最多额外重试64次，区间在目标精度下过窄或重试耗尽会明确报错。seed、batch size、dtype、G 和重试上限固定时可重放。未来启用 uniform 必须先生成对应 YG，并校验观测强度元数据；元数据检查本身不生成或证明观测像素正确。

训练中前三次预测之后执行：

\[
Z_{i+1}=F_{\mathrm{V5}}(\widehat C_i,\widehat\xi_i,g_{i+1}),
\qquad i=0,1,2.
\]

下一状态由预测 clean 和预测 raw ξ 生成。真实 \(C\) 仅用于 clean 监督，不替换 \(\widehat C_i\) 构建中间状态；也不使用预存的真实较弱退化图替代预测状态。整个预测状态链保留 Full-BPTT，不对中间状态、预测 clean 或预测 ξ 做截断。

推理调度支持任意正整数 N、有限正 gamma：

\[
g_i=G\left[1-(i/N)^\gamma\right],\quad
t_i=1-g_i/4,\qquad i=0,\ldots,N.
\]

默认 G=4、N=4、gamma=1，得到：

```text
t = [0, 0.25, 0.5, 0.75, 1]
g = [4, 3,    2,   1,    0]
```

也可传入从 t0 到 1 的严格递增非均匀全局时间网格。网格长度决定 N（显式 N 必须与之相符）；检查有限性、端点和最终 dtype 下的严格单调，不静默排序或修补非法网格。改 G 时全局时间仍除以4，不把 t0 强行重置为0；Simulator Smooth 分母也保持4。

推理前 N−1 次预测之后使用 refine 后的 ξ：

\[
Z_{i+1}=F_{\mathrm{V5}}(\widehat C_i,\xi_i^*,g_{i+1}),
\qquad i=0,\ldots,N-2.
\]

最后一次预测发生在 \(t_{N-1}\)。\(t_N=1,g_N=0\) 是物理终点，不是第 N+1 次网络调用。推理同时记录末次直接 clean 与其 refine 后的零强度渲染。默认 N=4 时就是原第四步行为。

NFE 仅计网络前向次数；refinement 内的 Simulator 调用及零端点检查不计作额外网络预测。N/gamma/网格属于外层调度，不改变共享网络参数结构；支持可变 N 不代表已验证变步数的恢复质量。

## 4. 训练损失与推理 refinement

训练对四阶段平均：

\[
\mathcal L
=\frac14\sum_{i=0}^{3}
\left[
0.9\,\operatorname{MAE}(\widehat C_i,C)
+0.1\,\operatorname{MAE}
 \left(F_{\mathrm{V5}}(\widehat C_i,\widehat\xi_i,4),Y_4\right)
\right].
\]

MAE 对 RGB 全像素取平均，随后对阶段和 batch 平均。所有阶段的重退化都使用强度 4，并对齐最初观测 \(Y_4\)，不对齐当前递归状态 \(Z_i\)。训练 refinement=OFF。

推理 refinement=ON，默认四次预测均执行；若以后改变推理 N，仍每次预测后执行，包含末次。每次固定预测 clean，从本步预测 raw ξ 出发优化。下面目标保持当前 G=4 的已定形式：

\[
J_i(\xi)=
\operatorname{MSE}\left(
F_{\mathrm{V5}}(\operatorname{stopgrad}(\widehat C_i),\xi,4),Y_4
\right)+\lambda_\xi R(\xi,\widehat\xi_i),
\]

其中 \(R\) 为相对本步预测 raw ξ 的 proximity regularization，不是对标准高斯先验的正则。V4 已核对实现可作为兼容参考：

\[
R_{\mathrm{V4,ref}}(\xi,\widehat\xi_i)=
\frac{\|\xi_z-\widehat\xi_{z,i}\|_2^2+
      \|\xi_t-\widehat\xi_{t,i}\|_2^2}
     {d_z+d_t}.
\]

V4 参考形式将两块平方差求和后除以 raw ξ 总元素数，并非两个块各自取均值后相加。这里记录旧实现供后续保持可比性，**不在本轮替用户冻结 V5 的分块或精确归约设计**，也不自动沿用 V4 数值超参数。

本轮不冻结 \(\lambda_\xi\)、优化步数、学习率、精确归约及其他推理优化器配置，留到后续推理实现和训练方案讨论。此处只规定 MSE observation matching + raw-ξ proximity 的目标类型，以及四次开启的行为。

当前 Loss 与 refinement 目标均不改写为随机 G 版本。未来真正启用非4起点时，必须让原始观测及重退化匹配强度一致，并在调用方接入 YG 后再开启；本轮只提供起点 / 时间 / 元数据接口，不将非4调度支持冒充已完成的一般 G 训练或推理链。

若零端点正确，则：

\[
F_{\mathrm{V5}}(\widehat C_3,\xi_3^*,0)=\widehat C_3.
\]

因此末次 refinement 只能影响该 clean 在 \(g=4\) 下的观测一致性，不能被报告为最终 clean 恢复收益。等式是否在要求精度内成立由 Simulator 检查决定，不能仅由流程文字假定通过。

## 5. Simulator 候选及科学边界

保持 PiRN/P2S 的主体生成链路及 blur→tilt warp 顺序。除必要修复外，不另行引入替代 Simulator。

| 项目 | 本轮实现 / 检查要求 | 尚不能直接推出的结论 |
|---|---|---|
| S1 PSF Smooth | 候选 \(\widetilde h_g=(1-g/4)\delta+(g/4)h_g\)；保持空间变化核、归一化和边界语义，检查 \(g=4\)、近零及零端点 | 候选连续性不证明其真实大气物理有效性 |
| S2 Tilt / Warp | 核对 \(g^{5/6}\) 缩放；使 pixel-center 网格与 `align_corners=False` 一致；检查零位移、正负非零位移、边界和输入梯度 | 单个零位移样例通过不等于完整 Simulator 通过 |
| S3 Tilt FFT | 明确频谱表示、Hermitian 关系、变换维度与尺度、固定硬阈值、有效梯度方向；记录统计变化 | 压缩维从 512 改为 257 本身不证明物理和梯度修复完成 |

S1 的线性混合发生在 PSF / blur 阶段；不能把整个最终 warp 输出直接与 clean 混合。完整的 \(g=4\) 新旧输出无需相等，因为 Tilt/Warp/FFT 修复可能改变它；仍须报告差异及来源。

S3 若保留可比较的候选实现，应分别命名并记录选择状态。硬阈值和结构性零空间必须解释；不是要求全部原始 ξ 维度在所有图像、所有强度下都存在非零梯度。FFT 完成后重测 S2，并重新运行最终组合的 S1–S3 局部检查。

## 6. 局部数值检查与最终 Gate

初始局部检查及建议阈值保留为历史记录。2026-10-21 按用户授权，运行前将已提出阈值和独立样例冻结于 `gate_20261021/FROZEN_PLAN.json`。最终 Gate 仍有近零绝对上限与局部 Zernike 有限差分失败，详见最新报告；补充诊断不能将原始 FAIL 改写为 PASS。

| 检查 | 必须保存的证据 |
|---|---|
| 固定旧版回放 | V4 源码 / P2S 资产哈希、环境、dtype、固定矩阵身份、\(C,\xi\) 或 seed 及构造方式 |
| \(g=0\) identity | 不同图像与真实采样 / 预测风格 ξ 下的 `max_abs`、MAE / RMS；全部输出有限 |
| \(g=0\) 的 ξ 导数 | ξ 梯度为零是预期行为；记录实际数值，不能把零梯度本身判成缺陷 |
| \(g=0\) 的 C 导数 | 检查 identity Jacobian：对方向 \(v\)，\(J_Cv=v\)；或对上游 \(u\)，\(J_C^\top u=u\)，保存误差 |
| \(g\to0^+\) 连续性 | 固定 \(C,\xi\)，报告多个近零强度下 \(\|F(C,\xi,g)-C\|\) 曲线和数值地板；有限扫描提供数值证据，不等同数学证明 |
| 有效方向有限差分 | 对 C 和有效 ξ 方向比较 autodiff 与中央差分；记录方向构造、归一化、多个步长及相对 / 绝对误差 |
| Tilt/Warp | 位移尺度、方向、边界、零位移、分段可导位置及输入梯度；避开或单独标明插值折点 |
| FFT | 频谱形状 / 共轭对称性、归一化、位移均值 / 方差 / RMS / 极值、频段或方向响应、有效与无效方向 |
| 非整数强度与端点 | 多个浮点 \(g\in[0,4]\)，输出 / 梯度有限性及相对旧 \(g=4\) 的差异 |
| de36 NPU 环境与算子 | 主机身份、实际空闲资源、PyTorch / torch_npu / CANN 版本、FFT 与采样等关键算子的真实前向 / 反向 / 有限差分结果、CPU reference 对照 |

阈值必须绑定测试分辨率、dtype、设备、输入范围、seed 和方向选取规则，兼顾接近零的绝对误差与非零导数的相对误差。不得先把观察到的最大误差包装成已获确认的标准。CPU reference 与真实 de36 NPU 结果分别报告；若算子发生 CPU 回退，须记录回退边界、梯度连通性与时间，不能将其写为纯 NPU 算子通过。

## 7. 后续数据与训练登记

正式数据使用已研究接收的 \(F_{\mathrm{V5}}\) 重新生成全部 \(Y_4\)，保留原 Gate 未全通过的局限。旧 V4 缓存 \(Y_4\) 不能作为 V5 训练观测；可复用 clean 资源及与新划分兼容的身份信息，但须重新核实 source-disjoint 划分和 realization 兼容性。

已定数据协议为 clean source 层 Train:Val:Test=8:1:1，每张 clean 独立采样 4 个 ξ。旧 split 只有在同时满足这个比例和 source-disjoint 要求时才可直接复用。若 FFT 修改 raw ξ 的采样 / 统计语义，先重新确认其分布。训练在线采样内部时间，无需预生成连续强度的真实中间状态。

已定 clean 总量 2000/250/250，每张 4 个 ξ。D0 与 R0 从同一完整适配 GoPro U0 分叉，R0-D0Init 从 D0 完整 Val 最优 checkpoint 启动，额外累计整个 D0 训练与选择成本。每组最多 10k optimizer updates；AdamW 原主体/原 RGB LR5e-5，新增时间/raw LR1e-4，warmup300 后 cosine，wd1e-4，grad clip1，EMA OFF。effective batch 目标32，具体 microbatch/累积/卡数/checkpointing 由真实吞吐与显存决定。

每500 updates完整Val1000，refinement OFF；R0均匀四步，D0保持单步基线。每1000更新保存完整可续训checkpoint，额外best/latest；测试集在最终选择前不用于恢复评价。训练结束后进行各组best的完整refinement ON评价，数值超参数须另行明确记录，不因未冻结而阻塞训练。执行授权来自用户后续明确指令，不能反推为原Simulator Gate全通过。

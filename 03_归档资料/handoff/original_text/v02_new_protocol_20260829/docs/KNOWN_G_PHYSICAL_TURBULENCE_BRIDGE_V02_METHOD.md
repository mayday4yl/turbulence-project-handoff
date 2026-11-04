# Known-\(g\) Physical Turbulence Bridge V0.2
## Trajectory-Consistent Epsilon Refinement

**状态：算法方案冻结版；执行歧义已按 V0.2 协议闭合。**

本文是当前新 V0.2 的唯一方法定义。V0.1 仅作为数据、源码、checkpoint、结果和固定 protocol 的基线资产；任何此前停止的旧 V0.2 协议均不得恢复或补入本文。

本版本是对 V0.1 的定向修复，不改变老师认可的两个核心：

1. 沿 PiRN/P2S 构造的物理 severity bridge 逐步恢复；
2. 网络预测 turbulence realization \(\xi\) 后，利用原始观测对 \(\xi\) 做测试时物理优化。

V0.2 只修复 V0.1 已暴露的两个错位：

- 训练只约束终端退化重建，没有约束 \(\xi\) 的跨 \(g\) 传输能力；
- 推理优化只追求重建原始强湍流观测，可能破坏下一桥状态。

---

## 1. 研究范围

当前阶段严格研究 **known-\(g\)** 单帧湍流恢复：

\[
\boxed{(Y,g_Y)\rightarrow \hat C}
\]

其中：

- \(Y\)：单张湍流退化图像；
- \(g_Y\)：已知的初始湍流强度；
- \(\hat C\)：最终恢复的 clean 图像。

当前不让恢复网络预测 \(g\)。后续 unknown-\(g\) 版本可单独增加 severity estimator：

\[
Q_\phi(Y)=\hat g_Y,
\]

再将 \(\hat g_Y\) 接入已验证的 known-\(g\) 框架。

---

## 2. 数据与物理轨迹

第 \(i\) 条 trajectory 固定：

\[
C_i,\qquad
\xi_i=(\varepsilon_{\mathrm{Zernike},i},\varepsilon_{\mathrm{tilt},i}),\qquad
G_i\in\{1.50,2.25,3.00,3.75,4.50\},
\]

其中 \(G_i\) 是数据 manifest 中该 trajectory 的固定 terminal-severity metadata，不得在训练过程中给同一 trajectory 重新分配。物理状态只随 turbulence strength \(g\) 改变：

\[
\boxed{X_{i,g}=F(C_i,\xi_i,g)}.
\]

severity grid 保持 V0.1：

\[
\boxed{\mathcal G=\{0.75,1.50,2.25,3.00,3.75,4.50\}}.
\]

每条 trajectory 保存或可由冻结 manifest 确定性重放：

\[
\{C_i,\xi_i,X_{i,0.75},X_{i,1.50},X_{i,2.25},X_{i,3.00},X_{i,3.75},X_{i,4.50}\}.
\]

当前数据可直接复用。训练只使用满足 \(g\le G_i\) 的状态；所有新增监督均在同一条 trajectory 内使用，不依赖同一 Places source 的不同 augmentation 版本之间形成对照。若后续只读 identity audit 发现既有数据不满足固定 \((C_i,\xi_i,G_i)\) 或无法取得所需同轨迹状态，则报告 mismatch 并停止，不得静默重分配 \(G_i\) 或重生成 split。

---

## 3. 网络接口

沿用 V0.1 的 known-\(g\) 双分支网络：

\[
\boxed{
S_\theta(X_{i,g},g)
=
(\hat C_{i,g},\hat\xi_{i,g})
}
\]

其中：

\[
\hat C_{i,g}=S_{\theta,C}(X_{i,g},g),
\]

\[
\hat\xi_{i,g}=S_{\theta,\xi}(X_{i,g},g).
\]

\(g\) 是网络的显式条件，但 \(\hat\xi\) 的物理语义必须是 **severity-independent canonical realization**：

- 网络可利用已知 \(g\) 反演底层 realization；
- \(g\) 对 Zernike/tilt 幅度和物理成像的作用仍由 simulator \(F\) 完成；
- 不把已经和当前 \(g\) 混合的 blur/warp field 当成 \(\xi\) 输出。

V0.2 保持 V0.1 simulator-native full-resolution \(\xi\) 表示，不静默降维或更换 latent 定义。

---

## 4. 训练样本采样

对同一条 trajectory 先采样当前状态：

\[
\boxed{G_i\ge g\ge g_{\min}}.
\]

当且仅当 \(g>g_{\min}\) 时，下一目标固定为离散 grid 上与 \(g\) 相邻的较弱 severity：

\[
\boxed{g>h\ge g_{\min}}.
\]

含义：

- \(G_i\)：该条 trajectory 的终端强度；
- \(g\)：当前输入状态的已知强度；
- \(h\)：离散 grid 上与 \(g\) 相邻的下一较弱强度。

若 \(g=g_{\min}=0.75\)，不存在合法 \(h\)。该样本仍训练 clean 与 trajectory realization，但不定义或计算 bridge execution loss。

网络前向：

\[
(\hat C_{i,g},\hat\xi_{i,g})
=S_\theta(X_{i,g},g).
\]

---

## 5. 三个核心训练损失

统一图像距离采用与 V0.1 一致的 mean MSE：

\[
\boxed{
d_I(A,B)=\operatorname{mean}\big[(A-B)^2\big]
}
\]

图像范围、边界处理、dtype 和 simulator convention 必须继承 V0.1 实际代码；不得静默改变。

### 5.1 Clean restoration loss

\[
\boxed{
\mathcal L_C
=
d_I(\hat C_{i,g},C_i)
}
\]

作用：维持已经建立的 clean 恢复能力。

---

### 5.2 Trajectory realization loss

定义当前样本可用的去重 severity 集合：

\[
\boxed{
\mathcal S_i=
\begin{cases}
\operatorname{unique}\{G_i,g,h\}, & g>g_{\min},\\
\operatorname{unique}\{G_i,g\}, & g=g_{\min}.
\end{cases}
}
\]

使用真实 clean \(C_i\)，将同一个预测 realization 放到当前可定义的多个 severity 下：

\[
\boxed{
\mathcal L_T
=
\frac{1}{|\mathcal S_i|}
\sum_{s\in\mathcal S_i}
 d_I\!\left(F(C_i,\hat\xi_{i,g},s),X_{i,s}\right)
}
\]

当 \(g>g_{\min}\) 时，它同时要求：

\[
F(C_i,\hat\xi_{i,g},G_i)\approx X_{i,G_i},
\]

\[
F(C_i,\hat\xi_{i,g},g)\approx X_{i,g},
\]

\[
F(C_i,\hat\xi_{i,g},h)\approx X_{i,h}.
\]

V0.1 的 terminal degradation loss 被完整保留为 \(s=G_i\) 的一项；V0.2 只是将其扩展为跨 severity 的 trajectory consistency。

这里使用真实 \(C_i\)，因此 \(\hat\xi\) 不能通过吸收 \(\hat C\) 的误差来降低该损失。

V0.2 不要求：

\[
\hat\xi_{i,g}=\xi_i
\]

的 raw Gaussian tensor 数值完全一致，而要求 \(\hat\xi\) 在 simulator output space 中对整条 trajectory 有效。

---

### 5.3 Bridge execution loss

本项仅对存在合法 \(h<g\) 的样本定义。

按照真实推理路径生成下一状态：

\[
\boxed{
\hat X_{i,h}
=
F(\hat C_{i,g},\hat\xi_{i,g},h)
}
\]

监督：

\[
\boxed{
\mathcal L_B
=
d_I(\hat X_{i,h},X_{i,h})
}
\]

在 batch 中，\(\mathcal L_B\) 必须只对合法 transition 子集单独取 mean；不得把无合法 \(h\) 的样本以零填充后再除以整个 batch。若某个 batch 没有合法 transition，则该 batch 跳过 \(\mathcal L_B\) 项。

作用：直接约束预测 clean 与预测 realization 联合后，能否生成正确的下一张弱湍流图。

三个 loss 的分工为：

\[
\mathcal L_C:
\hat C\text{ 是否正确},
\]

\[
\mathcal L_T:
\hat\xi\text{ 是否跨 }g\text{ 有效},
\]

\[
\mathcal L_B:
(\hat C,\hat\xi)\text{ 联合后能否执行正确 bridge transition}.
\]

---

## 6. 总训练目标

\[
\boxed{
\mathcal L_{\mathrm{V0.2}}
=
\mathcal L_C
+
\lambda_T\mathcal L_T
+
\lambda_B\mathcal L_B
}
\]

训练执行协议允许对每个样本从 \(\mathcal S_i\) 均匀采一个 \(s_T\)，以无偏单-severity estimator \(\widehat{\mathcal L}_T\) 代替一次 forward 中的完整求和；其期望必须等于本节定义的 \(\mathcal L_T\)。Validation、checkpoint selection 和正式报告必须计算完整 \(\mathcal L_T\)，因此该 estimator 是计算实现，不是第四个 loss 或方法修改。

V0.2 方法主体不加入：

- raw-\(\xi\) regression；
- 不同步骤 latent equality；
- global consensus variable；
- cycle/re-identification/fixed-point network loss；
- multi-step rollout loss；
- 额外感知损失。

---

## 7. 推理阶段：预测、优化、物理重渲染

输入：

\[
\boxed{(Y,g_Y)}.
\]

### 7.0 唯一 schedule authority：SA-V02-ADJ-1

请求的 \(K\in\{0,1,3,5\}\) 定义为“请求执行的 physical transition 数”，不表示必须到达 \(g_{\min}\)。所有 transition 只能沿冻结离散 grid 移动到**相邻较弱** severity。

对起始强度 \(G\)，定义：

\[
K_{\max}(G)=\text{从 }G\text{ 沿相邻 grid 到 }0.75\text{ 的 transition 数},
\]

\[
\boxed{K_{\mathrm{eff}}(G,K)=\min\{K,K_{\max}(G)\}}.
\]

达到 \(0.75\) 后立即停止，不重复状态、不增加虚假 transition，也不渲染 \(g=0\)。\(K=0\) 始终为 one-shot，仅执行一次当前状态上的 clean prediction。

下表是训练诊断、validation、comparator 和 frozen test 的唯一 schedule authority：

正式 validation/comparator/test 对第 \(i\) 条 trajectory 均从固定 terminal state \((X_{i,G_i},G_i)\) 开始；训练中的随机当前 \(g\) 不改变评价起点。

| 起始 \(G\) | 请求 \(K=0\) | 请求 \(K=1\) | 请求 \(K=3\) | 请求 \(K=5\) |
|---:|---|---|---|---|
| 0.75（边界输入，非 terminal bucket） | `0.75`（\(K_{\mathrm{eff}}=0\)） | `0.75`（0） | `0.75`（0） | `0.75`（0） |
| 1.50 | `1.50`（0） | `1.50→0.75`（1） | `1.50→0.75`（1） | `1.50→0.75`（1） |
| 2.25 | `2.25`（0） | `2.25→1.50`（1） | `2.25→1.50→0.75`（2） | `2.25→1.50→0.75`（2） |
| 3.00 | `3.00`（0） | `3.00→2.25`（1） | `3.00→2.25→1.50→0.75`（3） | `3.00→2.25→1.50→0.75`（3） |
| 3.75 | `3.75`（0） | `3.75→3.00`（1） | `3.75→3.00→2.25→1.50`（3） | `3.75→3.00→2.25→1.50→0.75`（4） |
| 4.50 | `4.50`（0） | `4.50→3.75`（1） | `4.50→3.75→3.00→2.25`（3） | `4.50→3.75→3.00→2.25→1.50→0.75`（5） |

因此实际 schedule 为：

\[
\boxed{
g^{(0)}=g_Y>g^{(1)}>\cdots>g^{(K_{\mathrm{eff}})}\ge g_{\min}>0,
}
\]

且每一对 \((g^{(k)},g^{(k+1)})\) 都是 grid 上的相邻下降。

初始化：

\[
X_{g^{(0)}}=Y.
\]

对 \(k=0,\ldots,K_{\mathrm{eff}}-1\)，执行以下四个操作。若 \(K_{\mathrm{eff}}=0\)，跳过 7.2--7.4，直接将 7.1 的 clean branch 作为 one-shot 输出。

### 7.1 网络预测

\[
\boxed{
(\hat C_k,\hat\xi_k)
=
S_\theta(X_{g^{(k)}},g^{(k)})
}
\]

### 7.2 生成未经优化的下一状态提议

\[
\boxed{
\tilde X_{g^{(k+1)}}
=
F(\hat C_k,\hat\xi_k,g^{(k+1)})
}
\]

该提议由经过 trajectory training 的网络产生，代表优化前应保护的 next-state physical effect。

### 7.3 Trajectory-preserving epsilon refinement

沿用 V0.1 的核心：以 \(\hat\xi_k\) 初始化，只优化 \(\xi\)，并始终使用原始观测 \(Y\) 与原始强度 \(g_Y\) 作为物理锚点。

观测一致性项：

\[
\boxed{
\mathcal J_{\mathrm{obs},k}(\xi)
=
d_I\!\left(F(\hat C_k,\xi,g_Y),Y\right)
}
\]

下一状态保护项：

\[
\boxed{
\mathcal J_{\mathrm{keep},k}(\xi)
=
d_I\!\left(
F(\hat C_k,\xi,g^{(k+1)}),
\tilde X_{g^{(k+1)}}
\right)
}
\]

优化目标：

\[
\boxed{
\xi_k^*
=
\arg\min_\xi
\left[
\mathcal J_{\mathrm{obs},k}(\xi)
+
\beta\mathcal J_{\mathrm{keep},k}(\xi)
\right]
}
\]

固定：

\[
\theta,\quad \hat C_k,\quad Y,\quad g_Y,\quad g^{(k+1)}.
\]

优化变量只有：

\[
\boxed{\xi}.
\]

解释：

- \(\mathcal J_{\mathrm{obs}}\) 保留 V0.1 的真实观测校正能力，缓解高维 \(\xi\) 难以仅靠网络精确预测的问题；
- \(\mathcal J_{\mathrm{keep}}\) 防止 optimizer 为了拟合终端观测而任意破坏网络已学到的下一状态；
- 不直接使用 \(\|\xi-\hat\xi_k\|^2\)，因为 raw Gaussian latent 可能存在 functionally equivalent 解；保护的是 simulator output space 中的物理作用。

边界：

\[
\beta=0
\]

退化为 V0.1 terminal-only refinement；

\[
\beta\rightarrow\infty
\]

接近完全保留网络 next-state proposal。

### 7.4 生成实际下一桥状态

\[
\boxed{
X_{g^{(k+1)}}
=
F(\hat C_k,\xi_k^*,g^{(k+1)})
}
\]

随后进入下一步：

\[
X_{g^{(k+1)}}
\rightarrow
S_\theta
\rightarrow
(\hat C_{k+1},\hat\xi_{k+1})
\rightarrow
\xi_{k+1}^*.
\]

---

## 8. 最终恢复

按照已冻结的 SA-V02-ADJ-1，物理 schedule 在完成请求的 \(K_{\mathrm{eff}}\) 次相邻 transition 或到达 \(g_{\min}\) 时停止。终点可以高于 \(g_{\min}\)，但始终大于 0，因为当前 PiRN/P2S 不保证：

\[
F(C,\xi,0)=C.
\]

若 \(K_{\mathrm{eff}}>0\)，在实际终点进行最后一次网络前向：

\[
S_\theta(X_{g^{(K_{\mathrm{eff}})}},g^{(K_{\mathrm{eff}})})
=(\hat C_{K_{\mathrm{eff}}},\hat\xi_{K_{\mathrm{eff}}}),
\]

最终输出：

\[
\boxed{\hat C=\hat C_{K_{\mathrm{eff}}}}.
\]

当 \(K=0\) 时，\(K_{\mathrm{eff}}=0\)，最终输出就是输入状态上的一次 \(\hat C_0\)。任何 comparator 均不得将 true clean 替换为最终输出。

---

## 9. 完整算法流程

### Training

\[
\boxed{
X_{i,g},g
\xrightarrow{S_\theta}
(\hat C_{i,g},\hat\xi_{i,g})
}
\]

\[
\boxed{
\mathcal L_C
+
\lambda_T\mathcal L_T
+
\lambda_B\mathcal L_B
}
\]

### Inference

\[
\boxed{
X_{g_k}
\xrightarrow{S_\theta}
(\hat C_k,\hat\xi_k)
\xrightarrow{\text{physics refinement}}
\xi_k^*
\xrightarrow{F(\cdot,g_{k+1})}
X_{g_{k+1}}
}
\]

其中 \(k=0,\ldots,K_{\mathrm{eff}}-1\)，schedule 必须严格取自 SA-V02-ADJ-1。

一句话定义：

\[
\boxed{
\text{V0.2 先学习一个能跨 }g\text{ 重放 trajectory 的 }\xi,
\text{再利用真实观测对其做受下一状态保护的测试时优化，}
\text{最后沿 PiRN/P2S 物理桥逐步恢复。}
}
\]

---

## 10. V0.1 → V0.2 的最小变化

| 部分 | V0.1 | V0.2 |
|---|---|---|
| 输入接口 | \(S_\theta(X_g,g)\) | 不变 |
| 网络输出 | \((\hat C,\hat\xi)\) | 不变 |
| 物理桥 | 每步 PiRN/P2S 重渲染 | 不变 |
| \(\xi\) 优化 | 每步优化 | 不变 |
| clean loss | 有 | 保留 |
| realization supervision | 仅终端 \(G\) | \(G,g,h\) 跨 severity |
| next-state supervision | 无 | 新增 \(\mathcal L_B\) |
| refinement objective | 只拟合原始 \(Y\) | 拟合 \(Y\)+保护 next state |
| 最终输出 | 实际执行终点的 clean branch | 不变；\(K_{\mathrm{eff}}\) 由 SA-V02-ADJ-1 决定 |

---

## 11. 必须保留的机制对照

1. **V0.1 original**：原始冻结 20k checkpoint 与其原冻结 one-shot 路径；
2. **B0 continued**：旧训练目标继续相同 extra updates；主图像控制使用 one-shot，next-state 诊断可使用同 schedule 的 no-refine 路径；
3. **V0.2 \(K=0\)**：同一 V0.2 checkpoint 的 one-shot clean output，无 transition；
4. **V0.2-no-refine**：每步直接以 \(\hat\xi_k\) 重渲染下一状态，不优化 \(\xi\)；
5. **V0.2-old-refine**：新训练、\(\beta=0\) 的 terminal-only refinement；它不是 V0.2-full；
6. **V0.2-full**：新训练、validation 冻结的 \(\beta>0\) trajectory-preserving refinement；
7. **oracle-\(\xi\) bridge**：clean 始终由网络预测，但每次 renderer 使用真实 \(\xi_i\)，不替换 true clean；
8. **teacher-forced true-state bridge**：每一步网络输入真实 \(X_{i,g_k}\)，不使用上一阶段预测生成的 state。

所有需要 transition 的 comparator 必须使用 SA-V02-ADJ-1，不得各自定义 schedule。若没有任何 \(\beta>0\) 通过 refinement gate，则 V0.2-full 不存在，必须报告 refinement FAIL/STOP；不得以 \(\beta=0\) 或 no-refine 冒充。

这些对照分别识别：

- trajectory training 的贡献；
- V0.1 refinement 是否会重新破坏 next state；
- trajectory-preserving refinement 的贡献；
- 物理桥本身的 oracle 上限；
- rollout failure 是否来自生成状态偏离 true simulator states。

---

## 12. 方法边界

V0.2 首先只支持以下结论范围：

\[
\boxed{
\text{matched-simulator、synthetic paired、known-}g\text{ 单帧湍流恢复}
}
\]

它不自动证明：

- unknown-\(g\) 场景有效；
- 真实湍流数据泛化；
- raw Gaussian \(\xi\) 可唯一恢复；
- 任意连续 \(g\) 或其他光学 protocol 可直接使用。

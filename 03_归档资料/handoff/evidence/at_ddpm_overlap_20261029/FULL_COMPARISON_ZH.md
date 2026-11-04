# AT-DDPM 与当前方案的定向重合核查

## Verdict

**Level 3 — Medium Overlap（针对完整 V6；定向核查，不是全网新颖性放行）。** AT-DDPM 已覆盖单帧条件扩散、弱/强退化蒸馏和带噪观测初始化。当前 V6 的 learned raw-xi 物理递归/自生成中间观测/完整反传，与该文的具体流程不同。PiRN-SR 已覆盖物理一致性与扩散的组合，DiffUIR 则是现用恢复器本身，因此这两者也必须明确承认。

若把我们的 claim 缩减为“用条件扩散恢复单帧湍流”，该想法已经存在；不能靠把人脸换成通用场景、换 backbone 或改少量步数来声称方法首次提出。

## Delta

AT-DDPM 通过弱/强条件的噪声预测蒸馏恢复人脸，当前 V6 则让预测 Clean/ξ 经可微模拟器生成下一物理观测，并对后续恢复联合反传；其目标是在匹配预算下减少递归误差累积、改善 PSNR/LPIPS，但这个收益尚不能由本次文献核查或 D0 对比图认定。

因此目前可以描述方法差异，不能写成“已经带来可测提升”的定稿创新句。

## Decomposed claim

- **Problem framing:** 单张退化 RGB → Clean；固定 G4 与真实数据，推理不引入 GT/额外帧。
- **Core mechanism:** 原有 DiffUIR 条件残差扩散 + learned raw-xi + 独立物理时间 + 可微模拟器产生递归观测 + Full-BPTT。
- **Key insight:** 让训练适应恢复器自身生成的中间物理状态，并保持对原观测的解释；这是待验证假设。
- **Application domain:** 通用场景湍流恢复。

## Structured papers

- **AT-DDPM**
  - Title: AT-DDPM: Restoring Faces degraded by Atmospheric Turbulence using Denoising Diffusion Probabilistic Models
  - Date: 2022-08
  - Problem framing: Single degraded facial image to a restored face; synthetic and real face evaluation.
  - Core mechanism: Conditional Gaussian DDPM with noise prediction, progressive adaptation from a face prior/SR model, weak/strong-condition distillation, and noisy-input initialization.
  - Key insight: Transfer a stable face prior across degradation strengths and retain coarse input structure during sampling.
  - Application domain: Atmospheric-turbulence degraded faces; FFHQ/CelebA training or synthetic tests and LRFID/BRIAR real faces.
  - Overlap score: 2/4 (triage, not proof of novelty)
  - Source: https://arxiv.org/abs/2208.11284
  - Venue: WACV 2023
  - Provenance: User-provided paper; public-title search; verified downloaded PDF and official source.

- **PiRN/PiRN-SR**
  - Title: Physics-Driven Turbulence Image Restoration with Stochastic Refinement
  - Date: 2023-07
  - Problem framing: Single-frame turbulence restoration with fidelity and perceptual evaluation across synthetic and real scenes.
  - Core mechanism: Train a deterministic Swin-based restorer with differentiable simulator consistency using stored synthesis protocol/realization; separately train a conditional DDPM on its restored output.
  - Key insight: Physics consistency improves the restoration constraint; a constrained restored image guides stochastic perceptual refinement.
  - Application domain: General atmospheric-turbulence images, including scenes, text, and faces.
  - Overlap score: 2/4 (triage, not proof of novelty)
  - Source: https://arxiv.org/abs/2307.10603
  - Venue: ICCV 2023
  - Provenance: Existing project reference; public-title search; verified downloaded PDF.

- **DiffUIR**
  - Title: Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model
  - Date: 2024-03
  - Problem framing: A single model for five image restoration tasks and zero-shot transfer.
  - Core mechanism: Conditional residual diffusion with explicit and concatenated image conditioning, a shared-distribution term, residual L1 learning, and three-step inference.
  - Key insight: Preserve useful conditioning while mapping different degradation distributions to a shared distribution.
  - Application domain: General RGB image restoration; original tasks include blur, low light, rain, snow, and haze.
  - Overlap score: 1/4 (triage, not proof of novelty)
  - Source: https://arxiv.org/abs/2403.11157
  - Venue: CVPR 2024
  - Provenance: Actual imported backbone and project reference; public-title search; verified downloaded PDF and local official source.

## Comparison result

- **Proposed work: full V6, not D0 alone**
  - Title: Current V6 physical recursive alignment (working description)
  - Date: 2026-10 method snapshot
  - Source: ../v6_diffuir_l_20261027/method_impl/ab.py:145-171
  - Problem framing: one degraded RGB observation to clean general-scene RGB
  - Core mechanism: inherited DiffUIR sampler plus learned raw-xi, distinct physical time, simulator-generated weaker states and full rollout backpropagation
  - Key insight: align recovery to its own generated physical-state distribution, with observation consistency (hypothesis)
  - Application domain: atmospheric turbulence, general scenes

- **AT-DDPM**
  - Title: AT-DDPM: Restoring Faces degraded by Atmospheric Turbulence using Denoising Diffusion Probabilistic Models
  - Date: 2022-08
  - Source: https://arxiv.org/abs/2208.11284
  - Problem framing: Single degraded facial image to a restored face; synthetic and real face evaluation.
  - Core mechanism: Conditional Gaussian DDPM with noise prediction, progressive adaptation from a face prior/SR model, weak/strong-condition distillation, and noisy-input initialization.
  - Key insight: Transfer a stable face prior across degradation strengths and retain coarse input structure during sampling.
  - Application domain: Atmospheric-turbulence degraded faces; FFHQ/CelebA training or synthetic tests and LRFID/BRIAR real faces.
  - Axes fully matching: 2/4; partial matches remain flagged and are not silently counted as full matches.
  - Level: 3 — Medium Overlap (two axes differ).

- **PiRN/PiRN-SR**
  - Title: Physics-Driven Turbulence Image Restoration with Stochastic Refinement
  - Date: 2023-07
  - Source: https://arxiv.org/abs/2307.10603
  - Problem framing: Single-frame turbulence restoration with fidelity and perceptual evaluation across synthetic and real scenes.
  - Core mechanism: Train a deterministic Swin-based restorer with differentiable simulator consistency using stored synthesis protocol/realization; separately train a conditional DDPM on its restored output.
  - Key insight: Physics consistency improves the restoration constraint; a constrained restored image guides stochastic perceptual refinement.
  - Application domain: General atmospheric-turbulence images, including scenes, text, and faces.
  - Axes fully matching: 2/4; partial matches remain flagged and are not silently counted as full matches.
  - Level: 3 — Medium Overlap (two axes differ).

- **DiffUIR**
  - Title: Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model
  - Date: 2024-03
  - Source: https://arxiv.org/abs/2403.11157
  - Problem framing: A single model for five image restoration tasks and zero-shot transfer.
  - Core mechanism: Conditional residual diffusion with explicit and concatenated image conditioning, a shared-distribution term, residual L1 learning, and three-step inference.
  - Key insight: Preserve useful conditioning while mapping different degradation distributions to a shared distribution.
  - Application domain: General RGB image restoration; original tasks include blur, low light, rain, snow, and haze.
  - Axes fully matching: 1/4; partial matches remain flagged and are not silently counted as full matches.
  - Level: 4 — Low Overlap (three axes differ).

## 具体到实现的区别

|维度|AT-DDPM|当前 V6|
|---|---|---|
|训练状态|Clean 加高斯噪声，输入强/弱退化条件|内层沿用 DiffUIR；B 外层用模型预测经物理 F 生成状态|
|主要预测|Gaussian epsilon；有弱/强条件一致性蒸馏|Clean 通过残差采样生成；另预测 raw-xi|
|物理模型作用|训练图的弹性变换/模糊增强|可微 F 既参与物理 loss，也产生下一次恢复的观测|
|时间|扩散去噪时间|内层扩散时间与外层物理强度时间分开|
|训练展开|随机扩散时刻的噪声/蒸馏目标|B 对四次恢复及三次物理状态转移整体反传|
|必须承认的已有部分|条件扩散、蒸馏、带噪输入初始化|DiffUIR 恢复器本身来自已有论文；物理一致性＋扩散组合还需对照 PiRN|

本地实际代码：[B rollout](<../../NOT_INCLUDED.md#ref-2b0fbe9e190317cafbe8>)、[最新 D0](<../../NOT_INCLUDED.md#ref-cae550782fa1d54893b1>)。当前最新 D0 不带外层 ξ/模拟器，不能用其输出图当作完整物理递归有效的证据。

AT-DDPM 证据：§3.2，§3.3 Eq.13–15，§4.1–4.3；PiRN 证据：§2.2.1、Eq.4/6、§2.3、§3.6；DiffUIR 证据：§3.2 Eq.11/Algorithm 1–2、§4.2。完整提取文本位于 papers/，源码核对片段位于 SOURCE_EVIDENCE.json。

## 边界与执行说明

三个目标 PDF 均真实下载并提取正文，只读查看现有科学代码；0 推理、0 训练更新、未改 checkpoint。六源检索脚本因 requests 缺失未成功运行，已用公开论文题名与原文做定向替代，不能声称穷尽全部相关论文。自动审批拒绝了包含内部方法描述的外部查询；该查询未执行，之后不再向外部发送内部机制。未触及训练依赖环境。


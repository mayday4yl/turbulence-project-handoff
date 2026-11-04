# 论文阅读顺序与项目对应关系

来源核对日期：2026-11-04。下面链接指向论文原文、正式会议页面或作者官方仓库；论文表现不等于本项目湍流实测。当前优先读P2S模拟器论文和UniDB；接手方具体consistency论文/代码确认后再接着读。其余骨干与基准论文按需查阅，用于溯源和历史背景。

|优先级|论文与来源|在本项目中的用途／先看什么|
|---|---|---|
|当前必读 · 模拟器|**Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform**，Zhiyuan Mao、Nicholas Chimitt、Stanley H. Chan，ICCV 2021。[论文](https://arxiv.org/abs/2107.11627) · [正式PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Mao_Accelerating_Atmospheric_Turbulence_Simulation_via_Learned_Phase-to-Space_Transform_ICCV_2021_paper.pdf)|理解P2S如何把相位表示转为空间PSF基系数；对照本项目36→100映射、平均核/基核卷积。V5的Smooth端点与时间接口是项目改动，不能统称论文原版。|
|当前必读 · 扩散基线|**UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control**，Kaizhen Zhu等，ICML 2025（官方仓库标注）。[论文](https://arxiv.org/abs/2502.05749) · [官方代码](https://github.com/UniDB-SOC/UniDB)|理解条件桥、SOC与终端惩罚；随后读`UniDB-GOU`的MatchingLoss、反向均值与100步采样。项目使用DIV2K SR gamma_1e7，commit固定为`1bb63e318891b932c6e641b98729643e46b91e06`，不是UniDB++。|
|接入参考 · 一致性基础|**Consistency Models**，Yang Song、Prafulla Dhariwal、Mark Chen、Ilya Sutskever，ICML 2023。[正式论文页](https://proceedings.mlr.press/v202/song23a.html)|理解沿轨迹的一致映射、边界与训练/蒸馏差别。这是基础阅读，**并未确认它就是会议中接手方的具体实现**。|
|溯源参考 · 湍流代码|**Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model**，Mao等，ECCV 2022。[论文PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790419.pdf) · [官方TurbNet](https://github.com/VITA-Group/TurbNet)|了解PiRN/P2S资产来源、单帧湍流任务与基线。TurbNet是恢复模型，P2S是生成模拟器；不要混称。本项目曾未取得TurbNet官方checkpoint，不能把TurbWNet结果写给它。|
|历史参考 · V5恢复网络|**Efficient Concertormer for Image Deblurring and Beyond**，Pin-Hung Kuo等，ICCV 2025。[正式PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Kuo_Efficient_Concertormer_for_Image_Deblurring_and_Beyond_ICCV_2025_paper.pdf) · [官方代码](https://github.com/setsunil/Concertormer)|V5采用官方GoPro恢复主体，再加连续时间与raw-ξ输出；D0保留clean-only形式。先看恢复主体/残差输出，再对照项目适配，不能把新增物理头归给原论文。|
|历史参考 · V6|**Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model**，Dian Zheng等，CVPR 2024。[论文](https://arxiv.org/abs/2403.11157) · [正式页](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Selective_Hourglass_Mapping_for_Universal_Image_Restoration_Based_on_Diffusion_CVPR_2024_paper.html) · [官方代码](https://github.com/iSEE-Laboratory/DiffUIR)|V6 DiffUIR-L来源；区分其内部S3与外层物理A/B。项目曾改变训练目标和强度，不是全程官方原生对照。|
|历史参考 · 候选排查|**ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting**，Zongsheng Yue、Jianyi Wang、Chen Change Loy，NeurIPS 2023。[论文](https://arxiv.org/abs/2307.12348)|理解残差迁移与当时考虑的latent路径。项目的VQ往返漂移是特定codec诊断，不是该论文所有设置或所有latent模型无效。|
|历史参考 · V4骨干|**Simple Baselines for Image Restoration**，Liangyu Chen等，ECCV 2022。[论文](https://arxiv.org/abs/2204.04676)|NAFNet来源；项目使用SIDD width64主体再改输入/头。历史匹配对照见V4报告，不与UniDB不同协议直接排名。|
|历史参考 · V4骨干|**Restormer: Efficient Transformer for High-Resolution Image Restoration**，Syed Waqas Zamir等，CVPR 2022。[论文](https://arxiv.org/abs/2111.09881)|早期V4的Transformer恢复骨干；本项目A100 fresh-U0是共同随机初始化，不能写成官方去噪预训练迁移。|

## 会议中的CGDB：找到公开候选，但不代替本人确认

检索发现公开匿名稿 **Consistency Geodesic Bridge: Image Restoration with Pretrained Diffusion Models**，[OpenReview PDF](https://openreview.net/pdf/31fe5d6fe18cc375881bad6828d181d878d608fb.pdf)。该文件中有“Consistency Geodesic Diffusion Bridge Training / CGDB”术语，页眉为ICLR 2026审稿稿件。

**仅凭缩写不能认定就是接手老师的工作，也不据此判断最终发表状态或作者身份。** 交接时请接手方确认论文版本、代码仓库/commit、所用solver和loss。公开候选保留为对照入口，当前不把其中公式写进本项目已实现方法。

## 原论文与本项目证据怎么一起读

1. 原论文解释方法动机和定义；本项目冻结源码决定实际执行。
2. 读模拟器时同时看[项目公式与限制](SIMULATOR_GUIDE.md)；原论文没有为本项目所有FFT、坐标和Smooth选择背书。
3. 读扩散时区分论文目标、官方代码目标和本项目某一阶段的实际目标；避免用论文结论解释尚未定位的失败。
4. 外部论文表格只能注明引用；只有同数据、同指标口径的结果才适合匹配比较。不能把引用数字写成本项目已复现。

精简BibTeX入口：[references.bib](references.bib)。其中公开匿名候选只用`misc`并标待确认，不填写推测作者或录用信息。

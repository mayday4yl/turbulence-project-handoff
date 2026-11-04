# DeblurDiff 与 DiffBIR v2：对照会后要求

2026-11-01。论文与固定官方源码只读核查；0模型创建、0权重载荷下载、0推理、0训练、0加速卡任务。旧D0/A/B和选型报告不变。

## 结论

二者都符合“预训练生成扩散＋可学习图像条件控制”的架构思路；二者都不符合当前用户明确的“不采用latent、RGB像素空间扩散”。不扩散恢复残差Y−C，并不意味着它是RGB像素空间模型。

会议原始转录《智能纪要：新录音 3 2026年11月1日.md》05:25出现“原始图像扩散”，05:40又建议“预训练的diffuser…control net…IR”。这是两处实际转录依据，不能擅自替老师裁定“已同意采用latent”。当前执行筛选仍服从用户明确的无latent要求；若未来要选下面两种，需明确这是放宽空间约束，而不是声称它们属于像素扩散。

## 对照

| 项目 | DeblurDiff | DiffBIR v2 |
|---|---|---|
| 正式发表 | NeurIPS2025主会 | ECCV2024；v2为作者标注的ECCV论文版本，v2.1是后续不同权重 |
| 基础先验 | Stable Diffusion2.1 | Stable Diffusion2.1-base |
| 扩散空间 | AutoencoderKL编码的4通道latent，H/8×W/8 | AutoencoderKL编码的4通道latent，H/8×W/8 |
| 是否扩散恢复残差Y−C | 否，clean latent加高斯噪声、预测epsilon | 否，clean latent加高斯噪声、预测epsilon |
| 图像条件 | LQ latent＋LKPN每步恢复latent，拼成8通道hint；输入ControlNet | 首阶段恢复器去退化，再VAE编码为图像condition，输入IRControlNet |
| 条件随输入变化 | 是；LKPN还读取当前z_t、t、文本，每步预测空间变化5×5核 | 是；不同图像产生不同condition；固定condition由随z_t和t变化的ControlNet处理 |
| 原生训练 | 训练ControlNet＋LKPN，冻结SD U-Net/VAE/CLIP；epsilon MSE＋LKPN latent MSE | 训练IRControlNet，冻结SD U-Net/VAE/CLIP；latent epsilon目标 |
| 公开配置成本 | 默认50步、CFG1，非tiling每步LKPN＋ControlNet＋SD U-Net各一次 | 官方v2示例50步、CFG4，spaced sampler每步调用cond/uncond两次：100 SD U-Net＋100 IRControlNet；另有首阶段/VAE等 |
| 当前资产证据 | 作者Drive链接，未验权重库存/strict load | 作者v2.pth及SD/首阶段权重入口，未下载或strict load |
| 当前工程限制 | LKPN依赖CuPy RawKernel、自定义CUDA正反向，非CUDA抛错；现Ascend不能直接运行 | 本轮未验证NPU运行、输入梯度、完整循环成本；不能称即插即用 |
| 对湍流的证据 | 去模糊任务更接近其中的blur，但没有本轮湍流质量验证 | 通用盲恢复框架，首阶段如何处理湍流几何扭曲尚无本轮证据 |

以上次数均是特定非tiling配置的静态源码计数，不是实测延迟；CFG、tiling、额外restoration guidance改变实际成本。不要与D0 3NFE直接按步数推导速度比例。

## 原文与代码依据

- DeblurDiff NeurIPS主会：[正式论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html)。固定官方commit 8c56b25a4740e3c6b6f08befa863bb053b800c29。model/cldm.py为VAE、LKPN、ControlNet、epsilon返回；model/lkpn.py为CUDA/CuPy限制；train.py训练ControlNet＋LKPN；model/gaussian_diffusion.py联合epsilon和LKPN latent MSE。详见deblurdiff/EVIDENCE.json，10个固定源码链接/行号摘要，不伪称全部源字节已本地归档。
- DiffBIR [ECCV正式页](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7690_ECCV_2024_paper.php)、[原文§3.2–3.3/§4.1](https://arxiv.org/html/2308.15070)。固定官方commit 5c2d6c177c3f2965e6d33a32931248d242ef7b3d；8份源字节包见diffbir_v2/SOURCES.json，全部Git blob精确核验见SOURCE_VERIFICATION.json。
- DiffBIR model/cldm.py:143–172明确prepare_condition经VAE、forward先ControlNet后SD U-Net；train_stage2.py:131–148明确GT经VAE成z0、首阶段SwinIR生成condition；spaced_sampler.py:144–158明确CFG分支两次完整model调用。README明确v2是ECCV版本，不能混成v2.1。
- 文本prompt并不等于自动提供了湍流task prompt。两者保留SD文本条件，但恢复主要依据可学习图像控制；DeblurDiff默认positive为空，DiffBIR论文也使用空positive和质量相关negative。不能据有CLIP/text分支就宣称已经提供输入自适应的湍流退化prompt。

## 对当前选型的含义

若保持像素空间硬要求，二者只作latent控制路线的机制参考，不加入符合条件的主榜。若未来允许latent，DeblurDiff值得研究空间变化条件与迭代核预测，DiffBIR值得研究成熟的两阶段IRControlNet。此时还要独立核验VAE表征损失、几何保真、原尺寸/递归输入梯度与成本，不能把现成latent权重去掉VAE后当作可用pixel权重。本轮没有提出或执行新的训练。


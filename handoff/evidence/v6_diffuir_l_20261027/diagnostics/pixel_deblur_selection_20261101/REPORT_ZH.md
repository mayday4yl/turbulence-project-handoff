# 无VAE、像素空间条件扩散：去模糊专项核查

2026-11-01。仅研究及官方资产metadata核查；0模型运行、0训练、0加速卡任务、0权重/数据载荷下载。现有checkpoint与冻结报告不变。

## 结论

有。**若优先要求官方代码与真正存在的去模糊权重，IR-SDE（ICML2023）是本轮最完整的可核候选。** 它在RGB图像空间运行，作者已发布去模糊专用checkpoint。但它属于mean-reverting条件SDE，不是纯高斯终点的标准DDPM；框架通用而非只做去模糊。

如果同时坚持前面“完整清晰RGB高斯扩散、不扩散恢复残差”的机制，**Multiscale Structure Guided Diffusion（ICCV2023，简称MSG）**与**DP-IR（NeurIPS2024）**是真实匹配，不过本轮未找到作者完整代码＋checkpoint发布，不能称已可运行。不能为了满足代码权重条件，将latent方法、残差模型、重模糊生成或第三方复现塞入主榜。

| 候选 | 正式发表 | 状态与条件 | 资产/成本边界 |
|---|---|---|---|
| IR-SDE | ICML2023 | 全RGB清晰图通过均值回归过程走向以退化图为均值的噪声状态；reverse网络每步读当前RGB和退化RGB，无VAE | 官方代码＋单文件metadata已核ir-sde-deblurring.pth，548648748B；默认reverse100步；未下载strictload或验NPU/输入梯度 |
| MSG | ICCV2023 | 清晰RGB直接加高斯噪声，预测epsilon；输入blurryRGB＋多尺度结构条件feature，无VAE与恢复残差状态 | 论文/补充/作者research页已核；未找到官方代码/CP。Table12有20–1000步实验，不宣称某个快速配置为已发布默认 |
| DP-IR | NeurIPS2024 | 全维sRGB清晰图加高斯噪声；首阶段FFTFormer恢复图、每步MIRNet-S去噪结果与时间经0.73M Fusion形成conditional score | 论文有GoPro/HIDE；完整作者code/denoiser/fusion权重未找到。作者5reverse＋1IR=6NFE口径，不等于系统仅6次网络调用 |

## IR-SDE的实际发布证据

[正式ICML论文](https://proceedings.mlr.press/v202/luo23b.html)、[作者代码](https://github.com/Algolzw/image-restoration-sde)。
官方根目录→pretrained_models→pretrained-deblurring→ir-sde-deblurring.pth；三层目录与单文件metadata回读一致。文件ID1MZzKbY8I63id4Ro3T5TWGNYM0-JIbpB2，548,648,748字节。[官方权重页面](https://drive.google.com/file/d/1MZzKbY8I63id4Ro3T5TWGNYM0-JIbpB2/view)。仅库存元数据，不是权重SHA/完整state加载验收。

固定commit2598d73cde3a1e2cf78c4ba4923414e1f4b90393：
- codes/config/deblurring/options/test/ir-sde.yml：GoPro test_full，ConditionalUNet in/out3，T100。
- test.py：直接sde.noise_state(LQ)，没有VAE编码。
- sde_utils.py：mu_bar为mu+(x0-mu)*exp(-theta_sum*dt)，state维持RGB，网络预测noise/score。
- DenoisingUNet_arch.py：网络输入cat([xt-cond,cond])，有图像中心化差分，但这不等于把最终恢复残差Y-C作为扩散状态与目标。不能说源码完全没有做减法。
- denoising_model.py：官方训练是反向一步期望与optimal-step目标的maximum-likelihood训练，不是照搬当前D0 raw Residual L1。
- 同一repo也含Refusion/Latent-Refusion，**本轮只核IR-SDE去模糊路径**；不能由README出现latent就把所有分支都判latent，也不能误取Refusion权重。
- 原公开train YAML中train与val路径都指向作者本地gopro/test_full。这需要未来核实实际训练数据配置，不能直接照抄启动；亦不能仅凭模板路径断言发布checkpoint已经泄漏测试集。本轮未取得checkpoint训练日志，训练来源/兼容性仍未完整验收。

## MSG与DP-IR为何是真匹配

[MSG论文§3.1–3.3](https://arxiv.org/html/2212.01789)直接以clean RGB构造xt、输入[xt,blurredRGB]，epsilon L1；结构指导网的多尺度灰度回归feature作为条件，文中latent space指中间特征，并非VAE压缩的扩散状态。正式会议为[ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ren_Multiscale_Structure_Guided_Diffusion_for_Image_Deblurring_ICCV_2023_paper.html)。[作者research页](https://www.mengweiren.com/research/)只找到arxiv/BibTeX；有界检索未找到官方代码/权重，不断言永未开源。

[DP-IR NeurIPS正式页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b213d9a999b82cb6fcd03a0d5a7498be-Abstract-Conference.html)、[论文§3.1–3.3/§4/附录H](https://arxiv.org/html/2411.05993)。扩散变量与原始数据同维度，sRGB训练；DDPM的hierarchicalVAE/latentvariables只是概率模型术语，不是图像VAE降维。GoPro/HIDE为单帧deblur，burstJDD-SR是另一个任务，不混成多帧。论文称曾在8张Ascend910上训练，但我们未取得完整代码，也未进行运行验收；附录列出的FFTFormer/BSRT/SwinIR资产不能替代DP-IR的denoiser＋fusion权重。方法适合读条件注入和少步采样，资产尚不足以承诺直接部署。

## 不混入主榜

- DvSR（Deblurring via Stochastic Refinement，CVPR2022）：无VAE、像素空间，但显式扩散clean−initial predictor残差；常见Github为非官方复现。若未来放宽非残差要求可单列，不伪称当前严格匹配。
- UCDIR（NeurIPS2023）：恢复初步指导图的残差，按前面严格偏好排除；动态核条件可作为机制参考。
- HI-Diff（NeurIPS2023）：扩散compactpriorfeature，非直接cleanRGB。
- EGDeblurring（CVPR2025）：扩散event/motionpriorfeature，再由恢复器输出RGB，非直接cleanRGB。
- ID-Blau（CVPR2024）：diffusion做清晰→模糊数据增强，不是扩散恢复器。
- DDBM（ICLR2024）：旧已核官方发布仅Edges2Handbags与DIODE权重，未查到去模糊专用checkpoint，本轮不把它列GoPro现成模型。
- DiffPIR/BlindDPS：无条件先验加算子/测量引导，与学习好的退化图条件去模糊网络分列。
- DeblurDiff/DiffBIR：SD＋VAE latent，仍不满足用户无latent要求。

## 当前建议与边界

先以IR-SDE做“可获取官方像素去模糊模型”的后续技术验收候选；MSG/DP-IR作为更直接的fullRGB条件扩散论文阅读。先完成权重严格加载、VAE无关路径、输入条件梯度/原尺寸与采样成本等核验，才有依据决定是否用于湍流。100步代价可能影响递归训练，不能直接改成3步再称官方方法；去模糊权重对湍流warp与空间变化PSF是否有效本轮未知。

权重库存真实存在不等于湍流恢复质量已证实。此报告没有授权或执行新模型推理、训练、A/B。


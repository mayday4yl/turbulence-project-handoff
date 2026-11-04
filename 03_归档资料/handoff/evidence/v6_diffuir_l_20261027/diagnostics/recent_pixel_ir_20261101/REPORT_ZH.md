# 2025–2026 顶会像素条件图像恢复：近期选型核查

2026-11-01。主范围2025–2026，晚2024作为约两年范围补充。只有论文、固定源码与小型metadata；0权重/数据载荷、0模型/训练/加速卡、A/B保持暂停。旧科学资产不改。

**有，近期原生像素条件IR不必只举IR-SDE。** UniDB（ICML2025）、ResFlow（CVPR2025）、UDBM（ICML2026）、GenDR-Pix和LearnIR（ICLR2026）均已核正式发表，但“无VAE”“非恢复残差”“代码权重可运行”不能混成一个PASS。

| 模型 | 正式主会 | 状态与条件/目标 | 作者资产 |
|---|---|---|---|
| UniDB / 发布UniDB-GOU | ICML2025 | RGB3，每步6ch图像条件；noise/score与反向ML，不直接回归Y−C | 代码/训练配置/Drive链接；库存、strictload未验 |
| ResFlow | CVPR2025 | RGB与辅助变量增广flow，从LQ出发；velocity，无VAE | 官方README only、Code Coming Soon；未找到CP |
| UDBM | ICML2026 | RGB不确定性条件桥，Stage1＋bridge；发布实现pred_res=LQ−GT L1 | 官方代码、S/M/L双权重入口；未下载 |
| GenDR-Pix | ICLR2026 | 去VAE，以pixelshuffle无损重排RGB；一步蒸馏恢复，HQ目标 | 完整官方Pix实现/CP未找到，parent仍under construction |
| LearnIR | ICLR2026 | RGB动态分辨率＋learned posterior correction，无VAE；仍有residual-noise目标 | 官方repo只有README/assets，Code Coming Soon |

正式会议：[UniDB](https://proceedings.mlr.press/v267/zhu25o.html)、[ResFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Reversing_Flow_for_Image_Restoration_CVPR_2025_paper.html)、[UDBM](https://proceedings.mlr.press/v306/tu26d.html)、[GenDR-Pix](https://proceedings.iclr.cc/paper_files/paper/2026/hash/acb7ce5aab6e134300a2361dd90a501f-Abstract-Conference.html)、[LearnIR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/51480fa7058146264ac6fc145eeb13b4-Abstract-Conference.html)。

## 实际优先级与限制

**沿用用户此前“不以恢复残差为目标”的偏好，近期已有官方可运行代码的原生RGB条件IR，UniDB-GOU较实际。** 它是新的SOC/GOU桥框架、仍用RGB条件U-Net，默认100次网络，发布任务去雨/SR/人脸inpaint；未发现去模糊专用配置。不是新DiT基础主干，也不是标准纯高斯终点DDPM。权重链接不等于库存/加载验收。

UniDB和UDBM是不同作者的不同论文，不是同方法两版本。UDBM正式会议为ICML2026，修正检索中的CVPR2025误标签，历史冻结原件不改。UDBM默认1桥步＋1Stage1是两次不同网络，不能报整个系统1NFE；发布代码恢复残差参数化与论文x0叙述区分，严格非残差主榜单列。

ResFlow和GenDR-Pix原生目标均非旧D0恢复残差，方法可读，但官方完整资产未找到，不能用第三方starter或旧latent GenDR权重代替。GenDR-Pix最终无VAE、192ch像素重排，原生训练含教师/对抗/感知/频域目标，不能冒充随机t普通DDPM。ResFlow属于增广flow，不和Gaussian-DDPM混写。

LearnIR真正避免预训练VAE，但DRM是图像分辨率变换，前向与目标仍明确Rz/residual-shifted noise。官方repo当前无model源码和CP，不得只因About写“implementation”就称代码已发布。已保存实际tree和README。

PMRF（ICLR2025）为RGBflow，有作者代码/人脸权重，但主式是端点条件，v(zt,t)不逐步额外接rawY；conditional_on_Y是独立对照。主要face512/受控face256，NATTEN/CUDA，非当前通用场景首选。

若目标是老师第二条“像素预训练生成模型＋条件支路”，PixelDiT（CVPR2026）仍应重点比较；它有生成权重，却不是现成原生IR，也无本轮湍流质量。IR-SDE留历史/技术基线即可，不以年代代替质量判断。

## 边界核查

- DPIR（CVPR2025 Dual Prompting）用SD3＋ControlNeXt，为latent；与NeurIPS2024的DP-IR不是同篇。后者fullRGB可读但完整作者资产缺口仍在。
- E-Bridge（ICLR2026）公开FLUX/AE encode-decode，不进strictpixel榜。
- BDG（ICLR2026）、DGSolver（NeurIPS2025）、Resfusion（NeurIPS2024）为恢复残差/resnoise边界。
- FoundIR-v1（ICCV2025）已核核心与DiffUIR相同，不因新年份称独立新backbone。
- RDBM（CVPR2026）不能只凭标题排除：公开target=x_start，恢复残差噪声乘子被注释，实际RGB条件桥/cleanL1；与paper残差调噪有实质差别，CP未找到，标源码/论文对应缺口。
- PixRestore、PixelSR是2026像素条件IR预印本追踪，当前未核正式顶会接收；PixelSR官方under review。
- DiP/PixelDiT等生成主干与原生IR分列；DAPS等算子求逆与学习好的退化图条件恢复器分列。

## 可追溯性与完成边界

native_recent含UDBM/PMRF，unidb含8份精确Gitblob；conference_scan含GenDR-Pix及latent/生成/预印本排除；boundaries含E-Bridge/RDBM等固定源码及resflow补证。LEARNIR_SOURCE_EVIDENCE保存官方README和实际tree。CANDIDATES.json/csv保留目标、条件、任务、采样口径和资产边界。

这是有界调研，非全领域穷尽性证明；没有候选模型运行、训练、NPU或湍流效果证据。后续选型优先级不是未经测量的质量排名。


# PixelDiT：V6无VAE条件扩散选型核查

2026-11-01，仅论文、官方小型源码与权重metadata。0模型实例/前向、0训练、0权重/数据载荷下载、0加速卡调用。本文只讨论PixelDiT第一代，不能把同repo PixelDiT2配置或其他PiD decoder权重混进来。

## 明确判断

**PixelDiT应进入“像素预训练生成模型＋图像条件适配”重点候选。** 前面的研究仅提ADM作为像素生成先验不够完整，这里补充较新的真实官方选择。它满足无VAE/直接RGB、顶会、有生成预训练权重；它不是现成的去模糊/湍流恢复器，尚不能直接拿官方权重做D0并期待恢复。

对照老师会议中“用预训练diffuser，再通过ControlNet形式做IR”这条路线，PixelDiT在架构类别上适合；若目标是立即使用已发布去模糊权重，IR-SDE更接近ready的技术验收。两种优先级分属不同路线，不是未经实测的湍流质量排名。

## 论文、状态、目标

PixelDiT: Pixel Diffusion Transformers for Image Generation，**CVPR2026主会**，官方仓库标注Oral及BestPaperFinalist，NVIDIA会议议程支持Oral。[正式会议论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_PixelDiT_Pixel_Diffusion_Transformers_for_Image_Generation_CVPR_2026_paper.html)、[作者论文](https://arxiv.org/html/2511.20645)、[官方仓库](https://github.com/NVlabs/PixelDiT)。

原生状态是RGB3图像，学习与采样都不经过VAE/VQ图像编解码。patch-level DiT建模全局语义，pixel-level PiT处理细节；内部token compaction/feature projection不意味着另有图像VAE状态。类条件配置明确in_channels3；不要把类构造器默认in_channels4当作发布配置。

论文§3.4为**Rectified Flow＋velocity matching**，不是旧D0恢复残差Y−C建模。其训练可含DINOv2表示对齐REPA，DINO仅监督中间特征，不是将RGB先压成latent再扩散。T2I后续stage停止REPA；不能笼统说所有公开训练stage都必须保留REPA或无声接旧ResidualL1/SDT/Adam。

## 条件类型不能混淆

| 版本 | 公开条件 | 是否已具备模糊RGB恢复条件 |
|---|---|---|
| PixelDiT C2I | ImageNet类别标签，时间；类别嵌入进入AdaLN | 本轮没有找到，y是ClassEmbedder标签，不是LQ图 |
| PixelDiT T2I | Gemma2文本embedding与时间，通过MM-DiT交互 | 本轮没有找到，y必须为B×L×D文本特征 |
| 论文FlowEdit编辑演示 | 训练外FlowEdit图像编辑协议 | 编辑保背景展示不是去模糊或湍流恢复训练权重 |

源码forward(x,t,y,s=None,mask=None)不能按变量名把y、s或mask解释成退化RGB条件。C2I的s默认由当前noisy图的patch分支生成，是内部上下文；mask为注意力相关输入。T2I检查y.dim()==3并调用文本投影。要做配对湍流恢复，仍需新增/适配有空间信息的图像条件路径并训练，不能仅把“remove turbulence”写进prompt当成已完成条件恢复。

## 公开权重、规模、成本

官方第一代C2I有256/512权重；论文PixelDiT-XL约**797M参数**。T2I生成主干约**1.3B**，另需Gemma2-2B文本编码器，官方发布最高1024训练模型。模型论文中的B184M结构不等于已发现官方B预训练checkpoint。本轮未下载/strict-load或验证NPU。

官方C2I示例采用FlowDPMSolver100步。固定源码每步将cond/uncond拼成2B一次net调用，因此是**100次批量前向、200次单分支等效评估**；guidance区间外仍执行两支。不能写100步等于100个batch1条件网络计算，也不能推导为D0的100/3倍耗时。T2I公开默认50步，文本编码、CFG和新增条件支路成本另计，未做完整实际NFE/耗时测量。

官方HF仅做库存/文件页面metadata，UI近似体积不是下载后实际SHA/容器解析。没有即刻权重加载或恢复质量PASS。所有未来大文件获取按用户要求走服务器网络，本轮仅小型文本/metadata。

官方代码采用NSCLv1，限定非商业研究/评估，科研用途对应但不能称任意商用许可。[LICENSE](https://github.com/NVlabs/PixelDiT/blob/f7d7c35f71893d81c6e71d2576ea80601a1e8529/LICENSE)。此为许可证原文范围描述，非法律判断。

## 像素ControlNet式条件分支并非完全没有参考实现

[PixelControl](https://arxiv.org/abs/2608.15705)，2026-08预印本，有[作者代码](https://github.com/linxin0/PixelControl)。它使用冻结PixelDiT-T2I与14层零初始化adapter，处理depth/seg/edge及组合。公开代码和配置存在；不能沿用“所有像素控制支路都需完全从零写”的说法。

但当前条件不是退化RGB，未查到去模糊/湍流任务；control checkpoint下载未找到，README/usage要求自备本地base/control权重。此预印本不能写成已核顶会，也不能当现成IRControlNet或自动宣称迁移有效。详见control_boundary/EVIDENCE.json和4份固定源码精确Gitblob验收。

## 当前建议

如果重点转为老师第二路线，优先深入PixelDiT的**256像素模型＋空间图像条件适配**可行性；这只是选型建议，未实现新支路或启动训练。要先弄清主干权重严格加载、条件注入位置、NPU算子/原尺寸/梯度与实际计算预算，再决定下一阶段。高质量生成FID和FlowEdit细节保留，不足以证明湍流PSNR/结构保真或warp恢复。现有D0 checkpoint与科学结果全部保留。


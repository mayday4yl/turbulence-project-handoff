# V5-v1 Backbone 候选比较

核查日期：2026-10-21。范围：官方论文、固定版本源码、配置与公开权重入口；不下载 checkpoint、不运行 Backbone NPU 前后向或训练。本报告是选型建议，不是恢复效果或硬件验收。

**第一推荐：Concertormer full，优先进入 V5 适配验证；跨架构备选：BioIR-L；AdaIR 暂缓。** 主要理由是 Concertormer 有明确的单帧去模糊配置和官方权重发布入口，50.54M 的完整配置提供了中大型容量。这个排序是任务和结构匹配判断，不是“参数越大越好”，也没有证明它在 NPU 上最快、最省显存或湍流效果最好。若 full 仅因容量/计算预算不合适，可先考虑官方 lite；若是 CSA 算子兼容性问题，lite 仍有相同算子，应转向 BioIR-L。

## 1. 官方结构、配置与规模

参数为按官方源码构造器与配置在 CPU/meta 上求和的**原模型注册参数**，未加载权重，不含 V5 新时间条件和双头。FLOPs 为各论文的 256×256 RGB 口径，不是本次测速；不同论文的统计工具未统一。

| 候选 | 实际结构 | 官方主要配置 | 精确参数 | 论文单次 FLOPs |
|---|---|---|---:|---:|
| **Concertormer full** | 四尺度 U-shaped Transformer/Hybrid；CSA 内含卷积与门控，局部/跨区域及通道交互 | dim48；encoder `[6,8,24]`，middle2，decoder `[24,8,8]`；heads1/2/4/8，window8 | **50,535,664** | 220.2G |
| Concertormer lite | 同一算子体系，减少深度 | dim48；encoder `[4,4,12]`，middle2，decoder `[12,4,4]` | 28,985,644 | 116.79G |
| **BioIR-L** | 三尺度 U-shaped 卷积/动态混合 Hybrid；像素—区域交互替代标准多头自注意力 | dim48；三阶段 `[6,6,14]`，镜像decoder，refinement4，共56块 | **15,847,328** | 125.64G |
| **AdaIR** | 四尺度 Restormer Transformer U-shape，通道注意力+3个频率模块 | dim48；blocks `[4,6,6,8]`，refinement4，heads1/2/4/8，FFN2.66 | **28,784,824** | 147.45G |

BioIR-T/B 分别为1,316,728/7,280,200参数；本轮中型候选选L，不人为扩大宽深后仍沿用官方L权重。BioIR-L 有18,432个已注册但未被forward调用的参数，表中保留官方原模型总数。三个模型都不能简单归为纯CNN；BioIR也不能仅凭块类名就归为标准自注意力Transformer。

来源：[Concertormer 配置与源码](https://github.com/setsunil/Concertormer/tree/94269c556277805051644343e3a2cc76c4cd7aaf)、[ICCV正文](https://openaccess.thecvf.com/content/ICCV2025/papers/Kuo_Efficient_Concertormer_for_Image_Deblurring_and_Beyond_ICCV_2025_paper.pdf)及[补充表10](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Kuo_Efficient_Concertormer_for_ICCV_2025_supplemental.pdf)；[BioIR 源码](https://github.com/c-yn/BioIR/blob/ff61e477ecf64db2731ef080a1e586f3f9d4248d/All_in_One/net/model.py)及[NeurIPS论文表9](https://proceedings.neurips.cc/paper_files/paper/2025/file/73ba81c7b25134a559c8a9c39ec1a4c3-Paper-Conference.pdf)；[AdaIR源码](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/net/model.py)及[ICLR论文表17](https://proceedings.iclr.cc/paper_files/paper/2025/file/fb44a668c2d4bc984e9d6ca261262cbb-Paper-Conference.pdf)。

## 2. 预训练核实到哪一步

| 候选 | 已核实的官方发布情况 | 去模糊相关性 | 尚未核实 |
|---|---|---|---|
| Concertormer | [官方Drive包](https://drive.google.com/file/d/1NvbGroZm4vVgvWJtmgW-6Fwd97PiukiL/view)页面及元数据可访问；`CSA_models.zip` 1,298,912,519字节（约1.21GiB） | 官方[GoPro测试配置](https://github.com/setsunil/Concertormer/blob/94269c556277805051644343e3a2cc76c4cd7aaf/options/test/GoPro/Concertormer.yml)指向full/lite；另有RealBlur-R/J、REDS专用路径 | 包内部各文件、tensor内容、哈希、strict load。不能只凭配置路径说权重已加载可用 |
| BioIR-L | [官方公开目录](https://drive.google.com/drive/folders/1tBv-4ixucMUxPCnVr3ZHf9EHlUnVM1vD)列出 `BioIR5D.ckpt`，182.3MB | 5D包含GoPro去模糊；未找到专用GoPro大模型权重。3D不含去模糊；`LOLBlur.pth`是Tiny联合低光/模糊任务，不能当L权重 | 字节下载、哈希、严格加载及与固定源码逐层对应 |
| AdaIR | [官方目录](https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH)与[5D预览](https://drive.google.com/file/d/1Th3Tg8Qck38HQfg1SkjC8Fq4dKn7aGuG/view)可访问；`adair5d.ckpt` 345,795,801字节 | 5D包括GoPro；当前公开目录未见deblur-only文件，deblur测试默认用5D | 同上，另需澄清下面的频带实现行为 |

所有数值计数均不代表成功加载预训练。官方从零训练配置、公开预训练入口和第三方未知checkpoint是不同证据；本轮没有采用第三方权重。Concertormer的GoPro train配置从零训练，RealBlur配置从GoPro继续训练，这与已发布预训练入口并不矛盾。

## 3. 接入 V5 的共同改动与边界

三个项目本身都接单张RGB，无需扩成6通道。V5保留 `Nθ(Z_RGB3,t) → (Ĉ, ξ̂)`，单独新增连续时间embedding/MLP，在各尺度注入条件；不把t拼成第4图像通道，不额外给网络g/G/N/gamma。时间调制末端可零初始化，减少对预训练主体初始特征的扰动。AdaIR现有 `noise_emb` 形参没有被使用，不能冒充已实现时间条件。

三个官方模型都是加性RGB残差输出，现有V5继承的双输出则为**sigmoid绝对clean RGB与无界raw-ξ**。因此应复用兼容主体、接既定clean/raw头，记录新增头和时间层初始化；不能宣称完整继承官方RGB恢复映射或直接把残差头加载为绝对RGB头。依据：[现有双头参数化](<../v4_nafnet_sidd116m_candidate_20261005/ADAPTATION.md>)。Concertormer官方多尺度RGB输出、MSL1FFTLoss，以及另两者原训练损失均不引入V5。

可取末端256²特征：Concertormer/BioIR-L为48通道，AdaIR为96通道。raw分支上采样×2，保持 `epsilon_zernike=[B,36,512,512]`、`epsilon_tilt=[B,512,512,2]`；不改成位移场/PSF，不添加raw裁剪。NFE属于外层循环，四次调用共享同一套参数。**这些是源码支持的适配建议，本轮尚未实现完整网络，也未冻结具体头宽度。**

## 4. 四步 Full-BPTT 的成本与 Ascend 风险

以B1、RGB256、raw512、FP32为例，单次38通道raw输出本身为38MiB，四次合计152MiB。Simulator每个不同预测clean的100个RGB基卷积结果为 `[1,3,100,256,256]`，单份75MiB；跨步clean不同，不能错误地当作同一clean缓存反复复用。还要保留物理映射、FFT、warp及跨步反向所需中间量。因此只看参数显存远远不够，四步也不能通过detach来省显存而仍称Full-BPTT。

| 候选 | 核心算子及风险 | 源码指定张量估算，非峰值显存 |
|---|---|---|
| Concertormer | 5D matmul、softmax、einsum、Conv3d `(3,3,1)`、频繁布局变换；Python自定义LayerNorm backward。核心不依赖FFT、unfold或自定义CUDA kernel | full所有softmax输出累计约756MiB/次，四次约2.95GiB；不含QKV、Conv3d输入等 |
| BioIR-L | einsum产生像素—区域affinity，3×3/5×5 unfold，reflection pad、depthwise、插值。没有核心复数FFT、Conv3d、自定义CUDA | full-resolution affinity每块64MiB；全网该类张量累计约1.297GiB/次，四次约5.188GiB。unfold在通道均值后作用于单通道，不按全C展开 |
| AdaIR | 通道attention是每头通道×通道，不是HW×HW；另有3次FFT+6次IFFT、复数abs、动态整数切片。无核心einsum/unfold/自定义CUDA | 本轮未统计全部保存激活；不能因通道注意力矩阵小就推断总内存小 |

**前两行统计的是不同类别的张量，不能据此给峰值显存排名。** 这些是无激活重计算时的逻辑张量量级，不是测得的NPU内存，更不是“能装进某张卡”的保证。实际峰值、合法batch、吞吐、秒/step、混精策略与四卡收益均未知。激活重计算可能保留Full-BPTT并降低存储，但会增加计算，需后续验证。

论文FLOPs简单乘四：Concertormer full约880.8G、lite467.16G、BioIR-L502.56G、AdaIR589.8G，仅为四次原骨干前向，不含新头、Simulator、反向或refinement。Concertormer作者RTX4080 Super单次full/lite为83.46/45.03ms，不能换算成de36训练耗时，也不用于承诺速度排名。

现有de36记录环境为Ascend910B3、Torch2.7.1/torch_npu2.7.1.post2；本轮没有使用NPU。以上风险来自源码，**三个完整候选均未在该环境验证**。Simulator既有FFT证据不覆盖AdaIR的复杂FFT/abs反传；标准PyTorch算子也不自动代表Ascend兼容。Concertormer应使用基础 `Concertormer` 接入训练，官方 `ConcertormerLocal` 有B=1限制和CUDA调用，不能原样移植TLC路径后沿用论文质量结论。三个官方训练入口里的CUDA/Lightning/NCCL配置都需要接入现有NPU框架。

## 5. AdaIR 暂缓的具体原因

官方 `FreModule.fft` 用 `h_=(h//128*sigmoid_threshold).int()` 决定低频掩码半宽。RGB256的三个频率模块特征尺寸为32/64/128：前两者整除结果为0，128尺度在sigmoid未饱和为1时也得到0。原样CPU模块检查三者均出现空低频掩码、低频输出全零，阈值预测层梯度为None；整数切片本身也不向阈值层传梯度。来源：[固定源码337–364行](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/net/model.py#L337-L364)、[本次CPU结果](<../../NOT_INCLUDED.md#ref-4008a2a34d3b24d2de55>)。

这是频带分离机制的明确疑点，**不意味着整个AdaIR不能学习**；其他路径仍可训练。修改掩码会改变原算法和checkpoint解释，因此本轮只记录，不擅自修补后称“原版AdaIR”。这也是暂缓它的主要原因，而不只是笼统的NPU算子担忧。

## 6. 预训练、随机初始化与三组公平性

优先考虑可验证的去模糊预训练主体，原因是可能减少学习低层图像表征的成本；运动模糊与湍流几何扭曲、空间变化模糊仍有差异，收益是待验证假设。随机初始化更直接考察V5自身训练，来源清楚，但可能需要更多数据与预算。不能先断言哪种一定更好。

| 组别 | 应记录的初始化和比较边界 |
|---|---|
| D0 | 单步直接双输出基线；在当前V5约定下保留clean与重退化监督，不悄悄换成clean-only基线 |
| R0 | 与D0从**完全相同的适配后U0**分叉：同一主体权重/随机state、同一新clean/time/raw参数。训练从G4开始，在线内部时间、共享参数Full-BPTT |
| R0-D0Init | 从该D0预先规定节点继承；明确主体/clean/time/raw各部分、优化器是否重置。记录D0阶段和递归阶段各自及累计成本 |

同一比较中不能D0用预训练、R0随机，再把差异解释为递归收益。D0只见t0=0，其时间条件尚不代表学会全路径；即使D0的raw头有重退化监督，也不能把它叫作已恢复真实生成ξ。R0-D0Init额外付出了D0训练成本，仅对齐第二阶段步数不构成等总预算比较。主比较应固定source划分、观测、增广与评估；同时报告数据暴露和计算成本。预训练vs随机可作为独立初始化轴，当前不替用户扩大实验矩阵或冻结预算。

## 7. 选型后再执行的最小步骤

先核验所选官方权重字节、哈希、键名和兼容主体加载；随后做关键算子CPU/NPU前反向对照，再接RGB3+t双头，验证末步loss到首步clean/raw的梯度连通。最后才测256/512、B1四步Full-BPTT的实际峰值与时间，决定精度/batch，再讨论正式数据和训练。当前只提交上述建议，**等待用户确认Backbone后进入下一阶段**。

完整证据与复算脚本：[Concertormer](<../../NOT_INCLUDED.md#ref-5d6b225317ef8bf476d0>)、[BioIR](<../../NOT_INCLUDED.md#ref-fe64300b4e09f72a7799>)、[AdaIR](<../../NOT_INCLUDED.md#ref-ecbc894dec7ada75e457>)。源码分别固定于 `94269c556277805051644343e3a2cc76c4cd7aaf`、`ff61e477ecf64db2731ef080a1e586f3f9d4248d`、`ccb8b98e49614e07badd0641e5163fa7635c2f02`；本目录保留源码、论文、参数计数和公开入口元数据，没有模型权重。

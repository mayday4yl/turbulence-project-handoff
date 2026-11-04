# V6 像素空间条件扩散选型：原生 IR 与预训练先验适配（不限定桥）

日期：2026-11-01，Asia/Shanghai。此次按用户最新要求取消桥优先。桥是可选轨迹家族，条件图像如何注入是另一条轴，不能把“有条件”与“必须是桥”等同。本轮不训练、不恢复A/B、不运行候选模型、不下载完整权重或数据。

## 推荐及其依据

**严格像素扩散先看 DA-CLIP 与 Resfusion；WeatherDiffusion保留为常规条件DDPM参照；若接受conditional flow，PixRestore-S另列低调用候选；ADM256＋新ControlNet分支保留为工程路线。** 这是源码与资产可核程度、条件机制、采样成本和迁移工程的选型，不是湍流质量排名。所有新候选均未strict-load，也没有256/NPU/梯度/质量验收。

1. **DA-CLIP / Universal-IR（ICLR2024）：机制贴合度最高。** 已有随输入图像生成的内容、退化两种条件，无需额外输入任务文本。RGB扩散主状态保留256，官方编码器和恢复器文件名、大小、LFS SHA可核。主要代价是100次UNet，另有每图一次编码调用、内部两条视觉网络；全局退化向量不等于稠密位移场。对我们的模糊＋warp是否有益仍需同样本验证。[官方代码](https://github.com/Algolzw/daclip-uir)、[官方权重](https://huggingface.co/weblzw/daclip-uir-ViT-B-32-irsde)。
2. **Resfusion（NeurIPS2024）：低步数的经典条件扩散候选。** 默认LinearPro,T12由实际生成循环导出5次UNet，目标为resnoise MSE，使用RDDM式UNet。它改变前向过程／目标，不代表独立全新骨干。官方ckpts.zip存在，但必须先确认无mask任务权重与其调度；不能把默认5NFE直接认定为每个包内权重的已验证配置。[官方代码与权重](https://github.com/nkicsl/Resfusion)。
3. **WeatherDiffusion（TPAMI2023）：纯条件噪声回归基线证据清楚。** 原生RGB LQ拼接、标准噪声预测和DDIM，官方64/128图块权重HEAD均200。25步每步需要多个重叠图块，不能称25次整图前向；局部图块对大幅几何变形的影响是待验证风险。[官方代码](https://github.com/IGITUGraz/WeatherDiffusion)。
4. **PixRestore-S（2026预印本）：独立DiT／低调用扩展。** 原生RGB conditional flow，公开的是GAN单步EMA；1次DiT之外还需DINO条件特征。主RGB256仅静态支持，DINO独立预处理448，不会把当前缓存和GT改512。若要求经典DDPM，应单列，不把它与纯ResidualL1混成原版复现。[官方代码](https://github.com/csslc/PixRestore)、[官方权重](https://huggingface.co/VCLab-PolyU/PixRestore)。
5. **ADM256＋ControlNet式图像支路：保留为更重的研发方向。** 官方像素生成权重HEAD200，约2.211GB；实际没有建立可直接加载的通用LQ-ControlNet恢复权重。需新增多尺度条件支路，不能搬SD的latent ControlNet权重。官方ADM示例250生成步不是本项目的低步数IR结论。[ADM](https://github.com/openai/guided-diffusion)、[ControlNet](https://github.com/lllyasviel/ControlNet)。

## 哪些是真正变化，哪些只是初始化对照

**FoundIR-v1不能当独立新骨干。** 官方固定源码AST比较：29个共同顶层定义中28个一致，唯一不同是Trainer；Unet、UnetRes、ResidualDiffusion、gen_coefficients完全相同。发布主体同dim64×(1,2,4,8)、pred_res、sum_scale.01；FoundIR测试4步、delta_end.0014，官方DiffUIR测试3步、delta_end.0018。不同大规模真实数据预训练可能有价值，但只能称同架构权重初始化对照，不是新扩散机制。[原始AST证据](<../../../../NOT_INCLUDED.md#ref-0c3a4b3d23612c87e8c3>)。

Resfusion主要改变扩散路径、目标与调度，仍使用RDDM式UNet；DA-CLIP引入原生内容／退化条件和SDE恢复器；PixRestore的DiT／flow则属于更明显的模型家族变化。UDBM和I²SB仍可平行比较；桥标签不赋予额外优先级，取消桥优先也不等于排除桥。

## 条件注入：实际送入网络的是什么

| 模式 | 已核实例 | 输入与注入位置 | 对湍流的局限 |
|---|---|---|---|
| RGB空间拼接 | WeatherDiffusion、Palette、AT-DDPM | 每步将LQ与noisy RGB在通道维拼接 | condition存在不等于能估计位置变化；需从配对数据学习 |
| 空间LQ＋随图全局退化／内容条件 | DA-CLIP | 每步[xt−LQ,LQ]；退化向量经learned prompt加到time embedding，内容单token经cross-attention | 两个B×512向量不是dense warp；不是用户输入的文字prompt |
| 图像特征引导 | SRDiff、PixRestore | RRDB局部特征／DINO dense特征注入恢复器 | SR接口可能不匹配；DINO原生预处理和梯度都需核 |
| 多尺度零卷积侧支 | 提议ADM-ControlNet | LQhint与noisy state/time进入复制的encoder/middle，再将zero-conv残差注入主干 | 新侧支需适配训练；现有SD权重不兼容 |
| 测量算子引导 | DiffPIR、DDRM、DPS | 已知H及数据一致性／投影／测量梯度 | 有Simulator不等于真实推理时知道ξ、PSF、warp |

DA-CLIP在100步之前只调用一次encode_image(control=True)，内部是visual_control和受控clip.visual两次视觉网络前向，得到内容、退化各B×512全局向量。退化变量名text_context不是推理文本输入；内容unsqueeze成单个token。LQ的空间拼接始终保留。因此可学习condition特征不等于采用有损VAE扩散latent。重训DA-CLIP退化controller还涉及原文caption/退化文本监督；不能默认当前配对RGB就能完整复现controller训练。

## 采样成本统一口径

下表均为源码／配置导出的调用数，不是本机延迟、FLOPs、显存或湍流质量实测；不同分辨率、网络大小和图块batch不能按NFE直接排名。若未来做四次物理递归，100内步可能产生400次UNet，再加条件编码和反向成本，不能默认沿用当前3步速度。

| 候选 | 核心调用 | 另计成本／边界 |
|---|---|---|
| 当前DiffUIR D0 | 3 UNet | 现有参考，不是本轮新跑 |
| DA-CLIP / IR-SDE | 100 UNet | DA-CLIP另1次编码调用，内部2视觉网；plain IR-SDE无该编码器 |
| Resfusion | 默认5 UNet | T12是完整调度长度；包内每个CP配置未验 |
| WeatherDiff64 /256² /stride16 /25steps | 4,225个patch-example求值 | 整图batch1、patch batch64时75次批量调用；不是4,225个整图NFE，也不是25整图NFE |
| Palette | 原文1000 UNet | 第三方sample_num8只影响快照保存 |
| SR3 | 原文推理预算≤100 | 2000训练区间与第三方2000调用都须分开；级联分别计 |
| SRDiff | 100 denoiser | 另1次RRDB；原生SR接口 |
| UCDIR | 当前SID入口200 denoiser | 另1次guidance；README50不覆盖当前入口 |
| RDDM | 已核10step×2UNet=20 | 只针对所核配置 |
| FoundIR-v1 | 4 UNet | 同架构权重对照 |
| PixRestore-S | 1 DiT | 另DINO；公开阶段含GAN，不是经典DDPM |
| ADM＋新ControlNet | 恢复管线未实现 | ADM官方生成示例250步，每步还需新侧支；不承诺3步效果 |
| DifFace | 默认512人脸100 denoiser | 另1次SwinIR；非通用256湍流checkpoint |
| AT-DDPM | 发布默认60 denoiser | 仅学生推理；EMA弱图教师训练专用 |

ADM256按官方构造器作纯整数计数得552,814,086参数；复制encoder/time/middle为210,904,064，按每个skip/middle加zero projection为8,334,080，合计约772.05M＋hint encoder。它有18个input blocks＋middle共19注入点，SD原ControlNet是12＋middle共13，latent4ch、hint下采样8倍也不兼容。该构造只是工程量估计，没有实现或实例化模型。[算术证据](<../../../../NOT_INCLUDED.md#ref-36307edd77e941ad690d>)。

## 官方代码／预训练资产完整对照

权重“入口”“HEAD成功”“LFS/Release元数据”和“完整payload SHA／strict load”是不同等级。下表没有任何候选通过最后一级。

| 模型 | 分类／定位 | 官方资产现状 | 核心调用＋辅助开销 |
|---|---|---|---|
| DA-CLIP / Universal-IR | RGB conditional IR-SDE; OU/bridge-related, not forced binary | Official HF: universal-ir.pth195990371B + daclip_ViT-B-32.pt1744725787B; paired names/LFS SHA known, no payload | 100 UNet calls；1 encode_image call per image, internally2 visual-network forwards; context reused across100 steps |
| Resfusion | RGB conditional resnoise DDPM; RDDM-style UNet | Official ckpts.zip1584356819B release asset; internal task configs not inspected | Default LinearPro,T12 -> T_acc5 ->5 UNet calls；No separate semantic encoder in inspected path |
| WeatherDiffusion | Conventional paired RGB conditional DDPM/DDIM | Official WeatherDiff64/128 HEAD200:1327834207/1370274225B, payload0 | Documented25 DDIM steps;256²/patch64/stride16 =>4225 patch-example denoiser evaluations；75 batched model invocations for image batch1, patch batch64; overlaps are not25 whole-image NFE |
| IR-SDE | Mean-reverting RGB OU/SDE, bridge-related connections | Official Drive link; exact selected file inventory not verified | 100 UNet calls in inspected config；No DA-CLIP encoder |
| UCDIR | RGB residual conditional diffusion plus guidance network | Official Baidu link; specific weight inventory not verified | Current public SID entry200 denoiser calls；1 initial guidance network; README50 and unused DDIM code are not executed200->50 proof |
| Palette | Conventional conditional RGB DDPM | No verified author checkpoint release; third-party CelebA-HQ/Places2 inpainting weights only | Original1000 refinement calls; third-party Places2 test1000 too；sample_num8 in third-party code means saved snapshots, not8NFE |
| SR3 | Conventional pixel conditional SR diffusion | Third-party face-SR links only inspected; not official general restoration | Original paper inference budget<=100;2000 is training grid；Unofficial shipped configs use2000 actual calls; cascades counted separately |
| SRDiff | RGB residual diffusion for super-resolution | Official release ZIPs132578889B CelebA /135424285B DIV2K; no archive inspection | 100 denoiser calls；1 RRDB feature extraction |
| FoundIR-v1 | Same core model implementation as DiffUIR; different pretraining | Official GitHub release model-2000.pt580829530B; payload/strict-load0 | Official real-world test4 UNet calls；No extra semantic encoder |
| RDDM | RGB residual/noise diffusion; close DiffUIR family | Official task-weight links, file inventory/strict-load not verified | Inspected10steps x2UNets =20 forwards；No semantic encoder |
| PixRestore-S | RGB conditional flow matching; released one-step GAN extension | Official HF GAN EMA128709340B + separate DINO; LFS SHA known, native multistep teacher not located in inspected HF tree | Released config1 DiT forward；DINO extraction; mainRGB256 statically supported, DINO preprocess448 |
| ADM256 + proposed ControlNet-style branch | Pretrained RGB generation prior plus NEW restoration condition adapter | Official256uncond.pt HEAD200,2211383297B; no ready LQ-ControlNet weight established | No IR sampler implemented; official generation example250 denoiser calls；Control side branch per step; architecture-derived552814086 base +219238144 branch +hint parameters |
| DifFace | Public pixel prior + degraded-image estimator, not ControlNet | Official release prior639362581B + estimator90204361B; main512 face path | Default512face path100 diffusion calls；1 SwinIR estimate; public256 pipeline is face inpainting |
| DiffPIR | Pretrained pixel prior + operator/data-fidelity guidance | Official ADM-related model links; operator-specific restoration code | Paper/README reports <=100 NFE; inspected deblur iter_num100/iter_num_U1; actual model-call count not instrumented；Operator solve/projection cost must also be counted |
| DDRM | Pixel generative prior + known linear inverse operator | Official model links and test code; no payload fetched | Official paper/config20 steps；SVD/operator cost |
| DPS | Pixel generative prior + measurement gradient | Official public code and prior paths; no full model load | Inspected default1000 denoiser steps；Per-step measurement gradient/backward cost, not1000 forwards alone |
| AT-DDPM | Conventional RGB conditional DDPM for turbulence-degraded faces | Official Dropbox link only; actual folder files/access not verified | Released100point respacing minus skip40 =>60 denoiser calls；1 inference denoiser, no teacher ensemble |
| TSTRNet | Turbulence paper: SSM coarse stage + conditional DDPM refinement | No verified official checkpoint/code in scoped search | NFE not established；First stage + refinement details unverified |
| UDBM | Optional RGB conditional bridge; bridge not required | Official two-stage weight link; exact payload/strict-load not verified | Default1 bridge-denoiser call；1 auxiliary uncertainty network plus native random/clamp/abs protocol |
| I2SB | Optional RGB Schrodinger bridge | Official9 task-checkpoint download script; no payload/strict-load this research | Configurable NFE; no verified3step turbulence performance；Large ADM-family denoiser; sampler detach/no_grad requires future audit |

## 未进入首选的原因与科学边界

- Palette、SR3的常见Janspiry实现明确为非官方；不能把这些实现和检查点写成作者发布。SRDiff虽有作者代码／权重目录，但SR不是同尺寸湍流退化。
- AT-DDPM确有单帧湍流研究和RGB条件代码，但采用人脸先验／渐进蒸馏，发布权重链接内容未核；不能把人脸实验推成自然场景泛化。原文与发布代码的起始索引细节也需后续数值核验。
- TSTRNet的作者页与摘要可确认任务相关，但本轮未建立可部署官方代码／权重，不把论文宣称当可运行结果；可继续保留为线索。
- DifFace证明像素生成先验适配确实有公开实现，但默认是512人脸先验＋SwinIR；256公开管线是人脸inpainting。不能称现成通用场景ControlNet。
- DiffPIR/DDRM/DPS是测量算子引导；DDRM的已知线性SVD约束、DPS的已知可微测量模型与真实未知湍流算子不同。不能使用GT ξ作为部署条件。
- Edify有pixel ControlNet的论文先例，但这次没有建立完整公开部署资产；Iris的3B／原生4×1024tile恢复协议不适合作当前256低成本默认。所核DiffBIR、ResShift发布latent路径和IRBridge的SD/VAE实验不满足硬条件；DiffIR的compact prior diffusion也不是RGB状态扩散。
- 原始目标、调度和预处理优先保留。不同模型不能机械套用旧SDT、旧Residual L1或DiffUIR的Adam。图像condition是否有效须用固定样本验证，不将“有prompt”“非桥”“更新”当质量证据。
- 所有采样器／条件编码器的no_grad、detach、PIL/uint8边界都需要未来独立梯度验收；冻结参数不等于允许切断对输入的梯度。不能以像素空间自动推出Full-BPTT可行。

## 后续技术验收建议，尚未执行

先按研究候选获取精确配套权重并strict完整加载；再原生256形状、有限值、完整seed/调用数和固定Val小样本质量、显存／端到端时间。随后才核条件分支和采样的输入梯度。若要训练，另立所选模型原生目标的有限协议，本次没有训练授权执行。权重若后续需要获取，只通过服务器网络，不经本地热点。

本轮保存三组独立源码／小元数据清单、FoundIR AST对照、Resfusion调度算术、Weather图块控制流、ADM参数算术、原始检索和失败记录。既有模型、缓存、Test1000、冻结结果和A/B保持原状；旧版研究不覆盖。

# 像素空间、非恢复残差、图像条件扩散：顶会新增核查

核查日期：2026-11-01。仅文献、官方源码及权重目录元数据；没有下载模型权重载荷、创建模型、推理或训练。旧两版选型报告保持原样。

**有，前面的搜索遗漏了几篇，不能说只剩 Palette、WeatherDiffusion 和 SR3。** 本轮新增核实 GSAD、PyDiff、ShadowDiffusion、IDM-SR；还找到论文级 IDM-Face，以及扩散反射/照明分量的 Diff-Retinex。前四项均有作者代码，但“有权重链接”不代表当前权重已获取、完整加载或能在 NPU 上运行。

## 采用的判据

本轮“非残差”特指不以恢复残差 Y−C（或 C−Y）为扩散状态/预测目标。对 clean RGB 加高斯噪声后预测 ε 属于非恢复残差扩散；骨干内 ResBlock、skip connection 不影响这一分类。RGB 空间的金字塔缩放也不等于 VAE/VQ latent。反射/照明图、RAW、无条件先验推理引导和两阶段预恢复条件分别列出，不混作直接 clean RGB 条件模型。

## 新增正式顶会候选

| 方法 | 正式发表 | 扩散状态与图像条件 | 公开入口的成本证据 | 作者资产与主要限制 |
|---|---|---|---|---|
| **GSAD** | **NeurIPS 2023 主会** | 完整 clean RGB3 加高斯噪声；LQ3 与 noisy RGB3 拼成6通道，预测 ε | **20次 denoiser**，来自 LOLv1 测试配置与循环 | 作者代码、三种 LOL 权重名称和官方 Drive 入口；未下载/strict load。原训练有结构与不确定性正则，不是纯基础 ε 损失 |
| **PyDiff** | **IJCAI 2023 主会** | clean RGB 在不同空间分辨率上扩散；LQ、直方图均衡图、位置编码一起作为条件，输入13通道 | **4次 denoiser＋3次 GlobalCorrector**，官方默认配置静态推导 | 作者代码、LOLweights.pth 入口；未下载/strict load。额外条件预处理、金字塔和校正器增加迁移工作 |
| **ShadowDiffusion** | **CVPR 2023 主会** | clean RGB3 扩散；input RGB3＋shadow mask1＋noisy RGB3，预测 ε并更新mask | **5次 denoiser**；完整系统入口的额外调用数未验收 | 作者代码/权重链接。依赖初始阴影mask；另有 shadow transmission 路径，不能直接作为任意湍流条件 |
| **IDM-SR** | **CVPR 2023 主会** | 直接对 HR RGB3 加噪，预测 ε；EDSR 编码 LR 条件并有scale调制，无VAE/VQ | 默认 **2000次**；可选DDIM分支静态实际 **199次** | 作者代码。README声明8×face权重，但本轮公开Drive只返回空列表，未取得可用文件库存；超分辨率接口与高采样成本是限制 |
| **IDM-Face** | **ICCV 2023 主会** | 论文为完整clean RGB条件DDPM，拼接degraded RGB，intrinsic训练预测 x0 | 论文报告 **10步**，没有官方循环或实测NFE验收 | 本轮有界搜索未找到作者代码/权重；人脸任务，论文级备选 |

所有次数均是特定公开配置的静态源码/论文结论，不能当作运行延迟或 NPU 兼容性测试。尤其 **PyDiff 的4步不等于全系统只跑4次网络**；IDM-Face的10步只有论文证据。

一手会议/作者依据：[GSAD NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc034d186280f55370b6aca7a3285a65-Abstract-Conference.html)、[PyDiff IJCAI](https://ijcai.org/proceedings/2023/199)、[ShadowDiffusion CVPR作者代码](https://github.com/GuoLanqing/ShadowDiffusion)、[IDM-SR CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_Implicit_Diffusion_Models_for_Continuous_Super-Resolution_CVPR_2023_paper.html)、[IDM-Face 作者发布页](https://research.google/pubs/towards-authentic-face-restoration-with-iterative-diffusion-models-and-beyond/)。

## 两项不能省略的评测差异

GSAD原生test.py用GT灰度均值与输出灰度均值的比值校正恢复图，然后算指标。PyDiff infer.yaml开启use_kind_align，验证时用GT全RGB均值校正uint8输出后保存、计分；README明确自有数据应关闭。**两者都不能将这种GT辅助后处理结果当成真实部署输出。** 这不是GT输入噪声网络，但会影响质量数字与视觉。今后若获授权评测，应保留raw结果并将原论文对齐口径单列；本轮没有修改代码或执行评测。

ShadowDiffusion活跃SR3代码确实对HR RGB加噪，不对恢复残差扩散；但其辅助mask标签由HR−SR的均值阈值生成。不能把“非残差扩散”扩大成“完全不使用图像差分”。此外README命令与已抓infer.py入口存在差别：infer.py只调用test()，test_d(h_hat)存在但不能据config宣称一定启用。因此只确认5次噪声网络循环，不编造完整系统5+1次调用。

## 更贴近湍流与其他边界

- **PiRN-SR，ICCV 2023**：论文clean RGB扩散，以PiRN初步恢复图为条件，而非原始退化图。公开默认路径却是恢复RGB加噪暖启动、每步model_kwargs=None、3输入通道，未核实论文显式图像条件的可复现实现。论文15 DDIM与公开默认10 denoiser＋1预恢复不能混写。文件夹脚本还平均后续图像并缩放512；不能直接冒充严格单帧。它是重要任务参考，但不应作为“已就绪的直接Y条件模型”推荐。
- **Diff-Retinex，ICCV 2023**：无VAE/VQ、无恢复残差扩散，但状态为RGB反射图＋1通道照明图，不是单分支clean RGB。官方三模块文件元数据已找到，合计236,231,423字节，载荷下载为0。现入口两支各5步，总10 denoiser＋10 correction＋1 TDN，不能只写5NFE。
- **AT-DDPM，WACV 2023**：代码确实有clean RGB高斯扩散＋每步退化图拼接，但人脸限定，WACV不混入CVPR/ICCV/ECCV主榜。
- **DR2 / GDP（CVPR 2023）、PGDiff（NeurIPS 2023）、BlindDPS（CVPR 2023）**：无条件先验加推理约束/梯度引导，与训练好的退化图条件网络分列。
- **DGSolver（NeurIPS 2025）**明确估计恢复残差，不能为了补一篇新顶会而加入本轮严格主榜。
- **ExposureDiffusion（ICCV 2023）**是RAW物理曝光噪声过程；**DiffTSR（CVPR 2024）**使用AutoencoderKL，均不满足当前严格RGB条件。

## 对当前湍流项目的建议

**下一步优先深入GSAD和PyDiff的迁移可行性，而不是宣布必须继续用残差模型。** GSAD的6通道条件接口更直接、来源与作者资产入口明确；PyDiff采样成本低，但有额外corrector、直方图/位置条件与金字塔接口，不能视作无改动接入。这里是工程与机制优先级，不是实测湍流性能排名。

ShadowDiffusion更适合借鉴条件注入/辅助退化估计的设计，不能凭去阴影成绩推断几何扭曲恢复。IDM-SR方法是真匹配，但公开权重库存和采样成本不利于当前马上使用。IDM-Face仅用于阅读；PiRN-SR需先解决论文与公开默认代码路径的差异。

这些候选本身的结构/不确定性、global correction、mask监督等原生设计不能被无声替换为当前D0 Residual L1。选择“非残差clean RGB扩散”本身就涉及训练目标与状态定义的决定；本轮只调研，不作任何方法修改或启动实验。

## 文件与可追溯性

- CANDIDATES.csv / CANDIDATES.json：7项新增/边界候选，条件、采样计数、权重状态、源码固定提交和局限。
- rgb_lowlight_shadow/：GSAD/PyDiff/ShadowDiffusion的17份固定源码、PROVENANCE、逐项SHA及精确Git blob核验；PyDiff采样算术与独立复核结论。
- native_models/：IDM-SR、IDM-Face、Diff-Retinex、ExposureDiffusion、DiffTSR的论文/代码/权重metadata依据。
- turbulence_sr/：PiRN-SR/AT-DDPM源码及论文差异核查。
- prior_and_condition/：DR2/GDP/DGSolver/PGDiff等排除依据与查询失败记录。
- RESEARCH_ACCEPTANCE.json / DELIVERY_SHA256.json：纯调研交付完整性，不代表权重或运行已通过验收。

这是有界扩充，不是全领域穷尽性证明。低光/阴影/超分权重能否恢复湍流尚无本轮证据；0模型执行、0训练、0数据/权重载荷下载。


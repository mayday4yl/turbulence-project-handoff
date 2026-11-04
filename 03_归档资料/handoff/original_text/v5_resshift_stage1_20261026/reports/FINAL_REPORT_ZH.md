# ResShift Stage 1：官方基线审查与最小推理验收

日期：2026-10-26，北京时间。状态：**Stage 1 完成，停止于原版推理验证；未进入 Stage 2 / Training。**

官方 `journal` 的去模糊模型和两个官方 checkpoint 已在单张 Ascend 910B3 上完成真实推理。直接运行官方 CUDA 入口会失败；通过华为 `torch_npu.contrib.transfer_to_npu` 做设备调用适配后，未改动上游科学源码、配置、Forward / Reverse、Condition、Backbone 或 Loss，官方整图入口成功。随后完成固定 V5 Val 的 4 张 G4 图像 Zero-shot 推理及 4 张真实 Clean 的 VQ 编解码重建，权重加载通过逐张量精确核对。

这说明所测推理路径可在当前 NPU 环境执行，**不等于复现论文的 GoPro 基准数值，也不等于湍流恢复有效**。四张湍流图的平均 PSNR 从输入 17.7892 dB 降到 16.9712 dB；没有证据据此宣称比现有 V5 更好。四张均属 category 254、来自四个不同 source，只是固定小样本运行检查。

## 1. 代码、环境及隔离边界

| 项目 | 本次固定身份 |
|---|---|
| 上游 | [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift)，`journal` |
| commit | `bb03b7d21614cace01787e097c8a6ab6b945227d` |
| 配置 | `configs/deblur_gopro256.yaml`；SHA256 `5c98de721db21627b4d1fa35957338c4eb4a69823a55575c2ab380f8c4770dec` |
| 本地独立根 | `/Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026` |
| 远端独立根 | `/data2/yanglin/v5_resshift_stage1_20261026`，`liteserver-de36` |
| 实测设备 | Ascend 910B3，物理卡 2；`ASCEND_RT_VISIBLE_DEVICES=2`，进程内只有 `npu:0` |
| OS / Python | Linux aarch64，kernel 5.15.0-91-generic；Python 3.10.12 |
| 框架 | torch `2.7.1+cpu` + torch_npu `2.7.1.post2`；torchvision `0.22.1` |
| CANN / 驱动 | CANN `8.5.2`；驱动 `25.5.1` |
| 实测推理精度 | FP32 权重，FP16 autocast；conv / matmul HF32 均关闭；请求关闭 internal format |
| 主要依赖 | numpy 1.26.4、scipy 1.13.1、Pillow 10.4.0、einops 0.8.1、omegaconf 2.3.0、timm 0.6.13、scikit-image 0.24.0、albumentations 1.3.1、opencv-python-headless 4.11.0.86 |

官方 README 指向 torch 2.1.2，而 requirements.txt 固定 torch 2.1.1 / xformers 0.0.23；本次没有伪称完全复刻该 CUDA 环境。xformers 未安装，执行的是上游已提供的 vanilla attention 路径。该路径运行成功，但没有做 CUDA/xformers 与 NPU 的数值一致性比较。

独立 `env/` 通过 `.pth` 只读复用已有 `/data0/yanglin/envs/physical_bridge/lib/python3.10/site-packages` 的 torch/NPU 二进制；新增依赖安装在新 env 内。它是共享底层运行库的隔离依赖目录，不是完全封闭的容器。没有向基础环境安装或升级依赖，没有修改驱动或 CANN。新目录内保存各类缓存；启动脚本先 source CANN，并保留其 PYTHONPATH。完整版本见 [pip_freeze.txt](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/pip_freeze.txt>) 和 [CANN_RUNTIME.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/CANN_RUNTIME.json>)。

初查物理卡 2–7 均没有计算进程；本阶段仅使用卡 2，未启动六卡任务。结束快照确认卡 2 已无运行进程。旧 V5 保护范围内的 74 个代码/协议/配置/日志等文件 SHA256、大小和 mtime 均未变；32 个 checkpoint 的大小、mtime 和 inode 均未变，**未重新读取全部 checkpoint 计算哈希**。前后证据见 [ISOLATION_FINAL.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/ISOLATION_FINAL.json>)。

官方仓库跟踪了 `.pyc`：早期导入自动重建了其中 8 个字节码缓存，科学源码和配置未变。已保存自动生成的字节码，再恢复其官方原始字节；后续设置 `PYTHONDONTWRITEBYTECODE=1`。最终远端全部 577 个跟踪文件与固定提交清单逐一一致，本地 git 工作树干净。原始差异和恢复记录保留在 `ISOLATION_COMPARISON.json`、`BYTECODE_RESTORE.json`，没有将中途差异隐藏为从未发生。

## 2. 模型与 Condition：当前实现事实

模型是 **带 Swin 模块的 UNetModelSwin，在 VQ f4 的连续 latent 空间执行 4 步 residual-shifting diffusion**。不是像素空间扩散，也不是纯 DiT。UNet 基础通道 160、通道倍率 `[1,2,2,4]`、每级 2 个 ResBlock，Swin depth 2 / embed dim 192 / window 8；模型输出 3 通道 clean latent。VQ codebook 为 8192×3，256×256 RGB 编码成 3×64×64。

设 clean / degraded RGB 经 `2*x−1` 归一化后为 `C / Y`，连续编码为 `z_C=E(C)`、`z_Y=E(Y)`。本配置 `scale_factor=1`，`sf=1`。`lq` 有两条同时存在的路径：

```text
RGB Y [B,3,256,256]
 ├─ VQ encoder → quant_conv → z_Y [B,3,64,64]
 │    └─ 决定前向桥均值和推理初始随机状态
 └─ UNet.feature_extractor（卷积及两次下采样）
      → condition feature [B,64,64,64]
      → 与缩放后的 noisy latent [B,3,64,64] 拼接
      → UNet 首层实际输入 67 通道 → predicted clean latent
```

原 `lq` 是条件 RGB，`x_t/z_k` 是不断更新的随机 latent，二者不能互换。条件 CNN **接收归一化 RGB，而不是直接接收 `E(lq)`**。本次执行 hook 实测四步均为上述 condition shape；采样索引实测 `[3,2,1,0]`。整数扩散索引经 160 维 sin/cos、MLP 成为 640 维时间 embedding，通过 ResBlock 的 scale/shift 注入；它不表示 V5 的物理时间 `t_phys=1−g/4`。

前向边缘分布及本配置预测目标：

\[
z_k=(1-\eta_k)z_C+\eta_k z_Y+\kappa\sqrt{\eta_k}\epsilon,
\quad\epsilon\sim\mathcal N(0,I),\quad\kappa=2,
\qquad f_\theta(z_k,k,Y)\rightarrow\hat z_C.
\]

模型实际收到的 noisy latent 还除以 `sqrt(1+κ²η_k)`。当前 target 为 `xstart`，即连续 clean latent，不是 epsilon。配置的 `etas_end=0.99` 实际用于 sqrt_eta 末值，因此 eta 末值为 0.9801；首值为 0.01。推理从 `z_Y+κ*sqrt(eta_last)*noise` 开始，这是近似终端先验，不与严格前向终端均值完全相等。

令 `alpha_k=eta_k−eta_prev,k`，逆过程使用预测 clean 替代真实 clean，均值和方差为：

\[
\mu_\theta=\frac{\eta_{prev,k}}{\eta_k}z_k+
\frac{\alpha_k}{\eta_k}\hat z_C,
\qquad\sigma_k^2=\kappa^2\frac{\eta_{prev,k}}{\eta_k}\alpha_k.
\]

逆序执行四次；`k>0` 加后验噪声，`k=0` 不再加噪声。最后的 clean latent 经 **codebook 最近邻量化 → post_quant_conv → VQ Decoder → RGB**，官方 sample_func 将最终 RGB clamp 到 `[-1,1]`，再映射回 `[0,1]`。编码端 `encoder→quant_conv` 不量化，默认解码端会量化；不能把 VQ 当作完全连续且无损的恒等变换。

当前提交训练时每样本随机抽取一个扩散索引，而非四步 Full-BPTT。正常有限值分支的 loss 是 latent MSE + 4×LPIPS，LPIPS 对照为重建 GT `D(E(C))`；该最新源码行为不能证明较早发布权重已按这项修改重训。默认配置冻结 VQ；量化反向采用 STE。完整公式、异常分支、梯度边界和固定提交链接见 [模型与 Condition 专项审查](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/reports/MODEL_CONDITION_AUDIT_ZH.md>)。

| 源码入口 | 当前职责 |
|---|---|
| [inference_resshift.py:124](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/inference_resshift.py:124>) | 选择 deblur 配置与两个官方权重；创建 sampler 并调用 inference |
| [sampler.py:212](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/sampler.py:212>) | 图像读取/归一化、分块、合并及保存 |
| [sampler.py:164](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/sampler.py:164>) | sample_func 构建 lq kwargs、采样、最后 RGB clamp；包装调用显式 mask=None |
| [gaussian_diffusion.py:420](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/models/gaussian_diffusion.py:420>) | encode lq → prior → 四次 p_sample / p_mean_variance → predicted clean latent |
| [unet.py:682](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/models/unet.py:682>)、[unet.py:865](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/models/unet.py:865>) | 条件 CNN、拼接、扩散时间 embedding、UNet 前向 |
| [autoencoder.py:41](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/ldm/models/autoencoder.py:41>) | 连续 encode 与默认量化 decode |
| [trainer.py:741](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/trainer.py:741>)、[trainer.py:974](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/upstream/trainer.py:974>) | 随机采样扩散索引、传递原 RGB lq、latent MSE 与 LPIPS；仅静态审查，未训练 |

## 3. 官方权重实际获取与加载验收

使用官方入口 `_LINK` 给出的 GitHub v2.0 release，HTTPS 下载后上传新远端目录；本地/远端 SHA256 一致。release API 未提供 digest，因此下列 SHA 是本次下载及传输完整性指纹，并非作者额外签发的校验值。

| 文件及官方地址 | 实测字节数 | 文件结构 | 实际加载 |
|---|---:|---|---|
| [resshift_deblur_s4.pth](https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_deblur_s4.pth) | 610,280,984 | dict，顶层 `state_dict`、`vae` | `state_dict` 的 572 个 tensor；UNet 118,714,255 参数 |
| [autoencoder_vq_f4.pth](https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth) | 221,364,711 | 原始 OrderedDict | 205 个 tensor；VQ 55,322,782 参数 |

Deblur SHA256：`7b2bb6ddc9c29dfe845028ce32fa146081caf7647f772113eb26502b92dec515`。

VQ SHA256：`e967eaa2c380152811c53e312990d6967a8a957c4f7590f19490ec72110b2889`。

当前明确使用 `deblur_gopro256.yaml` 的 `tune_decoder=False`，所以 deblur 文件额外的 `vae` 项不进入模型；VQ 来自独立官方 autoencoder 文件。这与官方配置及 sampler 的实际加载路径一致，没有擅自改用 tunedecoder 配置。

官方 reload_model 之后，验证脚本独立比较 checkpoint 与已加载模型的键、shape、dtype、每个 tensor 的数值；两个模型的 missing / unexpected / shape_mismatch / value_mismatch / dtype_changes 均为空。加载审计 SHA 再与下载清单交叉核对通过。证据：[CHECKPOINT_AUDIT.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/outputs/val4_fp16/CHECKPOINT_AUDIT.json>)、[WEIGHTS_MANIFEST.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/WEIGHTS_MANIFEST.json>)、[DELIVERY_SUMMARY.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/DELIVERY_SUMMARY.json>)。没有随机初始化冒充预训练或跳过加载后继续计分。

## 4. 原版推理成功范围与环境问题

直接调用原 `inference_resshift.py` 在 `sampler.py:90 .cuda()` 失败：`Torch not compiled with CUDA enabled`。独立包装 [official_cli_npu.py](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/scripts/official_cli_npu.py>) 导入华为设备适配器，把 CUDA API 分发到 NPU，再用 runpy 执行原入口；另加观测 hook 记录次数、有限值、同步耗时与 allocator 内存。不替换模型前向或采样算法。适配器还会禁用 torch.jit.script，故不能宣称原 CUDA/JIT 路径完全一致。

| 实际遇到的问题 | 已完成的最小修复 / 证据 |
|---|---|
| 新 venv 未自动继承基础 venv 的 site-packages；pip 试图重新解析 torch | 停止该安装，使用 `.pth` 只读引入已验证 NPU 运行库；其余依赖安装在新 env，见 `NESTED_VENV_RUNTIME_FIX.json` |
| 未 source CANN 时 libhccl 导入失败；随后启动脚本覆盖 PYTHONPATH 导致 `No module named tbe` | source CANN，并在前置上游目录时保留原 PYTHONPATH；保存 attempt1 日志 |
| `ASCEND_CACHE_PATH` 指向尚未创建的目录 | 启动前 mkdir 新目录内缓存；保存 attempt2 日志 |
| 记录器读取 `torch.npu.config.allow_internal_format` getter 报 AttributeError | 该版本可设置但无对应读取属性；记录 requested=False，不伪称读回确认；保存 attempt3 日志 |
| xformers 不存在 | 使用官方已有 fallback；没有改写 attention 或安装不匹配 CUDA 扩展 |
| 原始 CUDA 设备硬编码 | 进程内使用华为 transfer_to_npu；原入口失败与成功适配分开记录 |

上述三次 NPU 入口尝试均在去噪网络调用前退出，`unet_calls=0`；它们不计为成功推理。最终官方整图运行返回 SUCCESS、exit code 0、28/28 个图块输出有限；固定 Val4 返回 PASS、exit code 0。未遇到阻止**本次推理路径**的 NPU 不支持算子。训练 backward、多卡 HCCL、其他分辨率及 CUDA 数值对齐均未验证。

官方输入为仓库 `testdata/deblur/lq/im_138.png`，原尺寸 **1280×720**。官方 `--chop_size 64` 在 `scale=1` 下转换为实际 256×256 patch、stride 192，产生 28 个图块；没有只跑一个裁剪后声称整图成功。该仓库目录未提供对应 GT，因此不报告官方示例 PSNR / SSIM 或论文复现误差。输入和预测见 [官方整图对照](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/outputs/official_cli/comparison.png>)。

## 5. 固定 Val4 Zero-shot 与真实 Clean 的 VQ 重建

按冻结 sample_manifest 原始顺序选择前 4 个不同 Val source 的第一条 G4 记录，没有看恢复效果再筛选。读取的是原始 FP32 CHW NPY，逐文件 SHA 校验通过；不更换划分、不重生成退化、不把输入转 PNG 量化、不 clamp 输入、不裁剪/缩放。Test1000 无图像读取或评价。完整选取依据见 [V5 数据接口报告](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/reports/V5_DATA_INTERFACE_ZH.md>)。

每张都使用相同 seed 12345，调用原 sample_func、noise_repeat=True、mask=None。指标为完整 RGB、range=1、逐图 PSNR 后平均；SSIM 使用 Gaussian 11×11、sigma 1.5、population covariance。官方输出保留其自带 clamp；另外保存 clamp 前 decoder 数值并计分。PNG 仅用于显示，指标从未量化 NPY 计算。LPIPS 未测：本阶段隔离缓存没有 AlexNet 权重，没有为此另行下载。

| 固定样本 | Input PSNR | Zero-shot PSNR | GT→VQ→RGB PSNR | Input SSIM | Zero-shot SSIM | GT VQ SSIM |
|---|---:|---:|---:|---:|---:|---:|
| val_s0040_r00 | 17.7122 | 16.4947 | 22.3112 | 0.1529 | 0.1299 | 0.7029 |
| val_s0041_r00 | 18.5010 | 17.5481 | 26.9665 | 0.4748 | 0.4417 | 0.8391 |
| val_s0042_r00 | 15.2796 | 15.1794 | 21.4325 | 0.2686 | 0.2662 | 0.7515 |
| val_s0043_r00 | 19.6641 | 18.6624 | 29.8171 | 0.5250 | 0.4960 | 0.8711 |
| **4 来源均值** | **17.7892** | **16.9712** | **25.1318** | **0.3553** | **0.3334** | **0.7912** |

预测 clamp 前均值为 PSNR 16.9709、SSIM 0.3334，和官方 clamp 后的差异很小；本次退化不能归因于输出 clamp。GT VQ 重建使用原 encode/decode、包括默认 codebook 量化、不 clamp 计分；四图 decoder 范围总体为 −0.07422 至 1.11523，显示 PNG 才裁剪到 [0,1]。VQ 重建 PSNR 的跨图范围 21.4325–29.8171 dB，说明编解码不是无损的。

逐图观察：官方示例的运动模糊有明显减轻；湍流预测出现局部边缘增强，但仍保留严重几何错位，也出现块状纹理。GT VQ 重建保留了较多轮廓与纹理，存在细节/色调差异，未出现同等程度的湍流预测失真。这是定性观察，不能仅由四图确定误差因果，**VQ 重建也不是经证明的恢复性能上限**。

![固定 Val4，列顺序：G4 输入、Zero-shot、GT、GT VQ 重建](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/outputs/val4_fp16/ALL4_COMPARISON.png>)

这四张均属同一类别、输入范围恰好在 [0,1]；未覆盖 Val1000 中含越界数值的样本。不要把四图均值与旧 V5 D0/A/B 的完整 Val1000 均值直接比较，也不要将其称为“湍流训练后的性能”。

## 6. NFE、内存与耗时：仅列实测

| 项目 | 实测值及口径 |
|---|---|
| 扩散 NFE | 每个 256×256 输入 / 图块均为 4，索引 3→2→1→0；不把 VQ 编解码计入 NFE |
| 官方 1280×720 整图 | 28 图块 × 4 = **112 次 UNet 调用**；不能把整图计算量写成 4 次调用 |
| 官方入口总耗时 | **30.4252 s**，从 wrapper 开始计时，含 import、建模加载、图像处理和保存，不含外层 shell 启动 |
| 官方图块 sample_func | 首块 1.2736 s；后 27 块平均 0.3747 s、范围 0.3661–0.3891 s；28 块合计 11.3915 s |
| Val4 单图 sample_func | 顺序为 **1.1908 / 0.3966 / 0.4015 / 0.4000 s**；后三张平均 0.3994 s |
| GT VQ encode+decode | 四张分别 0.0230 / 0.0235 / 0.0258 / 0.0223 s，均值 0.02365 s；此时模型和算子已由推理预热 |
| 官方图块 allocator 峰值 | allocated **1220.71 MiB**，reserved **2146 MiB** |
| Val4 推理 allocator 峰值 | allocated **1181.89 MiB**，reserved **2146 MiB** |

采样计时前后均同步 NPU；包含观测 hook，Val4 还包含 decoder 输出 clone 与 seed reset。它不含加载、CPU 指标、传回主机和文件保存。首个调用可能含首次执行/编译开销；其余只是这次顺序观测，未进行重复吞吐测试。单卡上没有本项目并发任务，服务器其他卡仍有他人任务。

内存是 torch_npu allocator 在采样段的峰值，包含当时驻留模型；**不是进程全部 HBM、不是全程加载峰值、不是训练显存需求**。没有持续采样 npu-smi 来获取整进程 HBM 峰值。NPU 包使用官方已有非 xformers 路径，本表不能用于宣称与论文 GPU 等硬件效率。

原始计时和逐次 hook 见 [OFFICIAL_CLI_RESULT.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/OFFICIAL_CLI_RESULT.json>)、[Val4 RESULT.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/outputs/val4_fp16/RESULT.json>)。

## 7. 后续 Condition 接口位置与尚未验证项

仅把同尺寸 RGB 湍流图作为原版 lq 时，已由本阶段独立 NPY adapter 完成接口验证，**无需改变 Condition 网络**。后续训练数据接入的最小位置是 paired dataset 输出 `gt/lq` 与显式归一化，先保留原始 FP32 数值和冻结身份。

若未来要设计自定义 Condition，需分别讨论以下位置；本阶段没有实施任何一个：

| 修改对象 | 最小定位点与必须区分的影响 |
|---|---|
| UNet 可见的条件内容 | `trainer.py` 的 model_kwargs、`sampler.sample_func` 的 model_kwargs、`UNetModelSwin.forward(lq=...)`；训练和推理应一致 |
| 条件通道 / 尺寸 | `UNetModelSwin.__init__` 的 feature_extractor 与首层通道；更改后须单独定义新参数初始化和 checkpoint 加载规则 |
| 前向桥及初始化 | `training_losses(y=...)`、`encode_first_stage`、`prior_sample`；改变这里会改变 Forward / Reverse 的输入身份，不能当作仅换条件特征 |
| 物理时间 | 独立于现有 diffusion timestep embedding；目前没有物理时间分支 |
| 与 V5 Full-BPTT / raw-ξ 结合 | 原采样及 VQ encode 有 no_grad，原 raw 头依赖 Concertormer 特征；不能直接套上现有接口就宣称梯度链保持 |

尚未验证：CUDA/NPU 与 FP32/AMP 数值一致性、原论文 GoPro 全量质量、checkpoint 的原始训练日志及新 loss 对其训练身份的适用性、Val1000/真实湍流、多种随机种子、超出当前尺寸和范围的数据、训练反向与优化器、多卡 HCCL、其他五张获授权卡的模型执行、物理递归与 raw-ξ 接入。LPIPS 仍缺实测。

建议下一次讨论先以“两条 lq 路径”和已有 VQ 重建误差为事实基础确定 Stage 2 的问题与公平对照，再决定是否以及如何设计新条件。本报告不授权或预先实施 Stage 2。原 V5 D0/A/B、物理递归框架和 Test1000 继续保留。

## 8. 复核入口与交付目录

当前权威状态为 [STAGE1_STATUS.json](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/evidence/STAGE1_STATUS.json>)；中途失败日志不覆盖最终 PASS。复现脚本位于新根 `scripts/`，远端使用以下命令；第二条需选择新的输出目录，脚本拒绝覆盖既有结果：

```bash
cd /data2/yanglin/v5_resshift_stage1_20261026
bash scripts/run_env.sh scripts/official_cli_npu.py \
  -i testdata/deblur/lq/im_138.png \
  -o /data2/yanglin/v5_resshift_stage1_20261026/outputs/official_cli_recheck \
  --task deblur --scale 1 --chop_size 64 --seed 12345

bash scripts/run_env.sh scripts/minimal_validate.py \
  --root /data2/yanglin/v5_resshift_stage1_20261026 \
  --device npu:0 \
  --dataset-root /data2/yanglin/v5_concertormer_10k_20261021/data/v5_places_c50_s2000_v250_t250_g4_r4_20261021 \
  --output /data2/yanglin/v5_resshift_stage1_20261026/outputs/val4_fp16_recheck \
  --precision fp16 --seed 12345
```

复现前仍须检查卡 2 是否空闲；本报告命令没有自动提交后续运行。

- `reports/`：本总报告、模型/Condition 审查、V5 数据接口报告。
- `evidence/`：固定提交与哈希、下载清单、依赖版本、原生失败与三次包装失败、最终成功运行、资源释放和隔离性证据。
- `outputs/official_cli/`：完整官方预测和输入/输出对照。
- `outputs/val4_fp16/`：四张各自 input / gt / prediction_official / prediction_raw_decoder / gt_vq_reconstruction_raw 的 NPY、对照 PNG、逐图 METRICS、CHECKPOINT_AUDIT、完整 RESULT，以及四图汇总图。
- `weights/`：两个已核验官方 checkpoint；`upstream/`：固定官方源码；`scripts/`：设备启动、观察/验证与证据复核脚本。

最终交付文件哈希见 `evidence/DELIVERY_MANIFEST.json`。本阶段无训练任务、无后台实验或定时任务。

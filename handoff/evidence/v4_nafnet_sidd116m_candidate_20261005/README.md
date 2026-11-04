# V4 NAFNet-SIDD width64 候选资产

用户指定准备 **NAFNet-SIDD 116M 骨架及 D0/R0/R08/L08 适配**，并于 10-06 明确要求使用**官方预训练权重**。目录日期是准备开始日期；部分验收在 10-06 凌晨完成。本目录与已有 Restormer 正式 run 完全独立。

## 结构与参数量

官方来源：[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet/tree/2b4af71ebe098a92a75910c233a3965a3e93ede4)，固定 commit `2b4af71ebe098a92a75910c233a3965a3e93ede4`，官方 SIDD width64 配置：编码器 `[2,2,4,8]`、middle 12 块、解码器 `[2,2,2,2]`，共 36 个原版 NAFBlock。

| 模型 | 可训练参数 |
|---|---:|
| 官方 RGB NAFNet-SIDD width64 | 115,982,915（115.98M） |
| V4 六通道 + 条件 + clean/raw ξ 双头 | 117,767,209（117.77M） |

保留官方 NAFBlock、SimpleGate、SCA、LayerNorm2d（eps=1e-6）、编码/解码尺度和相加 skip。V4 接口为 `Concat(Z,Y)`、raw `[g,t]` FiLM；输出 clean RGB `B×3×256×256`、raw Zernike `B×36×512×512` 和 tilt `B×512×512×2`。clean 用 sigmoid，raw ξ 无界。官方 RGB ending 和输入 residual 被 V4 双头接口替代；没有引入官方 SIDD 的 loss、训练 schedule 或测试期 TLC。

当前默认资产为 **SIDD 官方预训练主体 + 新 V4 接口**，四方法共享同一份适配后的预训练 V4 U0。NAFBlock 的全部已训练权重（包括 beta/gamma）原样迁移；intro 扩成 `[W/2,W/2]`，保留原 bias；主干各级 FiLM 初始为恒等。

官方原模型和 V4 适配模型分别保留：

- **完整官方去噪模型**：`assets/pretrained/NAFNet-SIDD-width64.pth`，经原版 RGB NAFNet strict load 验证，115,982,915 参数。可用 `nafnet_v4.official_model()` 加载原始三通道去噪模型。
- **四方法预训练主体适配**：`protocol/pretrained_common_init.pt`，117,767,209 参数。官方 ending 是加性残差输出，而 V4 是 sigmoid 绝对 RGB，因此 V4 clean head、新条件和 raw ξ 分支新初始化；并非所有新增参数都经过预训练。详见 `protocol/PRETRAINED_COMMON_INITIALIZATION.json` 中逐项载入清单。

不导入 Restormer 或 NPU checkpoint。早先完成的随机 U0 及根目录随机验收报告保留为独立参考，**不作为当前默认初始化**。当前预训练验收证据位于 `reports/pretrained/`。详细改动见 [ADAPTATION.md](<ADAPTATION.md>)。

## 四方法科学定义

| 方法 | 中间状态 | RGB mixing | 预测次数 |
|---|---|---|---|
| D0 | direct | 0 | 1 |
| R0 | 预测 clean/raw ξ 的 physical recursion | 0 | G |
| R08 | 同 R0 | T0=0.8，逐步降至 0 | G |
| L08 | 预测 clean 与原观测构造 linear intermediate-state | T0=0.8，逐步降至 0 | G |

共同使用原版 `L = mean(|C_hat-C|) + 0.2 mean(|F(C_hat,xi_hat,G)-Y|)`；递归先对每条路径求均值，再对 batch 求均值。保持 G1–4、原观测 Y 条件、同一条路径的 eta、原 simulator 和 full-BPTT。R0/R08 refinement 为独立 inference 开关，沿用 per-sample serial 的 40 步实现，不参与训练。

## 入口与边界

- `nafnet_v4/model.py`：骨干和接口；`upstream.py`：校验源文件 SHA 后隔离导入未改写的官方类，避免 BasicSR 的无关插件依赖。
- `nafnet_v4/pretrained.py`：官方权重严格加载、来源/统计身份检查、共同预训练 U0、四方法 factory；`factory.py` 保留原随机参考入口和共享 AdamW。自定义 LayerNorm2d 正确进入不施加 weight decay 的归一化参数组。
- `nafnet_v4/inference.py`：四方法推理及 R0/R08 独立 refinement。
- `nafnet_v4/metrics.py`：复用 PSNR/SSIM/LPIPS 定义；固定评价网络权重只用于 LPIPS。
- `nafnet_v4/checkpoints.py`：显式绑定实验身份、model/optimizer/scaler/scheduler/RNG/采样位置/计数的保存与恢复。拒绝其他骨干或 run 的 checkpoint。
- `configs/groups_pretrained/` 与 `configs/runtime_pretrained.json`：当前预训练四方法配置。`groups/` 和 `runtime.json` 保留原随机参考配置。
- `protocol/`：官方源、继承科学代码、Train2000 统计、共同预训练 V4 U0、旧随机参考及授权边界。
- `scripts/accept_pretrained_cpu.py`：官方预训练权重映射和真实256尺寸特征 parity。`accept_cpu.py` 保留原随机结构的32尺寸梯度 parity 与方法语义 fixtures。
- `scripts/accept_native_pretrained.py`：实际 256/512 尺寸、B1/G4、一次临时优化器更新和 save/restore 检查。**不是 B32 正式训练验收。**
- `reports/`：实测结果；最终状态见 `reports/pretrained/ACCEPTANCE_SUMMARY.json`。
- `scripts/verify_assets.py`：无需 PyTorch/GPU 的完整资产 SHA 校验。

Train2000 仅用于这次有界验收：引用既有 canonical 数据、固定 train-only 统计，不重新生成、复制或抽样数据；只读取 train 样本，Test 锁定。后续若改用更多图片，必须新建数据身份并重算对应 train-only 统计，不能沿用本验收统计。

**正式 NAFNet 训练尚未启动。** 新数据规模、正式预算、batch/吞吐、GPU 布局及正式启动协议仍待下一阶段决定。当前已有的 Restormer A100 训练继续运行；NPU 保持停止。本目录没有自动训练控制器，`require_training_authorization()` 明确拒绝候选阶段正式训练。

## 环境与存储

A100 候选目录：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_nafnet_sidd116m_candidate_20261005`。

复用现有 `py310_torch271_cu126_v1` 环境，Torch 2.7.1+cu126，无需安装旧 BasicSR。simulator 资产和 canonical 数据通过验收配置引用原冻结目录，不产生额外数百 GB 数据副本。官方预训练文件约0.464 GB、共同预训练 V4 U0 约0.47 GB，独立保留旧随机 U0 约0.47 GB；临时优化器 checkpoint 验收后自动清理，不保留为正式训练结果。

原生 CUDA 验收用 BF16 骨干、FP32 simulator/loss 和非重入 activation checkpoint；验收用时包含加载、校验和 checkpoint I/O，不能作为训练 ETA。

**预训练版**独立完成了四方法 B1/G4、state restore、G2/serial40 refinement 及指标接口验收；当前结论见 `reports/pretrained/ACCEPTANCE_SUMMARY.json`。详细边界见 [验收记录](<../../NOT_INCLUDED.md#ref-bf07530a0971d98c6d65>)。评价依赖与缓存位置记录在 `protocol/ENVIRONMENT.json`；复用已有 Restormer 的 deps 和 `assets/torch`。

运行验收前先确认 GPU 没有其他进程，再通过 `CUDA_VISIBLE_DEVICES=<已核实的 GPU UUID>` 限定设备。脚本还会再次检查 UUID 占用，不能直接复用过去空闲的 GPU 编号。

```bash
CUDA_VISIBLE_DEVICES='' python scripts/accept_pretrained_cpu.py
CUDA_VISIBLE_DEVICES=<空闲GPU_UUID> python scripts/accept_native_pretrained.py --group D0 --gpu-uuid <空闲GPU_UUID>
```

`execution/`、`v4/`、`npu_runtime/` 是哈希封存的历史 primitives，仅复用明确的接口；其中旧 scheduler、NPU setup、Restormer checkpoint purpose 等定义不构成新实验计划或启动入口。NAFNet 公开 checkpoint API 是 `nafnet_v4/checkpoints.py`。


预训练迁移的数值边界：六通道卷积的求和顺序不同于原三通道卷积，FP32 下 intro 仅有约1e-6差异，但经过已训练深层网络后可放大。逐项参数一致、相同intro特征时主体输出完全一致；不声称六通道适配与原始三通道去噪模型端到端bitwise相同。

验收发现预训练版 R08/L08 在强噪声路径下初始梯度偏大（B1/G4样本分别约60.95和2.32e5，clip前），已有clip=1后更新finite。短数值补查见 `reports/pretrained/SHORT_NUMERICS.json`；正式有效batch/稳定性/学习率仍需新run验收。这里不能用去噪预训练或一次smoke推断湍流任务效果。

`UPSTREAM_IDENTITY.json` 是最初克隆源码时的身份快照，其中下载权重标记是当时状态；当前官方权重来源、哈希和载入方式以 `PRETRAINED_SOURCE.json` 与 `PRETRAINED_COMMON_INITIALIZATION.json` 为准。

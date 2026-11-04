# TurbWNet 官方预训练 G4 Zero-shot 验证：停止报告

执行日期：2026-10-28至2026-10-29，Asia/Shanghai。

**官方权重真实取得，完整模型严格加载通过；NPU Smoke Test和Val1000未完成。本次共卡启动触发D0现有排他守护，导致D0训练中断，已停止TurbWNet。** 我未在共卡前检查这条排他保护，这是执行疏漏。事件因果链、残留进程和已核验checkpoint见 [事件报告](D0_SUPERVISOR_INCIDENT_ZH.md)。

## 1. 官方权重

来源为 [官方仓库README](https://github.com/FallenLeaves-Creator/TurbWNet/blob/05627b725241c28a09b03c00faefafa861558ebf/README.md) 指定的 [百度网盘](https://pan.baidu.com/s/1ZJA359Ia4W-WT5Rolm11mw?pwd=sv8d)，提取码sv8d。网页直接下载遭客户端限制，之后由用户通过客户端下载并放入本次独立 `weights/`。本轮对实际文件执行了大小、SHA256、ZIP CRC以及CPU参数读取核查；本地和远端副本哈希一致。

| 实际文件 | 字节数 | SHA256 |
|---|---:|---|
| net_g_300000.pth | 130,013,973 | `a039c67176a0704a9db22cfefe6f58e78d09681e50c03ce73bfad34290b0843f` |
| net_g_deblur.pth | 46,211,772 | `a4b21e40b34163dd7a790b363e32a56bed33c0d454c30568579fc49b812f5638` |
| net_g_detilt.pth | 46,208,316 | `9db43e5d15e8ed37c3bbecf1a52f6202f970a44c3e23a8c2f56fda92730fd0b8` |

三个文件均只有 `params` 分支，全部张量为有限值。完整模型709个张量、32,430,816参数；独立deblur为274个张量/11,526,395参数，detilt为274个张量/11,525,531参数。原始清单见 `evidence/WEIGHTS_RECEIVED.json` 和 `CHECKPOINT_INVENTORY.json`。公开来源没有给出作者校验和，以上SHA用于固定本次已取得文件，不声称与作者另外发布的SHA交叉验证过。

## 2. 源码、模型与严格加载

固定 [官方源码版本](https://github.com/FallenLeaves-Creator/TurbWNet/tree/05627b725241c28a09b03c00faefafa861558ebf)：`05627b725241c28a09b03c00faefafa861558ebf`。39个跟踪文件的SHA256及干净工作树状态见 `evidence/SOURCE_PROVENANCE.json`。

完整路径为：单张RGB退化图 → deblur U-Net及四级特征 → detilt U-Net估计两通道偏移 → 四级特征按对应flow作bilinear/grid_sample/border/align_corners=True warp → 最终恢复decoder及RGB输出。没有把GT、真实tilt、物理参数或额外帧传给网络。

采用官方deturb配置：dim48、blocks=[1,2,2,1]、refinement1、channel/spatial heads=[1,2,4,8]、window8、spatial_dim_head16、ffn2.66、WithBias、scale1。完整 `params` 自身包含 `deblur_model.*`、`detilt_model.*` 与最终decoder，strict=True加载全部709个张量通过，missing/unexpected均为空，加载后逐张量相等检查通过。

两个独立子模型也各自strict=True加载通过。完整模型的deblur参数与单独deblur文件不同，detilt相同；因此运行使用完整checkpoint，未将单独子模型重新覆盖进完整模型。

官方 `source/` 未改。独立 `runtime/official_arch.py` 只移除BasicSR注册和未使用的torchstat/torchvision transforms导入，类结构、方法、算子及前向路径经AST对照相同。没有修改网络结构、跳过阶段或用随机参数补缺。差异与哈希见 `runtime/ADAPTATION.json` 及 `OFFICIAL_ARCH_ADAPTATION.diff`。

## 3. 单帧接口与CPU技术验收

输入为 `[1,3,256,256]`、RGB、float32、[0,1]，不作额外均值/标准差归一化。官方数据读取中的BGR会转换到RGB；本次既有G4数组已是RGB，不再次换通道。沿用官方16倍数反射padding和原尺寸裁回，本数据256×256无需padding。调用 `eval()`、冻结参数及 `torch.inference_mode()`。

CPU完整加权前向在固定 `val_s0040_r02` 上通过，三个阶段各调用1次，输出形状保持 `[1,3,256,256]`，输出范围约[-0.0027144,0.9840679]。CPU参考耗时51.515秒，浮点数组保存于 `protocol/CPU_REFERENCE.npy`，仅用于预定的CPU/NPU数值对照；不是CPU或NPU的完整恢复效果报告。

为减少共享NPU负载，原指标函数计划在CPU执行。对4张已保存D0输出的检查中，PSNR/SSIM差异均为0，LPIPS最大绝对差异1.19e-7。指标源文件未经修改，SHA256为 `ea241b2477a07b67d2704d47f8db9e1e38a47ac9593524203f222807db05ef2d`。证据见 `evidence/CPU_ACCEPTANCE.json`。

## 4. NPU尝试、资源与D0影响

实际选择物理NPU7，batch1。初始化前最近100个四rank共同完成更新的临界均时约0.72556秒，D0吞吐82.695图/秒。显存约16.8/64 GiB已用，约47.2 GiB空闲。没有占用NPU0–3。

前两次尝试都停在NPU环境初始化：独立启动脚本覆盖了CANN Python路径，修正后仍缺少独立环境中的CANN依赖。该环境问题已在CPU侧修复并通过导入检查；第三次尝试在设备所有权检查时发现D0训练worker缺失，未再启动NPU子进程。

第二次尝试的PID3206173被D0原守护记录为foreign并触发SIGTERM，D0于00:09:40–41异常中断。**这是本任务引起的守护冲突，不能称为“D0未受影响”。** 完整说明见事件报告。没有显存耗尽证据，也没有得到模型算子兼容性或推理速度结论。

- TurbWNet实际NPU图像前向：**0张**。
- 5–10图NPU Smoke Test：**未通过/未完成**。
- 单图NPU速度及模型峰值显存：**无实测值**。初始化的110 MiB上下文占用不等于完整模型峰值显存。
- D0共享安全性：**失败，现有守护不允许独立外来进程共卡**。
- 已退出TurbWNet模型进程，后续启动已用停止标记封锁。没有擅自终止D0残留rank、重启D0或修改其守护/依赖。

## 5. Val1000与逐图结果

原Val1000已冻结为250个独立Clean×4个G4 realization，sample_id集合与旧D0全部1000行一一匹配。未读取锁定Test图像。指标仍为RGB、range1、逐图PSNR后平均，原Gaussian SSIM11×11/sigma1.5/population covariance，官方LPIPS-Alex v0.1/normalize=True/FP32，以及MAE；计划使用未经clamp/crop/resize的浮点输出，PNG只做展示clip。

| 项目 | PSNR(dB) | SSIM | LPIPS | MAE | 本轮状态 |
|---|---:|---:|---:|---:|---|
| 输入G4 | 18.795938 | — | — | — | 原有基准参考，未重新全量计算 |
| 旧D0 U14927 EMA | 21.060202 | 0.54878696 | 0.52907033 | 0.05615627 | 已有结果及其哈希核验 |
| TurbWNet官方预训练 | — | — | — | — | **Val1000未执行** |

没有TurbWNet单图指标、逐图配对改善、全量均值或NPU恢复视觉图。不能对其G4恢复质量作判断，也没有使用CPU单图或仓库示例图冒充本次NPU评估结果。

## 6. D0最后完整状态

最后登记完整checkpoint为 `.../distributed/V6_D0_U14927_official_residual_microopt_01/checkpoints/U030500.pt`，大小580,757,443字节，SHA256为 `e1b64746e451e57f412a36048f9a088a11c3255087b884352aae63b188bd65b4`。本轮只读CPU反序列化通过，含raw/EMA/Adam/调度/RNG/缓冲等状态。未发现U30500之后的checkpoint；rank0最后记录完成U30920，但停机保存失败。

00:14:30现场仍有3个D0 rank残留于NPU5–7。此次没有对它们发信号或执行恢复。详见 `evidence/D0_LAST_CHECKPOINT_READONLY.json` 和事件报告。

## 7. 文件与复现入口

- `RESULT.json`：机器可读停止状态，包含权重、加载与D0事件。
- `weights/`：三个真实官方文件；`source/`：固定原版源码。
- `runtime/`：不改变前向的独立适配副本与原指标实现。
- `protocol/`：固定Val1000、8个预选Smoke ID和CPU参考。
- `evidence/`：原始来源、环境、CPU验收、D0只读故障现场和最后checkpoint核查。
- `outputs/smoke01`、`smoke02`、`smoke03`：三次尝试的完整记录，失败目录均保留。
- 远端独立目录：`/data2/yanglin/turbwnet_zeroshot_20261028`。

无NPU的验收复核命令为 `bash scripts/run_env.sh scripts/inspect_weights.py` 和 `bash scripts/run_env.sh scripts/accept_cpu.py`，工作目录为上述远端独立根。后者包含1次CPU前向与既有D0输出的指标核对，没有训练。

**不要直接重跑共享推理命令。** 本任务已停止，`evidence/STOPPED_AFTER_D0_SUPERVISOR_CONFLICT.json` 会阻止启动。D0残留进程/恢复及其排他守护需要先另行处理，才能决定是否继续本基线；本轮没有自动修改这些设置，没有改用其他设备、随机权重或训练。

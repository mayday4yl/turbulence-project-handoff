# V4 D²Turb-Restormer 候选

本目录把 **D²Turb 作者公开的 Restormer 骨干结构** 接入 V4 的 D0、R0、R08、L08，使用同一份全新随机初始化。它是候选模型资产与正确性验证包；正式训练尚未授权，也没有选择新的训练预算或学习率计划。

## 来源与结构

- 权威代码：[HertzDot222/D2Turb](https://github.com/HertzDot222/D2Turb/tree/bbc28441b7212cc55319124a081ed6c0001879ef)，固定 commit `bbc28441b7212cc55319124a081ed6c0001879ef`。
- 使用 `vendor/restormer/d2turb/code/models/restormer.py`。该源码 SHA256 为 `70ba46b5f28b3d1e4780725bd742e7826642be0e488aac6e906891cdcb13331a`。
- 通道 48/96/192/384；encoder 4/6/6/8 blocks；decoder 6/6/4 blocks；refinement 4 blocks，共 44 个 Transformer blocks。保留 MDTA、GDFN、上下采样、skip、作者版 `nn.LayerNorm(eps=1e-5)` 和 temperature 参数布局。
- 作者原 RGB 网络参数为 **26,126,644**；V4 适配后可训练参数为 **27,032,525**。
- 作者仓库公开的是推理结构，未提供可核验的训练脚本和完整训练配置。因此这里只称“对齐作者公开骨干”，不声称复现其未公开训练过程。
- `vendor/restormer/` 中 swz30 标准 Restormer 资产仅供比较；不参与候选实例化。D²Turb 的 ASPI / flow rectifier 未加入 V4。D²Turb 固定版本树内未发现 LICENSE/COPYING，标准 Restormer 的许可证不能替代它。

## V4 适配

输入为原尺度 FP32 `Concat(Z,Y)`（6×256×256），raw `[g,t]` 经原 JointCondition 和各 stage 的 FiLM 注入。输入不 clamp、不增加标准化。RGB 输出使用 sigmoid；因此没有照搬作者 RGB 残差输出层。独立 ξ 分支输出不限制范围的 Zernike 36×512×512 和 tilt 512×512×2。

| 方法 | 起始 t | 路径 | 网络调用次数 |
|---|---:|---|---:|
| D0 | 0 | 单次直接恢复 | 1 |
| R0 | 0 | 原物理模拟器递归 | G |
| R08 | 0.8 | 原物理模拟器递归 | G |
| L08 | 0.8 | 原线性路径递归 | G |

共同保留 RGB 全图 MAE 的 `LC + 0.2 LR`、路径平均后 batch 平均、full BPTT、固定同一路径 eta。训练中无精修。13 个科学逻辑/统计文件与既有 SwinIR 实验逐字节一致，见 `protocol/INHERITED_IDENTITY.json`。归一化统计沿用冻结的 train500/8000 记录；没有访问测试集。

`protocol/common_init.pt` 与 `COMMON_INITIALIZATION.json` 是四组唯一共同 U0；从零生成，没有加载恢复预训练权重。物理模拟器依赖的既有 P2S 资产属于固定前向模拟器。

## 验证及使用边界

汇总以 `reports/CANDIDATE_ACCEPTANCE.json` 为准，详细原始记录在 `reports/acceptance_*` 下。

1. 完整宽度/深度、32×32 CPU：与作者骨干特征及输入/全部骨干参数梯度比较，六通道映射和 FiLM 隔离方式写入报告。
2. 真实 256 图像、512 模拟器：混合 G=1/4 的前向、逐样本一致性、真实 MAE、G4 完整反传及故意断图的负控。
3. 检查四组共同 U0、完整 RAW/EMA/空 optimizer/RNG 保存恢复，以及损坏身份、状态、目标 EMA 和错误 optimizer 绑定的拒绝行为。正式训练入口主动拒绝，全部 optimizer 调用为 **0**。

显存限制必须保留：无 offload 的 R0 G4/B1 测试在 allocator cap 0.35（约 27.74 GiB）下发生 OOM，记录在 `reports/acceptance_final_v1/native/R0/acceptance.json`。R0 后续使用仅限验收的 CPU saved-tensor offload，保留完整图和 FP32，不重算、不 detach。其通过不代表无 offload 可训练。D0 同时有原生 CUDA 和 offload 对照；R0 验收结束后 GPU6 被他人任务占用，资源护栏停止了后续 CUDA 测试，R08/L08 的真实尺寸检查在 CPU 完成。CPU 证据不替代这两组的 CUDA 专项验收。

尚需在正式训练方案确定后验证：B32 / micro2 的显存与梯度累积、激活 checkpoint 或其他内存方案、实际 optimizer 更新和有 moments 的断点恢复、完整精修集成、训练速度与 ETA、验证集表现。不能根据本验收推断比 SwinIR 更好、已经收敛或可以扩大训练预算。

## 入口与位置

- `restormer_v4/model.py`：骨干与 V4 接口；`factory.py`：四组、共同初始化、固定模拟器；`checkpoint.py`：严格 U0 存取。
- `restormer_v4/inference.py`：四方法推理适配，完整 optional refinement 仍待专项集成验收。
- `configs/runtime.json`、`configs/groups/*.json`：候选配置；`protocol/CONTROL.json` / `AUTHORIZATION.json`：范围。
- `scripts/accept_candidate.py`：分层验收；`scripts/run_bounded_acceptance.py`：有限序列、GPU6 身份/空闲/显存及 host RAM 检查。不会自动抢卡或启动训练。
- 远端：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_restormer_candidate_20261002`。
- 远端依赖：既有 `shared/envs/py310_torch271_cu126_v1/bin/python`；无新环境、无依赖安装。

继承 `v4/training.py` 内的历史 LR 等辅助函数不代表当前候选选择了该排期；本目录没有正式训练 launcher。

磁盘审计独立见 `reports/STORAGE_AUDIT_20261002.md`，审计不等于删除授权或已释放空间。

# NAFNet-SIDD 116M → V4 四方法

## 改动位置

1. **输入**：官方 intro 从 3 通道改为 6 通道，接 `[Z,Y]`。保留 3×3、padding1；从官方预训练权重扩为 `[W/2,W/2]` 并复用原 bias，使 Z=Y 时保持原 intro 映射；输入与原实验一致，不 clamp、不额外标准化。
2. **条件**：复用 V4 `JointCondition`，raw `[g,t] → 256 → 256`。在四个 encoder、middle、四个 decoder 前加 FiLM；decoder 的注入位置是 upsample+skip 之后。当前预训练版主干 FiLM 的 scale/shift 投影初始全零，以保留预训练主体的特征路径。条件 MLP 在首步可能是零梯度，投影更新后再学习，这是预期初始化，不是断图；新 raw ξ block 内 FiLM 仍按旧 V4 接口初始化。
3. **主干**：官方 width64 / `[2,2,4,8]` / 12 / `[2,2,2,2]`；所有 NAFBlock 和其官方 LayerNorm 自定义反向代码保持不变。当前使用已训练 checkpoint 中的 beta/gamma，不将其重置为零。原随机参考版才采用官方从零训练的 gate=0 初始化。
4. **输出**：用 width64 feature 接 3×3 clean RGB head+sigmoid，另接与现有 V4 同型的 realization projection、nearest×2、卷积/SiLU/FiLM block、36+2 raw heads。官方 `ending` 预测加性残差，与 V4 的绝对 RGB sigmoid head 语义不同，因此不直接误载该输出头；完整原去噪模型仍保留于官方 checkpoint。V4 clean head 与 ξ decoder 新初始化，raw heads 零初始化、无界、在 simulator 内应用 severity。未对 ξ 改坐标系或缩小 native resolution。
5. **数值/内存**：BF16 只限模型 autocast 范围；输出转回 FP32，simulator/loss 保持 FP32。stage activation checkpoint 不 detach，不改变完整递归反传。必要 CPU activation offload 仅为无损存储选项，是否使用以验收报告为准。
6. **优化器**：沿用 V4 AdamW lr1e-4、betas(.9,.999)、eps1e-8、weight decay1e-4；bias 和 normalization 参数不做 decay。NAFNet 的自定义 `LayerNorm2d` 显式纳入该规则。这里只是验收配置，没有采用官方 SIDD 400k/PSNRLoss 的训练配方。

额外可训练参数共 **1,784,294**（约原版的 1.54%）。适配后四方法结构/参数量相同，差别只在原有 direct/physical/linear rollout 和噪声 schedule。

## 验收的意义

当前默认使用官方预训练主体。其 256 尺寸真实预训练 feature parity、逐项 exact tensor transfer 和四方法 native 重新验收见 `reports/pretrained/`。以下小空间非零 gate 描述是早先随机骨架的结构测试，不混作预训练权重证据。

- **官方路径 parity**：full-width/depth、小空间32的 CPU 对照，匹配所有共享参数；临时把 FiLM 置零，RGB intro 分成两份权重接同一张图。为避免零 beta/gamma 造成大量平凡零梯度，在两个仅用于 parity 的模型上将 gate 同设为 .1；这些模型不保存为 U0。
- **语义 fixtures**：混合 G1–4、路径长度、L1/.2 权重、同一 eta、clean target 不进入网络、terminal BPTT 和故意 detach 的反例。这是方法语义检查，不是主干速度或恢复效果。
- **真实尺寸**：冻结 canonical train 样本、256/512、G4，检查四方法 finite、clean/ξ 分支梯度、训练/inference OFF 一致性和一次优化器更新。
- **递归依赖**：仅从末步 clean loss 反传到第一步预测；R0/R08 应连通 clean 与 raw ξ，L08 应连通 clean，而对 raw ξ 的这一条跨步依赖按定义不存在。L08 的 raw ξ 仍由各步重退化损失训练。
- **恢复**：保存后主动扰动网络/optimizer，再按绑定 identity 恢复，核验完整参数/optimizer hash、RNG、scaler、计数和采样位置；错误 run identity 必须被拒绝。单次状态恢复不能替代未来长训的断电/异常/分布式恢复验收。

本候选并不声称已验证大模型更优、已找到最快 batch、已完成 B32 或长训收敛。扩大数据量、训练预算和实际比较必须另建正式 run，不能把这些临时验收值并入 Restormer 训练曲线。

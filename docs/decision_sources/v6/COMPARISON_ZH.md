# UniDB 与 PixelDiT＋图像控制：当前单帧湍流任务选型

日期：2026-11-01。结论是基于原文和固定官方源码的适配判断，不是湍流质量排名。

当前优先建议：先验证 UniDB 的可运行性和原生配对恢复接口。PixelDiT＋图像控制作为后续投入较大的生成先验适配路线。两者均无本项目湍流实测，不能承诺超过 D0。

| 比较项 | UniDB-GOU | PixelDiT1 C2I-XL256＋待设计图像控制 |
|---|---|---|
| 来源 | ICML 2025；原生图像恢复扩散桥 | CVPR 2026；原生类别条件像素生成／Rectified Flow |
| 状态空间 | RGB，无 VAE | RGB，无 VAE |
| 已有空间条件 | 每步拼接当前状态减 LQ 与 LQ，并提供时间 | 类别条件及内部噪声上下文；需新增退化 RGB 的空间控制 |
| 效果预期 | 配对恢复和输入约束更直接，适合作可核验的恢复基线；几何纠正效果未知 | 生成先验可能改善纹理／感知质量，但真实细节、几何和文字保真均待验证 |
| 方法与工程改动 | 数据范围／配置、原生目标、保存恢复与五档验证接入 | 新条件编码／注入位置／初始化与优化策略；需处理原类别条件和原生 flow 目标 |
| 参数规模 | nf64/depth4 配置静态整数算术约 137.15M | 论文约 797M 主干＋尚未定义参数量的控制分支 |
| 训练计算 | 随机时间，一次网络前向／microbatch；通常全网训练 | 随机时间，一次主干前向并计算条件分支；冻结主干仍需向条件特征反传 |
| 官方推理参考 | 默认 100 次 denoiser 调用 | 默认 100 次 2B CFG 批量调用，约 200 单分支等效评估，另加控制分支 |
| 成本结论 | 接入和排错成本较低；实际速度／显存未测 | 工程和单步计算负担预计更高；冻结可省优化器状态，不能由可训练参数数目直接推算耗时 |
| 现有资产证据 | 官方代码与权重链接已核；权重库存、完整加载未核，没有已验证湍流权重 | 官方生成权重库存已核；没有已验证退化 RGB 恢复权重或可直接加载的湍流 ControlNet |

## 对当前项目的含义

1. 160K 缓存由 10K Clean×4 ξ×4 强度组成，内容独立性是 10K Clean。两条路线原则上均可复用配对数据，无需仅为更换模型重新生成；必须核对原生范围、裁剪、划分和实际条件接口。
2. 应保留固定五档 Val1000 及原来源身份，分开比较 PSNR／SSIM／LPIPS、几何及文字结构保真和单图延迟。生成 FID 不构成湍流恢复优势证据，参数比约 5.8 也不是耗时比。
3. UniDB 的训练目标是其原生 noise／score 与反向一步 ML 匹配；PixelDiT 为 velocity matching，C2I 公开预训练还含 REPA。均不能直接继承 D0 的 Residual L1、SDT 或 Adam／EMA 状态并声称是原方法。
4. 两者训练通常仅抽一个时间步，100 步采样不意味着每个训练更新运行 100 次网络。五档 5000 图完整验证仍可能成为主要时间开销。
5. 冻结 PixelDiT 参数时，控制注入之后的主干需保留对控制特征的导数。把整个主干包在 no_grad 中会阻断该学习路径。未来控制方案不一定复制整个主干，当前不能提供精确可训练参数／显存／小时预算。
6. PixelControl 提供了基于 PixelDiT-T2I 的 depth／seg／edge 控制代码参考，但不是已验收的 C2I-256 湍流恢复支路；不能直接照搬其 14 层 adapter 或 SD 的注入位置。

若近期优先取得可信湍流恢复结果，UniDB 更适合作为第一候选；若目标转为研发预训练像素生成模型的恢复控制机制，PixelDiT 更贴合该研究方向，代价与不确定性也更大。

## 原文与代码依据

- UniDB 原文：https://proceedings.mlr.press/v267/zhu25o.html
- UniDB 官方代码，固定 commit 1bb63e318891b932c6e641b98729643e46b91e06：https://github.com/UniDB-SOC/UniDB
- PixelDiT 原文：https://arxiv.org/abs/2511.20645
- PixelDiT 官方代码，固定 commit f7d7c35f71893d81c6e71d2576ea80601a1e8529：https://github.com/NVlabs/PixelDiT
- PixelControl 官方代码，固定 commit fc5dc8957d13efa811fc7c25f1702a5a4054a0af：https://github.com/linxin0/PixelControl
- 逐项代码依据和参数算术：unidb/EVIDENCE.json、unidb/PARAMETER_ARITHMETIC.json、unidb/MODULE_UTIL_SOURCE.json、pixeldit/EVIDENCE.json；继承源码位于 recent_pixel_ir_20261101 与 pixeldit_fit_20261101 的固定来源清单。

本轮仅来源与静态算术核查：0 模型实例／前向，0 权重载荷下载，0 训练／NPU 作业；A/B 未恢复，既有 checkpoint、冻结配置和历史结果未改。

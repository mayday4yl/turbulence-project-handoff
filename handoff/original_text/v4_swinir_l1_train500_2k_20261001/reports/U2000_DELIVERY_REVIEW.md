# V4 SwinIR + L1 + 独立 2k LR：U2000 完整交付核验

本机核验完成时间：2026-10-02T16:55:48.448015+08:00（UTC 2026-10-02T08:55:48.448015+00:00）。服务器墙钟偏快，本文完成时间使用本机时钟。

四组均已完成预设的 2000 次正式 optimizer update（各 64000 次 start-record exposure），完整交付已下载并核验。实际进程扫描、历史 PID 核对、空 worker 列表及 tmux 退出共同确认本实验全部进程退出。预算到点停止；不自动延长，也不恢复旧 U-Net 实验。方法下一步等待用户决定。

本次预算内，D0 的验证 PSNR、SSIM、LPIPS 均优于三种递归组；四组验证 PSNR 均低于 Input 基线。此结论限于本次冻结配置、数据和训练种子，不能据此宣称收敛、递归方法普遍无效或某个单独组件导致结果。

## 主结果：共同 U2000 / RAW / refinement OFF / val800

Input PSNR = 20.539433 dB。验证 population 为 100 个 source、800 个 start。PSNR/SSIM 越高越好，LPIPS 越低越好。

| 组 | 验证 PSNR (dB) | SSIM | LPIPS | 相对 Input (dB) | 完整 train8000 PSNR | Train−Val (dB) |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 20.473245 | 0.520175 | 0.648212 | -0.066188 | 20.597070 | +0.123825 |
| R0 | 20.443722 | 0.516498 | 0.653578 | -0.095710 | 20.552908 | +0.109186 |
| R08 | 20.419548 | 0.516473 | 0.653467 | -0.119885 | 20.536331 | +0.116783 |
| L08 | 20.400306 | 0.514626 | 0.658067 | -0.139127 | 20.516222 | +0.115917 |

完整训练集的表现同样有限；较小的 Train−Val gap 本身不能证明恢复效果好或泛化成功。monitor400 仅作固定子集监控，表中训练指标全部来自真正的 fulltrain8000。

最终 best 仅在共同 [0,2000] 的 0、250、…、2000 节点按 RAW/OFF val800 PSNR 选择，1e−6 dB 容差取较早节点。四组均选中 U2000，SSIM/LPIPS 同 checkpoint，因此无需非里程碑 best 的条件 fulltrain 补评价。ON 精修不参与训练或 best。

## 配对比较

先对每个 source 的 8 个 start 聚合，再对 100 个 source 配对 bootstrap：10000 次重采样，seed=20260929。以下差值及 CI 已在本地复算。CI 描述此训练种子下的 source 变异，不覆盖训练种子变异或精修随机性。

| 比较 | 平均 PSNR 差 (dB) | 95% CI | Source 胜率 |
|---|---:|---|---:|
| R0-D0 | -0.029522 | [-0.040470, -0.018662] | 28% |
| R08-R0 | -0.024174 | [-0.030785, -0.017642] | 18% |
| R08-D0 | -0.053697 | [-0.064953, -0.042484] | 18% |
| L08-D0 | -0.072939 | [-0.086621, -0.059638] | 13% |
| R08-L08 | +0.019242 | [+0.013377, +0.025494] | 76% |

R0、R08、L08 相对 D0 的配对 PSNR CI 均为负。R08 优于 L08 的这项对比仍未超过 D0。递归输出与自身第一步的 progression 不是独立 D0 对照，不能把 progression 正值直接等同递归净优势。

## 与共同 U1000 及旧配置的有界比较

| 组 | 新 U1000 PSNR | 新 U2000 PSNR | 新 1k→2k 增量 | 旧 U2000 PSNR | 新−旧 U2000 | 旧共同 [0,2k] best PSNR / update |
|---|---:|---:|---:|---:|---:|---|
| D0 | 20.208763 | 20.473245 | +0.264481 | 20.047625 | +0.425620 | 20.636709 / U1000 |
| R0 | 20.160170 | 20.443722 | +0.283552 | 19.907147 | +0.536575 | 20.542166 / U1000 |
| R08 | 20.218231 | 20.419548 | +0.201317 | 19.741949 | +0.677599 | 20.545230 / U1000 |
| L08 | 20.113800 | 20.400306 | +0.286506 | 19.917520 | +0.482786 | 20.535361 / U1000 |

新配置四组从 U1000 到 U2000 的验证 PSNR 均上升；U1000 时 R08 相对 D0 的小幅领先没有保留到 U2000。共同 U2000，新配置 PSNR/SSIM 高于旧配置，但 LPIPS 更差。共同 [0,2k] 按相同规则选择 best，旧配置四组均为 U1000，新配置四组均为 U2000；新配置四组的 PSNR/SSIM 更低、LPIPS 更高。故不能把旧 U2000 单点作为“新配置全面改善”的依据。

新旧比较同时改变 backbone、loss 和 LR，应整体归因；MSE 与 MAE training loss 不直接比较。旧 U2000 缺少 fulltrain 时保留 N/A，不用 monitor400 代理；旧 U5000 仅属历史。旧运行、文件与 checkpoint 保持只读。完整数值和原始来源见 [matched updates](../stage_deliveries/U02000/old_vs_new_matched_updates.md)、[common-grid best](../stage_deliveries/U02000/old_vs_new_common_grid_best.md) 与 [OLD_VS_NEW](../stage_deliveries/U02000/OLD_VS_NEW.md)。

## 仅作辅助的 U2000 ON−OFF 精修

R0/R08 在 val800 使用已冻结的串行 40 步原 MSE observation + xi proximity 精修；OFF 复用 RAW。以下为一次正式调用，未为核验额外重跑精修。

| 组 | 指标 | ON−OFF | 95% CI | Source 胜率（依该指标方向） |
|---|---|---:|---|---:|
| R0 | psnr | -0.010462 | [-0.013776, -0.007306] | 24% |
| R0 | ssim | -0.000854 | [-0.001000, -0.000716] | 6% |
| R0 | lpips | -0.001161 | [-0.001368, -0.000953] | 88% |
| R08 | psnr | -0.000069 | [-0.000473, +0.000318] | 50% |
| R08 | ssim | +0.000008 | [-0.000003, +0.000019] | 58% |
| R08 | lpips | -0.000166 | [-0.000193, -0.000139] | 89% |

R0 的 ON PSNR/SSIM 下降，LPIPS 小幅改善；R08 的 PSNR/SSIM 变化 CI 跨零，LPIPS 小幅改善。不能据此声称 PSNR 精修收益或推断重复精修稳定性。正式完整调用 R0=1581.88s、R08=1666.70s；仅为已有计时，不作额外推理。

## 交付与正确性证据

- 原包 manifest SHA256：`abb91320b0c12ba03c35ace4848d68853995f8a595ed9db38762773658c1a3d6`；1795 个清单文件，2,012,572,411 bytes（约 2.01 GB）。远端逐文件和本地下载 hash 均 PASS。
- 4 个实际完整 checkpoint 已 CPU 加载；model/EMA tensor hash、734 项 optimizer state、update/exposure/stream/scheduler、评估权重与 checkpoint 对应关系均 PASS。没有运行新 forward、backward、optimizer 或验收。
- RAW 各组 fulltrain8000、val800、derived monitor400 的身份、人口、逐行有限值与配对 PASS；derived monitor 按冻结格式核验父 COMMIT/hash/checkpoint 与行内容，不修改格式。14 个嵌套 COMMIT / 1604 个清单文件、72 个曲线输入 COMMIT、28 个新旧来源评价目录核验 PASS。
- 67 个冻结源文件和 2 个既有验收 evidence 文件共 69 项 hash PASS，execution/protocol 身份未变。
- 原包 89 张 PNG 与 5 个 PDF 的 5 页已实际视觉检查。主表长 B 标签的原始展示越界保留原包，单独本地修正阅览版；修正 PDF 的实际渲染已检查，4 位小数数值对源 JSON/PDF text 核对 PASS。
- 最终进程核验 PASS：本实验实际 process rows 与历史 owned PID rows 为空，tmux session 已退出，worker/gpu mapping 为空，RELEASE 为 stop/PRESET_BUDGET_ENDPOINT。

Execution hash：`617b6adfc710f040a8263f46cf72b692448dbb8f13e80c635d1445b2dcc9cbfd`。

Protocol hash：`533e75ca59247a8fe4d6938451b0ea06486363ffbd7ecc26eeea9388a0e8ff08`。

2k 为预设预算，未宣称收敛。科学配置、冻结执行和旧证据均未改；没有预算扩展、抢卡、第三卡、验收重跑或旧实验恢复。test200 保持禁用。

## 阅览与审计文件

- [修正主表 PDF](U2000_REVIEW_EDITION/main_performance_table.pdf) / [PNG](U2000_REVIEW_EDITION/main_performance_table.png) / [DISPLAY_IDENTITY](U2000_REVIEW_EDITION/DISPLAY_IDENTITY.json)
- [远端完整交付审计](U2000_REMOTE_DELIVERY_AUDIT.json)
- [本地下载与统计复算审计](U2000_LOCAL_DOWNLOAD_AUDIT.json)
- [全部视觉检查记录](U2000_VISUAL_REVIEW.json)
- [最终进程退出审计](U2000_FINAL_PROCESS_EXIT_AUDIT.json)
- [冻结 hash 审计](U2000_FROZEN_HASH_AUDIT.json)
- [原包 README](../stage_deliveries/U02000/README.md)

自动跟进关闭状态另见本次 HEARTBEAT 与 AUTOMATION_COMPLETION 记录；只有工具确认删除成功才记录为已删除。

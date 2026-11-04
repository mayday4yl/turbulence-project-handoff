# Known-g Physical Turbulence Bridge v0.1 最终报告

## 结论

progressive bridge 未达到冻结成功判据。G=4.50 上 Bridge + refinement 相对 one-shot 的 paired PSNR 变化为
0.0000 dB，95% bootstrap CI 为
[0.0000, 0.0000]；机制分类为
`validation_selected_one_shot_progressive_not_supported`。冻结 K=0，refinement learning rate=0.03。

## 原始假设

沿固定 realization 的 PiRN/P2S 六级 severity trajectory 逐步恢复，可能优于同一网络直接 one-shot 恢复。

## 执行与冻结

- 数据：Places365-Standard，9400 条 trajectory，六级状态与 simulator-native FP32 ξ 完整保存。
- 训练：完成 20000 updates；使用 EMA checkpoint；effective batch 固定 16。
- checkpoint：`/data2/yanglin/physical_bridge/runs/bridge_v01_seed20260826/checkpoints/frozen_checkpoint.pt`，SHA-256 `34dc4dce03bd13ba4079100d4b195fcc0e8c3b07699e06ae12ca91e0aba48f47`。
- test 前冻结：checkpoint、refinement LR、K 均来自 val；test 未调参。
- implementation fixes：NPU device port、真实 512 ξ shape、batched scalar-G、decoder 512 上采样、indexed tar、BF16/FP32 边界、DDP/checkpoint/resume、数值与磁盘保护。
- method changes：无。

## Frozen Test L1

| G | one_shot | bridge_no_refinement | bridge_refinement | oracle_xi | teacher_forced |
|---:|---:|---:|---:|---:|---:|
| 1.50 | 0.0385 | 0.0385 | 0.0385 | 0.0385 | 0.0385 |
| 2.25 | 0.0450 | 0.0450 | 0.0450 | 0.0450 | 0.0450 |
| 3.00 | 0.0508 | 0.0508 | 0.0508 | 0.0508 | 0.0508 |
| 3.75 | 0.0568 | 0.0568 | 0.0568 | 0.0568 | 0.0568 |
| 4.50 | 0.0597 | 0.0597 | 0.0597 | 0.0597 | 0.0597 |

## Frozen Test PSNR

| G | one_shot | bridge_no_refinement | bridge_refinement | oracle_xi | teacher_forced |
|---:|---:|---:|---:|---:|---:|
| 1.50 | 24.4206 | 24.4206 | 24.4206 | 24.4206 | 24.4206 |
| 2.25 | 22.9769 | 22.9769 | 22.9769 | 22.9769 | 22.9769 |
| 3.00 | 22.1642 | 22.1642 | 22.1642 | 22.1642 | 22.1642 |
| 3.75 | 21.0343 | 21.0343 | 21.0343 | 21.0343 | 21.0343 |
| 4.50 | 20.6272 | 20.6272 | 20.6272 | 20.6272 | 20.6272 |

## Frozen Test SSIM

| G | one_shot | bridge_no_refinement | bridge_refinement | oracle_xi | teacher_forced |
|---:|---:|---:|---:|---:|---:|
| 1.50 | 0.7381 | 0.7381 | 0.7381 | 0.7381 | 0.7381 |
| 2.25 | 0.6700 | 0.6700 | 0.6700 | 0.6700 | 0.6700 |
| 3.00 | 0.6145 | 0.6145 | 0.6145 | 0.6145 | 0.6145 |
| 3.75 | 0.5710 | 0.5710 | 0.5710 | 0.5710 | 0.5710 |
| 4.50 | 0.5550 | 0.5550 | 0.5550 | 0.5550 | 0.5550 |

## Frozen Test LPIPS

| G | one_shot | bridge_no_refinement | bridge_refinement | oracle_xi | teacher_forced |
|---:|---:|---:|---:|---:|---:|
| 1.50 | 0.2784 | 0.2784 | 0.2784 | 0.2784 | 0.2784 |
| 2.25 | 0.3826 | 0.3826 | 0.3826 | 0.3826 | 0.3826 |
| 3.00 | 0.4432 | 0.4432 | 0.4432 | 0.4432 | 0.4432 |
| 3.75 | 0.4751 | 0.4751 | 0.4751 | 0.4751 | 0.4751 |
| 4.50 | 0.4795 | 0.4795 | 0.4795 | 0.4795 | 0.4795 |

## 机制解释

- Oracle-ξ 相对 one-shot 的 G=4.50 PSNR 变化：0.0000 dB。
- Bridge no-refinement 相对 one-shot：0.0000 dB。
- Bridge + refinement 相对 one-shot：0.0000 dB。
- G=4.50 Bridge + refinement paired PSNR wins/ties/losses：0/200/0。
- G=4.50 Bridge + refinement 的 LPIPS improvement：0.000000（正值表示优于 one-shot）。
- teacher-forced branch 的完整六级指标见 `test/teacher_forced_branch/summary.json`。

## 过程证据

- 数据退化曲线、replay histogram、range/clipping、物理 descriptor 与 split 分布：`data/places_p2s_bridge_v01/statistics/`。
- 训练 loss、gradient norm、LR、lambda_D 与 val per-g 曲线：`plots/training_losses_gradients_schedule.png`、`plots/validation_metrics_per_g.png`。
- LR/K 冻结曲线：`plots/validation_lr_and_k_selection.png`。
- paired delta histogram、per-step quality/next-state/residual、runtime/NFE/simulator/memory：`plots/`。
- 逐条 paired delta：`test/paired_deltas.jsonl`；完整分 terminal bootstrap 与 residual/runtime 统计：`test/paired_statistics.json`。

## 可复现性与安全

所有 manifest、source/checkpoint hash、环境、逐步 tensor、每步 residual/NFE/runtime、paired rows 和完成标记均保存在本 run 目录。训练和评估仅使用物理 NPU 4–7；数据仅写入 `/data*/yanglin`。数据 tar、必要 checkpoint、完整 ξ 和固定定性样本被保留；只清理 checksum 验证后的临时 spool。

# U=5000 真实阶段交付

四组共同 U=5000，epoch=20，每组 160000 start-record exposures。原完整交付和 FIGURES_AND_TABLES 均已提交、下载并通过全部哈希核验；训练已按授权继续到 U=7500，最终上限仍为 U=10000。

主比较为 val800、RAW / refinement-OFF：

| Method | full train8000 PSNR | val800 PSNR | val800 SSIM | val800 LPIPS | ΔPSNR vs Input |
|---|---:|---:|---:|---:|---:|
| D0 | 26.2923 | 19.6670 | 0.4832 | 0.4449 | -0.8725 |
| R0 | 26.8101 | 19.4339 | 0.4599 | 0.4755 | -1.1055 |
| R08 | 26.4038 | 19.5665 | 0.4795 | 0.4588 | -0.9729 |
| L08 | 26.4790 | 19.3841 | 0.4677 | 0.4688 | -1.1553 |

Input Y：val800 PSNR=20.5394、SSIM=0.5261、LPIPS=0.4860。当前四组验证 PSNR 均低于 Input；四组截至5k的最佳 RAW val PSNR 均在 U=1000。训练/验证表现分离，当前结果不能作为泛化有效或递归改善的证明，5k也不作收敛判断。

R08−L08 的 RAW/OFF 配对差值为 +0.1824 dB，source win-rate=74%，source-cluster bootstrap 95% CI=[0.1074,0.2572]。R08−D0 为 −0.1005 dB，CI=[−0.2057,0.0027]，区间包含0；不能只用 R08−L08 说明全面优于直接恢复。

R0/R08 精修 ON−OFF 分别为 +0.1147/+0.0741 dB，source win-rate=98%/99%。精修属于辅助推理比较，不进入 matched-budget RAW/OFF 主表或 best-val 选择。两卡并行完整 val800 ON invocation 实测约27.5分钟；纯 solver 约25.9/26.1分钟，分别列于表3。

剩余完整交付预计：U7500 13.7–16.5小时；U10000 28.0–33.9小时。包含训练、RAW评价、保存/报告、精修和两卡调度；是工程估计，后续持续以实际耗时更新。

首个正式 optimizer update 至完整5k阶段提交用时约27.2小时（不包含开训前测速和首个update之前的U0工作）。

## 固定八项图表

- [01_train_val_psnr_curve](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/01_train_val_psnr_curve.png>)
- [02_train_val_loss_curve](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/02_train_val_loss_curve.png>)
- [03_progression_by_G](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/03_progression_by_G.png>)
- [04_final_inference_performance_table](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/04_final_inference_performance_table.png>)
- [05_paired_method_ablation_table](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/05_paired_method_ablation_table.png>)
- [06_refinement_ablation_table](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/06_refinement_ablation_table.png>)
- [07_final_inference_visual_comparison](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/07_final_inference_visual_comparison.png>)
- [08_refinement_visual_comparison](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/FIGURES_AND_TABLES/08_refinement_visual_comparison.png>)

三图另有PDF；三表另有CSV/Markdown。未生成PPT/PPTX，没有为整理重新运行模型或精修。

本地原交付文件1786项、新包23项全部哈希一致，8张PNG与3张PDF均已视觉检查。原80张横/竖固定图完整保留。原纵图标题重叠另以 [CPU排版修正副本](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/reports/U05000_VERTICAL_LAYOUT_CORRECTION/README.md>) 修正，使用全部40个原冻结start的已有FP32输出，原证据和主包不覆盖。

[原阶段说明](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/stage_deliveries/U05000/README.md>) · [完整性与视觉审计](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/reports/U05000_LOCAL_DELIVERY_AUDIT.json>) · [本次跟进证据](</Users/lucas/Desktop/turbulence IR/v4_train500_matched_v2_perf1_eval5kfull/reports/HEARTBEAT_20260930T215306Z.json>)

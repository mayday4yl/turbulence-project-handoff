# D0 5-source / 40-epoch fitting diagnostic

已完成：仅 D0，从原共同随机初始化训练 40 epoch；80 starts/epoch，共 3200 exposures、400 optimizer updates。
每个 epoch 在同一 raw checkpoint 下评估 train80 与固定 val64；三指标使用原始 FP32 输出。无 refinement，无参数搜索。

源图：Places365_val_00005914.jpg, Places365_val_00030817.jpg, Places365_val_00013160.jpg, Places365_val_00012149.jpg, Places365_val_00014491.jpg

| Population | Output | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Total loss |
|---|---|---:|---:|---:|---:|
| train80 | input | 21.273289 | 0.586683 | 0.481968 | — |
| train80 | raw | 22.773847 | 0.620750 | 0.502949 | 0.007438328 |
| train80 | ema | 16.276225 | 0.462863 | 0.655949 | 0.026771125 |
| val64 | input | 19.583533 | 0.482851 | 0.496491 | — |
| val64 | raw | 17.534017 | 0.453970 | 0.623281 | 0.020991160 |
| val64 | ema | 15.633327 | 0.401235 | 0.699440 | 0.033237060 |

## 解释范围

本轮用于诊断小训练集拟合能力。train80 是全部训练起点，val64 来自不同 source，只是固定验证子集，不是 val800。40 epoch 对应 400 updates；不能与旧实验按 epoch 数直接等同。
主图为 raw；EMA 仅作末期附录，其曝光累计较少可能明显滞后。历史 train100 统计 buffer 只保留身份，D0 不用 mu/s 构造输入。
每步在线 train loss 和每个 epoch 重新评估的 train loss 分别保存在 online_training_loss.csv、epoch_metrics.csv。后者与 val loss 口径一致。
指标：RGB PSNR range=1；SSIM Gaussian 11/sigma1.5/population covariance；LPIPS AlexNet v0.1。没有预处理 clamp/resize/PNG 量化。

训练 update 累计时长：207.36 s；训练+评估+checkpoint 墙钟：497.84 s。
验收：AUDIT.json=PASS；真实双卡 batch8 相对逐样本梯度见 implementation/reports/preflight.json；验收 optimizer step=0。

[图像原位置：Train and validation](<../../../../NOT_INCLUDED.md#ref-d3d99d4750f92dca07e0>)

复现入口：独立 implementation 内 `bash scripts/run_all.sh`；存在训练日志时拒绝重复开跑。协议为 protocol/FROZEN.json。

更正后的训练图命名：Train source 01–05；[查看训练结果图](<../../../../NOT_INCLUDED.md#ref-0be19bb1db41766bea12>)；[源文件对应表](<../../../../NOT_INCLUDED.md#ref-764318ca7cd8a93c7369>)。

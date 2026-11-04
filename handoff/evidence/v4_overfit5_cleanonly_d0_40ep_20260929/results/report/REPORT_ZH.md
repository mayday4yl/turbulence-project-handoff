# Clean-only D0 5-source / 40-epoch fitting diagnostic

已完成：仅 clean-only D0，L=clean MSE，从原共同随机初始化训练 40 epoch；80 starts/epoch，共 3200 exposures、400 optimizer updates。
每个 epoch 在同一 raw checkpoint 下评估 train80 与固定 val64；三指标使用原始 FP32 输出。无 refinement，无参数搜索。

源图：Places365_val_00005914.jpg, Places365_val_00030817.jpg, Places365_val_00013160.jpg, Places365_val_00012149.jpg, Places365_val_00014491.jpg

| Population | Output | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Clean-only loss |
|---|---|---:|---:|---:|---:|
| train80 | input | 21.273289 | 0.586683 | 0.481968 | — |
| train80 | raw | 22.590230 | 0.617636 | 0.504719 | 0.007589480 |
| train80 | ema | 16.316924 | 0.464728 | 0.659877 | 0.023872476 |
| val64 | input | 19.583533 | 0.482851 | 0.496491 | — |
| val64 | raw | 17.549631 | 0.453242 | 0.625304 | 0.019606015 |
| val64 | ema | 15.646254 | 0.402295 | 0.701362 | 0.030003751 |

## 解释范围

本轮用于诊断小训练集拟合能力。train80 是全部训练起点，val64 来自不同 source，只是固定验证子集，不是 val800。40 epoch 对应 400 updates；不能与旧实验按 epoch 数直接等同。
本轮完全跳过xi分支和simulator；xi参数冻结且不进入optimizer。与旧D0的主体/clean路径初始化及数据、顺序、batch、LR、评估完全配对。
主图为 raw；EMA 仅作末期附录，其曝光累计较少可能明显滞后。历史 train100 统计 buffer 只保留身份，D0 不用 mu/s 构造输入。
每步在线 train loss 和每个 epoch 重新评估的 train loss 分别保存在 online_training_loss.csv、epoch_metrics.csv。后者与 val loss 口径一致。
指标：RGB PSNR range=1；SSIM Gaussian 11/sigma1.5/population covariance；LPIPS AlexNet v0.1。没有预处理 clamp/resize/PNG 量化。

训练 update 累计时长：74.53 s；训练+评估+checkpoint 墙钟：300.48 s。
验收：AUDIT.json=PASS；真实双卡 batch8 相对逐样本梯度见 implementation/reports/preflight.json；验收 optimizer step=0。

[图像原位置：Train and validation](<../../../../NOT_INCLUDED.md#ref-d3d99d4750f92dca07e0>)

复现入口：独立 implementation 内 `bash scripts/run_all.sh`；存在训练日志时拒绝重复开跑。协议为 protocol/FROZEN.json。

## 与上一轮 D0 的配对比较

| Population | Method | PSNR | SSIM | LPIPS | Clean MSE |
|---|---|---:|---:|---:|---:|
| train80 | input | 21.273289 | 0.586683 | 0.481968 | 0.010217412 |
| train80 | old_D0 | 22.773847 | 0.620750 | 0.502949 | 0.007313226 |
| train80 | clean_only | 22.590230 | 0.617636 | 0.504719 | 0.007589480 |
| val64 | input | 19.583533 | 0.482851 | 0.496491 | 0.013420782 |
| val64 | old_D0 | 17.534017 | 0.453970 | 0.623281 | 0.019750237 |
| val64 | clean_only | 17.549631 | 0.453242 | 0.625304 | 0.019606015 |

同样只给400次更新。结果回答该固定预算内是否改善拟合；不能直接证明网络容量上限。

[图像原位置：Fixed paired images](<../../../../NOT_INCLUDED.md#ref-59a6f446c46ee502d7ca>)

[图像原位置：Paired curves](<../../../../NOT_INCLUDED.md#ref-1527b5d6a6a488a072b3>)

更正后的训练图命名：Train source 01–05；[查看训练结果图](<../../../../NOT_INCLUDED.md#ref-59a6f446c46ee502d7ca>)；[源文件对应表](<../../../../NOT_INCLUDED.md#ref-764318ca7cd8a93c7369>)。

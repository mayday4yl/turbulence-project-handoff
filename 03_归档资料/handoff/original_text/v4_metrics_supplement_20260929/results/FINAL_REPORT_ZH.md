# V4 完整 val800 补充评估：最终结论

全部 9 项（输入 Y；D0/R0/R08 raw 与 EMA，全部 refinement-off；R0/R08 raw refinement-on）已各完成 800 个 start。最终浮点/协议审计通过，3,600 个 FP32 输出文件的 SHA256 已核验，22 个预先固定图像对照已生成。评估进程已退出，GPU5/6 已空闲。本轮没有训练或重新选择 checkpoint。

在这轮固定 last_32k 的 val800 上，所有模型设置的 PSNR、SSIM、LPIPS 均未超过输入 Y。主比较中，R08 相比 R0 三指标都改善；相较 D0，raw R08 的 SSIM 略高，PSNR 略低且 LPIPS 更差。EMA 中 R08 的 PSNR/SSIM 最好，D0 的 LPIPS 最好。

逐样本 refinement 对两组的三指标都带来平均改善：R0 的 ΔPSNR/ΔSSIM/ΔLPIPS = +0.211692 / +0.011246 / −0.015509；R08 = +0.081215 / +0.003949 / −0.005586。仍未超过 Y。这些 on/off 比较单独列出，不混入 refinement-off 主表。

主表全部 5,600 个逐样本 PSNR 与原结果完全相同。Refinement-on 重放的 PSNR 最大绝对差异为 R0 0.008640dB、R08 0.002284dB；平均有符号差异分别 +0.00001148dB、−0.00000033dB。逐样本差异均保留，新表三指标均来自本次相同浮点输出，没有拼接旧 PSNR。

[原样收回的机器报告](REPORT_ZH.md)、[最终审计](AUDIT.json)、[逐样本指标 CSV](per_start_metrics.csv)、[全部配对差值](PAIRED_METRICS.json)、[补充 EMA−raw 配对差值](MAIN_PAIRED_METRICS.json)、[固定图像索引](VISUAL_INDEX.json)、[浮点输出哈希清单](FLOAT_OUTPUT_MANIFEST.json)。完整浮点输出留在服务器；本地保留全部标量、日志、固定参考数组与图像。

以下为机器生成表格、指标口径及固定图像。

# V4 last_32k 完整 val800 补充评估

范围：100 个独立 clean source × 2 个 realization × 4 个 G，共 800 个 start；均为既有 last_32k checkpoint。训练 optimizer step = 0。
三指标均在同一份未经量化或 clamp 的 FP32 输出上计算；所有输出另存 NPY 并核验 SHA256。正式推理沿用原固定 batch=2、样本分组、eta seed，refine 始终逐样本执行。

## 主比较：全部关闭 refinement

| 结果 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Y | 20.539433 | 0.526077 | 0.485965 |
| D0_raw_off | 19.198257 | 0.448009 | 0.604885 |
| D0_ema_off | 19.954701 | 0.495130 | 0.584217 |
| R0_raw_off | 18.880515 | 0.427755 | 0.634793 |
| R0_ema_off | 19.719150 | 0.479047 | 0.603684 |
| R08_raw_off | 19.189480 | 0.450221 | 0.625278 |
| R08_ema_off | 19.982513 | 0.496818 | 0.590685 |

## 单独的 raw refinement 比较

| 结果 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| R0_raw_off | 18.880515 | 0.427755 | 0.634793 |
| R0_raw_on | 19.092207 | 0.439001 | 0.619284 |
| R08_raw_off | 19.189480 | 0.450221 | 0.625278 |
| R08_raw_on | 19.270695 | 0.454170 | 0.619691 |

## 配对差值（A − B）

PSNR / SSIM 正值更好，LPIPS 负值更好。先在每个 source 内平均 8 个 start，再平均 100 个 source。改善数以独立 source 为单位。

| A − B | ΔPSNR | ΔSSIM | ΔLPIPS | 改善 source 数 P/S/L（共100） |
|---|---:|---:|---:|---|
| D0_raw_off minus Y | -1.341176 | -0.078068 | +0.118920 | 0/0/0 |
| R0_raw_off minus Y | -1.658918 | -0.098322 | +0.148828 | 0/0/0 |
| R08_raw_off minus Y | -1.349953 | -0.075856 | +0.139313 | 0/0/0 |
| R0_raw_off minus D0_raw_off | -0.317742 | -0.020254 | +0.029907 | 14/4/2 |
| R08_raw_off minus R0_raw_off | +0.308965 | +0.022466 | -0.009515 | 89/93/61 |
| R08_raw_off minus D0_raw_off | -0.008777 | +0.002211 | +0.020392 | 46/53/18 |
| D0_ema_off minus Y | -0.584731 | -0.030947 | +0.098252 | 5/9/0 |
| R0_ema_off minus Y | -0.820283 | -0.047030 | +0.117719 | 1/2/0 |
| R08_ema_off minus Y | -0.556920 | -0.029259 | +0.104720 | 5/7/0 |
| R0_ema_off minus D0_ema_off | -0.235551 | -0.016083 | +0.019466 | 16/4/5 |
| R08_ema_off minus R0_ema_off | +0.263362 | +0.017771 | -0.012999 | 87/95/82 |
| R08_ema_off minus D0_ema_off | +0.027811 | +0.001688 | +0.006468 | 49/54/31 |
| R0_raw_on minus R0_raw_off | +0.211692 | +0.011246 | -0.015509 | 100/99/100 |
| R08_raw_on minus R08_raw_off | +0.081215 | +0.003949 | -0.005586 | 98/98/99 |

## 与原 PSNR 的重放差异

| 结果 | 平均有符号 ΔdB | 最大绝对 ΔdB |
|---|---:|---:|
| Y | 0 | 0 |
| D0_raw_off | 0 | 0 |
| D0_ema_off | 0 | 0 |
| R0_raw_off | 0 | 0 |
| R0_ema_off | 0 | 0 |
| R08_raw_off | 0 | 0 |
| R08_ema_off | 0 | 0 |
| R08_raw_on | -3.33661663e-07 | 0.00228402841 |
| R0_raw_on | 1.1484289e-05 | 0.00864027576 |

新表保留本次同源三指标；原结果未覆盖。`per_start_metrics.csv` 同时记录原 PSNR、本次 PSNR 和差值。Refine 涉及 CUDA backward，差异按实测列出，不用旧 PSNR 拼接新指标。

## 指标口径与证据

PSNR：与原 evaluator 一致，RGB 全像素 FP32 MSE，data_range=1。SSIM：scikit-image 0.25.2，RGB 通道平均，11×11 Gaussian、sigma=1.5、population covariance、data_range=1、float64；窗口统计后排除 5 像素边缘。设置参见 [官方 SSIM 文档](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html#skimage.metrics.structural_similarity)。
LPIPS：官方 lpips 0.1.4，AlexNet v0.1，固定 pretrained trunk / learned weights，eval-mode，FP32，每次一个样本；normalize=True 做 2x−1。定义参见 [官方实现](https://github.com/richzhang/PerceptualSimilarity)。不 resize、clamp 或逐图标准化。
原 PyTorch 2.7.1+cu126、CUDA 12.6、cuDNN 90501 保持不变；metric 依赖安装在独立 deps。完整软件版本、源文件与权重哈希见补充实现 `protocol/FROZEN.json`。

可复现：在补充实现目录执行 `bash scripts/launch.sh validate`（已有同身份检查可跳过），`bash scripts/launch.sh run`，再以相同 PYTHONPATH=deps 和原 Python 执行 `scripts/report.py`。不要在同名任务运行时重复启动。
证据：`AUDIT.json`、`PAIRED_METRICS.json`、`FLOAT_OUTPUT_MANIFEST.json`、各 case 的 `results.json`、补充协议与 validation.json。

## 固定图像对照

样本在指标计算前固定：原 val64 的前 8 个 start，以及该固定子集全部 G=4 的 start，去重；未按好坏筛选。PNG 仅展示使用，指标来自浮点数组。各图统一列序和显示范围。

val_s0537_r00::G1 (G=1)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/00_G1.png>)

val_s0537_r00::G2 (G=2)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/01_G2.png>)

val_s0537_r00::G3 (G=3)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/02_G3.png>)

val_s0537_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/03_G4.png>)

val_s0521_r00::G1 (G=1)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/04_G1.png>)

val_s0521_r00::G2 (G=2)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/05_G2.png>)

val_s0521_r00::G3 (G=3)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/06_G3.png>)

val_s0521_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/07_G4.png>)

val_s0562_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/08_G4.png>)

val_s0565_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/09_G4.png>)

val_s0556_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/10_G4.png>)

val_s0528_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/11_G4.png>)

val_s0531_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/12_G4.png>)

val_s0596_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/13_G4.png>)

val_s0579_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/14_G4.png>)

val_s0548_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/15_G4.png>)

val_s0570_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/16_G4.png>)

val_s0586_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/17_G4.png>)

val_s0513_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/18_G4.png>)

val_s0503_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/19_G4.png>)

val_s0567_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/20_G4.png>)

val_s0518_r00::G4 (G=4)
![固定对照](</Users/lucas/Desktop/turbulence IR/v4_metrics_supplement_20260929/results/visuals/21_G4.png>)

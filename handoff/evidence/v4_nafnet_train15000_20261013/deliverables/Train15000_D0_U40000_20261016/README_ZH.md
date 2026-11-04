# Train15000 D0 U40000 正式结果

D0已完成从自身U20000追加20,000 updates，新增640,000 exposures，总1,280,000 exposures。冻结终点为global U40000。原worker已自然退出，锁释放。online为主结果，EMA为独立次要结果，均为refinement OFF。四法同预算比较等待其余三法完成。

## 主fullVal8000

同一原序8,000 inputs / 1,000 clean sources，每源8，每G2,000，Places365/LSDIR各4,000 inputs。Test未使用。

| 权重 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| online | 23.450488489 | 0.665960092 | 0.351608030 |
| ema | 23.452903633 | 0.665954301 | 0.352022130 |

两套正式结果分别完成250 parts的独立CPU验收，以及本机原序人口、逐文件SHA、8,000行CSV和FP64整体/各G/各域复算。模型保存已于前轮单次CPU验收，本轮复用该凭据。

## ValDetail500 补充人口

按原冻结STARTS与canonical双射从各自完整正式CSV提取原序子序列。全部G为500 sources / 4,000 inputs，G4为1,000 inputs。主表与补充人口分别报告。

| 权重 | 人口 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| online | ValDetail500_all_G | 22.557385382 | 0.633998850 | 0.377270621 |
| online | ValDetail500_G4 | 20.876111679 | 0.529997816 | 0.505327494 |
| ema | ValDetail500_all_G | 22.559200524 | 0.633979912 | 0.377667752 |
| ema | ValDetail500_G4 | 20.877369553 | 0.530016002 | 0.505670184 |

两套补充集各通过人口/原序、FP64、CSV/hash及本机4,000行复核。精确值、逐输入指标、各G/域结果和验收证据见包内CSV/JSON。

## 保存和执行证据

完整保存SHA256：`d0a66bb030a73527f0c86e9cd642fe8e704ac5316f48515fcbbf546b9711173b`。707 model / 707 EMA tensors / 704 AdamW states，step40000，原mu/s/RNG和data/resource/S5身份已验收。冻结终点next_ids为空、next_lr为None。

主训练R08已由原controller在D0自然完成后接入腾出的GPU7，其首次独立CPU父迁移和既有worst-G4峰值学习率GPU门禁另有验收证据。R0/R08继续，L08待排队。该进度是本轮快照，不表示其最终结果已经完成。

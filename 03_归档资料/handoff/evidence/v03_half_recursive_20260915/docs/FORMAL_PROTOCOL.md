# V0.3-half 正式训练协议（已冻结）

## 身份与范围

- 服务器：liteserver-de36；仅物理 NPU 0–3。
- 远端项目：`/data2/yanglin/v03_half_recursive_20260915`；正式 run：`runs/half_formal_01`。
- 训练源码提交：`edf3dbcd5d93941c31303710a07b2120c7a7975d`。
- 配置 canonical SHA-256：`d5b08f8516e317983cab6ae9c3f59ad36b634da899fc50b99f41da6f4f0511cc`。
- 原 V0.1 network/simulator/xi/inference 保持源码哈希不变。数据仅使用 source-level half view。
- 正式 seed=20260915，sampler seed=2026091503。全新 model/EMA/optimizer/scheduler/sampler；不复用任何 pilot 权重。

## 正式配置

| 项目 | 冻结值 |
|---|---|
| lambda_R | 0.2 |
| per-device microbatch | 上限 2 |
| actual global paths/update | 32 |
| gradient accumulation | 按实际平衡后 chunks，通常 4–5 |
| optimizer | AdamW，betas=(0.9,0.99)，weight decay=1e-4 |
| LR | 64 updates warmup 至 1e-4，cosine 至最大 10k 时的 1e-6 |
| grad clip | 1.0 |
| EMA | 0.996005996001，formal validation 固定使用 EMA |
| curriculum | 0–127 短路径偏重；128–511 中等；512 起长路径充分曝光 |
| 预算 | plateau 最早 4k；主预算 8k；满足预声明条件可到 10k 上限 |
| validation | 每 1k 固定 16 trajectories fast；每 2k 优先全 200 trajectories |
| inference coverage | 全六档 G，K=0/1/3/5，原始完整 refinement |

EMA 根据旧配置在 dataset passes 中的窗口计算 .999**4；这不是在本次短 pilot 中调优出的值。curriculum 使用 pilot 的同一组权重表，完整 conditional path probabilities 保存在 freeze JSON。每条 path 按 prediction count 平均，再对 global paths 等权平均。

## 固定预算与实测估时

| updates | sampled trajectories | network predictions | 训练小时 | 含 validation 小时 | 估算 NPU-hours |
|---:|---:|---:|---:|---:|---:|
| 4000 | 128000 | 291287 | 36.9 | 39.2 | 156.7 |
| 6000 | 192000 | 442273 | 55.9 | 59.4 | 237.4 |
| 8000 | 256000 | 592824 | 74.9 | 79.5 | 318.1 |
| 10000 | 320000 | 742949 | 93.9 | 99.7 | 398.9 |

以上是四卡持续可用时的累计运行估算，不含等待空闲、中午暂停和后续未授权的停机间隔。validation 耗时可能随 refinement early stopping 改变。预计今天中午前约 180–210 updates，不能达到 2k，也不代表已收敛。

## Checkpoint、停止与续训

- 每 500 updates 完整保存；2k 倍数永久保留；其他常规 checkpoint 保留最近 3 个。
- 保存 model、EMA、optimizer、scheduler、update、curriculum、sampler 和各 rank RNG。实际 save→resume、加载后逐组件 hash、EMA/DDP 同步均已通过。加载后数值状态一致不等同于保证 NPU 后续执行逐比特可复现。
- 今天 11:30 在完整 update 边界保存安全 checkpoint，优先做 fast validation，再原样恢复至 11:50 左右保存 resume_ready 并暂停。
- 暂停不等于收敛；中午后不自动续跑。后续获授权后从同一完整 checkpoint 继续，保持原 LR horizon 和 curriculum counter。
- plateau：4k 起，连续三个常规 1k validation 间隔未比此前最好 macro PSNR 改善至少 0.05 dB。
- 8k 继续条件：7k/8k 最好 macro PSNR 比截至 6k 的最好结果提高至少 0.05 dB，且当前 K0 距最好 K0 不超过 0.10 dB；缺少可比 validation 不授权延长。
- 相同 fast subset 上 K0 下降超过 3 dB 则停止审查。正式 run 不临时调整 lambda、curriculum、LR 或预算。

## Checkpoint 选择与结论边界

- 仅比较相同 validation subset、相同 EMA 权重类型的结果。最终推荐需要完整 200-trajectory validation。
- 先要求 K0 PSNR 距该集合最好 K0 不超过 0.10 dB，再按六档 G 和四个 K 等权的绝对 PSNR 排序。0.03 dB 以内按 LPIPS；LPIPS 差异不超过 0.005 时优先更早 checkpoint。
- progressive gate：同一 checkpoint 的 K1/3/5 平均 PSNR 相对 K0 至少 +0.10 dB；至少四档 G 为正、无一档低于 -0.20 dB；LPIPS 不恶化超过 0.005、SSIM 不恶化超过 0.001，并满足 K0 preservation。
- next-state MSE 与 observation residual 仅作机制诊断。必须保留每个 G/K 的 PSNR、SSIM、LPIPS、L1 与 Delta。
- 只有达到预声明训练结束条件、完成 full validation/checkpoint selection、满足上述 gates 和完整性检查后，才能报告 ready_for_frozen_test。当前不能报告。
- 短 pilot 选择 lambda=0.2，但其 K1/3/5 仍弱于 K0。没有正向 progressive gain 证据；不能把该选择写成方法成功。
- 结论覆盖完整 V0.3 protocol；单独归因于 recursive rollout 仍需 matched ablation。
- 不执行 held-out final test。

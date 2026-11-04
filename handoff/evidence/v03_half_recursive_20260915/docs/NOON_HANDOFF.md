# V0.3-half 中午安全交接

**已按约定于 2026-09-16 11:50:24（北京时间）安全暂停，完成 204 optimizer updates。正式训练尚未完成，尚未达到 frozen test 条件。**

## 1. 当前状态与恢复点

授权的 0–3 卡到 06:57 才可用；随后完成验收、四卡 benchmark 和公平 pilots，10:24 启动正式训练。正式运行窗口约 86 分钟，因此中午留下的是早期可恢复模型。

- de36 物理 NPU 0–3 已释放；11:56 实时检查四卡均无运行进程，健康状态 OK。其他人的任务未被停止。
- 正式训练累计 6,528 条 sampled trajectories、10,098 次 network predictions，约 1.632 个 trajectory-equivalent passes。
- 11:30 保存 update 173；完成 fast validation 后，四卡实际加载恢复审查全部通过，并继续训练到 update 204。
- 推荐用于续训的完整 checkpoint：`/data2/yanglin/v03_half_recursive_20260915/runs/half_formal_01/checkpoints/update-000204.pt`。
- checkpoint SHA-256：`8bb92a45c1fca32d4a8692a909dfc54120ad24bf5ee32ef9c28588c949c43b92`。
- 模型、EMA、optimizer、scheduler、update、curriculum、sampler、四 rank RNG 全部保留；21 项最终 CPU 审查全部通过。实际 173 save→resume 证明加载状态一致；不声称后续 NPU 运算逐比特可复现。
- 续训时继续原 10k scheduler horizon、原 curriculum 和采样序列。下一 update index=204，下一 global sample ID=6528。显式恢复入口需要 `--resume` 指向上述文件并带 `--continue-after-pause`。
- 自动跟进已暂停。没有在中午后启动续训；没有执行 held-out test。

## 2. 数据与训练身份

| split | sources | trajectories | degraded states |
|---|---:|---:|---:|
| train | 1000 | 4000 | 24000 |
| val | 100 | 200 | 1200 |
| test | 250 | 500 | 3000 |

固定 seed=2026091504 按可靠类别信息分层、在 source 层抽样；完整保留每个 train source 的四条、val/test source 的两条 trajectory。六档状态、clean、xi 均来自原始数据。47,000 个成员全部读取，42,300 个原始 SHA 校验匹配；source overlap=0、trajectory key duplicate=0，原数据未改动。test 仅做完整性核验。

- 独立项目：`/data2/yanglin/v03_half_recursive_20260915`；dataset view：`data/places_p2s_bridge_v03_half`。
- 旧 full-data V0.3 因 dataset_protocol_change 终止，旧权重没有进入本次正式训练。
- 正式 seed=20260915，sampler seed=2026091503；模型/EMA/optimizer/scheduler/counters 全新初始化。
- 训练源码 git commit：`edf3dbcd5d93941c31303710a07b2120c7a7975d`；源码和 V0.1 基线 hashes 在最终审查中仍一致。

## 3. 冻结配置与依据

- 四卡；每卡 microbatch 上限 2；global/effective batch=32 paths；按路径长度分配各 rank 工作；loss 权重保持 path-wise mean → global-path mean。
- 本次实际 gradient accumulation microsteps 集合：`[4, 5]`。
- λ_R=0.2。AdamW，LR=1e-4，64 updates warmup，cosine 至固定最大 10k 时的 1e-6；betas=(.9,.99)，weight decay=1e-4，grad clip=1。
- EMA=.999**4≈.996006，formal validation 固定用 EMA。这按旧配置的 dataset-pass 平均窗口推导，未在短 pilot 中调优。
- curriculum：前 128 updates 短路径偏重，128–511 中等，512 起使用充分长路径曝光的最终分布。最后阶段 one-shot 总概率约 .412、至少三次 transition 的总概率约 .271；G=4.5 的完整 K5 条件概率 .30。
- 4k 是 plateau 停止的最早检查点，8k 主预算，满足预声明改善与 K0 保持条件才能继续到 10k 上限。今天的停止是运行时间边界。
- 完整科学定义、采样概率、停止/选择/gate 规则见 [冻结协议](<FORMAL_PROTOCOL.md>) 和 [freeze JSON](<../../../NOT_INCLUDED.md#ref-8aad898fd7094991ce9c>)。

公平 pilot：每个候选相同初始化、数据/path 顺序、optimizer/LR/batch/curriculum、64 updates/2048 trajectories/4156 predictions；同一 8-trajectory raw-model validation 在 32/64 updates 执行。12 个训练后跨步梯度检查通过。

| λ_R | K0 PSNR | 全 G/K 绝对 PSNR | LPIPS |
|---:|---:|---:|---:|
| 0.05 | 19.8483 | 18.8935 | 0.7178 |
| 0.1 | 19.8139 | 18.8958 | 0.7184 |
| 0.2 | 20.0233 | 19.2318 | 0.7121 |

0.2 按预声明规则被选中，绝对 K0 和综合质量均较好，数值稳定。所有 pilot 仍未出现正向 progressive gain；选择配置不等于证明方法有效。没有从 pilot checkpoint 接续正式训练。

## 4. 四卡吞吐与成本

以下 mixed 对比使用同一 global32 数据/path workload；稳态剔除首轮和 checkpoint 更新，计入 data loading、network、simulator、backward、DDP 和 optimizer。

| 配置 | 秒/update | trajectories/s | useful predictions/s | allocated / reserved 峰值 GiB |
|---|---:|---:|---:|---:|
| micro1 | 54.42 | 0.588 | 1.372 | 14.16 / 15.01 |
| micro2 | 44.65 | 0.717 | 1.672 | 20.05 / 22.27 |
| micro2 + length balancing | 32.73 | 0.978 | 2.281 | 24.20 / 25.44 |

较大的 microbatch4 在有限诊断预算内未稳定完成 update；microbatch8 OOM，均已安全释放并保留日志。原因尚未完全定位，不能宣称当前硬件效率已达到理论最优。当前选择依据是已稳定完成的最高实测吞吐。

独立长度成本：global8、micro2、四卡，完整数据读取→EMA；同一长度的首 update 排除。该表只作长度成本诊断。

| 实际 predictions/path | 秒/update | trajectories/s | predictions/s | allocated / reserved GiB |
|---:|---:|---:|---:|---:|
| 1 | 3.27 | 2.448 | 2.448 | 5.13 / 6.72 |
| 2 | 6.58 | 1.216 | 2.433 | 9.48 / 11.14 |
| 3 | 9.87 | 0.810 | 2.431 | 13.77 / 15.51 |
| 4 | 13.20 | 0.606 | 2.425 | 18.05 / 19.85 |
| 5 | 16.50 | 0.485 | 2.425 | 22.32 / 24.20 |
| 6 | 19.79 | 0.404 | 2.425 | 26.60 / 28.56 |

正式 train+fast validation 实际约 **5.77 NPU-hours**；本次 half-data 全部 guarded benchmark/pilot/验收/正式阶段合计约 **18.72 NPU-hours**，包含失败工程尝试。口径为四卡×guard wall time，含启动/退出开销；不包含 CPU-only 等卡时间和旧 full-data 项目。

## 5. 正式训练曲线

[图像原位置：训练曲线](<../../../NOT_INCLUDED.md#ref-ec2274322f02060545c0>)

淡线为原始每 update 值，深线为明确标注的最近九个 update 中位数。不同 update 的 G/path 会变化，训练 loss 曲线不能代替 validation 或收敛判定。

## 6. 可用 validation

update 173 的固定 16 条 val trajectories，六档 G、四个 K，共 384 个记录；EMA，原 V0.1 完整 inference。

| K | PSNR | SSIM | LPIPS | L1 | ΔPSNR vs K0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 17.2925 | 0.5040 | 0.6794 | 0.10999 | +0.0000 |
| 1 | 16.1409 | 0.4791 | 0.7559 | 0.12833 | -1.1516 |
| 3 | 15.3683 | 0.4622 | 0.7766 | 0.14261 | -1.9242 |
| 5 | 15.2162 | 0.4594 | 0.7792 | 0.14567 | -2.0763 |

**当前多步结果弱于 K0，没有 progressive gain。** 目前只有一个早期 formal EMA validation；未完成主要训练、full validation 或最终 checkpoint selection。
EMA 初始参数的平均残留系数在 update173 为 0.5004。这解释了早期 EMA 的一个限制，但不能证明后续必然改善。pilot 的 raw model/8条子集/不同seed与这里的 EMA/16条不能直接横比。

| K | next-state MSE（path mean） | refinement 后 observation residual（path mean） |
|---:|---:|---:|
| 1 | 0.011094 | 0.005264 |
| 3 | 0.013317 | 0.008154 |
| 5 | 0.014175 | 0.008990 |

[完整 per-G/K 指标 CSV](<../artifacts/noon_handoff/validation_metrics_per_G_K.csv>) · [validation 原始汇总](<../../../NOT_INCLUDED.md#ref-d214dfe7fb645b993bcb>) · [训练数据 CSV](<../../../NOT_INCLUDED.md#ref-9c9a804a151b48ba8280>)

## 7. 后续与结论边界

- 续训恢复点推荐 update 204；已有验证的快照是 update 173。最终性能 checkpoint 暂不推荐，需完成冻结的 full-validation 选择流程。
- 后续获得续训指令后，从 update204 恢复同一正式实验，保留所有状态、LR horizon 和预冻结预算规则；当前不会自动继续占卡。
- 当前不满足 ready_for_frozen_test；不运行 held-out test。
- 本实验评价完整 V0.3 training protocol。即使后续成功，也仍需 matched ablation 才能将收益单独归因于 recursive rollout。

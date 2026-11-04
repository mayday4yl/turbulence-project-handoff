# V0.31 后续实验交付

T1_DIRECT 与 T2_FEEDBACK_R 各完成1000 updates并暂停，保留完整续训状态。I1完成预声明有限候选，开发集不支持任何新修复候选，按规则停止。

| raw1000任务 | G4 ΔPSNR | G3 ΔPSNR | G2 ΔPSNR | G1 ΔPSNR |
|---|---:|---:|---:|---:|
| T1_DIRECT | 0.793930 | 0.944353 | 0.976816 | 0.719382 |
| T2_FEEDBACK_R | -0.237647 | -0.055815 | 0.046907 | -0.049376 |

上表为原训练集1600起点相对旧 raw1000 无 refinement 的最终配对平均PSNR增益，单位dB。逐步曲线、source级中位数/胜负和原始FP32指标另存，不从PNG算指标。

## 成本与检查点

| 任务 | microbatch/accumulation | 完整update秒数合计 | 训练墙钟秒数 |
|---|---|---:|---:|
| T1_DIRECT | 8/1 | 344.10 | 367.26 |
| T2_FEEDBACK_R | 1/8 | 38497.21 | 38526.47 |

T1_DIRECT：`/data2/yanglin/v031_followup_20260918/T1_DIRECT/formal/checkpoint_001000.pt`；SHA256 `46cf6879e587089d645c8de767b128d48d585cb981bb6f0551c28808f9d7d7ff`。


T2_FEEDBACK_R：`/data2/yanglin/v031_followup_20260918/T2_FEEDBACK_R/formal/checkpoint_001000.pt`；SHA256 `80cbb6bd56d7c941395a833a56c9389fe1943d712110b495f65c4e6d0d4cae59`。

同1000 updates代表每个起点20次曝光，不能称为等算力。网络/模拟器计数、实际配置、资源保护记录和每个checkpoint身份随包保存。没有独立训练seed，不能宣称多seed稳定性。

## Heldout32

固定32个未见train-pool source、2组xi/source、共256起点，仅在raw1000评估。它是同分布新场景筛查，不是外部benchmark。

| 任务 | G4 ΔPSNR | G3 ΔPSNR | G2 ΔPSNR | G1 ΔPSNR |
|---|---:|---:|---:|---:|
| T1_DIRECT | 0.747673 | 0.562782 | 0.263565 | -0.543606 |
| T2_FEEDBACK_R | -0.077032 | -0.033302 | -0.002834 | -0.269635 |

## 文件与边界

I1_REPORT.md报告所有新候选与首跳诊断。TRAINING_REPORT.md、TRAINING_PAIRED_SUMMARY.json和HELDOUT_SUMMARY.json保存训练对照。固定示例沿用旧样本并加入预先固定的5个source。大checkpoint和FP32数组保留在de36新实验目录；数组路径、大小和hash另存。旧V0.31未续训，未覆盖旧数据、源码或结果。

I1确认集与候选heldout没有运行，因为没有开发候选过门槛；这是预声明停止规则，不能把最接近U0的负增益候选称为成功修复。

完整辅助对照、输入/旧K1/自身K1/T1配对及source统计见 [ALL_PAIRED_COMPARISONS.md](ALL_PAIRED_COMPARISONS.md)。[CHECKPOINT_AND_RESUME.md](CHECKPOINT_AND_RESUME.md) 提供完整状态位置与恢复入口；SOURCE_CHANGES.patch、各CONFIG_CHANGES.patch保留源码/配置差异；MISSINGNESS.json列明未运行分支。训练曲线在figures/training_update_*，全部合法K的固定图像在figures/all_steps_raw500与all_steps_raw1000。

## 未见source的绝对质量与结论

| heldout32 raw1000 | G4 PSNR | G3 PSNR | G2 PSNR | G1 PSNR |
|---|---:|---:|---:|---:|
| 输入Y | 18.833153 | 19.497087 | 20.479326 | 22.277710 |
| 旧模型 K1 | 17.835720 | 18.420005 | 19.225096 | 20.784585 |
| 旧模型 FINAL | 16.682420 | 17.431869 | 18.504187 | 20.784585 |
| T1 K1 | 17.430093 | 17.994651 | 18.767752 | 20.240979 |
| T2 K1 | 17.899242 | 18.450881 | 19.189448 | 20.514950 |
| T2 FINAL | 16.605387 | 17.398567 | 18.501353 | 20.514950 |

在这个预冻结heldout32上，三种模型各G组的最终平均PSNR都低于输入Y；相对旧模型的正增益不能等同于相对输入的有效恢复，也不能把训练集提升称为泛化成功。此结论限于本次32个同分布未见source。

T1在原训练集raw1000的四个G组均优于旧模型末步；它支持本次直接结构的拟合能力，但本对照同时改变xi分支和LR，不能把收益单独归因于某一项。

T2与旧模型的逐G胜负以配对表为准，不把局部改善称为稳定优势；两者都含LR，不能据此断言物理约束整体无用。I1没有通过开发门槛，按有限候选规则终止。

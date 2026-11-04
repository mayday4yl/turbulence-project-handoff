# V4 RGB Noise Bridge：三组首段训练与配对评估

三组均已完成各 32,000 次曝光、1,000 次参数更新；计划内评估全部完成。总曝光 96,000，正式阶段没有失败、重启、追加训练或协议变更。GPU5/6 上的本任务进程已经退出。

本轮没有得到验证集恢复质量优于输入的证据。R08 相对 R0 有小幅优势，但相对独立直接恢复 D0 基本持平；不能据此主张物理递归或噪声混合带来整体泛化收益。

## 1. 实现、协议与范围

本轮在独立训练目录中新增训练、DDP/梯度累积、EMA、完整 checkpoint 恢复、固定评估和结果汇总入口。历史 v3（V0.31/V0.3.1）及原 V4 correctness 实现、数据、资产和 checkpoint 保留。冻结后未修改正式源码；本地新增审计脚本只读取已生成产物。

共同执行配置：物理 A100 GPU5/6；FP32，autocast/TF32 关闭；全局 batch=32，每卡 microbatch=4，累积4次；网络评估 batch=2，固定 chunk/GPU 分组；正式精修始终 per_sample_serial，实际 refine batch=1。三组从同一初始化独立训练，使用完全相同的20轮 start 顺序。

R08 使用 T0=0.8 完整物理递归；R0 使用 T0=0 完整物理递归；D0 使用 Concat(Y,Y)、[G,0] 单次预测，并保留原始 G 的物理辅助损失。D0 是本训练计划授权的控制组。训练不精修；推理仅在非末级精修。未继承历史 Track A/B、FLUX、latent 或 ControlNet 方案。

协议 SHA256：`69ef8e95c2b3e6012a131eacb371aaf2a2707556ff3556c3cbf18fc7a0756b93`。

[冻结协议](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/protocol/FROZEN.json>) · [授权](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/protocol/AUTHORIZATION.json>) · [训练计划](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/docs/TRAINING_PLAN.md>)

## 2. 同预算主比较：last_32k / raw / refinement-off

训练集为100 sources ×4 realizations ×4 G =1,600 starts；独立验证为100 sources ×2 realizations ×4 G =800 starts。指标来自浮点输出：逐图 PSNR，再按 source/G 平均 realization，最后 source 与四档 G 等权。没有使用显示 PNG 计算分数。

| 组别 | 训练 PSNR | 相对训练输入 | 验证 PSNR | 相对验证输入 |
|---|---:|---:|---:|---:|
| D0 | 23.4650 | +2.8129 | 19.1983 | -1.3412 |
| R0 | 23.6457 | +2.9937 | 18.8805 | -1.6589 |
| R08 | 23.0287 | +2.3767 | 19.1895 | -1.3500 |

单位均为 dB。训练输入基准20.6520，验证输入基准20.5394。训练集输出有改善，但不代表充分拟合或收敛；三组验证输出仍均低于输入。

| 同 source 配对比较 | 平均差值/dB | 获得改善的 source 比例 |
|---|---:|---:|
| R0-D0 | -0.3177 | 14% |
| R08-R0 | +0.3090 | 89% |
| R08-D0 | -0.0088 | 46% |

R0−D0 不支持本轮递归方案的净收益。R08−R0 的正差值表示 mixing 与相应 t 条件的共同改变在本次配置下优于无噪声递归，不能单独归因于高斯项。R08−D0 约−0.009 dB，未超过直接恢复控制组。这里只有一个训练 seed，不作统计显著性或多 seed 稳定性主张。

## 3. 精修、逐级变化与辅助结果

| 组别 | val off PSNR | val on PSNR | on−off/dB | 改善 source 比例 | 完整 val800 精修时间 |
|---|---:|---:|---:|---:|---:|
| R0 | 18.8805 | 19.0922 | +0.2117 | 100% | 22.39 分钟 |
| R08 | 19.1895 | 19.2707 | +0.0812 | 98% | 22.42 分钟 |

两组精修均带来小幅改善，但精修后仍低于验证输入。两组使用各自固定 checkpoint、相同 start/eta 分组与相同精修设置。每组800 starts，共1,200次非末级逐样本精修、48,000次 xi 更新；这些不是训练参数更新。D0 没有额外精修。

| raw/ref-off 验证 first→final 增益/dB | G1 | G2 | G3 | G4 | 总体 |
|---|---:|---:|---:|---:|---:|
| D0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| R0 | +0.0000 | -0.4126 | -0.2921 | -0.1016 | -0.2016 |
| R08 | +0.0000 | -0.1816 | +0.0121 | +0.1655 | -0.0010 |

| 辅助结果 | last_32k EMA 验证 PSNR | 预定节点中 best_val/raw 验证 PSNR | best_val 曝光 |
|---|---:|---:|---:|
| D0 | 19.9547 | 20.2109 | 16000 |
| R0 | 19.7192 | 20.1265 | 16000 |
| R08 | 19.9825 | 20.2417 | 16000 |

三组 best_val 均位于 E=16,000，仍低于输入；继续到固定预算后训练质量上升而验证下降，表现与过拟合一致。主比较仍使用共同 E=32,000 的 raw 权重，没有临时挑 EMA、较早 checkpoint 或 refinement 状态。

![固定 train64 与 val64 的 raw/EMA 曲线](/Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/results/remote_run/learning_curves.png)

## 4. 正确性与证据边界

- 实际 A100 双卡、真实网络与 simulator 的 forward/backward 已验收；R0/R08 仅末级 clean 损失的跨级梯度检查和中间 detach 负对照通过。所选 B32/micro4 的梯度相对参考最大相对 L2 约1.33e−6，低于登记阈值1e−4。
- mu/s 的人工统计公式、真实 train artifact 身份、混合公式和 checkpoint buffer 往返由原 correctness 验收与训练准备验收覆盖；三组持续绑定同一冻结 artifact。
- 模型/EMA/optimizer/RNG/采样位置恢复和下一次更新连续性在独立临时模型上通过。正式三组未发生恢复或重启；不能将临时恢复测试描述为正式 run 实际重启过。
- 最终离线审计通过164份实际评估产物：checkpoint、数据、eta、完整名单、分组与GPU分片一致；从逐图记录独立重算 source 级指标与配对差。每条训练 start 恰好出现20次，每组四档各8,000曝光。
- 两组完整精修分别核验49,200个有限候选，逐样本 best 总J、两块合计9,961,472元素的正则维数、1,200个不同的全新 optimizer、40次更新、固定对象哈希不变、末级不精修及G1 on/off不变。所有最优候选恰好落在第40次更新，不据此增加更新预算。
- 正式完整验证日志保留标量候选与固定对象哈希；逐次 Adam 算术及最佳两块 xi 配对的 tensor hash 检查由冻结前的实际验收覆盖，没有在全部正式样本上重复保存这些大张量证据。CUDA simulator backward 的逐位确定性与批量 refine 等价性不在通过声明内。

[最终完成审计](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/reports/final_completion_audit.json>) · [双卡训练验收](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/reports/training_validation.json>) · [所选 batch 验收](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/reports/selected_validation.json>) · [R08 精修审计](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/reports/R08_refinement_audit.json>) · [R0 精修审计](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/reports/R0_refinement_audit.json>)

## 5. 成本与交付

| 组别 | 成功训练更新累计/分钟 | 已完成 off 评估累计/分钟 | 训练峰值分配显存/GiB | 训练 sample-network predictions | 训练 sample-simulator renders |
|---|---:|---:|---:|---:|---:|
| D0 | 31.23 | 2.63 | 10.51 | 32,000 | 32,000 |
| R0 | 86.03 | 11.11 | 39.86 | 80,000 | 128,000 |
| R08 | 86.32 | 11.15 | 39.86 | 80,000 | 128,000 |

以上网络/模拟器计数是逐样本等价调用量，不是 Python 批处理函数调用次数。成功更新耗时不包含 checkpoint、评估及I/O；off评估耗时为每份结果的两rank较慢耗时。固定分组使精修rank0承担G1/2、rank1承担G3/4，存在负载不均；本轮未因速度改变分组。递归组与D0不是等算力对照。

完整val800/ref-off：D0约7.94秒，R0约36.71秒，R08约36.85秒（两卡端到端评估吞吐时间，不能冒充单张串行推理延迟）。每个固定双样本 batch 的实际耗时保存在各 results.json.execution[].seconds。

本地已收集1,668个文件：全部正式日志/评估、逐start数据、source/G表、节点身份与固定可视化。33份完整 checkpoint（E0至E32000，各组11份）继续保存在A100原目录；没有复制为临时训练权重或改写历史实验。

远程结果根目录：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/runs/v4_overfit100_paired_20260929`。

[远程 checkpoint 清单与源码哈希核验](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/reports/remote_completion_inventory.json>) · [全部配对结果](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/results/remote_run/PAIRED_RESULTS.json>) · [source/G CSV](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/results/remote_run/last32k_source_G.csv>) · [完整证据包](</Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929/results/completed_evidence.tar.gz>)

可复现本地审计（不会运行训练或新增推理）：

```bash
cd "/Users/lucas/Desktop/turbulence IR/v4_noise_bridge_training_20260929"
python3 reports/audit_d0_completion.py
python3 reports/audit_refinement_results.py R08 R0
python3 reports/audit_final_results.py
```

首段意外中断时的原恢复入口为远端独立实现目录下 `bash scripts/run_formal.sh`；本次已经完整完成，勿将它当作新增训练命令。所有组保存完整 last_32k 状态。继续到更大曝光终点需用户另行指定共同预算及新阶段配置；当前冻结入口上限仍是32,000，未启用追加训练。

本轮无剩余运行 blocker。未验证的范围包括多训练seed、其他eta映射、收敛、test200、train500、与历史v3的严格配对，以及T0/lambda质量搜索；这些均未自动执行。

执行收尾：自动跟进 `v4` 已通过应用工具暂停（PAUSED）；GPU5/6 上本任务进程已退出，未启动追加训练或评估。独立 batch=1 串行推理延迟未另外测量，交付冻结 batch=2 的实际执行耗时。

# V6 D0/A/B 阶段交付报告：D0 完成，A/B 方法缺项待定

证据时点：2026-10-27 09:01:47 CST。本报告不是整个 V6 完成声明；D0 已完成 **10,000/10,000** 次正式更新，A **0/6,000**，B **0/4,000**。G0/G1、D0 范围的 G2/G3、G4 已真实通过；A/B 完整图、Benchmark、Pilot、训练以及 G7 均未执行。

## 结论与继续执行边界

D0 在固定 Val1000 上最佳 EMA 为 **U10000，PSNR 21.03393849 dB**。相对同协议官方 EMA U0 增加 2.44169825 dB，相对输入增加 2.23800097 dB；LPIPS 相对 U0 增加 0.01610304，即变差。不能概括为所有恢复指标都改善。

旧 V5 Concertormer D0 best U8250 的 PSNR 为 21.44350572 dB，当前相差 -0.40956723 dB。这个负结果已保留，**不构成暂停 A/B 的理由**；没有因此改 Loss、LR、时间/噪声、Batch 或提前减少预算。

真正受阻的是 Method Spec §4.2 的 ξ Head 物理时间调制定义：只写独立 MLP 64→128→128 和调制各级特征，未明确输入编码/激活、128维投影、加法或 scale/shift、注入点和初始化。此前 Q1 异步问题尚无明确回复；工程自主权不能替代这项方法决定。D0 best 和完整训练状态已经保存，后续从这里继续，不能重跑已完成的正确阶段。

## 方法、实现与资产身份

- 官方 DiffUIR commit `c71437ce86ad1b098aa9be4678dc82e8b99649df`，Large 64-(1,2,4,8)，36,261,187 参数、276 参数张量；严格载入官方五任务（含 GoPro）checkpoint 的 EMA，不能称 GoPro-only 权重。权重 SHA `b9d07ab6b32d37d895be65c50e99bbe6b2ae6e32823e7304ec2167bc5f18843f`。
- 像素空间，无 VQ。原生随机扩散时刻 τ 共1000个；每次 Clean 恢复采样实际3 NFE，不是总共只有3个训练时刻。原生 Condition、Forward/Reverse、SDT/clamp和采样噪声公式保持。
- D0 `method_impl/d0.py`：`L_res=0.5×official residual L1`，加真实可微三步采样 `L_C`，总损失 `L_res+L_C`。训练4次 denoiser，推理3次。这里是 V6 联合目标，不能与旧 Stage1 原生-only Pilot 混称。
- `training_io.py` 负责 Train10000/40000 完整池、无重复padding的类别均衡遍历、全状态原子保存；`run_d0.py` 负责真实六卡 HCCL、累积、Adam、立即 EMA 和日志；`evaluation.py` 固定 Val1000身份/噪声与选模。
- 执行中的6个 V6 依赖文件和所有只读上游文件 SHA 在结束后重新核对相同。配置/方法/训练授权/data hash 及环境身份在 `distributed/D0_formal10000_01/IDENTITY.json`；完整代码映射和未实现接口见 `reports/IMPLEMENTATION_MAPPING_ZH.md` 及兄弟目录 `v6_diffuir_l_method_review_20261027/`。本轮没有覆盖旧 V5、Stage1、ResShift 或旧 Benchmark。

## 数据完成与真实覆盖

从经核验的官方 Places365 train256 归档新增8000个 Clean，与原2000个组成 Train10000；50类各200个 source，每个4条 realization，总计40000对。归档实际1,803,460图像成员、26,103,685,120 bytes，合并 SHA `0ade5d7d38c682aa0203014a2b99ad3bf608bf1de0f151dc0d80efba54660a0a`。这是下载内容身份，不冒称官方发布checksum。

完整40000对逐文件SHA/FP32/shape/finite/来源对应验收完成；原2500源前处理逐像素重放、新旧各8条独立跨进程 ξ/G4物理重放误差0。后者是抽样物理重放，不是将全部40K再渲染一遍。新增源经精确hash、decoded/canonical hash与联合pHash/dHash检查；原Val1000/Test1000身份未改。

数据 fingerprint SHA `0eb85bfd1d783b25ec3145d65de02da71176bce7c880dde7e183c8aa140b53d0`。旧训练源历史来自官方validation pool，新增来自train pool，潜在域差异已披露：原/新G4输入平均PSNR18.71382/18.63926dB；类别一致不等于图像分布完全相同。

正式运行逐卡60000条更新日志联合审计：600000个global position各出现一次；40000条轨迹每条恰好15次，10000个Clean每个恰好60次；每类12000次呈现，原源120000/新增480000，每个realization index150000次。这里600000是重复呈现量，不是60万独立图片。每次使用真实训练文件；没有 Test 恢复评价或用于选模。

## 工程检查、Benchmark 与 Pilot

| 项目 | 实测结果 | 边界 |
|---|---|---|
| D0 双分支完整图 | 原版/可微同噪声采样差0；真实Adam/EMA、双分支与联合梯度验收PASS | 不替代A/B图 |
| 单卡完整图 | microbatch1/2/4/5 PASS；5另做3warmup+100timed | 6/8按显存余量SAFETY_SKIP，非OOM、非已证明硬上限 |
| 六卡候选 | B5×6×acc2约34.1527张/秒；B4×6×acc2约30.675 | 原池候选，临时权重丢弃 |
| 完整新池最终配置 | 5×6×2=60，3warmup+100timed，34.2225张/秒 | 冻结为D0配置，不能套给A/B |
| HCCL/恢复 | 实际六卡与串行梯度一致；新进程下一更新网络最大差1.49e-8，RNG/索引精确 | FP32数值等价，非NPU训练逐位复现 |
| 独立Pilot | 300更新、18000呈现、10000独立Clean，实际参数与有限梯度PASS | 正式从官方EMA重新开始，不继承Pilot |
| Pilot分支梯度 | raw U0/U300固定18个训练source；两分支有限非零、和等于联合梯度 | 18个源诊断，不外推全部Train |

正式冻结配置：FP32、HF32 OFF、256×256、物理 NPU2–7、microbatch5、accumulation2、effective batch60；无AMP/gradient checkpointing。Adam β=(0.9,0.99)、wd0、global clip1；LR峰值2e-5、前200更新warmup、cosine至2e-6；EMA β=.995每次立即更新。没有根据验证结果动态修改这些定义。

## 10,000 更新的实际运行结果

| 统计 | 实测 |
|---|---:|
| 完整 launcher 墙钟（含启动、验证、保存、退出） | 5.0831 小时 |
| 日志累计 update interval | 4.8465 小时 |
| update interval吞吐（含数据I/O、Adam/EMA/DDP；不含独立验证/保存） | 34.3894 张/秒 |
| 全墙钟摊销吞吐 | 32.7885 张/秒 |
| 启动/验证/保存/退出等合计额外时间 | 851.84 秒 |
| 峰值 allocated / reserved | 39.0875 / 41.7012 GiB |
| 外部 npu-smi 定时采样峰值HBM | 45.7861 GiB |
| 逐update最小空闲显存 | 18.2134 GiB |

显存数值分别是allocator峰值与设备采样，npu-smi不是连续硬件trace。每卡10000条Loss/梯度/更新记录均有限、联合梯度非零；每卡201次四denoiser输出梯度记录全部有限非零。结束时六卡参数逐位相同。实际CPU读取U0与U10000：276/276参数张量改变，最大绝对变化0.061188117，L2变化17.193457509；raw/EMA/Adam均有限，全部Adam step=10000、EMA更新=10000、6份rank RNG、游标600000、扩散buffer逐位不变。独立审计没有额外optimizer更新。

所有本次六卡NPU worker已按PID/start_ticks核实退出；未干预NPU0/1或其他用户进程。资源快照留在 `evidence/D0_COMPLETION_AUDIT.json`。

## 固定 Val1000 完整结果

1000条轨迹、250个独立Clean、50类；所有11个checkpoint使用相同source/trajectory ID、固定初始噪声和输入。原始输出计算指标，不以PNG显示clamp替代；完整3 NFE。完整Val每1000更新，Val100观察每250更新且不参与best选择。

| update | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | MAE ↓ |
|---:|---:|---:|---:|---:|
| 0 | 18.59224024 | 0.45175667 | 0.51510603 | 0.08007468 |
| 1000 | 19.76536213 | 0.50200805 | 0.58150140 | 0.06627411 |
| 2000 | 20.58730673 | 0.52888749 | 0.55929537 | 0.05990245 |
| 3000 | 20.76934009 | 0.53586950 | 0.54963865 | 0.05845923 |
| 4000 | 20.86761983 | 0.53983708 | 0.54378468 | 0.05768648 |
| 5000 | 20.92778477 | 0.54251455 | 0.53952252 | 0.05720605 |
| 6000 | 20.96788388 | 0.54436545 | 0.53647460 | 0.05688681 |
| 7000 | 20.99791262 | 0.54568497 | 0.53430609 | 0.05665939 |
| 8000 | 21.01602876 | 0.54654879 | 0.53279537 | 0.05651343 |
| 9000 | 21.02343436 | 0.54702201 | 0.53188937 | 0.05644493 |
| 10000 | 21.03393849 | 0.54745733 | 0.53120907 | 0.05637289 |

输入PSNR=18.79593752dB。Best选择按冻结协议：PSNR差≥0.02dB按PSNR；差<0.02dB按更低LPIPS，再相同保留较早checkpoint。独立重放11次选择结果与BEST指针一致。最终best U10000，原因“PSNR within 0.02 dB tie; lower LPIPS”。

逐图配对表 `reports/D0_FINAL_PAIRED_VAL1000.csv`：相对输入正增益 983/1000，相对U0 974/1000，相对旧V5 D0 229/1000。这1000条不是1000个独立场景；每source有4条realization。

旧V5 best同固定Val1000：PSNR 21.44350572、SSIM 0.57407242、LPIPS 0.45879667、MAE 0.05275667。逐图source ID完全对应，输入PSNR浮点差小于2e-5dB。旧模型Train2000、effective batch48、BF16网络、Clean-only MAE、单次网络前向，V6 Train10000、batch60、FP32、Residual+三步Clean，初始化亦不同。因此这是历史基线对照，不能把差异只归因于backbone或训练速度。当前不能从这些结果确定欠拟合、域差异、目标冲突或模型容量谁是主因。

## Checkpoint、复现与继续入口

- 远端最佳：`/data2/yanglin/v6_diffuir_l_20261027/distributed/D0_formal10000_01/checkpoints/U010000.pt`；SHA `505d1f865e46c429fccc5a7701c6216cc7f088176d1e78549960f6cc3f653e39`。最后checkpoint：`/data2/yanglin/v6_diffuir_l_20261027/distributed/D0_formal10000_01/checkpoints/U010000.pt`，SHA `505d1f865e46c429fccc5a7701c6216cc7f088176d1e78549960f6cc3f653e39`。大文件仍在de36，未伪称已下载本地。
- `gates/A_INITIALIZATION_READY.json` 只表示D0 best可用；状态是 `D0_BEST_AVAILABLE_A_METHOD_DETAIL_PENDING`，不是A已开始。
- 完整训练控制配置在 `distributed/D0_formal10000_01/CONFIG.json`、启动命令在 `LAUNCH.json`，每卡日志、HBM监控、PER_IMAGE、固定100图原始FP32 NPY/PNG均保留远端。本地同步状态/日志/指标，图像与大checkpoint不作重复下载。
- 独立结束审计：`PYTHONPATH=/data2/yanglin/v6_diffuir_l_20261027 python3 scripts/audit_d0_completion.py`；CPU checkpoint审计：`V6_DEVICES=2 bash scripts/run_env.sh scripts/audit_final_checkpoint.py`（只在CPU读取，不创建NPU模型）。报告复现：本地 `python3 scripts/report_d0_completion_local.py`。
- 方法Spec SHA `4c0672c761fe93d36e4a7f17953a5e2d61689d99199a28b2e68d607a38d0cbda`；配置/代码/data/checkpoint完整版本身份见机器结果和IDENTITY。修复历史与失败现场见 `reports/ENGINEERING_FIXES_ZH.md`、`evidence/code_history/`，没有删除失败来制造PASS。

## 尚未执行与所需研究决定

A/B网络实例化、真实联合Loss/Simulator反传、A随机g、B四阶段独立噪声与四节点等概率TF、B终点隔离Full-BPTT、各自Batch/Pilot和6000/4000正式预算均未执行。旧V5或D0梯度不能替代这些验收。A best后续仍必须按固定Val1000、固定4→3→2→1→0四步最终PSNR。

需要明确的Q1建议实现（**待决定，未实施**）：ξ Head采用SinEmbed64(1000t)→Linear64→128→SiLU→Linear128→128；每个resblock将128维投影到2C，第一GN后、SiLU前进行 `h=(1+scale(t))*GN(h)+shift(t)`，末端scale/shift投影全零初始化。若选择别的定义也应补入独立方法补丁，明确位置/初始化，再完成等价与真实梯度验收。不能用未回复当批准。

得到该定义后，继续实现/验证A并从上述D0 best初始化，A工程Pilot通过再执行6000，随后B完整图/Pilot通过执行4000；不会因为本报告的恢复负结果要求额外质量批准。当前交付诚实保留 **D0完成、全V6尚未完成** 的状态。

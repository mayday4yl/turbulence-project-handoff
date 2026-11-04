# 早期湍流项目交接：v1—v3 与几何校正支线

本文件按本地原始报告、冻结配置和验收记录整理，面向不掌握聊天记录的接手者。整理日期为 2026-11-04；历史完成、暂停与设备释放均指记录当时，不代表今天的服务器实况。当前 UniDB 午夜实验的最终状态由总交接另行收尾，本文件不授权恢复任何旧实验。

**先读结论：这一时期形成了可复用的模拟器、数据与递归训练工程资产，但没有建立未见图像上的稳定渐进恢复优势。** v2 正式训练未启动；v2.5 只留下早期 U204 恢复点；v3 在训练图像上有不精修递归收益，后续 heldout 检查揭示泛化回退。几何校正支线是另一个实验，完成了预定计算且 Gate1 失败，不能并入老师的 v2。

本文中的来源代号对应附录的绝对路径与完整 SHA256；机器可读事实、口径、远端资产引用和缺口见同目录 `legacy_evidence.json`。本次仅计算这些本地证据文件的 SHA，不加载 checkpoint 张量，不核验远端大数组现存状态，不重新计算模型结果。

## 1. 版本、数据、K：接手前必须区分

| 展示名 | 原文件标识 | 正确定位 |
|---|---|---|
| v1 | V0.1 / v01 | 原始物理桥接、双头恢复网络和推理基础 |
| v1.5 | V0.2 / v02 | 用户修改的物理约束与推理；另有 2026-08-29 新协议，不应与旧停版混写 |
| v2 | V0.3 / v03 | 2026-09-15 老师的预测状态递归训练，全数据分支在正式开训前终止 |
| v2.5 | V0.3-half / v03-half | source 层减半，1000-source 正式递归训练，只完成 U204 |
| v3 | V0.31 / V0.3.1 / v031 | 固定四档完整尾路径的 Overfit100；包括原递归和后续 T1/T2/I1 |
| 几何校正支线 | `analysis/v03_validation_20260909` 的历史 v0.3 | 四臂几何动作/训练分布验证，与 v2/v2.5/v3 身份独立 |

`v03_overfit100_g4321_n4/` 虽含 v03，实际是 **v3 的数据构建项目**。其 Places365 basename 有 `val` 也不等于本实验验证集：source 是从旧 **train pool** 选出的，v3 全 1600 起点属于训练人口。不能凭文件名、目录名或 README 的旧阶段描述判定最终状态。[INDEX, V3_DATA_DELIVERY, V3_FINAL_STATUS]

本文件的 LC 为 clean loss，LR（或 L_R）为重退化 loss；学习率另写 lr，不能把两者混读。

v1/v2/v2.5 的 K 和 v3 的 K 语义不同：v3 K 是网络预测总次数，包含最后一次 clean 预测，G4 尾路径 `[4,3,2,1]` 为 K4，one-shot 为 K1；v2.5 的 K0 是原推理 one-shot。跨版本表格须同时写 K 定义、G、权重类型和数据人口。[V25_PROTOCOL, V3_PROTOCOL]

## 2. v1：原始物理桥接基础

**日期与目的：** 2026-08-26授权，主seed20260826、run `bridge_v01_seed20260826`。精确开训/停训日历时间本轮未核，不能把seed直接当时间戳。研究固定realization的PiRN/P2S六档severity trajectory，渐进物理bridge＋推理时ξ精修能否优于同网络one-shot。[V1_FINAL_REPORT, V1_TRAIN_CONFIG]

**方法：** known-g单帧RGB，severity embedding/FiLM、共享编码器与clean/realization双解码器，clean sigmoid、ξ线性输出。训练clean MSE和**真实clean＋预测ξ**渲染terminal observation的MSE；degradation权重在U3000后渐升至.1。推理时才可优化ξ并渲染较弱状态。它不是扩散噪声matching；也不等于v2/v3使用预测clean＋预测ξ、全路径full-BPTT的目标。[V1_REMOTE_TRAIN, V1_TRAIN_CONFIG, V1_INFERENCE_CONFIG]

**数据：** Places365-Standard，256×256，ID `places_p2s_bridge_v01`；train2000sources×4=8000trajectories，val200×2=400，test500×2=1000；source跨split overlap=0。每trajectory保存固定clean、完整native FP32ξ和G=.75/1.5/2.25/3/3.75/4.5六级状态。terminal评估只分后五G档，不能把六级状态数当1000条最终评估人口。[V1_DATA_CONFIG, V1_GENERATION_SUMMARY]

**完成边界：** 配置最大30k，训练摘要stopped_early=true，实际完成20k updates/effectiveB16。归档审计连续20k训练行、4000 paired rows和有限数值通过；不是仅根据最大预算推断完成。val先选refinement LR=.03，再在K0/1/3/5选择K0，之后冻结checkpoint/K/LR执行Test1000。历史远端checkpoint `/data2/yanglin/physical_bridge/runs/bridge_v01_seed20260826/checkpoints/frozen_checkpoint.pt`，SHA `34dc4dce03bd13ba4079100d4b195fcc0e8c3b07699e06ae12ca91e0aba48f47`；评价使用EMA。[V1_FINAL_REPORT, V1_TRAIN_SUMMARY, V1_EVIDENCE_AUDIT]

**主要结果口径：** 冻结Test1000、五terminal-G bucket各200条、每trajectory最终输出一次；one-shot均值PSNR22.24463244dB、SSIM.62972416、LPIPS.41176186、L1.05016122。G4.5的200条为PSNR20.62721163/SSIM.55501783/LPIPS.47951543/L1.05971911。所有bridge/refinement/oracleξ路线相对one-shot配对ΔPSNR=0、CI[0,0]，因为最终冻结K0，路线实际全部退化为同一one-shot。**这只说明val没有支持渐进，不能据此判oracleξ没有价值。** val K5比K0低.914829dB。[V1_FINAL_SUMMARY, V1_TEST_ROLLOUTS, V1_FINAL_REPORT]

另有独立 `teacher_forced_summary.json`，覆盖1000trajectories×六severity=6000状态，平均PSNR23.087334；它不是上面的terminal Test1000，禁止替换主结果或与K0冻结表混平均。[V1_TEACHER_FORCED]

**入口与缺口：** 本地以 `reports/bridge_v01_retrospective/sources/original_final_report.md`、结构化summary及回顾证据审计为入口；执行实现保留于 `physical_bridge/`、`configs/`、`scripts/`，保存真实运行源码在回顾sources/remote_code。历史服务器liteserver-40b3，源 `/data0/yanglin/physical_bridge`，run/data在 `/data2/yanglin/physical_bridge/`。本次未查远端tar/权重现存状态；渐进失败根因未由这张冻结K0 test表因果隔离。checkpoint选择允许PSNR .05dB内按LPIPS再residual排序，不能只看最高PSNR断言选错模型。[V1_EVIDENCE_AUDIT]

## 3. v1.5：物理约束修改与新协议 A1

**日期与目的：** 2026-08-29新协议冻结，A1正式run截止08-30，追加可微性审计09-01。目标为改善ξ跨severity传输，以及“重建原观测”不足以约束下一个bridge状态的问题。`v02_src/`是更早被替代/停止方案；新协议明确禁止导入或恢复它，其预算/loss不能并入本次。[V15_METHOD, V15_PLAN, V15_AMENDMENT, V15_LEGACY_AUTHORITY]

**方法与初始化：** 保留known-g、双输出、完整nativeξ与同一simulator。clean MSE＋trajectory consistency LT（真实clean＋预测ξ、unique{G,g,h}）＋bridge execution LB（预测clean＋预测ξ→相邻弱态）；A0固定128batch选λT=.8/λB=.1。A1实际从v1 U20000 checkpoint的 **`model` RAW** 初始化，新建AdamW和EMA；评价为A1 EMA，不能写成从v1 EMA或旧v02_src续训。真实加载证据在保存trainer第121–122行。[V15_METHOD, V15_CONFIG, V15_TRAIN_SOURCE]

**优化与数据：** A1 3000updates/B16、48000 sample calls；AdamW lr2e-5、200warmup、betas(.9,.99)、WD1e-4、clip1、EMA.999，BF16网络/FP32 simulator与loss。复用v1数据但A1只训练2000trajectories、五terminal G各400，不能称全部8000train trajectories。fast-val100、每G20。[V15_CONFIG, V15_TRAIN_SUMMARY, V15_OBJECTIVE_REPORT]

**实际边界：** A0_PASS、TRAINING_COMPLETE、A1_STOP、SCIENTIFIC_STOP_A1；连续1—3000、数值健康，无nonfinite/clipped step。StageB未启动，full-val400/test/multiseed确认未完成。历史checkpoint `/data2/yanglin/physical_bridge/runs/v02_new_protocol_20260829/a1/checkpoints/update-003000.pt`，SHA `a48a7557a577d0affd745f1e06f32902801b96c2b97d1df53f68d59176ff033a`。[V15_OBJECTIVE_REPORT, V15_TRAIN_SUMMARY]

**首先纠正评价身份：** 正式evaluator使用seed2026082801，协议为2026082897，两个val子集仅19/100重合；同时对v1 comparator加载RAW，冻结推理应为EMA。正式evaluation execution因此被标invalidated，**不等于3000步训练权重损坏**，也不能直接拿旧正式表当论文主结果。[V15_OBJECTIVE_REPORT, V15_EVAL_SOURCE]

同一A1 U3000 EMA按协议seed＋v1 EMA的只读纠偏结果：

| Gate指标与人口 | 实际值 | 95% bootstrap区间 | 冻结门槛 | 判断 |
|---|---:|---:|---:|---|
| overall K0 ΔPSNR，100条 | −.14817378dB | [−.16795112,−.12803172] | ≥−.10dB | FAIL |
| G4.5 K1 ΔPSNR对v1 EMA，20条 | −.00192876dB | [−.06494431,+.05431180] | ≥+.10dB | FAIL |
| G4.5 next-state MSE下降，20条 | 1.34859367% | [.22347505%,2.47322876%] | ≥25% | FAIL |

四种seed×RAW/EMA comparator组合都STOP，不能只选其中K1正的组合救结论。K0总体100条和K1只G4.5的20条不可直接相减。同口径G4.5 A1 K0=20.57190967、K1=20.25016356、配对K1−K0=−.32174611dB；真实弱态teacher-forced K1=21.27747574dB，仅作机制诊断。[V15_GATE_SENSITIVITY, V15_CORRECTED_G, V15_OBJECTIVE_REPORT]

**诊断与未决：** controlled-render支持预测状态rollout、特别是predicted clean作为底图是主要问题位置，但未定位唯一根因。图连接、CPU有限差分和真实A0/A1 NPU各分支有限非零梯度支持simulator可微，不能沿用“不可微造成失败”的解释；NPU直接有限差分尚未完成，FFT resize warning算子定位未闭合。高G曝光、多目标方向和renderer放大等仍为假设；10条高G的ξ有效RMS不是根因定论。[V15_OBJECTIVE_REPORT, V15_DIFFERENTIABILITY]

**入口与缺口：** `v02_new_protocol_20260829/README.md`、METHOD/TRAINING_PLAN/config描述冻结方法；`analysis/v02_a1_20260830/source/`保存实际执行源码，`report/V02_A1_OBJECTIVE_HANDOFF_FOR_GPT.md`与derived CSV保存可复核纠偏。历史服务器liteserver-40b3，源 `/data0/yanglin/physical_bridge/v02_new_protocol_20260829`，run如上。该报告自称事后/inference-only诊断，不是正式replacement evaluation；本次未执行full-val/test或验证远端checkpoint现存性。

## 4. v2：全数据预测状态递归分支，正式训练未开始

**日期与目的：** 2026-09-15 创建独立项目，检验训练时就使用模型自身预测状态、保持完整反向传播，能否解决多步推理的状态错配。唯一基础实现来自 SHA 固定的 v1 `/data0/yanglin/physical_bridge`，不是旧 v1.5 或几何支线代码。[V2_README]

**方法：** 原 DualHeadUNet 输出 clean 与完整 native ξ；由预测 clean/ξ 渲染下一较弱强度状态，不 detach、不 teacher forcing、不在训练中精修。每一步使用全图 clean MSE 与对原始 Y/G 的 observation reconstruction MSE；路径内平均后对实际 global paths 等权平均。六档 G 独立均匀采样，requested K∈{0,1,3,5} 按原 schedule 构造并先去重。推理保持 v1 完整 schedule、prox/prior、候选规则、refinement LR0.03 和 16 像素精修边界；不能把这套推理与 v3 新精修混为同一个算法。[V2_README]

**实际完成边界：** 四卡 longest-path microbatch1/global4 工程验收完成 3 updates/12 paths，约 10.1 秒/热更新；另有恢复、混合路径和梯度验收队列。**这些是 disposable acceptance，不是正式训练或性能结果。** `DATASET_PROTOCOL_CHANGE.json` 明确在 2026-09-15 15:42:20 UTC 因 source-level half-data 协议变更终止，`old_formal_training_started=false`，旧权重不得初始化后继 v2.5。没有冻结正式完整训练预算或最终质量结果。[V2_EXECUTION, V2_TERMINATION]

**数据与入口：** 计划使用旧六档完整数据，后被 source 层减半协议替代。有效本地入口是 `v03_recursive_from_scratch_20260915/README.md`、`docs/EXECUTION_STATUS.md` 和终止记录；远端历史根为 `/data2/yanglin/v03_recursive_from_scratch_20260915`。工程 acceptance checkpoint 不是应部署或继续训练的正式模型。

**未决问题：** 这一 full-data 方案本身没有正式质量结果，不能比较“full 比 half 更好”；后继结果也不能倒填为 full-data 成绩。今天远端 acceptance 资产是否仍在，本次未核。

## 5. v2.5：source-half 正式训练，U204 安全暂停

**日期与目的：** 2026-09-15—16 执行 v2 的 source 层 50% 数据协议。2026-09-16 10:24 正式开始、11:50:24 安全暂停。其目标仍是完整 predicted-state rollout，不是只做 GT 状态的一步训练。[V25_HANDOFF]

**数据：** 确定性分层选择 train1000 sources×4=4000 trajectories/24000 六档 states；val100×2=200 trajectories/1200 states；test250×2=500 trajectories/3000 states。保留每个 source 的完整 realization 数，绝非“2000 sources 仅少一半轨迹”。47000 成员读取，42300 既有 SHA 匹配，source overlap 和 trajectory-key duplicate 均为0；仅新建引用原 tar 的 view，test 只做完整性检查。[V25_DATA]

**冻结方法与优化：** 沿用 v2 的 clean MSE+0.2×原 Y/G reconstruction MSE、完整 BPTT 和原推理；从全新 seed20260915 开始，不继承任何 pilot 模型。四卡/global32 paths、microbatch 上限2、按长度平衡，实际累积4或5 microsteps；AdamW betas(.9,.99)、WD1e-4、clip1、64-update warmup 到1e-4、固定10k horizon cosine 到1e-6；EMA=.999⁴。curriculum 前128短路径、128–511中等、512后长路径；主预算8k、满足条件可至10k，但**实际远未达到**。[V25_PROTOCOL, V25_CONFIG]

**完成边界与可用资产：** 实际 204 optimizer updates、6528 sampled trajectories、10098 network predictions，约1.632 trajectory-equivalent passes；formal training incomplete。历史 CPU 完整状态验收和 U173 真恢复后继续均通过，模型/EMA/AdamW/scheduler/curriculum/sampler/四rank RNG 保留。恢复点是 `/data2/yanglin/v03_half_recursive_20260915/runs/half_formal_01/checkpoints/update-000204.pt`，历史 SHA `8bb92a45c1fca32d4a8692a909dfc54120ad24bf5ee32ef9c28588c949c43b92`。这是**可恢复资产，不是性能推荐模型**。状态恢复一致不保证后续 NPU 执行逐比特重演。[V25_STATUS, V25_RESUME_AUDIT]

**主要质量结果：** 只有 U173 EMA 的固定 **16 val trajectories×六G×四K=384记录**，不是完整 val200，也不是 U204 最终质量。原完整 inference 的汇总为：

| K | 平均 PSNR dB | SSIM | LPIPS | ΔPSNR 对 K0 |
|---:|---:|---:|---:|---:|
| 0 | 17.2925 | 0.5040 | 0.6794 | 0 |
| 1 | 16.1409 | 0.4791 | 0.7559 | −1.1516 |
| 3 | 15.3683 | 0.4622 | 0.7766 | −1.9242 |
| 5 | 15.2162 | 0.4594 | 0.7792 | −2.0763 |

多步弱于 K0；λ pilot 选择0.2也没有产生正 progressive gain。没有完成 full-validation checkpoint 选择，`ready_for_frozen_test=false`，heldout test 未跑。训练+fast val约5.77 NPU-hours；该 half 项目全部受保护验收/pilot/benchmark/正式约18.72 NPU-hours，包含失败工程尝试，非纯训练时间。[V25_HANDOFF, V25_VAL_CSV]

**有效入口与缺口：** 先读 `docs/NOON_HANDOFF.md`，再读冻结 `FORMAL_PROTOCOL.md`、`artifacts/half_formal_01_freeze.json` 与上述完整状态验收。没有证据可把 U204 称为训练完成、收敛、验证最佳或最终泛化成功；本次没有查远端是否发生过未在本地归档的后续执行。

## 6. v3 数据构建：100 sources、固定四档，不是恢复实验

**日期：** 2026-09-16 19:31 北京时间完成数据交付与本地归档。[V3_DATA_DELIVERY]

从旧 train pool2000 sources 以固定 SHA 排序选100；每source一次确定性预处理得到固定 clean，4组独立完整 native ξ；同一 clean/ξ **分别直接渲染** G4/3/2/1。共400 trajectories、1600 states、1600起点、2500权威FP32数组，约17.279GB；不级联退化、不重新采ξ、不存FP16、不额外clamp、不生成G0。训练/验证/测试 source 身份不因 Places365 原文件 basename 而改变。

400组 realization/800块ξ seed重建逐元素一致，1600/1600状态磁盘回放 max_abs/MSE=0；独立CPU读取核验2516 hash files/2500 arrays。数据 hash_manifest历史SHA `b896f9060a8ff012cafe476ec6c0bb1075e6291bfdec1eab7ff791dc8936b03b`。该交付只标 `ready_for_overfit_training`，不包含任何恢复训练/pilot/推理。[V3_DATA_DELIVERY, V3_DATA_ACCEPTANCE]

权威远端数据路径为 `/data2/yanglin/v03_overfit100_g4321_n4/data/places_p2s_bridge_v03_overfit100_g4321_n4`；本地有配置、manifest和报告，大数组按历史交付留在远端，本次未重新访问。

## 7. v3 原递归：U1000 完成，训练集收益与精修负效

**日期与目的：** 2026-09-16—17 进行独立实现、验收和 AUTO_TO_1000 正式训练；09-18追加精修负效审查。目的为以固定四档完整尾路径，先检验该方法能否拟合100个固定 source，并验证真实 full-BPTT。[V3_STAGE1_REPORT, V3_FINAL_STATUS]

**方法：** 原32,101,673参数 DualHeadUNet，输入仅当前图像/当前g，预测 clean/zernike/tilt；下一输入 `F(C_hat,ξ_hat,next_g)`，每步（包含末步）以原始 G/Y 计算 `LC+0.2LR`，先按每条路径预测次数平均，再对32条路径等权。四档完整尾路径，不用旧六档curriculum；无detach、teacher forcing、raw-ξ监督、GT-next loss、训练精修或原Y旁路。BF16梯度parity失败保留，正式使用FP32；局部tilt导数通过同点条件参考及算子VJP，旧FP32有限差分失败未被改写。[V3_STAGE1_METHOD, V3_STAGE1_REPORT]

四卡B32/micro1×accum8、AdamW peak1e-4、50warmup、cosine至U1000的1e-5，EMA.996、clip1。1000 updates=32000 paths/80000 network predictions，均为同100sources/400trajectories/1600starts 的20轮曝光。训练 update 总耗时41928.9秒，计时为最慢rank，不是额外并发总卡时。U100仅按观测残差/耗时校准精修，固定LR.01/40iterations；**没有按恢复质量保证该配置最优**。[V3_PROTOCOL, V3_CONFIG, V3_U1000_REPORT]

**实际完成与完整状态：** 最终 `artifacts/WORKFLOW_STATUS.json` 为 PAUSED_AT_1000，1000训练及RAW/EMA×refined/unrefined各1600起点完成。README仍有早期“formal未启动”段落，应以这份最终状态和报告为准。完整checkpoint `/data2/yanglin/v031_overfit100_g4321_n4/runs/formal/checkpoint_001000.pt`，历史SHA `6eb3de6a064a0215a9a5b02935145528b23209d2f633b6dd50b60a999bad2454`；344 network/EMA张量、AdamW344参数状态、四rank RNG、sampler32000和scheduler U1000→next1001/lr1e-5验收通过。历史恢复即时状态exact通过，但未来数值bitwise重演失败被保留，不应宣称完全可重复执行。[V3_FINAL_STATUS, V3_U1000_STATE]

**主要结果口径：** 全部为原训练人口的1600起点；原始 float RGB、不clamp，先算每图全图mean MSE/data_range1 PSNR再平均。每G400起点，报告所有合法K，不事后挑best K。

| U1000 RAW/G | K1 PSNR | 完整不精修 PSNR | 完整精修 PSNR | 不精修末步对K1 |
|---:|---:|---:|---:|---:|
| 4 | 21.8539 | 22.8074（K4） | 22.2671（K4） | +0.953519 dB |
| 3 | 22.5287 | 23.0932（K3） | 22.6925（K3） | +0.564457 dB |
| 2 | 23.1473 | 23.3548（K2） | 23.1135（K2） | +0.207540 dB |
| 1 | 24.1221 | 同K1 | 同K1 | 无多步比较 |

这些支持训练集上不精修递归的拟合收益，**不证明泛化或相对于独立训练单次基线的优势**。精修较不精修G4/G3/G2约−.540/−.401/−.241dB。[V3_U1000_REPORT]

**精修审查（独立支线）：** 原U1000 RAW固定48起点，另12起点梯度/Adam位移诊断；0恢复网络更新。40次精修使观测J下降80.84%—85.97%，下一次恢复却各G16/16变差（G4−.845、G3−.551、G2−.346dB）；12点∇J/∇下一次恢复误差余弦均负，原完整Adam首步48/48已受损。独立solver/状态/返回值检查没有找到足以解释的实现错误，步幅会放大损伤，但不能唯一归因NPU、只说40步过多或宣称换GT-clean就能修好。研究价值是“观测目标下降并非恢复质量替代”，不是新修复已成功。[V3_REFINEMENT_FORENSIC]

**有效入口与缺口：** 配置 `configs/train_v031.json`；结果 `artifacts/V031_UPDATE1000_REPORT.md`；状态与完整性验收如上；实现位于 `code/v031/`。此次已核本地归档没有原组U1000之后的续训结果，本次不恢复。原始负效审查可复用，不能按不同人口的均值推断v2.5或当前UniDB同一根因。

## 8. v3 后续 T1/T2/I1：同起点曝光，不是等计算量

**日期与目的：** 2026-09-18—19 另建 `v031_followup_20260918/`，与旧U1000权重保持独立。T1检验简单直接恢复；T2检验去掉F作为中间输入生成器；I1在固定旧U1000上做有限精修候选验证。旧v3没有被续训或覆盖。[V3_FOLLOWUP_README, V3_T1_METHOD, V3_T2_METHOD]

**实际配方：** T1从共同未训练初始化提取原clean/shared结构，单次 `C_hat=D(Y,G)`、只clean MSE，不执行ξ分支/模拟器/递归，λR=0；T2从共同完整未训练初始化开始，下一输入直接为 `C_hat`，仍有native ξ和每步 `LC+.2LR`，模拟器只用于原G/Y重退化监督。T2后续g是步骤条件，不能声称反馈clean真的有该物理强度。两者FP32/raw-only/无EMA、四卡B32、原AdamW/LR schedule、各1000updates/32000starts；T1micro8×accum1，T2micro1×accum8。T1训练32000次网络预测，T2为80000次；同1000updates只意味着同曝光，不等FLOPs或卡时。[V3_T1_METHOD, V3_T2_METHOD, V3_T1_CONFIG, V3_T2_CONFIG]

**完成与资产：** 两项各U1000完整训练后暂停，主训练评估无缺项，完整RAW/AdamW/scheduler/四rank RNG/sampler通过历史CPU验收，local package/visual QA完成。T1完整update计时合计344.10秒、训练墙钟367.26秒；T2为38497.21/38526.47秒，两项包含的训练工作不同。checkpoint：

| 任务 | 远端完整checkpoint | 历史SHA256 |
|---|---|---|
| T1 | `/data2/yanglin/v031_followup_20260918/T1_DIRECT/formal/checkpoint_001000.pt` | `46cf6879e587089d645c8de767b128d48d585cb981bb6f0551c28808f9d7d7ff` |
| T2 | `/data2/yanglin/v031_followup_20260918/T2_FEEDBACK_R/formal/checkpoint_001000.pt` | `80cbb6bd56d7c941395a833a56c9389fe1943d712110b495f65c4e6d0d4cae59` |

[V3_FOLLOWUP_COMPLETE, V3_FOLLOWUP_RESUME]

**训练人口结果：** 原1600训练起点、RAW1000最终输出相对旧RAW1000不精修末步，T1 G4/3/2/1为 +.793930/+.944353/+.976816/+.719382dB；T2为−.237647/−.055815/+.046907/−.049376dB。T1是整体简单方案对照，同时移除ξ分支和LR等，不是单loss消融。[V3_FOLLOWUP_TRAINING]

**独立source检查：** 固定32个未见train-pool source×2ξ×4G=256starts；同分布新场景筛查，非外部benchmark、非多个训练seed。每G64记录，RAW1000的绝对平均单图PSNR：

| G | 原输入Y | 旧v3完整末步 | T1单步 | T2完整末步 |
|---:|---:|---:|---:|---:|
| 4 | 18.833153 | 16.682420 | 17.430093 | 16.605387 |
| 3 | 19.497087 | 17.431869 | 17.994651 | 17.398567 |
| 2 | 20.479326 | 18.504187 | 18.767752 | 18.501353 |
| 1 | 22.277710 | 20.784585 | 20.240979 | 20.514950 |

所有模型各G最终均低于输入；T1比旧模型强不等于已恢复得比原图输入好。[V3_FOLLOWUP_README, V3_HELDOUT]

**后补曲线诊断：** 仅追加既有checkpoint推理，未训练。固定train64与heldout256，U500→1000：T1 train21.3987→24.1839，而heldout19.5941→18.6084（−.9858dB）；T2 train20.4495→23.3164，而heldout19.4644→18.2551（−1.2093dB），两者32/32 heldout sources按source聚合均变差。5个采样checkpoint每G最终都低于Input。100source覆盖不足是待检验解释，尚不能排除架构/优化/content因素，未执行source-scaling；该heldout已反复查看，后续调参不能再称独立测试。[V3_HELDOUT_CURVE]

**I1有限候选：** 固定旧v3 RAW1000，dev8sources/24starts×14配置=336完整路径；12个新候选没有通过预声明门槛，最近者仍负，状态 `no_repair_candidate_supported_on_dev`。确认集与候选heldout按规则未开；这是有效停止，不是漏做最终成功验证。另4个冻结dev点梯度方向与下游误差冲突，不可泛化为唯一根因。[V3_I1_REPORT, V3_FOLLOWUP_MISSING]

**接手入口：** `reports/README.md`、`TRAINING_REPORT.md`、`CHECKPOINT_AND_RESUME.md`、`HELDOUT_SUMMARY.json`、`I1_REPORT.md`；后补曲线在 `reports/heldout_curve_20260919_2138/delivery/REPORT.md`。旧交付ZIP只含可同步资料，大checkpoint和FP32数组留在远端，路径/hash清单另存。本次不重新打包全部数组或将历史“完整状态验收”当今天远端现存核验。

## 9. 几何校正支线：完整执行后 Gate1 FAIL，非老师v2

**日期与问题：** 2026-09-08审计后，09-09凌晨完成计算；单帧、原G已知、现有P2S合成。验证标量弱化训练是否无法覆盖实际几何校正T(Y)，及实际轨迹是否优于简单解耦/普通增强。[GEOMETRY_PROTOCOL, GEOMETRY_REPORT]

**方法与数据：** 512训练sources、128验证、128保留测试，source/clean hash互斥并排除旧诊断129sources；每图2新ξ。scalar、tilt/blur解耦decoupled、RMS匹配generic、冻结v1估计actual四臂；相同22,533,571参数clean分支，输入[当前状态,原Y,冻结C0]，G/alpha条件；同初始函数，AdamW/MSE/BF16，128crop训练而完整256图模拟与评估。四臂×2固定随机顺序×5000updates=40000有效更新；两个seed是训练采样/crop/alpha顺序，非独立随机预训练初值。[GEOMETRY_REPORT]

**完成边界：** 8模型全部5000updates、完整验证；FINAL_STATUS为COMPLETE，Gate1 FAIL，Gate2 NOT_RUN_GATE1_FAILED；保留测试只生成缓存/完整性核验，未读恢复分数。累计1.7451596 NPU-card-hours、峰值并发4；工程评估错误修复并做输出等价核验，不当作方法没跑完。[GEOMETRY_STATUS, GEOMETRY_REPORT]

**主结果口径：** 主集合G2.25/4.5，先每source平均两ξ和两G，再128sources、10000次配对source bootstrap。actual(T(Y))相对decoupled(T(Y))两seed为−.0392dB [−.0755,−.0017]、−.0305 [−.0658,.0035]；相对generic(T(Y))为+.0518/+.0572dB，达不到两主对照各≥+.10dB及区间下界>0等门槛。actual相对旧C0仅+.0122/+.0260dB，区间均跨0，SSIM分别−.00129/−.00317。弱G下降/强G上升，不构成总体升级。[GEOMETRY_REPORT, GEOMETRY_GATE]

原C0的G2.25/G4.5为22.7625/20.6289dB；actual两个seed为22.6279/20.7879、22.6304/20.8130dB。前端T(Y)自身比Y提高约.51—.56dB，未等幅转化为最终恢复收益；“更像GT的中间图”不是恢复效果保证。

**保留什么、不下什么结论：** 保留冻结C0与完整负结果、8模型checkpoint路径/SHA清单、比较与失败样本。标量训练覆盖不足得到支持，但actual轨迹独立必要性、无损升级或blind真实场景性能未获支持。Gate0固定ξ标量曲线残差不能证明所有物理算子族不闭合；后续oracle/action方案只是报告草案，没有因这次整理而启动。该支线没有失败推翻老师v2，因为方法、数据和时间身份都不同。

**有效入口与缺口：** `analysis/v03_validation_20260909/REPORT.md`、`PROTOCOL.md`、`FINAL_STATUS.json`、`evidence/stage1_analysis.json`。原checkpoint路径为 `/data2/yanglin/v03_validation_20260909/models/stage1-<arm>-<seed>/resume.pt`，逐模型SHA保存在状态JSON。首轮启动前未逐次保存完整新脚本快照，需结合WORKLOG追溯；seed2起才有code_snapshots，不能把后补snapshot当早期冻结证据。本次未检查远端模型现存性。

## 10. 非主线资料索引与接手缺口

| 资料/分支 | 用途与边界 | 本地入口 |
|---|---|---|
| v2递归梯度smoke | 工程图连接证据，不是正式恢复训练 | v2 README引用 `/data2/yanglin/v01_recursive_gradient_smoke_20260915/attempt3`，本次只核文档引用 |
| v3 Stage1/BF16/tilt导数/NPU backward | 真实反向与数值验收；原失败保留，不把backend根因称全部定位 | `v031_overfit100_g4321_n4/artifacts/V031_IMPLEMENTATION_REPORT.md`、`docs/NPU_BACKWARD_DIAGNOSIS.md` |
| v3 HCCL/恢复/并行分片 | 工程重试和状态恢复，与方法质量分开 | `artifacts/STAGE2_RESUME_DIAGNOSIS.md`、`docs/STAGE2_HCCL_TIMEOUT_REPAIR.md`、`artifacts/evaluation_sharded_4card/` |
| v3训练目标梯度分解 | θ上的LC/LR训练梯度；不等于精修时ξ上的∇J/∇D | `artifacts/gradient_balance_update900_20260917_cpu02/REPORT.md` |
| v3精修取证 | 固定checkpoint的目标冲突和干预 | `artifacts/refinement_forensic_20260918/FINAL_REPORT_ZH.md` |
| v3后补数组/训练loss图 | 已保存结果的数据与可视化补充，不是新训练 | `v031_followup_20260918/reports/array_supplement_20260919/`、`reports/loss_curves_20260920/` |
| 几何支线文献/草案 | 支撑当时近邻调查与未执行后续设计；论文可用性/发表状态本次未联网更新 | `analysis/v03_validation_20260909/LITERATURE_UPDATE.md`、`REPORT.md` §6–7 |

最重要的交接缺口是：远端大权重/FP32数组本次未逐文件现存或重新hash；v2.5没有full-val/最终test；v3没有多训练seed、外部benchmark或source-scaling；I1确认未开是规则性停止；几何支线原始首轮代码快照不完备。以上缺口不能用工程PASS、训练loss下降、训练集正增益或后来版本结果补成成功。

对新接手者的执行边界：先读历史最终报告和对应验收，再核当前远端资产与完整身份；**这份交接不代表授权恢复训练、重跑评价、删除原始资产或改写历史失败**。当前午夜训练的最终验收/停止/备份另见总交接完成记录。

## 附录：本地证据路径与 SHA256

下表由本次实际读取的本地文件生成；远端checkpoint历史SHA单独保存在上述阶段与 `legacy_evidence.json`，没有冒充本次重新计算。

<!-- SOURCE_TABLE_GENERATED -->

| 来源代号 | 本地文件（可点击） | 本次 SHA256 |
|---|---|---|
| INDEX | [EXPERIMENT_INDEX.md](<../evidence/EXPERIMENT_INDEX.md>) | `b069b571b5ec32256196febcb51898343e64619d0e9ceaee34d5ea662c82c1ca` |
| V2_README | [v03_recursive_from_scratch_20260915/README.md](<../evidence/v03_recursive_from_scratch_20260915/README.md>) | `0d12990549fe7ab723b498a6508ac62bfab85a6f32085e082cef8b41231ec443` |
| V2_EXECUTION | [v03_recursive_from_scratch_20260915/docs/EXECUTION_STATUS.md](<../evidence/v03_recursive_from_scratch_20260915/docs/EXECUTION_STATUS.md>) | `fbc4b72f4a877441112a320f2505a0ff898431322bbeb42a11eadd68c9b5a3c7` |
| V2_TERMINATION | [v03_recursive_from_scratch_20260915/artifacts/DATASET_PROTOCOL_CHANGE.json](<../evidence/v03_recursive_from_scratch_20260915/artifacts/DATASET_PROTOCOL_CHANGE.json>) | `bde3f810a3251dde546cadc9c5ec7f6aa2b0ce0e0d857bfd0c56ff66589265d0` |
| V25_DATA | [v03_half_recursive_20260915/docs/DATASET_AUDIT.md](<../evidence/v03_half_recursive_20260915/docs/DATASET_AUDIT.md>) | `6fce31cb70c051a8f68150fcb921711df57cb2fa0076a16b2748b1d497e930e7` |
| V25_PROTOCOL | [v03_half_recursive_20260915/docs/FORMAL_PROTOCOL.md](<../evidence/v03_half_recursive_20260915/docs/FORMAL_PROTOCOL.md>) | `9f9cdaf3d5f7f8324a236d736dcf45e19df71b23ad637e8ada0933c9aef0135c` |
| V25_CONFIG | [v03_half_recursive_20260915/configs/half_formal_01.json](<../evidence/v03_half_recursive_20260915/configs/half_formal_01.json>) | `3ade4c5dfd327766727fe841339844e10dcb73bc134bc89e07fa2a2abca92d3f` |
| V25_HANDOFF | [v03_half_recursive_20260915/docs/NOON_HANDOFF.md](<../evidence/v03_half_recursive_20260915/docs/NOON_HANDOFF.md>) | `7e02635f073d733948ffa3ddef7c496a6afe07ffe47e165c72d7c49876fc8eb2` |
| V25_STATUS | [v03_half_recursive_20260915/artifacts/noon_handoff/handoff_summary.json](<../evidence/v03_half_recursive_20260915/artifacts/noon_handoff/handoff_summary.json>) | `17360f9e3c676002570d4af4dc5591cafc90cad0a08c89ae3a68da830b252c33` |
| V25_RESUME_AUDIT | [v03_half_recursive_20260915/artifacts/noon_handoff/noon_resume_audit.json](<../evidence/v03_half_recursive_20260915/artifacts/noon_handoff/noon_resume_audit.json>) | `57966ec028545960bc5f8ab603cd8db03f2f8cda2c4dff95da910d551fb75589` |
| V25_VAL_CSV | [v03_half_recursive_20260915/artifacts/noon_handoff/validation_metrics_per_G_K.csv](<../evidence/v03_half_recursive_20260915/artifacts/noon_handoff/validation_metrics_per_G_K.csv>) | `555727cf9f1d0e9d4ee370122657f04732e5def8cffbe61c998da54725825e1a` |
| V3_DATA_DELIVERY | [v03_overfit100_g4321_n4/DELIVERY_REPORT.md](<../evidence/v03_overfit100_g4321_n4/DELIVERY_REPORT.md>) | `8daccef59efbce20d8cd26bcceac0caa7202a01651839f3cca233953a2375665` |
| V3_DATA_ACCEPTANCE | [v03_overfit100_g4321_n4/artifacts/independent_cpu_acceptance.json](<../evidence/v03_overfit100_g4321_n4/artifacts/independent_cpu_acceptance.json>) | `bea2fef64bf6f69c1137809f5113c2cf0a326d528595e6ffcfa10eb87a22552b` |
| V3_STAGE1_METHOD | [v031_overfit100_g4321_n4/docs/V031_STAGE1_METHOD_AND_IMPLEMENTATION.md](<../evidence/v031_overfit100_g4321_n4/docs/V031_STAGE1_METHOD_AND_IMPLEMENTATION.md>) | `bbcb67a6c063778b426a737edc4e8af13d8756b3631f19d33732d7c9e137b8d5` |
| V3_STAGE1_REPORT | [v031_overfit100_g4321_n4/artifacts/V031_IMPLEMENTATION_REPORT.md](<../evidence/v031_overfit100_g4321_n4/artifacts/V031_IMPLEMENTATION_REPORT.md>) | `6cc9f150b4f3066f4a4c0256cfe40d7e9f34ce496526731fb151bf51d09728ce` |
| V3_PROTOCOL | [v031_overfit100_g4321_n4/artifacts/V031_TRAINING_PROTOCOL.md](<../evidence/v031_overfit100_g4321_n4/artifacts/V031_TRAINING_PROTOCOL.md>) | `502da957564a90ddb08bb25a72b0f8efd09c5bdcf2961777757ad3f0a0595122` |
| V3_CONFIG | [v031_overfit100_g4321_n4/configs/train_v031.json](<../evidence/v031_overfit100_g4321_n4/configs/train_v031.json>) | `ee4c2e9074b3bd11333364fcb2ce553a1985d8120a4f74974fa8cc842b50f706` |
| V3_FINAL_STATUS | [v031_overfit100_g4321_n4/artifacts/WORKFLOW_STATUS.json](<../evidence/v031_overfit100_g4321_n4/artifacts/WORKFLOW_STATUS.json>) | `4b1c15314b416d2fea252e9108a242f48fd97bd903dd1933d5eea8015a3bb9db` |
| V3_U1000_REPORT | [v031_overfit100_g4321_n4/artifacts/V031_UPDATE1000_REPORT.md](<../evidence/v031_overfit100_g4321_n4/artifacts/V031_UPDATE1000_REPORT.md>) | `afeef85610bf08995ff41c93f2465ad7d0cdc8f1a1e7302dce487597a33dda04` |
| V3_U1000_STATE | [v031_overfit100_g4321_n4/artifacts/CHECKPOINT_001000_STATE_CHECK.json](<../evidence/v031_overfit100_g4321_n4/artifacts/CHECKPOINT_001000_STATE_CHECK.json>) | `12c5faedae8eed28fc682b15781a79d4611f249497b501172431a2b63451cb77` |
| V3_REFINEMENT_FORENSIC | [v031_overfit100_g4321_n4/artifacts/refinement_forensic_20260918/FINAL_REPORT_ZH.md](<../evidence/v031_overfit100_g4321_n4/artifacts/refinement_forensic_20260918/FINAL_REPORT_ZH.md>) | `c296f326c690dd058ed9b54b18ff707ce1e8b36855a34aeb62e0b8a38284e4d7` |
| V3_FOLLOWUP_README | [v031_followup_20260918/reports/README.md](<../evidence/v031_followup_20260918/reports/README.md>) | `688f919c4c15d736ad9155fdbe0ba6191707ed9788b403d4c08c79573eeec2e8` |
| V3_FOLLOWUP_TRAINING | [v031_followup_20260918/reports/TRAINING_REPORT.md](<../evidence/v031_followup_20260918/reports/TRAINING_REPORT.md>) | `c8c757ca54fcb66e32c94c13aca2c37aaa0348368165d52b7db9d0f1ada8ad07` |
| V3_T1_METHOD | [v031_followup_20260918/shared/authority/01_T1_DIRECT.md](<../evidence/v031_followup_20260918/shared/authority/01_T1_DIRECT.md>) | `f16bfa0e0afd3a1ae351f25ae21fd90bdc01cd85ea66aea04f14083c142d2a0c` |
| V3_T2_METHOD | [v031_followup_20260918/shared/authority/02_T2_FEEDBACK_R.md](<../evidence/v031_followup_20260918/shared/authority/02_T2_FEEDBACK_R.md>) | `63fe8bcc0dd0788cf82760f572e9c62ecbaea07ea187e32cac9f1b3cbc54373d` |
| V3_T1_CONFIG | [v031_followup_20260918/configs/T1_DIRECT_formal.json](<../evidence/v031_followup_20260918/configs/T1_DIRECT_formal.json>) | `6bd86e78609069bc7589b39ee4a6313eacaca40fd480274396e8c57067274a0b` |
| V3_T2_CONFIG | [v031_followup_20260918/configs/T2_FEEDBACK_R_formal.json](<../evidence/v031_followup_20260918/configs/T2_FEEDBACK_R_formal.json>) | `73f1b4bdec8b4a3db3f0bd898f06b05914cb925129f91b48d3d09612bf9b84ff` |
| V3_FOLLOWUP_COMPLETE | [v031_followup_20260918/reports/COMPLETION.json](<../evidence/v031_followup_20260918/reports/COMPLETION.json>) | `8d10a4916ad0af2383a63e45cfcc06ab29d10c197bb6f4d15fda7c6f2bc72cf2` |
| V3_FOLLOWUP_MISSING | [v031_followup_20260918/reports/MISSINGNESS.json](<../evidence/v031_followup_20260918/reports/MISSINGNESS.json>) | `93921cdcb8ea91fdbb7751eff5ccb384b69f60034eef8fa9a0e216ddd7c18204` |
| V3_FOLLOWUP_RESUME | [v031_followup_20260918/reports/CHECKPOINT_AND_RESUME.md](<../evidence/v031_followup_20260918/reports/CHECKPOINT_AND_RESUME.md>) | `70529152b002c2c1ef9ae886c599ae2ddd8174c8f2bcdd28913ce57d53658daa` |
| V3_HELDOUT | [v031_followup_20260918/reports/HELDOUT_SUMMARY.json](<../evidence/v031_followup_20260918/reports/HELDOUT_SUMMARY.json>) | `2850308b82fb3ae20e0ea89388447c1abdbc65b6201012ccc04efacc36407d93` |
| V3_HELDOUT_CURVE | [v031_followup_20260918/reports/heldout_curve_20260919_2138/delivery/REPORT.md](<../evidence/v031_followup_20260918/reports/heldout_curve_20260919_2138/delivery/REPORT.md>) | `238addf01f872f45e1dafae8b5c023f4e5711d2730b7d830cffcd2b13d038698` |
| V3_I1_REPORT | [v031_followup_20260918/reports/I1_REPORT.md](<../evidence/v031_followup_20260918/reports/I1_REPORT.md>) | `b77e00b3ed8069c354d3b939f2a51d46ba5e524b1a7147cb308c80da21369468` |
| GEOMETRY_PROTOCOL | [analysis/v03_validation_20260909/PROTOCOL.md](<../evidence/analysis/v03_validation_20260909/PROTOCOL.md>) | `d04e74a274b841997918ad7fa2c7f377584cbaa6e36161641a04111e167e4bcb` |
| GEOMETRY_REPORT | [analysis/v03_validation_20260909/REPORT.md](<../evidence/analysis/v03_validation_20260909/REPORT.md>) | `e53e188fe6c5f95c25a1efb775acf2d81ae5b6c58d920758e29b6143ac45a402` |
| GEOMETRY_STATUS | [analysis/v03_validation_20260909/FINAL_STATUS.json](<../evidence/analysis/v03_validation_20260909/FINAL_STATUS.json>) | `cc7936f77cb449f6339bb8b6cced8a68a8cc29e787b9b1e7452721228544deff` |
| GEOMETRY_GATE | [analysis/v03_validation_20260909/evidence/stage1_analysis.json](<../evidence/analysis/v03_validation_20260909/evidence/stage1_analysis.json>) | `74232f19eb5a7a57e24777d1263a698d3efb5aa4fa86c9076864fa3da18fa908` |
| GEOMETRY_QA | [analysis/v03_validation_20260909/FINAL_QA.json](<../evidence/analysis/v03_validation_20260909/FINAL_QA.json>) | `b8e25bbda8ee4ef02b7ba9e2a0a59f6a5ad2a3825bc8e5a7657011b6b99d47ad` |
| V1_FINAL_REPORT | [reports/bridge_v01_retrospective/sources/original_final_report.md](<../evidence/reports/bridge_v01_retrospective/sources/original_final_report.md>) | `6bc192bca640fae51fbdc5fdc84d4ad9389034a1690b8ef712b8fb1040ea0dc1` |
| V1_FINAL_SUMMARY | [reports/bridge_v01_retrospective/sources/final_summary.json](<../evidence/reports/bridge_v01_retrospective/sources/final_summary.json>) | `65657e8fdabf651778e714e3fda5c47bdd842b990d39e6e65bba2943d88081e4` |
| V1_TRAIN_SUMMARY | [reports/bridge_v01_retrospective/sources/training_summary.json](<../evidence/reports/bridge_v01_retrospective/sources/training_summary.json>) | `7d53310b8be1cd83191ac6c34d98475c01ac1d1457b5968a01b80e9e86cac6e6` |
| V1_GENERATION_SUMMARY | [reports/bridge_v01_retrospective/sources/generation_summary.json](<../evidence/reports/bridge_v01_retrospective/sources/generation_summary.json>) | `701df72796488fb53417b29f34643b6e3406e7cb243e21c89608b48e45a5feb8` |
| V1_TEST_ROLLOUTS | [reports/bridge_v01_retrospective/sources/test_rollouts_summary.json](<../evidence/reports/bridge_v01_retrospective/sources/test_rollouts_summary.json>) | `fdce922d4d05226c31eff3abd72a61ac227d8b250584bb76d7e3928b9abc00e1` |
| V1_TEACHER_FORCED | [reports/bridge_v01_retrospective/sources/teacher_forced_summary.json](<../evidence/reports/bridge_v01_retrospective/sources/teacher_forced_summary.json>) | `01d343331d7f4b3bb0007708382e075f39b5beaa759d5005b96165177897806e` |
| V1_DATA_CONFIG | [reports/bridge_v01_retrospective/sources/configs/dataset_v01.yaml](<../evidence/reports/bridge_v01_retrospective/sources/configs/dataset_v01.yaml>) | `ed9d1c8983dc19d82729097de3e2526200edb3ea15da72c5ad5a12148f527a32` |
| V1_TRAIN_CONFIG | [reports/bridge_v01_retrospective/sources/configs/train_v01.yaml](<../evidence/reports/bridge_v01_retrospective/sources/configs/train_v01.yaml>) | `327535abc5034537d05f10246e7236f669cb3ba284334cf399bdfbe4bcfa5de9` |
| V1_INFERENCE_CONFIG | [reports/bridge_v01_retrospective/sources/configs/inference_v01.yaml](<../evidence/reports/bridge_v01_retrospective/sources/configs/inference_v01.yaml>) | `0cf635b4e72a1261404627e478117845d1a7fa8183f9de6ec9e7a65956e172c9` |
| V1_EVIDENCE_AUDIT | [reports/bridge_v01_retrospective/analysis/evidence_audit.md](<../evidence/reports/bridge_v01_retrospective/analysis/evidence_audit.md>) | `bfe0e2747f8ca4279d05057550424d66eb5ed500d2b84d432b414457ec7c8b21` |
| V1_REMOTE_TRAIN | [reports/bridge_v01_retrospective/sources/remote_code/train.py](<../evidence/reports/bridge_v01_retrospective/sources/remote_code/train.py>) | `95b2384368952c7f6fa5a716672243c4fbaba5dd8a6af7f26fd23ae7e329c5c5` |
| V15_METHOD | [v02_new_protocol_20260829/docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_METHOD.md](<../evidence/v02_new_protocol_20260829/docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_METHOD.md>) | `834723e3cd6883fc9873eeb48b59ad723560ac35265ef8c3f8b2d220fc1a0a7f` |
| V15_PLAN | [v02_new_protocol_20260829/docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_TRAINING_PLAN_CODEX.md](<../evidence/v02_new_protocol_20260829/docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_TRAINING_PLAN_CODEX.md>) | `64a166fdc6b5b00d6fff4253980431f5bfd402d85683d69d3b68101464a89f53` |
| V15_AMENDMENT | [v02_new_protocol_20260829/docs/EXECUTION_AMENDMENT_20260829.md](<../evidence/v02_new_protocol_20260829/docs/EXECUTION_AMENDMENT_20260829.md>) | `09adc0ec5cd5fdece2a363bdc07611bf9238b4151844fecbccb3e59ab94faf35` |
| V15_CONFIG | [v02_new_protocol_20260829/configs/v02.yaml](<../evidence/v02_new_protocol_20260829/configs/v02.yaml>) | `9a829762f47e48eb6e81ee9b47ea4d201800dbea025c83fa5b2ada5832e361d0` |
| V15_OBJECTIVE_REPORT | [analysis/v02_a1_20260830/report/V02_A1_OBJECTIVE_HANDOFF_FOR_GPT.md](<../evidence/analysis/v02_a1_20260830/report/V02_A1_OBJECTIVE_HANDOFF_FOR_GPT.md>) | `2f8e04efe2506aab0be15247e1490caef19f4057e11b0e69400aafb3db1f7ef6` |
| V15_QA | [analysis/v02_a1_20260830/report/qa_summary.json](<../evidence/analysis/v02_a1_20260830/report/qa_summary.json>) | `0d7c05027f1c4ac7757c41fed0b3c4560d6afb9722e6d39cc63f36fed4983172` |
| V15_TRAIN_SUMMARY | [analysis/v02_a1_20260830/data/a1/training_summary.json](<../evidence/analysis/v02_a1_20260830/data/a1/training_summary.json>) | `7a0906c943e055cfbbb9cf213d4a682df9e5adb7a9fcff144a4ea2fa029bfb3c` |
| V15_ORIGINAL_EVAL | [analysis/v02_a1_20260830/data/eval/v02_pilot_report.json](<../evidence/analysis/v02_a1_20260830/data/eval/v02_pilot_report.json>) | `71efc2992f186537edc47ee960a00620bdd83ccec81c001961c6bc6915448498` |
| V15_GATE_SENSITIVITY | [analysis/v02_a1_20260830/derived/gate_sensitivity.csv](<../evidence/analysis/v02_a1_20260830/derived/gate_sensitivity.csv>) | `1992c99d9a3fa1c17489609087b762f12a3110d8635fa39ad76d9413ac035573` |
| V15_CORRECTED_G | [analysis/v02_a1_20260830/derived/corrected_per_g.csv](<../evidence/analysis/v02_a1_20260830/derived/corrected_per_g.csv>) | `319b0e5ebcaac3ac1d690ad4b7dd353879664baac4256b32ff30661387d8a86c` |
| V15_CHECKPOINT_G450 | [analysis/v02_a1_20260830/derived/checkpoint_progression_g450.csv](<../evidence/analysis/v02_a1_20260830/derived/checkpoint_progression_g450.csv>) | `7c5ca68c887db89ce1961257946854d1cd4ed0b7cef7e882ffd9fd74135a9992` |
| V15_DIFFERENTIABILITY | [analysis/v02_a1_20260830/derived/differentiability_summary.json](<../evidence/analysis/v02_a1_20260830/derived/differentiability_summary.json>) | `c675176469d0a2dc010b44170b5cbdfd4ee4f784c4d6ee984e16893bd1963c8a` |
| V15_TRAIN_SOURCE | [analysis/v02_a1_20260830/source/v02_train_pilot.py](<../evidence/analysis/v02_a1_20260830/source/v02_train_pilot.py>) | `62f4e8f461646844add7cb918d4227ab5c85e195a805c2166f21ea756e9957c6` |
| V15_EVAL_SOURCE | [analysis/v02_a1_20260830/source/v02_evaluate_pilot.py](<../evidence/analysis/v02_a1_20260830/source/v02_evaluate_pilot.py>) | `672464174b8c2c1ce1588bf0a33cef01dd2c0f85b3406b27ccaddb133b6b277b` |
| V15_LOSS_SOURCE | [analysis/v02_a1_20260830/source/v02_losses.py](<../evidence/analysis/v02_a1_20260830/source/v02_losses.py>) | `229ee5211920d8078506136301b9df096705e87ce0a6abbc97eaf142bd915008` |
| V15_LEGACY_CONFIG | [v02_src/configs/train_v02.yaml](<../evidence/v02_src/configs/train_v02.yaml>) | `fd0072de546ea5a4b00b5d1975604fcb053d51a01c272d9072c3591be0ae91e1` |
| V15_LEGACY_AUTHORITY | [v02_src/docs/AUTHORITY.md](<../evidence/v02_src/docs/AUTHORITY.md>) | `dd2c695077d0433019f1afb0803a8f12e7d8192aeaaa5cc9b74c1f369d5a6267` |

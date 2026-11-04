# 湍流图像恢复项目交接

**接手重点：读懂现有 V5 模拟器，跑通最新 UniDB D0，再由接手方接入 consistency 方法。**

状态基准：2026年11月4日交接快照（北京时间），不是服务器实时监控。现有训练已结束；本次交接资料与运行示例已整理，接收人访问和独立重跑仍需实际确认。

项目任务是单帧大气湍流图像恢复：从含空间变化模糊、几何扭曲的RGB图恢复清晰图，并检验物理恢复路径能否优于匹配条件下的直接恢复D0。下面先介绍现在可以交接和运行的模块；完整历史过程收在页末附录。

## 1. 现在接什么

|模块|当前状态|从哪里开始|
|---|---|---|
|**V5 simulator：可复用的物理退化模块**|输入清晰图、固定raw ξ和强度g，执行空间变化模糊、Smooth与warp；可构造随时间变化的物理路径。保留原Gate局限和研究接收记录。|[原理、公式与接口](docs/meeting_20261104/SIMULATOR_GUIDE.md) · [对应源码](source_snapshot/v5_v1_simulator_20261020/)|
|**UniDB D0：最新像素空间扩散基线**|原路线best＝latest **U67180**；另一路best **U10000**、latest **U40848**。已有完整状态、最终评价及NPU恢复示例；当前没有接入外层模拟器或consistency loss。|[NPU逐步运行](docs/meeting_20261104/NPU_RUNBOOK.md) · [最终权重登记](handoff/registers/WEIGHT_REGISTER.csv)|
|**会议提出的下一步：物理路径＋consistency**|由接手方提供其原consistency/CGDB实现，确认时间方向、目标、采样器与梯度需求后接入；目前尚未实现，也没有新实验性能结论。|[方法说明与接口表](docs/meeting_20261104/METHODS_AND_NEXT_STEP.md#3-接手方要接的接口)|

**当前恢复质量仍有限：**最新UniDB两路线的最终五档PSNR均低于相应输入基线；跨真实场景的稳定强湍流恢复优势、物理多步稳定优于自身第一步，以及UniDB退化的唯一原因均未确立。训练结束和工程能够运行不代表科研目标已达到。

## 2. 接手老师按这个顺序看、跑

1. 打开[离线交互图解](docs/meeting_20261104/visual_guide.html)（下载/解压后在浏览器打开），或[文字总指南](docs/meeting_20261104/START_HERE.md)，先看当前模块和固定ξ强度示例。
2. 读[模拟器说明](docs/meeting_20261104/SIMULATOR_GUIDE.md)，对照[逐函数代码导航](docs/meeting_20261104/CODE_MAP.md)，理解两类raw ξ、PSF权重、像素位移与归一化采样网格。
3. 按[NPU运行指南](docs/meeting_20261104/NPU_RUNBOOK.md)取得现有de36访问方式，确认资产与解释器，先跑CPU模拟器，再在分配到的NPU上跑模拟器及UniDB最终EMA的100步恢复。
4. 用[接收清单](docs/meeting_20261104/MEETING_CHECKLIST.md)确认访问和重跑结果，再安排新方法；[完整性复查](docs/meeting_20261104/COMPLETENESS_REVIEW.md)列明已覆盖内容及仍待实际对接的事项。

现有NPU入口是`/data2/yanglin/v6_diffuir_l_20261027/env/bin/python`，它通过`.pth`依赖多个旧环境目录。模拟器位于`/data2/yanglin/v5_v1_simulator_20261020`，UniDB位于`/data2/yanglin/unidb_d0_20261102`；完整账号、路径和逐步命令以运行指南为准，不能仅拷贝一个venv文件夹。

本次已完成CPU/NPU模拟器前向与一阶反向、UniDB最终EMA的CPU前向及NPU100步恢复示例，**0次优化器更新**。范围和原始记录见[运行验证](docs/meeting_20261104/VALIDATION.md)；这些检查与原正式实验评价分别记录。旧训练器绑定旧目录、截止和源清单，新训练需要按运行指南第6节另行适配。

原论文先看P2S、UniDB，再结合接手方的一致性方法：[论文阅读顺序与项目对应关系](docs/meeting_20261104/PAPERS.md)。已有论文草稿、19点汇报和最终结果素材见[论文与汇报素材索引](docs/meeting_20261104/PAPER_MATERIALS.md)。

## 3. 最新模型与正式结果

UniDB双路线于**11月4日00:00**截止边界安全停训，原路线U67180、改进路线U40848；午夜后没有新启动的优化器更新。两线完成五档强度、每档1000张的EMA评价及独立RAW监控，完整状态与保存预测的四指标CPU复核通过。三个唯一完整权重已在本机备份并核验SHA，见[权重登记](handoff/registers/WEIGHT_REGISTER.csv)。三阶段自有训练、评价与CPU收尾进程均已核验退出，自动跟进已结束。

**只比较 UniDB 匹配协议下的 G4 全量验证。** 每档1000张＝250个来源×4个湍流实现；模型使用EMA、原生100步SDE。这里不是每档100张的小监控集，也不是锁定Test或真实场景测试。`best` 指既定完整评价候选池的选择结果。

| 候选 | PSNR ↑ | SSIM ↑ | MAE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| 未恢复的 Input | **18.795938** | **0.451751** | **0.080178** | 0.571629 |
| 原路线续训前 D0 U45411 | 15.894991 | 0.302786 | 0.107844 | 0.425333 |
| **原路线 best＝latest U67180** | **16.211783** | **0.315317** | **0.103124** | **0.411531** |
| 改进路线 best U10000 | 14.708872 | 0.126323 | 0.134535 | 0.771947 |
| 改进路线 latest U40848 | 13.761130 | 0.106128 | 0.151208 | 0.745103 |

原路线比续训前提高约 **0.317dB**，仍比输入低约 **2.584dB**；LPIPS改善，但PSNR、SSIM、MAE仍弱于输入。改进路线最终也未达到恢复目标。“改进路线”是实验名称，不代表已经取得改进。两路起点、loss、学习率及卡数不同，不能将结果单归因于某一因素。

训练集为Places365的50类、10000个clean来源，组成G1–G4共160000个训练配对。原路线累计67180次更新、约**40.308个等效epoch**，其中U45411之后新增约13.0614轮；改进路线约24.5088轮。等效epoch按累计呈现量计算，含重复采样，不代表独立内容数量或收敛。

**直接看最终图片和曲线：**

- 原路线：[全程train loss](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_TRAIN_GLOBAL.png) · [全量验证四指标](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_VAL_4METRICS_FULL1000.png) · [监控集四指标](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_VAL_4METRICS_MONITOR100.png)。
- 最终Input / Best / Latest / GT固定样本：[G1](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_FINAL_BEST_LATEST_G1.png) · [G2](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_FINAL_BEST_LATEST_G2.png) · [G3](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_FINAL_BEST_LATEST_G3.png) · [G4](handoff/evidence/unidb_dual_20261104_until00/delivery/original_route/ORIGINAL_FINAL_BEST_LATEST_G4.png)。原路线best与latest是同一权重，两列相同；固定样本不代替全量结果。
- [完整五档四指标CSV](handoff/evidence/unidb_dual_20261104_until00/report/FULL_QUALITY_COMPARISON.csv) · [最终实验报告](handoff/evidence/unidb_dual_20261104_until00/report/FINAL_REPORT_ZH.md)。

最终状态依据：[主交接说明](handoff/documents/MAIN_HANDOFF_FINAL_v2.md) · [UniDB实验收尾与验收](handoff/experiment/README_ZH.md) · [模型角色与身份](handoff/provenance/UNIDB_FINAL_HANDOFF_STATE.json)。历史文件中的`RUNNING`、`pending`与计划步数只描述当时。

## 4. 资料范围与交接边界

本仓库公开提供技术交接资料，包含原11页[Word](handoff/documents/Turbulence_Project_Handoff_FINAL.docx) / [PDF](handoff/documents/Turbulence_Project_Handoff_FINAL.pdf)、[Markdown同稿](handoff/documents/MAIN_HANDOFF_FINAL_v2.md)、七份登记表、结果图、证据、选定源码及本次会议补充。原PDF/Word是午夜收尾快照；当前接手先读上面的会议后指南。对外交付统一使用一个无修订号后缀的交接ZIP；历史源码和证据文件的原名保留用于核验。

数据、完整权重和全量预测未进入Git，按[未包含资产清单](handoff/NOT_INCLUDED.md)、[资产登记](handoff/registers/ASSET_REGISTER.csv)和[权重登记](handoff/registers/WEIGHT_REGISTER.csv)取得。指南按现有de36环境验证；迁到新机器仍要补齐官方vendor、载荷及环境依赖。部分旧A100资产尚未核实，不能由连接失败推断文件丢失。

本仓库从2026-11-04交接快照开始Git版本管理，保存小文件资料与选定源码，不是全部实验磁盘镜像。原`handoff/`与`source_snapshot/`保持清单中的字节身份；详细冻结范围、离线校验和第三方许可说明在下方附录。需要追溯具体实验时再查[原交接入口](handoff/README.md)及[资产操作说明](handoff/history/OPERATIONS_GUIDE_ZH.md)。

后续研究目标、实验负责人、资源、consistency代码版本和论文分工由双方对接确定。当前没有排队中的追加训练，原控制器和启动脚本保留作历史源码。

<details>
<summary><strong>历史背景：方案演进、失败记录与未决问题</strong></summary>

完整的[换方案理由、失败记录与未决问题](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md)已按**31个节点**逐项整理：每项写清当时为何改、实际改什么、哪里失败或停止、还缺什么证据，并链接原件。未记录的动机明确标未知，不用事后结果补造理由。

| 转向 | 为什么换／为什么做 | 哪些仍没有解决 |
|---|---|---|
| [v1→v1.5](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#l1) | 原观测重建误差低，不保证ξ跨强度传输或下一状态正确；补轨迹与弱态约束。 | A1评价身份纠偏后仍未过继续门槛；不是模拟器不可微的定论。 |
| [v2→v2.5→v3](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#l2) | 用自身预测递归；随后改半量协议，再做小样本拟合诊断。 | v2未正式开训，v2.5因时间边界暂停；v3能拟合却在留出来源回退。 |
| [T1/T2、I1与几何支线](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#l5) | 分别检查直接恢复、物理反馈、精修保护和几何轨迹的必要性。 | 留出回退、精修候选及几何Gate未通过；未开的确认集不是隐藏的成功证据。 |
| [V4接口、骨干、数据与平台](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#v4-01) | 新方法加入固定原观测；继续检查不同骨干、预训练和更大数据；A100迁移及NAFNet由用户指定。 | 不能当单因素骨干排名。Train8000的RAW/OFF主表中直接恢复PSNR最高；Train15000按用户要求停止，部分停点没有full。 |
| [V4→V5及A→B](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#v5-01) | 新连续时间/模拟器规格，选择Concertormer；先中间态训练再对齐真实递归。 | Simulator原FAIL带局限接收；B胜独立D0却末步弱于首步；首个失稳原因未定位。 |
| [ResShift→DiffUIR-L](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#v5-05) | 所测冻结VQ四次RGB往返累积漂移，随后选像素空间恢复器。 | 只约束所测codec组合，不否定所有latent方法；短pilot工程通过没有证明质量。 |
| [V6目标与强度调整](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#v6-3) | 用户要求切回官方原生目标、扩至G1–G4，再有限比较G4偏重。 | 强弱档取舍、追加门失败；ATSyn有限正结果与OTIS/官方能力退化并存。 |
| [V6→UniDB→双路线](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#u-1) | UniDB已有RGB空间条件和恢复权重，接入较直接；诊断后用户要求原配方与新配方并行。 | 选型不等于湍流排名；微调权重与随机采样有不良交互，唯一根因未定；双线不是单因素loss对照。 |
| [中午→18点→午夜→交接](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#u-4) | 用户明确延长训练时限，之后决定结束研究、整理交接。 | 延期不是质量门通过；最终验收完成但恢复质量目标未达到。 |

**接手不用猜的部分：**[12项未决问题及核实入口](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#open-issues) · [模型、资产、复现与失败解释FAQ](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md#handoff-faq)。跨版本常同时改变数据、初始化、目标或预算，以上是决策过程，不是同条件排行榜。几何校正独立于老师的v2；旧文件名保留。

</details>

<details>
<summary><strong>附录：七份注册表、各版本源码、冻结清单、校验与迁移条件</strong></summary>

## 七份注册表

| 注册表 | 接手时用来查什么 |
|---|---|
| [VERSION_REGISTER](handoff/registers/VERSION_REGISTER.csv) | 展示版本、历史名称与版本范围 |
| [EXPERIMENT_REGISTER](handoff/registers/EXPERIMENT_REGISTER.csv) | 逐实验方法、实际执行状态、结果入口 |
| [DATA_REGISTER](handoff/registers/DATA_REGISTER.csv) | 数据来源、划分、强度、人口和原路径 |
| [CODE_ENV_REGISTER](handoff/registers/CODE_ENV_REGISTER.csv) | 源码、配置、环境与复现依赖 |
| [WEIGHT_REGISTER](handoff/registers/WEIGHT_REGISTER.csv) | checkpoint 角色、路径、SHA 和核验层次 |
| [ASSET_REGISTER](handoff/registers/ASSET_REGISTER.csv) | 数据、模拟器、模型及归档资产位置 |
| [EVIDENCE_REGISTER](handoff/registers/EVIDENCE_REGISTER.csv) | 结论依据、原文件路径及已记录 SHA |

## 仓库内容与源码导航

`handoff/` 是已验收便携包的原样副本，内有文档、注册表、历史叙述、图和证据；其 [MANIFEST.json](handoff/MANIFEST.json) 校验 561 个文件，清单不包含自身字节。[SOURCE_MAP.csv](handoff/SOURCE_MAP.csv) 连接包内位置与原位置；[NOT_INCLUDED.md](handoff/NOT_INCLUDED.md) 登记未包含的大资产。可读 Markdown 的部分链接在原便携包制作时已改写，原文另存 `original_text/`；原文历史链接不保证在新机器可打开。

`source_snapshot/` 包含计划中 754 个独立文件，共 5,941,856 字节，保留原相对目录与原字节。计划记录每项 SHA、来源、用途、已匹配的历史冻结声明及缺口；其中 394 项具有计划记录的冻结匹配，不能把全部文件都称为同一历史运行的冻结代码。该目录是选定源码快照，不涵盖所有诊断、候选或曾出现过的临时脚本。更完整的实验史仍在 `handoff/`。

| 版本 / 分支 | 源码入口（均在 source_snapshot 下） | 范围提示 |
|---|---|---|
| v1（原 V0.1） | [physical_bridge](source_snapshot/physical_bridge/)、[configs](source_snapshot/configs/)、[scripts](source_snapshot/scripts/) | known-g 双头物理桥接，保留实际运行代码与本地依赖 |
| v1.5（原 V0.2） | [v02_new_protocol_20260829](source_snapshot/v02_new_protocol_20260829/)、[A1诊断](source_snapshot/analysis/v02_a1_20260830/) | A1 U3000；正式评价与事后纠偏分别保留，未纳入已替换的早期 v02_src |
| v2（原 V0.3） | [v03_recursive_from_scratch_20260915](source_snapshot/v03_recursive_from_scratch_20260915/) | 正式训练未启动；部分本机工作树与初始三步预检提交不同，见计划 gaps |
| v2.5（原 V0.3-half） | [v03_half_recursive_20260915](source_snapshot/v03_half_recursive_20260915/) | source 层半量、实际 U204 formal，21项正式 release SHA 已核对 |
| v3（原 V0.31 / V0.3.1） | [数据构建](source_snapshot/v03_overfit100_g4321_n4/)、[递归训练](source_snapshot/v031_overfit100_g4321_n4/)、[T1/T2/I1](source_snapshot/v031_followup_20260918/) | 数据目录的 v03 属于 v3；不是老师的 v2，也不是几何校正支线 |
| V4 初期与匹配比较 | [Noise Bridge](source_snapshot/v4_noise_bridge_training_20260929/)、[U-Net Train500](source_snapshot/v4_train500_matched_v2_perf1_eval5kfull/)、[SwinIR](source_snapshot/v4_swinir_l1_train500_2k_20261001/)、[Restormer](source_snapshot/v4_restormer_train2000_npu_20k_20261003/) | NPU 和 A100 来源有别；缺失或不同 SHA 的文件不能冒充冻结版本 |
| V4 NAFNet 与诊断 | [NAFNet候选主体](source_snapshot/v4_nafnet_sidd116m_candidate_20261005/)、[Train8000](source_snapshot/v4_nafnet_sidd116m_train8000_20261006/)、[Train15000](source_snapshot/v4_nafnet_train15000_20261013/)、[Raw–Field](source_snapshot/de36_g4_raw_vs_field_12h_20261015/)、[连续差分pilot](source_snapshot/v4_followup_pilot_lr5e6_attempt2_20261014/) | 各自原配置与冻结修订；Train15000仅含本地可核验部分 |
| V5 | [Simulator](source_snapshot/v5_v1_simulator_20261020/)、[Concertormer D0→A→B](source_snapshot/v5_concertormer_10k_20261021/) | 正式运行及 B 保护续训源码；未补造本地未找到的 D0/B 完整 RUN_CONFIG |
| V6 | [DiffUIR-L stage1依赖](source_snapshot/v5_diffuir_l_stage1_20261026/)、[V6正式与续训](source_snapshot/v6_diffuir_l_20261027/) | 原生 residual、多强度、G4 reweight 和历史物理 A/B；未附权重或 NIQE 参数数组 |
| UniDB | [八卡 D0](source_snapshot/unidb_d0_20261102/)、[双路线至中午](source_snapshot/unidb_dual_20261103_until_noon/)、[延至18点](source_snapshot/unidb_dual_20261103_until18/)、[延至午夜](source_snapshot/unidb_dual_20261104_until00/) | 训练、完整状态审计和后处理源码；官方 vendor 未收录，历史控制器不应直接启动 |

展示名称按项目约定使用新名；旧路径、数据 ID、checkpoint 名及配置保持原名。文件名和旧 README 中的计划、RUNNING、pending 只解释当时快照。实际最终状态以已验收报告和注册表为准。

## 克隆后先做离线完整性检查

克隆公开仓库并使用 Python 3 标准库运行：

```sh
git clone https://github.com/mayday4yl/turbulence-project-handoff.git
cd turbulence-project-handoff
python3 tools/verify_repository.py
python3 tools/receiver/verify_supplement.py
```

也可从任意位置执行该脚本，或用 `--root /path/to/clone` 检查另一份副本。成功时退出码为 0 并输出一行精简 JSON；失败时退出码非零并列出错误。校验读取本仓库的相对路径，不访问清单中的原机绝对路径，不联网，不导入模型或执行实验代码。

校验包括 handoff 清单及全部源码 `repository_path` 的字节数、SHA 和文件集合，以及工作目录中的符号链接、常见模型/数组载荷和大于 50,000,000 字节的文件。`checkpoints/` 等历史证据目录名本身不被禁止，里面的 JSON/CSV 等小证据仍可保留。`.git/` 内部对象不参与工作目录扫描。这个检查证明本地文件与所带清单一致；它不证明上游授权、科学结论、环境可运行或远端大资产仍然存在。

## 如需复现，仍须补齐的条件

1. 从注册表选定具体实验、阶段、真实 update、RAW/EMA 与评价人口；按原清单另取数据、split、canonical 观测、raw ξ 或 seed/shape/SHA、模拟器 basis/PSF、完整 checkpoint 及对应权重。它们被登记，未进入 Git。
2. 对照该阶段的配置与冻结源码身份。源码保留 `/Users/lucas/...`、`/data2/yanglin/...`、A100 根目录等历史绝对路径，迁移需要在新工作目录显式配置路径，不应修改原冻结副本来掩盖差异。18点 UniDB CONFIG 的留存副本位于午夜目录的19点快照中，计划保留了真实证据路径。
3. 重建对应 CUDA 或 Ascend/torch_npu/CANN/HCCL 环境，并核实解释器与实际模块位置。环境文本和 pip freeze 不等于完整 OS、驱动、ISA 与 wheel 镜像；部分 `.pth` 复用旧 stage1/ResShift/基础环境，不能只安装一份 requirements 就视为完成。
4. 补齐外部 PiRN/P2S、UniDB 等依赖及许可；当前排除的 vendor、缺失配置和 SHA 不一致项逐项写在源码计划的 `groups[].gaps`、`excluded` 与 `explicit_vendor_exclusions`。本仓库没有这些组件的完整可执行闭包。
5. 新负责人确定研究目标、设备和运行范围后，在新目录做必要的加载与身份检查，再开展新评测或训练。旧 launcher、run_env、controller 可能设置设备、启动任务或处理进程，不能作为只读查看工具；旧 PID 和卡号不代表当前现场。

主文档第10—12节说明备份、旧资产未核实项及未解决问题。此仓库保留交付证据与研究源码，未授予项目代码的统一开放许可证；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

</details>

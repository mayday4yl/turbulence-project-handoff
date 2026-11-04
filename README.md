# 湍流图像恢复项目交接

**当前进度：实验已结束，资料已交接；当前研究到此收尾，未安排新训练。**

状态基准：2026年11月4日交接快照（北京时间），不是服务器实时监控。

项目想解决的是：把大气湍流造成的**模糊、几何扭曲**图像恢复成清晰图像，并检验多步物理递归是否比直接恢复更有效。研究从物理桥接与递归，发展到多骨干匹配比较，再转向像素空间条件扩散；目前交接的是完整研究过程、可复核结果和资产入口。

> **工程收尾已完成，科研目标尚未达到。** 有局部正结果，但尚未建立跨真实场景稳定的强湍流恢复优势；最新 UniDB 两路线的最终五档 PSNR 均低于对应输入基线。不能把训练完成、loss 下降或图片更锐利等同于恢复成功。

## 1. 现在做到哪里了

| 项目 | 当前进度与交接内容 |
|---|---|
| **最新训练** | UniDB 双路线已在 **11月4日00:00** 截止边界安全停训。原路线 U67180；改进路线 U40848。午夜后没有新启动的优化器更新。 |
| **最终评价** | 两线均完成五档强度、每档1000张的 EMA 评价及独立 RAW 监控；完整状态与保存预测的四指标 CPU 复核通过。 |
| **最终模型** | 原路线 **best＝latest U67180**；改进路线 **best U10000、latest U40848**。三个唯一完整权重已在本机备份并核验 SHA，位置见[权重登记](handoff/registers/WEIGHT_REGISTER.csv)。 |
| **运行收尾** | 本次三阶段所有自有训练、评价与 CPU 收尾进程已核验退出；本次自动跟进已结束。 |
| **交接资料** | 已完成11页[Word](handoff/documents/Turbulence_Project_Handoff_FINAL.docx) / [PDF](handoff/documents/Turbulence_Project_Handoff_FINAL.pdf)、版本历史、七份登记表、结果图、证据及选定源码；本仓库已设为私有。 |
| **仍未解决** | 跨真实场景的稳定恢复优势、物理多步在未见来源上稳定优于自身第一步、UniDB 退化的唯一原因均未确立。部分旧 A100 资产当前未核实；数据和大权重需按清单另行转交。 |

最终状态依据：[主交接说明](handoff/documents/MAIN_HANDOFF_FINAL_v2.md) · [UniDB 实验收尾与验收](handoff/experiment/README_ZH.md) · [模型角色与身份](handoff/provenance/UNIDB_FINAL_HANDOFF_STATE.json)。历史文件中的 `RUNNING`、`pending`、计划步数只描述当时，不能当作现在仍在运行。

## 2. 从最初到现在，做了哪些尝试

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

## 3. 最后一次实验，结果到底怎样

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

## 4. 接手的人从哪里开始

1. **先了解结论、转向与限制：**先读[决策／失败／未决补充](docs/DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md)，再读11页[主交接PDF](handoff/documents/Turbulence_Project_Handoff_FINAL.pdf)，或[可编辑Word](handoff/documents/Turbulence_Project_Handoff_FINAL.docx) / [Markdown同稿](handoff/documents/MAIN_HANDOFF_FINAL_v2.md)。
2. **再定位具体实验：**按[交接资料入口](handoff/README.md)、下方注册表和[操作说明](handoff/history/OPERATIONS_GUIDE_ZH.md)核对版本、评价口径、源码和模型身份。
3. **需要复现时再转交大资产：**数据、完整权重和全量预测没有进入Git；按[未包含资产清单](handoff/NOT_INCLUDED.md)与权重/资产登记取得对应文件。旧A100路径未核实项保留为待查，不能推断文件丢失。
4. **如决定继续研究：**由新负责人先确定目标和新实验范围。当前没有排队中的追加训练；旧控制器和启动脚本仅作历史源码，不应直接运行。

本仓库从2026-11-04交接快照开始Git版本管理，旧实验历史保存在报告与登记表中。它保存已验收的小文件资料和选定源码，不是全部实验磁盘镜像；源码身份、依赖、第三方许可及离线校验方式见下方。

<details>
<summary><strong>展开详细目录：七份注册表、各版本源码、克隆校验与复现条件</strong></summary>

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

取得该私有仓库的访问权限后，克隆并使用 Python 3 标准库运行：

```sh
git clone https://github.com/mayday4yl/turbulence-project-handoff.git
cd turbulence-project-handoff
python3 tools/verify_repository.py
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

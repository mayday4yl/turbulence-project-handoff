# 湍流图像恢复项目交接

这是全项目的私有研究交接仓库：保存已验收的交接文档、结果证据和按白名单选取的源码、配置。覆盖 v1—v3、V4 Noise Bridge、V5 Concertormer、V6 DiffUIR-L、UniDB 及重要支线。**它不是全部实验磁盘的镜像，也不是克隆后即可训练的独立软件包。**

本仓库从 2026-11-04 的交接快照开始版本管理；旧实验时间线保留在报告和清单中，不代表恢复了历史 Git 提交。

## 从这里开始

- **主交接说明，11页：**[可编辑 Word](handoff/documents/Turbulence_Project_Handoff_FINAL.docx) · [PDF](handoff/documents/Turbulence_Project_Handoff_FINAL.pdf) · [Markdown 同稿](handoff/documents/MAIN_HANDOFF_FINAL_v2.md)。
- [交接资料入口](handoff/README.md)：版本历史、最终报告、证据副本与已验收小文件包。
- [操作与复核说明](handoff/history/OPERATIONS_GUIDE_ZH.md)：如何按具体实验查结果、状态和资产。
- [本仓库源码清单及身份记录](provenance/SOURCE_SNAPSHOT_PLAN.json) · [第三方来源与许可边界](THIRD_PARTY_NOTICES.md)。

截至 2026-11-04 的交接记录，最新 UniDB 双路线已在北京时间 00:00 截止边界安全停训，最终评价、完整状态、保存预测 CPU 复核和自有进程退出审计已完成。原路线 **best＝latest U67180**；改进路线 **best U10000，latest U40848**。四个角色对应三个唯一完整 checkpoint，已在原本机资产位置备份验 SHA，**这些权重未放入本 Git 仓库**。实际角色和完整身份见[最终状态](handoff/provenance/UNIDB_FINAL_HANDOFF_STATE.json)及[权重登记](handoff/registers/WEIGHT_REGISTER.csv)。这些是交付时的证据，不是服务器的实时状态。

科研结论仍有限：最新原路线和改进路线的最终五档 PSNR 均低于各自 Input；全项目尚未建立跨真实场景稳定的强湍流恢复优势。工程验收通过不等于恢复目标达成。完整指标、有限正结果和失败边界请读主文档，勿用单张图或一个 PASS 代替结论。

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

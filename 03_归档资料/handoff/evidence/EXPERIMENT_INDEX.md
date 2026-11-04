# 湍流项目版本与任务索引

生效日期：2026-09-21（北京时间）。依据：用户在本任务中明确指定的新命名。

## 版本名称

| 后续统一称呼 | 历史名称 | 定位 | 现有目录（保留原名） |
|---|---|---|---|
| **v1** | V0.1 / v01 | 最初的物理桥接实验 | `physical_bridge/`、`configs/`、`scripts/`；归档 `reports/bridge_v01_retrospective/` |
| **v1.5** | V0.2 / v02 | 用户自行修改的物理约束与推理方案 | `v02_src/`、`v02_new_protocol_20260829/`、`analysis/v02_a1_20260830/` |
| **v2** | V0.3 / v03（老师的递归方案） | 基于预测状态递归展开的常规训练方案 | `v03_recursive_from_scratch_20260915/` |
| **v2.5** | V0.3-half / v03-half | v2 的半量数据训练分支 | `v03_half_recursive_20260915/` |
| **v3** | V0.31 / V0.3.1 / v031 | 100-source Overfit100 递归诊断及后续实验 | `v031_overfit100_g4321_n4/`、`v031_followup_20260918/` |

本次修改的是展示名称和项目管理约定。已有目录、代码标识、配置字段、数据集 ID、服务器路径、checkpoint 文件名、清单哈希和历史证据保持原样；需要查找或执行时仍使用实际旧路径。引用历史原文时可写“v3（原 V0.31）”。

### 避免混淆

- **v2 是常规训练方向，不是 Overfit100 的别名。** 全数据正式训练未启动即因数据协议变更终止，转向 v2.5。依据：`v03_recursive_from_scratch_20260915/artifacts/DATASET_PROTOCOL_CHANGE.json`。
- **v2.5 的实际半量是 source 层抽样。** 训练 1,000 sources × 4 trajectories；验证 100 × 2；测试 250 × 2。早期“保留 2,000 sources、轨迹减半”的建议不是最终执行配置。2026-09-16 交接记录为 update 204 暂停，这是历史快照，不是服务器实时状态。依据：`v03_half_recursive_20260915/docs/NOON_HANDOFF.md`。
- **v3 不仅是数据量缩小。** 它还固定了 g=4/3/2/1、完整尾路径、预测次数计数与独立推理配置。数据为 100 sources、400 trajectories、1,600 states / starts；其 overfit 结果与后续 heldout 诊断应分别陈述。
- **v3 的数据目录仍含 v03。** `v03_overfit100_g4321_n4/` 是 v3 使用的数据构建项目，不能仅凭名字归入 v2。服务器权威数据路径保持为 `/data2/yanglin/v03_overfit100_g4321_n4/data/places_p2s_bridge_v03_overfit100_g4321_n4/`。
- **历史几何校正支线也曾叫 v0.3，但不属于本次老师递归版 v2。** `analysis/v03_validation_20260909/` 后续称“几何校正支线（历史 v0.3）”，不强行编号；其 Gate 1 结果也不能套到 v2/v2.5/v3 上。
- 旧演示图中的“v1/v2”可能是当时的局部对照标签，必须结合图注和所用 checkpoint 判断，不能直接套本表。项目外单独名为 `v2` 的目录或其他项目也不在本次范围内。
- 方法有效性、数据验收、训练完成和泛化是不同结论；改名不改变证据边界。实时进度以最新日志和现场检查为准。

## 任务分类与标题

此次按 Codex 任务列表和实际对话内容整理，范围为 `/Users/lucas/Desktop/turbulence IR` 项目内已找到的 11 个对话。一个对话可以含多个版本的讨论，标题标识主要用途，不重写历史内容。

界面整理修正（2026-09-21）：最初建立的三个侧栏分区让对话从原项目下隐藏，给用户造成查找困难，现已撤销。11 个对话恢复在原 `turbulence IR` 项目下显示，保留新版标题。下表的分类仅用于索引，不代表独立侧栏分区；后续默认保持对话在原项目下，不再移出。

| 内容分类（非侧栏分区） | 原标题 | 新标题 | 主要内容 |
|---|---|---|---|
| 湍流｜v3 当前实验 | 100张数据集 | **v3｜Overfit100 数据构建与版本管理** | 数据构建、交付及本次版本总表 |
| 湍流｜v3 当前实验 | 创建 v0.3.1 版本 | **v3｜递归训练与 refinement 对照** | 实现验收、正式训练、逐步评估及 refinement 对照 |
| 湍流｜v3 当前实验 | v0.31-2实验+1推理 | **v3｜T1/T2 对照与 I1 推理诊断** | 后续两项训练对照、一项推理诊断和泛化分析 |
| 湍流｜v3 当前实验 | 整理 V0.31 研究复盘材料 | **v3｜研究复盘与证据打包** | 汇总已有配置、源码、日志、评估与可视化证据 |
| 湍流｜历史版本与支线 | 湍流修复v0.1 | **v1｜初版实验与方法讨论** | 初版项目及跨版本方法、网络和资源讨论 |
| 湍流｜历史版本与支线 | 湍流修复v0.2 | **v1.5｜物理约束改进与结果复盘** | 自主改进方案、A1 实验及与 v1 的对照 |
| 湍流｜历史版本与支线 | 湍流修复v0.3 | **v2 / v2.5｜递归方案与半量训练** | 老师的递归方案、半量训练；另含 v3 数据独立验收 |
| 湍流｜历史版本与支线 | 检查NPU访问状态 | **支线｜几何校正实验与失败复盘** | 历史几何校正路线；曾使用 v0.3 名称，但不是老师递归版 |
| 湍流｜汇报材料 | 规划PPT三部分结构 | **材料｜算力、平台与团队 PPT 规划** | 三部分结构规划与本人可用算力图表 |
| 湍流｜汇报材料 | 拆解图片为可编辑PPT | **材料｜算力与团队图示重建** | 算力资源和团队协作图的可编辑重建 |
| 湍流｜汇报材料 | 拆解为可编辑PPT | **材料｜实验保障可编辑 PPT** | 单页实验保障原生可编辑 PPT |

### 任务 ID 索引

- v3｜Overfit100 数据构建与版本管理：`01a0a9ce-d0f3-7bf1-be34-a9c54d229426`
- v3｜递归训练与 refinement 对照：`01a0aab1-bd26-7530-a5f4-c50f027cb391`
- v3｜T1/T2 对照与 I1 推理诊断：`01a0b54e-a4ed-7e11-ab66-79016674c429`
- v3｜研究复盘与证据打包：`01a0af62-2d61-7a90-89cf-82ef43f77896`
- v1｜初版实验与方法讨论：`01a03ea1-9c4a-7793-a4b9-227cb5f08936`
- v1.5｜物理约束改进与结果复盘：`01a04806-8eaa-72b3-ada6-5174e93b7e7d`
- v2 / v2.5｜递归方案与半量训练：`01a0a47c-1cb4-7161-bf0a-60419a152cce`
- 支线｜几何校正实验与失败复盘：`01a07cc3-ab0a-7b02-9d2d-f71908f675a2`
- 材料｜算力、平台与团队 PPT 规划：`01a075a9-4b93-7c62-92ae-5fe7bc995b2f`
- 材料｜算力与团队图示重建：`01a075b0-145e-7190-8485-e7f434d80336`
- 材料｜实验保障可编辑 PPT：`01a075ad-e756-7b51-b6ba-d4c059b68fe3`

## 后续使用

1. 新报告、解释与任务标题使用本表的新名称；第一次提到旧证据可并列历史名称。
2. 每次继续实验先确定具体版本、数据、checkpoint 和任务范围；不要从标题推断训练是否启动或完成。
3. 现有 v3 任务分别负责数据、训练评估、后续诊断、证据整理。需要跨版本对照时，在同一段或表格中明确版本。
4. 共享入口为根目录 `AGENTS.md` 与本文件。既有窗口通过一条仅同步命名的消息读取；新窗口在这个项目工作时先读取该入口。
5. 本次整理不新增训练、推理、数据生成或远端操作授权，不自动恢复已经暂停的实验。

## V4 候选资产（2026-10-02）

- [D²Turb-Restormer 四方法候选](<v4_restormer_candidate_20261002/README.md>)：以作者公开的 Restormer 结构为权威，从零共同初始化，适配 D0/R0/R08/L08。骨干及四方法有限正确性检查通过；正式训练未启动，预算/学习率未指定。R0 的 CUDA 反传依赖验收用 CPU 激活暂存，R08/L08 的真实尺寸验收在 CPU 完成；训练显存及批量方案尚待验证。
- [A100 磁盘审计](<../NOT_INCLUDED.md#ref-6920a75f7b8a50c2cee5>)：清理前的只读候选清单与扩数据容量估算；账号额度由用户声明为 2T，剩余额度未核实。
- [A100 已授权清理记录](<storage_archives/A100_cleanup_20261002/CLEANUP_RECEIPT.md>)（北京时间 2026-10-03 完成）：安装缓存及 93 个已本机归档、SHA256 核验通过的临时/旧中间权重已删除，项目占用由 287.92 GiB 降至 242.15 GiB，释放 45.77 GiB。正式权重、数据、环境及指定保留节点的元数据核查通过；本机归档约 40.29 GiB，NPU 未修改。

## V4 Train2000 NPU 长训练（2026-10-03）

- [独立实验入口](<v4_restormer_train2000_npu_20k_20261003/README.md>)、[执行记录](<../NOT_INCLUDED.md#ref-9083982bdde0fd4cdf1a>)、[NPU 兼容说明](<../NOT_INCLUDED.md#ref-60e9d13f51cdb07aee35>)。新 canonical 数据 ID 为 `places_p2s_canonical_20261003_c50_s2000_v250_t200_g4321_v1`；Places 50 类、2000/250/200 source-disjoint clean，验证 same-category / unseen-image generalization。不能与历史 NPU train2000 或 A100 train500 数据混用。
- 本阶段授权：de36 NPU 0/1/2/3 分别执行 Restormer D0/R0/R08/L08，各 effective batch 32、20,000 updates；A100 只准备镜像与未来 CUDA 资产，禁止本阶段正式训练。运行状态以远端 `reports/WORKFLOW_STATUS.json`、各方法 `runs/formal_20k/<method>/STATUS.json` 和完整 checkpoint 为证据；本索引不是完成声明。

### V4 Train2000 最新平台授权更新（2026-10-03）

用户最新明确要求：停止 NPU，A100 从共同随机 U0 重新训练，**不迁移 NPU checkpoint**。该要求覆盖上文 NPU-only / A100-assets-only 阶段限定。NPU 四方法已安全停止并保留结果；A100 物理3/5两卡四方法已经正式启动，新 run 为 `runs/a100_fresh_u0_20k`，预算每方法20k、effectiveB32及全部科学定义不变。见 [最新授权](<../NOT_INCLUDED.md#ref-5257a9a5a70034bde354>) 和 [U0初始化核验](<../NOT_INCLUDED.md#ref-ccf14b9b2dc6db766773>)。A100执行父目录的 `a100_resume_20261003` 仅是历史准备名称，不表示从NPU续训。

## V4 NAFNet-SIDD 116M 候选资产（2026-10-05/06）

- [独立候选入口](<v4_nafnet_sidd116m_candidate_20261005/README.md>)、[四方法适配说明](<v4_nafnet_sidd116m_candidate_20261005/ADAPTATION.md>)。用户指定官方 NAFNet-SIDD width64（115,982,915 参数）；V4 六通道、[g,t] 条件及 clean/raw ξ 双头适配后为 117,767,209 参数。复用固定的 D0/R0/R08/L08 科学代码。用户随后明确要求预训练模型，当前默认改为官方 SIDD 预训练主体的共同 V4 U0；六通道输入扩展，新 V4 条件与 clean/raw ξ 头初始化单独记录，完整官方去噪 checkpoint 同时保留。
- 当前范围是资产与有界验收，不是正式 NAFNet 训练；Test 未用于验收。既有 Restormer fresh-U0 run 继续运行，NPU 保持停止。验收证据、后续需要决定的数据/预算与 batch 事项见候选目录；不能将候选验收值视作正式模型效果。

### V4 NAFNet Train8000 最新执行授权（2026-10-06）

- [独立新实验](<v4_nafnet_sidd116m_train8000_20261006/README.md>)、[科学协议](<v4_nafnet_sidd116m_train8000_20261006/protocol/SCIENTIFIC_SPEC.json>)、[无人值守交接](<../NOT_INCLUDED.md#ref-d445b002de9a66ce8876>)。用户已授权官方SIDD预训练主体的全参数微调，以及新Train8000/Val1000的构建和必要验收后自动开训。此授权覆盖前述候选-only范围；并把旧Restormer的主比较/收尾点改为matched U5000，已有U>5000结果保留。
- 新数据为Places4000+LSDIR4000训练、Places500+LSDIR500验证，旧Test200锁定。source合并去重/冻结已完成；真实状态以新目录报告为准，source READY不能视作退化数据或正式训练 READY。
- 新实验只有A100物理3/5可分配，账号总占用不得超过两卡；de36 NPU0–3仅允许在空闲时生成唯一canonical新数据。正式首阶段各640000 start exposures，B/LR/precision待规定的实际GPU gates冻结；不自动延长。所有历史目录、checkpoint及科学资产不改名。

## V5-v1 Simulator 候选（2026-10-20）

- [独立入口](<v5_v1_simulator_20261020/README.md>)、[Method Spec](<v5_v1_simulator_20261020/docs/METHOD_SPEC_ZH.md>)、[局部检查报告](<../NOT_INCLUDED.md#ref-e67085f389653af3e219>)。V5 从 V4-R0 演进，须区分新旧时间语义、三通道输入、Simulator 和数据；不覆盖旧目录或把旧 V4 结果当作 V5 结果。
- 当前授权为 de36 独立目录 `/data2/yanglin/v5_v1_simulator_20261020/` 内实现 Smooth、Tilt/Warp、FFT 并进行数值检查。2026-10-21 用户进一步授权最终 Simulator Gate，暂定采用旧 P2S 完整复数 IFFT，保留 raw-ξ、分布及物理参数，非平稳协方差作为局限记录。用户允许使用任意实时空闲 NPU，最多同时四张，此项覆盖本任务此前的历史卡号限制。
- 严格训练内部时间 `0<t1<t2<t3<1` 已实现，端点、重复及舍入退化整组三点重采样；原推理默认 `[0,.25,.5,.75,1]`，NFE=4。[原冻结 Gate](<../NOT_INCLUDED.md#ref-cea8c9c2e478f8c22aef>) 未全通过的记录保留。2026-10-21 用户提出接收后，另行登记[保留局限的研究接收决定](<v5_v1_simulator_20261020/acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md>)：Simulator 不再阻塞 V5 下一阶段，原近零与差分 FAIL、物理定义及阈值不改写。未启动正式数据生成、恢复模型训练或效果实验，未新增物理采样方法；研究接收不等于启动这些作业。
- 2026-10-21 [起点与通用时间调度接口](<../NOT_INCLUDED.md#ref-0ff08de97d5870bd3a4e>)：网络只接 `(Z_RGB3,t)`；训练实际 fixed G4/NFE4，uniform仅显式预留且需真实对应YG；推理支持N/gamma及合法非均匀网格，全局t=1−g/4。Simulator与原冻结时间模块未改。
- 同日完成 [BioIR / Concertormer / AdaIR 候选比较](<v5_backbone_review_20261021/BACKBONE_COMPARISON_ZH.md>)。推荐属于选型建议；Backbone待用户确认。未下载大权重、未进行Backbone NPU测试、未实现完整训练链或启动训练。

## V5-v1 Concertormer 首轮 10k 实验（2026-10-21 后续授权）

> 最新资源规划（同日后续）：用户明确 de36 的 2–7 号六张 NPU 属于本项目，排除 0、1；接受 effective batch 36，并明确提出 microbatch 4 × 累积2 × 六卡 = batch48、增加样本呈现。当前首选规划为三组统一 B48，详见[六卡训练规划](<../NOT_INCLUDED.md#ref-c1c54e7430c19742187a>)。本轮仅规划；未启动测速或正式训练、未终止现有占卡任务，原开训确认限制仍有效。下文四卡设置与测速作为历史依据保留。

- 用户已确认 **Concertormer full + 官方 GoPro 初始化**，并授权完整训练链、新数据生成、必要自动预检通过后直接启动 D0 / R0 / R0-D0Init，每组最多 10,000 optimizer updates。这覆盖上文历史候选阶段的停止边界。入口见 [实验说明](<../NOT_INCLUDED.md#ref-0178ac8e2679e51869c7>) 和 [本轮协议](<../NOT_INCLUDED.md#ref-b0b36fc62b8a40fb941a>)。
- 同日后续用户要求**选好训练 setting 后先确认，不直接训练**，覆盖上一条历史自动启动授权。正式启动需用户确认最终设置；有界测速不计入正式训练预算。
- 后续平台对照交给既有窗口 **V5｜A100 Simulator 验收与 NPU 对照**（`01a15d25-ce27-7423-bba3-321d09166f86`）：先准备同训练链、报告所需资源；账号最多两张A100，等用户确认资源后才开始GPU测速，正式训练仍禁止。原V5窗口继续负责NPU。见[本次交接](<../NOT_INCLUDED.md#ref-dd3ad938e3e3415c2511>)。
- clean 使用官方 RGB 残差恢复，完整加载官方恢复主体与 RGB 层；不沿用 V4 sigmoid clean，不加入官方多尺度辅助损失。新增连续时间条件和原生 raw-ξ head，D0/R0 共用完整适配 U0。Simulator FP32、物理递归及 Full-BPTT 保持。
- 新数据 clean 2000/250/250，各 4 个独立 ξ，全部重新生成 V5 Y4。每 500 updates 完整 Val1000，refinement OFF；每 1000 保存完整 checkpoint，另有 best/latest。测试集在最终选择前锁定。
- de36 独立根目录 `/data2/yanglin/v5_concertormer_10k_20261021`，最多使用四张实时空闲 NPU。官方权重与新数据已验收；优化配置完整四步、四卡有效B32及checkpoint检查通过，D0/R0分别21.17/87.77秒每更新。见[吞吐报告](<../NOT_INCLUDED.md#ref-41249ebede3beeb5a081>)与[待确认设置](<../NOT_INCLUDED.md#ref-ac2e5e017effea26add5>)。正式训练仍未启动，等待用户确认，未自行迁移A100；训练状态以实际run日志和完整checkpoint为准，本索引不作完成声明。
- 同日后续用户授权继续定位大卷积反向并做等价 FP32 优化，仍禁止正式训练。[独立优化结果](<v5_concertormer_10k_20261021/optimizations/psf_adjoint_20261021/REPORT_ZH.md>)：保持原前向和反射边界，将固定 PSF 核输入梯度改用等价普通卷积；CPU64、NPU FP32、完整四步梯度及四卡检查通过。同日四卡 B32 对照 D0 21.45→9.60 秒、R0 87.28→39.09 秒，约 2.23 倍吞吐；三组各10k纯训练外推约22.69→10.16天。候选只在独立入口启用，原60个冻结源码文件及U0不变，尚未替换正式默认或获得开训确认。

### V5 最新协议修订：Clean-only D0（2026-10-21）

用户已明确恢复并要求直接推进：D0为无时间、无raw-ξ、无Simulator的纯RGB恢复，Clean MAE权重1；D0最多10k后，以最终best初始化新时间/raw头的R0-D0Init最多10k。共同U0直接R0暂缓。六卡B48、日志10、完整Val1000及checkpoint250、监控每小时。以上覆盖旧双输出D0、三组交替和暂停条款；实况见[当前执行记录](<v5_concertormer_10k_20261021/reports/CURRENT_EXECUTION_ZH.md>)。

### V5 最新后续协议：A 6k → B 4k（2026-10-22）

用户[最终执行指令](<../NOT_INCLUDED.md#ref-d8664604f6a4a3437a80>)要求保护正在运行的Clean-only D0，随后从其最终best进行A阶段6000更新的预生成物理中间态单步训练，再从A best进行B阶段4000更新的真实四步Full-BPTT Alignment。A按当前g/Yg做observation matching，B按G4/Y4；A/B均在原Val1000真实四步推理上选择best，另建固定Val-A作单步诊断。原后继10k R0-D0Init被替代，直接R0仍暂缓；新run为 `runs/alignment_A6k_B4k_20261022/{A,B}`。自动完成准备、必要检查、训练及结果汇总，阶段交接不等待审批，不增加预算；实时状态仍以[当前执行记录](<v5_concertormer_10k_20261021/reports/CURRENT_EXECUTION_ZH.md>)引用的远端日志为准。

## ResShift 独立诊断入口（2026-10-26）

- [Stage1 官方基线审查](<v5_resshift_stage1_20261026/reports/FINAL_REPORT_ZH.md>)：固定 journal 提交及官方 Deblur/VQ 权重；官方图与少量 Val 湍流图的原版推理。Zero-shot 结果不代表湍流训练性能。
- [反传可行性与 VQ 表示审查](<../NOT_INCLUDED.md#ref-eb340f5374615a721294>)：FP32 四步及固定真实ξ的两阶段物理反传通过；FP16完整递归仍有 NPU 算子阻塞。严格官方VQ路径覆盖全部Val250 clean和已有弱状态50张。无optimizer更新、正式训练或Test1000评价；旧V5 D0/A/B资产保留。此处A/B诊断不得与旧V5训练阶段A/B混淆。
- [四次VQ编解码漂移](<../NOT_INCLUDED.md#ref-7149dd84fc13c5d7f387>)：同一Val250、FP32连续4次官方VQ往返，平均PSNR 28.73→24.95dB、MAE满量程2.55%→3.99%，250图PSNR均逐轮下降。新增证据支持暂缓原样采用“冻结VQ＋四次RGB物理递归”组合，先讨论表示/往返方案；单次恢复基线可保留。此为纯codec诊断，不是实际恢复性能或上限；无训练、Simulator或Test图像评价。

## DiffUIR-L 官方审查与短 D0 Pilot（2026-10-26 后续授权，已完成）

- 用户明确将下一版 V5 Clean Restoration 组件改为 **DiffUIR-L（CVPR 2024）**，不继续 ResShift 正式训练；上述三份 ResShift 成果保留。[原地归档回执](<v5_diffuir_l_stage1_20261026/archive/ARCHIVE_RECEIPT_ZH.md>)记录7,321文件完整SHA清单和二次核验，不覆盖旧代码、权重或结果。此决定覆盖旧报告中的后续ResShift训练建议。
- 新独立目录[入口](<../NOT_INCLUDED.md#ref-bcd85a1613352b59190f>)、[最终报告](<v5_diffuir_l_stage1_20261026/reports/FINAL_REPORT_ZH.md>)。官方commit `c71437ce86ad1b098aa9be4678dc82e8b99649df`，Large64-1248、36,261,187参数；官方model-300 raw/EMA严格匹配。公开来源支持含GoPro的五任务通用预训练，不称GoPro-only。
- de36独立根`/data2/yanglin/v5_diffuir_l_stage1_20261026`，910B3物理NPU2完成FP32真实三步推理、原生Loss一次真实Adam更新及完整三步输入/模型反传；原固定Val4的官方EMA zero-shot为17.5407dB，输入17.7892dB，不构成完整泛化结论。
- 三项gate通过后，按[冻结配置](<../NOT_INCLUDED.md#ref-60f3a9b73c92d1888367>)完成100更新D0 Pilot，g=4、Y4→C、官方raw初始化、原生残差L1，未新增ξ头、物理时间或Condition。实际200个realization/91source/49类；Val50 EMA从U0 18.73989到bestU25 18.74114、U100 18.48104dB，技术可行但未证实稳定收益。EMA处于默认warmup，不能解释为成熟滑动平均效果。
- U0/U25/U50/U100完整checkpoint保存在远端新目录，最终CPU读回通过；本地保存日志、评估和哈希清单。旧V5 107受保护文件、32旧checkpoint元信息及600个选用数据文件核验通过，NPU2已释放。没有Test1000、AMP、多卡、外层四步物理B或大规模训练；正式D0须另行冻结预算与评估协议。

## DiffUIR-L Batch / 显存 / 吞吐工程 Benchmark（2026-10-26，已完成）

- [独立入口](<../NOT_INCLUDED.md#ref-60f667a68e41aa4f6498>)、[完整中文报告](<v5_diffuir_l_benchmark_20261026/BENCHMARK_REPORT_ZH.md>)、[机器结果](<v5_diffuir_l_benchmark_20261026/RESULTS.json>)。仅使用已验收FP32 Large和原Train8000，用户后续授权多空闲卡并行；2–7号卡容量筛查后，在NPU2串行复测主要档位。
- 原生真实forward/backward/Adam/EMA：已测最大稳定及最高吞吐microbatch16，NPU2约0.7095秒/update、22.5505张/秒，allocator allocated/reserved约31.95/38.15GiB，设备HBM采样峰值41.62GiB。推荐microbatch16、accumulation1–2；只有accum1实测，未开始正式训练。
- 完整三步采样反传：已测最大batch8，约1.0407秒/update、7.6872张/秒，HBM采样峰值53.28GiB；所有阶段、输入及参数梯度检查通过。与同卡同B8原生训练相比耗时2.67倍。此为末端Clean L1诊断，未加入D0训练目标，未实现外层物理A/B。
- 12次运行PASS，6次首次环境启动FAIL（CANN路径覆盖，0更新，修复后另目录重测），0个OOM；原生B32与采样B16按安全预算未测试，不能称已找到硬件极限。旧9,552项资产记录保护对比PASS，所有测试模型副本丢弃，NPU2–7已无相关进程，无Test1000、AMP或gradient checkpointing。

## V6 DiffUIR-L 方法审查与独立执行入口（2026-10-27）

- 当前方法入口为用户提供的 `V6-method-v1.0`；旧 Concertormer 方法仍叫 V5。历史 DiffUIR Stage1/Benchmark 路径保留 `v5` 原名，不能当作 V6 联合训练结果。
- [方法理解报告](<v6_diffuir_l_method_review_20261027/METHOD_REVIEW_ZH.md>)、[Spec→Code](<v6_diffuir_l_method_review_20261027/SPEC_TO_CODE_MAPPING_ZH.md>)、[问题与验收清单](<../NOT_INCLUDED.md#ref-109df4e557bbbb663df3>)。完整440行规范原件已保留，34个选定源码/结果文件本地与de36 SHA一致；审查本身未运行模型、训练或新数据生成。
- 用户随后提供 Training Handoff 并正式授权 G0–G7 自主执行，独立根 `v6_diffuir_l_20261027/`。最新补丁取消基于恢复效果的 conditional-go/no-go；保留方法、数据、梯度、资源和checkpoint硬门槛。预算 D0 10000 / A 6000 / B 4000；A best按固定Val1000固定四步最终PSNR。普通工程问题自主修复，需改方法定义的问题停止受影响部分。此条是授权记录，不是Gate通过或启动训练的声明。

## V6 五卡恢复当前入口（2026-10-27）

- 最新授权为恢复训练，仅NPU3–7；D0已完成，A从原六卡U2410完整状态继续至总6000，后接B10000。新入口v6_diffuir_l_20261027/execution_20261027_five_npu/，原six-card目录及STOP为历史保留。
- 五卡A effective40、B10；五卡真实恢复与正式启动验收PASS。状态须查新STATUS/进程/日志，不把本条当实时进度。见FIVE_NPU_STARTUP_REPORT_ZH.md及authority/USER_RESUME_FIVE_NPU_AUTHORIZED_20261027.md。

### V6 B前置并行补丁（2026-10-28）

- 当前CPU调度入口升级scripts/five_npu_pipeline_parallel.py，以execution_20261027_five_npu/ACTIVE_CONTROLLER.json和PIPELINE_LAUNCH02为准。原控制器正常交接退出，A worker未重启。B实际best后并行单卡全图、四卡初始梯度与CPU核验；完整五卡训练型检查依赖仍保留。见parallel_preflight/B_PARALLEL_PREFLIGHT_REPORT_ZH.md。此为编排接入，不表示B科学Gate已执行。

### V6 A完成与B工程修复当前入口（2026-10-28）

- A累计6000正式更新已完成，终点与best完整保留，详见v6_diffuir_l_20261027/reports/A_COMPLETION_REPORT_ZH.md。B初始分支诊断的JSON文件读取错误已保存现场，新增v2仅修解析；原56文件未变，新冻结59文件。当前CPU入口scripts/five_npu_pipeline_parallel_v2.py，以ACTIVE_CONTROLLER和PIPELINE_LAUNCH03为准；旧并行控制器退出不可重启。详见execution_20261027_five_npu/engineering_repair_20261028_0310/REPAIR_REPORT_ZH.md，实时Gate仍查远端，不将本索引当作正式B通过声明。

### V6 B正式训练启动（2026-10-28）

- B实际A-best的五卡全图Benchmark、独立恢复、Pilot100与同15例初终Full-BPTT梯度门槛已真实PASS，正式B_five_npu_formal10000_01已启动。方法/预算/FP32不变，正式U0从A best EMA重新初始化，未继承Pilot。启动验收及首个U250完整状态已独立核验；见v6_diffuir_l_20261027/reports/B_PREFLIGHT_PILOT_AND_STARTUP_REPORT_ZH.md。本条是启动验收，不能当作B已完成10000更新；实时查远端STATUS/日志。

### V6 四卡D0限时续训（2026-10-28）

- 用户最新要求暂停B、继续D0至2026-10-29中午12点（Asia/Shanghai），仅4张卡。B已在U9500安全保存并停止，不自动恢复A/B。新执行入口 [execution_20261028_d0_until_noon](<../NOT_INCLUDED.md#ref-8c6ee178828c830f73d5>)，从原D0 U13334完整状态续训，NPU4–7，micro5×4×acc3=60，原Loss/FP32/LR2e-6不变。先真实四卡恢复验收后正式续训；本索引不能代替实时Gate/日志证据。新旧STOP及原A/B/D0资产全部保留。

- 2026-10-28用户随后明确要求暂停以修改：D0已安全停U14927、完整状态及SHA/CPU独立读回PASS，四worker与控制器退出，自动跟进暂停。此前明日中午执行计划已被覆盖；未经新授权不得恢复D0/A/B，见V6当前执行策略和USER_PAUSE_REPORT_ZH.md。

### V6 D0 U14927 切换官方单步残差续训（2026-10-28）

- 用户最新明确授权覆盖暂停，独立 [execution_20261028_d0_official_residual](<../NOT_INCLUDED.md#ref-9b26e6fe749606fcb051>)。从U14927完整状态接续，仅训练改为原生随机τ一次U-Net raw残差L1，删除0.5和三步Clean训练分支，推理仍原三步。四卡4–7、effective60、FP32、恒定LR2e-6不变，正常至少至2026-10-29 12:00，再保存/完整Val1000。父EMA全Val1000起点PSNR21.06020173，门槛进度看实时Gate；该实验不是从官方权重开始的纯原生对照，旧资产及STOP保留，A/B不启动。

- 原生残差微批补充短测已完成：四卡micro5×acc3=60约82.13图/秒，micro15×acc1=60约79.63图/秒，无收益保留5×acc3。从正式U16701完整接续，新入口execution_20261028_d0_microopt，最终汇总两原生正式段，旧资产保留。详见MICROBATCH_BENCHMARK_REPORT_ZH.md与FORMAL_RESUME_ACCEPTANCE，实时进度查新STATUS。

## TurbNet 官方预训练 Zero-shot 核查（2026-10-28）

- 独立[入口](<../NOT_INCLUDED.md#ref-297972462e31b76ac2dc>)与[阻碍报告](<turbnet_zeroshot_20261028/reports/FINAL_REPORT_ZH.md>)：官方源码固定 `4ec7239b47328d286a834d50636411cf82741ee8`。仓库和Releases没有权重，官方Drive仅列出空的PiRN Models子目录；配置要求的 `turb_current49.pth` 未取得。按用户要求停止于权重可用性门槛，未验证预训练严格加载、Smoke或Val1000，没有TurbNet性能结论。
- 本轮只读核查D0/NPU4–7及旧U14927 checkpoint/已有Val1000哈希，没有新NPU进程、训练、微调或其他模型替代；没有修改D0运行环境、配置和checkpoint。共卡减速与TurbNet耗时均未实测，不将显存余量当作无干扰证据。机器状态及原始来源响应存于独立目录。

## TurbWNet 官方权重验收与共卡中止（2026-10-28/29）

- 独立[入口](<../NOT_INCLUDED.md#ref-748ebe083da21916c8d3>)、[停止报告](<turbwnet_zeroshot_20261028/reports/FINAL_REPORT_ZH.md>)。用户从官方百度网盘下载三个真实权重，大小/SHA/CRC及完整strict=True加载通过，官方commit `05627b725241c28a09b03c00faefafa861558ebf`；完整模型32,430,816参数、709张量，含全部恢复阶段。1张CPU技术前向通过，NPU图像前向为0，Val1000未执行，没有TurbWNet质量结果。
- **本次NPU7共卡尝试触发D0现有排他守护，导致D0训练中断。** foreign PID3206173与TurbWNet smoke02一致；00:09:40–41触发SIGTERM和rank0 DataLoader失败，详见[事件报告](<../NOT_INCLUDED.md#ref-b146989b01c8ff86c5f6>)。最后完整U30500 checkpoint SHA/CPU读回通过，U30920停机保存未完成；00:14只读检查仍有3个残留rank。本任务已停止且封锁后续推理，未擅自清理/重启D0或修改其环境/守护；本条不能当作D0已恢复的证据，需查现场。


## V6 D0 共卡保护中断后的工程恢复（2026-10-29）

- D0于00:09因独立TurbWNet外来PID触发既有排他守护而中断，U30920未保存。原资产/故障日志保留；从完整U30500在独立run `V6_D0_U14927_residual_recovery_20261029_01` 已实际恢复，原训练入口/config不改，四卡完整状态与前100次重放数值验收PASS，420次未保存更新单列重放，不计作新增有效更新。见[恢复报告](<../NOT_INCLUDED.md#ref-f2cd3d865cad9df2f8c0>)。新CPU与RANK身份、实时进度查ACTIVE_CONTROLLER和日志；原NPU4–7、effective60、LR2e-6及今日12:00后保存/完整Val1000计划保持，A/B不恢复。此处不是阶段完成声明。

## TurbWNet CPU 10图预训练 Zero-shot（2026-10-29）

- 用户明确选择“现在用CPU测10张”，独立[结果报告](<../NOT_INCLUDED.md#ref-e0ed67bf22eceaba1ee6>)、[逐图CSV](<../NOT_INCLUDED.md#ref-63d229d21b0c2b4b82cd>)。固定10个独立Clean、6类、每源1个G4，官方完整权重strict加载，CPU FP32/batch1/2线程，总耗时495.9秒。TurbWNet PSNR18.177932、SSIM0.377687、LPIPS0.383202、MAE0.082831；同组Input PSNR17.903898、旧D0 U14927 EMA20.177678。PSNR对Input +0.274033 dB（7/10提高），对D0 -1.999747 dB（1/10提高）；LPIPS明显低于D0，不能以锐度代替几何/像素保真。
- 本轮关闭NPU后端自动加载且逐图确认无NPU PID；D0前后四PID不变并继续训练，无D0修改/信号、训练、微调或Test。原NPU停止标记保持，NPU前向仍未验收，未执行Val1000；小样本结果不外推到全量。


## V6 D0 最新时间追加授权（2026-10-29）

用户明确要求“训到下午6点，加训练”，把原生残差D0本轮结束时间延长到2026-10-29 18:00（北京时间）。NPU4–7、micro5×4×acc3=60、LR2e-6及原生残差单步训练/原三步固定Val1000协议保持，A/B继续暂停。独立[追加授权](<../NOT_INCLUDED.md#ref-c00f27ff175b2ec55b6b>)和[执行计划](<../NOT_INCLUDED.md#ref-2b08e40200988cad80a5>)记录六项父状态/截止元数据的允许变化。旧中午run正常保存与完整验证保留，持久CPU734414/start_ticks553077830等待原完整状态验收与进程退出后，从真实latest完整接续新run；等待不等于新阶段已开训或PASS。启动和结束以远端新STATUS、EXTENSION_STARTUP_ACCEPTANCE和FINAL_ACCEPTANCE为准。每小时监控已按18点更新，不能因中午子阶段完成而结束最新任务。


## V6 D0 用户停止与最终Val1000完成（2026-10-29）

用户明确要求现在停止，覆盖原18点计划。D0已安全停U95544、完整checkpoint/CPU全状态读回与SHA通过，独立最终Val1000真实完成且0optimizer更新；latest和原冻结规则best均U95544。PSNR21.16514237、SSIM0.55626089、LPIPS0.49768281；U14927起点为21.06020173/0.54878696/0.52907033。见[最终报告](<v6_diffuir_l_20261027/execution_20261029_d0_user_stop/D0_OFFICIAL_RESIDUAL_FINAL_REPORT_ZH.md>)。全部本阶段训练、评估和CPU控制器退出，NPU4–7已释放；中央STOP保留，D0/A/B禁止自动恢复。自动跟进已删除。新测试集验证未执行，不能当作本次选模证据。


### 2026-10-29：V6 D0 U95544 EMA 的 OTIS / ATD 外部测试

仅原尺寸S3外部推理，0训练更新；OTIS全部5195帧恢复、16序列4328帧固定Mask计分，ATD38段各5帧共190帧定性。OTIS主指标下降：PSNR-M 10.3721→8.5155，SSIM-M11 .3112→.2052，补充LPIPS-valid64 .2745→.3897；工程通过不等于科研成功。固定样本存在细节丢失/伪影，不根据测试改方法或选模，D0/A/B保持停止。入口：[中文报告](<v6_diffuir_l_20261027/external_eval_20261029_otis_atd/OTIS_ATD_EVALUATION_REPORT_ZH.md>)、[OTIS图库](<../NOT_INCLUDED.md#ref-79c9dddee59e29499249>)、[ATD图库](<../NOT_INCLUDED.md#ref-836963d6b9bb1739a3e8>)、[验收](<v6_diffuir_l_20261027/external_eval_20261029_otis_atd/FINAL_ACCEPTANCE.json>)。


### 2026-10-29：V6 D0 U95544 当前 Train / Val Loss

在线Train最近1000更新原生残差L1均值0.0891821093；当前U95544 EMA完整Val1000同目标补测0.0876820027，250 Clean×4轨迹，真实1000次1NFE前向，0训练更新。历史Val残差未记录，仅补测当前点，历史三步Clean MAE终点0.0551166998另画，不能混为同一Loss。D0/A/B保持停止。入口：[曲线与说明](<v6_diffuir_l_20261027/diagnostics/d0_train_val_loss_20261029/TRAIN_VAL_LOSS_ZH.md>)。


### 2026-10-29：现存G1–G4强度数据盘点

只读CPU核验：旧Places Train2000每档G1–G4各8000张训练观测；旧Places4000+LSDIR4000每档各32000张，两个集合有复用不可相加。当前V6 Train10000完整训练缓存仅40000张G4，D0未多强度混训。3个固定Train来源同轨迹15个数组独立SHA/shape/finite核验并原尺寸显示Clean/G1/G2/G3/G4，非模型恢复、不生成新退化、不恢复训练。入口：[现存强度清单](<v6_diffuir_l_20261027/diagnostics/data_strength_inventory_20261029/DATA_STRENGTHS_ZH.md>)。


### 2026-10-29：U95544 EMA同批Val1000的G1–G4评估

每档同一固定Val1000（250 Clean×4 realization），4000次新S3恢复，12000真实NFE，0训练。按当前原物理seed/ξ生成G1–G3，1000条G4像素重放误差0；G4指标及100张旧浮点输出也精确复现。Input→D0 PSNR：G1 22.2418→15.3881，G2 20.3064→16.3748，G3 19.4719→19.0816，G4 18.7959→21.1651。G1/G2三主指标恶化，G3混合，G4改善；不是跨强度恢复成功，不由此单独判定扩散机制失效。固定图片不按质量挑选，D0/A/B均未恢复。入口：[中文报告](<v6_diffuir_l_20261027/diagnostics/d0_u95544_g1_g4_val1000_20261029/G1_G4_VAL1000_REPORT_ZH.md>)、[同批恢复图库](<../NOT_INCLUDED.md#ref-9ae12a074e3899de16db>)、[各强度指标CSV](<../NOT_INCLUDED.md#ref-9ebdbf7fd8904e5b9419>)。


### 2026-10-29：六档物理强度预览（训练保持停止）

32个独立Clean、同ξ的G0.25/G0.5/G1/G2/G3/G4共192张退化与Clean对比，前8个样例与原展示一致；G1–G4像素精确重放，0恢复推理/0训练更新。入口：[图库](<../NOT_INCLUDED.md#ref-8436e0f2c2e379427268>)、[说明](<v6_diffuir_l_20261027/diagnostics/strength_preview_g025_g4_20261029/PREVIEW_REPORT_ZH.md>)。当前仅供用户确认强度，未授权或恢复混合强度训练。


### 2026-10-29/30：D0 U95544 160K四档混合强度续训（最新授权）

用户批准并最终改为原Train10000 Clean×4原始ξ×G1/G2/G3/G4完整组合，共160K、每档40K/25%；每次更新effective60保持。原生Residual L1系数1、train1NFE/原S3验证、四卡NPU4–7、FP32/HF32off、恒定LR2e-6及完整U95544父状态继承。每1000新增更新完整保存、每2000五档各Val1000，G0.5仅验证；best增加G4≥20.96514237保留约束。**2026-10-30中午仅保存/验证/汇报，之后继续训练等用户决定，不自动停止。** A/B保持暂停，历史STOP和原资产保留。

独立[执行入口](<../NOT_INCLUDED.md#ref-4bc4dfa250aa21a8ebf0>)和[最新授权](<../NOT_INCLUDED.md#ref-edbb3628f32258b97a19>)。持久控制器已启动，160K四卡数据生成正在进行；旧40K准备子阶段按新指令安全退出，完整匹配产物核验后复用。父U95544全状态CPU读回及160K两轮遍历/跨轮恢复位置CPU测试已PASS；实际数据Gate、S3基线和四卡训练恢复Gate未完成不得视为已PASS或正式已开训。实时以新ACTIVE_CONTROLLER/PIPELINE_LAUNCH02、STATUS和各Gate为准。每小时自动跟进v6-d0-160k已启用，正常静默。

- 2026-10-30 01:26实际更新：160K全量SHA/组合与分割Gate、父五档基线（G0.5新测、G1–4明确复用）、真实四卡连续2步/重启末1步均PASS；3次临时更新丢弃。正式`V6_D0_U95544_multistrength160k_formal01`已从父完整状态启动，前20正式更新独立CPU验收PASS，最新本轮巡检快照U95818。此为工程启动与健康验收，尚无新正式2k五档质量结果，不代表科研成功；后续以实时日志和Gate为准。

- 2026-10-30中午查看后，用户最新要求“继续训练到下午六点”。**正常结束时间现为今天18:00 Asia/Shanghai（Unix1793354400）**，覆盖上文开放结束安排。复用原四worker/原CPU控制器、不改123冻结训练源码或CONFIG；独立纯CPU定时守护368139/start561945433已实际ARMED，到时写现有安全STOP，完整保存、最终五档Val1000后退出。新[授权](<../NOT_INCLUDED.md#ref-b70f3a14ef0a2d26b298>)与[定时执行入口](<../NOT_INCLUDED.md#ref-ed2c9d1b8d24eb23c354>)为准，18点结束尚未发生不得提前PASS；每小时自动跟进已更新，A/B仍暂停。

- 2026-10-30 13点，仅CPU定时守护预防性加固为`deadline_guard_safe_boundary_v2.py`，最新PID513476/start562212030，以同目录ACTIVE_GUARD/PIPELINE_LAUNCH02/GUARD_V2_HANDOFF_ACCEPTANCE为准；旧368139正常控制交接退出，不重启。原训练五进程身份不变、原123训练源/CONFIG和原4项守护源SHA不变。只读既有RNG元数据确认验证前后CPU Torch RNG变化，v2规避验证后重复保存已有update的全状态断言风险；当前未发生训练故障。18点记录到时状态，在安全新update边界写STOP，若正在验证则安全收尾后处理，真实STOP时间/预算照实保留。新3源manifest及6个纯CPU边界测试PASS，0新NPU任务，见[工程说明](<../NOT_INCLUDED.md#ref-f2317e1d0bf93a0b12da>)。


## V6 D0 G4重加权有限对照（2026-10-30 用户授权）

- 独立执行[execution_20261030_d0_g4_reweight_5k](<../NOT_INCLUDED.md#ref-68256a85b687d5383366>)：从同一完整Mixed最新U157505建立Control与G4-biased，各5000更新；G4改50%、其余原比例归一化，网络/原生Residual L1/LR/effective60/EMA/三步推理不改。NPU2/3各独立单卡，1号已有进程不动，4–7原任务推理继续，A/B暂停。
- 共同父、数据身份、原完整五档、逻辑四rank梯度均值与独立保存/重启已实际核验；活动CPU入口以执行目录ACTIVE_IMPLEMENTATION/PIPELINE_LAUNCH03及实时PID为准，不从本索引判定状态。原v1/v2 CPU审查失败、独立数值审查与活worker接管均保留，未修改训练actor或旧历史资产。0/1k/3k/5k固定五档Val1000，biased仅在预冻结改善/弱档保留条件通过后最多再5000，不搜索比例，不自动启动A/B。

## UniDB SR 原生D0八卡微调（2026-11-02 最新授权）

- 独立入口：[执行计划](<../NOT_INCLUDED.md#ref-394679605c3791ad50d9>)、[用户授权](<../NOT_INCLUDED.md#ref-456f0db4ac4fe47ca2cc>)。官方DIV2K SR `gamma_1e7.pth`，完整严格加载后全参数微调；原生网络、桥过程和MatchingLoss L1保持，无task prompt/Adapter/raw-ξ头，A/B仍暂停。新目录不自动命名为V7。
- 原预算（已被11月2日18:39明确停训覆盖）：de36最多八卡，1号卡可与已有服务共存但不终止服务；取消原5000更新上限，原定截止2026-11-03 12:00 Asia/Shanghai，安全保存后完成最终五档评价。下载慢允许本地下载转入。
- 官方SR载荷548649183字节，本地下载及远端独立SHA一致：`a6bb76bfabe32b8403d8b112f0ce7876114d830d20b9f4a4119aa860afef64b4`。原160K训练组合及五档验证相关175250文件已重新核SHA/shape/dtype/finite；源/分割与组合检查PASS，不生成新数据，不clamp。
- 初始CPU/NPU对照FAIL保留。逐层和单算子诊断发现NPU默认GELU与CPU精确GELU不一致；采用原GELU的FP32 erf公式作等价运行修复，不改官方源码、参数结构或loss。修复后在原容差下数值验收PASS；八卡实测选择micro12×8=96，完整保存重启和200更新Pilot工程门PASS。Pilot质量未获肯定，临时权重不继承。
- 2026-11-02 01:51快照：正式run已从官方SR重新初始化，151项raw/EMA张量全部逐一相等、Adam为空、update/cursor为0。当前八卡完成训练前U0完整五档基线中，已落盘1000/5000预测，尚无正式optimizer更新；随后自动训练至截止。实时状态仍以服务器证据为准。
- 2026-11-02 02:24后续快照：U0五档5000完整基线完成，正式已到U540；U500完整checkpoint独立CPU读回PASS（773张量、151 Adam、8rank RNG），8rank与控制器身份正常。仍按固定截止继续，不把正常推进当作质量结论。
- 2026-11-02 06:23后续快照：正式U11550、保存U11500，八卡运行正常。U10000完整五档暴露中期质量问题：G4 15.937571dB，较U0下降2.284828dB；弱档较U0改善但五档仍低于Input PSNR。全部5000已保存预测独立CPU重算/SHA/配对核验PASS，固定首来源图可见G4错误纹理。已记录警示，继续原授权截止和方法，不以此声称根因或最终结论。
- 2026-11-02 18:39用户明确提前停训并要求保存、终评和诊断，覆盖原11月3日中午训练预算。已在完整更新边界停止于U45411、4359456次呈现，最终完整checkpoint SHA c59a997659f8b3f63996344cde29f4bab2d5916de35a750ca23eb76f08da3bca；原8rank仅完成最终五档验证。STOP_REQUEST和NO_TRAINING_RESUME保留，不得恢复训练。后续仅收尾/诊断推理获授权，见新STOP_AND_DIAGNOSIS_AUTHORITY与CURRENT_EXECUTION；尚未完成全部交付。
- 2026-11-02最终收尾完成：[最终报告](<unidb_d0_20261102/FINAL_REPORT_ZH.md>)、[最终验收](<unidb_d0_20261102/USER_STOP_FINAL_ACCEPTANCE.json>)、[本地交付核验](<../NOT_INCLUDED.md#ref-b7fb1cca8e0f8d28f097>)。U45411五档各1000全部完成，G4 PSNR/SSIM/LPIPS=15.894991/0.302786/0.425333；强恢复目标未达成。U0/U10000/U45411三份独立本地完整checkpoint全文件SHA PASS，230份报告/CSV/源码/验收/图文件同步核SHA PASS；原始远端资产保留，自有worker和CPU控制器退出。诊断200图G4验证子集：U0→最终EMA在原生SDE为18.008→15.761dB，在官方mean-ODE为18.531→18.525dB（Input18.561），主要退化定位到微调权重与随机采样路径交互；确定性输出也未实现强恢复改善，不代表外部泛化成功。1600份mean-ODE预测独立CPU核验及6图目视QA完成。`unidb-sr`自动跟进已删除，STOP/NO_TRAINING_RESUME继续保留。
- 远端根 `/data2/yanglin/unidb_d0_20261102`；实况以STATUS、ACTIVE_PIPELINE对应PID/start_ticks、工程gate和真实checkpoint为准。自动跟进`unidb-sr`已建立；计划、下载和数据PASS不是训练或质量成功。


## UniDB D0 双路线3＋5卡限时训练（2026-11-03新授权）

用户已明确授权原路线3卡继续、改进路线5卡重开，两线至2026-11-03 12:00（Asia/Shanghai）安全保存停训，随后完整验证与验收；取消旧08:00及500/2000步建议上限。独立[执行入口](<../NOT_INCLUDED.md#ref-4e96165a8cee215554df>)、[具体配方](<../NOT_INCLUDED.md#ref-c57df930ebf3ef357f83>)、[用户授权](<unidb_dual_20261103_until_noon/protocol/USER_AUTHORITY.json>)。远端同名独立目录；原实验、旧STOP与封存forensic audit保留，V5/V6不改、不恢复。

原路线物理0/1/2，从UniDB U45411完整RAW/Adam/EMA/八逻辑rank RNG继续原生MatchingLoss L1、LR1e-5；改进物理3–7，从官方SR完整gamma_1e7权重fresh Adam/EMA，使用GT离散一步后验均值对齐的ε_opt MSE、200更新warmup至2e-6、墙钟cosine到2e-7、gradclip1.0。两线有效B96/FP32，原生网络/bridge/G1–G4等比例保持，无task prompt/Adapter/raw-ξ/A/B或物理递归。多项配方与起点、资源不同，属于实用路线比较，不能单因果归因。

真实8→3全参数/Adam/EMA/RNG数值门和两路线新格式恢复门已通过；首批正式实际更新、每次96样本和八个活rank身份独立验收见evidence/FORMAL_STARTUP_ACCEPTANCE.json。下载/工程PASS/首批更新不代表训练完成、收敛或泛化改善。根STATUS与runs/{original,improved}日志/完整checkpoint/验证才是实况；最终还须五档各Val1000、独立纯CPU保存输出四指标复算、固定图/曲线目视与本地交付，自动跟进ID unidb-3-5仅在这些完成后删除。

- 2026-11-03 07:25中期全量质量警示：改进路线U10000五档各1000完成，G4 PSNR14.708872dB，低于官方SR起点18.222398及Input18.795938。独立checkpoint/EMA/全部验证身份审阅PASS，保存输出四指标CPU预审计进行中；尚未实现强恢复，最终交付条件未满足。两路线按授权继续至12:00，不改方法或截止。

### UniDB 双路线延长至18点（2026-11-03最新授权）

用户在中午明确要求“继续训练到下午6点”，覆盖上述12点截止。新独立[执行入口](<../NOT_INCLUDED.md#ref-130539bcb64b379709f4>)和[授权](<unidb_dual_20261103_until18/protocol/USER_AUTHORITY.json>)，远端同名目录。两路3＋5卡，从中午完整原U55760、改进U20000保留RAW/Adam/EMA/RNG/cursor接续，绝对截止2026-11-03 18:00 Asia/Shanghai。原LR1e-5，改进延长段固定原cosine末端的2e-7，不重新warmup或抬高LR；原调度历史保留。网络/桥/loss/B96/FP32/数据和EMA不变。旧until_noon及旧STOP、封存报告/原实验不改。

两父完整CPU验收PASS，已核身份停止本任务旧评价与收尾actor以交接资源，外部任务不动。新同卡数完整恢复/重启门通过后自动正式继续；实际启动及完成仍以新STATUS、PID/start、日志、完整checkpoint和全量验证为准，不能从此授权或工程准备推断开训/成功。最终保存输出CPU四指标、图/曲线目视、进程退出和本地交付条件保持，自动跟进unidb-3-5已更新为18点。

- 2026-11-03 12:50接续实况：两路线恢复/重启实际NPU门和完整CPU读回均PASS，门内4更新全部丢弃；正式已从U55760/U20000接续并产生真实更新，controller PID1518007/start596530122。正式START与父RAW/Adam/EMA/方法RNG/cursor逐字节一致、首批每次96真实位置及8rank身份独立PASS，见新目录evidence/FORMAL_EXTENSION_STARTUP_ACCEPTANCE.json。改进延长段LR2e-7，原路线1e-5。资源门曾0更新拒绝新增外部小任务，已归档；按精确身份与至少48GiB free HBM共存，未停止外部任务。实际当前更新数仍查远端STATUS和日志，尚非18点完成/质量成功。

### UniDB 双路线延长至今晚24点（2026-11-03最新确认）

用户“继续训练到12点”已明确确认“今晚24:00（11月4日00:00）”，新绝对截止2026-11-04 00:00 Asia/Shanghai（1793721600），覆盖旧18点最终截止。新独立[执行入口](<../NOT_INCLUDED.md#ref-3a58c66eb3a03ea55897>)与[授权](<unidb_dual_20261104_until00/protocol/USER_AUTHORITY.json>)。3＋5卡、原MatchingLoss L1/LR1e-5和改进ε_opt MSE/clip1/LR2e-7均保持，不重warmup、不重初始化；完整RAW/Adam/EMA/RNG/loader/cursor与历史调度接续。当前先准备新root，旧18按原自然截止保存，再从CPU验收完整latest交接；准备/授权不代表已新开训。旧冻结源码、既有结果与partial保留，外部任务不动。完整恢复门PASS丢弃gate后自动formal，最终全量评价/保存输出CPU审计/新图目视/退出/本地交付条件不变；自动跟进已更新午夜截止。

- 17:48午夜交接准备已PASS，CPU协调器PID1949554/start598372909持久等待原18点自然保存。新恢复门及formal仍待父完整CPU验收与安全交接，不将armed误报已开训。另按用户最新要求准备今天19:00实际快照汇报PPT，与训练独立推进。

- 18:14午夜交接实况：原U61764/改进U30000自然截止完整父状态CPU验收及RAW全rank摘要绑定PASS，旧12自有PGID核身份交接退出且外部signals0。新controller1969261/start598504915已进入同3/5恢复工程门，首轮各1更新和CPU读回PASS；重启门/正式仍以远端证据为准。

- 18:25午夜正式续训已核：4次门更新丢弃后18:16从两父正式启动，原U61933/改U30307真实完成更新，8rank活、冻结源一致。两START与父RAW/Adam/EMA/方法RNG/cursor/祖先完整逐字节独立CPU验收及首3全局B96更新PASS（52ceff1a…337864），审计CPU已退出；正式训练仍进行，不代表最终质量或泛化成功。

- 19:00实际冻结状态汇报已完成，11页可编辑PPTX、约5分钟讲稿及4条答疑位于[汇报交付目录](<../NOT_INCLUDED.md#ref-6a8508bf828e1d05f4d3>)。快照原U62602、改U31565；文件、数值与全页目视已验。仅PPT一次自动任务删除，午夜训练与最终验收继续。

## 全项目交接整理（2026-11-03新增要求）

用户明确要求“训练完过后，从头到尾整理整个湍流项目，我不干了，搞一个交接的”。新增[交接准备入口](<../NOT_INCLUDED.md#ref-cd48940019e639459664>)和[请求/验收条件](<project_handoff_20261104/HANDOFF_REQUEST.json>)，覆盖全部历史版本、重要支线、数据/代码/环境/模型/结果、复现方法、已知问题和待办。本次UniDB仍按2026-11-04 00:00截止与原最终验收执行，随后补齐实际最终结果并交付总资料包；当前为准备中，非交接完成。不得因此删除或改名旧资产、重跑实验或向外部人员发送材料。现有`unidb-3-5`后续范围包括实验完整交付与全项目交接，二者完成后才删除跟进。

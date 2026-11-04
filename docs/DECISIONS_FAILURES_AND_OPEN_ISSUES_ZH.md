# 为什么换方案、哪里失败、还有什么没解决

本页是2026-11-04总交接的决策补充，随Git仓库维护。它与[首页](../README.md)、[最终主交接说明](../handoff/documents/MAIN_HANDOFF_FINAL_v2.md)配套；原报告、Word/PDF、源码和验收包保留原字节。**项目已收尾，本文没有启动新的训练或评测。**

接手者先读下方“转向总览”，再按节点查原始依据，最后看未决事项和常见问题。`U`是优化器更新次数；`D0`通常是该版本的直接恢复基线，不同版本的D0并非同一个模型；`ξ`是模拟器原生随机变量，不能直接等同于可见形变或有效模糊核。

## 读这份决策记录的口径

- **有记录的决定：**能指向当时的协议、用户授权、方法说明或停止记录。写明执行目的，不用后来的失败补造当时动机。
- **观察到的失败／限制：**评价未达到具体目标、工程故障、协议错误、用户停止和“未执行”分别记录。停止本身不证明方法无效，工程修复也不证明恢复有效。
- **解释与未知：**支持某种解释的证据和仍未隔离的原因分开；缺少转向理由的节点直接标注缺口。时间上接续不等于前一方法已被严格否定。
- **下一步：**只说明接手者如何核实缺口或设计新的判定；均是未执行的建议，不是待自动恢复的旧任务。跨版本通常改变了数据、目标、初始化或预算，不能直接用绝对分数排优劣。

## 转向速查

| 阶段／节点 | 已有记录能回答的“为什么” | 失败或停止的性质 |
|---|---|---|
| [L1：v1→v1.5](#l1) | 修跨强度传输与精修可能破坏下一状态的问题 | 评价身份先纠错；纠偏后仍未达继续门槛 |
| [L2–L3：v2→v2.5](#l2) | 新的预测状态递归；随后改source-half协议 | v2未正式训练；v2.5按时间边界暂停，不能称充分训练失败 |
| [L4–L6：v3、T1/T2、I1](#l4) | 分离拟合/泛化、直接恢复/物理反馈、精修目标/恢复效果 | 训练人口可拟合；留出回退；12个精修候选未过门 |
| [L7：独立几何支线](#l7) | 检查真实几何轨迹是否比解耦/普通增强更有价值 | Gate1失败，按规则未开Gate2 |
| [V4-01–02：新接口与诊断](#v4-01) | 用户新方法；加入固定观测条件，检查拟合及物理歧义 | 小样本拟合不转为稳定验证优势 |
| [V4-03–05：骨干与平台](#v4-03) | 候选扩展、用户指定A100随机重开及NAFNet预训练 | 非公平骨干淘汰赛；候选门通过不等于质量成功 |
| [V4-06：扩数据与表示支线](#v4-06) | 扩source覆盖；另查连续更新与Raw–Field表示 | 主线用户停止且部分停点无full；支线混合/负结果 |
| [V5-01–02：新模拟器与Concertormer](#v5-01) | 用户新规格；选择有官方去模糊权重的恢复主体 | 原Simulator FAIL带局限接收；选型不等于最优证明 |
| [V5-03–04：A→B与失稳收尾](#v5-03) | 先中间态对齐再真实递归；失稳时保护原方法恢复 | B胜独立D0但末步仍低于首步；首个失稳原因未定位 |
| [V5-05–06：ResShift→DiffUIR](#v5-05) | 候选探索；冻结VQ重复RGB往返有累积漂移 | VQ风险针对所测组合；未进行ResShift正式训练 |
| [V6-1–4：联合→原生目标及暂停](#v6-1) | 用户要求回到官方原生训练路径；工程短测与恢复 | 缺干净目标消融；A/B多步优势不足；性能候选未达门 |
| [V6-5–7：混强度、偏重与外部验证](#v6-5) | 扩强度覆盖，再检查G4权重及域外能力 | 强弱档取舍；追加门失败；真实域与官方能力有退化 |
| [U-1–3：UniDB选型、诊断、双路线](#u-1) | 原生RGB条件接入较直接；用户要求原线与新配方并行 | loss下降未转为强档恢复；综合配方非单因素对照 |
| [U-4–5：两次延期→总交接](#u-4) | 用户明确延长时限，随后决定研究收尾交接 | 延期不是质量门通过；工程终验已过、质量目标未达 |

以下共31个节点，每个都给出理由、失败/限制、未决与可点击的证据。查共性问题可跳到[12项未决台账](#open-issues)；资产和继续工作的常见疑问见[接手FAQ](#handoff-faq)。

## 早期路线：先改约束，再检验预测状态递归

<a id="l1"></a>

### L1｜v1 → v1.5：原观测重建正确，不保证下一个弱态正确

**为什么换（方法文件明确记录）。** v1的验证选择K0单步，K5比K0低约0.915dB；v1.5方法书明确要修两处错位：训练只约束终端观测、没有约束ξ跨强度传输；推理只压低原观测误差，可能破坏下一桥状态。因此增加跨强度轨迹一致性和相邻弱态执行约束，同时保留双头网络与原模拟器。更早已停用的`v02_src`不是本次A1，不可混并。

**失败在哪里。** A1完成3000步，但原正式评价seed与协议不符、v1比较器误用RAW；纠正为协议seed＋v1 EMA后，K0仍下降0.148dB，高G一步增益及next-state改善都未过门，Stage B未开。

**还没解决。** 预测clean参与渲染的状态错配得到诊断支持，尚无唯一根因；可微性检查已有非零梯度证据，不能再把“模拟器不可微”当定论。全量val400、Test和多训练seed确认未完成。

证据：[v1.5方法书开头](../handoff/evidence/v02_new_protocol_20260829/docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_METHOD.md) · [A1纠偏与停止报告](../handoff/evidence/analysis/v02_a1_20260830/report/V02_A1_OBJECTIVE_HANDOFF_FOR_GPT.md) · [v1冻结报告](../handoff/evidence/reports/bridge_v01_retrospective/sources/original_final_report.md)。

<a id="l2"></a>

### L2｜v1.5之后的v2：训练时就承受自己的预测误差

**为什么换（方法变化有记录，唯一因果没有定论）。** v2采用老师的预测状态递归：每步用预测clean与ξ渲染下一输入，整条路径反向传播，不用GT状态替代。设计上让训练暴露于自身预测状态；尚未用正式匹配消融验证状态错配的因果作用。源码从冻结v1重新建独立分支，不能称v1.5 checkpoint续训，也不能写成“v1.5失败已证明只能这样改”。

**实际停止原因。** 全数据分支只做3次可丢弃工程更新，正式训练未启动，就因数据协议改为source层减半而终止。不是全量训练失败，也没有正式质量成绩。

**还没解决。** 完整数据上的该配方尚未被正式验证；不能拿后继半量结果补作v2结果，或比较full与half谁更好。

证据：[v2冻结方法](../handoff/evidence/v03_recursive_from_scratch_20260915/README.md) · [执行状态](../handoff/evidence/v03_recursive_from_scratch_20260915/docs/EXECUTION_STATUS.md) · [明确的协议变更终止记录](../handoff/evidence/v03_recursive_from_scratch_20260915/artifacts/DATASET_PROTOCOL_CHANGE.json)。

<a id="l3"></a>

### L3｜v2 → v2.5：改变数据规模；U204是时间边界，不是收敛判断

**为什么换（可确认到协议层）。** 已有终止与交接记录明确改为source层50%抽样：训练1000来源、每来源仍4条轨迹，从新随机状态开始，旧工程权重不接续。记录足以确认“改了什么”，不足以将主观动机唯一归为算力、过拟合或速度。

**实际没完成什么。** 授权卡到06:57才可用，完成工程检查与pilot后10:24开训，按约定11:50安全暂停，实际仅204步。U173的16条轨迹诊断中K5比K0低2.076dB，但这不是U204终点或完整val200；停止记录明确说当天是运行时间边界。

**还没解决。** 尚无full-val模型选择与冻结Test结论；早期多步回退不能证明充分训练后必然失败或必然变好。U204是可恢复资产，不是推荐性能模型。

证据：[中午交接，含时间线和停止原因](../handoff/evidence/v03_half_recursive_20260915/docs/NOON_HANDOFF.md) · [冻结协议](../handoff/evidence/v03_half_recursive_20260915/docs/FORMAL_PROTOCOL.md)。

<a id="l4"></a>

### L4｜v2.5之后的v3：先把“能不能拟合”与“能不能泛化”分开

**为什么换（新实验目的明确；不倒推上一版失败）。** v3另取100个固定训练来源，用G4/3/2/1完整尾路径检验小样本拟合与真实Full-BPTT；同时改了路径、预测次数和推理定义，不是只把v2.5的数据调小。现有记录未建立“U204质量失败导致v3”的唯一决策链。

**哪些成了、哪些失败。** 1000步后，训练1600起点不精修递归较自身首步有收益（G4约+0.954dB）；精修却降低质量。后续未见来源筛查暴露回退。因此小样本能拟合成立，稳定泛化在已测留出人口上未获支持。

**还没解决。** 数据规模是否为主因没有source-scaling实验；也没有多训练seed或外部benchmark。BF16梯度一致性未过，正式使用FP32；状态恢复通过不等于之后NPU计算逐位可重演。

证据：[v3方法与实现要求](../handoff/evidence/v031_overfit100_g4321_n4/docs/V031_STAGE1_METHOD_AND_IMPLEMENTATION.md) · [U1000结果](../handoff/evidence/v031_overfit100_g4321_n4/artifacts/V031_UPDATE1000_REPORT.md) · [留出集曲线](../handoff/evidence/v031_followup_20260918/reports/heldout_curve_20260919_2138/delivery/REPORT.md)。

<a id="l5"></a>

### L5｜v3 → T1/T2：分别问“直接恢复够不够”和“物理生成下一输入有无必要”

**为什么改（独立问题预先写明）。** T1从共同未训练初值取clean结构，去掉ξ与重退化，只训练一次直接恢复；T2保留ξ和重退化监督，却直接把预测clean送入下一步，不让模拟器生成下一输入。旧v3权重不被续训或覆盖。两者各1000步、同32000起点曝光，但T1/T2网络预测次数为32000/80000，不是等算力对照。

**失败在哪里。** T1在训练集四档都优于旧递归末步，T2无稳定优势；但32个未见来源上三个模型各档末步都低于Input。U500→1000训练变好时，T1/T2留出平均PSNR反降约0.986/1.209dB。

**还没解决。** T1同时改变ξ分支、监督和计算量，不能归因于某一个loss；T2仍有重退化监督，不能据此否定全部物理约束。来源不足只是待检验解释。

证据：[T1问题与定义](../handoff/evidence/v031_followup_20260918/shared/authority/01_T1_DIRECT.md) · [T2定义](../handoff/evidence/v031_followup_20260918/shared/authority/02_T2_FEEDBACK_R.md) · [正式结果总览](../handoff/evidence/v031_followup_20260918/reports/README.md) · [后补留出曲线](../handoff/evidence/v031_followup_20260918/reports/heldout_curve_20260919_2138/delivery/REPORT.md)。

<a id="l6"></a>

### L6｜精修负效 → I1有限修复：降低观测误差不等于提高恢复质量

**为什么改（诊断后做有限候选验证）。** v3精修取证在固定48起点发现观测目标下降80.84%–85.97%，下一次恢复却各组16/16变差；没有找到足以解释的solver/状态/返回值实现错误。I1固定旧RAW1000、不更新网络，比较L/P输出保护与较弱精修等预声明候选。

**失败在哪里。** 开发与确认来源都曾被旧模型训练，不能当未见来源验证。8个开发来源、24起点上的12个新候选全未过门，最接近未精修基线者仍为负。按规则不启动确认集和候选heldout，不存在“挑一个最不坏候选作为修复成功”。

**还没解决。** 首跳梯度冲突只在预设少量点被确认，不能外推为全体样本唯一根因；减少步长只能减轻损害，尚没有通过质量门的修复。未运行确认集是规则性停止，不是遗漏的成功证据。

证据：[精修独立取证](../handoff/evidence/v031_overfit100_g4321_n4/artifacts/refinement_forensic_20260918/FINAL_REPORT_ZH.md) · [I1候选与停止结果](../handoff/evidence/v031_followup_20260918/reports/I1_REPORT.md)。

<a id="l7"></a>

### L7｜独立几何支线：从标量弱化改问实际几何动作是否有额外价值

**为什么换（独立协议，非老师v2）。** 2026-09-08/09的几何实验检验：按标量强度训练是否覆盖实际几何校正T(Y)，以及真实轨迹是否优于简单解耦或普通增强。固定四臂、两种采样顺序，共8模型各5000步；不能与随后v2的身份合并。

**失败在哪里。** actual轨迹未同时达到相对decoupled与generic各≥0.10dB等预设条件，Gate1失败，Gate2未开。前端T(Y)自身比Y提高约0.51–0.56dB，也没有等幅转化为最终恢复收益。

**还没解决。** 真实轨迹的独立必要性、无损升级、blind真实场景能力未获支持；后续oracle/action只是草案。首轮执行前源码快照不完整，也不能拿后补快照冒充当时冻结代码。

证据：[几何支线协议](../handoff/evidence/analysis/v03_validation_20260909/PROTOCOL.md) · [完整负结果与后续草案](../handoff/evidence/analysis/v03_validation_20260909/REPORT.md) · [最终Gate状态](../handoff/evidence/analysis/v03_validation_20260909/FINAL_STATUS.json)。

## V4 与 V5：接口、规模与恢复器的更换

<a id="v4-01"></a>

### V4-01｜v3 → V4 Noise Bridge：改造递归接口，不是已证明旧方案只有一个缺陷

- **为什么换：有明确新方法指令，缺完整排他性选型论证。** 2026-09-29记录的是用户提供新算法并要求独立实现：将当前状态Z与固定原观测Y一起输入，加入[g,t]条件，比较直接恢复D0、物理递归R0、加噪路径R08，后来加入线性路径L08。方法意图是检查条件与路径设计；不能仅因v3留出回退，就断言“缺原观测条件”已被定位为唯一原因。
- **失败/限制：** 首轮三法各1000更新，val800 PSNR为19.1983/18.8805/19.1895，均低于Input20.5394。R08优于R0，却没有超过D0。小样本可以拟合不等于新图像恢复有效。
- **还没解决：** 各路径在未见来源上稳定优于直接恢复；从v3到V4多个变化各自的贡献。
- **证据：**[V4实现授权与接口](../handoff/evidence/v4_noise_bridge_20260929/README.md)、[首轮最终报告](../handoff/evidence/v4_noise_bridge_training_20260929/results/FINAL_REPORT_ZH.md)、[V4历史](../handoff/history/v4_history.md)。

<a id="v4-02"></a>

### V4-02｜从正式小训练转向拟合与物理诊断：检查“到底能不能学”，而非把低验证分直接归罪于模拟器

- **为什么做：已记录诊断对象；不能把结果当换骨干的唯一理由。** 五来源完整D0与clean-only配对、续至200epoch、单来源和单对实验分别检查训练拟合；固定算子与盲补偿支线检查物理可辨识性。曾有一个clean-only续训200分支因用户纠正对象取消、零更新，不是训练失败。
- **失败/限制：** clean-only五来源训练PSNR升至31.0374，val64降至15.7443；完整D0有同类回退。单对可到34.6935只证明记住该对。有限算子非单射和局部C–ξ补偿证据不等于所有自然图像不可恢复，也不证明“模拟器不能反传”。
- **还没解决：** 拟合与泛化的差距来源、自然图像先验能消除多少歧义。
- **证据：**[clean-only续训报告](../handoff/evidence/v4_overfit5_cleanonly_resume_plateau_20260929/results/report/REPORT_ZH.md)、[完整D0续训](../handoff/evidence/v4_overfit5_full_d0_resume_20260929/results/report/REPORT_ZH.md)、[固定算子](../handoff/evidence/v4_fixed_physics_operator_20260929/results/report/REPORT_ZH.md)、[盲补偿](../handoff/evidence/v4_blind_ambiguity_20260929/results/report/REPORT_ZH.md)。

<a id="v4-03"></a>

### V4-03｜U-Net → SwinIR → Restormer：扩展骨干与预算，不能写成已完成单因素淘汰赛

- **为什么换：新候选与新协议有记录，旧骨干被“证明不行”的因果结论没有。** SwinIR正式分支同时改L1、日程和骨干；Restormer采用D²Turb公开结构、共同随机U0，并把训练扩大到2000来源。PiRN骨干候选只做功能门、正式更新0；不是完整PiRN复现成绩。
- **失败/限制：** U-Net共同U5000时D0最高19.6670，仍低于Input20.5394；SwinIR U2000 D0为20.4732，仍略低于输入且三递归组更低。Restormer共同U5000 R0较D0仅约+0.0391dB，不能和不同人口的后续NAFNet主表直接排名。
- **还没解决：** 骨干、loss、数据和预算分别贡献多少；多个seed的稳定性。旧U-Net上限10k不是已核实完成10k。
- **证据：**[U-Net U5000读数](../handoff/evidence/v4_train500_matched_v2_perf1_eval5kfull/reports/U05000_READOUT.md)、[SwinIR科学协议](../handoff/evidence/v4_swinir_l1_train500_2k_20261001/protocol/SCIENTIFIC_SPEC.json)、[Restormer匹配U5000](../handoff/evidence/v4_nafnet_sidd116m_train8000_20261006/reports/OLD_U5000_FULLVAL_SUMMARY.md)、[历史全表](../handoff/history/v4_history.md)。

<a id="v4-04"></a>

### V4-04｜Restormer NPU → A100：用户指定的平台变更，不是已证明NPU导致质量失败

- **为什么换：明确用户指令。** 10月3日先要求停NPU转A100，随后明确A100从共同随机U0重开，不继承NPU权重。现有记录不足以把迁移解释成质量、稳定性或吞吐的唯一因果结论。
- **实际终态：** NPU D0/R0/R08/L08分别在U669/196/198/201完整保存；A100是四个新run。10月6日NAFNet接替后，Restormer主比较收于共同U5000，旧>5000结果保留；旧20k预算失效。
- **还没解决：** 平台差异的严格同权重对照；完整环境可迁移性。目录名`a100_resume`不能当从NPU续训的证据，旧暂停也不表示现在仍占卡。
- **证据：**[迁移与随机重开指令](../handoff/evidence/v4_restormer_train2000_npu_20k_20261003/README.md)、[最终阶段口径](../handoff/history/v4_history.md)。

<a id="v4-05"></a>

### V4-05｜Restormer → NAFNet-SIDD＋Train8000：用户指定预训练主体并扩大数据

- **为什么换：明确指定，不是先做公平比较后证实最优。** 用户要求116M的NAFNet-SIDD，随后明确用官方预训练；四法共享适配U0，全参数训练，数据扩大到Places4000＋LSDIR4000来源。预训练复用主体，但六通道、条件和clean/raw头并不等于原完整去噪映射。
- **失败/限制：** 四法各20k完整结束，RAW/OFF fullVal8000中D0/R0/R08/L08 PSNR为23.0908/22.9195/22.7363/22.8114；D0的PSNR、SSIM、LPIPS均值最好。较强主体和更多数据没有建立递归的稳定优势。
- **还没解决：** 去噪预训练的独立增益、共同计算预算下的路径收益；不能把不同人口、初始化和精度的跨骨干差异当消融。
- **证据：**[用户指定与适配边界](../handoff/evidence/v4_nafnet_sidd116m_candidate_20261005/README.md)、[Train8000协议](../handoff/evidence/v4_nafnet_sidd116m_train8000_20261006/protocol/SCIENTIFIC_SPEC.json)、[最终REV4资料](../handoff/evidence/v4_nafnet_sidd116m_train8000_20261006/deliverables/U20000_FOUR_METHOD_RESULTS_20261016_REV4/README_ZH.md)。

<a id="v4-06"></a>

### V4-06｜Train8000 → Train15000及支线：扩大覆盖，同时检查状态表示和更新规则

- **为什么做：** 主线从各法U20000完整续训并新增7000来源；连续差分、Raw–Field是另建的小规模诊断，不是主线无缝替换。它们分别检验另一种连续更新、raw变量与物理场坐标；现有资料不能把这些设计的提出原因归为单一已定位缺陷。
- **失败/限制：** 10月18日用户要求停止，D0/R0到U40000，R08/L08仅U32891/U21559，后两停点未补full。连续差分是Train1000/G4、Val200来源×2＝400输入、单seed，末级比D0低0.3329dB；Raw–Field另为Train128、Val32来源×2＝64，未见Field的PSNR收益，两者末级都弱于首级，虽Field的SSIM/LPIPS略好。
- **还没解决：** 未完成的共同终点比较、大样本/多seed泛化；这两个支线不是同人口或纯坐标消融，小集合结论不能外推全部强度。原始raw精修目标J下降也不保证GT质量提高。
- **证据：**[Train15000终态](../handoff/history/v4_history.md)、[连续差分结果](../handoff/evidence/v4_followup_pilot_lr5e6_attempt2_20261014/PILOT_REPORT.md)、[Raw–Field最终报告](../handoff/evidence/de36_g4_raw_vs_field_12h_20261015/reports/FINAL_CPU_AUDIT/FINAL_REPORT_ZH.md)。

<a id="v5-01"></a>

### V5-01｜V4 → V5：新的用户方法规格，同时改变接口、时间、模拟器与损失

- **为什么换：明确新规格；不是某一个V4失败已被修复。** V5沿用clean/raw ξ预测与物理递归，改为只输入当前RGB与连续时间，调整零端点、Smooth/Tilt/FFT路径与损失权重。规格明确依据用户V5交接文档；现有证据不能排他性说明每项改动为什么优于原式。
- **失败/限制：** Simulator近零g=1e-4最大误差约0.002176，超过原0.002阈值，部分有限差分门仍失败。用户按已有梯度、端点和解析参考支持证据“带已知局限研究接收”，原FAIL没有改PASS；A100独立门也未全通过。
- **还没解决：** 非平稳协方差、raw冗余/零方向、负PSF和越界等原始局限，以及它们对恢复质量的贡献。训练成功不能追认物理模型全验证。
- **证据：**[方法规格](../handoff/evidence/v5_v1_simulator_20261020/docs/METHOD_SPEC_ZH.md)、[带局限接收理由](../handoff/evidence/v5_v1_simulator_20261020/acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md)、[A100独立报告](../handoff/evidence/v5_v1_simulator_a100_20261021/a100_reports/A100_GATE_REPORT_ZH.md)。

<a id="v5-02"></a>

### V5-02｜选Concertormer，而非BioIR或AdaIR：任务与权重来源匹配优先

- **为什么选：有当时的选型论证。** Concertormer有明确单帧去模糊配置、官方权重入口和50.54M完整主体；预训练可能减少学习低层表征成本，但运动模糊到湍流的迁移仍是待证假设。BioIR-L作跨结构备选；AdaIR暂缓的具体原因是256输入下频率模块出现空低频mask及阈值层无梯度，不能泛称整个网络不能学。
- **实际改变：** 最终采用官方GoPro主体和RGB残差输出，再加time/raw头；早期审查的sigmoid提案没有进入最终同一形式。四步Full-BPTT计算很重，PSF等价反传优化是工程加速，不是质量提升。
- **还没解决：** 同协议下三个候选的恢复排名；静态选型没有证明Concertormer最快或最好，未开展BioIR/AdaIR正式对照。
- **证据：**[当时的骨干比较](../handoff/evidence/v5_backbone_review_20261021/BACKBONE_COMPARISON_ZH.md)、[最终采用形式](../handoff/history/v5_history.md)、[等价加速报告](../handoff/evidence/v5_concertormer_10k_20261021/optimizations/psf_adjoint_20261021/REPORT_ZH.md)。

<a id="v5-03"></a>

### V5-03｜D0后不直接做旧10k递归，改A6k → B4k：先学中间态，再对齐自身预测路径

- **为什么换：10月22日用户明确覆盖旧后继方案。** A先从D0 best继承，对预生成真实物理中间态作单步训练；B再从A best出发，在自身预测产生的四步路径上Full-BPTT。这一训练结构检验中间态预训练与真实递归对齐；B并非看到A最终失败后才临时补救，A/B是在先约定的序列。
- **失败/限制：** 同原Val1000，D0/A/B的best PSNR为21.4435/20.1567/21.7361。B较D0好0.2926dB、明显减轻A的末级回退，但B第四步仍比自己的第一步低0.1053dB。A更容易的Val-A单步27.0828不能冒充原G4效果。
- **还没解决：** 多步为何仍不稳定胜过首步、预训练/额外成本与对齐各贡献多少、锁定Test及真实域推广性。
- **证据：**[A/B用户覆盖指令与实际定义](../handoff/evidence/v5_concertormer_10k_20261021/reports/CURRENT_EXECUTION_ZH.md)、[三段统一最终结果](../handoff/evidence/v5_concertormer_10k_20261021/reports/alignment_A6k_B4k_20261022/UNIFIED_VAL1000_FINAL_ZH.md)。

<a id="v5-04"></a>

### V5-04｜失稳暂停与性能候选：恢复原配方，不把未通过的优化偷偷当新方案

- **为什么暂停：明确数值异常。** B日志曾出现有限但极大loss/梯度；固定U750隔离前向复现网络/物理递归放大，FP32也发散，故“仅换FP32”不是已验证修复。最终从完整U1000恢复，增加只停止、不改梯度的guard，保持方法、精度、LR和原预算。
- **失败/限制：** 短CPU的原LR/半LR/四分之一LR均未复发，不能证明降LR更好；固定U750复现也不是逐更新重放，尚未定位历史首个失稳update。合并同步、trusted-strength等性能候选未满足逐值一致/至少5%收益，没有采用；一次测速漏恢复scheduler的工程错误已修复，不是正式模型更新。
- **还没解决：** 首次异常的唯一因果链。U1001–4000后来健康不消除历史失败；未提交尾部与诊断成本单列。
- **证据：**[暂停、复现、保护恢复与性能窗口记录](../handoff/evidence/v5_concertormer_10k_20261021/reports/CURRENT_EXECUTION_ZH.md)、[最终收尾口径](../handoff/evidence/v5_concertormer_10k_20261021/reports/alignment_A6k_B4k_20261022/FINAL_REPORT_ZH.md)。

<a id="v5-05"></a>

### V5-05｜Concertormer后考察ResShift：候选探索，不等于已证明扩散会解决递归问题

- **为什么转候选：** 已有V5结果保留；ResShift作为官方去模糊权重、四步latent diffusion候选被审查。现有最终报告未保留“为何优先它而非所有其他模型”的完整排他理由，不用事后结果补造。
- **失败/限制：** 仅4张固定湍流图zero-shot，Input17.7892→16.9712dB，不能当全量终局。FP32恢复/固定真实ξ两阶段反传可行，但FP16递归有NPU算子阻塞；未验收完整预测ξ四物理阶段训练，也未开始正式ResShift训练。
- **还没解决：** 训练后质量、四阶段稳定性、适合预算下的成本；技术可反传不等于选型已经成功。
- **证据：**[ResShift首阶段报告](../handoff/evidence/v5_resshift_stage1_20261026/reports/FINAL_REPORT_ZH.md)、[反传与VQ诊断](../handoff/evidence/v5_resshift_grad_vq_audit_20261026/reports/FINAL_REPORT_ZH.md)。

<a id="v5-06"></a>

### V5-06｜ResShift → DiffUIR-L：VQ往返风险后转试像素空间恢复

- **为什么换：有针对组合的诊断及后续改选记录。** 目标仍考虑高保真四次RGB物理递归；250张clean连续四次官方VQ往返，PSNR28.7253→24.9492，250/250逐轮下降，因此建议先调整表示/往返方式，随后用户改选像素空间DiffUIR-L；已有记录不证明这一风险是唯一改选理由。改变了表示空间、网络和扩散定义，不是单换U-Net。
- **失败/限制：** codec诊断没有UNet、Simulator或训练，不能据此否定全部latent diffusion。DiffUIR-L 100更新pilot虽工程通过，Val50 U0/最佳U25/U100为18.7399/18.7411/18.4810，仍低于Input18.9536；未证明稳定收益。144次benchmark更新全丢弃，不能计作正式训练。
- **还没解决：** 像素空间是否真正解决多步和泛化问题；V6正式训练是之后的新授权，不由短pilot的PASS自动推出。
- **证据：**[四轮VQ风险与选型建议](../handoff/evidence/v5_resshift_vq_repeat4_20261026/reports/FINAL_REPORT_ZH.md)、[DiffUIR-L替换范围和pilot](../handoff/evidence/v5_diffuir_l_stage1_20261026/reports/FINAL_REPORT_ZH.md)、[有界benchmark](../handoff/evidence/v5_diffuir_l_benchmark_20261026/BENCHMARK_REPORT_ZH.md)。

## V6 与 UniDB：为什么切换、哪里没有达到目标

<a id="v6-1"></a>

### V6-1｜从直接回归转向 DiffUIR-L，并保留 D0→A→B 物理链

- **当时的理由与改变：**方法规范选择像素空间条件扩散作为新的 Clean 恢复器，继续检验外层物理递归；不是已经证明“扩散比 Concertormer 好”。D0从官方EMA网络参数开始，初始目标为 `0.5×原生残差L1＋三步末端Clean MAE`；A增加物理时间条件和独立ξ头，B连接完整四步物理反传。ResShift/VQ不进入V6。
- **结果与失败边界：**D0 U10000在G4 Val1000达到21.033938dB，高于Input18.795938；但LPIPS比官方U0差。不能把局部PSNR改善当四指标全优，也不能与不同数据规模的V5作单因素骨干比较。
- **未解决：**初始额外Clean目标、数据扩展、初始化与骨干各自的作用没有分别隔离；物理递归仍需自己的效果证据。
- **证据：**[当时方法审查及规范映射](../handoff/evidence/v6_diffuir_l_method_review_20261027/METHOD_REVIEW_ZH.md)、[V6实际训练链](../handoff/history/v6_history.md)。

<a id="v6-2"></a>

### V6-2｜A/B没有自动推进到“成功”：暂停和最终停点按用户决定保留

- **当时的理由与改变：**A有暂停后恢复记录；最终用户要求暂停B、继续D0。记录能确认执行取舍，不能把它改写成算法自动判定B无效。
- **结果与失败边界：**A实际6000更新，best U1000四步末级19.861209dB，低于自身第一步20.997425。B实际U9500完整保存；既定best U7000末级21.045838，仍比第一步低0.038261dB。最近full是U9000；计划10000和停止点U9500都不能冒充已完成full。
- **未解决：**未证明多步稳定优于自身第一步；U9500没有完整Val1000终点评价。新负责人不能从旧计划自动补训或补评。
- **证据：**[A/B实际结束与选模说明](../handoff/history/v6_history.md)、[D0续训时的B暂停边界](../handoff/evidence/v6_diffuir_l_20261027/execution_20261028_d0_until_noon/USER_PAUSE_REPORT_ZH.md)。

<a id="v6-3"></a>

### V6-3｜U14927从联合目标切回官方原生残差L1

- **当时的理由与改变：**有明确用户指令要求回到官方单次随机扩散时间的原生训练路径：删除三步Clean训练分支与0.5系数，不改推理S3；目的在于按该定义继续检验D0。记录不支持“已经证明Clean loss导致失败”。RAW/Adam/EMA/RNG全部从旧联合目标U14927继承。
- **结果与失败边界：**原生段到U95544安全停止，G4由21.060202升到21.165142dB，四指标相对该父改善。它是旧模型切目标后的续训，不是从官方U0开始的纯原生对照；训练loss量纲也变了，不能跨切换直接比较降幅。
- **未解决：**新目标与继续训练时长的贡献没有隔离。最终停止是用户明确要求，不代表跑满原18点计划或通过科研成功门。
- **证据：**[直接授权](decision_sources/v6/USER_D0_OFFICIAL_RESIDUAL_RESUME_20261028.md)、[实际终局及用户停止原话](../handoff/evidence/v6_diffuir_l_20261027/execution_20261029_d0_user_stop/D0_OFFICIAL_RESIDUAL_FINAL_REPORT_ZH.md)。

<a id="v6-4"></a>

### V6-4｜吞吐尝试与资源故障：工程变化没有变成新科学配方

- **当时的理由与改变：**为提高端到端速度，短测同B60的micro15×acc1；它比micro5×acc3慢3.039%，未达预设至少10%收益门，因此保留旧配置。另一次外部TurbWNet占用触发排他资源保护，从完整U30500恢复，没有改loss/LR。
- **结果与失败边界：**U30920尾部没有保存成功，420次失败尾计算另计成本，不计入保留状态链。监控曾因串行读日志无共同快照而误触断言，修复读取器，不是训练数学故障。
- **未解决：**低利用率不等于可以增大microbatch提速；历史重放和停机成本不能从最终报告中抹掉。这些修复不解释恢复质量。
- **证据：**[原生段故障恢复、微批短测和监控修复附录](../handoff/evidence/v6_diffuir_l_20261027/execution_20261029_d0_user_stop/D0_OFFICIAL_RESIDUAL_FINAL_REPORT_ZH.md)。

<a id="v6-5"></a>

### V6-5｜G4专训转G1–G4混合，40K准备改成160K

- **当时的理由与改变：**目的改为让既有G4模型适应多档强度；用户明确要求“改成160k的吧，这样匹配一点”。同10000个clean×4个ξ的每组补齐G1–G4，25%等比例，G0.5只验证。不是新增160000个独立clean；父U95544、原生目标和LR保持。
- **结果与失败边界：**到U157505，弱档改善，但G4相对父下降0.274417dB，SSIM/MAE也变差，LPIPS改善；latest未守住父−0.20dB保护线。按既定综合规则保留best U109544，不能把各档峰值拼为一个模型。
- **未解决：**同一模型兼顾弱档和强档的取舍仍在。训练继续至指定截止是用户授权，不等于质量门通过；A/B没有因混强度结果而自动恢复。
- **证据：**[160K与中午检查点的直接授权](decision_sources/v6/USER_D0_MULTISTRENGTH160K_UPDATES_20261029.md)、[最终五档结果与选模](../handoff/evidence/v6_diffuir_l_20261027/execution_20261029_d0_multistrength160k/final_20261030_1800/D0_MULTISTRENGTH_REPORT_ZH.md)。

<a id="v6-6"></a>

### V6-6｜从混强度终点拆出Control／G4-biased，而不是继续盲目延长

- **当时的理由与改变：**要隔离“在当前状态提高G4采样比例”的效果，用户指定同父U157505、两支各5000更新；Control四档均匀，biased将G4提高到50%，其余各1/6。保留网络、目标、LR、B60与推理S3。
- **结果与失败边界：**biased较Control的G4提高0.087707dB，但G1–G3 PSNR与LPIPS付出代价；3k→5k只增0.018622dB，低于冻结0.02趋势要求，部分弱档保护也失败。**追加门未过，额外训练为0。**
- **未解决：**这不是已找到最优比例，也没有自动替换旧综合best；原逐位重启与部分moment检查的FAIL保留，不能写成全部状态逐位相同。没有依据据此承诺进入A/B就能修复强档。
- **证据：**[同父有限对照授权](decision_sources/v6/USER_D0_G4_REWEIGHT_5K_AUTHORIZED_20261030.md)、[5k最终结果与拒绝追加原因](../handoff/evidence/v6_diffuir_l_20261027/execution_20261030_d0_g4_reweight_5k/final_20261031_0255/D0_G4_REWEIGHT_FINAL_REPORT_ZH.md)。

<a id="v6-7"></a>

### V6-7｜从同域分数转向真实图、官方原任务和外部单帧验证

- **当时的理由与改变：**检查模拟湍流内的收益能否迁移、微调是否保留官方能力；这批是冻结模型评价，不是新训练或重新选模。不能将结果事后包装成提前设计的完整因果实验。
- **结果与失败边界：**U95544在OTIS有GT的16序列/4328帧，全部序列PSNR下降；官方13子集诊断中，各子集微调模型的Y-PSNR均值均低于官方EMA，RGB LPIPS均更差；不与湍流RGB PSNR混口径。ATSyn固定单帧999配对则有有限平均改善（Input21.381520，final21.583900dB）。正负结果需同时保留，协议不同不混平均。
- **未解决：**模拟器域差、目标切换、权重遗忘、尺寸/灰度协议各占多少尚未隔离；ATSyn不是DATUM默认多帧协议，Text final全400 OCR仍未完成。OTIS失败不等于所有真实场景都失败。
- **证据：**[V6外部评价与人口边界](../handoff/history/v6_history.md)、[官方13子集完整诊断](../handoff/evidence/v6_diffuir_l_20261027/diagnostics/official_five_tasks_20261030/all13_final_delivery_20261031_0650/FINAL_REPORT_ZH.md)、[固定真实图与机制支线](../handoff/history/extra_branches.md)。

<a id="u-1"></a>

### U-1｜为什么优先试UniDB，而不是直接上PixelDiT＋新控制分支

- **当时的理由与改变：**选型记录明确比较的是适配难度：UniDB已有RGB退化图空间条件、配对恢复目标和官方恢复权重；PixelDiT公开模型主要是类别条件像素生成，需要新设计退化图控制。于是优先核验UniDB原生恢复基线，并保留PixelDiT为投入更大的研究方向。此判断不是湍流性能排名。
- **结果与失败边界：**实际采用官方SR `gamma_1e7` 初始化；首轮不加prompt、Adapter、raw-ξ、A/B或物理递归，不继承V6的loss/Adam。早期建议B60/10k只是准备，后续正式B96/U45411以实际报告为准。
- **未解决：**SR先验能否适配几何湍流、像素生成控制能否更好均未被选型调研证明；不能把后来UniDB的失败倒写成当初已经知道应选另一模型。
- **证据：**[当时UniDB/PixelDiT取舍](decision_sources/v6/COMPARISON_ZH.md)、[首轮原生适配建议及未启动边界](decision_sources/v6/RUN_PLAN_PROPOSED_ZH.md)、[实际UniDB主线](../handoff/history/unidb_history.md)。

<a id="u-2"></a>

### U-2｜D0 U45411停止后先做forensic，而不是把loss下降当作成功

- **当时的理由与改变：**D0按用户提前停止要求保存于U45411，随后针对“原生loss下降但最终恢复变差”做独立审计；停止本身是用户决定。审计0optimizer更新，原权重和STOP封存。
- **结果与失败边界：**完整G4 Input18.795938、U0 18.222398、U45411 15.894991dB；LPIPS改善而PSNR/SSIM下降。所查范围未确认GT反接、DDP重复缩放、Adam错步或全局RGB尺度错误。固定子集mean-ODE只回到输入附近，方差规则干预可挽回部分PSNR仍低于Input，支持权重与随机路径的不良交互。
- **未解决：**没有唯一根因；L1与Gaussian平方风险不同不等于L1已被证明是主因。审计原结论没有证据支持原配方长训，也没有证据要求直接换loss/Adam/backbone；后续继续属于新的用户决策。
- **证据：**[D0历史最终报告](../handoff/evidence/unidb_d0_20261102/FINAL_REPORT_ZH.md)、[forensic结论与反证](../handoff/evidence/unidb_d0_20261102/forensic_audit_20261102/FINAL_FORENSIC_AUDIT.md)。

<a id="u-3"></a>

### U-3｜warmup疑问之后采用3＋5双路线，属于整套配方尝试

- **当时的理由与改变：**用户明确要求“一条继续训练原来的，另外找一个改进的也开始训练”，时间有限，不以吞吐比较为目的。原3卡保持L1/LR1e-5与U45411完整状态；新5卡从官方U0重开epsilon_opt MSE、较低LR、200步warmup、cosine和clip1。
- **结果与失败边界：**独立warmup审查发现官方参考配置本就关闭warmup；其补丁未应用，不能说此前漏做官方步骤。双路线同时改变初始化、loss、LR和卡数，后来改进U10000全量G4已退化，不能凭短monitor峰值或工程PASS宣称成功。
- **未解决：**没有单独warmup或loss因果对照；从U45411后补warmup也不能检验U0初期的反事实。epsilon_opt MSE是有数学动机的候选定义，不是被结果证明的修复。
- **证据：**[用户双路线原话及配方](../handoff/evidence/unidb_dual_20261103_until_noon/protocol/USER_AUTHORITY.json)、[warmup未应用审查](../handoff/evidence/unidb_warmup_review_20261102/REVIEW_ZH.md)、[最终路线历史](../handoff/history/unidb_history.md)。

<a id="u-4"></a>

### U-4｜中午→18点→午夜：为何继续、哪些内容没有重来

- **当时的理由与改变：**两次延期分别来自“继续训练到下午6点”和“今晚24:00”的明确授权；**不是恢复质量达标触发，也不证明尚欠训练。**新root从已验收完整父恢复，保留祖先调度；改进原cosine结束后固定2e-7，不重warmup、不抬LR。
- **结果与失败边界：**原线45411→55760→61764→67180，改进0→20000→30000→40848。恢复工程门更新全部丢弃；旧源/STOP保留。午夜8rank最后更新都在截止前开始，只完成在途更新后保存，没有午夜后新启动的优化器更新。
- **未解决：**更多训练没有建立恢复优势。原best/latest67180的G4较D0父提高0.316792dB仍低于Input2.584155dB；改进latest40848低于自身full-best10000。不能把延期当收敛证明或把best理解为每个更新全量验证后的全局最优。
- **证据：**[18点授权](../handoff/evidence/unidb_dual_20261103_until18/protocol/USER_AUTHORITY.json)、[午夜明确时间与交接规则](../handoff/evidence/unidb_dual_20261104_until00/protocol/USER_AUTHORITY.json)、[实际终验与指标](../handoff/experiment/README_ZH.md)。

<a id="u-5"></a>

### U-5｜最终停训后转为全项目交接

- **当时的理由与改变：**用户明确要求训练结束后整理整个项目并退出当前研究，因此午夜后只做保存、评价、CPU核验、备份和交接，不安排新实验。停止研究不等于证明所有湍流恢复或扩散方法无效。
- **结果与失败边界：**原best＝latest U67180，改进best U10000/latest U40848；3个唯一完整权重和4个角色、全部保存预测与自有进程退出均有最终验收。工程链闭合，但最终两路线五档PSNR均低于Input，未达到预期质量目标。
- **未解决：**真实域泛化、唯一退化原因、一步训练到多步自由轨迹的关系、模拟器适用范围仍是开放问题。接手者先读现存证据再决定研究目标；交接文件不授权复活历史任务。
- **证据：**[用户交接要求原话（当时为准备态）](../handoff/evidence/project_handoff_20261104/HANDOFF_REQUEST.json)、[最终主交接说明](../handoff/documents/MAIN_HANDOFF_FINAL_v2.md)、[实际实验收尾](../handoff/experiment/README_ZH.md)。

<a id="open-issues"></a>

## 未解决问题台账：接手后查什么、缺什么

这里的“未解决”不等于已经证明无法解决。表中建议补做的核实或新实验尚未执行，也没有被自动排进训练队列。新负责人可以只保管成果，或据此另立目标。

| 编号／问题 | 已知事实与不能再混淆的地方 | 接手者先查哪里／还缺哪种证据 |
|---|---|---|
| **O01 · 物理递归的稳定收益** | v3可拟合训练人口；V5 B胜过独立D0，但B第四步低于自身第一步；V4多配置末级回退。尚未证明在未见来源上多步稳定优于单步，也未证明所有递归都无效。 | 查[v3](../handoff/history/legacy_v1_v3.md)、[V4](../handoff/history/v4_history.md)、[V5](../handoff/history/v5_history.md)逐步表。若继续，需固定数据、初始化、训练曝光和评价人口，同时比较自身第一步、匹配D0与各后续步；不能只报B对另一个D0的均值。 |
| **O02 · raw ξ是否是合适的预测表示** | 有限定算子的非单射结构和局部clean–ξ补偿证据；Raw–Field小集合只呈现混合指标结果。没有自然图像全局唯一性或不可恢复性证明。 | 查[V4可辨识性与Raw–Field记录](../handoff/history/v4_history.md)。先确认raw坐标、有效field及PSF语义；若继续，只检验明确算子、人口和恢复目标，不用局部witness替代完整恢复结果。 |
| **O03 · Simulator已接收但并非所有门通过** | V5近零/固定差分原FAIL保留；研究接收支持在记录范围内继续，不是完整物理模型认证。非平稳协方差、raw冗余及少量越界等仍有局限。 | 先读[研究接收原文](../handoff/evidence/v5_v1_simulator_20261020/acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md)。新范围若涉及分辨率、参数或任意预测ξ，需独立验证；不能改阈值后覆盖旧FAIL。 |
| **O04 · V5最初失稳的唯一原因** | 现场、工程修复和隔离对照均保留；恢复后健康不等于已找到首次异常更新。短面板的原LR/半LR/四分之一LR未复发，不能称降LR已证实修复。 | 查[V5故障与成本记录](../handoff/history/v5_history.md)。若研究因果，需要从同一异常前完整状态重演并定位最早分叉；现有证据不支持指定某一个LR、骨干或模拟器为唯一原因。 |
| **O05 · V6真实域细节丢失和能力退化** | OTIS负结果、官方13子集退化与ATSyn单帧有限正结果并存。固定20图受控检查未见RGB/BGR、范围、EMA或保存错误；按训练预处理复跑后ATD文字/杆件仍被抹掉。 | 查[V6外部评价](../handoff/history/v6_history.md)及[真实图诊断](../handoff/history/extra_branches.md)。缺同协议、完整人口下能维持结构且不过度遗忘的证据；固定少量图片和4帧OCR不替代未完成的全400文字OCR。 |
| **O06 · UniDB退化究竟由什么主导** | 当前活动loss与所查官方实现一致；未确认GT接反、DDP缩放、Adam错步或粗大RGB尺度错误。固定干预支持微调权重×随机路径的不良交互，但未隔离唯一根因；mean-ODE主要接近输入，非成功恢复。 | 查[独立forensic总报告](../handoff/evidence/unidb_d0_20261102/forensic_audit_20261102/FINAL_FORENSIC_AUDIT.md)。若继续，需从同一初始化、数据、预算做能区分目标/采样/优化的最小控制；3＋5综合路线不能回答单一因素的因果份额。 |
| **O07 · 看着锐利却指标差** | PSNR/SSIM/MAE与LPIPS不总同向；错位、亮度偏差或错误纹理可同时产生清晰观感和较低PSNR。原UniDB最终LPIPS优于Input，但其余三项差；这是取舍，不是全面成功。 | 查[最终四指标及固定图](../handoff/experiment/README_ZH.md)。先核同Input/GT、EMA、seed、浮点范围；显示clip01不等于指标也clip。不能以恢复MAE补画未记录的同目标val loss。 |
| **O08 · 公平比较与泛化证据** | 版本间数据、初始化、目标和预算经常同时变；V6/UniDB所述完整验证每G1000含250来源×4实现，并非1000独立内容；monitor和反复用于选择的heldout不是新Test。 | 查[数据登记](../handoff/registers/DATA_REGISTER.csv)、[实验登记](../handoff/registers/EXPERIMENT_REGISTER.csv)。继续前先锁目标人口与比较协议，按source做相关性处理；未知真实域或锁定Test结果就标未评，不能补造。 |
| **O09 · 跨机器能否逐位复现** | 源码/权重SHA、完整状态和部分重放验收已有；同seed跨ARM/x86随机流或重建simulator buffer曾不一致。环境有.pth继承链，未形成覆盖全部实验的驱动、wheel和系统镜像。 | 查[操作指南第3—5节](../handoff/history/OPERATIONS_GUIDE_ZH.md)、[源码快照计划](../provenance/SOURCE_SNAPSHOT_PLAN.json)的gaps。先固定canonical参考、解释器/模块路径、环境与分支身份，再在新环境做有限一致性验证，不能承诺一份requirements就逐位复现。 |
| **O10 · 旧资产能否取回** | 本次UniDB三个唯一完整状态已有本机全SHA备份；不等于所有旧权重都备份。A100五项当前未核，不能说丢失；部分旧清理checkpoint未找到归档去向，不能保证恢复。 | 查[资产登记](../handoff/registers/ASSET_REGISTER.csv)、[权重登记](../handoff/registers/WEIGHT_REGISTER.csv)、[归档说明](../handoff/history/OPERATIONS_GUIDE_ZH.md)。按原路径/receipt逐项交接，不必问原负责人猜位置；找不到已建立去向的项目保留“不可保证恢复”。 |
| **O11 · 论文主张与第三方资产边界** | “条件扩散用于单帧湍流”已有先例；机制不同不等于效果或新颖性成立。部分上游vendor/许可与依赖未纳入本仓库。 | 查[AT-DDPM重合记录](../handoff/history/extra_branches.md)、[第三方说明](../THIRD_PARTY_NOTICES.md)。若发表，需针对最终实际方法重审主张及许可；本交接没有新颖性放行或统一开放许可证。 |
| **O12 · 部分历史转向动机未留下书面记录** | 具体新方法、实际授权和结果通常可查，但并非每次都有足以证明唯一动机的决定书。本页明确标缺口，不将后来的失败自动当作当时原因。 | 按本页节点链接查协议与历史稿。若只有方法变更记录，接手时就以“决定已发生、唯一理由未记录”作为事实边界；无需向原负责人索要一个事后解释来补成确定结论。 |

<a id="handoff-faq"></a>

## 接手者常见问题：直接答案与查找入口

**现在接的是一个可以直接部署的恢复模型吗？**

不是已证明跨真实场景有效的成品。可交接的是实现、数据与状态身份、有限正结果及负结果。最新UniDB已完整收尾，但未达到恢复质量目标；当前没有继续训练的自动任务。[当前总状态](../README.md)

**应该拿哪个best，为什么有多个best？**

先定版本和评价协议。最新UniDB原线best/latest都是U67180；改进线best是U10000、latest是U40848。它们只在既定full候选池选取，monitor峰值不自动成为full-best，也不是全项目跨版本冠军。[角色与三个完整文件](../handoff/experiment/README_ZH.md)

**代码、数据、权重到底在哪，为什么clone之后还不能训练？**

代码在`source_snapshot/`，版本与依赖缺口在[SOURCE_SNAPSHOT_PLAN.json](../provenance/SOURCE_SNAPSHOT_PLAN.json)；数据、完整权重、预测大文件通过[NOT_INCLUDED](../handoff/NOT_INCLUDED.md)、[DATA_REGISTER](../handoff/registers/DATA_REGISTER.csv)、[WEIGHT_REGISTER](../handoff/registers/WEIGHT_REGISTER.csv)、[ASSET_REGISTER](../handoff/registers/ASSET_REGISTER.csv)定位本机/服务器原路径。只有登记不代表已经复制进Git，stat存在也不等于载荷全SHA验收。访问权限由接手者取得，不在仓库存放密码或私钥。

**哪些失败已经修了，哪些不能算修了？**

v1.5评价seed/comparator身份已纠偏，原invalid结果仍保留；训练平台、恢复和最终交付的工程问题按收据验收。V5原Simulator门仍有FAIL；V5首个失稳原因、V6真实域能力、UniDB强档恢复没有因工程PASS而变成“已修复”。“修复实现/评价”与“达到恢复目标”分开查。

**多训练、加warmup、换loss是否就能解决？**

现有结果不能支持这个承诺。UniDB官方所查SR参考warmup关闭；后来另一路同时改起点、目标、日程和clip，最终仍差。延长到中午、18点、午夜是用户批准的训练时限，不是质量门通过证明。不同loss的数值也不处于同一量纲。[warmup原审查](../handoff/evidence/unidb_warmup_review_20261102/REVIEW_ZH.md)

**哪些东西不能再拿来当证据？**

旧README中的RUNNING/预计步数、只完成monitor的权重、部分full、准备稿和工程门都不能替代最终质量表。v2未正式训练、v2.5 U204没有完整终评、V6 B U9500没有full SUMMARY、V4部分停点没有full；按各节点已完成的真实人口说明。训练后固定图是特定checkpoint重放，不是训练过程截图。

**我先做什么，才不依赖原负责人？**

先读首页和本页，选具体实验登记行→最终报告→数据/环境/权重身份→已有曲线和固定图；运行仓库的离线完整性校验即可核资料字节。需要继续研究时，由新负责人明确一个未决问题与新协议，另建目录并核必要大资产；不要直接启动历史launcher、按旧PID操作或自动补跑Test。所有未执行项在本页保留“缺什么证据”，而非默认为旧任务待恢复。

## 本次补充的覆盖边界

本页覆盖版本转向、主要训练目标/数据/骨干/平台变更及关键诊断分叉；逐个实验的全部运行状态继续以[EXPERIMENT_REGISTER](../handoff/registers/EXPERIMENT_REGISTER.csv)为准。新增文字只重建现有记录能支持的决策过程，不改写历史资产，也不把未记录的私人动机伪装为事实。若继续维护，请新增有日期和证据的记录，保留既有失败与未知项。

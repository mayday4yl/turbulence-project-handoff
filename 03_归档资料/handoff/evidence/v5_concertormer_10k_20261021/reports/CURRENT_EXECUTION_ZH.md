完成验收（2026-10-26 00:32，北京时间）：直接恢复基线10000、连续时间单步预训练6000、四步递归对齐4000更新均完成；最终Val1000与完整checkpoint验收通过，控制器子任务exit0、训练进程退出且NPU2–7已释放。三阶段PSNR-best依次为U8250/21.443506、U5000/20.156652、U2750/21.736105dB；后两阶段以四步输出计。统一Val1000汇总已完成，授权隔离对照已完成且未部署，v5小时监控已删除。refinement OFF、Test锁定；未声明历史递归失稳根治。见 reports/alignment_A6k_B4k_20261022/UNIFIED_VAL1000_FINAL_ZH.md 和 reports/heartbeat_20261026_0032/AUTOMATION_STOP_RECEIPT.json。

最新巡检（2026-10-25 23:31，北京时间）：正式递归对齐U3936/4000正常，六rank guard无新abort，梯度及显存峰值稳定；最新完整Val/ckpt仍U3750、best U2750。预计剩约48分钟，约10月26日00:19完成。配置保持，见 reports/heartbeat_20261025_2331/REPORT_ZH.md。

最新巡检（2026-10-25 22:28，北京时间）：正式递归对齐U3843/4000正常，六rank guard无新abort，梯度及显存峰值稳定；最新完整Val/ckpt仍U3750、best U2750。预计剩1.86小时，约10月26日00:19完成。配置保持，见 reports/heartbeat_20261025_2228/REPORT_ZH.md。

最新巡检（2026-10-25 21:27，北京时间）：正式递归对齐U3753/4000正常，新完成U3750完整Val1000 PSNR21.718557，best仍U2750；新checkpoint已CPU完整回读，六rank guard无新异常、显存峰值稳定。预计剩2.87小时，约10月26日00:20完成。配置保持，见 reports/heartbeat_20261025_2127/REPORT_ZH.md。

最新巡检（2026-10-25 20:27，北京时间）：正式递归对齐U3671/4000正常，六rank guard无新abort，梯度及显存峰值稳定；最新完整Val/ckpt仍U3500、best U2750。预计剩3.91小时，约10月26日00:22完成。配置保持，见 reports/heartbeat_20261025_2026/REPORT_ZH.md。

最新巡检（2026-10-25 19:26，北京时间）：正式递归对齐U3583/4000正常，新完成U3500完整Val1000 PSNR21.722656，best仍U2750；新checkpoint已CPU完整回读，六rank guard无新异常、显存峰值稳定。预计剩4.92小时，约10月26日00:22完成。配置保持，见 reports/heartbeat_20261025_1926/REPORT_ZH.md。

最新巡检（2026-10-25 18:24，北京时间）：正式递归对齐U3498/4000正常，六rank guard无新abort，梯度及显存峰值稳定；最新完整Val/ckpt仍U3250、best U2750。预计剩5.95小时，约10月26日00:20完成。配置保持，见 reports/heartbeat_20261025_1823/REPORT_ZH.md。

最新巡检（2026-10-25 17:24，北京时间）：正式递归对齐U3412/4000正常，六rank guard无新abort，梯度及显存峰值稳定；最新完整Val/ckpt仍U3250、best U2750。预计剩6.91小时，约10月26日00:19完成。配置保持，见 reports/heartbeat_20261025_1724/REPORT_ZH.md。

最新巡检（2026-10-25 16:28，北京时间）：正式递归对齐主快照U3328，随后确认U3329/4000正常。新完成U3250完整Val1000 PSNR21.722826，best仍U2750；新checkpoint已CPU完整回读，六rank guard无新异常、显存峰值稳定。预计剩7.77小时，约10月26日00:14完成；配置保持，见 reports/heartbeat_20261025_1628/REPORT_ZH.md。

最新巡检（2026-10-25 15:24，北京时间）：正式递归对齐U3241/4000正常，六rank guard无新abort，梯度及显存峰值稳定；最新完整Val/ckpt仍U3000、best U2750。预计剩9.19小时，约10月26日00:36完成。配置保持，见 reports/heartbeat_20261025_1524/REPORT_ZH.md。

最新巡检（2026-10-25 14:29，北京时间）：正式递归对齐U3162/4000正常，六rank guard无新abort、梯度及显存峰值稳定；最新完整Val/ckpt仍U3000、best U2750。预计剩10.15小时，约10月26日00:38完成。配置保持，见 reports/heartbeat_20261025_1429/REPORT_ZH.md。

最新巡检（2026-10-25 13:28，北京时间）：正式递归对齐U3075/4000正常，新完成U3000完整Val1000 PSNR21.721940，best仍U2750。U3000 checkpoint已CPU完整回读，六rank guard无新异常，显存峰值稳定。预计剩11.22小时，约10月26日00:41完成；配置保持，见 reports/heartbeat_20261025_1328/REPORT_ZH.md。

最新巡检（2026-10-25 12:24，北京时间）：正式递归对齐U2992/4000正常，六rank guard无新abort，梯度与显存峰值稳定；最新完整Val/ckpt及best均仍U2750。预计剩11.79小时，约10月26日00:12完成。配置保持，见 reports/heartbeat_20261025_1224/REPORT_ZH.md。

最新巡检（2026-10-25 11:29，北京时间）：正式递归对齐U2911/4000正常，六rank guard无新abort，梯度与显存峰值稳定；最新完整Val/ckpt及best均仍U2750。预计剩12.48小时，约10月25日23:58完成。配置保持，见 reports/heartbeat_20261025_1129/REPORT_ZH.md。

最新巡检（2026-10-25 10:27，北京时间）：正式递归对齐U2817/4000正常，新完成U2750完整Val1000 PSNR21.736105，best更新至U2750（较原best仅+0.002734dB）。新checkpoint已CPU完整回读，六rank guard无新异常、显存稳定。预计剩13.60小时，约10月26日00:03完成；配置保持，见 reports/heartbeat_20261025_1026/REPORT_ZH.md。

最新巡检（2026-10-25 09:25，北京时间）：正式递归对齐U2733/4000正常，六rank guard无新abort，梯度与显存峰值稳定；最新Val/ckpt仍U2500、best U1000。预计剩14.94小时，约10月26日00:22完成。配置保持，无新诊断或训练，见 reports/heartbeat_20261025_0925/REPORT_ZH.md。

最新巡检（2026-10-25 08:18，北京时间）：正式递归对齐U2634/4000正常，六rank guard无新abort，梯度与显存峰值稳定；最新Val/ckpt仍U2500、best U1000。预计剩16.14小时，约10月26日00:27完成。配置保持，无新诊断或训练，见 reports/heartbeat_20261025_0817/REPORT_ZH.md。

最新巡检（2026-10-25 07:15，北京时间）：正式递归对齐U2543/4000正常，新完成U2500完整Val1000 PSNR21.726726（较输入+2.930788），best仍U1000。U2500 checkpoint已CPU完整回读，六rank guard无新异常、显存稳定；预计剩17.17小时，约10月26日00:26完成。配置保持，见 reports/heartbeat_20261025_0715/REPORT_ZH.md。

最新巡检（2026-10-25 06:16，北京时间）：正式递归对齐U2464/4000正常，六rank guard无新abort，梯度与显存稳定；最新Val/ckpt仍U2250、best U1000。预计剩17.93小时，约10月26日00:12完成。配置保持，无新诊断或训练，见 reports/heartbeat_20261025_0615/REPORT_ZH.md。

最新巡检（2026-10-25 05:20，北京时间）：正式递归对齐U2382/4000正常，六rank guard无新abort，梯度与显存峰值稳定；最新完整Val/ckpt仍U2250、best U1000。预计剩18.77小时，约10月26日00:07完成。配置保持，无新诊断或训练，见 reports/heartbeat_20261025_0520/REPORT_ZH.md。

最新巡检（2026-10-25 04:18，北京时间）：正式递归对齐U2290/4000正常，新完成U2250完整Val1000 PSNR21.705290（较输入+2.909352），best仍U1000。U2250 checkpoint已CPU完整回读，六rank guard无新异常，显存峰值保持；预计剩19.89小时，约10月26日00:12完成。配置不变，见 reports/heartbeat_20261025_0418/REPORT_ZH.md。

最新巡检（2026-10-25 03:19，北京时间）：正式递归对齐U2210/4000正常，六rank guard无新abort、梯度有限。张量峰值稳定，框架保留峰值仅较上轮增加2MiB，未见OOM或明确泄漏，配置保持。最新Val/ckpt仍U2000、best U1000；预计剩21.13小时，约10月26日00:28完成。见 reports/heartbeat_20261025_0319/REPORT_ZH.md。

最新巡检（2026-10-25 02:20，北京时间）：正式递归对齐U2125/4000正常，六rank guard无新abort，梯度与显存稳定；最新完整Val/ckpt仍U2000、best U1000。预计剩21.65小时，约10月26日00:00完成。配置保持，无新诊断或训练；见 reports/heartbeat_20261025_0220/REPORT_ZH.md。

最新里程碑（2026-10-25 01:18，北京时间）：递归对齐U2000完整Val1000及checkpoint已完成，已自动继续到U2031/4000。PSNR21.732800（较输入+2.936862），best仍U1000=21.733371；四步21.843081→21.787927→21.755634→21.732800。U2000 checkpoint已CPU完整回读，六rank guard无新异常、显存稳定，配置保持。预计剩22.92小时，约10月26日00:13完成。见 reports/heartbeat_20261025_0117/REPORT_ZH.md及reports/alignment_A6k_B4k_20261022/milestones/B_U002000_ZH.md。

最新巡检（2026-10-25 00:13，北京时间）：正式递归对齐U1944/4000正常，六rank guard无新abort，梯度与显存稳定；最新完整Val/ckpt仍U1750、best U1000。预计剩23.78小时，约10月26日00:01完成。配置保持，未重跑诊断；见 reports/heartbeat_20261025_0013/REPORT_ZH.md。

最新巡检（2026-10-24 23:08，北京时间）：正式递归对齐U1847/4000正常，六rank guard无新abort，梯度与显存稳定；最新完整Val/ckpt仍U1750，best U1000。预计剩24.99小时，约10月26日00:08完成。配置保持，无新训练或诊断，见 reports/heartbeat_20261024_2307/REPORT_ZH.md。

最新巡检（2026-10-24 22:12，北京时间）：正式递归对齐U1761/4000正常，U1750完整Val1000 PSNR=21.719547、best仍U1000=21.733371。U1750完整ckpt已CPU回读确认optimizer/scheduler/sampler及六rank RNG；六rank guard无新abort、显存稳定，配置保持。预计剩25.98小时，见 reports/heartbeat_20261024_2210/REPORT_ZH.md。

最新巡检（2026-10-24 21:07，北京时间）：正式递归对齐U1673/4000正常，六rank guard无新abort，显存与上一轮一致。最新Val/ckpt仍U1500、best U1000；预计剩27.18小时。配置保持，无新训练/诊断，见 reports/heartbeat_20261024_2105/REPORT_ZH.md。

最新巡检（2026-10-24 20:11，北京时间）：正式递归对齐U1590/4000正常，六rank guard无新abort、显存稳定；最新Val/ckpt仍U1500、best U1000。预计剩27.97小时；配置保持，不重跑已完成隔离。见 reports/heartbeat_20261024_2008/REPORT_ZH.md。后续可复用本地 reports/monitoring_tools/collect_readonly.py 经SSH stdin从远端实验根目录运行，只读采集，不分配NPU或写服务器。

最新巡检（2026-10-24 19:10，北京时间）：正式递归对齐U1501/4000正常，U1500完整Val1000 PSNR=21.733024、best仍U1000=21.733371。U1500完整ckpt已CPU回读核对optimizer/scheduler/sampler和六rank RNG。六rank guard无新abort，第二次验证后显存峰值仍稳定，配置保持；预计剩29.23小时。见 reports/heartbeat_20261024_1908/REPORT_ZH.md。

最新巡检（2026-10-24 18:02，北京时间）：正式递归对齐U1409/4000正常，六rank guard无新abort；显存与上一小时相同，保持设置。最近完整Val/ckpt仍U1250，best U1000；剩余约30.46小时。隔离对照已完成且未部署、不重跑。见 reports/heartbeat_20261024_1801/REPORT_ZH.md。

## 2026-10-24 17:09：显存保持配置，隔离对照已完成

正式递归对齐主快照U1323、随后U1326正常；U1250首次恢复后完整Val1000与checkpoint均完成，PSNR21.691619，best仍U1000=21.733371。CPU回读新checkpoint确认model/optimizer/scheduler/sampler及六rank状态完整。六rank guard无新abort。

用户要求显存没有问题就不修改。整卡约63.3/64GiB，账面余量约0.70GiB偏紧；张量峰值54.98GiB、框架保留峰值57.76GiB，U1260–1320及短采样稳定，无OOM或持续增长证据。保持训练设置，不额外占用NPU；风险不为零，继续监控趋势。详见reports/heartbeat_20261024_1703/REPORT_ZH.md。

根因隔离5个worker全部exit0完成，91.07分钟，3分支各3次虚拟更新全部丢弃，正式未改。原LR/半LR/四分之一LR在一次纠正+两次独立更新后均未在预定小面板复发，没有降LR优越证据；warmup/长期预防未验证。主干权重交叉支持主干变化参与递归放大，不能归为仅time/raw，也未定位历史最早致因update。完整结论reports/gradient_spike_audit_20261024/root_cause/isolation_followup_20261024/CONCLUSION_ZH.md。禁止重复启动旧isolated_lr_probe和isolation_followup；不自动部署混合权重、改LR或扩诊断预算。当前预计正式剩约32.7小时，后续现场刷新。

---

巡检补记（2026-10-24 16:01，北京时间）：正式递归对齐U1240正常，六rank guard无新abort，U1000完整ckpt哈希一致；剩余约32.68小时。隔离gradient已完成，mechanism仍运行，候选纠正持续性分支尚待依赖；无新最终结论。详见 reports/heartbeat_20261024_1557/REPORT_ZH.md。正式run及超参未改，勿重复启动。

## 2026-10-24 15:32：已启动后续失稳机制与复发隔离对照

用户进一步授权继续查清训练进入失稳及如何避免复发，方法/Loss保持，正式训练先别改。独立CPU控制器PID1192107已启动，状态 reports/gradient_spike_audit_20261024/root_cause/isolation_followup_20261024/WORKFLOW_STATUS.json；脚本/预定方案 repair_20261024/isolation_followup/PROTOCOL_ZH.md。不得重复启动或将候选部署到正式run。旧单步学习率探针已经完成，不要重新运行。 15:36首个机制前向已完成并精确复现旧CPU坏例（total34720.847656、末步clean峰值77947.578125）；正式B现场U1208，loss0.04930、preclip0.16610。冻结88源码SHA全一致。预估完整隔离对照2–3小时，实际以各worker RESULT与exit为准。

先并行计算固定坏例Full-BPTT梯度和A/B750模块权重交叉前向；再自动并行原LR、半LR、四分之一LR分支：一次已知例纠正+两次相同独立微批更新，检查已知例、同图未见时间、未参与更新的样本是否复发。另测无当前梯度而保留AdamW历史动量的反事实，避免把旧动量作用误判为当前梯度修复。仅CPU FP32、微批2，训练函数及Loss原样；不是六卡B48/BF16重放，也不是warmup预防验证。最多48CPU线程，每worker硬限2小时、共最多9次丢弃的虚拟更新；Test锁定。三个必要checkpoint已硬链接保留并核对SHA，避免滚动清理；U250/U500和逐步权重不存在，不能断言恢复出历史最初失稳更新。

监控继续优先保护正式B：仍原六卡/原超参受guard保护；同时读取隔离WORKFLOW_STATUS、各worker的STATUS/RESULT及log。只有退出码0且RESULT存在才算该worker完成，部分失败不得宣称全实验成功；有界诊断失败保存证据，不自动重复大试验。全部完成后汇总“进入机制”和“纠正持续性”的实际证据及局限，通知用户，不自动部署。正式训练不因该对照暂停或被修改。

---

## 2026-10-24 15:22：学习率隔离对照完成，未部署

CPU隔离诊断已完成（1596.76秒，exit0）。同一固定U750坏例、同一裁剪梯度和原AdamW状态，各一次虚拟更新：原LR、半LR、四分之一LR均将末步clean峰值从77947.58降到约1.01，总loss从34720.85降到约0.08。原LR也有效，不能归因于降LR，更不能声称长期递归失稳已解决；warmup未测试。这是两个参与更新的Train样本上的CPU FP32局部诊断，不是六卡B48正式验收。所有虚拟更新丢弃，正式run暂不改，禁止自动部署或重复跑。15:21现场正式训练U1186/4000。详见reports/gradient_spike_audit_20261024/root_cause/ISOLATED_LR_CONCLUSION_ZH.md及ISOLATED_LR_RESULT.json；远端原始RESULT位于同目录isolated_lr_probe/RESULT.json。

---

## 2026-10-24：根因诊断进展与隔离对照授权（不修改正式训练）

最新用户要求根因修复而非仅加拦截，并已回复“允许隔离对照，正式 run 暂不改”。CPU64/官方原生 Conv3d 完整四步已复现固定 B U750 坏例：clean 峰值6.965→56.786→520.958→77901.342，末步回投MAE2777517。初次网络已有局部异常，后续物理核出现巨大正负系数；未找到等价算子修复，不能将本次称作根治。详见 reports/gradient_spike_audit_20261024/root_cause/ROOT_CAUSE_ZH.md。

正式 B 仍沿用下方已验收受监控恢复协议，不能重复启动或改学习率/optimizer。最近核对 U1126/4000 正常，88份冻结源码无差异；实际进度现场刷新。新隔离诊断在 CPU，脚本 repair_20261024/root_cause/isolated_lr_probe.py，状态 reports/gradient_spike_audit_20261024/root_cause/isolated_lr_probe/STATUS.json。最多35分钟，原Loss/Full-BPTT，固定两个Train样本的同一梯度与AdamW状态对比原步长/半步长/四分之一步长，各一次虚拟更新全部丢弃。CPU FP32且仅微批诊断，不得冒充六卡B48/BF16验收或warmup预防结论；完成后先读RESULT.json，不能自动部署或重复跑。许可详见同目录 ISOLATED_LR_AUTHORIZATION.json。

用户询问结果为何无refinement：继续遵守常规验证OFF、Test锁定。D0 clean-only不适用raw refinement；A/B最终ON评价尚需确定推理超参，不能为掩盖失稳擅自开启。

---

## 2026-10-24 13:16后：已验收受监控恢复（优先于下方历史暂停记录）

用户最新明确要求“继续修复，恢复训练”。已完成U1000三历史窗口720微批/1440次输入复查，最大total loss0.08777，无大值异常；六卡原完整状态临时U1001前后向更新检查通过，模型/optimizer全量有限且六rank一致，跨步梯度有效，临时更新已丢弃。修复的是有限巨大值被遗漏的监控漏洞，不宣称旧U750异常映射已彻底治愈。科学公式、88份冻结代码、模型/Loss/精度/Full-BPTT与预算保持原样。

**实际正式训练已恢复到U1003/4000**，现场loss0.049039、preclip梯度0.097090，最近一步41.13秒；controller4001534、torchrun4003914，六rank4003975–4003980。后续状态必须现场刷新。直接恢复基线及连续时间单步预训练已完成，禁止重跑。

新controller为repair_20261024.guarded_resume.resume_controller，runner为repair_20261024.guarded_resume.guarded_runner。运行凭据repair_20261024/guarded_resume/FORMAL_UPDATE_CONFIRMED.json，恢复验收RESUME_ACCEPTANCE.json，正式run仍runs/alignment_A6k_B4k_20261022/B。原PAUSE_RECEIPT保留作历史，不再代表当前暂停，禁止监控重复启动或误杀恢复任务。model/optimizer/scheduler/六rank RNG/采样从U1000完整恢复；未提交1001–1220日志已归档至reports/alignment_A6k_B4k_20261022/recovery/B/1792818823275497228，220次尝试的8968.07秒训练成本另存guarded_resume_20261024，不扩大4000有效update预算。

新增只暂停的guard：任一micro total>1、clean>1、observation>10或loss非有限，在backward前全rank退出；原preclip norm>1000或非有限，在optimizer.step前退出。原clip=1不变，不clamp、不跳样本、不换随机路径、不改变loss或gradient。新guard异常有GUARD_ABORT_NO_AUTOMATIC_RETRY凭据，不得自动重试或放宽阈值。普通I/O短暂错误才沿用完整checkpoint有限重试。guard阈值不是物理/数值Gate阈值，也不把历史失败改PASS。

六卡2–7、B48、BF16网络/FP32 Simulator、日志10/完整Val1000及checkpoint250保持。Test锁定、refinement OFF。暂估剩35–37小时（恢复初期约41秒/update加已测验证/保存成本），后续按正式日志修正。v5监控保持每小时，服务器guard在每次更新前实时检查。详见reports/gradient_spike_audit_20261024/repair_20261024/REPAIR_AND_RESUME_ZH.md。

---


## 2026-10-24 梯度异常排查暂停（优先于下方历史运行状态）

本轮异常暂停不撤销用户此前自主排查和修复的授权。下一轮应先定位随机路径的首个异常张量；只有在保持既定方法、超参、精度和完整恢复状态且有正确性证据时，才能恢复原预算训练。不能仅凭后续finite正常就直接恢复，也不能静默修改Loss或截断梯度。

四步递归对齐训练在 U280/U620/U790 日志窗口出现已核实的有限但巨大 loss/梯度尖峰，最高裁剪前梯度约1.277646e11。按用户异常协议暂停本项目控制器和其训练子进程，禁止监控自动恢复或视为临时 I/O 故障。先阅读 reports/gradient_spike_audit_20261024/PAUSE_RECEIPT.json。模型、Loss、精度和预算未改变；日志尾部完整保留，最近完整可恢复 checkpoint 为 U1000（含 optimizer/scheduler/RNG/sampler）。待原因及有效性核实后再决定恢复。此次是发现历史异常后的安全暂停，不是新的NaN/OOM，也不能把 anomaly=null 当全程稳定证据。

### 2026-10-24 12:22后：已复现前向发散，继续暂停

本轮隔离诊断固定B U750权重，复用U781–790的240个微批/480个不同Train样本及原随机时间，在六卡前向扫描中复现1个巨大值微批；具体为rank5/micro3、数据时间标签U783、第二张train_s0166_r03。首次网络输出clean范围−3.30～6.06、raw-Zernike−34.25～36；随后网络/物理递归放大，末步该样本G4回投MAE约1.16e7。仅前向、无backward/optimizer也可重复出现，排除单纯梯度统计错误。相同样本/时间/固定U750的FP32网络对照仍发散（回投MAE约2.78e6），不能用切FP32作为已验证修复。U1000在同一例的BF16回投MAE约0.03674正常，但只是一例，不足以解除暂停。固定U750不等于历史逐更新重放，不能认定783是历史真正触发步；尚需定位第一层异常及原Full-BPTT/裁剪后状态。

无正式update、无科学配置改动，原88文件哈希一致；六卡诊断150.16秒、单卡对照49.26秒，结束已释放卡2–7，0/1外部任务未触碰。监控提示已更新，禁止把本次暂停误当临时I/O而自动重启。详细证据 reports/gradient_spike_audit_20261024/FORWARD_REPRODUCTION_ZH.md。恢复点仍U1000，1001–1220日志尾部保留，旧ETA无效。

# V5 当前执行协议：D0 best → A 6000 → B 4000

2026-10-22 用户最终执行指令覆盖旧的 D0 后直接进行10k R0-D0Init方案。原文保存在 protocol/ALIGNMENT_USER_DECISION_20261022_ZH.md；机器计划和授权为 protocol/ALIGNMENT_A6K_B4K_20261022.json、protocol/ALIGNMENT_AUTHORIZATION_20261022.json。用户已要求离开后继续推进、自主修复并完成真实训练，不等待逐阶段批准。

- 保护当前D0：runs/clean_only_D0_then_R0D0Init_B48_u250/D0，最多10000更新；不改模型、Loss、Optimizer、数据和schedule，不中断。按最终完整Val1000选择best并冻结身份、配置和曲线。
- 新run：runs/alignment_A6k_B4k_20261022/{A,B}；新控制器状态：reports/alignment_A6k_B4k_20261022/workflow_STATUS.json；准备目录：preparation/alignment_20261022。
- 旧控制器已禁止启动后继作业：原计划已备份并添加后继废止标记，启动时固定的计划SHA与新文件不同，其现有保护将在D0自然结束后阻止旧后继子进程。没有向D0或旧控制器发送终止信号。证据 reports/alignment_A6k_B4k_20261022/OLD_SUCCESSOR_DISABLED.json。旧控制器随后出现 Execution plan changed while workflow was active 属于预期退役，不代表D0训练失败；检查新控制器和D0本身状态。
- A：D0最终best继承Backbone及RGB，新时间调制输出零初始化和固定seed raw-ξ头；检查初始clean映射与D0一致。六卡2–7，micro2×accum4=B48，BF16网络、Simulator/Loss FP32、HF32 OFF、等价PSF反传、AdamW、wd1e-4、clip1、EMA OFF；旧层LR5e-5、新层1e-4，warmup300和6k cosine。
- A每样本单次预测，Loss=0.9 MAE(clean,C)+0.1 MAE(F(clean,rawξ,g),Yg)，使用当前g及Yg；无四步展开、跨强度损失或GTξ监督，Simulator反向保留。
- A数据：原Train8000，同一真实ξ生成物理中间态，25% Y4/t0和75%内部态；每轨迹10个等宽时间层各随机1点。每版最多80000条无损FP32缓存、版本和种子可复现。分段推进到2000/4000/6000只为分批生成新缓存，同一Optimizer/Scheduler/RNG/采样状态续训，没有审批停点。生成不和六卡主训练竞争NPU。
- B：继承A best完整模型及已经训练的时间/raw头；新Optimizer/Scheduler，旧层LR2.5e-5、新层5e-5，warmup100和4k cosine。Y4起点，严格三个随机内部时间，四次共享网络调用和预测物理中间态，完整Full-BPTT，无detach或identity/surrogate梯度。Loss为四stage平均的0.9 Clean MAE+0.1 MAE(F(clean,rawξ,4),Y4)。
- 审计Train/Val/Test来源、类别、重复与泄漏、预处理、realization及物理身份；Test只做数据审计，模型评价锁定。原Val1000不替换；新增固定Val-A使用同250源/1000realization和A时间分布，只作单步诊断。
- 日志10；完整验证及可恢复checkpoint250；保留best/latest、最近2份rolling及1000步里程碑。A每次运行完整Val-A单步和原Val1000真实均匀四步；B运行原Val1000四步。A/B best均按原Val1000最终四步PSNR，refinement OFF；Val-A不替代实际递归指标。
- 自动顺序：D0 best冻结→审计→Val-A/首版缓存→必要正确性检查→A6k→A best→B4k→B best→D0/A/B统一Val1000分析。阶段交接不等待回复，不超预算；直接共同U0 R0和旧10k R0-D0Init不启动。
- 临时设备/I/O故障保留证据并优先完整checkpoint续训；无法恢复的OOM、持续NaN/Inf、梯度错误、checkpoint损坏或泄漏停止并报告。不改方法/随机种子，不结束外部进程。

2026-10-23 01:04 更新：D0 已完成10000更新，最终 best U8250（Val1000 PSNR21.443506dB）。A首版80000物理内部态及固定Val-A1000已就绪，原Y4全量重放最大差0；D0→A实际初始化FP32/BF16首步clean最大差0。首次预检环境选项读取兼容错误已保留失败证据并修复，仅改变记录选项的读取接口。修复后六卡前后向、有效梯度、FP32 PSF等价反向、四步有限值推理及完整checkpoint精确恢复均通过。A已正式推进至U12，数值正常，控制器PID3751220、torchrun PID3796255，六rank3796437–3796442；初期U7–12约10.00秒/update，剩余纯训练约16.63小时，完整验证与剩余缓存需另计；以实时日志修正ETA。B尚未开始。见 reports/alignment_A6k_B4k_20261022/A_INITIAL_LAUNCH_REPORT_ZH.md。D0代码、训练配置和结果未修改。

每小时监控ID v5 读取本新协议，正常推进保持安静，首次正式更新、首轮完整验证、里程碑、失败、完成或需用户处理时通知。服务器控制器持续执行，监控不重复启动。当前聊天模型档位没有已确认的修改途径，不宣称已降低。

全量数据审计与原ξ复现：DATA_AUDIT PASS；10000条realization/20000个raw字段SHA逐一完全一致，见新cache/audit/RAW_REPLAY.json。物理Y4数值重放在D0结束后的缓存生成中继续逐条验收。

最近监控（2026-10-23 18:00）：A已完成5k里程碑并自动继续至U5038/6000；U5000原Val1000四步PSNR20.1567dB（输入+1.3607dB），成为新best。四stage为21.7734→20.5569→20.2875→20.1567，仍低于D0 best的21.4435dB。U5000完整checkpoint已CPU实读，optimizer/scheduler/六rank RNG和采样状态完整；六卡与数据/代码身份正常。预计今晚21:10–21:15完成A，再按协议自动进入B。见 reports/alignment_A6k_B4k_20261022/A_U5000_MILESTONE_ZH.md。

采样确认（2026-10-23）：A保持6000 updates、每2000更新一批80000内部态，共3批。已只读枚举全部采样计划：每批72000条内部态无重复，使用率90%，时间层近似均衡、所有8000原轨迹覆盖；Y4每批每轨迹恰好3次。非每轨迹严格9张。U4000完整checkpoint实读核实model/optimizer/scheduler/六rank RNG及next_update4001，刷新不重置进度。见 reports/alignment_A6k_B4k_20261022/SAMPLING_AND_RESUME_CONFIRMATION_ZH.md。


2026-10-23 利用率优化窗口（用户请求“你再优化一下训练”）：独立扩展在 `optimizations/utilization_20261023/`，未修改原88份冻结代码、科学计划或正式权重。等待A U5250的两份完整验证及完整checkpoint落盘后，由 `boundary_window.py` 暂退本项目旧controller，独占卡2–7比较原循环与合并loss有限值检查（四个micro及六rank全部检查，仍在optimizer.step之前）。每种方法baseline→candidate→baseline repeat，每组5个临时完整B48计算updates（前2预热后3计时），最后baseline增加1个分段计时update，均丢弃；A短测上限360秒，B900秒。测试有全梯度和首次参数更新逐值对照、异常标记注入；只在逐值一致、前后基线差≤5%、提速≥5%时按方法启用扩展。否则原实现续训。窗口内控制器的预期退役不是训练数值故障，不要另起重复controller。状态见 `reports/alignment_A6k_B4k_20261022/utilization_20261023/WINDOW_STATUS.json`；finally自动用扩展controller从U5250完整恢复A，B仍等A最终best。正式预算、Full-BPTT、微批/累积、BF16/FP32、科学config hash不变；额外代码身份见扩展MANIFEST，启用后每run另记RUNTIME_EXTENSION.jsonl。不能将短测耗时计作正式训练update，也不能漏记额外墙钟成本。


### 2026-10-23 18:49 北京时间：利用率优化短测修复与 ETA

- 首次 U5250 窗口在导入 `runtime.controller` 时失败，尚未执行任何 A/B 对照 update，不能宣称有加速。失败原因是按绝对脚本路径启动时模块搜索路径缺少工程根目录。原始 WINDOW_EVENTS / RESUME_CHECKPOINT / WINDOW_FINISHED 均保留。
- A 从 U5250 完整 checkpoint 安全恢复；18:48:52 已到 U5269，anomaly=null，仍使用原始训练 runner。当前近期正式训练均值约 10.23 秒/update。
- 已修正搜索路径并增加停止训练前的依赖预检；CPU import preflight PASS。新的 CPU watcher PID 1756051，等待 U5500 完整验证与 checkpoint，再进行既定 A/B 有界短测。当前 controller 1706848、child 1709899 正常训练。新窗口状态在 utilization_20261023/retry_5500/WINDOW_STATUS.json，基准结果仍输出上一级 benchmark_A.json / benchmark_B.json，旧失败记录不覆盖。
- 短测只在完整梯度/更新/loss一致、基线稳定且实测提升至少 5% 时选择优化 runner，否则原方案续训。禁止重复启动窗口、禁止将该受控窗口退出误判为模型失败或恢复旧方案。所有窗口耗时累计记录。
- B 尚未正式启动。2026-10-21 同四步训练链的六卡 B48（micro2×accum4）真实 5 次 update 稳态均值为 42.25188 秒（CV 1.37%）；对应 B4000 纯训练约 46.95 小时，加 16 次完整验证/保存及准备约 48–50 小时。该数为既有实测外推，不是新 B 生产测速，更不是优化后结果。
- 按当前速度，A 剩余含三次完整双验证及短测约 2.5–3 小时；新短测预计约 1 小时后开始出结果，取决于 U5500 验证完成和有界短测耗时。


### 2026-10-23 19:57 北京时间：U5500 优化脚本 scheduler 修复

正式 A 正常续训；19:55 查询已到 U5606/6000，数值有限、anomaly=null。U5500 原 Val1000（250 sources）PSNR20.090539dB、SSIM0.503629、LPIPS0.510240、输入增益1.294601dB；四stage21.777170→20.542273→20.245392→20.090539dB。best仍U5000 PSNR20.156652dB。完整checkpoint保存2.58秒，磁盘剩余约4521GiB，原冻结代码无差异。

U5500短测失败发生在测速脚本第一次scheduler.step之前：测速副本加载了A model/optimizer但漏恢复scheduler进度，触发“scheduler must advance by exactly one optimizer update”。没有产生正式update、没有候选加速结论。服务器finally已从U5500完整checkpoint恢复正式A；当前controller1866849、torchrun1869535、六rank1869607–1869612。此次短测/控制器交接窗口59.62秒（另外初始化开销以正式运行记录为准）。原始失败、checkpoint审计和恢复证据保留在 utilization_20261023/retry_5500。

已仅修复扩展benchmark：每个A对照分支都恢复checkpoint scheduler，B仍从独立warmup100/4k schedule的第1步开始。新增CPU回归测试读取真实U5500 checkpoint：A5501–5506及B1–6，baseline/candidate/baseline_repeat六个分支的LR序列与参考逐值一致；报告 CPU_PROBE_SCHEDULER_TEST.json PASS。原正式runner、模型、Loss、随机采样与预算不变，扩展MANIFEST单独更新、旧代码按SHA保留。

CPU watcher1951542已部署，当前只等待U5750完整双验证与checkpoint；状态 utilization_20261023/retry_5750/WINDOW_STATUS.json。届时独占六卡进行既定有界对照，输出仍为上级benchmark_A.json、benchmark_B.json，finally自动续训。不要重复启动，不要将该受控窗口导致的controller退出视作正式训练数值故障。只有逐值正确且至少5%实测提升才启用。

19:52 A近期200更新10.385秒/update，余412步加两次双验证约1.41小时；另计未来短测与恢复约20余分钟，暂估21:35–21:45完成A，B随后自动开始。B仍仅有此前同四步六卡B48基线42.2519秒/update的外推：4000步含验证保存约48–50小时，不能宣称本次加速或B正式测速已完成。此次为新的测速脚本失败并已修复，通知一次；不需要用户处理。


### 2026-10-23 20:57 北京时间：利用率对照完成，未采用候选

U5750窗口已完成并安全恢复。A原实现9.845秒/update、减少同步候选9.945；B原实现40.138、候选40.707。没有达到至少5%收益且未满足逐值一致要求；相同baseline重复本身也有相近数值差异，不能将全部差异归因于候选。不放宽阈值，不采用优化，不再追加这项候选重测。

当前controller2137690、torchrun2140354继续原alignment_training.runner，20:54 A已到U5805/6000、数值正常。U5750完整checkpoint已审计，旧88份冻结代码和科学配置不变，Test锁定。A预计21:35–21:45完成；B将继承最终A best，fresh optimizer按4000预算正常自动启动。B基于本次六卡四步实测40.138秒/update暂估46–48小时（含验证保存等余量，等待正式速度校正）。这一ETA改善不能归因于未启用的候选优化。

窗口已结束，没有待执行watcher；不要重复启动任何优化窗口。详见utilization_20261023/BENCHMARK_CONCLUSION_ZH.md与benchmark_A/B.json。服务器resume_controller会自动按A/B各自adopt=false选择原实现。所有既有监控/续训协议继续。


### 2026-10-23 21:56 北京时间：A6000完成，B正式启动

A已完成6000/6000，完整Val1000最终选出的PSNR-best为U5000：20.156652dB（输入+1.360714dB）、SSIM0.505831、LPIPS0.509330；四stage21.773382→20.556912→20.287529→20.156652dB。Last U6000为20.098987dB（输入+1.303049dB），四stage21.783297→20.551658→20.254641→20.098987dB。A最终四步仍较D0 PSNR-best21.443506dB低1.286853dB，不据A结果宣称递归收益。A记录耗时19.47小时，加D025.05小时累计44.52小时；缓存/优化准备另有记录，不能漏计进总成本。

B已自动继承最终A PSNR-best U5000，parent SHA869a5c39012f87d9e37127b0a8c4a990afaf60db1f8d190adddcd3a52746168a。1180个模型tensor全部逐值继承，time/raw未重置；fresh optimizer/scheduler，warmup100+4k cosine，LR旧层2.5e-5/新层5e-5。六卡正式链检查 READY_FOR_TRAINING，参数梯度完整且有限，末stage loss到首stage clean/raw的跨步梯度非零且有限，FP32等价PSF回归PASS，六rank model/optimizer/scheduler/RNG/下批索引精确恢复PASS。检查的临时updates已丢弃。

21:53 B已正式完成23/4000；21:55继续至26，loss和梯度有限、anomaly=null。首次正式update及0步完整恢复checkpoint均已记录。controller2137690、torchrun2337966，六rank2338078–2338083占用卡2–7，实际alignment_training.runner，未启用利用率优化候选。网络BF16、Simulator/Loss FP32、micro2×accum4=B48、Full-BPTT和原G4/Y4重投影目标不变。

B初期U11–20均值40.42秒/update，含冷启动前20步均值43.11秒，当前单步约40–42秒，尚无长期稳态和B完整Val数据。B预计剩约46–48小时，预计10月25日晚完成，正式完整Val后再校正；首个U250完整Val1000预计10月24日00:30–00:40。最近单卡allocator峰值约54.94GiB、reserved55.95GiB，npu-smi每卡进程59086MiB/全卡约62445MiB；六卡健康无OOM。磁盘剩4520GiB，冻结原代码无差异，Test评价锁定、refinement OFF。

A里程碑与B首个正式update应通知一次；后续正常进度静默监控。不停在2k，不重复启动，不延长B超过4000updates。best一词应明确为PSNR-best，并非所有指标综合最优；用户指出D0后期MAE/SSIM/LPIPS改善是正确的，选择规则仍按已确认的full-Val PSNR，不擅自更换已使用的初始化checkpoint。

### 2026-10-23 23:44 北京时间：用户再次要求优化，新增 B 有界窗口

用户在本聊天明确要求“现在利用率还是太低了，优化”。本次不重复已失败的合并loss检查候选；三个并行代码审计确认新的候选：只对原CPU时间路径已经严格校验且未修改的强度，省去Simulator重复的NPU强度范围布尔回读，每rank每B48 update共56次。原FP32物理算式及顺序、FFT、P2S、卷积、Full-BPTT、微批/累积与随机序列不变；未知或修改的强度回退原检查。候选位于独立 `optimizations/b_utilization_20261023/`，88份冻结科学源码不改。

CPU协议测试 `TRUSTED_STRENGTH_CPU_TEST.json` PASS：原时间路径/全局RNG不变、非法值拒绝、四步tiny rollout输出/参数梯度逐值一致、末stage到首stage clean及两种raw梯度非零有限；不据此声称NPU验收通过。`CPU_RUNTIME_PREFLIGHT.json` PASS：当前真实B checkpoint配置/六rank采样状态、所有分支实际和模拟U250恢复后的6次4k schedule推进一致；完整导入预检PASS。候选SHA ec03403dd6bc5e2578e295cb1558bf402f495bb2665c610fcbf1661b38b530bb。

已部署CPU等待器PID2664991，状态 `reports/alignment_A6k_B4k_20261022/b_utilization_20261023/WINDOW_STATUS.json` 为 WAITING_FOR_CHECKPOINT。目标为B U250完整Val1000与完整checkpoint落盘；23:43正式B U185、anomaly=null，原controller2137690/child2337966仍正常训练。等待器不会提前占用NPU。

U250保存后只退出已核实的本项目controller及其子进程，独占卡2–7，以同一完整B checkpoint/model/optimizer/scheduler/六rank RNG/采样状态做baseline→trusted_strength→baseline_repeat，各5次丢弃更新（2次预热、3次吞吐计时）；另记录原实现一个丢弃update末micro的rank0低开销CPU+NPU profile。整段torchrun限1100秒；失败或超时也进入finally从原U250完整恢复。卡0/1、外部进程不触碰，临时update不计正式4000预算。预计北京时间10月24日00:40–00:50产生结论，尚无实测加速。

仍保留此前性能窗口的严格准入：首更新裁剪后梯度/参数和5步loss逐值一致、前后baseline差≤5%、提速≥5%，CPU证据与候选SHA绑定；不调低门槛将失败记PASS。必须等整个torchrun（含profile）成功退出后，新的 `optimizations.b_utilization_20261023.resume_controller` 才可能选用独立optimized_runner。未通过则原alignment_training.runner续训。新controller同时复用旧恢复保护并核对新扩展MANIFEST。

**监控注意：**不要重复启动窗口或控制器。窗口内旧controller预期退出不等于正式数值故障；先读取新WINDOW_STATUS、WINDOW_EVENTS、benchmark_B.json、WINDOW_FINISHED和resume_controller.log。WINDOW_FINISHED记录恢复PID、probe退出码、耗时；完整probe退出0且adopt=true才能采用。最终报告单列本窗口额外墙钟。任务完成后应核实正式B从U250继续推进，并报告实际收益或没有收益，不能只交付代码。用户未授权更改方法/精度/超参数或延长预算。


## 2026-10-24 00:16 再次优化进展

用户再次授权优化。B正常U234；既有PID2664991的U250完整checkpoint窗口仍在等待，禁止重复启动。新增LN复用减法候选CPU逐字节五组PASS，但未NPU实测、未采用，不自动新增第二窗口。低优先级CPU解析PID2743137等待现有短测完成并生成分析；不得把解析问题当训练故障。详情见 `alignment_A6k_B4k_20261022/OPTIMIZATION_FOLLOWUP_20261024_ZH.md`，先核对WINDOW_FINISHED/benchmark与正式B恢复后成功更新，再报告收益。


### 2026-10-24 00:55 北京时间：B首次Val1000、U250窗口完成并恢复

B U250完整Val1000最终21.664724dB、SSIM0.575843、LPIPS0.444084，相对输入+2.868786dB、A best最终+1.508071dB、D0 best+0.221218dB。四stage21.809475→21.713570→21.702364→21.664724；递归退化明显缩小但最终仍低于第一步。早期结果不作为完成/收敛声明，继续原4000预算。

可信强度候选对照已完成：baseline39.9671、candidate39.7025秒/update，表面0.67%小于baseline前后1.075%波动且未通过逐值一致，adopt=false。不再重复该候选测试、不放宽阈值。窗口899.63秒（14.99分钟），probe状态全丢弃，原完整U250恢复成功；当前controller2855013、torchrun2858765、ranks2858888–2858893，原alignment_training.runner已正式推进U257。没有待执行NPU watcher。CPU后处理COMPLETE，分析位于b_utilization_analysis_U250_20261024。

正式最近200update40.803秒，Val294.54秒、保存2.83秒；剩约44–46小时，中心10月25日20:35。六卡健康、anomaly=null、冻结88代码无差异、磁盘4519GiB。详细证据见alignment_A6k_B4k_20261022/MONITOR_20261024_005505_ZH.md。此次优化结论和B首次完整验证通知一次；之后健康常规进度静默。

LN候选已完成基于当前profile的离线取舍：预计仅删除376次重复subtract/micro，对照本次232845个kernel，且未进入top热点、没有NPU收益实测。决定本轮归档不采用，不额外暂停B；详见 optimizations/layernorm_reuse_20261024/PROFILE_TRIAGE_ZH.md。不得把CPU PASS当NPU采用依据或据此调整ETA。

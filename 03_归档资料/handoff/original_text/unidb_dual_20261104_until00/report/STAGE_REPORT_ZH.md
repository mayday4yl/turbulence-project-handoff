# UniDB 两路线阶段报告

生成于2026-11-03T22:59:47+08:00。状态：**IN_PROGRESS**。原续训线累计新增20525次更新、改进线累计新增40000次更新；本次午夜延长段分别新增4172、10000次更新。训练截止2026-11-04 00:00 Asia/Shanghai（11月3日24点），截止后可完成最终评价。工程gate、CPU数学PASS或小monitor不代表正式训练完成、收敛或泛化成功。

当前根执行状态：`FORMAL_RUNNING`。

尚待：original worker/full evaluation not complete；original BEST complete-state CPU audit missing；original latest_full_summary missing；improved worker/full evaluation not complete；improved BEST complete-state CPU audit missing；improved latest_full_summary missing；controller model-validation completion receipt missing；CPU audit not bound to actual summary /data2/yanglin/unidb_dual_20261104_until00/runs/original/validation/U00065411_ema_full/SUMMARY.json。

## 两条路线与实际工作量

原线从父Final RAW U45411及Adam/EMA/8逻辑RNG继续，原native mean-L1、lr1e-5；3张物理卡模拟8条逻辑B12流。改进线从官方完整SR U0重新初始化，fresh Adam，epsilon_opt MSE、LR2e-7→2e-6 warmup200后wall-clock cosine→2e-7、global clip1.0；5卡局部B20/19/19/19/19。18点段由完整父U55760/U20000接续；午夜段再由各自18点阶段完整父（见授权与下表）接续。两次延长均原LR保持1e-5、改进LR保持2e-7，不重新热身、不回升LR，完整RAW/Adam/EMA/RNG/cursor及旧日程锚点继承。两线均全参数、B96、FP32/HF32off、原网络/桥、无task prompt/Adapter/raw-xi/A/B或物理递归。

这是起点、objective、schedule和资源都不同的实用综合recipe比较，不是单变量因果ablation或吞吐benchmark。原L1没有被证明实现错误；epsilon_opt目标不是实际forward epsilon，loss为mean((noise−epsilon_opt)²)，不再重复除c²。

| 路线 | 卡 | 起始U | 午夜段父U | 3+5累计新增更新 | 累计呈现 | 午夜段新增 | 延长段呈现 | 最近完整保存checkpoint U | 本快照已完成日志U | full best U | 全campaign监控最优U | 延长段监控最优U | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 3 | 45411 | 61764 | 20525 | 1970400 | 4172 | 400512 | 65911 | 65936 | 65411 | 64411 | 64411 | TRAINING |
| improved | 5 | 0 | 30000 | 40000 | 3840000 | 10000 | 960000 | 40000 | 40000 | 10000 | 3000 | 39000 | EVALUATING |

| 路线 | 首正式actor启动(北京时间) | 停训检测(北京时间) | 终评完成(北京时间) | 三阶段正式/恢复actor数 |
| --- | --- | --- | --- | --- |
| original | 2026-11-03T01:40:36+08:00 | 待完成 | 待完成 | 3 |
| improved | 2026-11-03T01:40:36+08:00 | 待完成 | 待完成 | 3 |

资源共享披露：`protocol/SHARED_CONTEXTS.json`记录卡0、2与外部训练任务共享；3+5是设备分配数，不代表独占算力。共享负载会影响实际时延，现有记录不能将时延变化因果归于具体任务或估算具体百分比；未停止外部任务。

交接更新间隔按直接父阶段最后一条committed更新完成→新extension首条committed更新开始计算；包含旧验证、资源交接、工程门与初始化，不能当作纯空闲时间：

| 路线 | 父阶段 | 接续阶段 | 父阶段最后完成U | 完成(北京时间) | extension首更新U | 开始(北京时间) | 记录更新间隔(min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | until_noon | until18_extension | 55760 | 2026-11-03T12:00:00+08:00 | 55761 | 2026-11-03T12:49:14+08:00 | 49.2261 |
| original | until18_extension | until00_extension | 61764 | 2026-11-03T18:00:00+08:00 | 61765 | 2026-11-03T18:17:26+08:00 | 17.4392 |
| improved | until_noon | until18_extension | 20000 | 2026-11-03T11:54:14+08:00 | 20001 | 2026-11-03T12:49:16+08:00 | 55.0259 |
| improved | until18_extension | until00_extension | 30000 | 2026-11-03T17:23:49+08:00 | 30001 | 2026-11-03T18:17:33+08:00 | 53.7387 |

original约49.23分钟；original约17.44分钟；improved约55.03分钟；improved约53.74分钟。该间隔不是纯空闲损失。

两线累计新增更新是当前有效轨迹global update−原始base（45411/0）；本次延长段新增是global update−授权18点阶段父（见父checkpoint表）。连续曲线合并noon、18点段和午夜段的committed日志，以完整父与global update计数，不把新日志行数冒充累计训练量；恢复回放不会被重复计为有效新更新。以下额外计算只来自各线UNCOMMITTED_RECOVERY_rank0*.json中已完成但未保留的记录；同一update若在不同恢复事件再次丢弃，按不同已发生执行计额外工作，未记录的in-flight耗费无法完整重建。无此类文件时明确为0。

| 路线 | 有效轨迹新增更新 | 有效新增呈现 | 额外已完成未保留更新 | 额外已记录呈现 | 恢复记录文件 |
| --- | --- | --- | --- | --- | --- |
| original | 20525 | 1970400 | 0 | 0 | 0 |
| improved | 40000 | 3840000 | 0 | 0 | 0 |

工程门均丢弃：旧noon的2更新门/1更新恢复门及原8卡reference单列；18点段和午夜段各自每线1更新门+1更新恢复门，每段四门合计4更新/384样本呈现，均不带入正式训练。以下按各进程实际完成量计，不用累计new_updates代替该进程次数：

| 阶段 | 工程门 | 路线 | 卡 | 该进程更新 | 该门呈现 | 范围 |
| --- | --- | --- | --- | --- | --- | --- |
| until_noon | improved5 | improved | 5 | 2 | 192 | PASS_GATE_ONLY |
| until_noon | improved_resume | improved | 5 | 1 | 96 | PASS_GATE_ONLY |
| until_noon | original3 | original | 3 | 2 | 192 | PASS_GATE_ONLY |
| until_noon | original_ref8 | original | 8 | 2 | 192 | PASS_GATE_ONLY |
| until_noon | original_resume | original | 3 | 1 | 96 | PASS_GATE_ONLY |
| until18_extension | improved | improved | 5 | 1 | 96 | PASS_GATE_ONLY |
| until18_extension | improved_restart | improved | 5 | 1 | 96 | PASS_GATE_ONLY |
| until18_extension | original | original | 3 | 1 | 96 | PASS_GATE_ONLY |
| until18_extension | original_restart | original | 3 | 1 | 96 | PASS_GATE_ONLY |
| until00_extension | improved | improved | 5 | 1 | 96 | PASS_GATE_ONLY |
| until00_extension | improved_restart | improved | 5 | 1 | 96 | PASS_GATE_ONLY |
| until00_extension | original | original | 3 | 1 | 96 | PASS_GATE_ONLY |
| until00_extension | original_restart | original | 3 | 1 | 96 | PASS_GATE_ONLY |

累计/延长段呈现分别=96×对应有效更新，不是新增独立图。ACTIVITY_AND_COST.csv中的seconds是rank0已记录更新段之和，包含范围有限，**不是CPU核时、完整分布式critical time或GPU利用小时**；验证/保存/工程gate、恢复与空档另由controller actor记录。不能由1号卡driver context推断某OS进程存活或被终止。

## 训练与固定验证曲线

两种loss和梯度norm量纲不同，分图只看各自动态，不能横比绝对值或以loss下降宣称恢复改善。EMA monitor每G100图=25来源×4ξ；完整验证每G1000图=250来源×4ξ，使用不同marker。旧U0/父Final水平线仅在同25来源重聚合的monitor子集可得时绘制，未把完整5000均值冒充monitor基线。

![动态](../figures/CAMPAIGN_DYNAMICS.png)

## 完整质量结果

**best覆盖同一条3+5路线从noon到午夜所有已完成5000全量验证的checkpoint，按G4 PSNR最大、LPIPS再择优选出。** 历史已完成full保留可选best角色，可能仍指向旧noon或18点根；最终要求每条指针与完整state/EMA digest/SUMMARY SHA完成绑定；当前是否满足以pending及验收状态为准。旧U0/父Final只作外部基线，不纳入best。各阶段交接父若未完成full，其partial现场保留，不作为full候选。全campaign监控最优U由三根目录已完成EMA monitor合并后按G4 PSNR、LPIPS、早U规则选出；延长段监控最优U仅为当前extension的BEST_MONITOR marker，二者分列，不写回worker marker。监控最优未必做过完整full验证，不能当作full-best或泛化成功，也不增加既定full候选池。两个role指向同一验证时不算重复模型证据。

| 模型/路线 | 角色 | 阶段 | U | G | PSNR↑ | SSIM↑ | MAE↓ | LPIPS↓ | ΔInput dB | ΔU0 dB | Δ父Final dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | best_full | until00_extension | 65411 | 0.5 | 21.6202 | 0.642259 | 0.0509484 | 0.0746854 | -3.55059 | 5.35815 | 0.928042 |
| original | best_full | until00_extension | 65411 | 1 | 20.1551 | 0.55033 | 0.0612349 | 0.0908435 | -2.0867 | 3.99907 | 0.620061 |
| original | best_full | until00_extension | 65411 | 2 | 18.3187 | 0.448077 | 0.076942 | 0.126819 | -1.98763 | 1.20749 | 0.401446 |
| original | best_full | until00_extension | 65411 | 3 | 17.3773 | 0.3992 | 0.0873201 | 0.204581 | -2.09452 | -0.889644 | 0.509262 |
| original | best_full | until00_extension | 65411 | 4 | 16.1919 | 0.314749 | 0.103388 | 0.411431 | -2.60405 | -2.03051 | 0.296901 |
| improved | best_full | until_noon | 10000 | 0.5 | 17.3321 | 0.297447 | 0.0954324 | 0.486859 | -7.83877 | 1.06998 | -3.36013 |
| improved | best_full | until_noon | 10000 | 1 | 16.8667 | 0.259004 | 0.100373 | 0.500926 | -5.37507 | 0.710702 | -2.66831 |
| improved | best_full | until_noon | 10000 | 2 | 16.0705 | 0.200685 | 0.110927 | 0.566699 | -4.23584 | -1.04072 | -1.84676 |
| improved | best_full | until_noon | 10000 | 3 | 15.2654 | 0.155589 | 0.12414 | 0.667634 | -4.20648 | -3.0016 | -1.6027 |
| improved | best_full | until_noon | 10000 | 4 | 14.7089 | 0.126323 | 0.134535 | 0.771947 | -4.08707 | -3.51353 | -1.18612 |
| U0 | external_fixed_baseline | 外部基线（不适用） | 0 | 0.5 | 16.2621 | 0.362127 | 0.10891 | 0.42995 | -8.90875 | 0 | -4.43011 |
| U0 | external_fixed_baseline | 外部基线（不适用） | 0 | 1 | 16.156 | 0.326916 | 0.107658 | 0.4149 | -6.08577 | 0 | -3.37901 |
| U0 | external_fixed_baseline | 外部基线（不适用） | 0 | 2 | 17.1112 | 0.329318 | 0.0942681 | 0.391977 | -3.19512 | 0 | -0.806047 |
| U0 | external_fixed_baseline | 外部基线（不适用） | 0 | 3 | 18.267 | 0.361694 | 0.083558 | 0.453543 | -1.20488 | 0 | 1.39891 |
| U0 | external_fixed_baseline | 外部基线（不适用） | 0 | 4 | 18.2224 | 0.361802 | 0.0855983 | 0.636238 | -0.573539 | 0 | 2.32741 |
| parentFinal | external_fixed_baseline | 外部基线（不适用） | 45411 | 0.5 | 20.6922 | 0.58822 | 0.056392 | 0.0801451 | -4.47863 | 4.43011 | 0 |
| parentFinal | external_fixed_baseline | 外部基线（不适用） | 45411 | 1 | 19.535 | 0.511612 | 0.0656127 | 0.0957673 | -2.70676 | 3.37901 | 0 |
| parentFinal | external_fixed_baseline | 外部基线（不适用） | 45411 | 2 | 17.9173 | 0.423474 | 0.080868 | 0.137962 | -2.38907 | 0.806047 | 0 |
| parentFinal | external_fixed_baseline | 外部基线（不适用） | 45411 | 3 | 16.8681 | 0.373765 | 0.0939092 | 0.234101 | -2.60378 | -1.39891 | 0 |
| parentFinal | external_fixed_baseline | 外部基线（不适用） | 45411 | 4 | 15.895 | 0.302786 | 0.107844 | 0.425333 | -2.90095 | -2.32741 | 0 |
| Input | input_baseline | 不适用 | 不适用 | 0.5 | 25.1708 | 0.758794 | 0.0353639 | 0.0749475 | 0 | 8.90875 | 4.47863 |
| Input | input_baseline | 不适用 | 不适用 | 1 | 22.2418 | 0.618509 | 0.0494776 | 0.106024 | 0 | 6.08577 | 2.70676 |
| Input | input_baseline | 不适用 | 不适用 | 2 | 20.3064 | 0.513581 | 0.0637261 | 0.191671 | 0 | 3.19512 | 2.38907 |
| Input | input_baseline | 不适用 | 不适用 | 3 | 19.4719 | 0.475845 | 0.0724754 | 0.361375 | 0 | 1.20488 | 2.60378 |
| Input | input_baseline | 不适用 | 不适用 | 4 | 18.7959 | 0.451751 | 0.0801783 | 0.571629 | 0 | 0.573539 | 2.90095 |

最新已完成RAW monitor另列（每G100图，不与EMA full混称）：

| 路线 | U | 阶段 | G | 图数 | PSNR↑ | SSIM↑ | MAE↓ | LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 47411 | until_noon | 1 | 100 | 20.1809 | 0.563564 | 0.0583806 | 0.0843896 |
| original | 47411 | until_noon | 4 | 100 | 16.067 | 0.337717 | 0.102738 | 0.422758 |
| improved | 2000 | until_noon | 1 | 100 | 16.4789 | 0.262288 | 0.102791 | 0.492207 |
| improved | 2000 | until_noon | 4 | 100 | 15.0256 | 0.17192 | 0.126208 | 0.661731 |

所有主结果为原生100步native reverse SDE、相同来源/seed、raw float RGB、无clip/crop/resize；RAW小monitor与EMA完整结果分开。LPIPS改善可能与PSNR/SSIM下降并存，逐项报告。验证隔离并恢复训练RNG由worker/独立门检查，本报告不加载权重、构造恢复网络或重新算神经指标。完整性和恢复目标是否达到是不同结论。

## 预声明固定图

固定源为MONITOR_SELECTION中前2来源的G4、各按sample_id排序取首ξ，不按输出挑图；记录于FIXED_IMAGE_SELECTION.json。每列Input、旧U0、父Final、新original、新improved、GT。阶段图仅用已有已完成验证；可能显示明确标注的monitor候选，空白列表示尚无保存结果，不调用模型补图。图像仅显示时clamp[0,1]；指标仍为未经裁剪原浮点值，SHA/min/max/越界比例存FIGURE_MANIFEST。

![固定best](../figures/FIXED_G4_BEST.png)

![固定latest](../figures/FIXED_G4_LATEST.png)

图best所用新验证：{"original": {"path": "/data2/yanglin/unidb_dual_20261104_until00/runs/original/validation/U00065411_ema_full/SUMMARY.json", "update": 65411, "full": true, "campaign_stage": "until00_extension"}, "improved": {"path": "/data2/yanglin/unidb_dual_20261103_until_noon/runs/improved/validation/U00010000_ema_full/SUMMARY.json", "update": 10000, "full": true, "campaign_stage": "until_noon"}}。

图latest所用新验证：{"original": {"path": "/data2/yanglin/unidb_dual_20261104_until00/runs/original/validation/U00065411_ema_full/SUMMARY.json", "update": 65411, "full": true, "campaign_stage": "until00_extension"}, "improved": {"path": "/data2/yanglin/unidb_dual_20261104_until00/runs/improved/validation/U00039000_ema_monitor/SUMMARY.json", "update": 39000, "full": false, "campaign_stage": "until00_extension"}}。

## 验收、范围与限制

本报告生成器本次恢复网络调用=0、指标网络调用=0、指标模型构造=0、optimizer更新=0；只读已有日志、指标和预测。

历史已完成CPU审计来源：`/data2/yanglin/unidb_dual_20261103_until_noon/evidence/FINAL_OUTPUTS_CPU_AUDIT.json`，SHA `a884f0acc075d9b68b36e926dbddc17e45df8a41c0c035e91f07386c488c3b8e`。按validation signature去重后1套验证，历史记录指标forward 1250个batch、指标模型构造1次；这些是历史阶段已发生的审计成本，不是本次reporter调用或延长段新增开销。

当前尚无新extension wrapper的最终CPU审计invocation收据；复制的历史aggregate/cache不计作本次最终审计或延长段新增成本。最终wrapper会在完成后提供实际调用收据，历史/本次/本reporter分列于CPU_POSTPROCESS_WORK.csv，逐signature来源见CPU_AUDIT_VALIDATION_PROVENANCE.csv。

独立CPU指标审计绑定状态：尚未满足完整交付条件；固定图目视状态：待root/heartbeat对本次新图SHA重新目视确认。--final仅在两线best/latest完整full、controller完成及CPU保存输出审计PASS并绑定实际SUMMARY后允许生成最终报告。

G1–G4为同模拟器训练强度，G0.5仅作较弱强度验证；这里是内容留出验证，未建立真实外部湍流或Test泛化。午夜后评价可以继续；午夜后不得再开始新的训练更新，在途更新于完整边界保存收尾。改进延长段固定floor2e-7，属于明确披露的复合日程，不能把延长结果单归于loss或LR。3+5资源分配服务于实用训练机会，不证明某配方算力效率或单因果优越。旧父实验与封存forensic audit没有被本报告改写。

生成命令：`/data2/yanglin/v6_diffuir_l_20261027/env/bin/python /data2/yanglin/unidb_dual_20261104_until00/scripts/make_campaign_report.py`。输入文件snapshot SHA、脚本SHA、输出SHA见REPORT_MANIFEST.json；活动日志SHA对应实际读取的完整字节前缀，随后追加不会追溯改变本次CSV。

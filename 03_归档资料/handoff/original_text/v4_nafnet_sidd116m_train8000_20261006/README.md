## 2026-10-07 17时核验：旧 Restormer L08 U5000 全量 Train32000 已完成

核验本地时间：2026-10-07T17:39:01+08:00。`reports/L08_TRAIN32000_DUAL/COMPLETE.json` 与 `FINAL_CPU_AUDIT_20261007.json` 均PASS。32000 starts / 2000 sources，G1–4各8000，全部32027个COMMIT条目SHA通过；逐start身份和原顺序一致，原FP64聚合函数重新计算结果完全一致。RAW U5000、refinement OFF、batch1、原BF16评价及指标协议不变。

完整Train结果：path_total=0.04936820035142591，L_C=0.04747153133881511，L_R=0.009483345030821511，PSNR=22.900243981826208 dB，SSIM=0.6322163298183924，LPIPS=0.4139369083833881。最终CSV/Markdown与aggregate/COMMIT已镜像本地，分G表在 `reports/L08_TRAIN32000_DUAL/L08_U5000_TRAIN32000.md`。

旧dual controller及两个评价worker已结束，完整结果不再PENDING，禁止重新启动dual/repair/sweep或重复推理；原单卡parallel永久STOP和全部历史分片保留。旧补评ETA为已完成，历史47分钟/1.79小时估算失效。

本次验收快照：NAFNet R0 U7653、R08 U7593，原PID1117944/1118184及startticks未变，micro16×accum2/B32；D0/L08仍按持久队列从共同U0接续。账号只GPU3/5，原训练没有收到信号或重启。新训练仍未完成，90分钟自动监督保持ACTIVE。后续ETA需用补评退出后的正式新窗口计算，不复用共卡旧速度。



## 2026-10-07 micro16稳定验收完成

`reports/micro16_user_selection_20261007/COMPLETE.json`、`STABLE_RUNTIME.json`均PASS；四组16×2/B32。60秒预热后300秒共卡观察，R0/R08各20个正式updates，平均15.57/16.02秒，未见OOM/非有限值。账号仅3/5；当前实际训练PID1117944/1118184，CPUworkflow1115584/supervisor1115593、旧dual1155750及workers1156169/1156359均只是快照，后续必须刷新。R0自身U6898/R08自身U6894零回滚恢复；D0/L08 micro16仍U0排队；旧补评分片继续。

完整报告 `reports/micro16_user_selection_20261007/RESULT_REPORT.md`；更新ETA `ETA_AFTER_STABLE.json/.md`。旧8档ETA只为历史；当前四法含排队纯训练约103.5小时，含评价保存规划108–119小时，D0/L08仍为短测外推；旧L08补评19880/32000，剩余推理约1.79小时，合并hash另计。绝对时间用本地当前时刻加剩余时长，不能混用比本地快约16分钟的服务器时钟。

用户16档选择覆盖历史8档策略，原micro1/2/4不重新测速，所有已完成probe/guardian/observer不重启。工程v5在MICROBATCH_USER16_GATE/REVISION登记；原冻结科学config、preflight时间限制、sweep结果保持原样。90分钟监控正常继续，只有重大变化通知。


## 2026-10-07 最新配置：用户指定 micro16，切换已完成

四方法正式工程配置为 **microbatch16 × accumulation2，effective batch32，FP32**。R0从自身完整U6898、R08从自身完整U6894恢复，回滚0步；D0/L08仍从共同U0排队，配置均为16。四组16档完整恢复验收及精确源码兼容链PASS；CPU监督与旧L08双卡补评均已接续。只占GPU3/5。原模型/数据/损失/fullBPTT/LR/预算、原preflight/TIMEBOX及历史receipt均未修改。

最新依据：`reports/micro16_user_selection_20261007/COMPLETE.json`、`FORMAL_RESUMED.json`、`SOURCE_ACCEPTANCE.json`；稳定窗口结果见同目录 `STABLE_RUNTIME.json`（必须实际存在且PASS才算通过）。实际微批量由 `protocol/MICROBATCH_USER16_GATE_20261007.json` 和 `MICROBATCH_USER16_REVISION_20261007.json` 工程v5登记，原冻结config的micro4以及历史micro8属于历史配置。下文所有更早快照按此最新用户选择解释，不重做sweep，不重复切换。

# V4 NAFNet-SIDD pretrained / Train8000 / Phase-1

## 2026-10-07 已验收的microbatch工程调整

用户授权的短测后，R0/R08已从自身完整U6705无回滚接续为micro8×accum4，B32及全部科学定义不变；D0/L08保持micro4×accum8。保守短测收益15.4%/13.6%，不是最终ETA。[验收及恢复边界](reports/microbatch_tuning_20261007/RESULT_REPORT.md)与MICROBATCH_GATE/MICROBATCH_REVISION解释原冻结config micro4上限的有界工程覆盖。原config/LAUNCH_GATE/preflight TIMEBOX不修改。旧L08双卡评价已经从保留分片接续。


当前授权是2026-10-06新正文、五条补充、账号最多两张A100，以及用户明确授权的LSDIR登录/同意访问条件/下载操作。参见 `protocol/USER_EXECUTION_PROMPT.txt`、`protocol/AUTHORIZATION.json`、`protocol/SCIENTIFIC_SPEC.json`。

**当前：NAFNet已在A100 GPU3/GPU5两卡正式训练，分别R0/R08；L08/D0在持久队列自动补位。统一FP32/B32/peak LR5e-5、micro4×accum8和各20k预算不变。已核验GPU1非存活驱动记录并修正活跃配额误判，R0未中断。** 实时证据优先于本段快照。执行状态见 `NEW_RUN_LAUNCH_REPORT.md`、`WORKLOG.md`、`handoffs/UNATTENDED_HANDOFF_20261006.md`。

- 旧Restormer：只保留matched U5000主比较，逐组完整状态验收、持久STOP、graceful stop及无respawn验证。D0已有U>5000结果全部保留，不继续旧20k计划。旧NPU训练不恢复。
- 新数据：Places4000+LSDIR4000训练；Places500+LSDIR500验证；旧Test200锁定。原2450个来源及已有退化字节复用，不重新生成。全局精确重复/基础近重复审查覆盖跨数据集和Test来源。
- LSDIR：官方公开val只有250，本轮已按允许的fallback，采用官方train源池的4000train+500held-out；明确不是官方val500。选择源池为按固定seed预选的官方HR shard-09，尚未把“抽一个archive”描述为对全部LSDIR的均匀抽样。
- de36：唯一canonical generation，原已验收FP32 pipeline，raw ξ保留。只允许NPU0–3持续空闲后取分片，不挤占其他任务、不用5–7。生成和全量验收已经完成，DATA_READY已冻结；原数据控制器正常退出，不得重启生成。
- A100：仅3/5；账号全部任务可能超两张时等待，不用第三卡。新full-BPTT四方法全参数微调；同预训练共同U0、数据/样本流、effective B、LR/precision和640000 exposures；Phase-1后不自动延长。
- 新训练gates：Train8000统计、全部runtime RGB哈希、真实峰值LR≥10步/方法、吞吐B32/B64、FP32/BF16、调度及精确状态恢复。CPU工程检查已通过，不替代真实GPU验收。

A100版本：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_nafnet_sidd116m_train8000_20261006`。
de36版本：`/data2/yanglin/v4_nafnet_sidd116m_train8000_20261006`。
Canonical：`places_lsdir_p2s_canonical_20261006_p4000_l4000_v1000_t200_g4321_v1`。

本地镜像经已有SSH连接转发，未新增远端SSH权限。**镜像已完成，本机休眠或离开校园网络不再影响A100预检和训练推进**；服务器数据构建和旧实验收尾独立继续。A100_RUNTIME_READY后，GPU验收/新训练/队列/恢复由服务器守护独立执行。保留90分钟聊天跟进，健康等待不重复通知。

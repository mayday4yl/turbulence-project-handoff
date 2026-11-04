# V6 B 前置验收、Pilot 与正式启动报告

B 已通过工程及梯度硬门槛并自主进入正式10000更新；当前启动验收通过，不代表B全阶段已完成或科研成功。正式run为distributed/B_five_npu_formal10000_01，仅NPU3–7。

|检查|真实结果|
|---|---|
|实际A-best初始化完整B图|独立3次Adam更新，完整Full-BPTT及参数更新通过；临时权重丢弃|
|五卡稳定Benchmark|3 warmup + 100稳定更新，103次真实forward/backward/Adam/EMA；五卡末端参数digest精确一致|
|五卡独立恢复|10000更新生产LR周期，2步参考与从U1独立重启下一步一致；raw最大差7.45e-9、数据/时间/噪声/LR/RNG一致|
|独立Pilot|100更新、1000呈现/1000轨迹/1000 Clean、50类各20；临时权重不继承|
|初始及终点分支/末端梯度|同15条固定Train轨迹，30条记录/270项末端梯度有限非零；真实Pilot U0 core精确匹配；三组参数均更新|
|正式初始化与首个完整保存|正式U0 raw/EMA与A best EMA精确相同、Adam为空；U250的raw/EMA/Adam均有限，468个Adam计数250、五卡RNG、游标2500、生产调度10000及SHA通过|

五卡micro1×acc2，有效batch10，FP32/HF32off，无AMP/checkpointing。原Condition/Forward/Reverse、联合Loss=0.9×四步Clean均值+0.1×四步G4物理一致性均值+0.1×TF原生残差保持；每次micro有四阶段三步采样和一次TF随机扩散时间调用，13 NFE。TF四节点均匀协议、严格随机物理节点、四阶段独立扩散噪声和Full-BPTT保持。

|稳定测速指标|实测|
|---|---:|
|每更新耗时（100次，逐更新取五卡最慢后平均）|5.014849秒|
|五卡吞吐|1.994078图/秒|
|peak allocated|33.326 GiB/卡|
|peak reserved|35.088 GiB/卡|
|npu-smi采样真实HBM峰值（五卡最大）|39.126 GiB/卡|

正式训练启动审查到U311：五卡已记录更新连续、联合Loss权重一致、全部Loss及三组梯度有限非零，受检455项网络输出梯度来自完整13NFE；严格物理时间、四独立采样种子与TF节点范围核验通过。启动审查近100更新约5.20秒/update，仅是此时快照，后续实时进度查STATUS/日志。

初始完整Val1000按明确等价复用规则引用A best：4个固定例的全部四步Clean数组最大差均0，参数/core/扩散buffers/数据/时间/噪声/指标协议完全一致；明确REUSED_VERIFIED_EQUIVALENT且fresh_B_Val1000_inference_executed=false，没有重新运行B初始1000图推理。Pilot U100重新执行完整Val1000，四步PSNR20.39584990 dB（初始19.86120922），Step4−Step1为−0.59238761 dB；这是Pilot结果，不替代正式B或泛化结论。所有正式后续完整验证真实重新执行。U250只做Val100辅助诊断，不用于替代完整Val1000选best。

原并行初始诊断01在JSON读取阶段失败；新增v2修复后真实15例通过。原56文件逐项保持，新冻结59文件；成功重试复用已完成单卡图，成功图与分支实测重叠为0，不宣称测得并行提速。故障、精确diff和复验见engineering_repair_20261028_0310。

数据Train10000/40K与Val/Test冻结，不使用Test选模型。正式B从A U1000 EMA初始化，官方方法及预算不改；继续每250更新完整保存与辅助Val100、每1000及终点完整固定Val1000按四步最终PSNR选best。中文最终报告待B全部完成及退出验收后生成。

证据：execution_20261027_five_npu/B_PREFLIGHT_AND_STARTUP_ACCEPTANCE.json、B_FORMAL_U250_CHECKPOINT_AUDIT.json及各Gate/原始run；当前CPU控制器4081304/start_ticks541277305，以ACTIVE_CONTROLLER/PIPELINE_LAUNCH03为准。

# V6 DiffUIR-L 独立执行目录

唯一方法依据为 `authority/V6_DiffUIR_L_D0_A_B_METHOD_SPEC_v1.0.md`，训练/数据依据为 Training Handoff，并由 `authority/USER_EXECUTION_OVERRIDES_20261027.md` 记录最新覆盖指令。取消恢复质量 GO/NO-GO；保留方法、数据、梯度、设备与 checkpoint 工程门槛。

完整方法审查在相邻 `v6_diffuir_l_method_review_20261027/`。本目录保存所有新实现与执行证据；旧 V5、Stage1、ResShift 与旧 Benchmark 只读。

## 最新授权：D0续训5 epoch → A6000 → B10000

用户已重新明确授权连续训练并允许提前停止。当前D0续训正在新run `D0_continue_5epoch_01`运行，目标U13334；原U10000及全部旧成果保留。A/B通过各自实际继承权重的工程验收和独立Pilot后自动衔接。每250更新完整保存；每1小时自动巡检，正常静默。当前入口：[续训执行计划](execution_20261027_continuation/RUN_PLAN_ZH.md)、`execution_20261027_continuation/STATUS.json`及`gates/EXECUTION_POLICY_CURRENT.json`。以下Benchmark-only说明为历史阶段。

## 历史阶段：仅 A/B Benchmark 与配置

用户已批准 ξ Head 时间调制，随后明确“先把AB的显存测了，先不训练”“就是先把配置定一下，我还在看D0的结果”。最新依据为 `authority/USER_BENCHMARK_ONLY_20261027.md` 和 `gates/EXECUTION_POLICY_CURRENT.json`。**不得自动进入 A/B Pilot 或正式训练。**

A/B 完整图 Benchmark 及低利用率优化已完成。六卡各3次预热+100次稳定计时通过：A micro4×6×acc2，16.479轨迹/s；B micro1×6×acc2，2.361轨迹/s，较保留基线分别提升11.3%和37.7%。临时模型丢弃，相关NPU worker全部退出，v6自动跟进已暂停；A/B Pilot与正式训练均未启动。B仍未继承A best，本轮仅为初始完整图的容量与性能验证。最终优化报告：[AB_PERFORMANCE_OPTIMIZATION_ZH.md](reports/AB_PERFORMANCE_OPTIMIZATION_ZH.md)；原基准：[AB_BENCHMARK_REPORT_ZH.md](reports/AB_BENCHMARK_REPORT_ZH.md)；候选配置：`benchmarks/AB_OPTIMIZED_CONFIG_CANDIDATE.json`；终审：`optimizations/dispatch_20261027/COMPLETION_AUDIT.json`。正式runner尚未迁移/验收此候选运行时，后续训练须用户另行明确授权。

## D0 完成时的历史快照（2026-10-27 09:03）

D0正式10000/10000完成，Train10000/40000轨迹已真实验收；最佳U10000，固定Val1000 PSNR21.03393849dB。六卡训练worker已退出。A/B 0更新，因ξ Head时间调制缺项待明确。请先读[V6_D0_A_B_FINAL_REPORT_ZH.md](V6_D0_A_B_FINAL_REPORT_ZH.md)和`gates/GATE_STATUS_AT_D0_COMPLETION.json`；该报告明确整体V6尚未完成。

## 实时状态

实际工作根在 de36：`/data2/yanglin/v6_diffuir_l_20261027`。本地可能滞后；检查远端状态文件、对应进程的 PID/start_ticks、最新日志和实际 checkpoint，不以目录名称或任务标题判断完成。

- `evidence/DATA_PIPELINE_STATUS.json`：官方归档→source冻结→G4生成→独立重放→逐文件G1。
- `evidence/D0_PIPELINE_STATUS.json`：图/恢复/HCCL→Batch候选→G1→完整新池DDP复测→实际六卡恢复验收→Pilot300→正式D0 10000。
- `distributed/<run>/STATUS.json`、`LAST.json`、`BEST.json`：实际进度与可恢复/最佳checkpoint。临时Benchmark不得继承到Pilot/正式训练。
- `datasets/DATA_READY.json`：只有真实40000个配对文件验收完成才存在PASS。`SOURCE_READY`不是`DATA_READY`。
- `gates/`：分阶段工程证据；A/B未执行不得写PASS。

控制器通过同一NPU分配锁协调，只使用实时空闲的物理2–7号卡。不要重复启动已有live PID；故障先保存日志、源码版本和diff，再修复、重新验收。已完成且身份正确的阶段直接复用。

## 当前交付入口

- `reports/IMPLEMENTATION_MAPPING_ZH.md`：实现与规范对应、未完成边界。
- `reports/BENCHMARK_REPORT_D0_ZH.md`、`benchmarks/RESULTS_D0.json`：完整图实际Batch/显存/吞吐。
- `reports/OFFICIAL_EMA_VAL1000_BASELINE_ZH.md`：固定Val1000真实官方EMA U0基线，负结果如实保留。
- `reports/ENGINEERING_FIXES_ZH.md`、`evidence/code_history/`：工程修复及原失败证据。
- `WORKLOG.md`：人类可读交接；机器实况优先。

Q1 的 ξ Head 时间调制已由用户明确批准，见 `authority/V6_METHOD_Q1_XI_TIME_APPROVED_20261027.md`。当前不训练是用户随后要求先定配置，而非方法缺项。A best 必须按固定Val1000、固定4→3→2→1→0路径四步最终结果选择。

只读查询命令：在远端V6根执行 `PYTHONPATH=/data2/yanglin/v6_diffuir_l_20261027 python3 scripts/status_v6.py`，同时返回live控制器、生成数、Gate、checkpoint和NPU进程。`DATA_READY.training_started=false`仅描述数据冻结时点，后续训练状态以对应run实时文件为准。

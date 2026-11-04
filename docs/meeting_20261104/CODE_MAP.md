# 逐函数代码导航

行号由当前源码AST提取，SHA见[evidence/CODE_MAP.json](evidence/CODE_MAP.json)。链接指向Git源码快照；服务器根为`/data2/yanglin/<分支>`。

当前必读顺序：simulator_v5 → UniDB core → data → 午夜worker/validation。V5 training/core与alignment_training/core仅在追查旧递归实现时阅读。

运行用[独立示例](../../tools/receiver/demo.py)；原worker绑定旧root、截止、卡数与源SHA，不能直接当今天的通用launcher。

## 读代码时的重点

- `effective_state`把raw变成场；`render_prepared`先Smooth再warp；`flow`是归一化grid，位移在`effective_state.displacement`。
- `build_simulator`绑定原canonical buffers；`_P2S.forward`按上游复用fc2，不能当手误改fc3。
- V5 A回投当前g/Yg，B回投G4/Y4；下一状态来自预测，正式refinement OFF。
- UniDB `NativeObjective.forward`是原生一步均值L1；NPU初始化含精确GELU适配。
- 完整续训需要RAW、optimizer、EMA、RNG、loader、cursor和scheduler；加载EMA仅用于推理/新初始化。

## 函数索引

### v5_v1_simulator_20261020/simulator_v5.py

[打开文件](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py)

|函数|源码入口|
|---|---|
|`bind_canonical_buffers`|[L31](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L31)|
|`effective_state`|[L97](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L97)|
|`prepare_image`|[L133](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L133)|
|`sampling_grid`|[L145](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L145)|
|`warp`|[L151](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L151)|
|`render_prepared`|[L156](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L156)|
|`build_simulator`|[L180](../../source_snapshot/v5_v1_simulator_20261020/simulator_v5.py#L180)|

### v5_v1_simulator_20261020/vendor/v4_simulator.py

[打开文件](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py)

|函数|源码入口|
|---|---|
|`forward`|[L63](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py#L63)|
|`sample_realization`|[L262](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py#L262)|
|`effective_state`|[L289](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py#L289)|
|`prepare_image`|[L325](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py#L325)|
|`render_prepared`|[L347](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py#L347)|
|`render`|[L380](../../source_snapshot/v5_v1_simulator_20261020/vendor/v4_simulator.py#L380)|

### v5_v1_simulator_20261020/method_interface_20261021/schedules.py

[打开文件](../../source_snapshot/v5_v1_simulator_20261020/method_interface_20261021/schedules.py)

|函数|源码入口|
|---|---|
|`sample_training_paths`|[L209](../../source_snapshot/v5_v1_simulator_20261020/method_interface_20261021/schedules.py#L209)|
|`inference_paths`|[L253](../../source_snapshot/v5_v1_simulator_20261020/method_interface_20261021/schedules.py#L253)|

### v5_concertormer_10k_20261021/models/clean_concertormer.py

[打开文件](../../source_snapshot/v5_concertormer_10k_20261021/models/clean_concertormer.py)

|函数|源码入口|
|---|---|
|`forward`|[L28](../../source_snapshot/v5_concertormer_10k_20261021/models/clean_concertormer.py#L28)|
|`build_clean_model`|[L52](../../source_snapshot/v5_concertormer_10k_20261021/models/clean_concertormer.py#L52)|

### v5_concertormer_10k_20261021/models/v5_concertormer.py

[打开文件](../../source_snapshot/v5_concertormer_10k_20261021/models/v5_concertormer.py)

|函数|源码入口|
|---|---|
|`forward`|[L46](../../source_snapshot/v5_concertormer_10k_20261021/models/v5_concertormer.py#L46)|
|`forward`|[L112](../../source_snapshot/v5_concertormer_10k_20261021/models/v5_concertormer.py#L112)|

### v5_concertormer_10k_20261021/training/core.py

[打开文件](../../source_snapshot/v5_concertormer_10k_20261021/training/core.py)

|函数|源码入口|
|---|---|
|`unroll_training`|[L90](../../source_snapshot/v5_concertormer_10k_20261021/training/core.py#L90)|
|`infer`|[L212](../../source_snapshot/v5_concertormer_10k_20261021/training/core.py#L212)|

### v5_concertormer_10k_20261021/alignment_training/core.py

[打开文件](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/core.py)

|函数|源码入口|
|---|---|
|`unroll_training`|[L34](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/core.py#L34)|
|`infer`|[L61](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/core.py#L61)|

### v5_concertormer_10k_20261021/alignment_training/runner.py

[打开文件](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/runner.py)

|函数|源码入口|
|---|---|
|`save_training_checkpoint`|[L186](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/runner.py#L186)|
|`train`|[L263](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/runner.py#L263)|
|`main`|[L624](../../source_snapshot/v5_concertormer_10k_20261021/alignment_training/runner.py#L624)|

### unidb_d0_20261102/core.py

[打开文件](../../source_snapshot/unidb_d0_20261102/core.py)

|函数|源码入口|
|---|---|
|`sources_ok`|[L42](../../source_snapshot/unidb_d0_20261102/core.py#L42)|
|`initialize`|[L52](../../source_snapshot/unidb_d0_20261102/core.py#L52)|
|`configure_npu`|[L89](../../source_snapshot/unidb_d0_20261102/core.py#L89)|
|`process`|[L103](../../source_snapshot/unidb_d0_20261102/core.py#L103)|
|`forward`|[L116](../../source_snapshot/unidb_d0_20261102/core.py#L116)|
|`make_ema`|[L128](../../source_snapshot/unidb_d0_20261102/core.py#L128)|

### unidb_d0_20261102/data.py

[打开文件](../../source_snapshot/unidb_d0_20261102/data.py)

|函数|源码入口|
|---|---|
|`load_pool`|[L12](../../source_snapshot/unidb_d0_20261102/data.py#L12)|
|`validation_groups`|[L66](../../source_snapshot/unidb_d0_20261102/data.py#L66)|

### unidb_dual_20261103_until_noon/improved_objective.py

[打开文件](../../source_snapshot/unidb_dual_20261103_until_noon/improved_objective.py)

|函数|源码入口|
|---|---|
|`forward`|[L45](../../source_snapshot/unidb_dual_20261103_until_noon/improved_objective.py#L45)|

### unidb_dual_20261104_until00/training_worker.py

[打开文件](../../source_snapshot/unidb_dual_20261104_until00/training_worker.py)

|函数|源码入口|
|---|---|
|`source_checks`|[L67](../../source_snapshot/unidb_dual_20261104_until00/training_worker.py#L67)|
|`forward`|[L144](../../source_snapshot/unidb_dual_20261104_until00/training_worker.py#L144)|
|`save_checkpoint`|[L241](../../source_snapshot/unidb_dual_20261104_until00/training_worker.py#L241)|
|`load_resume`|[L297](../../source_snapshot/unidb_dual_20261104_until00/training_worker.py#L297)|
|`main`|[L406](../../source_snapshot/unidb_dual_20261104_until00/training_worker.py#L406)|

### unidb_dual_20261104_until00/validation.py

[打开文件](../../source_snapshot/unidb_dual_20261104_until00/validation.py)

|函数|源码入口|
|---|---|
|`evaluate`|[L12](../../source_snapshot/unidb_dual_20261104_until00/validation.py#L12)|

## 官方UniDB源码

Git快照未包含全部官方vendor。de36实际目录是`/data2/yanglin/unidb_d0_20261102/vendor/unidb_gou/`。权威映射在该root的`protocol/UPSTREAM_SOURCE.json`，固定commit为`1bb63e318891b932c6e641b98729643e46b91e06`。

|文件|查看内容|
|---|---|
|`DenoisingUNet_arch.py`|`ConditionalUNet.forward(xt,cond,time)`，xt−cond与cond拼接|
|`sde_utils.py`|随机桥状态、reverse_optimum_step、reverse_sde_step_mean、100步reverse_sde|
|`loss.py`|MatchingLoss L1归约|
|`module_util.py`|时间嵌入、残差块、注意力|

[官方固定版本](https://github.com/UniDB-SOC/UniDB/tree/1bb63e318891b932c6e641b98729643e46b91e06/UniDB-GOU)。项目wrapper与官方模块分开登记，不用上游最新main替代冻结版本。

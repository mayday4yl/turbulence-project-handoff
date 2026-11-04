# V5 Concertormer A100 有界测速准备

NPU 最新候选参考已更新：固定 PSF 输入梯度的等价 FP32 优化，四卡 B32 的 D0/R0 为 9.603/39.094 秒，见 [核对结果](reports/NPU_REFERENCE_UPDATE_ZH.md)。旧约 22.8 天为原实现外推；最新候选三组各 10k 纯训练外推约 10.16 天。A100 尚未验证该改写的数值和性能，未改变执行入口。

2026-10-21 后续更新：用户已允许共享 zhuzhiyu 的卡。实时占用与原全重算结果支持尝试低显存有界探针，见 [最新共享显存判断](reports/SHARED_MEMORY_ASSESSMENT_ZH.md)。下文两卡独占/75 GiB 是原速度优先方案，不是所有测试的最低条件；现有执行入口尚未切换为共享模式。本次判断没有启动 GPU 测试。

已完成 CPU/静态准备，GPU 尚未测试，正式训练未启动。详见 [准备结果与资源申请](reports/PREPARATION_AND_RESOURCE_REQUEST_ZH.md)。

本地目录 `/Users/lucas/Desktop/turbulence IR/v5_concertormer_a100_benchmark_20261021`；A100 独立镜像 `/home/zhuzhiyu/yanglin/turbulence_ir/versions/v5_concertormer_a100_benchmark_20261021`。原 NPU 工程、原 A100 Simulator 验收及现有作业保持不变。

当前入口为 `scripts/check_preparation_cpu.py` 和 `scripts/launch_bounded_cuda.py`。后者默认仅打印命令，不启动 GPU。`--execute` 需要本轮明确资源确认、绑定计划与源码哈希、准确 GPU UUID 和有效时间窗口，并在 CUDA 初始化前检查实时占用。`protocol/BENCHMARK_RESOURCE_AUTHORIZATION.json` 当前未批准；正式训练开关也保持未批准。

建议资源为两张 A100 80GB 独占最多 60 分钟，账号总占用不得超过两张，启动时每卡至少 75 GiB 空闲。先单卡 D0/R0 真实 B2 预检，再双卡 B32 短测。尚无 CUDA 训练链路显存或吞吐实测，不沿用 Simulator 单测的显存判断。

`protocol/A100_BENCHMARK_PLAN.json` 是本轮计划。其他 `PROPOSED_*`、历史 NPU launcher/config 和 `source_snapshot/` 仅用于来源追溯，不能作为 A100 启动入口。这里只有与 NPU 测速相同的 32 个 train 样本，不是完整训练数据集。当前有界测速不需要也不读取 Val/Test 图像。

可在远端独立目录中使用既有 Python 环境运行以下无 GPU 命令：

```sh
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python scripts/launch_bounded_cuda.py
```

CUDA 完整模型执行、跨步梯度、NCCL、实际显存/吞吐、完整 CUDA checkpoint 写读恢复必须在资源明确批准后实测。当前通过的 CPU 小模型语义检查不替代上述项目。

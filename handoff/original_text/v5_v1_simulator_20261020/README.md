# V5-v1 Simulator · de36 NPU

独立实现 V4-R0 的 Simulator 演进候选，包含 S1 Smooth、S2 Tilt/Warp、S3 FFT 和严格内部时间采样。**2026-10-21 后续决定：按保留已知局限的研究候选接收，Simulator 不再阻塞 V5 下一阶段。** 原冻结 Gate 未全部通过的事实、近零/差分失败及阈值全部保留；接收依据与测试结果分开记录。正式数据生成和模型训练尚未启动。

远端目录：`liteserver-de36:/data2/yanglin/v5_v1_simulator_20261020/`。本轮只运行固定测试样例和有界数值检查，未生成正式数据、加载恢复网络或启动训练。V4 源码、历史数据和模型权重未修改。

2026-10-21 实验准备更新：网络接口明确为 **N(Z_RGB3,t)**；训练仍 fixed G4/NFE4，独立采样接口预留 uniform；推理支持 G/N/gamma 或合法非均匀全局时间网格，默认均匀四步。只更新方法接口并分析 Backbone，尚未选定或改造 Backbone，不实现完整训练链、不生成新数据。

## 交付入口

- [当前研究接收决定](acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md)、[当前机器状态](CURRENT_STATUS.json)
- [最终 Gate 结果与明确建议](gate_20261021/GATE_REPORT_ZH.md)、[最终机器判定](gate_20261021/GATE_RESULT.json)、[独立样例与复现说明](gate_20261021/README.md)
- [局部数值报告与阈值建议](reports/LOCAL_CHECK_REPORT_ZH.md)
- [历史局部状态](reports/SUMMARY.json)、[原建议 Gate](reports/PROPOSED_GATES.json)（原件保留；数值结果以最终 Gate 为准，后续研究接收状态见单独决定）
- [Method Spec](docs/METHOD_SPEC_ZH.md)、[V4→V5 变更表](docs/V4_TO_V5_CHANGELOG_ZH.md)
- [Simulator 实现](simulator_v5.py)、[冻结历史时间采样](time_paths.py)
- [当前起点与调度接口、CPU验证](method_interface_20261021/README.md)、[当前默认配置](method_interface_20261021/DEFAULT_V5_V1.json)
- [Backbone 候选比较与推荐](../v5_backbone_review_20261021/BACKBONE_COMPARISON_ZH.md)
- [FFT 决策及统计边界](audit_fft/FFT_DECISION_ZH.md)
- [NPU 算子与实际 kernel 证据](reports/npu_probe_summary.md)
- [后续数据接口和开销估算](docs/DATA_INTERFACE_ZH.md)
- [原始交接文档](docs/ORIGINAL_HANDOFF.md)

## 重放局部检查

所有命令从本目录执行。已有 fixture 应直接复用，不能重新生成后仍称作相同样例。de36 使用现有 `/data0/yanglin/envs/physical_bridge/bin/python` 与已安装 CANN；`run_npu.sh` 配置动态库、FP32、编译及 V5 私有缓存。

运行前先查看 `npu-smi info` 和设备进程归属，仅选择实时空闲卡，最多同时四张。下面的 `npu:3` 是示例，不表示固定预留。

```bash
./run_npu.sh --device npu:3 --part forward --output reports/replay_forward.json
./run_npu.sh --device npu:3 --part warp --output reports/replay_warp.json
./run_npu.sh --device npu:3 --part gradients --output reports/replay_gradients.json
./run_npu.sh --device npu:3 --part directional --output reports/replay_directional.json

./run_npu.sh --device cpu --part gradients --double-reference --output reports/replay_cpu64.json
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/data0/yanglin/envs/physical_bridge/bin/python scripts/check_time_paths.py
```

独立时间采样脚本在 de36 执行前同样需要 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，因为该环境的 PyTorch 会自动加载 torch_npu 扩展。检查脚本保存原始 JSON/NPZ、执行源码快照及运行起止输入哈希；进程正常退出不等于所有 `local_checks` 都为真，应读取报告中的具体结果。

本轮冻结测试尺寸为 RGB256、raw512。CPU64 是独立数值参考，不是 NPU 回退。现有结果只建立该固定配置下的执行和数值证据；未验证训练吞吐、多卡扩展或恢复质量。

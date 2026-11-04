# Overfit100 数据集交付报告

**状态：ready_for_overfit_training。** 数据构建、完整回放和独立 CPU 验收全部通过；可进入后续 trainer 适配与验收。本次没有训练、pilot 或恢复模型推理。

验收与本地归档完成：2026-09-16 19:31（北京时间）。

## 位置与数量

- 服务器：`liteserver-de36`；本次只使用物理 NPU 0，任务结束后已确认释放。
- 远端项目：`/data2/yanglin/v03_overfit100_g4321_n4`。
- 权威数据：`/data2/yanglin/v03_overfit100_g4321_n4/data/places_p2s_bridge_v03_overfit100_g4321_n4`。
- 数据大小：`17278657719 bytes`，约 **17.279 GB / 16.092 GiB**。
- 本地保留代码、配置、manifests、报告和日志；FP32 大数组留在远端。

| 对象 | 数量 |
|---|---:|
| 原图 / 固定 clean | 100 |
| 独立 native xi / 物理 trajectory | 400 |
| 四档真实 degraded state | 1,600 |
| 起点路径 | 1,600 |
| FP32 权威数组文件 | 2,500 |

## 数据协议检查

从原 train pool 的 2,000 个唯一 source ID 中，按 SHA256("source-select|20260916|" + source_id) 排序固定取前 100；ID 为原 manifest 的原始 basename，UTF-8 编码，不做字符串归一化。与原 val/test source overlap 均为 **0**。原图从原始 tar 读取，未复用旧 trajectory 的随机增强 clean。

每个 source 一次确定性 EXIF/RGB/resize/center crop；已重新从原始图片回放预处理并逐元素核对。每个 source 的四条轨迹引用同一份 clean/hash。每条轨迹只采一次两块原生 FP32 xi，并用同一 clean/xi 分别直接渲染 g=4/3/2/1。没有级联退化、重新采 xi、FP16 存储、额外 clamp 或 g=0 状态。

| 起始 G | 初始真实图 | severity_path | num_predictions |
|---:|---|---|---:|
| 4 | X_4 | [4,3,2,1] | 4 |
| 3 | X_3 | [3,2,1] | 3 |
| 2 | X_2 | [2,1] | 2 |
| 1 | X_1 | [1] | 1 |

num_predictions 表示网络总预测次数，包含最后在 g=1 的 clean 预测；one-shot=1。未来在 G=4/3/2 的同一初始观测上也可评估 one-shot。本任务只构建真实状态和路径元数据，不生成 predicted states，不冻结训练抽样频率、loss、优化器或训练预算。

## Replay 与完整性结果

- 全部 **1,600/1,600** 状态从磁盘加载 clean/xi 后重新渲染，全部逐元素相同。
- 实际全局 **max_abs=0，max MSE=0，mean MSE=0**。四个 g 分组各 400 项，误差均为 0。
- 全量构建前已冻结容差：`max_abs <= 1e-6` 且 `MSE <= 1e-12`；未在全量检查后调整。
- 全部 **400 个 realization / 800 块 xi** 按各自 seed 重建，全部逐元素相同；800 个 seed 唯一，两块数组的 400 份内容身份分别唯一。
- 原 source archive、原 source manifests、原训练 manifest、原模型/模拟器/推理代码和 assets 的前后 hash 一致。
- 完整性检查和独立 CPU reader 均通过。独立复核覆盖 **2,516 个 hash 文件与 2,500 个权威数组**。本地同步的 16 个数据元文件再次匹配远端 hash manifest。

### 未 clamp 的 simulator 输出统计

| g | 最小值 | 最大值 | <0 像素比例 | >1 像素比例 |
|---:|---:|---:|---:|---:|
| 4 | -0.06300162 | 1.045473 | 6.014506e-06 | 0.0006532542 |
| 3 | -0.001797102 | 1.00137 | 4.590352e-06 | 0.001327311 |
| 2 | -0.0001967438 | 1.001274 | 2.848307e-06 | 0.002227834 |
| 1 | -0.005013 | 1.00436 | 4.463196e-06 | 0.00276769 |

## 辅助视图

| 视图 | trajectories | starts | 不同 source |
|---|---:|---:|---:|
| overfit_1traj | 1 | 4 | 1 |
| overfit_16traj | 16 | 64 | 16 |
| overfit_400traj | 400 | 1600 | 100 |

三个集合按固定 SHA-256 顺序取前缀，嵌套关系、全部引用与主 manifest 一致；未复制数据或另采 source/xi。

## Manifest 身份

hash_manifest.json 的 SHA-256：

b896f9060a8ff012cafe476ec6c0bb1075e6291bfdec1eab7ff791dc8936b03b

hash manifest 覆盖所有权威数组、主/子 manifests、数据摘要与完整性报告，明确排除自身。

| Manifest | SHA-256 |
|---|---|
| source_manifest.jsonl | `cfcca03afc4fd61f17970973f1ad45eb0317170c45b3e7d4f0742c9306ff64f1` |
| start_manifest.jsonl | `9ee10f77a842f316301bff91adcda3ce3aa896c2c22cc25023866d573a7dd6d6` |
| trajectory_manifest.jsonl | `26870ae5b48cb175ca57c46c8c3eb3f6d86df05ef54cec2120979329f868767e` |

构建配置 SHA-256：`770d5ff52c9bee09ea254019b756a2f14ed63ada6e7aacde3842f657db4b5667`。

## 复查入口

- [完整性报告](data/places_p2s_bridge_v03_overfit100_g4321_n4/integrity_report.json)
- [独立 CPU 验收](artifacts/independent_cpu_acceptance.json)
- [全量逐状态 replay](data/places_p2s_bridge_v03_overfit100_g4321_n4/replay_metrics.jsonl)
- [逐 realization seed replay](data/places_p2s_bridge_v03_overfit100_g4321_n4/seed_replay_metrics.jsonl)
- [运行环境与原始输入 hash](artifacts/run_manifest.json)
- [设备释放与监督日志](logs/build_01/guard_result.json)
- [独立读取器](code/dataset.py)

原 simulator 的 protocol() 内保留原六档常量作为原版本信息；本数据的实际强度来自独立 build config，明确为 4/3/2/1。原 simulator 数学行为与所有原实验文件保持不变。

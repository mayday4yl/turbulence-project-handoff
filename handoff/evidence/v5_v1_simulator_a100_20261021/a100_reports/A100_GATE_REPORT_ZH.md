# A100 V5-v1 Simulator 验收

状态：**A100_GATE_COMPLETE_NOT_PASSED**。

相同候选在 A100 上仍未通过全部冻结要求；正式 V5 数据生成和训练不放行。

## 执行身份与范围

- A100：`h3c-gpu-server01`，账号 `zhuzhiyu`，物理 4 号卡；用户明确授权与现有作业并行。
- GPU UUID：`GPU-1a36bf37-948b-bd1e-a7cc-d47ec16a469b`；每个 CUDA 进程 allocator 上限 8 GiB，始终仅一个 CUDA 验收进程。
- 同环境 CPU/CUDA：PyTorch 2.7.1+cu126、CUDA build 12.6、cuDNN 90501；TF32 关闭、无 AMP。FP64 显式升级两个复数缓冲区为 complex128。
- Simulator、canonical 资产、四组固定样例和原 Gate 计划保留。三独立样例为 smooth/edges/natural，另保留原 fixed_case 全图差分。
- 新 harness 只适配设备/同步/保存；原脚本、阈值、步长和失败证据未改。FP64 是 canonical FP32 资产的精度提升，不重建物理滤波器。
- 本次没有生成正式数据、加载恢复网络或启动训练。
- 全部 CUDA 检查中峰值 allocated 1.925 GiB，峰值 reserved 3.697 GiB，均低于 8 GiB 上限。

全部 12 次主样例中，零端点输出/C identity VJP/ξ 梯度误差均精确为 0；独立 FFT 前向最大绝对误差 2.1314264e-06，FFT 梯度最大相对 L2 4.5768922e-07，均通过原阈值。S1 核质量、显式卷积和常量保持检查通过。

## 三独立样例

| 样例 | 设备/精度 | g=1e-4 最大差（≤0.002） | 局部 Zernike 连续两步 | 全图 C/Zernike/Tilt 连续两步 | 失败断言 |
|---|---|---:|---|---|---|
| smooth | cpu / float32 | 2.42590904e-05 | FAIL | PASS/PASS/PASS | local_FD |
| smooth | cpu / float64 | 2.42106721e-05 | FAIL | PASS/PASS/PASS | local_FD |
| smooth | cuda:0 / float32 | 2.42590904e-05 | FAIL | PASS/PASS/PASS | local_FD |
| smooth | cuda:0 / float64 | 2.42106721e-05 | FAIL | PASS/PASS/PASS | local_FD |
| edges | cpu / float32 | 0.00217600144 | FAIL | PASS/PASS/FAIL | g1e-4_atol_2e-3, local_FD |
| edges | cpu / float64 | 0.00216673668 | FAIL | PASS/PASS/PASS | g1e-4_atol_2e-3, local_FD |
| edges | cuda:0 / float32 | 0.00217600144 | FAIL | PASS/PASS/FAIL | g1e-4_atol_2e-3, local_FD |
| edges | cuda:0 / float64 | 0.00216673668 | FAIL | PASS/PASS/PASS | g1e-4_atol_2e-3, local_FD |
| natural | cpu / float32 | 0.0013962388 | PASS | PASS/PASS/FAIL | 无 |
| natural | cpu / float64 | 0.00139001796 | PASS | PASS/PASS/PASS | 无 |
| natural | cuda:0 / float32 | 0.0013962388 | PASS | PASS/PASS/FAIL | 无 |
| natural | cuda:0 / float64 | 0.00139001796 | PASS | PASS/PASS/PASS | 无 |

FP32 全图差分保留为原压力诊断；FP64 全图差分和局部差分仍按冻结的连续两步规则判定。全图通过不覆盖局部失败。

## 后端一致性

| 比较 | 最大输出绝对差（≤2e-5） | 最大梯度相对 L2（≤1e-3） | 结果 |
|---|---:|---:|---|
| 同机 CPU/CUDA smooth_float32 | 2.3841858e-07 | 0.00082080601 | PASS |
| 同机 CPU/CUDA smooth_float64 | 5.5511151e-16 | 7.8982849e-14 | PASS |
| 同机 CPU/CUDA edges_float32 | 1.5258789e-05 | 0.00021291004 | PASS |
| 同机 CPU/CUDA edges_float64 | 2.797762e-14 | 6.2586903e-15 | PASS |
| 同机 CPU/CUDA natural_float32 | 8.4638596e-06 | 0.00052837248 | PASS |
| 同机 CPU/CUDA natural_float64 | 1.7930102e-14 | 6.9162239e-15 | PASS |
| 同机原 fixed_case CPU/CUDA float32 | 1.0073185e-05 | 8.2098144e-05 | PASS |
| 同机原 fixed_case CPU/CUDA float64 | 1.9151347e-14 | 5.2423132e-15 | PASS |
| 跨机 CUDA32/NPU32 smooth | 2.3841858e-07 | 0.00012694038 | PASS |
| 跨机 CUDA32/NPU32 edges | 1.5318394e-05 | 0.00021288601 | PASS |
| 跨机 CUDA32/NPU32 natural | 8.4638596e-06 | 0.00053072789 | PASS |

跨机 NPU 对照同时存在软件栈、主机和加速器差异；判断 CUDA 后端差异优先看同机 CPU/CUDA。原 NPU32 为 de36 torch 2.7.1+cpu / torch_npu 2.7.1.post2。

## S1/S2、严格时间和原固定样例

| 设备/精度 | 部件必检 | 原 fixed_case 全图 Tilt 最小相对误差 | 连续两步通过 | 峰值 allocated / reserved GiB |
|---|---|---:|---|---|
| cpu_float32 | PASS | 0.010777268 | FAIL | CPU |
| cpu_float64 | PASS | 5.5853523e-10 | PASS | CPU |
| cuda_0_float32 | FAIL | 0.010776496 | FAIL | 0.915 / 2.010 |
| cuda_0_float64 | FAIL | 1.519146e-09 | PASS | 1.822 / 3.697 |

部件包含：S1 g4 与未 Smooth blur、原 V4 控制、S2 零/整数/非整数/±300px 边界及梯度、mixed batch、非法 g、目标设备上 10000 条严格内部时间。边界原绝对误差与冻结的独立 FP64 累加相对误差分开保存。

冻结 V4 wrapper 只支持 FP32。FP64 部件 harness v2/v3 让旧版原样重放控制保持 FP32，其余 V5 检查保持 FP64；早期 harness 错误日志完整保留，这不是物理修复。CPU64 v3 仅缓存 raw-ξ 差分中不变图像的卷积结果；C 的差分与全部 autograd 仍完整执行。

## 有界计时

| 设备/精度 | 完整前向中位数 s | 完整前向+反向中位数 s |
|---|---:|---:|
| cpu_float32 | 1.47981 | 3.99876 |
| cpu_float64 | 4.57256 | 10.9972 |
| cuda_0_float32 | 0.0230426 | 0.100302 |
| cuda_0_float64 | 0.0179446 | 0.0998036 |

每项为 B1/RGB256/g4，1 次预热 + 3 次测量，CUDA 前后显式同步。共享卡同时跑其他作业，不是独占吞吐，也不能外推训练 ETA；只可作为后续 Y4 单样例计算开销的观测。

## 结论与停止线

1. A100 可执行同一 Simulator 的 CUDA FP32/FP64 前向、反向和数值验收。具体数值通过范围以上表为准。
2. 边缘图 g=1e-4 的绝对误差与局部 Zernike 原步长失败在 CPU/CUDA、FP32/FP64 中复现时，不能归咎为 NPU 特有问题，也不能靠换硬件消除。
3. FP64 对小步长数值差分的改善，与原近零/分段断言是否通过是不同问题。更小步长诊断不覆盖原 FAIL。
4. 旧 P2S 完整复数 IFFT、非平稳协方差、冗余 raw 方向与 PSF 负系数的局限仍保留；未作完整物理认证。
5. 在原 Gate 缺口关闭前，不足以支持正式 V5 数据与训练。本轮验收后停止。数值断言未通过不等于已经证实网络无法训练或无法收敛；本轮没有做该实验。

关键文件：`A100_GATE_RESULT.json`、`A100_EXECUTION_PLAN.json`、`HARNESS_DIFF.patch`、`HARNESS_REPAIR_V2.json`、`../a100_results/`、`../a100_logs/`、`POSTFLIGHT_A100_FINAL.json`。

## 新发现的旧版控制差异

CUDA 的 all-off V4 重放最大差 4.2617321e-06，仍超过原 1e-6 标准，保留 FAIL。逐层比较 z、displacement、weights、blur 均为 0；第一处差异是旧 grid 的标量 /255 与 tensor /255，grid 最大差 2.38418579e-07。
在诊断中只换回原 V4 grid，输出与原 V4 最大差为 0；当前正式候选 S2 使用 /256，其标量/张量写法差为 0。因此本样例可定位为旧控制分支的 CUDA 算术舍入差异，不是本次观察到物理缓冲区或 P2S 输出变化。诊断不改变原 FAIL，也没有修改候选。

## 同机 CPU64 失败分解

- smooth：g=1e-4 full 最大差 2.42106721e-05，仅 blur 1.71688336e-07，仅 warp clean 2.4158113e-05，最大位移 0.00231644463 px。
  局部 Zernike 补充 h=3e-4/1e-4 相对误差为 6.112e-11/3.726e-10，逐层 ReLU 变号数见原始 JSON；这些新步长只用于解释分段，不覆盖冻结的 .03/.01/.003/.001 失败。
- edges：g=1e-4 full 最大差 0.00216673668，仅 blur 1.48167722e-05，仅 warp clean 0.00215612058，最大位移 0.00182856139 px。
  局部 Zernike 补充 h=3e-4/1e-4 相对误差为 1.61e-12/2.936e-12，逐层 ReLU 变号数见原始 JSON；这些新步长只用于解释分段，不覆盖冻结的 .03/.01/.003/.001 失败。

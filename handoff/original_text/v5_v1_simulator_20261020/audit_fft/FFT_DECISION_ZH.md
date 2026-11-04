# V5-v1 Tilt FFT 候选审计

**2026-10-21 决策更新：** 用户已明确 V5-v1 暂定采用本文的旧 P2S 完整复数逆变换语义，保留 raw-ξ 接口、采样分布和物理参数；非平稳协方差作为局限记录，不新增采样方法。按既有提议阈值执行的最终 Simulator Gate 见 [最新报告](../gate_20261021/GATE_REPORT_ZH.md)：独立 FFT 检查通过，整个 Simulator 仍有其他必检失败。下文第 5–6 节的“待确认”属于当时审计建议，最新执行状态以本更新及最终报告为准。

本审计确认：保持 `[B,512,512,2]` 的 raw Tilt 接口及其两组独立标准正态采样，可以修复完整复数频谱被误作 one-sided 输入的问题。最小候选是逐方向执行完整复数逆变换并取实部，与归档旧 P2S 的两次 `torch.ifft(...)[...,0]` 数学等价。本报告只认证这个等价关系、数值性质和局部梯度；没有认证旧 P2S 的目标物理协方差，也没有执行正式数据生成或训练。

本机参考环境：`/opt/anaconda3/bin/python`，PyTorch `2.9.1`，CPU、2 个线程。NPU 必须用同一 fixture 独立检查，CPU PASS 不能代替 NPU PASS。全部数字、源码/资产 SHA 和候选阈值见 `FFT_AUDIT.json`。

## 1. 修复目标与保留项

令 `S[k]` 为现有 `tilt_s_half_unit`，`a(g)=g^(5/6)`，`c=tilt_constant`，`ξ_x,ξ_y` 为原接口的两个独立实标准正态数组。候选为：

```python
frequency = S[None, :, :, None] * a[:, None, None, None] * xi
D = torch.fft.ifft2(frequency, dim=(1, 2), norm="backward").real * c
```

保留固定滤波器、阈值 `0.02`、强度幂、常数、原 raw 数组维度及分布、两个独立 x/y 方向和最终 crop。`norm="backward"` 的二维逆变换归一化为 `1/(N*N)`，没有另行加入振幅补偿。滤波器虽命名 `s_half`，实际保存的是 `N×N` 完整复数数组，名称不构成半谱依据。

V4 的 `stack(S.real,S.imag)` 配合实数 `irfft2(...,dim=(1,2),s=(512,512))`，把两个实/虚分量分别分配给 x/y，且沿最后变换轴只消费前 257 列。旧 P2S 原先对 x、y 各自用完整的复数 S。这是已证实的接口语义差异。

参考源码：

- `v4_nafnet_sidd116m_candidate_20261005/physical_bridge/simulator.py`。
- `output/v4_preparation/sources/PiRN_12b9ef6a32c7/assets/simulator_new.py`。
- `output/v4_preparation/sources/TurbulenceSim_P2S_5b743872bc66/simulator_old.py:78`。
- `output/v4_preparation/sources/TurbulenceSim_P2S_5b743872bc66/turbStats.py:95`。

PyTorch 官方接口说明：[irfft2](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft2.html)、[ifft2](https://docs.pytorch.org/docs/2.14/generated/torch.fft.ifft2.html)。前者解释输入为 Hermitian one-sided 频谱，后者接受完整复数输入；调用的变换轴与实/虚表示不能相互替代。

## 2. Hermitian 问题及等价的半谱表达

`S*ξ` 一般不是 Hermitian 频谱，完整 IFFT 的虚部也不要求接近零。候选里的 `.real` 是旧 P2S 明确规定的实部投影，不是声称原频谱已经 Hermitian。本次固定样例的虚部/实部 RMS 比约 `0.3673`。

如果因后端兼容性需要等价的 `irfft2` 表达，必须先构造：

`H[k] = (F[k] + conj(F[-k])) / 2`，其中 `F=S*ξ*a`，两个频率轴均按模 N 取负。

再取 `H[:, :, :257]` 执行 `irfft2(...,s=(512,512),dim=(1,2))`。这是 `Re(IFFT(F))` 的严格等价式，后半频率通过 H 进入计算。不能直接把 F 或 raw ξ 裁至 257：本次直接裁切的位移 RMS 误差为 `0.844224 px`。

## 3. 数值与梯度结果

下表使用 SHA 校验通过的归档 canonical 固定滤波器；本机重建滤波器的同项也通过。

| 检查 | 实测 | 当前判定 |
|---|---:|---|
| FP64 完整 IFFT 与独立 NumPy 参考最大绝对误差 | `5.329e-15 px` | PASS |
| 完整 IFFT 与 Hermitian 投影后 irfft2 误差 | `7.105e-15 px` | PASS |
| 修改 raw 后 255 列，V4 输出变化 | `0` | 旧问题复现 |
| 同一修改，候选输出最大变化 | `2.922070 px` | PASS |
| 后 255 列 raw 对裁剪后 probe loss 的最大梯度 | `1.100e-3` | PASS |
| FP64 有效方向有限差分，ε=`1e-2…1e-5`，最差相对误差 | `2.202e-14` | PASS |
| `g=0` 位移与对 raw ξ 梯度 | 均严格为 `0` | PASS |
| `g=0,1e-8,1e-6,1e-4,.01,.25,1,2.75,4` 的 `g^(5/6)` 缩放 | 误差低于 `1e-10 px` | PASS |
| FP32 位移与同输入 NumPy FP64 参考最大绝对误差 | `2.341e-6 px` | PASS（CPU） |
| FP32 同项相对 L2 | `7.814e-8` | PASS（CPU） |

有限差分目标是 `mean(D[:, :256, :256] * probe)`，方向为该目标 raw ξ 梯度的 RMS 归一化。它验证至少一个有效方向和完整线性 FFT 反向链，不把无效方向排除后宣称所有 raw 维度有效。整体图像 loss 的 warp/边界梯度检查属于主验收。

## 4. 零梯度子空间

现有硬阈值只保留 **41/262144** 个频点；其余位置的零梯度是固定滤波器的定义。本次修改所有被 mask 的 raw 分量，候选输出变化严格为 0，梯度也严格为 0。

此外，实部投影与近似共轭配对带来额外冗余。对 256×256 crop 取 418 个空间探针，在相对奇异值阈值 `1e-7` 下，41 个有效频点的一方向算子数值秩为 21。这个值是已保存 FP32 滤波器的有界诊断，不是全物理模型秩的证明；不能要求 512×512 的所有 raw 参数都收到观测项非零梯度。

因此本轮不移除硬阈值、不压缩网络输出、不改变 ξ 邻近正则。若以后更改这些项，应单独定义方法变化。

## 5. 物理统计边界：旧 P2S 不是自动正确的物理验收基准

对一个方向，设 `M=N²`，`p,q` 为像素位置，则候选的精确协方差是：

`Cov(D[p],D[q]) = c² a² / (2M) * ( Re(IFFT(|S|²))[p-q] + Re(IFFT(S²))[p+q] )`。

第二项依赖 `p+q`，因此该实 iid 频域采样器一般不是平稳随机场。这个结论来自原线性算子的解析推导，本次还直接求和核验：方差公式与频点系数平方和最大误差 `4.44e-16 px²`。

canonical 的 `g=1` 单方向方差：

| 像素位置 | 解析方差（px²） | 32768 组独立有效频率抽样的经验二阶矩 |
|---|---:|---:|
| `(0,0)` | `2.421848` | `2.423636` |
| `(17,19)` | `2.472148` | `2.466592` |
| `(128,128)` | `2.920839` | `2.923079` |

crop 内最大/最小方差约差 20.6%。x/y 使用独立 raw 数组，所以其总体交叉协方差为零，单次或 32 次有限场抽样的相关系数不必接近零，不能用少量样本的 RMS 比例充当物理各向同性验收。

这不是新增的候选统计机制，而是恢复旧 P2S 数学式时继承的性质。将 ξ 改解释为空间白噪声再 FFT、改为独立 Hermitian 半谱高斯采样、重做 covariance PSD / `ifftshift` / 负特征值处理，都可能改变位移统计及物理定义。本轮没有执行这些替换。

**决策建议：** 将当前式标记为“旧 P2S 完整复数语义修复候选”，允许其用于 V5 局部数值检查；最终 Gate 前确认接受此统计定义，或单独规定目标协方差后再研究采样器。当前数学检查 PASS 不等于物理 Gate 已通过。

## 6. 可复现阈值与 NPU 参考

阈值均为建议，尚待最终 Gate 确认：

- CPU FP64：完整 IFFT / NumPy / Hermitian 等价最大绝对误差 `≤1e-10 px`。
- CPU FP64：有效方向有限差分相对误差 `≤1e-8`。
- NPU FP32：同 fixture 相对 NumPy64，逐点 `atol=2e-5 px, rtol=1e-4`；梯度相对 L2 `≤1e-4` 且最大绝对误差 `≤1e-6`。先观察实测再判断是否应调整，不能为让失败通过而事后无说明放宽。
- 零强度位移及 ξ 梯度应严格为 0；有效后半轴扰动需有非零响应。mask 区域无响应是预期。

`FFT_REF.npz` SHA256：`b0de24ed8c7cfb712f90a46de50d8c0ff967701483700099d272bd58d5d893ad`，约 16.1 MiB。

包含 `xi[2,512,512,2]`、完整 `spectrum[512,512]`、`g=[1,4]`、amplitude、constant、CPU FP32 / NumPy FP64 完整位移、`probe[2,256,256,2]`、CPU FP32 raw 梯度。没有 clean 图像、网络权重或正式数据。

```bash
/opt/anaconda3/bin/python v5_v1_simulator_20261020/audit_fft/audit_fft_candidate.py
/opt/anaconda3/bin/python v5_v1_simulator_20261020/audit_fft/check_fft_fixture.py --device cpu --output v5_v1_simulator_20261020/audit_fft/CPU_FIXTURE_CHECK.json
```

在已经核验可用的 NPU 上，用对应 PyTorch/torch_npu 环境执行 `check_fft_fixture.py --device npu:0 --output <report.json>`。该脚本不会自动回退 CPU；若 complex/IFFT/backward 不支持，返回 `UNSUPPORTED_OR_ERROR` 及错误，需另行验证数学等价的后端适配。

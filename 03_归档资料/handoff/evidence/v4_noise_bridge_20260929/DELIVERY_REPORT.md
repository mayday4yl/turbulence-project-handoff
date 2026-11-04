# 新 V4 implementation + correctness 交付

状态：**当前逐样本精修实现已通过下述真实 A100 验收；严格 raw-xi 重复性和原批量精修轨迹对照未通过，不能宣称全部数值确定性通过。** 本轮已停止，未启动正式训练或效果 Pilot。

## 1. 改了哪些地方

独立本地目录：`/Users/lucas/Desktop/turbulence IR/v4_noise_bridge_20260929`。
A100 独立目录：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_rgb_20260929`。
实际主机 `h3c-gpu-server01`，SSH `10.94.2.1`，验证账号 `zhuzhiyu`；仅物理 GPU5。

| 文件 | 变更 |
|---|---|
| `physical_bridge/model.py`、`simulator.py` | V3 原始快照逐字复制；哈希与原件一致。没有改 simulator 数学。 |
| `v4/model.py` | V3 网络继承适配：6 通道、raw `[g,t]` 的 2→256→256 SiLU MLP；保留双 decoder、clean Sigmoid、两块 raw-xi 布局和正式零头初始化。 |
| `v4/data.py`、`normalization.py` | 已有真实数据的 train/Overfit100 身份绑定与读取；FP64 Chan 统计、四档等权、独立 artifact、FP32 非训练 buffer 与 artifact hash。 |
| `v4/core.py` | 固定原始 Y、每样本独立且路径共享 eta、完整尾路径、active 索引、每级原始 G/Y 回投、path mean→batch mean、full-BPTT。 |
| `v4/refinement.py`、`inference.py` | 当前 xi_hat 初始化/固定中心、合计元素分母、固定 Adam/41候选；公共执行逐样本串行，每样本每级新 optimizer；末级不精修。 |
| `v4/runtime.py` | 实际 import/源码/资产哈希核验、FP32 CUDA 设置、完整临时 checkpoint 与独立噪声 RNG 往返。 |
| `scripts/`、`tests/` | 真实 CUDA 验收、CPU 小型合同检查、失败诊断与持久 optimizer-step 限额。 |

原 V3 代码、数据、checkpoint 未写入；A100 已有数据/固定资产只读。22 个交付源码/配置/说明文件与服务器逐项一致，实际 import 均来自独立实现目录。[代码映射](<../../NOT_INCLUDED.md#ref-15bfe0c4944589d9905c>) · [源文件清单](<../../NOT_INCLUDED.md#ref-a1b3f38c7ebc3abfd58d>) · [实际 import 与释放记录](<../../NOT_INCLUDED.md#ref-540fe4e168e0c729225c>)

工程修复记录：系统 Python 版本不足时改用已核验的项目 Python3.10；区分生成前 source plan hash 与生成后 source manifest hash，后者绑定现有 ASSET_LOCK；修正 CPU 测试预期值的整数 dtype。它们均未改变方法。

## 2. 是否与当前图及文字一致

没有发现实质图文矛盾，也没有方法定义偏差。当前实现为 T0=0.8、K=G、g=G…1、K−1 次状态生成、原始数值 `[g,t]`，不独立输入 G。训练每级 `LC+0.2LR`，含末级，LR 匹配原始 Y/G。训练不精修、不 detach、不引入 GT-xi/弱状态监督；原 simulator 参数冻结、输入图保留。

推理保持原目标和全部指定超参数：两块 raw-xi 合计 `d_xi=9,961,472`；lambda=.01；Adam lr=.01、betas=(.9,.999)、eps=1e-8、wd=0；40 更新；从初始化与更新后共41个候选按总 J 选择成对结果。没有 clamp、投影、梯度归一化或新停止规则。

唯一需明确的执行选择是**逐样本串行精修**：原向量化40步对照未通过原定 raw-xi 容差，已退出公共执行路径。用户要求允许逐样本实现，这不改变每个样本的算法；它也没有解决 CUDA 本身的重复运行非确定性。

实际统计绑定 **Overfit100 训练子集**，400条轨迹×4档=1600个唯一状态，每档400张，仅全幅256 RGB固定视图。mu = `[0.4587469524, 0.4475286356, 0.4201560764]`，s = `[0.2491986447, 0.2523667266, 0.2715907645]`。未把它冒充完整500张训练集的统计；将来训练总体改变需重新绑定统计 artifact。

[统计 artifact](<../../NOT_INCLUDED.md#ref-024daf98b6db8b07b676>) · [真实统计复核](<../../NOT_INCLUDED.md#ref-7d8e81a6cef0dcd53ea8>)

## 3. 哪些已通过、哪些未通过或未验证

全部 CUDA 检查使用实际32,038,377参数网络、256 RGB、512 native raw-xi、真实固定 P2S 资产和现有 A100 数据，FP32，TF32/autocast关闭。CPU/toy 与真实 CUDA 结果分开记录。

| 检查 | 实际结果及证据 |
|---|---|
| CPU统计与精修合同 | **20 tests通过**：流式/直接统计、块划分、档间均值差、非法数据、冻结buffer、公式/不clamp、合并分母、独立Adam、候选和错误处理。[日志](<../../NOT_INCLUDED.md#ref-9695c0c521726d5b3cbd>) |
| G=1/2/3/4 前向 | **通过**。乱序混合 `[2,4,1,3]`、不同源图，active从 `[0,1,2,3]→[0,1,3]→[1,3]→[1]`。混合/逐样本最终clean最大差8.5831e-6，路径loss差0；完整G4四次预测、三次transition；没有g0。[证据](<../../NOT_INCLUDED.md#ref-6b72b29e553d6d504388>) |
| 混合噪声/输入 | **通过**。t=0精确等于当前物理状态；直接RGB与归一化逆公式最大差2.3842e-7；真实Z范围至少 `[-0.5674,1.4062]`，入口hook确认未截断；Y固定、eta按原索引对应。见前向证据。 |
| 仅末级LC的full-BPTT | **通过**。正式零头初始化下，G4前三阶段clean、两块raw-xi和生成状态梯度均非None、有限、非零；无需改初始化或启用诊断替代模型。[证据](<../../NOT_INCLUDED.md#ref-1773dbe84defad7742a2>) |
| 中间detach负对照 | **通过**。测试hook在stage2入口断链，stage1四项梯度均None，stage2/3仍连接；前向差0。没有靠共享参数或局部损失证明连接。见BPTT证据。 |
| 混合G梯度/reduction | **通过**。混合batch反传与逐样本loss/B累计的全部参数梯度比较，global relative L2=`8.6816772e-7`，低于预设1e-4；参数和simulator均不更新。[证据](<../../NOT_INCLUDED.md#ref-4071ae4a0341b8c0ff88>) |
| 逐样本真实精修 | **通过**。5个真实样本-阶段求解共200次xi更新，205个候选；同一次实际梯度驱动独立float64数学Adam核对，参数最大差1.9291e-8，低于预设atol2e-7/rtol1e-6。所有选中两块xi哈希与相应最小J候选一致。[证据](<../../NOT_INCLUDED.md#ref-edec6ae1abb387b081d1>) |
| 精修不可变对象与reset | **通过**。网络/simulator参数、clean、Y/G、当前中心哈希不变；G4每级独立optimizer、G1和末级无精修；G1 on/off输出精确相同。见精修证据。 |
| 临时参数更新/checkpoint | **通过，恰好1次训练optimizer step**。参数实际改变且有限；simulator不变；模型、optimizer、mu/s、artifact identity、CPU/CUDA/独立eta RNG精确往返，临时checkpoint已删除。[证据](<../../NOT_INCLUDED.md#ref-61019cb14714182c837e>) · [总次数账本](<../../NOT_INCLUDED.md#ref-479f0736e8c7a8fea0d4>) |
| 原批量40步raw-xi对照 | **未通过、公共执行已停用**。初次最大差约0.0624，没有放宽原容差或改判。[原失败](<../../NOT_INCLUDED.md#ref-651bc91898350c28a0d0>) |
| 同一样本40步raw-xi重复性 | **未通过严格容差**。独立重复最大差约0.0999，即使总J只差约1e-6也不能称xi轨迹一致。[诊断](<../../NOT_INCLUDED.md#ref-e9e6ea5194769124ef4f>) |
| 同一点的精修梯度尺度 | **通过**。批量/单样本zernike梯度relative L2≤1.532e-7，norm ratio≈1；tilt精确一致，排除所测条件的1/B梯度缩放错误。这不能替代40步轨迹对照。[证据](<../../NOT_INCLUDED.md#ref-b743f8aa01777f2b91ab>) |

严格确定性模式在真实反传直接报告 `grid_sampler_2d_backward_cuda does not have a deterministic implementation`。这证明当前后端不支持严格确定性保证；不能仅据此断言它是所有40步差异的唯一成因。

可复现命令、环境路径与步骤见 [README](<README.md>)。典型检查峰值：混合反传约23.98GiB；单条G4末级反传约10.06GiB。上述为本轮检查的实测，不是训练吞吐或资源预算。

## 4. 真正 blocker 与停止状态

**当前支持的逐样本算法实现没有缺失资产、断图或目标/更新规则 blocker。** 尚未验收的向量化精修加速路径和严格raw-xi重复性仍受上述问题阻塞；不能承诺它们已解决。当前保存的固定噪声seed能复现eta，不承诺精修xi逐元素复现。

本轮完成 implementation + correctness 后停止：正式runs目录仍为空，只做1次临时模型optimizer更新，未启动正式训练、独立T0对照、收敛/PSNR效果Pilot或T0/lambda搜索。最终GPU5无计算进程，临时checkpoint无遗留。[最终释放/身份检查](<../../NOT_INCLUDED.md#ref-540fe4e168e0c729225c>) · [机器可读摘要](<../../NOT_INCLUDED.md#ref-bc9d8969cdc5f1632f28>) · [证据哈希](<../../NOT_INCLUDED.md#ref-f0ed0768b373948eceb8>)

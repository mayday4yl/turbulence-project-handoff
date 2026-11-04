# PiRN 四方法适配验收

**结论：D0 / R0 / R08 / L08 四组真实 GPU 功能验收全部通过。正式训练未启动，网络 optimizer updates = 0。**

本次是 backbone 替换工程验收，未使用 test200。验收中的 metrics 来自未训练的共同初始化和少量固定训练样本，不能作为方法效果或泛化结果。

## 实测资源

GPU5/6 共卡、每卡依次两组。从启动前资源快照至最后一组完成为 140.12 秒；不包含前期 CPU 准备、代码编写和最后文件下载。其他用户进程保留，验收进程已全部退出。

| 方法 | GPU | 检查阶段用时 (s) | 峰值已分配显存 (GiB) | B32 micro1/micro2 梯度相对 L2 |
|---|---:|---:|---:|---:|
| D0 | 5 | 24.69 | 3.59 | 3.297e-05 |
| R0 | 6 | 61.49 | 14.19 | 8.663e-06 |
| R08 | 5 | 60.86 | 14.19 | 4.893e-06 |
| L08 | 6 | 43.16 | 13.84 | 4.055e-06 |

梯度一致性阈值为 1e−4，四组均通过。每个进程的 PyTorch allocator 上限为 32 GiB，组开始前要求至少 48 GiB 空闲。检查阶段用时包含该组多项检查，不是稳定训练 sec/update；本次没有估计完整训练 ETA。

## 已验证内容

- 当前 train500 冻结 μ/s、8000 starts 与共同初始化身份一致；四组都是同一份全新 PiRN 权重。
- G1/G2/G3/G4、单样本与混合批次对齐；原 Y/G/η 保持身份；clean GT 只参与 loss。
- D0 单次预测且 clean/ξ 全部纳入 optimizer；R0/R08 物理递归；L08 精确使用原线性状态公式，并保留物理 LR。
- 原 path-mean LC + 0.2 LR、full BPTT 与截断负对照通过；L08 末端 clean loss 对早期 ξ 没有依赖是预期语义，不是断图。
- 有效 B32（各 G 八个 starts），microbatch1 与 microbatch2 梯度和 loss 一致，全部参数梯度有限；没有执行 network optimizer.step。
- Y/G-only 统一推理接口与训练 objective 的 RAW 输出一致；D0/L08 的 refinement-ON 拒绝为 N/A。
- R0/R08 在真实 G4 样本各经过 3 个合法精修阶段，每阶段逐样本 40 步、41 candidates；网络和模拟器哈希不变。临时 ξ Adam 更新不属于训练网络更新。
- U0 RAW/EMA/空 optimizer/RNG/数据游标保存恢复；错误方法、架构、旧统计、stream、初始权重身份或缺失统计 buffer 均被拒绝。
- 实际 PSNR/SSIM/LPIPS 与 LC/0.2LR/total 管线完成样本 COMMIT，所有文件哈希核验通过。
- 38 项父来源文件、16 项原训练状态/日志/指针等文件保持不变；旧正式训练仍按用户要求停止。

## 使用入口及范围

统一建模入口为 `pirn_v4.factory.build(config, group, device)`，当前 `v4.matched.build` 已接入该 factory；统一推理入口为 `pirn_v4.inference.infer_group`。四组配置在 `configs/groups/`。

共同初始化参数量：17,611,761。文件 SHA256：`d5e928344c05bebb9a0939472b193f5a4ef7c2601ccc38c9713be248bf21a512`。

名称为 PiRN 仓库 SwinIR/RSTB backbone + V4 条件/双输出适配，未加载预训练 PiRN 权重，不称为整套官方 PiRN 方法复现。

后续正式训练仍需确定预算与执行冻结，并接入新正式运行的完整状态管理。本次只验证 U0 保存恢复，未验证非零 optimizer moments 续训；未运行学习曲线、全量 train8000/val800 或方法效果比较。`scripts/train_worker.py` 仍拒绝直接开训。

## 证据存储

本地包含代码、配置、共同初始化、全部 JSON/日志及指标管线样本。五份较大的 U0 验收状态 `.pt` 保留在远端独立目录；其逐项哈希验证在 `FINAL_ACCEPTANCE_AUDIT.json`，本地未把这些验收状态冒称为正式训练 checkpoint。

CPU D0 验收先完成；CPU R0 在用户授权 GPU5/6 共卡后被结束，以避免重复计算，证据保留在 `cpu_acceptance_v1/INTERRUPTED_FOR_GPU_VALIDATION.json`。最终结论由完整 `gpu_acceptance_v1` 四组 PASS 支持。

# 当前交接版本：先读、再跑

状态基准：2026-11-04。现有训练已结束；本次交接验证仅运行小规模模拟器/恢复示例，0次优化器更新。

## 1. 两个可以直接接着使用的模块

|项目|当前交接对象|用途|远端入口|
|---|---|---|---|
|物理模块|V5 Smooth + pixel-center warp + legacy full IFFT Simulator|生成物理退化路径；后续可接入新consistency方法|`/data2/yanglin/v5_v1_simulator_20261020/simulator_v5.py`|
|恢复基线|UniDB D0，原生条件UNet与100步reverse SDE|运行最新恢复权重，作为新方法的比较和工程参考|`/data2/yanglin/unidb_d0_20261102/core.py`|

V5是当前复用模拟器的版本名；UniDB是最新恢复实验。名称不代表两者已经耦合，UniDB当前没有raw-ξ头、外层模拟器或会议提出的新consistency loss。

## 2. 最新权重从哪里开始

默认示例使用**原路线U67180的EMA**：它是这条路线的best＝latest，不是跨项目所有模型的最优声明。

```text
/data2/yanglin/unidb_dual_20261104_until00/runs/original/checkpoints/U00067180.pt
SHA256: 0381ae3435d2939a2b8b4fbc215c3f78ac6c3775f254bead31dee3e9cdb78e98
```

完整checkpoint中RAW键为`network`，EMA键为`ema.network`。用于推理读取EMA；完整续训还需optimizer、RNG、loader/cursor等状态，不能只载EMA就称为resume。另一条实验路线的best是U10000、latest是U40848，详见[权重登记](../../handoff/registers/WEIGHT_REGISTER.csv)。

## 3. 当前效果与限制

原路线U67180的正式G4全量EMA结果为16.211783dB，对应输入为18.795938dB；最终五档PSNR均未超过相应输入。因此本次交接的是已完成实验、可运行模块及其局限，尚未达到恢复质量目标。表格与固定样本见[最终实验报告](../../handoff/evidence/unidb_dual_20261104_until00/report/FINAL_REPORT_ZH.md)。

模拟器已有局限包括原Gate部分近零/差分FAIL，以及未认证的时间导数、高阶梯度和任意物理参数/分辨率；详见[模拟器说明](SIMULATOR_GUIDE.md)。

## 4. 接手当天的运行顺序

1. 用[NPU运行指南](NPU_RUNBOOK.md)中的授权登录方式进入de36。
2. 指向现有V6解释器，核对三条.pth依赖；对新交接包执行完整性校验。
3. 执行`demo.py inspect`检查前置文件/小型配置；这个PASS不代替SHA或设备兼容性检查。
4. 跑CPU模拟器，查看固定ξ多强度图；在分配到的NPU上重复示例。
5. 严格加载U67180 EMA，再跑原生100步恢复。当前已有这两类示例的[实测记录](VALIDATION.md)。

所有新输出写接收者新目录。训练数据、权重和vendor继续按现有登记取得；本包没有迁移全部服务器资产。

## 5. 下一步要接什么

接手方提供其实际consistency/CGDB代码与论文版本后，确认时间方向、共享ξ、模型输出、目标函数、采样器和导数需求，再将物理路径接入。具体[接口清单](METHODS_AND_NEXT_STEP.md)已列出。

历史worker绑定旧目录、截止、源SHA与资源，接手后的正式训练需要在新副本里适配；本次没有把旧命令包装成未经验证的新训练入口。

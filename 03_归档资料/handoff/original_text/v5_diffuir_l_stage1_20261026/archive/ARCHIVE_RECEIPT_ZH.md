# ResShift 原地归档与保留回执

2026-10-26，北京时间。用户已明确将下一版 V5 Clean Restoration Backbone 改为 DiffUIR-L；ResShift Stage 1、梯度/VQ 审查和四次 VQ 往返成果保留为对照，不继续开展 ResShift 正式训练。

本次归档完成于 **21:27:35**，逐文件重新读取验收于 **21:27:48** 通过。归档方式是保留三个完整原目录，建立全量内容 SHA256 清单，并在本目录复制关键报告和身份记录。未移动、删除、覆盖、重命名或 chmod 任何旧文件；未运行模型、访问远端或启动训练。本地根索引未由本归档任务修改。

这是一份原地留存与内容身份归档，不是另建完整副本或异地备份。三个原目录仍是全量权重、代码、日志、图像与数组的保存位置。

| 保留目录 | 已哈希文件数 | 已哈希字节数 | 归档时状态 |
|---|---:|---:|---|
| `v5_resshift_stage1_20261026` | 655 | 1,122,123,970 | 官方 NPU 推理完成；未训练 |
| `v5_resshift_grad_vq_audit_20261026` | 4,888 | 2,669,495,822 | FP32 梯度检查与 VQ 表示审查完成；保留 FP16 递归阻塞 |
| `v5_resshift_vq_repeat4_20261026` | 1,778 | 1,106,025,165 | 四次 VQ 往返及交付完成 |
| **合计** | **7,321** | **4,897,644,957（4.5613 GiB）** | **原地保留，禁止据旧建议自动开训** |

清单仅排除 `.git` 内部对象/索引和 Finder `.DS_Store`；这些排除项也未删除。源码、已有 `.pyc`、两份官方权重、浮点数组、图片、报告、日志、失败记录及既有 tar 包均纳入内容哈希。官方仓库的 commit、origin 和工作树状态另记于清单；归档时工作树干净。没有符号链接。

另外复制了 **33 份关键证据，1,560,771 字节**，位于 `snapshots/`，每份副本都与原件 SHA256 相同。包括三组最终报告、状态、官方配置、权重身份、数据选择、保护记录和四轮总体统计。完整逐图数组及模型权重继续存放原目录，避免无必要的重复占用。

## 归档时的结果与限制

**Stage 1。** 固定官方 `journal` commit `bb03b7d21614cace01787e097c8a6ab6b945227d`，以官方去模糊配置及两个官方 checkpoint 完成设备适配后的真实 NPU 推理。官方示例是完整 1280×720 图，28 个块、112 次 UNet 调用。固定 V5 Val4 的 zero-shot 平均 PSNR 为 16.9712 dB，输入为 17.7892 dB；四张属于同一类别，不能视为完整 Val1000 结论。`STAGE1_STATUS.json` 记录未开始训练、未使用 Test。

- Deblur 权重：`7b2bb6ddc9c29dfe845028ce32fa146081caf7647f772113eb26502b92dec515`。
- VQ 权重：`e967eaa2c380152811c53e312990d6967a8a957c4f7590f19490ec72110b2889`。

**梯度与 VQ 审查。** FP32 单步训练 Loss backward、完整四步采样、固定真实 ξ 的两阶段物理连接通过；没有 optimizer 更新。FP16 完整物理连接保留 `Conv2DBackpropFilter 507035` 失败，不能写成全部梯度验收通过。严格官方 helper 路径的全部 Val250 Clean 自重建平均 PSNR 为 28.7253 dB；已有弱态50张另行统计，不是新增恢复训练。

**四次 VQ 往返。** 最新 `evidence/STATUS.json` 已更新为 `COMPLETED`，结束时间为北京时间 **21:08:16**，最终报告已经补齐。全部 Val250、50类、1000次往返，四轮 PSNR 均值为 28.7253 / 27.4151 / 26.0999 / 24.9492 dB；250/250 图的 PSNR 每轮严格下降。该实验只有 codec，未调用去噪网络、Simulator、backward 或 optimizer；扩散 NFE 为0。不能把这组数字作为未来真实四步恢复的上限或损失预测。

旧 `outputs/val250_fp32/PROGRESS.json` 仍为 `RUNNING`，但其计数已经是250/250、1000次。该文件没有结束回写；不能据此否定 `RESULT=PASS`、退出码0、最终 `STATUS=COMPLETED` 和交付验收。归档保留该原始不一致，未改写旧记录。

先前梯度报告中“可考虑未来小规模 ResShift 训练”的建议没有执行；用户本次明确选择 DiffUIR-L 并要求停止 ResShift 正式训练，当前执行边界以这项新决定为准。旧建议作为历史记录保留，不删除或改写。

## 验证与后续使用

`ARCHIVE_VERIFICATION.json` 为 PASS：再次完整读取并哈希全部7,321个文件，核对文件大小、mtime、目录清单以及33份证据副本，未发现变化。该验证检查归档内容完整性，没有重新执行 PSNR/SSIM/LPIPS 或模型。

本次新清单不重新读取旧 V5 D0/A/B 大 checkpoint。原实验已有保护记录称107个代码/配置/日志文件未变、32个 checkpoint 的大小/mtime/inode未变；这是原诊断的证据范围，不把它扩写成此次全字节重哈希。DiffUIR 根任务负责远端保护与实时状态核实。

机器清单 SHA256：`e5c9030e5f852fc5df969342958194794202c2915ace801b68f5aec5ca1a142a`。

- [全量机器清单](ARCHIVE_MANIFEST.json)：每文件原路径、字节数、SHA256、mtime及模式；三目录统计和排除项。
- [二次核验](ARCHIVE_VERIFICATION.json)：验收时点、重新哈希数量与错误列表。
- [归档状态](ARCHIVE_STATUS.json)：最新执行决定、保留状态及三项目完成摘要。
- [重验脚本](build_archive.py)：`--verify` 只读核对源资产；创建模式拒绝覆盖已有清单。
- [Stage 1 原始最终报告](</Users/lucas/Desktop/turbulence IR/v5_resshift_stage1_20261026/reports/FINAL_REPORT_ZH.md>)。
- [梯度/VQ 原始最终报告](</Users/lucas/Desktop/turbulence IR/v5_resshift_grad_vq_audit_20261026/reports/FINAL_REPORT_ZH.md>)。
- [四次 VQ 往返原始最终报告](</Users/lucas/Desktop/turbulence IR/v5_resshift_vq_repeat4_20261026/reports/FINAL_REPORT_ZH.md>)。

后续 DiffUIR 对照应引用固定 Val4 选择身份及原始浮点输入；保留 ResShift 的推理精度、输出 clamp、随机种子和评价口径差异。需要改动 ResShift 来做新研究时另开独立分支或目录，不覆盖本次归档的证据。

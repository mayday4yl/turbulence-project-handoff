# A100 湍流项目清理完成

本地交付日期：2026-10-13（北京时间）。操作目录沿用授权时的 `20261012` 名称。

用户明确授权“开始删除”，并限定“只能删除我们湍流项目的东西，不能动其他人的东西”。已按审阅过的 A/B/C/D 四组逐文件白名单执行，服务器完成凭据为 `PASS_AUTHORIZED_TURBULENCE_ONLY_CLEANUP_COMPLETE`；不得重复运行删除脚本。

| 项目 | 实测值 |
|---|---:|
| 湍流目录删除前占用 | 887.2007 GiB |
| 湍流目录删除后占用 | 326.4393 GiB |
| 实际减少 | **560.7614 GiB（602.1130 GB）** |
| 已删除文件 | 18,114 |
| 已移除旧 xi 空目录 | 8,901 |

实测范围仅为 `/home/zhuzhiyu/yanglin/turbulence_ir`。这是目录磁盘块占用变化，不是账号配额剩余值，也不是内存变化。

| 已批准并完成的组 | 文件数 | 文件磁盘块占用 |
|---|---:|---:|
| A：旧 Train2000 在 A100 的 xi 副本 | 17,800 | 332.2444 GiB |
| B：NAFNet 四法早期完整 checkpoint | 86 | 148.1716 GiB |
| C：已结束 preflight 的私有恢复权重 | 7 | 12.3476 GiB |
| D：历史湍流实验中间权重 | 221 | 67.9679 GiB |

文件块占用合计 560.7315 GiB；与目录实际减少量的小差额来自空目录等元数据及新增清理日志。

删除前已对 de36 保留的全部 17,800 个原始 xi 文件读取实际内容，SHA256 均匹配冻结清单，读取 354,630,681,600 字节；de36 本轮无写入。A100 上旧 Train2000 的 RGB、原图、manifest 和 metadata 仍保留；当前 Train8000/Val1000 runtime 未删除。若未来需要旧 raw xi 重放，须从 de36 原档恢复，不能称 A100 仍有完整本地 raw xi 档案。

四法最终 `full_U20000.pt`、共同 U0、官方预训练及关键科学配置/验收凭据共 12 个文件，在删除前后实际整文件 SHA256 全部通过。其余 580,442 个保留文件的路径、设备、inode、大小、mtime、链接数等身份指纹前后完全一致；这项是身份核查，并非对全部保留文件重新做内容哈希。104 项评价、逐样本 parts、指标、图像、G4 四步输出、报告、源码和冻结门禁均保留。

老师的 OWT/AR700/ckvhigh700 归档与 zhouqile 项目不在白名单，未删除、移动或修改。未向任何进程发信号，未重启训练或工作流，未重置任何重试计数。旧 Train500 的另一份约 97.05 GiB xi 缺少独立备份核验，未纳入删除。

原 Phase1 验收时全部千步 checkpoint 存在的记录保持原样；本清理是之后经用户授权的保留策略变更，早期 checkpoint 已裁剪，不能再声称其当前仍全部存在。四法 U20000 与 104/104 正式评价的完成事实不变，不自动重建、重训、恢复旧任务或启动镜像。

证据：`CLEANUP_RECEIPT.remote.json`、`STORAGE_BEFORE.json`、`STORAGE_AFTER_AND_REMOTE_RECEIPT.json`、`BACKUP_FULL_SHA_PROGRESS.jsonl`、`EXECUTION_AUTHORIZATION.json`、`EXECUTION_SOURCE_LOCK.json`、逐文件白名单及删除日志。服务器原始完成记录位于 `/home/zhuzhiyu/yanglin/turbulence_ir/maintenance/storage_cleanup_20261012/COMPLETE.json`。

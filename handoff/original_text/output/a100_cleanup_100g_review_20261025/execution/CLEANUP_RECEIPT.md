# A100 yanglin 湍流项目归档清理记录

流水线及文档完成时间：2026-10-25T21:04:14.986361+08:00（本机时区）。空间扫描结束于2026-10-25T21:03:51.405792+08:00。本次按用户直接授权执行，不启动或恢复实验。

已将 **126,050份文件**归档至 `liteserver-de36:/data2/yanglin/storage_archives/A100_cleanup_20261025_train15000_data`，源端SHA256、de36独立逐文件回读、完整文件集合与fsync、删除前源端复核均通过后，才逐一删除A100清单内副本。

释放原分配空间 **101.7426 GiB（109.25 GB）**。文件内容大小为106.18 GB，两者因文件系统分配块不同而不相等。

| 清理后空间 | 实测值 |
|---|---:|
| yanglin 湍流项目占用 | 340.22 GiB / 365.31 GB |
| zhuzhiyu 共享账号占用 | 1.9332 TB / 1800.44 GiB |
| A100 `/home` 共享文件系统可用 | 3.749 TB |
| 共享文件系统使用率 | 89% |

账号占用是整个zhuzhiyu共享账号，不能归为yanglin个人。du是单次非原子扫描；共享盘在其他任务持续写入时，可用空间变化不必恰好等于本次释放量。系统配额未核实；若沿用用户声明的2TB管理上限，剩余额度估算为66.80 GB，不能当作系统强制配额。

保留文件清理前后元数据指纹一致：**626,299份**；23份重点模型的内容SHA256一致。Train8000、约97GiB原始ξ、最终及最佳模型、全部结果、预测、指标、报告、日志、科学源码/配置/统计/数据身份元数据保留。其他人的项目不在操作范围。

归档的Train15000数据是新增7000个源的original/clean/states；未来续训需要先取回这些文件并核对SHA256。此次不改旧STATE、SOURCE_LOCK、科学配置、保存指针或计数，不自动续训。归档 `files/` 保持原项目相对路径。

直连通道超时记录及不完整目标文件保留在归档transport目录；随后改用四路不重叠的SSH流式传输和无损传输压缩。19:39四路SSH曾同时超时，根因未确定；两端连通恢复后，独立SHA256核对并复用103,534份已完整文件，仅补传22,516份缺失或不完整文件。全部传输结束后，仍对全部126,050份原始文件重新完成独立回读、完整集合和fsync验收，之后才进行源端复核与删除。错误、源码及恢复证据保留在本机recovery_20261025_1942及归档transport对应目录。

证据：SOURCE_MANIFEST.json、ARCHIVE_ACCEPTANCE.json、ARCHIVE_FILES_VERIFIED.jsonl（de36）、PREDELETE_ACCEPTANCE.json、DELETED_FILES.jsonl、CLEANUP_RECEIPT.json、KEPT_BEFORE.json / KEPT_AFTER.json、POST_ACCOUNT_SPACE.jsonl。恢复说明见RESTORE_README.md。

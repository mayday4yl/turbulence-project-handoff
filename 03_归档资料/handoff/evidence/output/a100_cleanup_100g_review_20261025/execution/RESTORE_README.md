# A100 yanglin 湍流项目归档与恢复

本归档由用户于 2026-10-25 明确授权：将已确认的 101.742561 GiB / 126,050 份候选文件移至 de36，逐文件核验后清理 A100 对应副本。

- 归档服务器：liteserver-de36
- 归档根目录：/data2/yanglin/storage_archives/A100_cleanup_20261025_train15000_data
- 原服务器：zhuzhiyu@10.94.2.1（h3c-gpu-server01）
- 原项目根目录：/home/zhuzhiyu/yanglin/turbulence_ir
- `files/` 保持相对于上述原项目根目录的完整路径。
- `SOURCE_MANIFEST.json` 每行记录原路径、大小、分配块、文件身份及 SHA256；`ARCHIVE_FILES_VERIFIED.jsonl` 为 de36 独立读取的校验记录。
- 只有 `ARCHIVE_ACCEPTANCE.json` 与最终 `CLEANUP_RECEIPT.json` 均为 PASS 时，才表示归档及 A100 清理均完成。仅存在本说明或部分文件不代表完成。

归档包括 Train15000 新增 7000 个源的 original/clean/states 共126000份，以及9份临时验收数据和41份旧 Restormer 中间评价权重。

保留 A100 上的整个 Train8000 数据集、约97GiB原始ξ、最终及最佳模型、全部指标/预测/报告/日志/结果包，以及科学源码、配置、统计量和数据身份元数据。此次归档不授权启动或恢复任何实验。

需要恢复 Train15000 时，先另行明确恢复范围，再将所需 `files/<相对路径>` 复制回原项目根目录下同名相对路径。逐文件复核原清单的大小和SHA256；不得覆盖已有且内容不同的文件。原数据中的 UID1036 是 A100 共享账号身份，不是项目个人归属；项目属于 yanglin。恢复后再按对应实验原有门禁检查完整性，不能仅凭旧 DATA_READY 或 STATE 自动续训。

目前选择的中间权重仍可从归档取回；旧报告中的原路径和结果不会为本次搬迁改写。源目录中的历史保存指针、SOURCE_LOCK、科学配置、统计量和实验计数均保留原值。

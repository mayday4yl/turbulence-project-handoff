# UniDB 双路线最终实验资料

两路线已按2026-11-04 00:00（Asia/Shanghai）绝对截止安全结束，完整模型评价、保存预测的独立CPU验收、最终固定图目视、本地完整权重备份及自有进程退出证据现已绑定并通过实际严格validator。这里的“验收”表示实验交付内容已核，不表示已在聊天发送，也不替代全湍流项目交接文档的最终验收。

## 先看结果

- [完整中文实验报告](../report/FINAL_REPORT_ZH.md)：配方、三阶段更新/呈现量、五档四指标、基线、曲线、成本和局限。
- [完整五档质量CSV](../report/FULL_QUALITY_COMPARISON.csv)与[全量/监控验证历史](../report/VALIDATION_HISTORY.csv)。完整验证每G1000张=250来源×4；monitor每G100张=25来源×4，两者分别解释。
- [原路线专用附件](original_route/README_ZH.md)：从U0到U67180的完整train loss、PSNR/SSIM/MAE/LPIPS监控及全量曲线；G1–G4各两个固定来源的Input/Best/Latest/GT图。
- [三阶段动态总图](../figures/CAMPAIGN_DYNAMICS.png)、[实际full-best固定图](../figures/FIXED_G4_BEST.png)、[实际latest固定图](../figures/FIXED_G4_LATEST.png)。原路线最终Best=Latest，两列相同有明确标注。

| 完整EMA候选 | G4 PSNR↑ | SSIM↑ | MAE↓ | LPIPS↓ |
|---|---:|---:|---:|---:|
| 原线U67180，best=latest | 16.211783 | 0.315317 | 0.103124 | 0.411531 |
| 改进线U40848，latest | 13.761130 | 0.106128 | 0.151208 | 0.745103 |
| 改进线U10000，full-best | 14.708872 | 0.126323 | 0.134535 | 0.771947 |

匹配协议的G4 Input PSNR为18.795938；原线较D0 U45411的15.894991提高约0.316792dB，仍低于Input约2.584155dB。改进线在当前原生100步SDE全量评价下退化。以上不是恢复成功或泛化成功；两线起点、objective、LR日程及资源均不同，不能单因果归因于warmup、学习率或loss。仅monitor完成的历史U3000不属于full-best候选。

## 实际训练量与停止边界

原线全程U67180、累计6449280呈现；其中D0 U45411之后三阶段新增21769次/2089824呈现，午夜延长段新增5416次/519936呈现。改进线从官方SR U0起U40848、累计3921408呈现，午夜段新增10848次/1041408呈现。呈现量包含重复采样，不是独立图像数。

最后在途更新均在午夜前开始：原线在截止后1.717842秒完整完成，改进线在截止后0.353840秒完成。8rank全日志摘要的截止后新启动计数均为0，随后仅完成保存与评价。终评分别于01:19:26（原）和00:47:54（改）完成；这些时刻不是训练更新截止。

## 已备份的三个unique完整状态

四个逻辑角色对应三个文件，合计6584061917 bytes（约6.132GiB）。每份包含RAW、Adam、EMA、RNG、cursor和调度/祖先信息。原线8逻辑RNG bank、改进线5物理RNG bank均有独立完整CPU状态验收。

| 实际角色 | 本地完整权重 | checkpoint SHA256 |
|---|---|---|
| original best与latest U67180 | [U00067180.pt](../checkpoint_backup_staging/original/U00067180.pt) | 0381ae3435d2939a2b8b4fbc215c3f78ac6c3775f254bead31dee3e9cdb78e98 |
| improved latest U40848 | [U00040848.pt](../checkpoint_backup_staging/improved/U00040848.pt) | 11a19694f8fc2f4f343e97fab6d3a326759253dbb8eb6b68d2275e144e2cbfc9 |
| improved full-best U10000 | [已验历史备份U00010000.pt](../../unidb_dual_20261103_until_noon/delivery/checkpoints/improved/U00010000.pt) | 728c30638ee41259c6302b53979f23e115e618c08dca7c2156db8f15e1105a75 |

两份新latest以原TRANSFER全文件SHA和当前size/mtime绑定；旧U10000复用原完整字节SHA核验，并新增实际最终角色绑定。未重复下载旧候选或下载全量预测。完整预测留远端，CPU审计覆盖三个unique候选各5000张；本地保留指标/全SHA证据和指定固定小图。

## 验收入口与来源边界

- [实验最终内容adapter](experiment_full_delivery_receipt.json)：四角色身份、完整state、EMA digest→SUMMARY、CPU预测验收、最终图QA、小文件同步及最终退出/备份绑定。
- [实际unique权重备份验收](unique_checkpoint_backup_audit.json)：三文件/四角色的原全SHA、当前stat和最终角色对应。
- [全部自有进程退出adapter](all_owned_process_exit_audit.json)：02:03:51的三root原生终验覆盖468身份；02:10:26再核较早快照中额外66个loader/resource_tracker PID/start/PGID/SID身份。来源联合覆盖534，额外66均已消失，无same identity/zombie/未解释匹配；没有把原生468改称原生534。全部0signals，外部任务保留。指定本地copy runner/rsync PID也已只读确认不存在。
- [实际严格validator输出](EXPERIMENT_FINAL_RECEIPT_VALIDATION.json)及[其输入绑定](FINAL_RECEIPT_REGISTRY_INPUT.json)。状态`PASS_EXPLICIT_NATIVE_FINAL_RECEIPT`，0模型/optimizer/NPU/网络调用；大权重只stat，未再次读取载荷。
- [145项独立最终元数据审阅](../evidence/final_metadata_review/README_ZH.md)、[原生保存预测CPU审计](../evidence/FINAL_OUTPUTS_CPU_AUDIT.json)、[最终小文件同步验收](../evidence/FINAL_READY_SMALL_SYNC_ACCEPTANCE.json)。

CPU finalizer、原附件生成器和TRANSFER原记录中保留当时的pending字段；本次通过新的独立SHA绑定QA/同步/退出/角色验收补齐对应要求，没有手改旧记录。`receipt_history/pre_historical_identity_reconciliation/`是补查66身份前的adapter历史版本，当前入口以本目录三个adapter和最新validator为准。

实验源、旧noon/18/D0/forensic/V5/V6资产均保留。全项目交接、最终Word/PDF/逐页目视和便携小资料包由[项目交接入口](../../project_handoff_20261104/README.md)承接，不能据本README声称那些文档已最终交付。

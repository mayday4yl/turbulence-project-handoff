# G4有限实验最终交付补充（不修改冻结源码）

自动finalizer产生的报告/FINAL_ACCEPTANCE只是工程初验，先保存自动原件。只有双方真实5k各五档5000齐全、按冻结规则决定biased最多追加5k并实际完成后，才作最终交付。不得因5k checkpoint存在提前称完整验证。

最终正文补全：
1. 共同父U157505、双方matched5k以及真实biased额外终点（若有）的每档PSNR/SSIM/LPIPS与逐图源级配对CI。追加10k对Control5k为不匹配预算，主因果比较仍5k。
2. 保留U95544/U109544旧锚点，至少G4三指标及相对新候选恢复差距；不可只对比已退化的共同父并称恢复到旧峰值。全部旧best与CP不覆盖。
3. 全部实际强度呈现、effective60、真实训练更新/游标、原采样位置/τ/seed匹配；按实际分布解释1.875个160K呈现当量，不能说biased均匀完整遍历了所有160K组合。
4. 附两次CPU审查修复：原bitwise恢复FAIL保留，独立重启完整Adam数值Gate PASS；额外冷正式/热Probe逐元素Adam有FAIL，不假称全状态一致。正式u0对真实父全状态精确、前20协议健康PASS，两种证据分开。源码和阈值未改训练。
5. 独立重算完整0/1k/3k/5k逐图及冻结追加条件250源bootstrap；训练seed仅一条，bootstrap不代表训练seed方差。
6. 最终latest全状态真实SHA/Adam276/EMA/RNG四槽/buffers/恒LR/所有worker及CPU退出；32固定来源×五档真实配图和全局曲线目视QA，建议是否结束D0但A/B不自动启动。
7. 补充报告另存或保留自动原件后再更新，重算report_sha256与完整交付清单、本地远端逐项验收。原网络/native残差L1/原S3/LR全部不改，普通质量下降按结果报告。

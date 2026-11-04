# D0按用户指令安全暂停

D0停于U14927，当前更新后完整保存，所有worker及控制器已退出。不是原计划中午截止完成。

本次从U13334新增1593更新、95580次呈现；累计895620次呈现，原Train10000/40K身份不变。

最终完整checkpoint：/data2/yanglin/v6_diffuir_l_20261027/distributed/D0_four_noon_formal01/checkpoints/U014927.pt

SHA：0c7903726f262c56fb40333dcaf70e0e205109c618456bb39e216987bd48d9b6。已核验实际SHA与训练保存CPU读回；独立CPU细查另见CPU_CHECKPOINT_U014927_USER_PAUSE_AUDIT.json（以真实结果为准）。

当前best U14000，固定Val1000 PSNR 21.056258325645853；停止点U14927未完整评估，不冒称best。

自动跟进已暂停，当前策略关闭D0/A/B训练；旧B仍停U9500。保留STOP与全部历史资产，等待用户修改要求，不自行恢复。

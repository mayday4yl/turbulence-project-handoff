# 一次有界 canonical 兼容恢复候选

本次失败不是删除数据导致：原来源、runtime、seed和hash元数据仍在。same seed 的CPU randn在AVX与原ARM/scalar路径上不是逐字节一致；scalar路径对两域及train/val/test的20项raw-xi NPY SHA已精确通过。

随后RGB重放失败，因为x86和原生成主机重新计算固定模拟器buffer时不一致。原主机仅在CPU读取同版本、同SHA模拟器和三份资产并构建固定buffer，设备后端自动加载关闭，无NPU调用。使用原固定buffer后，CPU两域G1–4八项RGB重放全部通过：max_abs约1.97e-6，MSE最高1.85e-15，保留原max_abs<.003、MSE<1e-7，未放宽。

候选dispatch.py只对生成进程设置scalar RNG和使用这份canonical固定buffer；训练及评价进程数学、原科学源码、模型、数据来源、seed、B32/FP32、LR和U40000预算不改。BUILD_IDENTITY明确绑定本兼容REVISION。候选代码、编译与CPU证据已准备，尚未开展新增GPU验收。

原generation重试计数已经3（首次加两次自动重试均结束），STATE为GENERATION_BLOCKED，不能重置计数或直接再次启动。需用户新具体授权以下一次恢复：

1. 启动前重新核验最多两卡、空卡或获准chenchao共享、显存和磁盘门禁；仅一个候选生成worker，保留历史失败和计数3。
2. 新的独立恢复事务只运行一次GPU兼容门禁，最多15分钟。仍按原两域raw-xi精确SHA与八项RGB固定容差验收；失败保留证据并停，不自动再试。
3. 若通过，接续原已授权的28000新轨迹生成和119000runtime数组全量验收。DATA_READY和各法U20000完整迁移/恢复/数值/显存门禁全部通过后，D0/R0优先并行追加20000updates，再R08/L08补位。

这不是第三次小数据科学实验，不延长已完成pilot的TIMEBOX；不会重建已完成下载、旧数据、预检或旧训练，也不恢复NPU任务。

# V5-v1 数据就绪结论

**DATA_READY：数据已完整生成并通过验收，可供训练使用。** 这不代表恢复模型已完成验证或训练。

| 项目 | 结果 |
|---|---|
| clean train / val / test | 2000 / 250 / 250 |
| 每clean独立ξ数 | 4，固定G=4 |
| 新V5 Y4 train / val / test | 8000 / 1000 / 1000 |
| 旧V4观测复用 | **0** |
| 完整核验 | 2500 clean + 10000 Y4，FP32、形状、finite和文件/像素SHA全部通过 |
| 来源隔离 | 2500来源ID、原图SHA和预处理像素SHA全局唯一；旧2450张split保持；新增50张test排除历史扩展9200来源 |
| 原始ξ可重放 | 20000个raw块SHA唯一，保存独立CPU seed、固定RNG版本和形状；真实raw不进入训练输入或监督 |
| 六样本NPU重放 | raw哈希及Y4逐bit一致，最大误差0 |
| train样本CPU32/NPU32 | 最大绝对误差 **1.847743988e-6 ≤ 2e-5**，沿用原Gate阈值 |
| 测试集使用边界 | 只生成、检查完整性及Simulator重放；未运行恢复模型、未计算选模指标 |

原来的2450张clean均重新执行预处理并逐像素核对一致；新增test每类1张。联合pHash/dHash也未发现所选来源中的近重复，但该启发式不能证明不存在全部重裁切或语义相似图片。

三卡生成主循环最长 **1251.04秒（约20分51秒）**，不把三卡累计耗时写成墙钟；完整验收 **69.86秒**。单卡生成峰值PyTorch allocated为466,695,168 bytes；数据目录约 **9.22GiB**。完成后已释放数据生成及验收使用的NPU；后续卡状态由主控实时检查。

DATA_READY SHA256：`14173a9ad216664e884520cf14bec96ea549b7127a2d8027fab2ebea8ca613c7`。

数据根：`/data2/yanglin/v5_concertormer_10k_20261021/data/v5_places_c50_s2000_v250_t250_g4_r4_20261021`。

证据：[正式就绪记录](<../../../NOT_INCLUDED.md#ref-50fadbefb49c6e7216c2>)、[source隔离审查](<../../../NOT_INCLUDED.md#ref-fd9ce8ab891b783e3c77>)、[生成身份](<../../../NOT_INCLUDED.md#ref-2a1788129fc4471cd4c8>)、[完整生成报告](<../../../NOT_INCLUDED.md#ref-0d61b3ea5e4aa8ad1e41>)、[验收耗时及退出码](<../../../NOT_INCLUDED.md#ref-14037ce9fbd7e9b5f6c2>)、[实际Train8000/Val1000 loader检查](<../../../NOT_INCLUDED.md#ref-06cb77c7d462f1bf5028>)。

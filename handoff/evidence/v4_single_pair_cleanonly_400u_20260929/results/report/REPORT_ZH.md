# 单对 clean-only D0：400 updates

固定 Train source 01 / train_s0006_r00 / G4，同一张 C、同一份原始 raw-xi、同一张 Y。原共同初始化，batch1、GPU5一张A100、LC only。400updates=400次固定图像对曝光；没有val曲线，没有泛化结论。
# 单对 clean-only D0：400 updates

固定 Train source 01 / train_s0006_r00 / G4，同一张 C、同一份原始 raw-xi、同一张 Y。原共同初始化，batch1、GPU5一张A100、LC only。400updates=400次固定图像对曝光；没有val曲线，没有泛化结论。

| 数据 | Clean MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Input Y | 0.011374428 | 19.440704 | 0.530480 | 0.533658 |
| D0 clean-only update400 | 0.000339353 | 34.693485 | 0.936284 | 0.060474 |

训练更新耗时 41.35 秒；含评估/保存checkpoint的训练阶段墙钟 50.04 秒，不含预验收/作图/下载。原common_init复用、xi冻结跳过、simulator零调用、400次optimizer状态计数、最终checkpoint恢复输出逐元素一致、旧代码/资产哈希未变，验收PASS。

41个完整浮点评估节点，每10update一次。图为训练这同一对图的拟合程度，PNG仅显示、指标不使用PNG。曲线数据metrics_by_update.csv；完整配置在implementation/protocol/FROZEN.json。保留所有41个FP32输出、C/Y和固定xi备份、5份checkpoint以及逐update日志。
| 数据 | Clean MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Input Y | 0.011374428 | 19.440704 | 0.530480 | 0.533658 |
| D0 clean-only update400 | 0.000339353 | 34.693485 | 0.936284 | 0.060474 |

训练更新耗时 41.35 秒；含评估/保存checkpoint的训练阶段墙钟 50.04 秒，不含预验收/作图/下载。原common_init复用、xi冻结跳过、simulator零调用、400次optimizer状态计数、最终checkpoint恢复输出逐元素一致、旧代码/资产哈希未变，验收PASS。

41个完整浮点评估节点，每10update一次。图为训练这同一对图的拟合程度，PNG仅显示、指标不使用PNG。曲线数据metrics_by_update.csv；完整配置在implementation/protocol/FROZEN.json。保留所有41个FP32输出、C/Y和固定xi备份、5份checkpoint以及逐update日志。

[图像原位置：](<../../../../NOT_INCLUDED.md#ref-7ef0c7daedd3ab226592>)

[图像原位置：](<../../../../NOT_INCLUDED.md#ref-ca07473ad5a389663457>)

[图像原位置：](<../../../../NOT_INCLUDED.md#ref-98b408b489842d987b6c>)

# 1 source × 4 xi × 4 G：clean-only D0

已完成400updates/200epoch/3200exposures。batch8；A100物理GPU5/6。原共同初始化，LC only，raw主结果，末期EMA仅附录。
每epoch16个固定start各出现一次；每个start共出现200次。epoch40=80updates/640exposures，每个start出现40次。
train16与val64 source不相交；每5epoch在同一checkpoint状态评估，所有三指标来自FP32输出。

| 集合 | 节点 | Clean MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| train16 | input epoch0 | 0.009114789 | 20.745385 | 0.598811 | 0.449856 |
| train16 | raw epoch40 | 0.007773978 | 21.384321 | 0.617309 | 0.522765 |
| train16 | raw epoch200 | 0.000879904 | 30.560403 | 0.880661 | 0.163252 |
| train16 | ema epoch200 | 0.018579830 | 17.316280 | 0.472199 | 0.600830 |
| val64 | input epoch0 | 0.013420782 | 19.583533 | 0.482851 | 0.496491 |
| val64 | raw epoch40 | 0.062583643 | 12.808397 | 0.370437 | 0.710327 |
| val64 | raw epoch200 | 0.074237086 | 11.952586 | 0.263562 | 0.730166 |
| val64 | ema epoch200 | 0.048985667 | 13.734805 | 0.386106 | 0.740028 |

训练更新累计75.96秒；含评估/checkpoint总墙钟175.84秒，不含预验收和报告。
前五源图40epoch=400updates/每start40次；本轮40epoch=80updates，本轮200epoch=400updates/每start200次。相同更新次数不代表每start重复次数相同。
报告只解释此固定训练总体的拟合与固定val64表现，不等价于完整验证集或泛化结论。

![训练与验证曲线](train_val_curves.png)

![训练图像竖版](train_source01_G4_epoch40_200_vertical.png)

复现入口：独立implementation下bash scripts/run_all.sh。存在updates日志时拒绝重复训练。冻结协议protocol/FROZEN.json。

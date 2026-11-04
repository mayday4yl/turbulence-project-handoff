# Restormer matched U5000：完整验证主表

旧 Train2000 run，四组均 U5000 / 160000 exposures。完整 Val250 clean × 2ξ × 4G = 2000 starts；online/raw 权重、refinement OFF、CUDA BF16、batch1，指标与 population identity 已逐组核对。Test未使用。

|方法|PSNR ↑|SSIM ↑|LPIPS ↓|Path L1 total ↓|
|---|---:|---:|---:|---:|
|D0|22.7106|0.628662|0.410466|0.049782|
|R0|22.7497|0.628736|0.410140|0.049494|
|R08|22.7013|0.628487|0.409482|0.049942|
|L08|22.7123|0.628818|0.408198|0.049993|

本表是旧 Restormer 的匹配预算验证结果，不与新 NAFNet Train8000 混用。四方法差异较小；该单次训练结果不能单独证明某方法稳定优于其他方法。完整train评价、R0/R08 refinement等重型任务尚在低优先队列，本表不代表全部U5000分析交付完成。

可追溯的checkpoint、aggregate SHA、per-G和评价定义见同目录 OLD_U5000_FULLVAL_SUMMARY.json。

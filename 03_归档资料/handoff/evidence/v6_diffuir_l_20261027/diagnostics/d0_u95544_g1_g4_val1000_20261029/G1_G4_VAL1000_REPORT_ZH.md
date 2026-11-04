# D0 best U95544 EMA：同批Val1000的G1–G4恢复评估

**结论：当前U95544 EMA对G4有效，对G1/G2明显过度处理，跨物理强度泛化不足。** 同一固定Val1000中，G1/G2的PSNR分别下降6.8537/3.9315 dB，SSIM和LPIPS均恶化；G3结果混合（PSNR−0.3903 dB、SSIM+0.04838、LPIPS+0.04345变差）；G4为PSNR+2.3692 dB、SSIM+0.10451、LPIPS−0.07395改善。

PSNR提高的帧数依次为G1 2/1000、G2 41/1000、G3 420/1000、G4 982/1000。固定样本中，G1/G2可见过度增强、亮暗区域失真及纹理伪影；不是输入更清楚就一定恢复得更好。首个固定样本的G4本身也未提高，完整图库保留这些失败样例，没有按结果换图；总体结论以全部1000样本均值为依据。

所有1000条G4物理重放、G4逐图指标及原100张浮点可视化重放的最大误差均为0，支持本次没有混用旧模拟器缓存或改变D0采样方式。这个测试直接显示当前checkpoint的强度适用范围存在问题，但不能单独证明其原因必然是某项Loss、扩散机制失效，或解释全部OTIS/ATD问题。本轮没有修改权重/方法、重选best或启动训练。

本报告正文由原自动结果补充上述科学解释，自动原件保留于automatic_originals。独立从4000行CSV按math.fsum重算均值，与SUMMARY最大误差见INDEPENDENT_REVIEW.json；无新增模型执行。


完整完成4档×1000轨迹=4000次恢复，250个相同Clean来源、每个4个相同物理realization，四档使用匹配的DiffUIR初始Gaussian种子。原三步采样每次真实3NFE，合计12000 NFE，0训练更新。D0/A/B未恢复。当前checkpoint原先按G4 Val1000选出，本轮不重新选模。

本次观察：G1 PSNR变化-6.8537 dB（2/1000帧提高）；G2 PSNR变化-3.9315 dB（41/1000帧提高）；G3 PSNR变化-0.3903 dB（420/1000帧提高）；G4 PSNR变化+2.3692 dB（982/1000帧提高）。完整多指标如下，不把PSNR单项或工程PASS当成跨域科研成功。

## 原始输入基线

|物理强度|样本数|PSNR ↑|SSIM ↑|LPIPS ↓|MAE ↓|
|---|---:|---:|---:|---:|---:|
|G1|1000|22.241768|0.618509|0.106024|0.049478|
|G2|1000|20.306366|0.513581|0.191671|0.063726|
|G3|1000|19.471852|0.475845|0.361375|0.072475|
|G4|1000|18.795938|0.451751|0.571629|0.080178|

## D0 U95544 EMA 三步恢复

|物理强度|样本数|PSNR ↑|SSIM ↑|LPIPS ↓|MAE ↓|
|---|---:|---:|---:|---:|---:|
|G1|1000|15.388083|0.413802|0.391242|0.116472|
|G2|1000|16.374845|0.445874|0.397299|0.099781|
|G3|1000|19.081602|0.524226|0.404826|0.068229|
|G4|1000|21.165142|0.556261|0.497683|0.055117|

输入与输出严格同图配对，汇总为1000个逐图指标的均值。250个Clean×4realization不是1000个独立Clean；不能把所有帧作为独立样本作总体显著性结论。完整逐图、逐档差值及提高帧数见PER_IMAGE_ALL_G.csv和SUMMARY_BY_G.csv。按强度曲线见METRICS_BY_G.png。

## 同批、同ξ与当前数据版本

不使用旧canonical G1–G4缓存来替换当前V6 G4。当前固定Val1000的clean/sample/source/类别/物理seed均来自原冻结验证清单；raw epsilon_zernike/epsilon_tilt每条按原CPU B1 RNG复现并逐一匹配原raw SHA。原V5-v1 Simulator all-enabled配置和canonical buffers保持；G1–G3新生成于本诊断目录。全部1000条G4物理重放最大误差0，其中1000条逐像素精确相同；门槛沿原物理重放验收≤2e-5。四卡首样本另检查g0 identity。G4推理实际读取原验证文件。

每档都重新执行1000张，未用历史G4数值冒充新推理。新G4逐图指标相对原U95544最终Val1000最大差0；原100张已保存浮点可视化重放最大像素差0。这些一致性检查支持本次强度对比沿用了当前D0的数据及采样协议，而不是混入旧版本缓存。

D0只读取Yg，不额外输入g、物理时间、GT或ξ；GT和ξ仅用于合法生成评估观测及计分。本次每档从自己的原生初始分布完整三步恢复，不执行G4→生成G3→生成G2→生成G1的外层递归。不同g共享原G4验证seed，因此DiffUIR初始随机噪声匹配。

## 指标与图片

保持原256×256、原RGB浮点、range1，无resize/crop/Mask/颜色拟合/指标前额外Clamp。PSNR逐图RGB FP64 MSE后取均值，SSIM为RGB Gaussian11、sigma1.5、population covariance，LPIPS为原官方Alex v0.1 pretrained/normalize=True/FP32。此处LPIPS是标准整图RGB，与之前OTIS的luminance valid64指标不同。仅PNG/JPEG显示clip01和量化。

图库G1_G4_GALLERY.html含预固定100个sample。列为Clean / G1输入 / G1恢复 / G2输入 / G2恢复 / G3输入 / G3恢复 / G4输入 / G4恢复。聊天两页取sample_id顺序前8个独立Clean，未按结果选图。所有4000张原浮点输出与PNG在`/data2/yanglin/v6_diffuir_l_20261027/distributed/D0_U95544_Val1000_G1234_S3_20261029_01/outputs`，G1–G3输入原数组在本目录inputs，原件不改。

## 执行与复现

checkpoint `/data2/yanglin/v6_diffuir_l_20261027/distributed/V6_D0_official_residual_until18_20261029_01/checkpoints/U095544.pt`；SHA256 `1fd16df15c64cc496bda89ccb32fd6a9413c9de8e59942fc13cbb6592e757cc3`，严格完整加载ema.model，native/sampling buffer逐项一致；不创建optimizer、不做backward。物理生成与推理分开进程，避免生成端NPU配置改变原D0推理设置；只用NPU4–7，FP32/HF32off、原资源余量守护。

物理生成进程墙钟153.12秒，4卡评估进程墙钟369.84秒（含加载、原图基线/输出计分与保存），纯三步采样秒数合计607.24，各rank真实峰值见SUMMARY.json。不得将纯采样时间混作完整端到端吞吐。

输入身份PAIR_PLAN.jsonl、生成状态GENERATED_INPUTS.jsonl、DATA_GATE.json、源码/配置/协议SHA、原始命令GENERATE_COMMAND/EVALUATE_COMMAND/FINALIZE_COMMAND.json和原日志均保留。Test未做图像评估，原训练STOP保留；当前测试不恢复训练或修改best。尚不能由这一组物理强度结果确定OTIS/ATD失败的原因，或证明某项Loss变更的因果作用。

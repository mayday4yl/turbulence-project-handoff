# DiffUIR原五任务完整13子集：官方EMA与固定D0 best U109544

已完成13/13子集，共43321配对输入、86642恢复、259926次全量U-Net调用。各图每模型原native3NFE，0训练/微调；额外技术Smoke单列不并入这些预算。全部原尺寸，未缩256或切块；A/B保持暂停。

原PythonY PSNR方面D0在13/13子集均值低于官方，RGB LPIPS在13/13子集变差。退化不局限低光，去雨/去雪/去雾尤其明显。结果支持这些已测原任务能力下降，符合跨任务遗忘表现；这不是按训练阶段逐一对照，尚不能定位哪段续训或哪项Loss造成，也不证明扩散机制失效。固定旧D0109544，非新G4-biased。

![全套原任务指标](NATIVE_METRICS.png)

|测试集|图数|官方Y-PSNR|D0 Y-PSNR|差值dB|官方Y-SSIM|D0 Y-SSIM|官方LPIPS↓|D0 LPIPS↓|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LOL15|15|25.27394|16.62088|-8.65305|0.90433|0.55129|0.16119|0.41594|
|GoPro1111|1111|27.81591|25.93251|-1.88340|0.85014|0.79139|0.22104|0.28143|
|HIDE|2025|27.58086|24.77022|-2.81065|0.84524|0.76412|0.25492|0.34309|
|RealBlur-J|980|28.42781|27.40794|-1.01987|0.85829|0.83372|0.18897|0.22232|
|RealBlur-R|980|35.91006|33.93506|-1.97500|0.96693|0.95214|0.10521|0.11935|
|Rain100H|100|27.92963|13.63631|-14.29333|0.83343|0.36586|0.22283|0.64538|
|Rain100L|100|32.16672|25.56149|-6.60523|0.93304|0.80518|0.12030|0.32863|
|Test100实际98|98|28.01989|21.99364|-6.02624|0.88781|0.66298|0.11206|0.37825|
|Test1200|1200|33.49808|22.87079|-10.62728|0.92675|0.71388|0.06994|0.34234|
|Test2800|2800|33.04478|23.54586|-9.49892|0.93239|0.75659|0.05964|0.28710|
|Snow100K-S|16611|35.30348|25.29699|-10.00649|0.95469|0.81265|0.06885|0.28705|
|Snow100K-L|16801|29.65704|19.77302|-9.88402|0.90277|0.67616|0.13756|0.38961|
|SOTS-outdoor|500|35.28574|17.17378|-18.11197|0.99362|0.81615|0.00829|0.19638|

上表Y列沿用原Python量化RGB→浮点Y函数（历史字段psnr_y_uint8/ssim_y_uint8），RGB LPIPS补充；不是湍流Val的RGB口径。按数据集分别比较，不混成总PSNR。RainH/L同一批Clean不同退化，不把跨集样本当独立训练重复。

## RealBlur作者ECC注册RGB：独立口径

|子集|共同成功对数|官方RGB PSNR / SSIM|D0 RGB PSNR / SSIM|各模型失败数|
|---|---:|---:|---:|---|
|J|980/980|28.176195145993976 / 0.8476194108429014|27.643255851235573 / 0.8299185236911315|{'official': 0, 'D0_best': 0}|
|R|980/980|35.51480988105532 / 0.9450159590535555|34.9934390402447 / 0.9394578074398997|{'official': 0, 'D0_best': 0}|

作者ECC使用half-up uint8→float32、强度匹配、homography100/eps0、cubic inverse/reflect、nearest有效mask、RGB mask归一PSNR和Gaussian11/crop5 SSIM。仅共同成功交集配对，失败完整保留、不回退/静默删样本；未注册表覆盖完整各980。算法CPU验收通过，不宣称旧OpenCV/SciPy二进制逐位。

## SOTS来源公式uint8Y：独立补充口径

|模型|来源公式uint8Y PSNR|来源公式uint8Y SSIM|
|---|---:|---:|
|official|35.23592461|0.99244008|
|D0_best|17.17318278|0.81433948|

SOTS500输入/492前缀PNG GT，8次GT重用，均原尺寸一致，GT resize为同尺寸映射。来源公式用精确有理BT601+half-up uint8Y、255/RMSE和Gaussian11 sigma1.5 valid SSIM；全500两模型PNG独立CPU重算通过。原Python浮点Y与二次Y量化不混在一表，未执行旧MATLAB二进制，不宣称半整数浮点边界逐位相同。



配对数不等于独立Clean来源数：按GT文件SHA，Test2800为2800退化输入/200个不同GT，SOTS为500/492；Rain100H、Rain100L各100/98，作者Test100为98/95；Snow-S为16611/16604、Snow-L为16801/16789。完整13集的来源数量、尺寸直方图、原manifest与Gate SHA见EVIDENCE/SOURCE_DATA_INDEX.json，原有重复按作者协议全部保留，不重新加权或去重。

## 同一批图与全部逐图结果

- [LOL15完整报告](../LOL15_REPORT_ZH.md) · [恢复对比图库](../LOL15_GALLERY.html)
- [GoPro1111完整报告](../gopro1111_delivery_20261030_2015/GOPRO1111_REPORT_ZH.md) · [恢复对比图库](../gopro1111_delivery_20261030_2015/GOPRO1111_GALLERY.html)
- [HIDE完整报告](../hide_rain3_delivery_20261031_0530/HIDE/FINAL_REPORT_ZH.md) · [恢复对比图库](../hide_rain3_delivery_20261031_0530/HIDE/HIDE2025_GALLERY.html)
- [RealBlur-J完整报告](../realblur_jr1960_delivery_20261031_0645/RealBlur-J/FINAL_REPORT_ZH.md) · [恢复对比图库](../realblur_jr1960_delivery_20261031_0645/RealBlur-J/REALBLUR_J_GALLERY.html)
- [RealBlur-R完整报告](../realblur_jr1960_delivery_20261031_0645/RealBlur-R/FINAL_REPORT_ZH.md) · [恢复对比图库](../realblur_jr1960_delivery_20261031_0645/RealBlur-R/REALBLUR_R_GALLERY.html)
- [Rain100H完整报告](../hide_rain3_delivery_20261031_0530/Rain100H/FINAL_REPORT_ZH.md) · [恢复对比图库](../hide_rain3_delivery_20261031_0530/Rain100H/RAIN_Rain100H_GALLERY.html)
- [Rain100L完整报告](../hide_rain3_delivery_20261031_0530/Rain100L/FINAL_REPORT_ZH.md) · [恢复对比图库](../hide_rain3_delivery_20261031_0530/Rain100L/RAIN_Rain100L_GALLERY.html)
- [Test100实际98完整报告](../hide_rain3_delivery_20261031_0530/Test100_actual98/FINAL_REPORT_ZH.md) · [恢复对比图库](../hide_rain3_delivery_20261031_0530/Test100_actual98/RAIN_Test100_GALLERY.html)
- [Test1200完整报告](../rain_test1200_delivery_20261031_0540/Test1200/FINAL_REPORT_ZH.md) · [恢复对比图库](../rain_test1200_delivery_20261031_0540/Test1200/RAIN_Test1200_GALLERY.html)
- [Test2800完整报告](../rain_test2800_delivery_20261031_0620/Test2800/FINAL_REPORT_ZH.md) · [恢复对比图库](../rain_test2800_delivery_20261031_0620/Test2800/RAIN_Test2800_GALLERY.html)
- [Snow100K-S完整报告](../snow_s16611_delivery_20261030_2336/SNOW_S_FINAL_REPORT_ZH.md) · [恢复对比图库](../snow_s16611_delivery_20261030_2336/SNOW_S_GALLERY.html)
- [Snow100K-L完整报告](../snow_l16801_delivery_20261031_0243/SNOW_L_FINAL_REPORT_ZH.md) · [恢复对比图库](../snow_l16801_delivery_20261031_0243/SNOW_L_GALLERY.html)
- [SOTS-outdoor完整报告](../sots_out500_delivery_20261031_0620/REPORT_ZH.md) · [恢复对比图库](../sots_out500_delivery_20261031_0620/SOTS_OUT500_GALLERY.html)

每子集固定图库在推理前冻结（LOL为完整15图，其余固定32）。全量CSV/JSON、strict完整EMA/buffer/3NFE、数据与来源SHA、实际命令及退出证据均在链接的不可覆盖交付包；大尺寸float/PNG留服务器。不是以展示图代替全量。

## 时间与适用边界

|测试集|完整run墙钟秒|配对输入/秒|官方采样均秒/图|D0采样均秒/图|
|---|---:|---:|---:|---:|
|LOL15|51.681|0.29024|0.31845|0.19590|
|GoPro1111|3079.606|0.36076|0.83341|0.82703|
|HIDE|5372.775|0.37690|0.83031|0.82724|
|RealBlur-J|1512.160|0.64808|0.44302|0.43658|
|RealBlur-R|1270.229|0.77151|0.44127|0.43562|
|Rain100H|77.094|1.29712|0.18514|0.14780|
|Rain100L|80.654|1.23986|0.18058|0.14540|
|Test100实际98|80.246|1.22124|0.19484|0.15415|
|Test1200|876.010|1.36985|0.22035|0.21494|
|Test2800|1575.122|1.77764|0.19043|0.18666|
|Snow100K-S|11509.206|1.44328|0.23766|0.23583|
|Snow100K-L|11841.185|1.41886|0.23673|0.23532|
|SOTS-outdoor|394.055|1.26886|0.24042|0.22822|

完整NN run墙钟之和37720.022秒（10.478小时），含模型加载/CPU原生指标/图像读写/调度；不等同端到端下载加独立ECC CPU总时长，也不把采样均时当整体吞吐。

所有权重固定且未用本测试选模。作者canonical公开测试包未与不可下载的DiffUIR重打包逐像素核验，逐图uint32配对seed与原全局seed10流不同；本项目结果可追溯，但不冒称论文表格精确复现。Snow只S/L，Train图0；Test100作者包实际98，不虚报100，不拿Rain1400替Test2800。原失败/工程修复/自动pending原件保留。

全部相关原控制器和NN worker退出验收已通过；最终本地同步及套件图像QA以FINAL_ACCEPTANCE.json和DELIVERY_FULL_SHA256.json为准。不恢复任何D0/A/B。

本套件26行分模型指标与13项完整NN审计逐值交叉核对通过；13份图库共399幅文件完整性与26个本地报告/图库链接通过，新增RealBlur J/R各第1/17/32张和总图已目视核对。其余各阶段目视证据保留在各最终包，不把文件完整性验收说成逐幅人工审美评价。新总图只调整图例位置以免遮挡SOTS数据，自动初稿与冻结源保留。

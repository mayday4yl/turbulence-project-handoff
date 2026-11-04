> 修订4：R0/R08 refinement ON完整Val8000均已独立验收。保留已补齐的L08 G4四步结果及全部OFF/EMA、曲线和视觉。原REV2/REV3保留。D0/L08 ON不适用。

# 原Train8000四法 U20000 结果包

四法均已完成20,000 updates、640,000 exposures（5 epochs），原104项正式评价已完成。修订4保留全部已验收OFF和R0 ON结果，并补入R08 refinement ON完整Val8000结果。D0/L08的ON按原方法接口不适用。EMA是次要权重评价，与refinement开关分开。

## 先看哪些文件

- `U20000_RESULTS_OVERVIEW.pdf`：19页总览，含主表、EMA次表、不同输入G、固定G4逐递归步、四法loss及PSNR/SSIM/LPIPS曲线和Train/Val视觉效果。
- `U20000_RESULTS_TABLES.xlsx`：5个可编辑/筛选的数据页。主表已填R0/R08 ON，D0/L08 ON保留不适用。
- `figures/`：12张独立曲线图，均有PNG和矢量PDF。PSNR、SSIM、LPIPS各自成图；另有每个方法的四指标总览。
- `data/`：18份UTF-8 CSV及完整数值JSON。保留原始训练记录和绘图使用的最后记录两个版本。
- `visuals/`：Train/Val各两张固定G4样例的四法同图对比，含完整图和相同中心裁剪。`individual_png/`是24张独立显示图；`raw_arrays/`是对应未截断FP32数组。
- `source_artifacts/`：原最终评价的COMPLETE/IDENTITY/aggregate、训练日志、已完成G4逐步评价来源及SHA索引。包内不含大checkpoint。

## U20000主表

online/raw权重、refinement OFF、FP32、batch1、TF32 OFF；完整Val8000为1,000 clean sources × 2 trajectories × G1–4，Places365/LSDIR各500sources。Test未使用。

| 方法 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| D0 | 23.090775 | 0.646968 | 0.378897 |
| R0 | 22.919512 | 0.638654 | 0.401781 |
| R08 | 22.736275 | 0.629232 | 0.416175 |
| L08 | 22.811353 | 0.633398 | 0.410421 |

在这个固定验证人口上，D0的三项平均图像指标最好。这是描述性比较，没有在此包新增显著性或泛化结论。Path loss依照各方法原路径聚合，不能单凭loss大小判断最终图像质量。

## 曲线和G表怎么读

训练日志每20updates记录一次，图01展示训练batch损失及10条记录的后向中位数。四法各有1,000个唯一记录节点。L08历史恢复留下100条重复update记录：原始4100行完整保留，绘图每个update取最后一条，没有删除历史或重置计数。

固定小monitor的Train/Val各200inputs、50sources，每1,000updates评价一次。三项指标各自曲线展示这一固定人口，不能当成full Val8000曲线。图06另给出每4,000updates的full Val8000三指标曲线。

`val_u20000_by_G.csv`给出输入G1–4各自最终恢复指标，每个G为2,000inputs/1,000sources；online和EMA分行。`G4_recursive_prediction_steps.csv`是在同一2,000个G4输入上，R0/R08/L08预测C1–C4的结果。这两种G/步骤口径不能互换。L08四步结果已于10月13日完成并独立CPU验收；初版结果包漏收，修订2已补齐，未重新训练或推理。

## 视觉样例

样例沿用冻结Train/Val monitor中每域按start_id排序的首个G4输入，在比较输出质量之前已固定：Train为train_s0113_r00（Places365）和train_s4752_r00（LSDIR）；Val为val_s0092_r00和val_s8723_r00。

四法均为自身U20000 online、refinement OFF。D0/R0/R08复用此前接受的CPU导出；本次仅用CPU补出缺失的L08四张图。CPU图像指标与对应正式GPU记录的差值通过已有数值标准。视觉图片是补充展示，主表仍使用正式已提交GPU评价，未用显示图重算主指标。

显示PNG只统一clip到[0,1]并四舍五入到8-bit，不逐图拉伸、不锐化。中心裁剪统一为原256×256图像的[64:192,64:192]。原始数组和来源SHA均保留。

## ON补评状态

R0、R08各补同一Val8000、online权重、refinement ON，合计16,000正式输入。保持原native-xi算法：每个合法非终止阶段40次Adam更新、41个候选、学习率0.01和lambda_ref=0.01；不更新网络权重、不改原四法训练或OFF评价。

R0/R08 ON各完成250parts/8000inputs/1000sources，每源8、每G2000、每域4000。两法分别通过独立CPU原序FP64、人口/身份/指标/种子绑定及本机逐文件SHA/8000行复算。补评worker及队列已自然退出，原ATTEMPT、分段历史和时钟保留。

当前Train15000各自U20000→U40000续训独立进行，不混入本包。原四法U20000 checkpoint和其他项目资产未修改。

## 口径与来源

PSNR为逐输入RGB/H/W MSE换算后平均；SSIM采用原skimage定义；LPIPS为原AlexNet v0.1。指标用未clamp的原始FP32恢复RGB，不做crop/resize/quantization。精确指标契约见`data/metric_definitions.json`。

本次将104项原评价的COMPLETE/IDENTITY/aggregate与服务器当前SHA逐文件匹配，复用已有独立CPU正式验收，没有重跑这些已完成推理或大checkpoint审查。结果打包采用静态科研文件交付；CSV、Excel、PNG/PDF和FP32数组共享相同已核对数据。


## 修订4：R0/R08 refinement ON

同一原U20000 online与fullVal8000，D0/L08 ON不适用。

| 方法 | 开关 | PSNR ↑（dB） | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| D0 | OFF | 23.090775 | 0.646968 | 0.378897 |
| R0 | OFF | 22.919512 | 0.638654 | 0.401781 |
| R08 | OFF | 22.736275 | 0.629232 | 0.416175 |
| L08 | OFF | 22.811353 | 0.633398 | 0.410421 |
| D0 | ON | 不适用 | 不适用 | 不适用 |
| R0 | ON | 22.911773 | 0.637986 | 0.400823 |
| R08 | ON | 22.733954 | 0.629263 | 0.414917 |
| L08 | ON | 不适用 | 不适用 | 不适用 |

R0 ON相对自身OFF：PSNR -0.007739 dB、SSIM -0.000668、LPIPS -0.000959，PSNR/SSIM略降，LPIPS略好。

R08 ON相对自身OFF：PSNR -0.002321 dB、SSIM +0.000031、LPIPS -0.001258，PSNR略降，SSIM/LPIPS略好。两法均不据此声称总体优势。原生refinement固定40updates/41candidates、lr0.01/lambda0.01。

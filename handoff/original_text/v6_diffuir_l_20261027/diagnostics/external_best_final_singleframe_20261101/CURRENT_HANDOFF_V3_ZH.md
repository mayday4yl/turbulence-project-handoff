# 当前交付：单帧论文协议核实修订（2026-11-01）

最新6页PPT为output/singleframe_protocol_v3_delivery/V6_External_Generalization_Evaluation.pptx。SHA2c8e595c4fcb654300820a5bdd22cde0d7db9e1abdb700f6bb0f2dbbc8a16c21，7938300字节；最终全部6页视觉审查、56嵌图像素对源、12共同ROI、2原生表数字逐CSV/所有讲稿来源与0布局警告核验完成。原V1/V2和未交付V3草案保留，不覆盖。

严格主列Input、D0best109544 EMA、D0final157505 EMA、对应单帧论文方法、GT有才放；TurbWNet/两5k分支从外部PPT移除。PiRN Fig3/6与TurbNetFig9在同一页同一行，但论文精确输入帧未能证明，明确标论文展示参考并附原文Input缩略图，不称同帧/同ROI。OTIS候选Pattern1_027/Pattern9_027来自输入场景匹配，不按D0质量选帧；官方Pattern9是中湍流，helper名字strong不作为强度。

已实际读PiRN arXiv v1正文实验PDF5–8页、TurbNet ECVA PDF9–14页及固定官方测试源。PiRN公开inference_images.py67–98邻帧平均+512缩放，与本次单帧不符；TurbNet EvaluateData未定义变量、evaluate63返回-1、restore84–88保存输入tiles非J。utils/misc.py实际存在且已读取，PSNR RGB/[0,1]、SSIM Gaussian11/sigma1.5/零填充。论文定位单帧与发布demo不一致不能推出论文实验必然多帧；代码静态问题也不等于科学结论。官方权重未取得/核验时只用论文图，不随机实例化。独立supplement已查CVF/ECVA/Springer及arXiv源包但未取得，不声称已阅读。原页码、图表、代码行、自定规则及未知项在results/SINGLE_FRAME_PROTOCOL_AUDIT.csv及PROTOCOL_AUDIT_AND_SPEAKER_NOTES_ZH.md，每页讲稿也含依据。

六组5225 final NN与CPU汇总早已真实完成且退出，保留4534+691；旧best复用。历史OOM/显存缓存等价修复和有效输出/失败Smoke成本区分保留，不重新执行。Text final400恢复完成但final OCR未算，不用其他权重OCR顶替。OTIS序列macro灰度Mask与ATSynRGB指标不同不混比；论文AWDR/AD-LCS和本方AppleVision WordAcc/CER不同不排名。自然场景NIQE共同有限支持，不报伪PSNR。

ATSyn所有工作现已完成：官方1000test场景固定turb/24.png+gt.jpg共2000原ZIP成员CRC/长度/SHA，服务器既有proxy获取，不本地relay，不下载126GB完整包/Train。1个joint pHash2/dHash0近重复候选（Places365_test_00009761与V6Train源Places365_val_00009761）先排除，999正式。源ID/字节/预处理像素/联合近重复检查用于included样本，不承诺未知预训练全集无暴露。999两EMA各3NFE/native512/no extra noise，真实PROCESS_COMPLETE exit0/7worker和pipelineCPU退出，独立108源+3控制源/两权重SHA/所有数据和预测SHA/seed/finite/逐图PSNR及MAE重算差0/SSIMLPIPS全量finite均值重算PASS。固定首中末三组四列完整目视，图像场景与GT匹配但仍强模糊，SUMMARY/PER_IMAGE/12 full及12ROI小图本地同步完成。NN和CPU均不得重启。

Input/best/final ATSyn PSNR21.38151965/21.47726880/21.58389958，SSIM.58283467/.59024354/.59554888，LPIPS.55090750/.52245069/.49099741，MAE.05836579/.05688721/.05629226。只是小幅改善，非整体泛化成功。DATUM默认20帧/有效4帧/noise.0001，当前是自定单帧固定24，不当论文精确复现；CW-SSIM未测。

本轮无任何训练、A/B/RL，既有checkpoint/冻结实验不变。Heat404、独立supplement未取到、单帧权重不可验证、final OCR缺失已记录跳过，不编假表。NPU1用户授权但外部PID2938958未干预，实际7卡工作已退出。只同步小型结果、图和PPT至本地。全部本轮可行修订交付后删除v6-atsyn自动跟进。

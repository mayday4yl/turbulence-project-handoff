# 旧D0 U14927：固定Train–Val泛化分析

## 结论

三种候选中，当前证据**更符合Train/Val同时恢复性能不足**，没有检出明显的Train优于Val泛化差距。Train PSNR比Val低0.140351dB，而非明显高于Val；PSNR、SSIM和LPIPS差距的独立Clean层级bootstrap区间均跨0。两边恢复指标都处于约21dB PSNR、0.54–0.55 SSIM、0.53 LPIPS，结合此前恢复图细节不足，瓶颈也存在于训练集。

这是固定子集和单checkpoint的实测判断，**不是证明完全没有过拟合，也不是证明原因一定是优化欠拟合**。无法据此区分模型/采样过程/优化限制、退化歧义与信息损失；未做新增训练对照，也不把该判断外推到正在进行的新单残差续训阶段。

## 对象、固定索引与运行

只评估旧联合Loss阶段的U14927 EMA，checkpoint SHA `0c7903726f262c56fb40333dcaf70e0e205109c618456bb39e216987bd48d9b6`。训练阶段随机时间残差Loss没有在此作为评估指标。本次0 optimizer更新，没有重新初始化替代权重。

Train固定抽取300张独立Clean，50类每类6张，保留20%原数据/80%扩容数据：60原Clean+240新增官方训练Clean。每个Clean的4个G4 realizations全部评估，共1200个图像对。不是1200张独立Clean，不是Train10000全量结果。选择只依据固定seed和类/来源内SHA256排序，在读取任何恢复指标之前保存索引：FIXED_TRAIN300_G4_1200.jsonl，SHA `9c2ee4a196d92e9067fbe2a5caf2642b82aa621e8eff105d8f21058269baa818`。选择seed20261028，独立Clean内容SHA无重复；与Val的source ID及Clean SHA均不重叠。

Val为锁定的原Val1000，即250张独立Clean×4 realizations。复用此前对同一U14927 EMA新执行的完整Val1000（distributed/V6_D0_U14927_baseline01/validation/U014927_1000），明确本次没有重新运行全部1000图。另实际重放4个不同Val Clean的样本，四项指标与已保存值差异全部为0；原摘要/逐图SHA和重放证据见VAL1000_REUSE_ACCEPTANCE.json。

Train直接调用未修改的method_impl.evaluation.evaluate_rank，只在独立诊断进程将其数据列表适配为固定Train索引；采样、指标代码保持原函数。FP32/HF32off、256RGB、EMA、真实三步、初始噪声/时间网格/Reverse/Clamp均不变。种子规则为原BASE_SEED=2026102707及derive_seed(base, source_id, sample_id, 'outer', 0)，不因Train/Val切换而另设规则。PSNR为逐图RGB PSNR均值；SSIM为原Gaussian RGB定义；LPIPS为Alex v0.1 normalize=True。指标前不额外Clamp、裁剪、缩放或选取ROI，PNG显示按原代码处理。

## 实测结果

| 指标 | Train300（1200样本） | Val250（1000样本） | Train−Val | 差距95% CI |
|---|---:|---:|---:|---|
| psnr | 20.919850 | 21.060202 | -0.140351 | [-0.517082, +0.247022] |
| ssim | 0.543754 | 0.548787 | -0.005033 | [-0.024380, +0.014028] |
| lpips | 0.531770 | 0.529070 | +0.002699 | [-0.013070, +0.017674] |
| mae | 0.057347 | 0.056156 | +0.001191 | [-0.001583, +0.004005] |
| input_psnr | 18.542729 | 18.795938 | -0.253209 | [-0.606052, +0.104494] |


差距统一定义为Train−Val：PSNR/SSIM正值代表Train较好，LPIPS/MAE负值代表Train较好。置信区间使用5000次bootstrap、seed20261028：先平均同一Clean的4个realizations，再在50个固定类别内分别重采样Train独立Clean及Val独立Clean；不能将四次realizations当独立样本扩大显著性。

输入Train PSNR比Val低-0.253209dB，说明这个Train子集本身略难。Train相对输入平均改善2.377121dB，Val改善2.264264dB；改善量Train−Val为+0.112857dB，95%CI[-0.011579, +0.238651]也跨0。这进一步不支持“训练集恢复明显更好、验证集掉得多”的典型解释；这只是难度诊断，不是因果调整后的泛化证明。

按来源诊断：原60个Train Clean PSNR20.887239、SSIM0.538336、LPIPS0.527279；新增240个Train Clean PSNR20.928003、SSIM0.545109、LPIPS0.532892。两类都没有显著高于整体验证集的现象；来源内样本较少，不把这些均值当来源因果结论。

## Loss与恢复指标必须分开

训练时随机τ残差Loss衡量原生Forward噪声状态下的残差预测，旧阶段还联合了三步Clean训练Loss；本评估只测冻结EMA完整三步恢复后的PSNR/SSIM/LPIPS/MAE。两者数值不能直接比较，也不能用训练Loss下降宣布过拟合。当前数据没有显示明显Train–Val恢复差距。关于训练集本身恢复是否充分，应看这里的Train三步指标和固定可视化，而不是随机时间残差Loss。

## 资源与保护

仅使用事先确认空闲并取得独占项目锁的物理NPU3，不使用训练的NPU4–7；低优先级、CPU128–131、Torch线程2。主训练未停止、未发信号、未改配置/数据/Loss/LR/源码/checkpoint。Test图像读取/评估数0，未做Test选模。实际Train采样共3600次去噪器调用，等于1200×3；实测评估耗时311.47秒。无梯度、无optimizer、无参数更新，独立评估进程已退出。原checkpoint、采样/指标源码哈希验证，见COMPLETE.json及PROVENANCE_AND_EXIT.json。

逐图原始结果见train_eval/rank0.jsonl，逐独立Clean汇总见PER_SOURCE.csv，多指标表/CI见TRAIN_VS_VAL_METRICS.csv及METRICS_AND_GAPS.json，固定图像输出见train_eval/fixed_visual100。所有新写入仅本诊断目录；原锁文件仅作进程生命周期互斥，不改变训练资产。

## 复现

保留同一固定索引，不重新抽样。确认物理NPU3空闲后，执行LAUNCH.json记录的V6_DEVICES=3、原scripts/run_env.sh和evaluate.py命令；数据加载仅改变评估列表，不改原采样或指标实现。完成后同环境运行analyze.py和report.py。已完成输出不应覆盖；复现请复制诊断脚本与固定索引至新的独立输出目录，并保持权重/数据/SHA身份。

# 工件与复核入口

这是一套完成的只读审计。原训练永远停在U45411，本文件不授权重跑恢复模型或训练。不要将vendor中定义过optimizer的源码当作本次新建了optimizer。实际执行身份、协议、源SHA和输出记录分别绑定在各子目录；所有未来短训练仅见EXPERIMENT_PLAN设计。

|材料|生成源/复核方式|数据与边界|
|---|---|---|
|官方差异|official/upstream/GIT_TREE.json、FETCH_MANIFEST.json、SOURCE_DIFF.json|独立原commit blob、正式论文/官方HR-LR原图与当前执行源；不是官方完整SR复现|
|数学/时间|math/coefficient_audit.py、finite_gamma_counterexample.py、replay_cpu_tau.py|纯标量独立推导、当前CPU RNG重建、101时刻CSV；不加载网络|
|数据身份/图像统计|data_optimizer/audit_source_pixels.py、audit_data_images.py|10250原JPEG/完整Val输入＋固定train、原保存图；2D Sobel修复旧版归档|
|真flow/DC oracle|data_optimizer/reconstruct_warp_cpu.py、warp_decomposition.py、oracle_global_shift.py|原xi/seed/basis及冻结warp公式；oracle使用mean(full true flow)非部署，固定crop|
|Adam/参数/梯度|data_optimizer/audit_checkpoints.py、gradient_probe.py、gradient_postprocess.py、finish_gradient_analysis.py、cpu_backward_parity.py、audit_adam_dispatch.py|7完整checkpoint；96完整梯度；新backward零step；实际分派helper不构造Adam|
|新恢复轨迹/seed/方差/FD/SR|runtime/prepare_runtime.py、forensic_worker.py、controller.py；PROTOCOL.json|明确checkpoint/输入/种子/共享实际noise；全部完成且退出，不再次启动controller|
|教师噪声/条件依赖|teacher_noise_probe/worker.py、audit_analyze.py；condition_probe/worker.py、analyze.py|新前向已完成；CPU重算可直接读原无损数组，不重复forward|
|CPU闭环|cpu_rollout_parity/worker.py、controller.py、analyze.py、finish.py|已记录NPU dispersion直接注CPU，不用跨后端same-seed假定相同noise；完成两节点|
|采样指标/汇总|sampler_analysis/postprocess.py、summarize.py、sr_official_metrics.py、render_report.py、trajectory_difference_audit.py|保存图/状态SHA、PSNR/SSIM/MAE及CPU LPIPS；缓存图对避免重复指标调用|
|局部对比度|data_optimizer/local_contrast_audit.py、validate_local_contrast.py|6400身份/5700unique；稳定patch fallback，原1e-11门，初版REVIEW保留|
|全Val失败分层|failure_strata/analyze_strata.py、finalize_strata.py、render_strata.py|30000既有记录、同250源分组；不重新调用模型|
|root补充图/FFT/oracle|analysis/render_root_figures.py、periodicity_audit.py、oracle_controls.py|已有图、系数表和标量/高斯参考；高频不等保真|
|最终文档/假设/成本|analysis/write_final_report.py、write_hypotheses.py、write_cost_audit.py|只从完成证据组装；不加载恢复模型或创建optimizer|
|封存与交付|analysis/final_remote_inventory.py、check_inventory_exit.py、assemble_manifest.py、verify_delivery.py|完整远端SHA快照＋最终小文件增量SHA/本地验收；明确循环清单排除|

本地主要交付是报告/源文件/CSV/JSON/PNG/PDF，另有少量本地标量实验数组；96份完整梯度约52.7GB、64条无损轨迹、5000真flow及全部原始诊断预测保留远端，路径/尺寸/完整SHA在REMOTE_ASSET_INVENTORY.json。此快照记录采集时完整状态；最终小报告/图/manifest增量由LOCAL_DELIVERY_RECEIPT逐文件重新SHA绑定。不要将远端大数组索引称为这些数组已下载本地。

原7份checkpoint路径/完整SHA及状态读取在ORIGINAL_ASSETS_PRESERVED.json，所有原数据来源指纹在data_optimizer/DATA_PROVENANCE.json、SOURCE_PIXEL_REDECODE_RECORDS.jsonl等。原U0/U10000/Final本地完整独立checkpoint备份位于父目录saved_checkpoints，保持原文件。

主环境需按父目录run_env.sh使用原已安装依赖，实际import路径来自多个site-packages；官方和当前环境捕获在official/。没有伪装成锁定wheel的全封闭可重建环境。若复核数值，先验证源/输入SHA，并把新输出写到另一个复核目录，保留此交付的原始证据；不得修改原训练实现或继续训练。

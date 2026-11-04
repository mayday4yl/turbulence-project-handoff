# V5 历史交接准备：Simulator、Concertormer与候选更换

本材料于2026-11-04从本地既有报告、配置与验收收据整理；不代表本轮重新验收远端。没有运行模型、优化器、数据生成或网络查询。报告中的PID、空闲卡、磁盘与RUNNING均按原日期理解。逐文件当前本地SHA和分支证据见`v5_evidence.json`；历史checkpoint的SHA是原验收记录，交接准备没有重读大权重。

V5指Concertormer物理递归主线。ResShift与DiffUIR-L初审的目录虽以`v5_`开头，属于后续候选诊断；V6用户方法和正式联合训练另列`v6_history.md`。历史A/B标签在不同版本语义不同，不把V5 A/B与V6或原Track A/B混为一谈。

## 1. V4到V5实际改变了什么

`v5_v1_simulator_20261020/docs/METHOD_SPEC_ZH.md`是新方法与时间语义的入口。V5沿用预测clean/raw-ξ后物理递归的思想，但网络改为三通道当前状态`Z`和连续时间`t`，不直接接原始Y、g/G、总步数N或gamma。全局`t=1-g/4`；训练内部三点严格满足`0<t1<t2<t3<1`，端点、重复或浮点舍入退化整组三点重采样。默认真实四步推理g=4→3→2→1→0；任意N/gamma/非均匀时间是显式接口，不是所有正式训练都已运行的组合。

Simulator保留原生raw Zernike/tilt及P2S分布/物理参数，新实现Smooth、Tilt/Warp与FFT路径。正式骨干最终使用官方Concertormer RGB残差恢复`Z+restored_residual`，载入官方恢复主体和RGB层，不沿用V4的sigmoid绝对clean；新增time/raw-ξ head。未引入官方多尺度辅助loss。较早骨干审查中的sigmoid提案是历史准备，不能覆盖最终D0和A/B训练定义。

### Simulator研究接收有明确保留项

- `acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md`记录用户**带已知局限接收**；原`FINAL_GATE_COMPLETE_NOT_PASSED`保留，不能改写成所有数值门PASS。
- near-zero g=1e-4一项最大误差.002176高于.002冻结阈值；局部有限差分可能跨ReLU/插值分段边界，CPU64也没有消除全部失败。CPU/NPU梯度一致和负方向扰动等支持项独立保留。
- 非平稳协方差、raw ξ零方向、负PSF/输出越界等作为科学局限，不因工程能训练就消失。接收任务本身无正式数据生成/训练。
- A100独立审查`v5_v1_simulator_a100_20261021/a100_reports/A100_GATE_REPORT_ZH.md`同样是`A100_GATE_COMPLETE_NOT_PASSED`，CPU/CUDA32/64部分一致性通过，局部FD及近零边界失败保留；不是另一份训练完成声明。

## 2. 骨干选择与工程准备：哪些确实运行过

`v5_backbone_review_20261021/BACKBONE_COMPARISON_ZH.md`是CPU/static审查，没有下载checkpoint、运行GPU或optimizer。比较Concertormer full50,535,664、Lite28,985,644、BioIR-L15,847,328、AdaIR28,784,824参数，最终优先Concertormer。AdaIR整数频率阈值在256尺寸的空低频mask及阈值梯度问题只作为审查发现，未擅自修改算法。

`v5_concertormer_a100_benchmark_20261021/README.md`截至原报告仅静态/CPU准备，GPU benchmark及正式训练尚未执行；默认launch仅打印。它不是A100正式训练镜像已经开训的证据。32样本预备数据不等于完整新数据。

正式主线位于`liteserver-de36:/data2/yanglin/v5_concertormer_10k_20261021`。早期NPU四卡门完成，但最终正式D0/A/B为六卡NPU2–7、有效B48。独立`optimizations/psf_adjoint_20261021/REPORT_ZH.md`保持前向、反射边界和FP32数学，将固定PSF输入梯度改为等价普通卷积；CPU64/NPU/完整四步及多卡门通过。原四卡B32测得D0 21.45→9.60s/update、R0 87.28→39.09s/update。此结果是等价工程加速，不是恢复质量改善，更不是后来六卡正式成本的直接替代。

## 3. 实际正式主线：D0 10k → A 6k → B 4k

权威终态是2026-10-26 00:32的`reports/alignment_A6k_B4k_20261022/UNIFIED_VAL1000_FINAL_ZH.md`、`FINAL_RESULTS.json`及heartbeat收据，而非更早“训练未启动”“R0-D0Init 10k”或旧暂停计划。三段完整结束，最终B U4000于00:19保存/验证，controller及六rank退出、原自动监控删除。这里是历史终态，本次未重核当前设备。

1. **D0**：纯RGB一次直接恢复，无time、raw-ξ或Simulator，Clean MAE系数1，共10000更新。
2. **A**：从D0的最终best初始化新time/raw head，对预生成物理中间态做连续时间**单步**训练6000更新。按当前g/Yg做observation matching；模型选择仍在原Y4的真实四步Val1000上按末级PSNR。
3. **B**：从A best接续，真实四步物理Full-BPTT alignment4000更新，观测匹配用原G4/Y4。A/B原目标0.9Clean+0.1回投，均不detach、不clamp、不用GT状态替代、不做refinement。

数据为新V5 clean train2000/val250/test250，各4ξ、G4单帧；Train8000对、Val1000=250source×4实例。Test1000锁定未参与这次模型选择。不能和V4 Train8000 sources或V6 train10000 sources混同。网络BF16与Simulator FP32的混合配置按正式冻结记录；后来的DiffUIR FP32不能倒填本段。

### 同一原Val1000、各阶段PSNR-best的匹配结果

逐图RGB/range1、不clamp/crop/resize、refinement OFF；SSIM/LPIPS/MAE都来自相同PSNR-best，不分别择优。Input=18.795937524dB。

| 阶段 | 预算/最佳U | 推理 | PSNR↑ | SSIM↑ | LPIPS↓ | 最终clean MAE↓ |
|---|---|---|---:|---:|---:|---:|
| D0 |10000 / 8250|1次网络|21.443506|.574072|.458797|.052757|
| A |6000 / 5000|真实4步|20.156652|.505831|.509330|.066020|
| B |4000 / 2750|真实4步|21.736105|.582772|.453292|.051529|

B较D0 +.292599dB，较A末级+1.579453dB，是此固定验证人口上的增益。A best C1→C4为21.773382→20.156652，下降1.616729dB；B best21.841451→21.736105，仍下降.105346dB。因此“对齐显著减轻递归退化”有证据，“四步优于自身第一步/机制已根治”没有证据。A的Val-A单步27.082841来自更容易的中间态，不进此表。

最后U10000/6000/4000对应PSNR21.439008/20.098987/21.714916，best与latest都保留。三段Train loss的目标/状态和Val口径不同，不能直接以loss大小排名；A/B原Val loss为四步平均，并非最终图MAE。

### 完整性、故障与成本

最终B U4000完整CPU读回记录：612,255,280bytes，Adam1173状态/2组、六rank RNG、scheduler4000/4000/warmup100、cursor192000、B48/micro2/acc4/world6一致；SHA`5b1f2ff98e0cd2541899689cd6d7827a3b3f644465fc8b52ac0701f4405f21ed`。三个RUN_CONFIG共用dataset_root/data_ready SHA，40/24/16次Val250间隔完整覆盖终点；best身份及SHA查`FINAL_RESULTS.json`的selected_identities。

历史不稳定尾部、工程修复及隔离对照原样保留。最后恢复后的U1001–4000健康不代表最初失稳更新已被因果定位。授权隔离对照5worker退出、9次虚拟微批更新丢弃、未部署；恢复预训练backbone在已知坏例中压制反馈，但原LR/半LR/四分之一LR在短CPU面板均未复发，不能据此证明降LR更优。

| 段 | 训练计算h | 验证h | 保存h | 正式run计时h |
|---|---:|---:|---:|---:|
| D0 |23.86|.97|.030|25.05|
| A |16.78|2.61|.019|19.47|
| B |45.26|1.34|.014|46.66|

保留训练链正式run合计91.19h；未提交尾部另2.49h。控制器端到端97.10h含生成、等待、暂停、恢复，不把并行CPU对照91.07分钟、两优化窗口和未提交尾部再次机械相加。完整成本原文及机器字段保留。

## 4. ResShift候选：工程可行与VQ往返风险应分开

### 官方zero-shot

`v5_resshift_stage1_20261026/reports/FINAL_REPORT_ZH.md`固定journal commit`bb03b7d21614cace01787e097c8a6ab6b945227d`及官方Deblur/VQ权重。四步latent diffusion，VQ f4；UNet118,714,255参数，VQ55,322,782参数。NPU2用兼容迁移，不改原科学源码；实际FP16 autocast、HF32 off。

官方演示720p含28tile/112UNet调用；仅4张固定湍流图Input17.7892→16.9712dB，无完整Val1000、GoPro复现或正式训练，不能判定训练后上限。官方权重与VQ文件及其SHA保留于候选资产；本轮只哈希报告不重读模型。

### 梯度和codec诊断

`v5_resshift_grad_vq_audit_20261026/reports/FINAL_REPORT_ZH.md`零optimizer。FP32真实四步恢复反传和**固定真实ξ的两阶段物理**反传通过，detach负对照有效；没有证明预测ξ的完整四物理阶段训练已通过。FP16递归有NPU`Conv2DBackpropFilter 507035`阻塞，默认autocast缓存/输入梯度和下溢问题与loss-scale1024诊断记录保留，不把诊断配置当正式recipe。

严格官方VQ路径Val250 clean自重建PSNR28.7253/SSIM.8670/LPIPS.04217；50张弱态更高的32.5176不是恢复效果。`v5_resshift_vq_repeat4_20261026`对250clean连续四次codec往返、1000调用、无UNet/Simulator/训练：PSNR28.7253→27.4151→26.0999→24.9492，LPIPS.04217→.10711，MAE2.55%→3.99%；250/250逐轮PSNR下降。它支持谨慎采用冻结VQ+反复RGB物理往返，不能证明完整ResShift恢复必然如此或所有latent方法不可能。

用户随后改选DiffUIR-L，未启动ResShift正式训练。`v5_diffuir_l_stage1_20261026/archive/ARCHIVE_RECEIPT_ZH.md`于10月26日21:27对三目录7321文件/4.56GiB原地双重SHA核验；这是**原地归档/证据快照，不是新增全量备份**。旧PROGRESS的RUNNING由最终COMPLETE收据覆盖；源权重和数组仍在原目录，不能作为“已备份”理由删除。

## 5. DiffUIR-L初审与100更新pilot（仍非V6正式训练）

`v5_diffuir_l_stage1_20261026/reports/FINAL_REPORT_ZH.md`固定官方commit`c71437ce86ad1b098aa9be4678dc82e8b99649df`，Large64-1248，36,261,187参数、276参数张量/290状态。官方model-300 RAW/EMA strict匹配；含GoPro的五任务通用预训练，不叫GoPro-only。官方模型SHA`b9d07ab6b32d37d895be65c50e99bbe6b2ae6e32823e7304ec2167bc5f18843f`。

这是pixel-space六通道扩散状态/退化观测condition网络，原生训练随机1000个τ、一次denoiser raw residual L1。官方S3=999→665→332、eta0但仍有初始noise0.1、官方final clamp保留。此处网络的退化条件、扩散时间、SDT与显式task/strength prompt不同；未加task prompt、rawξ或外层物理递归。

NPU2 FP32固定Val4 Input17.7892→17.5407dB；1次真实Adam诊断更新丢弃，完整三步输入/参数梯度通过。随后正式短pilot100更新，从官方RAW重新初始化fresh EMA，G4 Y4→clean，原生残差L1；400pair冻结池中实际用200realization/91source/49类。Val50 EMA：U0 18.73989、最佳U25 18.74114、U100 18.48104，Input18.95359。小幅早期最优不证明稳定改善；EMA处默认warmup，U25/U100分别近似RAW U21/U91，不是成熟EMA。

U0/25/50/100完整checkpoint读回通过，best EMA SHA`055a8b95ccc4660f207a8c021b8184cde4a48db82b055cc5891fbc6ac530347e`。任务完成退出，旧V5资产保护对比通过；没有Test1000、多卡大规模训练或A/B物理递归。

## 6. DiffUIR-L工程benchmark

`v5_diffuir_l_benchmark_20261026/BENCHMARK_REPORT_ZH.md`：12次成功有界运行，6次初始环境FAIL为0更新，修复新目录重测；144次测试optimizer更新全丢弃，不算正式D0曝光。原生FP32最高**已测**micro16=.7095s/update/22.5505img/s；完整三步采样反传最高已测B8=1.0407s/7.6872img/s。B32原生与B16采样因安全预算未测试，不是OOM或硬件极限。

完整三步采样Clean L1是末端诊断目标，未因benchmark自动变更原生D0定义；后来的V6联合目标有独立明确授权。Train8000冻结池中实际测试覆盖986呈现/240pair/229source/50类，不是遍历全部池。benchmark fresh Adam LR8e-5等参数是工程测量配置，也不是后续V6正式LR。NPU/loader退出和旧9552资产保护验收通过；未运行外层物理A/B。

## 7. 资产、复现与未决

训练后补做的`reports/three_stage_stepwise_20261027/MANIFEST.json`绑定三阶段best D0U8250/AU5000/BU2750、4个固定Train和4个Val样例、共同Input/GT及数组SHA。它是**训练后best checkpoint重放**，不是逐update历史快照；只说明这8例轨迹，不能代替全Val。显示clip01但指标用原浮点，没有锐化；训练更新0、Test未用。

- V5正式run及权重在原de36根，完整恢复入口查正式`RUN_CONFIG`、`LAUNCH`、`FINAL_RESULTS.json`、`CURRENT_EXECUTION_ZH.md`及heartbeat完成收据。本地有报告/指标/固定图，不能据此假设全部大checkpoint本机已备份。
- 原Simulator未全通过的数值门、第一次失败与科学局限必须随包保留；不能为交接将状态改PASS。新接手人先读失败/研究接收，再决定是否额外验证，而不是盲目重跑旧controller。
- ResShift三个原目录保持冻结，归档清单只验证原地资产，不给删除许可。DiffUIR Stage1/benchmark为候选技术可行性，不代表V6正式结果。
- V5最终B有匹配Val1000平均增益，但强湍流仍未达到清晰结构恢复，且末级低于第一步；旧isolated control没有完成因果定位。Test未用，跨真实域泛化不由此表证明。
- 所有后续训练、恢复、资源使用或数据下载需要重新定位当前授权和完整状态。这里不提供自动启动命令；只读报告/配置和已有CPU审计入口足够复核历史结论。

复现读证据顺序：索引→Simulator Method/研究接收→正式RUN_CONFIG/数据SHA→三阶段FINAL_RESULTS/统一Val表→历史故障/恢复收据→对应完整checkpoint及逐图结果身份。禁止用V4数据、别的版本EMA导出或只含网络的权重替代完整恢复状态。

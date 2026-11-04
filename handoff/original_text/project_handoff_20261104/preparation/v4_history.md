# V4 历史交接准备：Noise Bridge、四方法、骨干迁移与Raw–Field

这是2026-11-04交接的历史材料准备稿。事实来自本地已有报告和验收，日期均按原报告；本次没有联网、读取远端进程、加载模型、运行评价或恢复训练。旧报告中的RUNNING、PID、GPU空闲或磁盘数字都是当时快照，不能用作当前状态。来源路径、SHA256及逐分支状态见同目录`v4_evidence.json`。大权重/数据本轮不重新哈希，下面的权重SHA是原验收记录。

V4不是v1/v1.5/v2/v2.5/v3的重命名。早期Track A/B/PiRN-Reimpl资产管理与后来的A100 Noise Bridge范围也不同；本交接不把旧窗口授权继承成可重新开训授权。

## 1. 主线与方法是什么

V4 Noise Bridge在RGB像素空间工作。网络接收`Concat(Z,Y)`六通道和原数值`[g,t]`，预测clean RGB与两块原生raw ξ（Zernike、tilt）。原始观测Y/G固定，预测clean/raw ξ经旧P2S Simulator生成下一物理状态；训练路径均值再batch均值、full-BPTT，训练中不做精修。早期网络clean用sigmoid；后来的V5不再沿用此clean参数化。

四方法应分开识别：D0单次直接预测；R0无RGB噪声混合的物理递归；R08物理递归加T0=.8逐步降噪混合；L08使用预测clean与原观测构造线性中间态，作为路径对照。不同日期实验可能使用MSE或L1等不同冻结loss，不能把最后的`LC+0.2LR` MAE口径倒填到全部早期拟合实验。R0/R08的40步逐样本raw-ξ精修是独立推理开关，网络参数不更新；D0/L08接口不适用时写N/A。

## 2. 资产与实现验收，尚不是正式训练

**2026-09-29 Noise Bridge实现。** 入口`v4_noise_bridge_20260929/DELIVERY_REPORT.md`。A100 `h3c-gpu-server01`物理GPU5；独立实现目录`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_rgb_20260929`。32,038,377参数，真实RGB256/raw512前反向、混合G reduction、末级loss跨级梯度与detach负对照通过；只做过1次临时训练更新，正式run为空。

当前公开执行选择是**逐样本串行精修**。原批量40步与同样本raw-ξ严格重复性未通过；原失败和容差保留。CUDA `grid_sampler_2d_backward_cuda`没有严格确定性实现，因此固定seed不等于精修ξ逐元素重现。工程通过不能解释成全部数值确定性、训练或恢复质量通过。

该阶段A100数据ID是`places_p2s_a100_20260928_g4321_s500_v100_t200_v1`：500/100/200 clean，4/2/2 realization，G1–4。Overfit100统计另有身份，不能沿用为train500/Train2000/NAFNet大数据统计。原环境入口`shared/envs/py310_torch271_cu126_v1/bin/python`；根注册及`versions/v4_noise_bridge/ASSET_LOCK.json`是历史资产身份入口，当前远端存在性未重新核查。

## 3. 小规模正式训练与拟合诊断

| 独立目录/分支 | 原报告确认的终态 | 可引用结果及边界 |
|---|---|---|
| `v4_noise_bridge_training_20260929` | 三组D0/R0/R08各32,000曝光、1,000更新全部完成；计划评估与进程退出验收完成 | train100×4ξ×4G=1600 starts；val100×2ξ×4G=800。RAW/OFF PSNR：D0 19.1983，R0 18.8805，R08 19.1895；同val Input20.5394。R08好于R0但未超过D0，均低于Input。 |
| `v4_metrics_supplement_20260929` | 9项固定last_32k val800补评完成，零新训练 | 输入、三组RAW/EMA OFF、R0/R08 RAW ON；同一浮点输出上PSNR/SSIM/LPIPS，不拼接不同调用。ON平均小幅改善仍未超过Input。 |
| `v4_overfit5_d0_40ep_20260929` | 完整D0五来源40epoch=400更新完成 | train80 RAW22.773847 / val64 RAW17.534017；是拟合诊断。 |
| `v4_overfit5_cleanonly_d0_40ep_20260929` | Clean-MSE-only五来源400更新完成，ξ冻结/跳过Simulator | train80 RAW22.590230 / val64 RAW17.549631；初始化/顺序等配对。 |
| `v4_overfit5_cleanonly_resume200_20260929` | **取消、未上传、零更新** | 用户纠正对象为完整D0，不可当训练分支。 |
| `v4_overfit5_cleanonly_resume_plateau_20260929` | 续至epoch200/总2000更新；达到上限，未确认plateau | train80 PSNR31.037437而val64降至15.744348；训练停止只看train，不用val早停。 |
| `v4_overfit5_full_d0_resume_20260929` | 配对续至epoch200/总2000更新完成 | train80 30.670328 / val64 15.803111；不代表泛化。 |
| `v4_one_source16_cleanonly_d0_20260929` | 单来源16 starts，400更新完成 | train16 30.560403 / val64 11.952586；同update不等于同曝光/重复次数。 |
| `v4_single_pair_cleanonly_400u_20260929` | 固定一对G4，400更新完成 | 19.440704→34.693485dB只说明该同一对可拟合；无val或泛化结论。 |

正式首段总曝光96,000，33份完整checkpoint据原报告留A100原run；本地已有完整标量、固定图和`completed_evidence.tar.gz`，不要假装所有大checkpoint都已复制本机。报告附有`reports/audit_d0_completion.py`、`audit_refinement_results.py`、`audit_final_results.py`本地只读重算入口；本轮交接未执行。

**物理可辨识性诊断另列。** `v4_fixed_physics_operator_20260929`对同一来源/真实ξ的FP64分析算子给出G1–4秩上界及非单射结构证据，但未找到预注册近零反例阈值的档位；没有算完整数值秩/SVD。`v4_blind_ambiguity_20260929`完成36×600=21,600次诊断变量更新，网络更新0；局部C–ξ补偿witness不等于盲恢复测试成绩或自然图像唯一性证明。详见各`results/report/REPORT_ZH.md`。

## 4. Train500四方法、SwinIR与PiRN骨干

旧U-Net四方法链见`v4_train500_matched_v2*`，`perf1_eval5kfull`为完整U5000阶段交付，不是任意目录名都代表最终完整结束。Train500×4ξ×4G=8000 starts；val800的共同U5000/RAW/OFF为：

| D0 | R0 | R08 | L08 | Input |
|---:|---:|---:|---:|---:|
|19.6670|19.4339|19.5665|19.3841|20.5394|

`U05000_READOUT.md`记载当时继续到U7500、上限U10000；本次未找到并独立核实后续全部终局，因此只把U5000称已验收阶段，不把旧预测ETA或上限当完成数。R08−L08 +.1824dB，但R08−D0 −.1005dB，不能用局部消融宣称全面收益。

`v4_pirn_backbone_matched4_20261001`四方法真实GPU功能验收通过，**正式optimizer更新0**；17,611,761参数，PiRN仓库的SwinIR/RSTB骨干加V4条件/双头、共同随机U0，不是完整官方PiRN复现或已预训练PiRN效果。四法G4/mixedG/full-BPTT/梯度累积、U0恢复通过，非零Adam moments正式恢复未验证。

`v4_swinir_l1_train500_2k_20261001`是新的SwinIR+L1+独立2k日程综合配置；截至2026-10-02 U2000完整交付，四组各2000更新/64000曝光，所有自有进程按原验收退出。

| 共同U2000 RAW/OFF val800 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| D0 |20.473245|.520175|.648212|
| R0 |20.443722|.516498|.653578|
| R08 |20.419548|.516473|.653467|
| L08 |20.400306|.514626|.658067|

Input20.539433；D0三指标优于递归组，但单seed有限预算不证明普遍无效。新旧同时改骨干/loss/LR，不能归因单一因素；旧U2000与旧共同网格best U1000也不是同一问题。

## 5. Restormer：候选→NPU停止→A100随机重开→matched U5000收尾

候选`v4_restormer_candidate_20261002`固定D²Turb作者commit`bbc28441b7212cc55319124a081ed6c0001879ef`，适配27,032,525参数。只对齐公开骨干，无作者训练recipe复现；共同随机U0。候选正式更新0；R0真实CUDA反传依靠验收专用CPU saved-tensor offload，R08/L08真实尺寸在CPU完成，不能说四组原生CUDA训练均已验收。D²Turb版本树未见LICENSE，标准Restormer许可不能代替它。

正式根`v4_restormer_train2000_npu_20k_20261003`：canonical Places50类2000/250/200 source，4/2/2ξ，G1–4。最初NPU0–3各法20k/B32计划已被用户迁移和后续收尾授权覆盖。NPU实停点：D0 U669、R0 U196、R08 U198、L08 U201，完整保存；**A100不是加载这些权重**，而是新共同随机U0，GPU3/5上`runs/a100_fresh_u0_20k`。旧目录名`a100_resume`不证明科学上续NPU。

10月6日NAFNet接替后，Restormer主比较收尾为共同U5000（既有>5000结果保留），不再按旧20k重启。权威匹配表`v4_nafnet_sidd116m_train8000_20261006/reports/OLD_U5000_FULLVAL_SUMMARY.md`：Val250×2ξ×4G=2000，RAW/OFF、CUDA BF16、batch1：D0 22.7106 / R0 22.7497 / R08 22.7013 / L08 22.7123dB。这个表不能和后续NAFNet fullVal8000直接当公平骨干排名。旧L08 U5000 fullTrain32000补评已于10月7日独立CPU完成，不能被更早PENDING/ETA覆盖。

## 6. NAFNet：候选与两个正式阶段

**候选。** `v4_nafnet_sidd116m_candidate_20261005`官方NAFNet-SIDD width64 115,982,915参数，V4适配117,767,209参数。官方SIDD文件strict load保留；V4只复用预训练主体，六通道intro、新FiLM、sigmoid绝对clean/raw头不是完整官方残差去噪映射。随机U0验收为旧参考，当前候选默认是预训练共同U0。候选B1/G4验收不是正式B32训练；R08/L08初始梯度偏大虽clip后finite，不等于任务已有效。预训练验收在`reports/pretrained/ACCEPTANCE_SUMMARY.json`。

**Phase1 Train8000。** `v4_nafnet_sidd116m_train8000_20261006`，Places4000+LSDIR4000 train；Places500+LSDIR500 val，旧Test200锁定。LSDIR held-out来自官方train源池，因为公开val只有250；不是官方val500、也不是对全部LSDIR均匀抽样。de36唯一canonical生成，A100两卡训练。实际最终工程配置micro16×acc2/B32/FP32，峰值LR5e-5、全参数、四法共享适配U0/样本流；较早micro4/8文本保留历史。每法20000更新/640000曝光=5epoch，104项评价于2026-10-12完整验收，原监督删除。

| U20000 RAW/OFF fullVal8000 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| D0 |23.09077535|.64696799|.37889746|
| R0 |22.91951237|.63865431|.40178125|
| R08 |22.73627484|.62923249|.41617541|
| L08 |22.81135343|.63339814|.41042081|

此处fullVal8000是1000clean×2ξ×4G，不能误读成8000独立来源。D0均值三指标最好。最终REV4结果包另补R0/R08 online refinement ON完整Val8000；D0/L08 ON为N/A。`deliverables/U20000_FOUR_METHOD_RESULTS_20261016_REV4`保存19页PDF、Excel、CSV、曲线、fixed raw数组与来源，不能用REV1/2的缺项覆盖REV4。

**Phase2 Train15000。** 从各法原U20000完整状态接续，新增7000源；globalU和phaseU分列。用户于2026-10-18直接停止：D0/R0已globalU40000完成；R08 U32891、L08 U21559原生保存停止，四法并未共同U40000完成。完整checkpoint含RAW/EMA/AdamW/RNG/sampler/nextLR，原`STATE.json`可能仍RUNNING但以USER_STOP收据为准，未获自动恢复授权。

同口径fullVal8000：D0 U40000 online23.450488489/.665960092/.351608030；R0 U40000 online23.329191248/.659748682/.367896976。R08已验收中期U27500 full23.026709205/.643240134/.390925936；停止点U32891未补评。L08停止U21559未补评，U21000 monitor不是full。ValDetail500是按细节用途筛选的500-source子集，不是独立Test或原Val1000随机代表人口；D0/R0 U40000子集分数另列，不混主表。

## 7. 连续差分与Raw–Field支线

`v4_followup_pilot_train15000_20261013`初次报告尚无共同里程碑；不能拿它当配对完成。真正完成的`v4_followup_pilot_lr5e6_attempt2_20261014`是Train1000/G4、D0与CONT_DELTA_R0各U1000、Val200源/400输入、FP32 online：D0末级18.753920，连续差分C4 18.421053，差−.332867dB，source-bootstrap95%CI[−.400783,−.275560]，PSNR胜率4%。原数据/日程和更新规则需查该独立冻结，不把它叫作只改raw坐标的实验。

`de36_g4_raw_vs_field_12h_20261015`于10月15日完成共同U1000与0/500/1000全val64；Train128source、Val32source×2ξ、G4-only、651,417参数、共同clean-only warm500初始化。warm不是独立D0 baseline。Raw状态raw512，Field是alpha/u的256场；两组共享初始模型/顺序、完整四级LC+.2LR，零GT状态监督/teacher forcing/mixing noise。

| U1000 full val64 OFF | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Raw |19.836685|.486571|.710429|
| Field |19.827981|.486735|.709152|

同Input20.133437。Field−Raw PSNR−.008704dB，探索性source-bootstrap95%CI[−.017554,−.000244]；SSIM/LPIPS略好，不能称全指标都差。两组C4均低于C1。另8-source refinement仅辅助；J降低并不保证GT恢复PSNR提高。报告记录首次CPU checkpoint审计意外初始化NPU0及修复后CPU-only重验，不能把这段成本隐去。真正最终权威入口是`reports/FINAL_CPU_AUDIT/FINAL_REPORT_ZH.md`、`AUDIT.json`、`CHECKPOINT_AUDIT.json`及`DELIVERY_COMPLETE.json`。`de36_raw_vs_field_quick_pilot_20261015`只检索到准备分支，当前未独立确认其完整终态，不代替上述12h正式结果。

## 8. 必须保留的资产与归档恢复

- Noise Bridge、旧U-Net/SwinIR/PiRN候选、Restormer、NAFNet的冻结源码/配置/数据身份/完整状态/逐图结果各自保留。**不同目录的U0、统计、数据人口、EMA和refinement不能混用。** 旧PID不用于当前kill/恢复。
- 2026-10-02清理：93个指定中间权重先逐SHA备份到`storage_archives/A100_cleanup_20261002/files/`后才从A100删除；缓存未另存。该目录是部分中间权重的唯一保留副本，不能当临时下载缓存清理。恢复查`PREFLIGHT.json`，只恢复所需相对路径，禁止整目录覆盖。
- Train15000用户停止时四法latest完整权重本机备份：`v4_nafnet_train15000_20261013/deliverables/Train15000_STOP_20261018_checkpoints/`，原验收总7,541,233,596 bytes。既有结果归档`Train15000_STOP_20261018_existing_results.tar.gz`含5710文件；原整体SHA`183a6204d4949df90448049f44f997e4ef7d829352a618e223780f8cac9a8385`。
- 2026-10-25存储清理：126,050文件先完整归档、逐文件回读和fsync，后删除A100清单副本，释放101.7426GiB。归档远端`liteserver-de36:/data2/yanglin/storage_archives/A100_cleanup_20261025_train15000_data/files/`保留原相对路径；新增7000源original/clean/states未来若需续训须先恢复核SHA。Train8000、约97GiB rawξ、最终/最佳模型和全部结果被保留。验收与恢复文档在`output/a100_cleanup_100g_review_20261025/execution/`。本次未重新联网确认该远端归档存在或当前磁盘余量。

## 9. 交给下一人的最低工作顺序

先读`EXPERIMENT_INDEX.md`及各实验最后的停止/完成收据，定位**相应数据和完整checkpoint**，再查readme/CONFIG/LAUNCH/环境/源码manifest。任何新操作均重新核远端主机、路径、文件SHA、授权及共享设备占用；历史存活声明不自动有效。恢复只能从对应完整状态，不拿显示PNG、EMA导出或其他骨干权重当等价完整恢复。

V4积累的主要证据是：实现与full-BPTT可以验收；小样本可拟合但验证下降；几种递归与精修方案在已测匹配人口上未表现为持续优于直接D0。它们是限定配置的负面/混合结果，不证明所有物理递归不可能有效。V5从V4-R0思想演进，但新Simulator/时间语义、RGB3输入、预训练恢复clean head及数据均另冻结；不能把V4结果表直接改名为V5。

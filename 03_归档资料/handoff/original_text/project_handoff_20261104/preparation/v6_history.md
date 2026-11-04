# V6 历史交接准备：DiffUIR-L正式链、混合强度与外部泛化

整理日期2026-11-04。所有终态和指标由本地已有历史报告/验收交叉整理；本轮零模型、optimizer、NPU、联网、远端修改和清理。运行状态注明原报告日期，旧PID或RUNNING不当实时。当前本地来源SHA与事实记录见`v6_evidence.json`，大模型/数据未重新哈希，历史权重SHA按原验收引用。UniDB后继不属于本文件，由总交接单列。

## 1. 方法与实际训练链应按修订分段

`v6_diffuir_l_method_review_20261027/{METHOD_REVIEW_ZH.md,SPEC_TO_CODE_MAPPING_ZH.md,OPEN_ISSUES_AND_ACCEPTANCE_ZH.md}`保留用户完整规范、代码对应和未决。该方法审查本身无训练。DiffUIR-L为pixel-space扩散状态+原退化图六通道condition、原生扩散τ/SDT；它没有因此自动拥有显式task/G prompt。V6 A/B为预测raw-ξ驱动的物理外层；D0是单帧恢复，内部S3不等于物理递归。

正式链有三次关键定义变化，不能把全部loss画为同一目标：

| 阶段 | 实际训练定义 | 真实结束/选择 |
|---|---|---|
| 初始D0联合目标 | `.5×原生raw残差L1 + 三步末端Clean MAE`；训练4次denoiser，推理S3 | U10000完成；续到U13334，再四卡续到U14927用户暂停 |
| 官方原生残差续训 | U14927完整状态起；删除.5与Clean分支，原生随机τ残差L1系数1，训练1NFE/S3推理 | U95544用户停止并完整验收；不是官方U0起的纯原生对照 |
| 多强度及重加权 | 继续同原生目标；G1–4等比例→同父Control/偏G4有限分支 | 多强度U157505结束；两5k支各U162505结束，追加门未通过 |

初始G4 Train10000clean×4ξ=40000pairs，固定Val250×4=1000。后期mixed160k是上述10000clean×4ξ×4G，不是新增160000clean。旧V5单帧D0较小训练人口、clean MAE、骨干和精度均不同，不能把指标差单因果归到diffusion或LR。

## 2. 初始D0、A/B及暂停的真实终态

### 初始D0联合目标与短续训

`V6_D0_A_B_FINAL_REPORT_ZH.md`日期2026-10-27 09:01，实际是**初始D0完成的阶段报告**，不是后续全部V6最终状态。六卡NPU2–7，FP32，B60=micro5×6×acc2；Adam LR2e-5、warmup200/cosine到2e-6、clip1，EMA每update即时更新。U10000/600000呈现/15遍，完整rank/状态保存。

| 同G4 Val1000 | PSNR | SSIM | LPIPS | MAE |
|---|---:|---:|---:|---:|
| Input |18.79593752|见原逐图表|见原逐图表|见原逐图表|
| 官方U0 EMA |18.59224024|.45175667|.51510603|.08007468|
| 初始D0 U10000 |21.03393849|.54745733|.53120907|.05637289|
| 续训U13334 |21.05311102|.54839314|.52982604|.05621891|

U10000相对Input PSNR改善，但LPIPS比官方U0变差，不能称全质量维度改善。U13334比U10000仅+.01917dB；+3334更新/200040呈现，不以延长自动证明还欠训练。初始权重SHA`505d1f865e46c429fccc5a7701c6216cc7f088176d1e78549960f6cc3f653e39`，续训`88c27a1997252542b5d40bd857921c44d62b9161f751679906ea005b21561b85`。

四卡限时段从13334到14927，+1593更新/95580呈现后用户暂停，SHA`0c7903726f262c56fb40333dcaf70e0e205109c618456bb39e216987bd48d9b6`；该暂停点本身当时没有新完整Val，后续作为原生分支起点才完整评价。U14000 best21.0562583不是U14927终点质量。

### A完成，B在9500暂停；不补成计划10000

`reports/A_COMPLETION_REPORT_ZH.md`确认A实际6000更新/259280呈现；六卡前2410步B48，后五卡2411–6000 B40，不能用6000×48反推曝光。A best U1000原Val1000四步末级19.86120922、第一步20.99742496；latestU6000末级19.79950879。较早Val-A容易中间态与末级均不能当D0匹配结果。

B从A best EMA新初始化，pilot丢弃；`reports/B_PREFLIGHT_PILOT_AND_STARTUP_REPORT_ZH.md`记录工程门和启动，不是完成声明。A-best初始Val1000复用有4份逐数组零差证明，不冒称新完整U0评价。

最新终态由`execution_20261027_five_npu/STATUS.json`和`execution_20261028_d0_until_noon/B_PAUSE_ACCEPTANCE.json`确定：用户要求暂停B、继续D0，B实际**U9500/cursor95000**完整保存，CPU读回与退出PASS，SHA`e49abaae1e487e63e73a3c3151b8f5585bd4945f9a15fde103ffc1bd572f7337`。旧G7/预算10000不等于已执行10000。

交接额外只读核实`distributed/B_five_npu_formal10000_01/BEST.json`及对应full SUMMARY：best U7000的末级PSNR21.04583810/SSIM.54874421/LPIPS.53296623/MAE.05630583；C1=21.08409921，C4−C1=−.03826111dB。best完整SHA`afa31bbfd0c40c270d41c92cbe00623f4046ef8159df981e8defa7e9dee2db8e`，BEST内验证字段与U7000原SUMMARY逐项相等（仅BEST额外记录update）。最近已保存full是U9000：21.05567491/.54914038/.53322902/.05623901；高PSNR但并列规则仍保留LPIPS更低的U7000，best不是严格PSNR峰值。停止点U9500未找到完整Val1000 SUMMARY，不拿U9250 monitor100或U9000冒充终点；未经新授权不恢复A/B或补模型评价。

## 3. U14927→95544：官方原生残差续训实际结束

权威最后报告`execution_20261029_d0_user_stop/D0_OFFICIAL_RESIDUAL_FINAL_REPORT_ZH.md`覆盖更早中午、microopt的中期终点。此段继承联合目标训练过的RAW/Adam/EMA/RNG，**不是**从官方checkpoint随机τ原生L1全程训练的对照。

NPU4–7，FP32，B60=micro5×acc3×4，常数LR2e-6，无新warmup/抬LR；唯一目标修订为原生raw残差L1系数1。最终U95544，比父+80617有效更新/4837020呈现，总5732640呈现。四正式run合计18.5432h，保留关键训练16.4174h；成本边界详原表。

| G4 Val1000 EMA | PSNR | SSIM | LPIPS | MAE |
|---|---:|---:|---:|---:|
| 父U14927 |21.06020173|.54878696|.52907033|.05615627|
| best=latest U95544 |21.16514237|.55626089|.49768281|.05511670|

完整SHA`1fd16df15c64cc496bda89ccb32fd6a9413c9de8e59942fc13cbb6592e757cc3`，CPU核RAW/EMA276张量、Adam、四RNG、cursor/buffers，原worker全部停止。旧U80171/21.13735575只是中期，不再当最终。

TurbWNet外部新进程触发资源guard，U30920未存尾部丢弃，从完整U30500恢复；420计算更新/25200呈现是额外成本，不算保留训练链。micro15×acc1同B60实测较baseline慢3.039%，未采用；3+20临时更新及恢复probe丢弃。不能以低设备利用率或临时吞吐替代完整端到端收益证据。

## 4. G1–4等比例多强度：全五档完整结果与强档退化

主入口`execution_20261029_d0_multistrength160k/final_20261030_1800/D0_MULTISTRENGTH_REPORT_ZH.md`。从U95544接续至2026-10-30 18:00安全更新边界U157505：+61961更新/3717660呈现，约23.2354个160k呈现当量。FP32/B60/constant2e-6/nativeL1/train1NFE/S3不改，仅训练等比例G1–4；G0.5仅评价。

各档Val1000=250source×4ξ，匹配输入/GT/固定噪声；下表为latestU157505，MAE按full报告四指标表，未按各指标择优。

| G | Input PSNR | latest PSNR | SSIM | LPIPS | MAE |
|---|---:|---:|---:|---:|---:|
| .5 |25.170834|29.92958440|.91963835|.05689790|.019876|
| 1 |22.241768|27.82736964|.87082889|.08860560|.025426|
| 2 |20.306366|24.81663086|.75823808|.16514046|.036972|
| 3 |19.471852|22.62914854|.64359417|.27629674|.047922|
| 4 |18.795938|20.89072573|.52813711|.47597871|.059168|

弱档显著改善，但父U95544 G4是21.16514237/.55626089/.49768281/.05511670，latest G4 PSNR−.27441664且SSIM/MAE变差，LPIPS改善。父模型弱G1约15.388083说明原先G4专训对弱档很差；不能只报告弱档收益掩盖强档代价。

既定综合best是**U109544**，G4 20.98248545/.53580052/.50575101；best标准先保护G4≥父−.20，再五档macroPSNR及近似平局LPIPS/SSIM/更早U规则，不是无条件G4峰值。latest低于20.96514237保护线；“best”和“latest”及目标应分列。

latest SHA`9e6b66e6a46434dc86b327dfce6be04d78aa1de61449d7791e282c62d52e0029`；best`5d7c57408af1a296f940997406c4bb2b2e876b9100ae20cd30ce2690a0d693a8`。`FINAL_ACCEPTANCE.json`及`LOCAL_FINAL_DELIVERY_ACCEPTANCE.json`记录全状态、五档完整评价、保存边界、退出与同步。CPU safeguard v2修复的是重复保存元数据核验，不是改训练科学代码。

## 5. Control vs G4-biased：必须保留的5k终局

两支从同一U157505完整父状态，各5000更新/300000呈现→globalU162505。Control均匀25%各G，biased G1/2/3各1/6、G4一半；同clean/ξ位置/τ/seed、native残差L1/LR2e-6/EMA不变。单物理卡micro5×acc12保留四逻辑rank槽；没有声称四卡HCCL逐位等价。所有工程/pilot更新丢弃后从精确父开始。

原报告`execution_20261030_d0_g4_reweight_5k/final_20261031_0255/D0_G4_REWEIGHT_FINAL_REPORT_ZH.md`确认双方完整1k/3k/5k、五档各Val1000，同输入GT噪声，父0点明确复用。

| G | Control5k PSNR / SSIM / LPIPS | G4-biased5k PSNR / SSIM / LPIPS |
|---|---|---|
| .5 |30.15297247 / .92384500 / .05296033|30.04724353 / .92208179 / .05437173|
| 1 |28.13339141 / .87930328 / .08375479|27.97340765 / .87524054 / .08538143|
| 2 |25.08305824 / .77048628 / .15962950|24.90353741 / .76263299 / .16171529|
| 3 |22.76327627 / .65209282 / .26877723|22.65574966 / .64580299 / .27378875|
| 4 |20.88218708 / .52769734 / .47289061|20.96989413 / .53304314 / .47828576|

G4-biased对同预算Control G4 +.08770705dB，250source配对95%CI[.07842587,.09601288]；对父+.07916840，对旧U95544仍−.19524824。相同预算下弱G1/2/3 PSNR分别低.15998376/.17952083/.10752660dB，LPIPS也较差；G4 PSNR/SSIM小幅回升但LPIPS较Control差。

**追加门未通过，两支均结束，额外0更新。** biased3k→5k仅+.01862229dB，低于预冻结+.02趋势要求，且部分弱档保护失败；不能因刚超旧G4保护底线.00475176dB就宣称本轮成功或自动替换综合best。无进一步比例搜索，A/B继续暂停。

实际Control呈现G1/2/3/4=74924/75054/75146/74876；biased=50018/49896/50088/149998。偏置组1.875当量不等于均匀遍历160k所有组合。两支各13完整CP，26全部SHA与CPU状态、全部新节点均值/逐图SHA/bootstrap已按原验收通过。

Control SHA`50a7162ba7c821e5e91ef79197218405dea3743b9bda955a50120537bb1027fb`；biased SHA`2dd267d35428c92433576bd5fae78a1af60efcb0f3e5bda7cda4e29f9341ba7d`。

原逐位重启FAIL仍保留；独立完整Adam既有数值标准PASS，冷正式/热probe额外moment逐元素FAIL未伪称全状态相同。CPU修复不改训练actor/数学，原失败与两个修复源随包。固定图仍明显模糊/几何结构伪影，小幅平均增益不等于强湍流清晰恢复。全局G4历史131节点和双分支图可复用，但不同目标阶段train loss不能直接连线归因。

## 6. 外部候选：取得资产与完成评价不是一回事

**TurbNet 2026-10-28。** `turbnet_zeroshot_20261028/reports/FINAL_REPORT_ZH.md`固定官方commit`4ec7239b47328d286a834d50636411cf82741ee8`。当时未找到可获取官方权重，按用户要求终止；环境/模型/smoke/Val1000/optimizer均0。静态demo缺陷不能证明效果差，随机初始化替代也没跑。

**TurbWNet原分支。** `turbwnet_zeroshot_20261028/reports/FINAL_REPORT_ZH.md`取得官方130,013,973bytes/709tensor/32,430,816参数checkpoint，SHA`a039c67176a0704a9db22cfefe6f58e78d09681e50c03ce73bfad34290b0843f`。CPU strict+一次真实forward通过；NPU7资源冲突触发D0guard后用户停止，本分支NPU完整smoke/Val1000未做。不是OOM、效果失败或所有后续评价都未做；下面10月31日独立外部任务后来实际完成TurbWNet OTIS4328。

## 7. 真实域与官方五任务：证据支持域差/遗忘，不能替代因果实验

### U95544原生单帧外部评价

`v6_diffuir_l_20261027/external_eval_20261029_otis_atd/OTIS_ATD_EVALUATION_REPORT_ZH.md`，零训练：OTIS共5195恢复，16个Pattern序列4328有mask GT定量；Door300/Car567只看定性。ATD38视频各5固定帧190无GT。16序列macro Input→D0：PSNR-M 10.372096→8.515455，SSIM-M11 .311215→.205198，LPIPSvalid64 .274517→.389673，MAE-M .260297→.292903；0/4328 PSNR改善，16序列均下降。

这是灰度luma+mask、全原尺寸无resize/tile的补充协议，LPIPSvalid64也不是常规RGB验证LPIPS；不能与G4 Val1000数字直接拼表。权重/采样finite检查通过，没有确认是加载错误；不能把simulated同域收益当真实域成功。

### 官方五任务13子集完整诊断

`v6_diffuir_l_20261027/diagnostics/official_five_tasks_20261030/all13_final_delivery_20261031_0650/FINAL_REPORT_ZH.md`对官方EMA与综合D0 bestU109544：13/13完整子集、43321配对输入/86642恢复/259926UNet调用、0训练。所有13项D0 Y-PSNR更低、RGB LPIPS更差，支持跨任务能力退化；不能分辨退化由何时/哪项训练目标造成。

例：GoPro1111官方27.81591→D0 25.93251 Y-PSNR，Rain100H27.92963→13.63631，SOTS35.28574→17.17378。量化Y/RealBlur ECC/各官方协议与本项目RGB不同，不把13项PSNR任意平均，也不叫官方论文全量逐项精确复现；Test100有98实文件等边界原样保存。

## 8. 外部离线包与ATSyn后续最终补全

`v6_diffuir_l_20261027/diagnostics/external_singleframe_20261031/HANDOFF_CURRENT_ZH.md`顶部的11月1日完成收据覆盖下面旧RUNNING/下载中。六固定组OTIS4328/Text400/ATD190/CLEAR64/RLR225/BVI2026十五张=5222输入，另Door3；三EMA候选bestU109544/Control5k/biased5k共15675恢复、47025UNet调用，另TurbWNet OTIS4328完整完成。0训练/无A/B。Text OCR8000词为本机Apple Vision诊断非论文AWDR；CLEAR为pseudo-reference非GT；NIQE只在共同finite支持集比较。

原离线包`v6_diffuir_l_20261027/diagnostics/external_singleframe_20261031/output/V6_最终组会离线包.zip`，80,115,677bytes，SHA`3cb516cd72be40e3cab98f160f0db75931f928d297c0690885c779ddf0fdf59a`；760文件本地/远端SHA及ZIP CRC接受，原自动跟进删除、实际NN/CPU退出。初期Heat404/ATSyn quota/Box404是当时阻塞，不可覆盖后面ATSyn独立完成。

### 最终ATSyn-Static固定单帧999配对

`v6_diffuir_l_20261027/diagnostics/external_atsyn_singleframe_20261101/FINAL_DELIVERY_ACCEPTANCE.json`及`v6_diffuir_l_20261027/diagnostics/external_best_final_singleframe_20261101/CURRENT_HANDOFF_V3_ZH.md`确认：官方1000test场景固定`turb24.png`/`gt.jpg`，选择性ZIP2000成员CRC/SHA；1个与Places近重复场景排除后999正式评价。Input、best109544、final157505，原生512/S3/EMA；1998恢复、2997逐图记录含Input。7worker及CPU退出、PSNR/MAE独立重算零误差、全部输出SHA/指标均值/固定图本地同步按原收据PASS。

| 同999场景 | PSNR | SSIM | LPIPS | MAE |
|---|---:|---:|---:|---:|
| Input |21.38151965|.58283467|.55090750|.05836579|
| D0 bestU109544 |21.47726880|.59024354|.52245069|.05688721|
| D0 finalU157505 |21.58389958|.59554888|.49099741|.05629226|

该固定单帧自定义协议上有小幅平均提升，但不是DATUM论文默认20帧/effective4/噪声.0001流程复现；CW-SSIM未测，不以局部提升掩盖OTIS失败。官方图例选帧来源不确定时明确“论文参考图/非逐帧对齐”，不得混入定量。

`CURRENT_DELIVERY.json`最后更新2026-11-01 14:10：最终本地可编辑PPT为`v6_diffuir_l_20261027/diagnostics/external_best_final_singleframe_20261101/output/singleframe_with_image_metrics_v4_final/V6_External_Generalization_Evaluation_With_Image_Metrics.pptx`，SHA`bda4f94db8df3074b6fa5ca350a1f1e63c76ce74763e84ebc123f9686b067359`，27图caption/数据集均值复核。Text final400恢复虽完成，**全400 final OCR未算**；新增OCR只有4帧/12图/60词，不能装成完整复测。此本地PPT修订零新恢复模型调用，未重新远端上传。

## 9. 独立诊断和候选研究的覆盖索引

以下目录均在`v6_diffuir_l_20261027/diagnostics/`下；这些任务的“完成”不代表产生新正式训练权重，也不自动重开A/B。

- **D0_diffusion_dependency_20261027_1955**：U13334与官方EMA，固定50独立source、4seed，零更新。D0原生C1→C3平均+.223054dB，40/50改善；状态/时间仍有响应，反驳“完全忽略状态/时间”，但不证明生成分布能力或优于独立训练普通恢复器。固定状态/时间干预离开联合训练分布，不能把巨大下降全算正常扩散收益。
- **diffusion_mechanism_audit_20261028**：官方/U10000/U13334三个独立CPU副本、12source、NFE1/2/3/4/6、合法q状态/敏感性/真实分支梯度，零optimizer/NPU。原生项与Clean项都有梯度，未发现活路径的意外detach/改目标；官方forward终端方差约1而采样init约.01的定义差异确有证据，未建立严格后验一致性，也未擅改成V6新算法。CPU与NPUGaussian不同，不将该CPU子集冒称重现完整NPU验证。
- **d0_u14927_train_val_generalization_20261028**：固定Train300source×4=1200，与已保存Val250source×4=1000配对口径。Train PSNR比Val低.140351dB，差距区间跨0；支持当时Train/Val都不足，不证明完全无过拟合或一定欠训练。不是Train10000全量结论，也不外推到后续原生loss阶段。
- **d0_u95544_g1_g4_val1000_20261029**：四档各1000完整，0训练。原G4-only模型对G1/2明显过处理，PSNR相对Input−6.8537/−3.9315dB；这是后续混强度动机，不是目标错误的因果证明。G4结果与原验证逐图精确对齐。
- **d0_train_val_loss_20261029**：U95544新增完整Val1000同目标残差L1=.087682003；最近1000在线RAW train=.089182109；三步clean MAE=.055116700另列。仅终点有真实Val residual点，不能伪造同目标Val历史。RAW在线和EMA固定τ/noise也不是完全等价采样。
- **strength_preview_g025_g4_20261029**：32source×6G=192退化、32clean，原ξ/同源，不是连续再次退化；G1–4相对旧128浮点输出最大误差0。只有物理模拟、恢复模型0、训练0，不作全量质量证据。**data_strength_inventory_20261029**是当时数据清单，旧Train8000缓存不自动成为当前V6轨迹数据，真实OTIS标签也未与g标定。
- **post_meeting_audit_20261101**：报告结论为受审范围未确认新实现bug，带科学限制；原160k数据/无source交叉、5000观测cache及250clean重SHA、150000旧逐图与额外5000结果复核等原证据保留。它不是P2S物理真值验证，也不证明整个项目不存在bug。
- **d0_global_progress_20261029_1436、d0_residual_progress_20261029_1423、d0_global_train_val_review_20261031**：已有历史/曲线/选模重放与图表交付，不另算模型训练。**group_meeting_d0_mixed_20261031**为组会资料，引用其上游数值而非把PPT作为首要实验证据。

### 11月1日候选调研与边界修订（没有新模型试验）

`pixel_diffusion_selection_20261101`与`pixel_diffusion_selection_v2_20261101`分开保留。v2按用户取消“桥优先”的新要求重新排序，考察DA-CLIP、Resfusion、WeatherDiffusion、PixRestore、ADM+新ControlNet等；源码/权重入口/HEAD/LFS元数据不等于payload取得、strict load、NPU或湍流质量验收。FoundIR与DiffUIR核心AST28/29相同，是同架构预训练初始化对照，不能只因年份更新称新骨干。NFE需同时记条件编码、DINO、patch和CFG支路成本。

`pixel_nonres_topconf_expansion_20261101`补GSAD/PyDiff/ShadowDiffusion/IDM等遗漏；`recent_pixel_ir_20261101`补近期UniDB/ResFlow/UDBM/GenDR-Pix/LearnIR并修正UDBM会议身份。后者部分当时只有README/资产缺口，不是可直接使用模型。“非恢复残差”特指不以Y−C状态/目标，不排除预测Gaussian ε，也不按骨干ResBlock判断。

`pixel_deblur_selection_20261101`核实IR-SDE官方去模糊单文件元数据548648748bytes，但未下载/加载；MSG/DP-IR为论文级待补资产。`pixeldit_fit_20261101`确认PixelDiT RGB生成/rectified-flow先验与可用生成权重，是需要新条件适配的研发方向，不是现成湍流恢复器；不能移植旧ResidualL1/SDT当原版训练。`deblurdiff_diffbir_fit_20261101`指出DeblurDiff/DiffBIR符合预训练条件控制思想但用VAE latent，不满足当时用户无latent约束。上述研究均零候选模型实例/权重载荷/推理/训练/加速卡任务，报告优先级不是效果排名。

UniDB相关`unidb_pretrained_audit_20261102、unidb_vs_pixeldit_20261101、unidb_pixeldit_cost_review_20261102、unidb_turbulence_plan_20261102、unidb_learning_site_20261102`是转入后继UniDB的资产/计划/学习材料，由总交接UniDB段接续；它们也不能代替UniDB实际最终评价。无版本前缀的真实图/DMAT/Heat/OTIS/AT-DDPM等顶层支线由总交接`extra_branches.md`负责，避免漏记或重复算成本。

## 10. 接手入口与不能自动继续的事项

- 正式数据/运行根是`liteserver-de36:/data2/yanglin/v6_diffuir_l_20261027`；分段execution目录、父SHA、STOP、source freeze与完整checkpoint逐一对应。A/B只读保留，最新D0所有支线已按各报告停止，不从旧controller/PID复活。
- 质量结论依据具体checkpoint、EMA/RAW、推理S3/物理四步、数据人口、指标定义。最优G4、综合五档best、latest和Control/biased终点不是同一模型。输出图片不是完整恢复checkpoint，EMA导出不能替代Adam/RNG/loader/scheduler。
- 复核按“原报告→FINAL_ACCEPTANCE/停止收据→逐图CSV/预测manifest→checkpointSHA/EMA身份→复现源码/环境/配置”读取。预测主要留远端，本地有小图/CSV/离线包，不默认全部预测和checkpoint已备份。
- 本轮看到的是保留资产路径和原验收，不代表当前远端文件/存储已核。任何新续训、重评、清理先核授权及身份，不因为交接列出复现入口就执行。
- 可复用原生D0、full五G结果、配对来源、固定图、完整训练链及失败证据；多强度与偏G4给出明确trade-off，外部/官方五任务表明跨域与遗忘问题值得保留。它们没有证实只加warmup、换LR、更多训练或重加权即可修复，也没有证实所有物理递归/扩散方法必然无效。

后续心跳应把本历史准备并入总交接，在最终UniDB实际模型/CPU审计/目视/本地交付完成后才形成总终稿；本文件完成不等于UniDB最终完成。

# UniDB 主线交接说明

证据整理日期：2026-11-04。D0和封存forensic保留原历史边界；双路线已完成停训、最终评价、独立CPU验收、全部自有进程退出和实际unique完整权重备份。实验原件交接与全项目主文档、资料包和聊天交付分别记录，后者不由本章自动宣称完成。文件索引及SHA见`unidb_evidence.json`。本稿不授权恢复任何训练或重新进行诊断推理。

## 这一阶段解决什么问题

这条路线将官方UniDB的像素空间条件扩散恢复网络用于单张RGB湍流图。它与前面的物理递归路线不同：网络没有预测raw-ξ，没有外层Simulator递归，也没有A/B训练阶段。原生100步SDE是扩散模型内部的恢复过程，不能写成100次物理重退化。

官方源为`UniDB-SOC/UniDB@1bb63e318891b932c6e641b98729643e46b91e06`，初始化为官方DIV2K SR `gamma_1e7.pth`。官方权重SHA：`a6bb76bfabe32b8403d8b112f0ce7876114d830d20b9f4a4119aa860afef64b4`。网络接收`concat([x_t-y,y])`与扩散时间t；退化图条件始终提供，没有额外显式强度/task prompt。原生MatchingLoss L1比较预测的一步反向均值与使用clean构造的目标均值，并不直接最小化最终100步图像的PSNR。

依据：[D0最终报告](../../unidb_d0_20261102/FINAL_REPORT_ZH.md)、[执行路径与数学审计](../../unidb_d0_20261102/forensic_audit_20261102/FINAL_FORENSIC_AUDIT.md)、[官方/本地差异](../../unidb_d0_20261102/forensic_audit_20261102/OFFICIAL_VS_CURRENT.md)。

## 数据、评价与不能混用的口径

- 仍是Places365，50类、10000个训练clean来源；每来源4个原始ξ实现×G1/G2/G3/G4，共160000组合，每档40000。组合数、呈现次数和独立来源数是三个不同量。
- 训练清单来自V6已经冻结的160K组合，UniDB重新核验身份/配对/分割/SHA/shape/finite后复用，没有在UniDB阶段生成新退化。
- 正式完整验证为G0.5/G1/G2/G3/G4各1000图，每档250个验证来源×4实例，总5000。G0.5仅验证，不混入训练。
- 监控为G1/G4各100图，每档25来源×4实例，共200。不能把监控均值和完整验证均值直接比较；诊断的50来源/8来源等子集同样单独标记。
- 主评价为EMA、原生100步SDE、固定种子、原始RGB浮点，不clip/crop/resize；展示图片可裁剪到显示范围。四指标为PSNR/SSIM/MAE/LPIPS；记录RAW monitor不能替代EMA完整主结果。
- 同模拟器、source-disjoint的验证只能支持这一协议内的内容留出结论；没有建立外部真实湍流或锁定Test泛化成功。

训练清单远端：`/data2/yanglin/v6_diffuir_l_20261027/execution_20261029_d0_multistrength160k/TRAIN_MANIFEST.jsonl`，SHA `d4898f32bff9deb2c40dbfdfce0c427e7dd4213c824e32c560e5a2afa8ba2a24`。最终交接台账应继续以实际DATA_GATE和清单记录核实，不依赖这段文字替代验收。

## D0原始八卡微调：已经完成的历史实验

独立目录`unidb_d0_20261102/`。官方网络/桥/MatchingLoss L1保持，effective B96，FP32/HF32关闭，Adam LR1e-5。早期CPU/NPU前向不一致定位到GELU数值路径，经原数学erf公式的等价修复后通过原容差；该历史FAIL与修复不能混为正式训练中途故障。

实际从官方U0重新初始化；200步工程Pilot不作为正式权重。正式于2026-11-02开始，用户18:39明确要求提前停止。完成在途更新后停在U45411，累计4359456次呈现，相当于160K组合27.2466轮。完整RAW/Adam/EMA/八rank RNG/cursor状态和五档验证完成，原自有进程退出，U0/U10000/U45411已独立备份本地并核全SHA。原STOP与NO_TRAINING_RESUME保留。

完整G4（每节点1000图）PSNR：Input18.795938，官方U0 18.222398，U10000 15.937571，U45411 15.894991。U45411的SSIM0.302786、MAE0.107844、LPIPS0.425333。LPIPS改善与PSNR/SSIM下降同时出现，不能写成“四项全退化”，也不能把更清晰或更低LPIPS等同于结构忠实恢复。

这份D0报告中的“Final”固定指U45411，不是后来午夜双路线的最终节点。D0报告曾把U0纳入完整G4最优定义；后续双路线使用已完成全量的正式训练候选池，交接时必须保留各自选模口径，不能把同名best当成同一个选择集合。

## 已完成的独立诊断与forensic审计

入口是`unidb_d0_20261102/forensic_audit_20261102/FINAL_FORENSIC_AUDIT.md`，配套`REPRODUCTION_GUIDE.md`、`FINAL_ACCEPTANCE.json`、`HYPOTHESIS_MATRIX.csv`、`REMOTE_ASSET_INVENTORY.json`。这套已封存审计有前向和只读反传，但新增optimizer更新为0；不应为了交接重跑它。

**已确认的事实：** 所查MatchingLoss活动执行路径与官方代码一致；没有发现GT接反、DDP重复缩放、Adam错步、全局RGB尺度错配或粗大CPU/NPU数值错误。固定GT-forward状态的一步目标改善，最早U500即同时出现强档最终恢复退化。它在所测训练来源和验证来源都存在，不只出现在验证集。

**有干预支持但有范围限制：** 固定50个Val G4来源×4实例的200图，U0→Final的原生SDE PSNR由18.008降到15.761；mean-ODE约18.531→18.525，Input18.561。去随机后主要回到输入邻域，没有实现强恢复。四个固定seed复核的Final−U0均负。只改变一组方差/边界约定使Final从15.761提高到16.892dB，但仍低于Input与同约定U0。由此支持权重与随机路径交互、方差规则有贡献，尚不足以判定唯一根因。

**不能写成确定结论：** L1数学风险不等于一般Gaussian KL平方风险，epsilon_opt一般也不等于forward真实epsilon；这些公式区别不自动证明L1是当前失败主因。有限方向局部扰动没有支持预设的“全面放大”；mean-ODE不是probability-flow ODE；高频增加不是正确纹理。SR域迁移、一步与多步状态、几何逆映射、优化和强度混合之间的因果份额没有通过受控新训练全部分离。

大数组（例如96份完整梯度、64条无损轨迹、true flow、原始诊断预测）主要保留远端，原库存有路径/尺寸/SHA。本地主要为报告、源码、CSV/JSON和图；清单存在不等于大数组已下载。环境实际import路径来自多个site-packages，并非完整锁定wheel的可重建镜像，这一可复现性局限需交给接手者。

## 3＋5双路线与最终实验验收

原路线从D0 U45411完整状态继续；改进路线从官方SR U0重开，并非从退化后的D0优化器继续。随后两次延期均在独立root完整验收接续，旧源和历史结果不覆盖。

|阶段|原路线起止U|改进路线起止U|各阶段角色|
|---|---:|---:|---|
|until_noon|45411→55760|0→20000|原3卡保持L1/LR1e-5；改5卡epsilon_opt MSE，200更新warmup到2e-6，墙钟cosine降到2e-7、clip1|
|until18|55760→61764|20000→30000|全部状态恢复；改进固定既有floor2e-7，不重warmup|
|until00|61764→67180|30000→40848|全部状态及祖先调度保持；午夜安全停训，终评、CPU验收、退出和实际完整权重备份均完成|

两线B96/FP32、原生网络和桥、等比例G1–G4保持。原3物理rank保留八逻辑rank RNG语义，改进保留五物理rank；同3/5真实恢复+重启工程门通过后丢弃门内更新，从验收父正式接续。两路线起点、loss、LR与卡数均不同，是综合路线尝试，不能作单一loss或LR的因果消融。

午夜最终更新的8个rank均在1793721600之前开始，只完成在途更新后以`reason=deadline`结束。原U67180/cursor6449280，改U40848/cursor3921408。原路线累计6449280呈现、相当于40.308轮160K组合，U45411之后新增21769更新/2089824呈现/13.0614轮；改进累计40848更新/3921408呈现/24.5088轮。午夜延长段分别5416更新/519936呈现与10848更新/1041408呈现。这里是样本重复呈现的等效轮数，不表示独立内容数量，也不是收敛结论。

### 实际最终角色与全量质量

原路线best和latest都为U67180，SHA `0381ae3435d2939a2b8b4fbc215c3f78ac6c3775f254bead31dee3e9cdb78e98`。改进latest为U40848，SHA `11a19694f8fc2f4f343e97fab6d3a326759253dbb8eb6b68d2275e144e2cbfc9`；eligible full-best保留noon U10000，SHA `728c30638ee41259c6302b53979f23e115e618c08dca7c2156db8f15e1105a75`。四逻辑角色只对应三个unique完整checkpoint和三个unique全量验证，不能当四份独立模型证据或重复计算备份/CPU成本。

best按同一条3＋5路线全部已完成5000 full候选的G4 PSNR、LPIPS等既定规则选择，官方U0和父D0只作外部基线；没有为monitor U3000增加新full候选。改U3000的G4 monitor100=18.615861是小监控最优，未full确认，不能替换U10000或当泛化成功。

|同G4 Val1000 EMA|PSNR↑|SSIM↑|MAE↓|LPIPS↓|
|---|---:|---:|---:|---:|
|Input|18.795937524|0.451751154|0.080178294|0.571628582|
|官方SR U0|18.222398420|0.361802154|0.085598340|0.636237582|
|父D0 Final U45411|15.894990900|0.302786184|0.107843530|0.425332534|
|原路线best=latest U67180|16.211782897|0.315317397|0.103123695|0.411530598|
|改进full-best U10000|14.708871735|0.126323485|0.134534767|0.771946728|
|改进latest U40848|13.761129645|0.106127686|0.151208202|0.745103065|

完整五档PSNR如下；同一节点所有SSIM/MAE/LPIPS均保留于`unidb_dual_20261104_until00/report/FULL_QUALITY_COMPARISON.csv`，没有分别按各指标择优。

|G 每档n1000|Input|官方SR U0|父D0 U45411|原best/latest U67180|改best U10000|改latest U40848|
|---:|---:|---:|---:|---:|---:|---:|
|0.5|25.170834|16.262089|20.692201|21.750354|17.332066|16.550630|
|1|22.241768|16.155999|19.535007|20.255327|16.866701|16.079718|
|2|20.306366|17.111245|17.917292|18.389684|16.070529|15.088655|
|3|19.471852|18.266974|16.868069|17.442456|15.265373|14.173301|
|4|18.795938|18.222398|15.894991|16.211783|14.708872|13.761130|

原U67180相对父D0 U45411的G4 PSNR改善0.316791998dB，但仍比Input低2.584154627dB、比官方SR U0低2.010615522dB。原LPIPS 0.411530598低于matched Input 0.571628582，同时PSNR/SSIM/MAE较Input差；不能称四项全退化，也不能把较低LPIPS或更锐利视为正确几何恢复。改进两选中节点在强G4的PSNR/SSIM/MAE/LPIPS均差于Input，U40848的PSNR又低于其full-best U10000。两条综合配方未建立强湍流恢复优势或外部真实域泛化成功。

EMA full每G1000=250独立来源×4ξ，五G总5000；RAW仅G1/G4各monitor100，另列于`report/VALIDATION_HISTORY.csv`（筛选weights=RAW/monitor）和最终报告。其人口、RAW/EMA和指标不能混用。科学结论不由工程PASS、loss下降或完成更多更新代替。

### 完整性 CPU成本 退出与备份

`evidence/STATE_AUDIT_{original,improved}_{FINAL,BEST}.json`完成四角色RAW/Adam/EMA/8逻辑或5物理RNG/cursor/日程/source检查；实际checkpoint SHA→EMA全参数digest→对应full SUMMARY均逐项绑定。`evidence/FINAL_OUTPUTS_CPU_AUDIT.json`对三个unique完整5000候选的全部预测/Input/clean SHA及PSNR/SSIM/MAE/全部LPIPS复核PASS，共15000角色去重记录。两个新终点实际CPU metric前向2500 batches，U10000完整已审缓存复用1250 batches；缓存没有被重复计成新调用，新增恢复模型/optimizer/NPU调用均0。CPU finalizer于unix 1793727575.7159739退出。

全部自有进程终验采用历史来源并集534身份：原生本次审计468，加上01:12预检中额外66个老loader/resource_tracker精确PID/start/PGID/SID身份；差集已逐项核相等，补查全部gone，无same/zombie/未解释存活或信号。37类来源覆盖、历史PGID/SID、rank、loader/resource_tracker、controller、CPU与本机8个指定传输PID均核退出；外部任务未动，card1 driver context不当OS存活断言。最终补查unix 1793729426.1490653，不能把旧534预检中的活跃sender混入最终结论。

实际本机unique备份如下，均已根代理全文件字节SHA核过，当前size/mtime及最终角色重新绑定；本次记录刷新不重新读2.19GB载荷。旧U10000完整备份合法复用，不重复下载。

- original best/latest U67180：`/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/checkpoint_backup_staging/original/U00067180.pt`，2194706863 bytes，SHA `0381ae3435d2939a2b8b4fbc215c3f78ac6c3775f254bead31dee3e9cdb78e98`。
- improved latest U40848：`/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/checkpoint_backup_staging/improved/U00040848.pt`，2194679415 bytes，SHA `11a19694f8fc2f4f343e97fab6d3a326759253dbb8eb6b68d2275e144e2cbfc9`。
- improved best U10000：`/Users/lucas/Desktop/turbulence IR/unidb_dual_20261103_until_noon/delivery/checkpoints/improved/U00010000.pt`，2194675639 bytes，SHA `728c30638ee41259c6302b53979f23e115e618c08dca7c2156db8f15e1105a75`。

全部unique字节合计6584061917。旧noon其他候选仍按原收据保留，既非额外最新权重，也不因已备份而获得删除许可。原报告/CPU收尾器里保留的VISUAL_LOCAL_PENDING是它们当时的历史状态，后续`delivery/experiment_full_delivery_receipt.json`、`all_owned_process_exit_audit.json`和`unique_checkpoint_backup_audit.json`合并关闭实际最终实验接收；不覆盖这些原件。

总图`report/VISUAL_QA.json`及原路线`delivery/original_route/VISUAL_QA.json`已root逐图绑定新SHA，分别三张总曲线/固定图与七张原路线全程曲线/G1–G4图。只引用已验字节，不重复绘图或运行恢复模型。19点PPT及21点状态图仍保留为历史快照，不能代替这些最终附件。

## 接手入口与科学局限

本地/远端以相同root名定位：`unidb_dual_20261103_until_noon`、`unidb_dual_20261103_until18`、`unidb_dual_20261104_until00`；远端host `liteserver-de36`、root前缀`/data2/yanglin/`。先读当前root `report/FINAL_REPORT_ZH.md`、`FULL_QUALITY_COMPARISON.csv`、`CPU_POSTPROCESS_WORK.csv`、四角色STATE_AUDIT、最终保存预测CPU审计以及上述三个实际交付adapter。数据、权重、状态与脚本入口由ASSET/WEIGHT/CODE_ENV/EVIDENCE注册表追溯；旧CURRENT_EXECUTION早期时间戳按历史快照理解，不复活旧PID、STOP或controller。

本阶段原生完整性、评价、退出与完整权重接收已经闭合；全项目主文档、可编辑DOCX、文档逐页QA、便携包和聊天交付另有流程，不能由实验终验自动宣布完成。`UNIDB_PENDING_FINAL.json`原件及其准备表收据保留为旧捕获，在当前实验表改归HISTORICAL_SNAPSHOT，不冒充最新状态。

科学局限仍包括模拟器和源域范围、训练一步目标与free-running状态差异、SR迁移/优化/强度混合及随机路径的未分离因果份额。当前独立全量Val不等于锁定Test或外部真实湍流部署成功。封存forensic的支持与反证不因后续综合配方失败而改写；没有证据把唯一根因定位为warmup、L1、Adam或backbone。环境多site-packages继承、未完全锁定wheel/OS/ISA/CANN，及诊断大数组主要留远端等复现边界继续保留。

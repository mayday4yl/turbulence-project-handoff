# V4 train500 四组训练计划：匹配更新次数与曝光

修订日期：2026-09-30。状态：**待审阅的训练计划；只修改计划，没有开训、远端操作或新的 GPU 验收。**

计划 ID：`v4_train500_four_arm_matched_v2`；补充修订：`reporting_eval_addendum_v1`。方法沿用本目录 `METHOD_AND_DATA_SPEC.json` 与 `LINEAR_METHOD_AND_DATA_DESIGN.md`；本文件补齐预算、优化、调度和评价规则，不改变原方法与数据定义。

本次只补best-val能力表、量化共同续训建议trigger、LR来源、同卡evaluation mutex、按需输出留存及固定图表交付。四组方法、数据范围、初始化、B32、10k共同首段和原LR数值均保持；提交用户确认后才可进入执行阶段。

## 1. 主问题与公平比较的口径

用户于2026-09-30明确：本轮首先回答 D0/R0/R08/L08 在训练充分后谁更好，尤其比较 R08 与 L08 的 physical intermediate state 和 linear interpolation。主方法消融应匹配 optimizer updates / start-record exposures，计算量另外报告。

会议中“训练量看整个计算量”不是“四个方法必须等计算小时”的规定。我此前自行引入的每组8个reference A100 hours，改变了本轮的主要研究问题；本版撤销这项预算及由单步耗时反推各组不同更新数的规则。原草案只保留于 [已撤销版本](<../../../NOT_INCLUDED.md#ref-d8ec1d0434836a2d063e>)，不得用于启动任务。

**主控制：共同初始化、train500、effective batch=32、相同训练start顺序、相同optimizer更新次数、同一update处相同LR/EMA规则、相同loss与评价协议。R08/L08同一曝光使用配对η。** 主表在相同更新节点比较四组；不能把较快方法多做的更新混入主方法结论。

共同第一阶段建议为 **10,000 updates / 320,000 start-record exposures / 40 epoch**。一个epoch=8,000 starts=250updates，每个start在首段恰好出现40次。四组均从同一未训练初始化开始。这是第一阶段的预算建议与检查点，不是老师规定的数值，也不预先保证训练充分；充分性与共同续训边界见第4、9节。

计算量单列：每个start在四档G等权下，D0平均1次网络预测/1次训练render，R0/R08平均2.5次预测/4次render，L08平均2.5次预测/2.5次render。它们是调用计数，不等于完整FLOPs。报告实际训练耗时、评价耗时、峰值显存与推理延迟，不能用这些成本改变组内训练预算。

四组共同曝光并不消除方法自身的多阶段监督、状态分布、ξ反馈和计算量差异；它把可控制的训练更新过程对齐。首要比较 R08−L08，其次 R08−R0、R0−D0。

## 2. 数据、网络与四组方法

| split | clean sources | realizations/source | G | starts | 用途 |
|---|---:|---:|---|---:|---|
| train500 | 500 | 4 | 1/2/3/4 | 8,000 | 四组共同训练 |
| val100 | 100 | 2 | 1/2/3/4 | 800 | 记录泛化与配对比较 |
| test200 | 200 | 2 | 1/2/3/4 | 1,600 | 本轮保留，不使用 |

数据 ID：`places_p2s_a100_20260928_g4321_s500_v100_t200_v1`。保持现有 256×256 FP32 RGB、固定预处理、原生 ξ 与物理参数，不生成新数据、不另增随机裁剪或增强。8,000 是 start 数，不是不同 clean 图数。

| ID | 网络入口 / 下一状态 | T0 | 预测次数 |
|---|---|---:|---:|
| D0 | `Concat(Y,Y), [G,0]`；无转移 | 0 | 1 |
| R0 | 当前状态与固定 Y；下一状态 `F(C_hat,xi_hat,g_next)` | 0 | G |
| R08 | 同上，按既定公式构造 Z | 0.8 | G |
| L08 | 下一状态 `(1-a_next)*C_hat + a_next*Y` | 0.8 | G |

L08 中 `a_k=(K-k)/(K-1)`，`t_k=0.8*a_k`，K=G；K=1 单独处理。初始状态 Y，每级从当前状态构造 `Z=(1-t)*X_tilde+t*mu+s*t*eta`。G=4 时线性状态为 Y → 1/3 C1+2/3 Y → 2/3 C2+1/3 Y → C3，然后第4次网络预测输出 C4。没有真实 clean 插值、弱退化真值替换、末级额外转移或 g=0。

四组均使用现有六通道 V4 小型 RGB 主体、`[g,t]` 条件 MLP、clean Sigmoid 和两块 native raw ξ。D0 是完整 D0，不是 clean-only；PiRN backbone 候选不进入本轮。

共同损失：每个 start 先对其 K 次预测平均 `L_C + 0.2 L_R`，再对 effective batch 平均。每级 L_R 都回投到原始 G、Y，包括末级。D0 的 K=1。训练无 refinement，预测递归保留完整 BPTT；simulator 参数冻结、输入导数保留。L08 的 ξ 仍受本级 L_R 监督，但不参与下一状态。

R08 与 L08 的主要区别是状态生成及 ξ 的跨阶段反馈，不是官方 E-Bridge 完整训练/求解器复现。四组也不是完整的“路径类型×噪声”二因素设计。

## 3. 共同初始化、统计与随机流

四组从同一份未训练网络参数独立开始；不续训旧 last_32k、tiny-overfit 或 PiRN 权重。优先核验原共同初始化（历史 SHA256 `2e6a29eb895de090bbaeaf1cfdd1aedd5550703fa1bb9bb80e099ca2fd920fc2`），仅取已核验的网络初始化参数；μ/s 用本轮 train500 artifact 替换。原初始化不可用时停止该步骤并记录缺失，不静默以另一份权重代替。

对 train500 的 8,000 个唯一 `(trajectory_id,g)` 原始物理状态重算一套 μ/s：每档 2,000 状态，四档等权，RGB per-channel，float64 Welford/Chan，总体方差 correction=0，包含档间均值差，std 下限 1e-6。沿用登记的完整 256×256 固定视图，不统计 clean、val/test、线性状态或预测 rollout。保存清单/代码/预处理/数组哈希、每档像素数、完整 μ/variance/std 与 artifact hash；checkpoint 中为非训练 buffer。

训练有效 batch 固定 **32**，单进程单卡，默认 **microbatch=2、accumulation=16**。若共卡最坏路径验收失败，可在正式冻结前四组统一降为 microbatch=1、accumulation=32；不改变有效 batch，不增加卡数，不做质量或 LR 搜索。

为使四组每次更新的 G 分布一致，每个 effective batch 含 G=1/2/3/4 各8个 starts。每 epoch 对四个 G 列表各自确定性 shuffle，依次每档取8条，再打乱32条的批内顺序；每 epoch 全部8,000 starts各出现一次，无 replacement/padding/drop。四组共享完全相同的有序流前缀与 microbatch 划分。每 epoch=250 updates；第一阶段四组统一完成10,000个32样本update，不补齐、不丢样本。

随机流固定 master seed=20260929，初始化使用其独立历史参数 artifact。数据顺序、η、诊断和可视化选择分别使用独立 namespace。η 的 seed 由 `SHA256(namespace|epoch|start_id)` 前8字节 little-endian、模 `2**63-1` 得到；按样本生成 FP32 `torch.randn((3,256,256), generator=独立CUDA generator)`，R08/L08 同一曝光使用相同 η，路径内共用。评估用 `eval|20260929|start_id`，不随 checkpoint 改变，不推进训练 RNG。将完整规则、Torch/CUDA 版本和实际样本顺序保存；不以共享进程 RNG 或进程快慢决定 η。

## 4. 共同首段预算、充分性检查与成本记录

| 项目 | D0 | R0 | R08 | L08 |
|---|---:|---:|---:|---:|
| 共同首段optimizer updates | 10,000 | 10,000 | 10,000 | 10,000 |
| start-record exposures | 320,000 | 320,000 | 320,000 | 320,000 |
| epoch / 每start出现次数 | 40 | 40 | 40 | 40 |
| effective batch | 32 | 32 | 32 | 32 |
| 每epoch各G曝光数 | 2,000 | 2,000 | 2,000 | 2,000 |

同updates加同batch使曝光一致；还要有相同实际start序列，不能只核对计数。网络预测次数和simulator调用次数因方法不同而不同，分别记录。

**共同续训建议使用开训前冻结的量化trigger，不能在看到正式曲线后改阈值。** 四组全部完成U=10,000后，对每组取完整val800、RAW/refinement-OFF、固定η下的逐epoch均值PSNR：`p_e`，epoch e对应U=250e。只用实际完整评价结果，不平滑、不插值补缺。

- 前窗口 A = mean(p_21,...,p_30)，对应U=5,250,...,7,500。
- 后窗口 B = mean(p_31,...,p_40)，对应U=7,750,...,10,000。
- 末段半窗差 D = mean(p_36,...,p_40) − mean(p_31,...,p_35)。
- β 为 p_31,...,p_40 对epoch的普通最小二乘斜率，单位dB/epoch。
- 预定义“末段持续下降”为：`β < 0 且 D <= -0.02 dB`。0.02为本版预登记的趋势阈值，允许微小回摆，不代表统计显著性。
- 该组trigger：`(B-A >= 0.05 dB) 且 非(末段持续下降)`。使用未舍入指标计算；原始窗口、差值、斜率与布尔值全部保存。

**四组中任一组触发，整体标记 `TRAINING_NOT_SATURATED`，只提出四组共同+5,000updates的建议，不自动执行。** 若所有组均未触发，标记 `EXTENSION_TRIGGER_NOT_MET`，不能改写为“已经证明收敛/达到容量上限”。四组或窗口指标缺失/非有限，标记 `REVIEW_INCOMPLETE`，不得缩小窗口、丢掉失败组或在数据不完整时给出整体判定。该trigger用于提出续训建议，不是训练提前停止器；四组仍全部完成共同10k。

阈值现定0.05dB与0.02dB，不在本轮搜索。正式`FROZEN.json`前，用独立临时冻结模型、完整val800、固定η/分组/dtype，在同卡mutex下对每组重复3次RAW/refinement-OFF评价，记录overall PSNR最大−最小、逐start差异。overall数值波动须<=0.005dB才接受上述阈值；这是精度检查，不是效果Pilot，不增加optimizer step。本轮仅修改计划，尚未测波动；若不通过，保留失败证据、先解决评价问题或在正式训练前另行修订并冻结阈值，不能在看完正式结果后调整。

在共同U=5,000/7,500/10,000节点仍保留完整train8000评价；train loss、末级MSE与三指标、val曲线及固定图片作为解释材料，不另建事后主观trigger。单看train变好不能证明泛化变好；本规则未触发也不能证明优化已充分。

若trigger触发，建议共同追加5,000updates（160,000 exposures / 20epoch），LR固定1e-5；各组恢复自己的AdamW/EMA/stream/RNG，不重置训练。**该追加未计入首段预算，没有自动执行授权，必须待用户另行确认。** 不允许只给某组追训，或把不同长度的best checkpoint混入matched-budget主表。

成本测量只用于资源验收、ETA与附表：不再进行用于倒算N_i的成本校准。新配置每组每个microbatch候选最多2次临时optimizer-step（最多两个候选、四组累计16次），用于实际更新与checkpoint正确性；全部丢弃并从共同初始化开始正式首段。吞吐预演可做有界forward/backward而不追加optimizer-step，正式训练获准后从已计入预算的更新日志估算剩余时间。

同卡共享时的进程墙钟含调度竞争，不能把两进程各自墙钟简单相加称为真实GPU-active hours。分别记录训练同步计时、GPU事件计时、进程总墙钟、每卡总占用时段及评估/保存时间，保留统计口径；有完整算子计数才另报FLOPs。开训前不会据测速给D0更多update，或者给R08/L08减少update。

## 5. 优化器与共同学习率

沿用AdamW：betas=(0.9,0.999)、eps=1e-8；weight_decay=1e-4，bias/normalization不衰减。每个effective batch做一次optimizer step；microbatch loss按样本数/32累积，之后统一global norm clip=1.0。

四组统一以首段N=10,000确定LR，第u次update前设置p=u/10,000：

- u=1,...,500：`LR=1e-4*u/500`；
- u=501,...,10,000：`LR=1e-5+0.5*(1e-4-1e-5)*(1+cos(pi*(u-500)/9500))`。

即共同500updates warmup（16,000 exposures / 2epoch），之后共同cosine至1e-5；同一update四组LR完全一致。EMA每update decay=0.996，buffer直接复制；FP32，autocast与TF32关闭。不得按各组运行小时、成本进度或val曲线分别改变LR。

LR来源说明：500/10,000=5%，与旧参考Pilot的50/1,000 warmup比例一致；仍是cosine从1e-4到1e-5。**本轮final LR=1e-5作为更长training horizon独立登记的新protocol choice，不是从用户所述旧Pilot的1e-6直接继承，也不在本轮搜索。** 四组LR-vs-update完全相同，公式与matched_v2补充前一致。

历史取证边界：已核对 [旧32k三组计划](<../../../NOT_INCLUDED.md#ref-8b686c093b7d326bfcf3>) 第5节，其实际参考比例为50/1,000、末值已是1e-5；尚未定位用户所指1e-6 Pilot的原始文件。因此不把“旧Pilot=1e-6”误写成所有历史V4实验的实测事实，也不改历史配置。本轮采用1e-5的决定不依赖该历史文件是否定位。

若另行确定共同续训，保留这段已执行LR历史，不追溯重算cosine分母；续训阶段四组使用同一个新约定，禁止某组重启warmup或重置optimizer。

## 6. 两卡四进程安排

仅 A100 服务器 `10.94.2.1`。拟定 GPU5：R08+D0；GPU6：R0+L08。每个实验单GPU单进程，无DDP，独立目录、optimizer、EMA、日志和状态。物理编号必须在执行时映射到已核验 UUID，不依赖历史编号猜当前资源。

单组正确性检查后，再测四进程并存时的完整 G4 和混合 G 前后向，以及一组完整评估/另一组已暂停但状态仍驻留的峰值。每卡实测占用加安全余量至少 `max(4 GiB,显存总量10%)` 不超过容量；同时验收主存、pinned-memory、磁盘空间。未通过 micro2 才共同改 micro1，之后重新验收并冻结，禁止在正式训练中某组单独降 microbatch。

新总调度器统一持有两卡的项目资源锁，并登记四个合法子进程、配置与 heartbeat；保留对外部占用的检查。不能直接沿用旧 launcher“发现任意 GPU 进程即退出”的逻辑，也不能删除锁来绕过冲突。旧实现与历史实验不改写。

四个训练进程各自完成本组评估，不额外起第五个GPU evaluator；需要顺序切换训练/EMA/metrics驻留，避免同卡出现第三个模型进程。较快的组达到共同首段节点后保存并等待其他组完成，不多做更新；首段结束后不自动增加任何组预算。

**同卡evaluation mutex：GPU5的R08/D0共享一个协调式evaluation lease，GPU6的R0/L08共享另一个；两张卡之间独立。仅锁两个evaluator而让peer继续训练不符合本协议。**

1. job到评价节点后，在完整optimizer→EMA boundary申请lease；协调器通知同卡peer暂停。peer须完成当前全部梯度累积及一次optimizer/EMA更新，释放该步临时计算图、CUDA synchronize，并确认暂停的U/E后，才可授予lease。此后peer不启动下一次forward/backward/update。
2. lease覆盖该节点整套evaluation：train-monitor400、到期的全train8000、完整val800、RAW/EMA、LPIPS、固定图片与metrics/hash提交。评价开始前必须收到所有仍存活peer的暂停确认；评价期间同卡最多一个job评价，另一job的U/E与sampler/RNG保持不变。
3. 两job同时申请按FIFO ticket处理；并列按卡内固定顺序（GPU5 R08→D0，GPU6 R0→L08）。协调器排队，不让双方各持锁再互等；若下一ticket已在等待，优先处理后再恢复训练，避免饥饿。
4. 评价结束并同步GPU、提交输出、恢复RAW训练模式与冻结buffer、清理临时LPIPS/EMA驻留后释放lease。已结束peer经进程/状态核验作为terminal确认；异常失联、超时或评价失败不视为成功暂停/完成，应保留错误并保持安全暂停，恢复时避免重复提交。
5. paused peer的model/optimizer及缓存仍计入显存峰值；暂停不等于显存已释放。不为通过验收自动offload、改变精度/微批、丢optimizer状态或新开GPU evaluator。锁等待时间单列runtime，不能推进LR、训练曝光或η流。

正式验收覆盖同时请求、一个job先结束、评价异常恢复；日志记录request/ticket、pause ACK的U/E、eval start/end、release与异常。须证明同卡评价区间不重叠，peer区间内无训练kernel且U/E不变，无死锁、无重复评价；本轮只定义规则，不声称mutex已实现。

训练数据不读取真值 ξ 作为监督；取消全量无限 pin-memory 缓存，使用有界缓存/只读数组，pinned 缓存上限暂定512 MiB/进程，普通CPU数组缓存上限2 GiB/进程，执行前用实际主存核验。不得通过减小图像/ξ尺寸、截断 rollout、换 simulator 或精度来通过资源门槛。

## 7. 评价频率、曲线与固定图片

**主结果统一 RAW、refinement OFF。** EMA 单列辅助；本轮四组比较不另加 refine-on 阶段。未来若单独做 R0/R08 refine-on，仍须固定逐样本精修，不与当前主表混合；L08没有 ξ 精修的状态入口。

固定 train-monitor400：100个训练source，其中旧Overfit100取20、新增400取80；每source固定1个 realization×4G。按独立 SHA256 选择规则抽取，已根据本地清单生成 [PLANNED_MONITOR_LISTS.json](<../../../NOT_INCLUDED.md#ref-54977ad8fa94cd5d027b>)，执行时核对远端身份再冻结；不按图像好坏筛选。这是训练集的一部分，不能标成全train8000。

验证曲线每次均使用**完整 val800**，没有 val64/val800 混写。train与val源图不重叠。曲线节点定义：

| 节点 | RAW | EMA | 内容 |
|---|---|---|---|
| 初始化与每个epoch（每250updates） | train-monitor400 + val800 | 每5epoch（每1,250updates） | 路径loss三分量、末级clean MSE、PSNR/SSIM/LPIPS |
| U=0 / 5,000 / 7,500 / 10,000 | 全train8000 + val800 | 同样完整评价 | 共同节点全量指标与output hash；FP32 image tensor默认仅保存固定40图 |

各组用完全相同的update节点，不再设按不同N_i计算的10%成本节点。主比较覆盖完整学习曲线及共同首段U=10,000；若充分性尚未建立，终点主表必须标记为第一阶段结果。未来共同续训的节点另行冻结。

同一个 checkpoint、固定评估 η 上，train-monitor 与 val 均计算：路径平均 L_C、0.2L_R、total；另存末级 C_hat_K 的 clean MSE。主性能PSNR、SSIM、LPIPS对末级输出计算，不把路径L_C转成末级PSNR；另从同一次rollout捕获first clean计算progression所需PSNR，不另跑不同路径。逐update在线训练loss另画图，因为其η和参数持续变化，不能冒充同checkpoint train评估loss。

曲线主横轴为optimizer updates / start-record exposures，附epoch坐标；计算成本单独做表或辅助成本曲线。所有点来自真实保存节点，不平滑、不插值制造未测指标。RAW与EMA分图，train-monitor/全train/val标注清楚；完整train仅在四个登记节点有指标，不伪造成逐epoch全train曲线。

固定图像：train5sources（旧100取1、新400取4）与val5sources，每source固定同一ξ、G=1/2/3/4，合计40 starts，预先登记。保存各评价节点这些starts的FP32输出；最终同时生成固定竖版与横版对照，RAW和EMA分别成套；标题明确split、source序号、G、组、RAW/EMA、update、exposure及实际耗时。最终并列 `Clean / Input Y / D0 / R0 / R08 / L08`。`Places365_val_...`只是原始文件名，不能据此把训练source标成验证集。

## 8. 指标实现与配对统计

沿用 `v4_metrics_supplement_20260929/scripts/metrics.py` 的已登记定义，执行时核对实际import、源码及权重hash，不能只核对包名：

- PSNR：原始浮点RGB、data_range=1；单图全部RGB像素MSE（FP32 CUDA），逐图 `-10 log10(MSE)`，再对图平均，不对平均MSE取PSNR。
- SSIM：scikit-image 0.25.2，float64 CPU，RGB channel_axis=2，gaussian_weights=True，win_size=11，sigma=1.5，use_sample_covariance=False，K1=0.01、K2=0.03，data_range=1。
- LPIPS：lpips 0.1.4，AlexNet，version0.1、pretrained、eval，FP32 CUDA、metric batch1，`normalize=True`只执行一次官方 `[0,1]→[-1,1]` 映射。
- 无额外clamp、resize、量化或per-image标准化。展示PNG不参与指标。网络评价batch固定2、同一排序与分组、单进程 world_size=1，尾批禁止补齐或丢弃；当前所有完整评价总体均可整除。

Input Y 在同样总体上计算三项指标作为基线。保存逐start、逐source、各G与总体结果；原始记录至少包含source/trajectory/start、G、checkpoint hash、权重类型、协议hash、η seed、三项指标、loss与输出hash。

首段主表仍严格使用四组相同U=10,000的RAW/refinement-OFF checkpoint，主结论优先看matched-update结果。所有配对也固定该共同节点，不把各组best-val节点代入主消融。val用于开发及checkpoint选择，不称为独立test；test200本轮完全不使用。

### 8.1 best-validation capability结果（辅助）

候选固定为共同41个RAW节点：`U in {0,250,500,...,10000}`，对应epoch0..40；每个节点使用完整val800、相同η与指标实现。它是在已预注册评估节点中选best，不声称搜索过区间内所有10,001次权重。

对每组先求 `m=max_U PSNR_val_raw(U)`，再从 `PSNR>=m-1e-6 dB` 的候选中取最早update；必须对完整41节点一次性选择，不能在线容差比较造成非传递偏差。候选缺失/非有限时标记选择结果不完整，不静默缩小候选集或换成EMA。checkpoint hash与原始未舍入得分一起保存。

`best_validation_capability_table`列：method、best_update、best_epoch=U/250、PSNR、SSIM、LPIPS、ΔPSNR_vs_Input；三项指标取同一个由RAW-val-PSNR选定的checkpoint，不能各自挑最高。输入基线是相同完整val800上的Input Y均值。完整候选RAW权重按原每epoch保存要求保留。

该表展示同一training protocol下观测到的最佳验证泛化表现；它经过val选择，有选择偏差，不是未见test的泛化估计，也不替代U=10k的公平主表。`main_performance_table`附B区引用相同数据，同时单独导出capability表，避免表格混读。

### 8.2 配对、source win-rate与区间

`paired_ablation_table`必须有五项：`R0-D0`、`R08-R0`、`R08-D0`、`L08-D0`、`R08-L08`。每项在U=10k、RAW、val800上逐start计算ΔPSNR，再对每source的2ξ×4G共8starts求均值d_s，最后对100sources平均。当前等量采样下它与800starts等权均值相同。

source win-rate=`count(d_s>1e-6 dB)/100`；`abs(d_s)<=1e-6`记tie，tie保留在分母中，另报win/tie/loss计数。不得把800个相关starts当800个独立source。

95% CI为source-cluster bootstrap：seed=20260929、NumPy Generator(PCG64)、10,000次重采样，每次有放回抽100个source并保留该source全部ξ/G和跨方法配对；所有五项复用同一bootstrap index matrix并保存hash。取重采样均值分布的2.5%与97.5% percentile，`method=linear`。区间反映固定checkpoint的source变异，不包含训练seed或best-val选择的不确定性。

### 8.3 first→final progression

对R0/R08/L08，在同一次RAW/refinement-OFF rollout里保存第1次网络clean输出的PSNR、末次clean输出的PSNR及两者output hash；保持同一Y/G/η/checkpoint/网络分组，不用另一个D0模型充当first。

`delta_prog(start)=PSNR(C_final,C_GT)-PSNR(C_first,C_GT)`，先逐start计算差再聚合。RAW的每epochtrain-monitor400/val800评价都记录该差；主progression图表固定U=10,000、完整val800中的G2/3/4。每个G含100sources×2ξ=200starts，先在source内平均两个ξ，再平均100sources；Overall等权平均G2/G3/G4，仅含600starts。G1无多步过程，不纳入Overall，可单列N/A。额外训练侧progression结果单列，不能混入验证主图。

主表每行还列first PSNR、final PSNR、ΔPSNR及source/start数，明确正值代表多步改善、负值代表损害第一次预测。保存未舍入逐start差，不能用显示后四舍五入值相减。

### 8.4 固定三张主图与三张主表

以下主图使用RAW/refinement-OFF，并导出PNG+PDF和绘图数据CSV/JSON；数据点来自实际评价，不平滑、不补造。EMA另存辅助图。

| 主图文件名 | 固定内容 |
|---|---|
| `train_val_psnr_curve` | x=optimizer updates，附epoch=U/250、exposures=32U；四方法颜色固定；train-monitor400与val800分面/明确区分；各自Input-Y PSNR虚线baseline |
| `train_val_loss_curve` | 四方法、train-monitor400/val800分清；三个loss分量L_total、路径L_C、0.2L_R；使用相同checkpoint/固定η的评价loss，在线训练loss另存 |
| `progression_by_G` | U=10k完整验证集，R0/R08/L08×G2/G3/G4的first→final ΔPSNR，标0线和改善/损害方向，不用best-val替换 |

SSIM与LPIPS完整train-monitor/val曲线和数据仍必须保存，RAW/EMA分清，不挤入上述三张主汇报图。

| 主表文件名 | 固定内容 |
|---|---|
| `main_performance_table` | A区：D0/R0/R08/L08在U=10k的val800 PSNR/SSIM/LPIPS/ΔPSNR_vs_Input；B区：各组best-val update/epoch及同checkpoint指标，明确辅助；另外导出同内容`best_validation_capability_table` |
| `paired_ablation_table` | 第8.2节五项对照，mean paired ΔPSNR、source win-rate、source-cluster bootstrap 95%CI，以及样本数/协议身份 |
| `progression_by_G_table` | U=10k，R0/R08/L08×G2/G3/G4/Overall，first/final PSNR和ΔPSNR；Overall仅G2–4 |

三张主表与capability子表导出CSV+JSON+可读Markdown，保留checkpoint/protocol/hash关联。最终qualitative至少含预注册train5+val5的固定40starts，每个split各自导出竖版和横版：`Clean | Input Y | D0 | R0 | R08 | L08`，列顺序固定、四组共同U=10k，RAW与EMA分别成套，注明source、G、split、update、权重类型。不得重新按结果选图；best-val若另做图片必须单列并逐列标实际update，不替代该共同终点对照。

## 9. Checkpoint、异常恢复与停止

每25updates原子更新rolling恢复checkpoint，保留当前及上一份；每5epoch（1,250updates）保留不可变完整checkpoint；每epoch保存可再评价的RAW/EMA权重。最终保存独立终点checkpoint。包含network、EMA、AdamW、scheduler进度、全RNG、stream位置、E/u、分项计时、μ/s artifact/hash与完整身份锁。

恢复前核对协议、代码、数据与初始化链，先查实际进程/tmux/status；禁止重复启动。仅从同组最后完整optimizer边界继续，不重复计已提交exposure；中断前后attempt ID及时间独立记录。不以旧实验checkpoint替代本轮缺失恢复点。

首段停止规则：四组统一达到U=10,000后完成首段评价并进入充分性审阅；不按不同耗时分配不同更新数，不分别以各自plateau早停。NaN/Inf、损坏checkpoint、资源冲突或实质方法偏离属于异常，保留失败证据并暂停受影响组，不能跳过坏batch伪装完成。工程修复需记录；改变正式数值/方法协议时新建版本，不能混合旧结果。

首段完成不等于充分训练后的比较已经完成；充分性结论与共同续训建议须显式记录。不自动续训到任意epoch、不搜T0/λ/LR、不换模型。依据第4节冻结trigger输出TRAINING_NOT_SATURATED / EXTENSION_TRIGGER_NOT_MET / REVIEW_INCOMPLETE，不能据未触发就断言容量极限或路径无效。

**输出留存默认采用流式指标与hash，不强制落盘全部image tensor。** 所有实际评价记录必须保存逐start/逐source/per-G/overall指标、checkpoint、output hash、固定qualitative样本FP32 output、manifest/protocol/logs；固定40starts沿用原逐评价节点保存，最终RAW/EMA全部保留。网络/指标计算仍使用原始FP32输出，禁止从展示PNG重算指标。

对非固定图，按batch计算完整指标与output hash后即可丢弃tensor，不先写55GB再删除。hash统一为原始输出contiguous、little-endian FP32、CHW、C-order字节的SHA256；另存shape/dtype、source/trajectory/start/G、first/final stage、U、RAW/EMA、checkpoint hash、η seed、协议与环境身份。hash身份指向当次实际评价输出，不是“重放必定bitwise相同”的保证。

需要图像时从保留的frozen checkpoint、数据清单、η/分组和数值协议重放。记录新输出hash及相对原记录的PSNR/SSIM/LPIPS差异；hash不同不能静默覆盖原评价。固定样本因保留FP32原图，还可直接数值对比；其余只有原hash/指标时不得声称比较过逐像素差异。重放仍遵守同卡evaluation mutex，不额外训练。

全train8000+val800、四组RAW+EMA终点image tensor约55.4GB只是可选容量估计，`required=false/default_enabled=false`。只有明确核验磁盘余量与I/O成本且决定启用时才另记配置；审计通过不以这些全量tensor常驻为前提，不删除历史结果腾空间。

首段完成核验四组U=10k、41个RAW评估节点、全部指标与hash、best-val选择、量化trigger、三图三表及固定双排版图；确认进程退出，返回第一阶段报告和共同续训建议状态后停止，不自动执行+5k。

## 10. 开训前必须实际完成的工作

1. 在新的独立实现目录复用原V4，参数化旧 hardcoded `overfit100/1600/32000/DDP2`；新增L08转移、单GPU四进程调度与同卡evaluation mutex、train500 stats、配对η、共同更新预算、路径loss/first-final evaluator及报告入口。历史文件不直接改写。
2. 只读核验服务器身份、GPU5/6实际资源、源码import与物理资产哈希、train/val当前文件。已有本地审计不替代当前远端可读性；test不加载评价。
3. 真实网络+真实simulator FP32验收 G1/2/3/4、混合G、路径reduction与噪声公式；G1只一次预测，末级不多走一步。训练与eval固定分组分别核验。
4. 末级clean-only诊断loss：R0/R08早期clean与两块ξ必须有真实跨级梯度；L08早期clean有跨级梯度，早期ξ没有这条梯度是预期。ξ本级L_R梯度另验。保存None/finite/范数与中间detach负对照；不凭共享参数或局部loss证明BPTT。
5. L08改变GT clean不得改变预测状态；改变ξ不得改变线性下一状态。对G2/3/4验证转移clean导数系数 `[1]`、`[3/10,1]`、`[7/45,22/45,1]`。统计人工数组/真实artifact/buffer往返、optimizer/EMA/checkpoint恢复及训练RNG不受eval影响均实际测试。
6. 验收batch32的累积loss/梯度缩放、stream每epoch不重不漏且每batch各G=8；共卡显存与指标fixture通过后写入新的 `FROZEN.json`、`BUDGET_LOCK.json`、环境/import/清单hash；锁定各组同U、同E和同一update处同LR，成本只作记录；同时冻结best-val候选与选择规则、续训trigger及其重复评价波动报告、mutex协议、图表schema与输出留存策略。

当前方法和数据范围已设计；L08实现、μ/s新artifact、单GPU适配与共卡验收仍未完成；10,000updates为本版共同首段建议，充分性尚未建立。它们是执行前工作，不能标为已有通过。本轮没有真正资源阻塞的现场证据，因为没有连接远端。

## 11. 证据与交付

- [方法与数据设计](<../../../NOT_INCLUDED.md#ref-4e5d5d9883c96b5c5fa9>)、[方法机器定义](<../../../NOT_INCLUDED.md#ref-087d4c8a3bb6a32d0202>)、[数据范围审计](<../../../NOT_INCLUDED.md#ref-01acfb07f06cbe8fbd78>)。
- [旧实际配置](<../../../NOT_INCLUDED.md#ref-09d437db3880c2e8285e>)：B32、两卡DDP、各32k/1,000updates，供复用核对，不是新预算授权。
- [旧训练实现](<../../../NOT_INCLUDED.md#ref-5d4f86e102815d091351>)、[旧启动器](<../../../NOT_INCLUDED.md#ref-3d64304afda491720619>)、[固定指标](<../../../NOT_INCLUDED.md#ref-cdd659c6c31e820d6d65>)。
- [会议转录](<../../../NOT_INCLUDED.md#ref-403fd8e0ca4b86c13a7b>)：09:40/10:03/11:44。
- [机器可读计划](<../../../NOT_INCLUDED.md#ref-3c99463f9614814855f2>)、[本地计划检查](<../../../NOT_INCLUDED.md#ref-b92300224b9dc51f39a4>)。后者只证明预算公式与文档引用检查，不证明GPU实现或训练已通过。

目标新实现根目录：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train500_matched_v2`；目标结果根目录：`/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/runs/v4_train500_matched_v2`。这两处为拟定新路径，当前未创建或核验。正式交付包含共同更新节点主表、独立计算成本表、训练充分性审阅、train/val loss与三指标曲线、固定竖版/横版图、逐记录指标与output hash、固定样本FP32结果、原始计时/配置/日志/恢复审计。

完成时间在共同配置的共卡测试与实际吞吐记录后给出；各组训练量不由ETA倒算。两卡每卡两个实验的资源安排保持，不能据“跑一晚”自行削减或增加某组更新预算。

# V0.31 冻结训练与评估协议

本协议依据统一补充和用户已激活的 AUTO_TO_1000 授权。科学定义沿用 Stage 1：四种真实起点，完整预测状态递归；每步 clean MSE 与原始 Y/G 重退化 MSE；路径内平均再对32条路径平均；lambda_R=0.2；完整 BPTT；训练不做 xi refinement。

恢复网络从 seed20260916 和原初始化重新建立，全部1600起点均匀洗牌、无放回。每轮50updates，共20轮到1000。AdamW peak1e-4、betas(0.9,0.999)、eps1e-8；bias/normalization不衰减，其余1e-4；所有组同一学习率。完成累积和四卡归约后 global norm clip1。EMA0.996，每步更新参数、复制 buffers。FP32主参数、优化器、EMA、simulator和loss；BF16因Stage1梯度parity失败被排除。

50updates线性warmup，之后cosine到update1000的1e-5，之后显式hold。此次自动终点1000，不重启schedule。当前唯一分批配置见train_v031.json和下方报告；设备仅de36物理0–3。没有开放超参数搜索。

每100updates完整保存network、EMA、AdamW、scheduler、四rank RNG、global sampler、计数与身份。初始checkpoint0只做固定64起点unrefined基线。update100用固定raw权重校准一次：固定4trajectory的G4/3/2共12起点，unrefined非末步共24probe；同一probe固定clean/Y/G/初始xi，不读取GT。固定Adam betas(0.9,0.999)/eps1e-8/无WD，LR[1e-3,1e-2,1e-1]，预算[10,20,40]复用40步前缀；每transition重置状态；完整两块xi同LR，不clamp/归一化/投影/clip/提前停止。返回包括初始化的最佳原始观测残差迭代。

各候选必须在24probe均finite且梯度连接，按平均相对残差下降A保留A>=0.95A_best，选累计实测最快，3%耗时tie优先少步再小LR；最多按已记录合格排序尝试3组真实完整桥集成。正式refined每次固定当前clean重新比较网络xi与前步refined xi，逐图整对选择，tie继承previous，选中者只优化一次，末步/one-shot不refine。低信号单列。数值校准不使用clean质量，不证明图像质量最优。

update100校准后所有G/raw/EMA/后续checkpoint共用一组数值。每100固定64起点、500和1000全1600（包含固定64，不重复计算），raw/EMA × unrefined/refined分开；每个G记录全部合法K，第一步为同一原始Y/G的one-shot。逐步网络预测xi的原始观测MSE与refinement候选/前后objective分列。原始float RGB全图mean MSE和data_range1单图PSNR后平均，不clamp指标；tie阈值1e-6dB，G1不算多步增益。只为预览clamp，样本ID在看效果前固定。

评估逐图执行，独立进程不恢复或污染训练RNG/sampler/optimizer。batch2的native xi数值对照失败，原阈值和失败记录保留；batch1两样本重复求解完全一致且完整路径通过。评估batch与训练分批独立。

服务器入口code/run_auto_to_1000.py按0→100校准→每100诊断→500全评估与报告→1000全评估与报告顺序执行。每个子进程由原资源守护器检查空闲/健康/归属并限定时长，只清理本任务已证明归属的进程；持久化日志/PID/进度并支持客户端断开。相同基础设施故障最多重试2次，先确认资源已释放；保留中断重放日志，正式连续日志只代表完整checkpoint接续轨迹。未授权资源、身份冲突、core gradient问题、正式nonfinite/确定性OOM或需改科学方法时停止报告。

1000训练和完整评估均完成才标PAUSED_AT_1000，不因此宣称收敛或泛化。之后须有新的明确授权，将冻结配置复制为续训配置并加入匹配的新run_until_update及continuation_authorization，然后从完整checkpoint_001000.pt用--resume和--authorize-boundary-extension继续；保留原schedule/code/数值身份，不重新初始化。当前入口始终禁止自动超过1000。

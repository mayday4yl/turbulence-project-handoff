# T1_DIRECT｜独立直接恢复对照（最高训练优先级）

必须同时读取`00_MASTER_PLAN.md`。本文件定义本轮第一个训练，四卡优先完整完成它。

## 1. 问题与范围

研究问题：保持原基础clean恢复结构，同数据同起点曝光，只训练一次直接恢复，基础恢复质量能达到什么程度？它与旧V0.31的K1及无refinement完整路径有何差异？

它是“简单整体方案”对照，**不是**仅隔离某一loss的单因素消融，也不是声称独立one-shot已达到最优。

数据仍是400物理trajectory的1600个真实起点。每个起点只做1次网络预测，**G=4的输入也只有K=1**，不能沿用旧helper将num_predictions算成G。

## 2. 前向与目标

\[
\hat C=D_\phi(Y,G),\qquad
L_{T1}=\frac1B\sum_b\operatorname{MSE}_{RGBHW}(\hat C_b,C_b).
\]

- 真实输入为原FP32 Y，G已知。
- 一个模型共同训练四档，不能拆成四个独立模型。
- 只用LC，不用LR、xi回归、频域/感知/边缘loss、refinement、蒸馏或GT-next-state监督。
- 不递归、不生成下一状态、不调用simulator参与训练。
- clean输出保留原sigmoid，不加Y+residual旁路，不换decoder/归一化/插值方式。

## 3. 复用结构和初始化

从原完整DualHeadUNet未训练初始state中按名称提取：

```text
severity
input
encoder
downsample
middle
clean_decoder
clean_head
```

共享部分与clean部分结构、参数值、强度embedding完全保持。原embedding使用0.75–4.50的固定归一化，不因为本轮四档而重新缩放；这是一种既有输入编码，**不代表使用旧六档路径**。

实现独立clean-only wrapper/模块，跳过realization_decoder、realization_up/block、zernike_head、tilt_head。无需“为了参数量一致”运行一个无监督的空分支。

公共初值必须先通过原完整构造函数和seed产生，再复制；不要先删分支再随机初始化造成公共层随机状态偏移。报告T1实际trainable参数量，**不能继续把T1称为3210万参数**，也不补额外容量来填平数量。

若只保留模块对象而没有物理删除无用分支，至少冻结并从optimizer/DDP可训练集合中排除它们；正式forward不得执行无用分支。优先使用干净的clean-only模块，避免unused-parameter副作用。

## 4. 配置与计数

所有优化和采样设置遵循总计划：FP32、raw-only、4卡、effective batch32、1000updates、seed20260916、AdamW与固定schedule、global clip1。

正式1000updates的预期计数：

```text
paths / start exposures = 32000
network clean predictions = 32000
true inputs = 32000
predicted next inputs = 0
simulator calls in training = 0
每个start出现20次；每G8000次
```

注意原`advance_counts`可能按severity_path统计递归预测次数。T1须用自身语义统计，不能原样调用后打印出80000次预测。

EMA关闭，checkpoint保存实际存在的raw/optimizer/scheduler/RNG/sampler/身份，不再声明需要或恢复EMA。

## 5. 最小验收后开训

A. **初始函数等价：** 固定真实四档样本，同一个原完整未训练state；T1输出应与原完整网络的clean输出在同backend、dtype下吻合，报告max_abs/MSE。若不符，先排查参数映射/forward，不能继续。

B. **目标与梯度：** 一个mixed-G batch实际loss等于逐样本full-image MSE平均；clean路径所需参数纳入optimizer。不同head是否有梯度以T1定义为准，不要求不存在的xi分支有梯度。

C. **DDP/累积：** 同一global32批次验证归约缩放正确；独立smoke覆盖一次真实AdamW更新和一次完整checkpoint恢复。使用原已接受的FP32恢复容差，区分“恢复瞬间exact”与“后续重算有界”，不虚称逐位相同。

D. **性能：** 按总计划四种microbatch/累积候选短测；选实测更快的可行配置。不能用原双头full-rollout测速代替T1。

验收通过、配置和source snapshot落盘后，按用户总执行请求直接训练，不需要为普通工程字段逐项等待。不是重新开展一轮Stage1方法审查。

## 6. 运行与评估

训练到500照常保存与排队评估，**不中断等结果**；继续到1000保存完整状态后暂停，不自动续训。

每100固定64起点raw诊断，500/1000全1600；1000另评heldout32。推理池执行评估，网络eval且不更新参数。T1只报K1，不伪造K2/K3/K4列。

比较：

1. T1_K1 vs 输入Y：基础恢复是否学到。
2. T1_K1 vs OLD_V031_raw_K1：不同训练方式下，首次恢复质量差异。
3. T1_K1 vs OLD_V031_raw_unrefined_final：简单一次恢复与旧复杂方案最终效果。

同update的结果放一起，500不对1000；raw不混EMA。报告训练数据与heldout32，不能用训练集胜负代替泛化。

## 7. 交付与解释边界

输出训练曲线与未平滑日志、各G原始输入/恢复指标、逐source配对、固定图、参数量、起点曝光、实际时间、checkpoint/hash、续训入口。

若T1更清晰：支持基础clean结构具备更好的拟合可能；不能单凭本对照断言LR或xi是唯一原因。

若T1较差：只说明当前相同曝光/固定超参数下的表现，不自动宣称桥已胜过训练充分的所有直接恢复模型。

若两者都糊：报告事实与训练趋势，不自动加高频loss、换backbone或增加updates。

T1完成1000并校验状态后交还四张训练卡供T2；最终评估可继续在推理池，不阻塞训练队列。

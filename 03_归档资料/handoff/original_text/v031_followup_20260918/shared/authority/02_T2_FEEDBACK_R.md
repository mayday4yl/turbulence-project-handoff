# T2_FEEDBACK_R｜直接反馈递归对照（保留重退化监督）

必须同时读取`00_MASTER_PLAN.md`。本文件是第二个独立训练；T1完成训练后接手全部四张训练卡。

## 1. 要隔离的变化

对照旧V0.31的**无refinement路线**，仅有意改变下一网络输入的生成方式：

```text
OLD: Z_next = F(C_hat, xi_hat, next_g)
T2:  Z_next = C_hat
```

保留同一个完整DualHeadUNet、参数共享、同一severity路径标签、每步LC/LR、完整反向传播及训练预算。

**不要把它称为“完全无物理模型”的递归网络。** Simulator仍用于LR，T2只去掉它生成中间输入的作用。它也不是所有普通递归方法的最强代表；没有加入每步原始Y旁路、hidden state或新网络。

## 2. 完整前向定义

每个起点读取`(C,Y,G,severity_path)`；路径为：

```text
G4 -> [4,3,2,1], K=4
G3 -> [3,2,1],   K=3
G2 -> [2,1],     K=2
G1 -> [1],       K=1
```

\[
Z_1=Y,\quad(\hat C_k,\hat\xi_k)=S_\phi(Z_k,s_k),\quad
Z_{k+1}=\hat C_k\quad(k<K).
\]

这里`s_k`保持原数值编码和次序，但在T2的后续输入上只是条件/步骤标签；**不能解释成Z_k真的具有对应物理severity**。

不要把G1额外变成一次identity训练，不引入g0，不读真实弱状态、不detach反馈clean。不在每步额外拼接原始Y。

## 3. 每步loss及聚合

\[
L_C^{(b,k)}=\operatorname{MSE}(\hat C_{b,k},C_b),
\]
\[
L_R^{(b,k)}=\operatorname{MSE}(F(\hat C_{b,k},\hat\xi_{b,k},G_b),Y_b),
\]
\[
L_{T2}=\frac1B\sum_b\frac1{K_b}\sum_k[L_C^{(b,k)}+0.2L_R^{(b,k)}].
\]

所有MSE为full-image RGB mean。LR始终使用原始G/Y，包括最后一次预测。simulator参数冻结但输入可微。完整展开、汇总后反向，完成有效batch后才更新。

训练与T2主评估均不做xi refinement；不把I1新优化器接进T2。不存在GT-xi或GT-next-state监督。

## 4. 梯度验收必须按T2的实际结构改写

**这一节是核心：不要错误地要求T2通过旧V0.31的全部xi跨步探针。**

### 4.1 应当存在的通路

仅用G4末步clean loss，检查其对早步实际反馈的`C_hat_1/C_hat_2/C_hat_3`有可用梯度；逐个在独立测试副本切断对应clean反馈，前向值不变，跨断点梯度消失。

正式trainer保留完整clean→下一输入→后步loss通路。负对照只存在测试副本。

### 4.2 不再要求存在的通路

T2的早步`xi_hat`不参与下一状态，因此终端clean-only loss对该早步输出张量本来就应无直接计算图通路。这里返回None不是失败；若意外有通路，反而要检查是否残留旧F-transition或数据旁路。

这是对输出张量的探针，不是对共享参数的探针。不能因为共享encoder有梯度，就误称early xi output参与下游。

### 4.3 xi仍然应从自身LR获得梯度

分别对各步自己的LR，查询实际进入F的clean/zernike/tilt输入梯度。不能因移除状态生成，就把xi分支一起冻结/删除或对LR的simulator使用no_grad。

在单步测试副本改变xi输出而固定clean时，LR可以变化，下一输入应仍严格等于clean；不运行下一状态simulator。

零梯度与断图区别对待；原零初始化造成的局部零导数应如实诊断，不能为了通过给正式网络加随机扰动或改初始化。

## 5. 模型、采样与计数

原完整网络结构与未训练初值不变，公共初始state来自总计划。不是从T1或旧1000权重微调。

优化参数与原V0.31相同的clean/shared/realization所有分组都训练；网络完整参数量应保持原定义，报告actual数值。strong/weak标签、embedding、sigmoid都不改。

1000updates、effective batch32、各起点20次时，预期：

```text
paths = 32000
predictions = 80000
true inputs = 32000
predicted clean-feedback inputs = 48000
LR simulator renders = 80000
transition simulator renders = 0
training xi-refinement iterations = 0
```

四种路径对应每条训练起点的simulator调用数为K，不是旧的2K-1。只读data loader无需加载真实中间状态或GT-xi。

## 6. 配置冻结、运行、评估

其余遵循总计划：FP32、4卡、raw-only、AdamW、相同LR/sampler种子、有效batch32，各1000updates。独立测microbatch，不默认T1或旧V0.31最优分批也适用T2。

针对性验收：四起点逐步trace、loss分母、上述不同梯度预期、DDP缩放、短smoke与checkpoint恢复。旧physics数学不改，不用重跑全量数据生成/旧stage1。

每100固定64、500/1000全量、1000共享heldout32。对每条路径保存所有clean预测指标、LR、最后输出；不事后用best-K替代完整路径主结果。

T2输出仍有xi，但它不是“已经估计真实realization”的证据。T2中后续clean loss不通过xi头反馈，这正是移除F-transition的结构后果；比较结果应据此解释。

## 7. 最终比较与边界

主比较：同数据/同起点曝光、同样结构与逐步loss条件下：

```text
OLD_V031_raw_unrefined_final  vs  T2_FEEDBACK_R_raw_final
```

两者网络预测次数一致，但simulator训练/推理调用不同，**不是完全等FLOPs**；记录成本，不能把额外物理调用隐去。

辅助比较：T2_final vs T2_K1，以及T2_final vs T1_K1。原始Y baseline也报告。

旧桥优于T2：支持“在这个受控替代方案下，F生成下一状态有价值”，不代表胜过全部递归算法。

T2相当或更好：说明当前F-transition尚未显示额外恢复收益，但不证明所有物理信息无用，因为LR两边都保留。

所有比较分训练样本/heldout32、source/G、checkpoint；保持raw口径。完成1000与规定评估后暂停，不自动做新训练消融。

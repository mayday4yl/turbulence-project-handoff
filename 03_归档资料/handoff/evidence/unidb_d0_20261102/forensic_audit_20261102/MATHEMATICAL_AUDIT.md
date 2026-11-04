# UniDB D0 数学与转移一致性独立审计

本审计读取冻结执行源，先从论文控制问题与线性高斯条件分布推导，再对照代码；没有训练、没有网络推理、没有修改原实现或旧报告。数值检验仅是标量系数及一百万个高斯随机数。结论不预设官方代码的概率解释完全自洽。

## 结论先行

1. 当前 `NativeObjective` 的**执行定义**是未加时间权重的一步均值 L1，与所复制的官方代码调用链一致。这不等于逐字实现论文全部公式，也不等于严格的高斯 KL 或无偏 epsilon 回归。
2. 论文把有限 gamma 的控制均值与另选的零端点插值方差组合起来；这个训练插值分布**不是同一有限 gamma 受控 SDE 的精确边缘**。该结构来自上游论文/代码，不能用“文件 SHA 相同”排除；但实际 gamma_inv=1e-7 时，t=1..99 的方差相对差最多约3.12e-5，单凭它不足以解释数 dB 退化。
3. 训练 target 是特定离散高斯 Markov 链的一步后验均值；采样执行的是 Euler 反向均值加 `g_t sqrt(dt)` 噪声。二者**均值存在离散化差异，方差也不等于该离散后验方差**。例如 t=2 的采样方差为 target 后验方差的2.912倍，t=99约1.108倍。它们可以是连续极限下的近似，但不能说是同一个有限步转移。
4. MatchingLoss 拟合的“noise”最优标签一般不是真实前向 epsilon，必须补偿 Euler/精确离散均值差。未显式加权也不表示时间梯度同权：epsilon 参数化下的隐含 L1 权重跨 t 差25.54倍。
5. 在只给网络 `(x_t,y,t)`、对隐藏 GT 取条件分布的总体风险中，L1 的最优解是逐坐标条件中位数；严格 score/条件均值性质通常需要相应二次风险。论文高斯 KL 推导中的平方项不能仅删常数就变成 L1。这里存在真实的概率解释缺口，不能靠“L1 官方原生”关闭。
6. 端点另有明确差异：论文 Algorithm 1 写训练 t=1..T，代码实际1..99；论文 Algorithm 2 写末步噪声 z=0，代码 t=1仍注噪。代码 t=100 的模型输出实际被丢弃，100次网络调用只有99次输出进入更新。实际 t=100 的 `f_sigma_1`/`r_sigma_1` 为NaN，但训练不调用该时刻、sampler走特殊分支，因此这不是正式训练发生NaN的证据。

这些是需要纳入最终解释的已核实结构，不等同于已证明其中某项是本次失败的唯一原因。尤其末步注噪单独对应约53.11dB的误差地板，显然不能直接解释15.9dB的恢复结果。

## 1. 来源与证据边界

实际复制来源为 `UniDB-SOC/UniDB` commit `1bb63e318891b932c6e641b98729643e46b91e06`。本节逐行对照当前 `vendor/unidb_gou/sde_utils.py`、`core.py` 与 `loss.py`；代码是否与GitHub逐字相同由官方差异审计单独复核，不能由数学结论代替。

主要数学主源为 [ICML 2025正式PMLR页面](https://proceedings.mlr.press/v267/zhu25o.html)及[正式PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhu25o/zhu25o.pdf)，已保存`UniDB_ICML2025_zhu25o.pdf`（28页）。正式第5页Eq18/Eq19、第6页Algorithm1/2和第18页Gaussian KL已渲染并目视读取，截图`icml_page05.png`、`icml_page06.png`、`icml_page18.png`。另取得[arXiv v3](https://arxiv.org/html/2502.05749v3)及[其PDF](https://arxiv.org/pdf/2502.05749v3)（27页）交叉复核：本报告所列主要式子/符号/算法边界问题在正式版仍然存在，并非只引用旧预印本的错字。预印本的第5/16/17/22页截图亦保留，页码不能混用。

独立重算 `coefficient_audit.py` 生成 FP64/FP32 两份逐时刻CSV；`ACTUAL_CPU_COEFFICIENTS.csv/json` 来自远端**实际未修改** `core.process('cpu')` 和 vendor 方法，未加载网络。前者用于数值稳定推导，后者用于确认真正执行的标量结果；它们因FP32舍入不要求逐位相同。

## 2. 独立的线性高斯概率推导

设 clean 为 x0，观察条件为 y，r=gamma^{-1}>0，lambda=30/255。记连续累计时钟 H(t)=integral_0^t theta(s)ds，a(s,t)=exp(-(H(t)-H(s)))，基础OU增量方差 b(s,t)=lambda^2[1-a(s,t)^2]。这里 y 是退化图条件，数学上的x0并非推理时可见输入。

基础OU的中心变量Z=X-y满足 dZ=-theta Z dt+g dW，g^2=2lambda^2 theta。有限终端二次代价给出的反馈漂移系数为

\[
\kappa_t=\theta_t+\frac{g_t^2a(t,T)^2}{r+b(t,T)},\qquad dX_t=\kappa_t(y-X_t)dt+g_t dW_t.
\]

这是从论文Eq20的概率模型出发，而不是先把代码的f_sigma当定义。状态转移基本解为

\[
\Phi(t,s)=a(s,t)\frac{r+b(t,T)}{r+b(s,T)}.
\]

所以其真实受控SDE边缘为

\[
q_{\rm SOC}(x_t\mid x_0,y)=\mathcal N\big(y+m_t(x_0-y),V_t^{\rm SOC}I\big),
\quad m_t=a(0,t)\frac{r+b(t,T)}{r+b(0,T)},
\quad V_t^{\rm SOC}=b(0,t)\frac{r+b(t,T)}{r+b(0,T)}.
\]

协方差可以直接积分 `Phi(t,s)^2 g_s^2`，或把基础OU终端用方差r的高斯观测软约束到y后作高斯条件化得到。两条推导一致。特别地，有限r时终端方差不是0，而是 `b(0,T)r/[r+b(0,T)]`。

论文Eq18与代码没有使用这个方差，而使用

\[
q_{\rm code}(x_t\mid x_0,y)=\mathcal N(M_t,V_tI),\quad M_t=m_tx_0+(1-m_t)y,
\quad V_t=\frac{b(0,t)b(t,T)}{b(0,T)}.
\]

论文4.3明确将此作为另行引入的随机插值噪声。r趋于0时两个方差一致；有限r时只改均值而固定GOUB方差，便不能把 `q_code` 直接说成有限gamma-SDE的真实边缘。

本配置定量结果：终端 `m_100=3.6125642e-8`、`V_code,100=0`、`V_SOC,100≈9.99992775e-8`。活动时刻1..99的最大 `(V_SOC-V_code)/V_SOC≈3.12237e-5`，发生在99附近。`COEFFICIENT_AUDIT.json`还包含独立高斯条件采样验证：一百万次样本方差约9.9950e-8，支持真实有限r端点方差非零。这个反例针对概率解释，绝不能说该微小差别已经解释图像退化。

普通DDPM把数据向与观察无关的噪声先验扩散；此处均值同时包含x0与y，方差在两个端点归零（代码插值）或在有限r终端保留小方差（真实SOC），是不同的条件过程。

## 3. 一步GT target的条件分布及额外假设

只知道每个时刻的边缘 `N(M_t,V_t)` 并不足以唯一决定跨时刻协方差。代码/论文追加了线性Gaussian Markov构造：

\[
F_t=\frac{m_t}{m_{t-1}},\quad Q_t=V_t-F_t^2V_{t-1},\quad
x_t=F_tx_{t-1}+(1-F_t)y+\sqrt{Q_t}z_t.
\]

在Q_t>=0时，这个构造确实保持指定边缘，并给出

\[
K_t=\frac{F_tV_{t-1}}{V_t},\qquad
\mu_t^*=M_{t-1}+K_t(x_t-M_t),\qquad
R_t=\frac{V_{t-1}Q_t}{V_t}.
\]

这一步显式使用 `Cov(x_{t-1},x_t|x0,y)=F_tV_{t-1}`；不是遗漏协方差，而是选择了特定协方差。代码 `r_mean_1` 展开与此一致，`reverse_optimum_step` 只是返回该值；GT条件变量没有接反。若把它解释成真实有限gamma受控SDE的最优posterior，则必须把V换成V_SOC，不能直接沿用V_code。

这里的目标没有网络参数；前向状态也由固定GT/y与随机数构造，所以target不反传本身没有不应有的stop-gradient。诊断时用GT计算free-running状态的目标是事后参考，不等于推理能使用GT。

在t=1，V0=0，故K1=R1=0，精确离散posterior的目标是x0、方差为0。在t=100，V100=0而m100>0，得到 `Q100=-3.12225e-12<0`（FP64），不存在该形式的合法实高斯创新。实际vendor方法也给NaN。训练避开t=100，不能用这个未执行分支反推正式loss出现过NaN。

## 4. epsilon到训练均值：符号、离散误差与真实标签

对代码前向高斯，真实条件score为

\[
s^*(x_t,x_0,y,t)=-\frac{x_t-M_t}{V_t}=-\frac{\epsilon}{\sqrt{V_t}}.
\]

当前模型 `epsilon_theta` 经 `get_score_from_noise=-epsilon_theta/f_sigma` 后，反向Euler均值为

\[
\hat\mu_t=(1+\kappa_th)x_t-\kappa_th y-c_t\epsilon_\theta,
\qquad c_t=\frac{g_t^2h}{\sqrt{V_t}}>0.
\]

负号由 `x - (forward_drift - g^2 score)h` 与score定义严格推出。当前代码负号与该推导相符；论文Eq19却打印为正号，Algorithm1/2还打印出不同的sigma次方。不能为了逐字贴合不一致的纸面式子而把当前代码改成正号。

令Euler均值匹配上节精确离散posterior，实际noise回归标签必须是

\[
\epsilon_t^{\rm opt}=\frac{(1+\kappa_th)x_t-\kappa_th y-\mu_t^*}{c_t}.
\]

将训练状态 `x_t=M_t+sqrt(V_t)epsilon` 代入：

\[
\epsilon_t^{\rm opt}=A_t(x_0-y)+B_t\epsilon,
\quad A_t=\frac{(1+\kappa_th)m_t-m_{t-1}}{c_t},
\quad B_t=\frac{\sqrt{V_t}(1+\kappa_th-K_t)}{c_t}.
\]

有限步长下它不是字面“预测真实epsilon”。例如FP64 t50：A=-0.038338，B=0.988804；B在t84附近约0.977381；|A|在t46附近约0.039593。小t较接近原始epsilon，但实际FP32的 `1-exp(-small)` 与schedule舍入也会产生小误差。已经生成逐t A/B与Euler-vs-exact系数，而不是仅凭名称判断一致。

## 5. MatchingLoss实际优化什么

对每张RGB256图，vendor `loss.py` 先逐元素L1，再平均3×256×256个元素，最后平均batch。每图只抽一个独立的整数t，因此时间期望由蒙特卡洛抽样实现，不是每次更新显式平均99个时间。DDP再平均八rank梯度；当前accumulation=1，有效batch96。

\[
L_{\rm code}=E_{g,x_0,y,t,\epsilon}\Big[\frac{1}{3HW}\|\hat\mu_t-\mu_t^*\|_1\Big]
=E\Big[c_t\frac{1}{3HW}\|\epsilon_\theta-\epsilon_t^{\rm opt}\|_1\Big].
\]

`is_weighted=False`表示没有额外显式权重，不表示epsilon参数化各时间同权。c_t从t1约0.00220995增至t99约0.05644968，比例25.5434；网络参数梯度还乘各状态下的Jacobian，不可能据样本数均分便断言四强度梯度贡献相等。均匀强度采样只保证样本概率，未测量的per-strength梯度冲突仍是未知。

论文Eq19还有 `1/(2 sigma_theta^2)` 的时间因子，官方/本实验未加它。若将Euler方差写完整为g^2 h，取消这个因子会改变时间梯度分配。即使这不改变无限容量逐t逐条件的点态L1最优值，也影响有限网络与SGD优化，不能只称整体常数。

固定 `(x_t,y,t)` 后，隐藏clean的条件分布会让mu*成为随机标签；L1的Bayes最优预测是逐坐标条件中位数，不是一般的条件均值。L2才给均值。`L1_MEDIAN_COUNTEREXAMPLE.json`用条件标签0/1、概率0.9/0.1说明：中位数0的L1风险0.1，小于均值0.1的L1风险0.18，但反向均值应为0.1。它只证明无普遍保证，不证明本项目条件后验确实多峰或该因素造成实测失败。

此外，对两个Gaussian的KL，均值差项严格为平方L2；正式论文第18页（arXiv v3第17页）Eq59本身也是平方。把它改成L1是视觉质量启发式/替代目标，不能通过“忽略常数”保留原Gaussian ELBO等价性。合理使用L1并非被数学禁止，但必须诚实区分：官方经验loss、概率似然目标、和当前湍流恢复指标。

## 6. 100步采样逐项审计

完整时间映射见两个CSV，共t0..100：训练支持1..99；采样网络调用100,99,...,1；返回状态0。`dt`不是基类初始的1/100，而被UniDB._initialize改为约0.104093804955；100个整数索引并不是100个间隔0.01的物理时间步。

对于t=1..99，训练的 `sde_reverse_drift` 与推理 `_1` 代数一致，区别只是tensor mask与Python分支表达；不存在这里的反号或索引差一。但它们均使用上述Euler均值，而target是精确离散均值。

采样反向转移为

\[
x_{t-1}=\hat\mu_t-\sqrt{E_t}z_t,\quad E_t=g_t^2h.
\]

负号不改变零均值Gaussian的分布，但固定随机数配对时必须保留同一符号和消耗顺序。E_t不是训练目标隐含的R_t：

|t|Euler采样方差 E_t|精确target后验方差 R_t|E_t/R_t|
|---:|---:|---:|---:|
|1|4.883017e-6|0|无穷/不定义|
|2|9.336371e-6|3.206162e-6|2.912009|
|10|9.245327e-5|7.150835e-5|1.292902|
|50|0.001458484|0.001361244|1.071435|
|99|0.002878800|0.002599044|1.107638|

这些是FP64独立重算，实际CPU标量另表记录。均值匹配没有同时匹配完整转移分布；但Euler连续SDE近似本来就不要求有限步严格等于离散posterior。应把它列为可测的结构差异，再由oracle/采样对照判断实际影响，而不是见方差比便宣称故障已证明。

### 首步与末步

代码起点x100=y。t100网络仍前向一次，但score强制为0，h-transform亦特殊清零；由于x=y，均值不动。SDE随后注入方差约0.002880817的噪声，故x99的分布是N(y,0.002880817I)。训练t99则均值为y+0.001042633(x0-y)、方差0.002600758，噪声方差相差约10.768%。mean-ODE首步保持y。这里的初始状态差异是精确可写的，不应笼统称“teacher forcing”。

最后t1精确posterior应是delta_x0；代码仍注入std≈0.00221的噪声。正式论文第6页（arXiv v3第22页）Algorithm2明确写最后z=0，与代码不同。其单独误差地板约53.113dB，量级远不足以解释最终15.9dB；仍应记录为真实paper/code差异。

### Mean-ODE不能直接解释成恢复均值

当前 `reverse_mean_ode`仅删除Brownian项，保留完整SDE score漂移；它不是采用半score的probability-flow ODE，不保证保持与SDE相同边缘，也不一般等于posterior mean图像。若模型在x=y时输出epsilon=0，所有无噪声状态可保持y，而SDE因每步噪声进入不同状态；因此“去噪声后PSNR回升”可以由近似identity与随机变化共同造成，不足以单独证明真正学到恢复。该构造不是对当前网络输出的实测结论，应由output-input能量、频谱、边缘与多种子审计判断。

## 7. 三层判断与未解事项

|问题|当前可下结论|不能越界的部分|
|---|---|---|
|A. 是否实现了官方代码loss|核心prediction/target/condition/L1路径对齐，逐t训练与推理漂移相同；上游SHA对照另审计|不等于全部论文公式相同；存在权重、符号印刷、末步处理差异|
|B. 数学目标是否合理|作为受控的经验一步L1目标可定义；活动t的离散posterior均值有明确Gaussian推导|不严格等价Gaussian KL/真实epsilon/有限gamma-SDE边缘及完整转移分布|
|C. 是否适合当前湍流任务|没有由上述数学提供PSNR/SSIM/LPIPS保证；适用性必须靠实验|当前实测失败不能由某一个小系数差异、L1名称或局部敏感性唯一归因|

最有价值的下一步检验（均不训练）：已知GT的解析oracle、真实epsilon与epsilon_opt两种oracle对照、将Euler方差与离散posterior方差分别用于同一oracle、末步z=0的纯诊断、完整保存每步实际噪声与均值、多个种子及共享状态扰动有限差分。父审计正在安排这些证据，任何新数值应另标测量，而非混入本节标量理论结论。

## 8. 可复核材料

- `coefficient_audit.py`：独立公式、FP64/FP32映射与Gaussian反例。
- `COEFFICIENTS_FP64.csv`、`COEFFICIENTS_FP32.csv`：每个t的m/V/F/Q/K/R、Euler均值系数、epsilon_opt的A/B、方差比与隐含时间权重。
- `ACTUAL_CPU_COEFFICIENTS.csv/json`、`ACTUAL_CPU_RAW.txt`：实际未修改vendor标量方法调用结果。
- `COEFFICIENT_AUDIT.json`：关键量和源文件SHA。
- `L1_MEDIAN_COUNTEREXAMPLE.json`：L1中位数/均值不等价的构造反例。
- `UniDB_ICML2025_zhu25o.pdf`、`UniDB_2502.05749v3.pdf`与七张论文原页截图：原始数学来源和符号/末步差异证据。

本报告既没有认定训练实现整体无bug，也没有认定已找出唯一失败原因；它明确撤回“官方相同即可推出数学完全一致”的推断。

## 9. 实际CPU timestep RNG全程重放结果

此项不把原日志未记录的逐图tau伪装成原始观测。从U0完整checkpoint每个rank的CPU Torch RNG开始，使用相同环境的`torch.randint(1,100,(12,1,1,1))`逐更新重放。结果：

- 8个rank × 45411更新，共363288条原始日志的tau_min/max逐条一致。
- U0/U500/U10000/U20000/U30000/U40000/U45411七节点 × 八rank，共56个CPU RNG状态逐字节一致。
- 共重建4359456个tau，保存`RECONSTRUCTED_TAU.npy`（8×45411×12，uint8），1..99全部出现、0/100未出现。
- 以实际训练indices关联g后，G1/G2/G3/G4平均t约49.9992/49.9738/50.0041/50.0008；平均隐含epsilon-L1系数c约0.0148434/0.0148329/0.0148401/0.0148443。未见G4被timestep抽样系统性偏置。
- 这不说明各g真实参数梯度相同，也不重建NPU高斯噪声；后者是独立RNG流，不能用CPU状态替代。

`CPU_TAU_REPLAY_AUDIT.json`记录全部56项状态SHA和重建数组SHA，`RECONSTRUCTED_TAU_HISTOGRAM.csv`含每t按rank/g计数，`RECONSTRUCTED_TAU_WINDOWS.csv`含各训练区间分布。7个checkpoint文件SHA在此helper中仅引用原sidecar身份，没有伪称重新hash了全部多GB文件；提取的CPU RNG均重新hash。实际CPU进程PID619292/start591191098已确认退出，零模型/NPU/训练调用。

## 10. 有限gamma家族的活动时刻反例（不代表本次实际NaN）

`finite_gamma_counterexample.py`在同一theta/硬桥方差下仅替换gamma_inv作纯标量反例：r=.01（gamma=100）时，t99的`Q=V99-F99^2 V98≈-2.0664e-4`；r=.1或1时t98/99均为负。因而这个“有限gamma均值+固定GOUB方差”的离散Gaussian链并非任意gamma都可定义。实际r=1e-7在t1..99的Q全部正；这一家族反例不能被写成当前loss曾NaN，也不是新的网络超参数搜索。它要求今后若改变gamma，先验证全时域创新方差而非只改变配置。


---

本顶层交付版完整来源为 `math/MATH_AUDIT_ZH.md`。原子报告与原始表/源码均保留；文中早期“后续检查”现由 `SAMPLER_AUDIT.md`、`OPTIMIZER_AUDIT.md`、`teacher_noise_probe/REPORT_ZH.md`、`cpu_rollout_parity/REPORT_ZH.md` 完成对应验证。综合因果排序以 `FINAL_FORENSIC_AUDIT.md` 为准，不能只从官方源码相同或某项数学差异推出根因。

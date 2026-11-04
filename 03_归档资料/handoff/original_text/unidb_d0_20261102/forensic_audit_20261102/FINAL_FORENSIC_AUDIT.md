# UniDB 单帧湍流 D0：独立 forensic audit

## 结论摘要

**本次确实是单张湍流 RGB 图像的独立 D0 全参数微调，没有接 A/B、物理参数预测、raw-ξ、Adapter、task prompt 或外层物理递归。原生100步扩散是网络内部恢复过程。** MatchingLoss 的活动执行路径与所取官方代码一致；本次没有发现GT接反、RGB/尺度错配、DDP重复缩放、Adam错步或粗大CPU/NPU数值错误。

但“忠实实现官方loss”不能推出“数学上是严格Gaussian似然”，也不能推出“适合单帧几何湍流”。本次最强的实验结论是：**微调权重和原生随机采样规则之间存在可重复的不良交互；训练目标的改善没有转化成强档结构与像素恢复。** 改变方差/边界约定可以挽回一部分质量，仍未恢复到输入或初始化水平。没有证据支持继续原配方长训，也没有证据支持直接宣判必须换loss、换Adam或换backbone。

本审计新增训练更新为0、optimizer对象/step为0。只读梯度重算有backward但没有更新。原训练停在U45411；原始代码、checkpoint、日志、数据和旧V5/V6文档未改，STOP/NO_TRAINING_RESUME保留。原报告只作为待核实材料，以下分别给出支持与反证。

## 1. 调查范围、来源和证据等级

本地审计目录为`unidb_d0_20261102/forensic_audit_20261102/`；远端对应`liteserver-de36:/data2/yanglin/unidb_d0_20261102/forensic_audit_20261102/`。全部新增代码/表/图/临时档案只在该目录。原附件在第9节开头截断，用户补充9–14完整，15节仅开头；没有伪造未提供的第15节。逐要求覆盖见[REQUIREMENT_COVERAGE.csv](REQUIREMENT_COVERAGE.csv)。

证据标签：已证实＝可由原件/公式/指定数据直接复算；强支持＝有明确配对或干预，但范围有限；候选/尚未验证＝仍缺区分性干预；强反对/所测反例＝只限定所测定义，不包装成排除一切可能。所有来源级CI将同source四ξ及多seed作为同源重复，不把它们当独立内容来源。

上游固定`UniDB-SOC/UniDB@1bb63e318891b932c6e641b98729643e46b91e06`，六份vendor文件与独立获取Git树blob逐项相同。数学主源是[ICML2025正式PMLR论文](https://proceedings.mlr.press/v267/zhu25o.html)及[正式PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhu25o/zhu25o.pdf)，与arXiv v3交叉核对，正式页5/6/18已目视读取。官方网络/数据/训练/测试的逐项位置见[OFFICIAL_VS_CURRENT.md](OFFICIAL_VS_CURRENT.md)，源码和原件SHA见manifest；没有只凭函数同名认定相同。

### 不同证据的真实规模

|证据|范围|不是哪种证据|
|---|---|---|
|正式完整验证|6节点×5G×1000=30000记录，250独立val源×4ξ|不是六组新的独立验证源|
|数据重解码|10250原JPEG全部与clean逐像素max0；全160K身份/组合检查|不是160K全部独立物理重render|
|旧2×2图像分析|3200保存图，50train＋50val来源×G1/G4×4ξ×4分支|不是mean-ODE全Val1000|
|新完整轨迹|64条，4EMA×8来源×G1/G4，首ξ；每条101状态|不是64独立来源|
|新恢复输出|2272张：native/新seed/方差/RAW/SR合计|不是2272个独立来源或训练呈现|
|多seed|U0/Final，50Val G4来源×4ξ×4固定seed|不是任意随机数总体期望|
|局部状态扰动|5760 FD记录；两共享状态族、10t、3方向、3幅度|不是完整Jacobian谱或全局Lipschitz|
|完整梯度|4RAW节点×4G×6t×B4=96梯度，4train来源|不是45411步历史梯度方向|
|教师噪声|4EMA×16来源/G组合×3t=192前向，48共同态|组合/时刻不增加独立来源数|
|SR功能保持|8Places x4退化族代理＋DIV2K0801一固定crop，4draw，288输出|不是完整官方SR benchmark|
|CPU闭环|一个固定Val G4来源首ξ，U0/Final各100步＋3same-state gate|不是全后端全历史等价证明|

## 2. 实际执行路径与最终失败是否真实

真实路径为`pipeline → run_env → torchrun worker → DDP(NativeObjective) → data.Positions → 官方桥状态/预测均值/GT target → MatchingLoss → 标准Adam → EMA`。没有实例化官方整套DenoisingModel训练器；本项目使用官方数学模块和自写训练/保存/评价wrapper。

每图输入为CHW RGB(3,256,256) FP32的同分辨率y，GT为对应clean。网络输入concat([x_t−y,y])及整数扩散t，没有显式g/ξ/task prompt。训练在1…99独立抽t，每次optimizer.step使用8×12=96张，无累积；loss逐RGB/像素均值、再batch均值，DDP平均梯度。正式45411更新、4359456呈现，160K组合约27.2466次遍历，不能称435万独立图。

原生SDE、EMA、raw-float RGB、无clip/crop/resize的完整每G1000图结果：

| G | Input PSNR | U0 PSNR | Final PSNR | Final SSIM | Final LPIPS↓ | Final MAE↓ |
|---|---|---|---|---|---|---|
| 0.500000 | 25.170834 | 16.262089 | 20.692201 | 0.588220 | 0.080145 | 0.056392 |
| 1.000000 | 22.241768 | 16.155999 | 19.535007 | 0.511612 | 0.095767 | 0.065613 |
| 2.000000 | 20.306366 | 17.111245 | 17.917292 | 0.423474 | 0.137962 | 0.080868 |
| 3.000000 | 19.471852 | 18.266974 | 16.868069 | 0.373765 | 0.234101 | 0.093909 |
| 4.000000 | 18.795938 | 18.222398 | 15.894991 | 0.302786 | 0.425333 | 0.107844 |

G4 Final对U0下降2.3274dB、对Input下降2.9009dB；来源bootstrap区间分别约[-2.409,-2.248]和[-2.989,-2.814]。G3/G4在50/50类别均下降，纹理、Sobel结构和输入难度四层也都下降；最高GT纹理四分位248图中0张Final PSNR高于Input。每类别只有5源×4ξ，类别排名不能作因果解释。详见[完整失败分层](failure_strata/STRATA_AUDIT_ZH.md)。

LPIPS与准确结构分离：完整G4有95.9%图LPIPS优于Input，却仅1.6%图PSNR优于Input，SSIM也更差。训练中没有LPIPS/感知loss；这种指标取舍不能反推“偷偷改了感知目标”。

U0/Final各完整5000保存预测在原收尾已独立CPU核SHA/PSNR/SSIM/MAE及LPIPS重聚合0差；此次再核30000记录配对、5000Input指标，并重读固定保存图。最终失败不是只靠一张图或监控子集宣布。U0仍是已完成完整G4评价节点的最佳；U10000仅是已完整评价的已训练节点中最佳，不是所有更新的全局最优。

![实际训练、梯度与参数动态](figures/OPTIMIZATION_DYNAMICS.png)

## 3. 官方一致性、数学合理性、任务适用性必须分开

|层次|审计结论|
|---|---|
|A 当前是否实现官方活动loss|是，所查prediction/target/condition/无权L1路径一致；训练与推理在相同x/t/noise预测下mean一致。并非整个官方SR recipe复现。|
|B 官方数学目标是否等于其概率解释|作为一步经验L1可定义；不严格等于一般Gaussian KL、真实epsilon回归、完整finitegamma SDE边缘或有限步转移分布。|
|C 是否适合当前湍流任务|本配置实测没有实现强恢复；理论没有PSNR/SSIM/LPIPS保证。何种目标/优化/几何因素主导仍需最小受控训练区分。|

### 独立推导的关键式子

设r=gamma^-1、lambda=30/255、H是theta的累计时钟、a(s,t)=exp[-(H(t)−H(s))]、b(s,t)=lambda²(1−a²)。有限gamma受控OU-SDE给出

`m_t=a(0,t)(r+b(t,T))/(r+b(0,T))`，

`q_SOC=N(y+m_t(x0−y), V_SOC I)`，`V_SOC=b(0,t)(r+b(t,T))/(r+b(0,T))`。

论文/代码实际训练插值却使用`V=b(0,t)b(t,T)/b(0,T)`，保留上述有限gamma均值。两者不是同一个有限gamma边缘；实际r=1e−7时活动t1…99相对方差差≤3.12e−5，不能据此解释数dB失败。

指定额外Markov协方差后，令`F=m_t/m_prev, Q=V−F²V_prev, K=F V_prev/V`，精确GT条件反向target为

`mu*=M_prev+K(x_t−M_t)`，`R=V_prev Q/V`。

预测使用反向Euler均值

`mu_hat=(1+kappa*h)x_t−kappa*h*y−c_t*epsilon_theta`，`c_t=g_t²*h/sqrt(V_t)`。

故native L1严格等价于`E[c_t |epsilon_theta−epsilon_opt|]`，其中`epsilon_opt=[(1+kappa*h)x−kappa*h*y−mu*]/c_t`，一般不是实际forward epsilon。t50约`epsilon_opt=−.03834(x0−y)+.98880 epsilon`；隐含c_t跨1…99差25.54倍。无显式权重不等于时间/强度梯度同权。

对于给定(x_t,y,t)而隐藏clean的总体风险，L1最优为逐坐标条件中位数，一般不等条件均值/score；Gaussian KL的均值误差是平方项。论文主文明确把L1作为视觉质量启发式替代，因此不是指控作者从未说明替代；但附录删常数并不能把平方风险等价变为L1。数学反例只证明缺少普遍保证，不证明本数据的后验多峰或L1就是主因。

![完整系数和隐含时间权重](figures/timestep_coefficients.png)

### 精确的时间与方差差异

训练t=1…99；采样调用100…1，输出状态0；dt实际约.1040938而非.01。t100网络虽计算但score强制0，x100=y时mean=y，首步产生`x99=y−sigma100 sqrt(dt)z`。其方差.002880817，而训练t99为.002600759，均值还带.0010426(x0−y)。

t1…99的训练/推理predicted mean没有发现反号/差一，但target为离散posterior、采样variance为Euler E_t=g_t²h，不是R_t。E/R在t2≈2.912、t10≈1.293、t50≈1.071、t99≈1.108。论文写最后z=0、代码仍加噪。最后一步单独地板约53.11dB，远不足解释15.9dB。GT解析oracle＋native噪声仍可达到该地板，故不能把这些结构差异误称不可避免的15.9dB上限。

实际t100的硬桥创新方差会给NaN，训练避开该时刻、sampler有特殊分支，没有正式run执行NaN的证据。若未来改gamma，活动创新方差不保证仍合法；纯标量反例已保留，非本次gamma搜索。

完整推导、Formal PDF符号差异、101时刻表及CPU实函数值在[MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md)。

## 4. loss为何下降：模型确实学好了部分一步目标

教师态固定相同x/y/epsilon，四EMA八来源、G1/G4、t10/50/90：G4 t10原生L1从.0032699降至.00149199，真epsilon相关性.6243→.9014；t50 L1 .0027776→.0013531、相关性.9482→.9871。预测RMS并非唯一证据，真epsilon与epsilon_opt分别报告。192预测和48组共享张量经纯NumPy核SHA/独立重算，最大标量差2.15e−7、L1重构差5.05e−9。

因此不能解释为“loss只是记录错了、网络什么都没学”。**它学好的，是指定前向桥样本的一步经验目标；这与把未知clean从自身100步随机轨迹中准确恢复，是不同的检验。**

![共享教师态noise与目标](figures/TEACHER_NOISE_TARGETS.png)

### 各自轨迹与相同状态的反证

64条轨迹每条保存101状态、100预测、100mean、100实际dispersion、99GT posterior。CPU逐项核`next=mean−dispersion`最大误差0，原生mean约1.19e−7、GTtarget约5.96e−7，训练/推理mean差0。48个跨权重配对均在t100/t99状态相同，首次有效mean差在t99、首次state差在t98；这是不同权重的正常作用，不是质量劣化起点。

四Val来源首ξ的G4，Final−U0从t52向0持续低于−.1dB，t39低于−.5，t31低于−1；末态17.7493 vs20.4401dB。**两者绝对PSNR中后段仍可上升，扩大的是两条轨迹之间的差。** 这些阈值是描述性跨越，不是统计因果变点。U500已有类似早期功能分离，G1并非相同退化方向。

重要反证：固定共同U0自由状态、4Val G4来源、十t等权L1，Final .0165086略优于U0 .0166242；固定GT-forward则.00177995→.00092847。不能声称“所有free-state一步预测都更差”，也不能仅凭own-Final态误差更大就宣判分布偏移是已识别主因。不同节点访问不同状态，状态分布、多步偏置与随机动力组合的因果份额尚未隔离。

![同实际随机序列的完整100步轨迹](figures/sampler_full_trajectories.png)

### 局部敏感性没有支持预设的‘全面放大’

5760条x方向FD与5120激活唯一键完整、无缺失/重复；共享两状态族、10t、三方向/幅度。Val G4共同U0状态Gaussian、delta RMS1e−3，在t1/2/5/10，Final局部mean增益约.832/.770/.754/.812，U0 .875/.839/.806/.837；t50约.9333/.9342。所测低t多数更小、高t接近，反对“这些方向普遍变得更敏感”。有限方向不能排除未测谱方向、多步非正规放大或非局部偏置。

条件y探针也未见网络依赖整体消失：G4 t10/50/90的Final网络noise预测自身局部增益约34.8/7.98/8.79，完整mean增益约.158/.0562/.0941；后者还包含显式mu漂移，不能只靠后者证明网络使用条件。wrong-y响应有时间依赖，不能将OOD错条件反应当无分布影响的因果证据。网络项与显式mu项带符号重构残差2.38e−7；RMS本身不能相加。见[条件诊断](condition_probe/REPORT_ZH.md)。

## 5. 随机采样的干预证据，以及Mean-ODE为何不是成功恢复

旧固定50Val G4源×4ξ=200图的四分支已重新读取SHA并补output–input LPIPS：

|分支|Out–GT PSNR|SSIM|LPIPS↓|Out–Input PSNR|Out–Input MAE|Out–Input LPIPS↓|
|---|---:|---:|---:|---:|---:|---:|
|U0 SDE|18.0076|.3682|.6259|27.6591|.02861|.5282|
|Final SDE|15.7611|.3196|.4183|18.1084|.08738|.6463|
|U0 mean-ODE|18.5306|.4537|.5376|42.9998|.004325|.0677|
|Final mean-ODE|18.5251|.4537|.5424|32.0563|.016225|.0438|

同subset Input PSNR18.5614。U0 mean-ODE的变化MSE仅原input误差MSE平均.3788%，Final约5.0385%；U0是强近identity，Final为小幅修改邻域而非严格identity。Mean-ODE只删除Brownian项、保留完整SDE漂移，不是probability-flow ODE，也不是被证明的posterior mean。PSNR反弹主要回到输入水平，未建立有效G4去模糊/几何恢复。

明确按[(Final_ODE−Final_SDE)−(U0_ODE−U0_SDE)]定义的PSNR交互为+2.24096dB，95%CI[+2.04471,+2.43491]；若反向作差则数值和区间同时取负。U0的mean-ODE是旧诊断中post-hoc补充；本次固定新seed不是重新选择最有利sampler。

### 四固定seed复核

| seed | Final−U0 dB | 95%下 | 95%上 | 负向来源/50 |
|---|---|---|---|---|
| 0 | -2.246492 | -2.422722 | -2.065964 | 50 |
| 3 | -2.359094 | -2.488838 | -2.227564 | 50 |
| 1 | -2.290633 | -2.458989 | -2.121623 | 50 |
| 2 | -2.308272 | -2.486066 | -2.137743 | 50 |

四seed每次50/50来源Final平均PSNR更差，同时50/50来源LPIPS改善；合并SSIM差约−.04906，LPIPS差约−.20960。这不是单个seed碰巧失败，但仍只代表这四固定seed和50来源。

### 唯一改变随机方差/边界约定

原生Euler mean不动、完整native z消耗顺序不动，将t100尺度设f_sigma99、t1…99尺度设r_sigma_1（t1=0）。该整套规则是显式反事实，不冒充精确finitegamma逆过程：

| 权重 | native PSNR | control PSNR | 差值 | 95%下 | 95%上 | native LPIPS | control LPIPS |
|---|---|---|---|---|---|---|---|
| FINAL_EMA | 15.761126 | 16.892102 | 1.130977 | 1.009498 | 1.257969 | 0.418330 | 0.364284 |
| U0_EMA | 18.007618 | 18.367968 | 0.360351 | 0.339964 | 0.381051 | 0.625879 | 0.495077 |

Final提升1.1310dB、50/50来源正向；但控制Final仍低于Input18.5614和控制U0的18.3680。**方差/边界规则是已经得到干预支持的贡献者，既不是全部解释，也不说明“修好sampler就等于训练方法成功”。** 这不拆分首步、末步与中间方差各自因果份额。

![四seed](figures/sampler_multiple_fixed_seeds.png)

![方差单因素对照](figures/sampler_variance_only_control.png)

## 6. 输出究竟变成了什么

同200图G4，clean高频绝对variance约.0016801，Input .0001474，U0 SDE .0014793，Final SDE .0029412；Final约为clean的1.75倍（总体均值比，不是逐图比均值）。Final mean-ODE .0001498仍接近Input。Final SDE的二维Sobel能量.0060816高于clean .0044296；GT边缘MAE从Input .049913、U0 SDE .061737升至Final .082199。结合SSIM下降，说明增加的高频并没有保持正确结构，不能只称锐化成功。

RGB均值漂移较小：Final SDE对GT通道均值漂移RMS约.00738，Input .00591；未见明显全局通道均值偏移；仅凭均值不能排除全局增益或对比度缩放。Final SDE越界像素平均3.03%，旧诊断clamp敏感性约.015dB，远不能解释近3dB下降。各通道mean/std、局部结构proxy、range、边缘和逐图PSD均保留。真正5×5/11×11每RGB reflect窗local std已补齐6400身份行/5700唯一文件，详见[局部contrast审计](data_optimizer/local_contrast/REPORT_ZH.md)。Val G4的11×11 local std均值，GT .081245、Input .030403、U0 SDE .049864、U500 .072941、U10000 .087522、Final SDE .093994、两mean-ODE约.0313/.0319；高局部对比不等于正确纹理。稳定patch fallback保留真实小方差，原1e-11门240点max1.744e-12 PASS；初版消减误差与旧REVIEW留档。

独立DFT统计6400条、5700唯一文件，period2/4/8轴向/对角峰与coherent checker投影均保留。Final SDE总体高频占比升至约4.99%（该脚本边界用radius≥.25），但exact period2轴向能量占比约12.4ppm，未见匹配幅度的全局周期峰增大。现有证据支持广谱高频过量，**没有证明棋盘格是主因，也没有排除局部非平稳伪影**。不要将径向PSD/单个周期proxy当完整伪影检测器。

![原始浮点径向频谱](figures/G4_RADIAL_SPECTRUM.png)

![周期性诊断](figures/periodicity_diagnostics.png)

## 7. 优化器、EMA、参数漂移与预训练功能

完整审计见[OPTIMIZER_AUDIT.md](OPTIMIZER_AUDIT.md)。标准PyTorch Adam，lr1e−5、betas(.9,.99)、eps1e−8、wd0、amsgradFalse，无clip/AMP/GradScaler/accumulation，constant LR。顺序zero_grad→forward→backward→只量测global norm→Adam.step→EMA.update。记录loss的all_reduce发生在backward后，不二次缩放梯度。

7完整checkpoint全部重新SHA；151 RAW/EMA张量及Adam一二阶moment有限，step与U一致，8rank RNG完整。全部8rank45411更新、4359456数据index重放匹配；CPU tau重放363288条min/max和56个CPU RNG状态逐字节一致，各G平均t和c_t接近。未发现“G4没训到”“抽t错了”或恢复时optimizer重置证据。

foreach/fused在checkpoint为None不是single分派证明。当前已安装Torch/NPU源码＋真实NPU Parameter默认helper返回fusedFalse/foreachTrue，指向multi_tensor Adam；历史没有逐kernel profile，不能追认每一步底层路径或历史包从未变动。环境实际import解析来自多个旧venv，版本/路径与旧记录一致，源冻结未覆盖完整依赖；可复现性边界公开保留。

RAW相对U0全网L2：U5001.576%、U10000 7.304%、U20000 10.314%、U30000 12.468%、U40000 14.241%、Final15.094%。151tensor都在500步已变，末端ups模块Final约22%，并非500步后参数饱和。历史global gradnorm均值前500约.001566→末1000约.004648，最大.01605且有限；增大本身不等于爆炸。

Adam实际更新不能用lr×grad代替。保存moment重构含bias correction的分量`-lr*mhat/(sqrt(vhat)+eps)`，Final全网norm约.010908、相对RAW约5.52e−5；它不是不存在的相邻step逐bit参数差，也不是稀疏checkpoint累计delta。每层/模块/组完整统计与moment分布均在子报告CSV。

### 梯度方向与多目标贡献

96个完整FP32梯度约52.7GB远端留存，FP64累加Gram。U0同t G1/G4全部正；U500、U10000个别t为负，Final所测6个t跨G全网cos均正。Final G4 t1vs99约−.1226，说明局部跨时间冲突存在，而非所有任务始终对冲。Final t10等G mixture方向投影G1/G2/G3/G4约54.27/35.61/7.06/3.06%，等样本比例不等优化贡献。四来源六t不足重建训练历史，不能唯一归因“多强度冲突”。

### EMA不是从第1步恒定.995

实际ema-pytorch0.3.2含update_after100、每10步、动态decay warmup。U500 decay约.9816，约U2930后达到.995；后期半衰期约1383optimizer更新=本B96约132768呈现。Final RAW比EMA在旧固定Val G4高约.199dB仍远低于Input；早期RAW/EMA小集另表，不能替代全RAW终评。没有证据把失败主要归因EMA公式错误。

官方SR训练期online验证，本项目主要EMA验证；发布裸gamma权重的原始RAW/EMA身份没有元数据。官方公开recipe B8、128GT/32LR、lr1e−4及900k多段衰减，与当前B96、256、constant1e−5、45411步明显不同；不是唯一backbone变化的受控比较。

### ‘参数变化’已补功能阳性，不直接等于全局灾难遗忘

取得真实DIV2K0801官方HR/LRx4文件对（Range/CRC/SHA），固定x4对齐256crop；另8Places源按官方退化族生成代理。相同四EMA及两sampler共288保存图。实际官方评价函数AST提取，uint8 clamp/round裁4的RGB/Y与clamped float LPIPS另表；原raw float数组不变。

真实0801一crop官方式RGB/Y结果（不是完整benchmark）：

| 权重 | 采样 | Input RGB | Out RGB | Out Y | RGB SSIM | clamped LPIPS↓ |
|---|---|---|---|---|---|---|
| U0_EMA | mean_ode | 30.801602 | 32.818521 | 34.232193 | 0.868769 | 0.253098 |
| U500_EMA | mean_ode | 30.801602 | 32.021002 | 33.421450 | 0.850675 | 0.319095 |
| U10000_EMA | mean_ode | 30.801602 | 30.441654 | 31.813939 | 0.829873 | 0.371846 |
| FINAL_EMA | mean_ode | 30.801602 | 27.305571 | 28.631245 | 0.793344 | 0.345928 |
| U0_EMA | sde | 30.801602 | 27.529572 | 28.911651 | 0.656648 | 0.345667 |
| U500_EMA | sde | 30.801602 | 20.459069 | 21.787628 | 0.454259 | 0.433992 |
| U10000_EMA | sde | 30.801602 | 24.806084 | 26.162009 | 0.713555 | 0.226938 |
| FINAL_EMA | sde | 30.801602 | 23.546550 | 24.873350 | 0.693978 | 0.265807 |

U0 mean-ODE在此crop有真实SR阳性：RGB30.802输入→32.818；随后U50032.021、U10000 30.442、Final27.306，支持**此局部SR功能下降**。8Places代理的mean-ODE也下降（raw Val四源26.967→25.122、train四源26.513→24.313），与参数漂移一致。不能仅凭参数norm宣布全部预训练能力丢失，也不能把一张crop扩展为DIV2K全体灾难遗忘。

官方默认SR test实际用SDE，而原U0 SDE此crop的PSNR也低于Input；这说明checkpoint＋采样协议在此有像素/感知取舍，不能由一个差SDE数直接宣判权重损坏或下载错。Strict load和SHA只证明身份/结构，不证明任务迁移合理。两份公开SR权重gamma1e7/5e5来源ID已核，本次始终使用指定1e7，没有暗换另一份或搜索gamma。

![原任务族保持对照](figures/sampler_SR_domain_controls.png)

## 8. 数据和单帧几何任务的真实难点

10k源×4ξ×G1–4组合身份一致，lowercase g计数各40000；历史uppercase G=4背景字段没有参与错档读取。原JPEG经EXIF/RGB/短边256/中心裁/255冻结预处理重解码10250张，clean像素差0；train/val sourceID与clean SHA无交集，存储pHash+dHash近重复筛查无候选。不存在可见的配对/缩放错误，仍不宣称排除语义近似或全160K物理算子问题。

真实simulator为随g^(5/6)缩放的PSF/tilt、PSF混合后再bilinear warp；G4没有identity混合，含border continuation。全Val1000 rawtilt从CPU种子重放SHA匹配、5000真实flow重建：G4 RMS6.719px，spatialmean位移norm平均6.460px，残差RMS1.582px，平均88.16%flow能量属于空间DC/均值平移分量；此数学分解未分离生成模型的物理global-tilt latent。该量不是低置信phase-correlation proxy。

G4仅保留clean高频能量逐图比均值12.55%、二维Sobel6.91%，颜色总体漂移较小。这是含绝对平移/局部warp/强blur的任务，与bicubic SR域不同；并不能据此证明单帧算子严格不可逆、后验Bayes下界或UniDB架构绝对无能力。

已知真实flow空间DC/均值平移的纯CPU反移oracle，固定32px共同内裁，G4 Input18.158→20.006dB（991/1000更好，来源CI增益[1.751,1.951]），SSIM .4053→.4584；wrong-sign反而16.412。该oracle使用未知物理元数据，非部署D0；crop指标不能与原全图PSNR混比，MSE下降35.45%不是可加的几何因果份额。它表明仍有可利用结构和几何headroom，也显示只校平移后强模糊仍在。

![确定规则选择的已知shift oracle](figures/ORACLE_GLOBAL_SHIFT_FIXED.png)

## 9. 三个最优先解释、反对证据和未解问题

完整H1–H16逐项支持/反对/证伪方式见[HYPOTHESIS_MATRIX.csv](HYPOTHESIS_MATRIX.csv)。下面按证据优先级排列，**不是三个已经独立识别的可加因果份额**：

1. **微调权重×原生随机动力的交互（强支持，干预证据最强）。** 2×2差异、多seed和variance-only均支持，方差规则可挽回Final约1.13dB但未解决任务。不能进一步偷换成已证Jacobian爆炸或“只要ODE就好了”。
2. **一步经验目标与最终结构恢复的有效性分离（已观察到；具体训练目标因果尚未识别）。** 教师态拟合确实更好，最终准确性更差；L1中位数、隐含时间系数、有限步target/采样方差不等提供需要验证的数学候选。共同U0态目标仍略优是重要反证，不能将整个原因浓缩成“分布外一步误差全面恶化”。
3. **SR先验迁移中的功能漂移及几何域差异（局部功能下降有证据；主要因果份额未知）。** 一个真实SR crop及8代理源显示功能保持下降，参数持续变化；真实flow和oracle表明新任务有SR没有的明显几何难度。尚未以独立训练干预区分lr、全参更新、数据几何与目标选择谁是上游原因。

**目前证据最强的解释**是有条件、可复现的权重—随机采样交互加上surrogate目标不预测最终恢复收益；不是已找到唯一loss代码bug。**最容易忽略的实现/协议问题**是官方target与实际有限步采样并非同转移、实际EMA warmup与验证分支、官方LPIPS原地clamp别名、混合环境依赖未全冻结、旧跨rank参数摘要并非全部137M参数相同证明。

**尚不能排除且可能影响较大的因素**：L1与条件均值风险差异、时间贡献分配、未测状态/方向的多步动力、B96/constantLR/全参迁移、全局tilt下条件歧义和SR功能遗忘因果。无受控新训练不能把这些定量归因；本次明确不训练。

**当前证据明确不支持的简化解释**：只因训练量不足；G4没抽到；loss记录假下降；GT/condition反了；训练用了外层递归或感知loss；单纯末步噪声/finitegamma微小误差足以造成近3dB；所测方向普遍更敏感；EMA单独造成全部失败；换Mean-ODE即证明恢复/泛化成功。

### 后续建议

停止延长原配方。未来若继续，按[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)的F或G先做一个至多500更新的可证伪对照：只改LR，或在控制时间系数后只改风险类型；同U0/数据/随机流/网络/采样，设共同质量停止门。H时间系数与I去global tilt分别另做，不同时改。所有短训练仅设计，原run不恢复，也不自动启动新骨干。

## 10. 数值平台、保全、成本与交付限制

CPU原生精确GELU用记录的实际NPU dispersion完整闭环，在固定Val灯塔样本上：U0最终CPU−NPU PSNR差−3.53e−7dB，Final+2.67e−5dB；最终state RMS约4.45e−8和9.58e−7。六个same-state gate在原容差PASS。两个完整B4 backward同状态比较也PASS，Final gradient cos=.999999403、relativeL2=.0010975。不能将旧默认GELU历史FAIL当当前FAIL，也不能由有限点PASS宣称全平台路径永远相同。

![同噪声CPU闭环和NPU原输出](figures/CPU_NPU_FIXED_IMAGES.png)

[ORIGINAL_ASSETS_PRESERVED.json](ORIGINAL_ASSETS_PRESERVED.json)重核17项原资产及7完整checkpoint SHA全匹配；8rank日志各45411连续、无新update。原SR SHA为`a6bb76bfabe32b8403d8b112f0ce7876114d830d20b9f4a4119aa860afef64b4`，Final完整checkpoint SHA为`c59a997659f8b3f63996344cde29f4bab2d5916de35a750ca23eb76f08da3bca`。本地U0/U10000/Final原独立完整备份保留，其余节点与所有大数组在远端。

本次只用空闲0/2/4/6做主前向、3做只读梯度/小探针；没有发信号给外部进程。物理1驱动仍登记PID2938958/2452MB，但当前/proc和ps不可见，**外部服务OS存活无法确认**；不能将driver登记当存活证明或臆断退出原因。原记录和首次观察保留，没有清理该上下文。

原run更新关键段累计11.854h/8卡94.84 NPU小时，原U0到终评流水线17.432h/139.46卡小时是历史训练/验证成本，非本审计新训练。新增成本单列`COST_AUDIT.json`：主前向/FD、多seed与工程恢复的三控制器区间分配代理约3.540卡小时（非纯核时）；只读梯度约.467卡小时；674条件/teacher前向进程约150s/.0417卡小时；206次CPU闭环前向约965s、上限4.29线程小时；CPU LPIPS/FFT/真flow/报告另计，墙时含IO/等待，不能当能耗/计费或纯核时。56800个batch4恢复前向与12800个batch1FD前向是成功提交科学任务的可重建调用数，不包括工程写文件失败前未提交SR调用，不能伪称精确总计算量。

本次自身工程错误均在新audit中留原源码/日志/失败证据：CPUprepare/sample_id含slash、空rank汇总门、SR路径写入、CPU汇总兼容、Sobel跨channel错误、补图输入缺失/undefined比值解析。只修工程与报告并复核；完整已提交预测/梯度复用，未改原训练定义或重跑原终评。两次补图失败没有模型调用，Sobel和local std数值消减修复仅重算CPU保存图统计，原严格门未放宽。

交付文件入口：

- [OFFICIAL_VS_CURRENT.md](OFFICIAL_VS_CURRENT.md)：官方/当前逐项差异与运行环境。
- [MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md)：完整独立概率推导/代码映射/反例。
- [OPTIMIZER_AUDIT.md](OPTIMIZER_AUDIT.md)：Adam/DDP/梯度/参数/EMA与所有限定。
- [SAMPLER_AUDIT.md](SAMPLER_AUDIT.md)：全部轨迹、seed、FD、variance、SR和CPU评价。
- [HYPOTHESIS_MATRIX.csv](HYPOTHESIS_MATRIX.csv)、[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。
- [AUDIT_MANIFEST.json](AUDIT_MANIFEST.json)：本地文件SHA、原资产/数据/7checkpoint/环境与远端大数组索引；不把仅远端数组假称本地备份。
- [FINAL_ACCEPTANCE.json](FINAL_ACCEPTANCE.json)：实际PID/start_ticks退出、完整计数、图QA、保全及交付复核。图像便利副本在`figures/`，每图数据/生成脚本/来源映射在`figures/FIGURE_INDEX.json`；原子报告/原始CSV/JSON/程序均保留。

边界：没有完整外部真实湍流测试；没有遍历所有采样seed；没有全部历史相邻参数/逐层梯度/每rank全参数快照；没有完整160K物理重render；一个SRcrop不是完整DIV2K；数学缺口和实测交互不自动给唯一根因。已确认这套直接SR全参迁移recipe没有达到本次强湍流恢复目标，原因已收窄到可检验的候选，而非以“再多训一点”替代验证。

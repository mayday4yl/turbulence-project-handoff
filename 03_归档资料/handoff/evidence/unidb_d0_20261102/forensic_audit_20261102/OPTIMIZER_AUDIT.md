# UniDB D0：优化器、EMA、参数漂移与完整梯度审计

本报告整合完整的优化器/训练日志审计、96次固定只读梯度诊断、两点CPU/NPU完整backward及Adam实际默认分派调查。审计对象为已停止的单帧D0，未恢复训练、未构造optimizer或调用step、未修改原权重/实现/数据。只读autograd不能被称为历史训练梯度；当前dispatcher验证不能冒充历史profiler。数据身份、真flow和oracle见[数据子报告](<../../../NOT_INCLUDED.md#ref-3d1d4723d37ecabeb3e7>)，loss数学及CPU timestep RNG重建见[数学报告](<../../../NOT_INCLUDED.md#ref-661930e1e084a20847ef>)，SR保持性能与采样对照见[采样报告](<../../../NOT_INCLUDED.md#ref-53d39ebc22dcbdd82979>)。

以下完整保留三份独立子报告的公式、数值、反例、源代码行号和覆盖限制。原子报告未改动；此顶层整合仅为文档操作，没有重跑任何模型。

## 1. UniDB D0：优化器、checkpoint和EMA独立审计

原子来源：[OPTIMIZER_AUDIT_ZH.md](<../../../NOT_INCLUDED.md#ref-8849ab56f3e97a57fc61>)。

审计对象为已安全停在U45411的独立D0。以下是代码、真实7份完整checkpoint和全部8rank日志的证据；本次未恢复训练、未调用optimizer.step、未修改原文件。新增梯度仅为另立协议的固定状态只读autograd，详见[梯度报告](<../../../NOT_INCLUDED.md#ref-1eff8c1939978e35b0f4>)。

### 结论及排除边界

未发现Adam step错位、学习率漏恢复、NaN/Inf、重复数据位置、loss又除一次world_size或误用AdamW的证据。该结论不等于超参数适合单帧湍流；小学习率也不代表45411步后的参数变化很小。RAW相对SR初始化的全网L2变化达到15.094%，且所有151参数tensor在U500就已变化。总loss下降期间，全局梯度norm逐步上升；二者可同时出现，单凭此不能认定爆炸或loss错误。

历史没有逐层/逐强度梯度，不能声称重建了历史梯度冲突。固定新探针发现部分早期状态存在负cos，但Final同t的G1–G4整体cos全为正，是“始终多档互相冲突”的反例。Final t10等权梯度mix中G4方向投影仅3.06%，G1为54.27%；均衡呈现量不等于方向贡献均衡。

### 真实代码调用链

- `worker.py:159–166`：初始化原网络→NativeObjective→标准`torch.optim.Adam`→EMA；Adam lr=1e-5、betas=(.9,.99)、weight_decay=0。不是AdamW。
- `worker.py:160–164`和实际CONFIG：FP32、HF32=False、microbatch=12、accumulation=1、world=8、effective_batch=96、无梯度裁剪；保存states同时验证这些配置。
- `worker.py:182`：DDP包装整个NativeObjective。`core.py:108–125`使用原MatchingLoss L1；每rank12样本及像素mean，DDP再平均8rank，得到96样本mean；`worker.py:211–212`仅按accumulation除一次。accumulation=1时不再缩放。
- `worker.py:203–219`：zero_grad→forward→backward→get_total_norm→Adam.step→EMA.update。`get_total_norm`只量测，无clip。loss日志的all_reduce/world在step之后，仅用于记录，未再次进入backward。
- `worker.py:137–145`：恢复RAW、optimizer、EMA、各rank RNG、update/cursor/counts；`111–134`在完整更新边界原子落盘/fsync/replace。无AMP，checkpoint scaler=None。
- `data.py:20–46,56–63`：确定性遍历，直接加载单张RGB clean/observation；没有物理递归、A/B接口、task prompt或xi输入。

### Adam foreach/fused：None不等于single dispatch

checkpoint保存foreach=None、fused=None仅是未显式指定。独立导出当前Torch2.7.1+cpu/torch-npu2.7.1.post2实际Adam类、functional adam、默认helper和supported-device函数源码，完整SHA见`adam_dispatch/SOURCE_AUDIT.json`。torch_npu注册privateuse backend alias为npu，foreach supported devices实际为[cuda,xpu,npu]。

functional Adam在两个参数均None时调用_default_to_fused_or_foreach(params,differentiable=False,use_fused=False)。真实物理NPU3上一元素FP32 torch.nn.Parameter只读helper实际返回(fused=False,foreach=True)；已知lr为float，不触发Tensor-lr且capturable=False的禁用foreach例外。因此依当前已安装源码和已知真实参数类型，当前有效选择为`_multi_tensor_adam`，不是从保存None臆测single。未构造optimizer、未调用step/NN，独立协议及actor退出见adam_dispatch目录。

该结果是当前源码＋真实NPU参数helper验证，不是历史optimizer profiler，不能追认训练每步底层kernel实际路径或历史package从未变化。数学Adam分量与真实多tensor实现的FP32运算/参数相减可能有舍入差；所保存moment重构量仍不能冒充实际相邻step逐bit差值。详见[Adam分派补充](<../../../NOT_INCLUDED.md#ref-4624569009a2b05ae8e0>)。

### 完整checkpoint独立CPU读取

每份文件重新SHA，读取所有RAW/EMA张量和optimizer一/二阶moment。全部151tensor、137147523参数有限；U0 optimizer fresh为空，其余151state的step全等于U。各checkpoint均包含8rank RNG、constant LR1e-5、scaler=None。

| U | 呈现量 | Adam step | RAW相对U0 L2 | EMA相对U0 L2 | RAW−EMA/RAW |
|---|---|---|---|---|---|
| 0 | 0 | fresh | 0.000000 | 0.000000 | 0.000000 |
| 500 | 48000 | 500.0 | 0.015763 | 0.010687 | 0.006845 |
| 10000 | 960000 | 10000.0 | 0.073043 | 0.063418 | 0.013370 |
| 20000 | 1920000 | 20000.0 | 0.103143 | 0.097544 | 0.008914 |
| 30000 | 2880000 | 30000.0 | 0.124682 | 0.120525 | 0.007157 |
| 40000 | 3840000 | 40000.0 | 0.142413 | 0.138918 | 0.006397 |
| 45411 | 4359456 | 45411.0 | 0.150938 | 0.147734 | 0.006032 |

逐tensor量纲不同，下表是151个tensor各自`||theta_U-theta_0||/||theta_0||`的无权重分位数，不能与全网比值混淆。

| U | 逐tensor漂移中位数 | p95 | 最大值 | 保存moment的Adam分量L2 | 分量L2/RAW L2 |
|---|---|---|---|---|---|
| 500 | 0.012875 | 0.031669 | 0.056112 | 0.008498 | 0.0000431 |
| 10000 | 0.048925 | 0.136483 | 0.195064 | 0.008016 | 0.0000406 |
| 20000 | 0.058169 | 0.182900 | 0.255947 | 0.010610 | 0.0000538 |
| 30000 | 0.064922 | 0.222762 | 0.306343 | 0.010969 | 0.0000556 |
| 40000 | 0.072836 | 0.261469 | 0.346085 | 0.010733 | 0.0000543 |
| 45411 | 0.075373 | 0.275250 | 0.360981 | 0.010908 | 0.0000552 |

保存moment分量按实际step及beta计算：`d = -lr * (m/(1-beta1**step)) / (sqrt(v/(1-beta2**step))+eps)`，eps=1e-8、weight_decay=0。它是该已完成step保存的累计moment所隐含的数学Adam分量；不是“当前gradient×lr”，不是稀疏checkpoint相减，也不是下一步新更新。实现舍入、foreach路径和参数相减的舍入可能使它不等于实际最后一步bitwise差值。U0无moment，不编造此量。

Final模块相对U0漂移：ups.3约22.35%、ups.2约21.85%、downs.0约17.27%；mid_attn约0.2%。mid_attn内部norm.g相对变化9.76e-7，但固定探针该层梯度非零；不能只凭小漂移判“死层”。本子任务未自行执行SR保持性能评估；独立runtime另设SR代理与官方DIV2K0801固定crop，实际完成范围和结果见[主采样审计](<../../../NOT_INCLUDED.md#ref-53d39ebc22dcbdd82979>)。15%漂移本身不能单独证明catastrophic forgetting。

证据：`CHECKPOINT_AUDIT.json`、`PARAMETER_DELTAS.csv`（1057行）、`MODULE_DELTAS.csv`、`ADAM_MOMENTS.csv`（906行）、`PARAMETER_ADAM_SUMMARY.json`。

### 全部真实训练日志及数据顺序

独立使用冻结`data.Order`重放所有4,359,456个8rank数据position/index；全部匹配。每rank45411条，日志LR唯一值1e-5、loss/全局gradnorm全部有限、tau_min/max均在1…99。8rank记录的全局gradnorm每步相同。最终g呈现量：G1=1,089,885，G2=1,089,834，G3=1,089,892，G4=1,089,845。相当于160K pool的27.2466次呈现；不是435万独立source/xi。

| 实际更新区间 | mean MatchingLoss | mean全局梯度norm | max全局梯度norm |
|---|---|---|---|
| 1–500 | 0.001256842 | 0.001566466 | 0.011468390 |
| 501–1000 | 0.001141577 | 0.001342137 | 0.004637970 |
| 9001–10000 | 0.000966608 | 0.003070430 | 0.009056781 |
| 19001–20000 | 0.000901976 | 0.003766452 | 0.015449101 |
| 29001–30000 | 0.000868783 | 0.004411665 | 0.016052131 |
| 39001–40000 | 0.000836382 | 0.004583832 | 0.010417181 |
| 44412–45411 | 0.000826733 | 0.004647503 | 0.014210884 |

全历史gradnorm均值.00373066、max .01605213、p99 .00922406。未出现日志级NaN或无限增长，但这不足以排除局部模块问题或错误采样轨迹。原日志只记录每rank每step的tau_min/max；单凭这些摘要不能反推逐图t或方向。独立math审计已从U0的CPU Torch RNG重放全部4,359,456个tau，全部日志min/max吻合，且7checkpoint×8rank=56个CPU RNG节点逐字节匹配；完整重建histogram见[数学审计](<../../../NOT_INCLUDED.md#ref-661930e1e084a20847ef>)和math/CPU_TAU_REPLAY_AUDIT.json。该强校验RNG重建仍须与原始逐图观测区分，且不重建NPU噪声或历史逐t loss/gradient。本次固定96探针不冒充这些历史量。

### EMA真实defaults与时间单位

当前安装`ema-pytorch==0.3.2`源码已独立导出到`../official/EMA_INSTALLED.py`。`core.py:128–129`仅指定beta=.995和update_every=10，实际默认还包括update_after_step=100、inv_gamma=1、power=2/3、min_value=0、karras_beta=False。update()依据调用前old step每10次才处理；old step≤100复制online，之后initted和decay ramp，最终cap .995。U500并非从开始一直固定.995；此时decay约.9816，到约U2930才达到cap。

后期.995每10optimizer update一次的半衰期约1383次optimizer update，即本实验batch96约132,768次呈现。官方SR batch8的相同步数约11,064次呈现，不能把EMA步数单位与图像呈现量混用。RAW–EMA差从U10000的1.337%到Final的.603%，但不是质量改进量。

当前checkpoint序列化EMA network/step/initted/beta/update_every/update_after_step，未序列化package所有defaults。官方requirements未锁定原作者当年的EMA版本，不能声称现有defaults就是其历史defaults。官方训练期`DenoisingModel.test`验证路径用online `self.model`，本项目evaluate默认EMA，这是验证recipe差异，不等价于EMA实现有bug。发布gamma_1e7裸state_dict缺少RAW/EMA身份元数据，不能从self.model属性倒推发布文件一定属于RAW或EMA。

### 验收措辞应纠正的证据覆盖

`core.py:162–168`的parameter_digest默认full=False：每tensor>32时仅取前/后16值；`worker.py:114–115,260`据此断言rank一致。因此旧`rank_parameters_equal=True`仅证明跨rank抽样digest一致，不能证明137M每元素完全相等。本次对保留的rank0完整state逐元素做finite/漂移检查，固定梯度前后使用full=True全参数digest；没有补造历史8份rank参数快照。

冻结RUN_SOURCE和INITIAL_EXECUTION_SOURCES.zip未覆盖run_env.sh、安装EMA源码、外部旧V5 image_metrics.py，存在环境代码冻结覆盖缺口。现在导出当前源码可提高可复查性，无法追溯性地保证历史包绝未变化。

[图像原位置：训练和优化动态](<../../../NOT_INCLUDED.md#ref-ab8ea00b055072814353>)

[图像原位置：模块漂移](<../../../NOT_INCLUDED.md#ref-8908e248d6f7923f4b4a>)

### 限制

此项审计支持“有限、顺序一致、公式为标准Adam/原生matching L1的真实优化确实发生”，不支持“科学目标一定合适”“多训就会好”或“全部实现/平台在全部状态完全正确”。本子任务没有自行完成SR评估，需结合独立SAMPLER审计；没有历史逐层grad、逐图原始tau观测或8rank全参数历史快照。历史tau的独立CPU RNG重建已通过56节点校验，不能称为缺失；它也不能重建历史逐层梯度。固定新探针的范围与数值对照必须单独标注。

## 2. UniDB D0：固定只读梯度、方向冲突和CPU/NPU完整backward对照

原子来源：[GRADIENT_AUDIT_ZH.md](<../../../NOT_INCLUDED.md#ref-1eff8c1939978e35b0f4>)。

### 范围与权重不可变保证

这是新增固定诊断，不是历史训练gradient。RAW U0/U500/U10000/U45411各24探针（G1…G4×t=[1,10,25,50,75,99]），共96 forward+96 backward，每次4张256²RGB。未构造optimizer，未调用step；各checkpoint前后full=True全137147523参数digest完全一致，保留于每个`WEIGHTS_UNCHANGED.json`。原权重、实现和数据未写入；STOP_REQUEST和NO_TRAINING_RESUME仍需存在才允许worker工作。

独立按固定SHA选4个不同category，再每category独立SHA选train source，取首个xi。未按模型输出或诊断结果选择。跨g同clean/source/xi，eps按`derive(921113,source_id,sample_id,t,GRAD_EPS)`，有意排除g/checkpoint；不同t噪声也被固定，可复算。

| category | source | xi |
|---|---|---|
| 341 | places365:train_standard:/t/tundra/00003018.jpg | v6_train_s09585_r00 |
| 49 | places365:train_standard:/b/beach_house/00000375.jpg | v6_train_s04180_r00 |
| 361 | Places365_val_00003715.jpg | train_s1472_r00 |
| 156 | places365:train_standard:/g/garage/indoor/00002082.jpg | v6_train_s06532_r00 |

冻结协议SHA：`9e0436b5fac36bcae0bb7d443e2f18a617f6d4e7d917e1ed5b334e8b33bfe6d8`，原始脚本SHA见`gradient_probes/PROTOCOL.json`。用原生GT-forward状态`x_t=f_mean(clean,t)+f_sigma(t)*eps`，原生network/state/y/t、原score换算、reverse_sde_step_mean和reverse_optimum_step，再原MatchingLoss('l1',False)，仅控制t/eps而不改变目标。

### 完整保存和定义

每个探针保留全部137147523元素FP32无损`.npy`，96份约52.665GB十进制；每份SHA在JSON和GLOBAL_GRADIENTS.csv，远端保留、本地只同步小统计。参数layout含name/shape/start/end/module。Gram/cos从全部元素分块FP64矩阵乘法累计，不是梯度采样或压缩近似。每checkpoint完整24×24 Gram和cos（有moment分量的为25×25）均保留。

统计包括global、16模块、151参数tensor的L2、mean/std/min/max、exactzero比例、abs<1e-12/1e-9/1e-6和abs>1e-4/1e-3/1e-2比例；阈值在运行前冻结，仅为描述性量。完整文件：`GLOBAL_GRADIENTS.csv`96行、`LAYER_GRADIENTS.csv`14496行、`MODULE_GRADIENTS.csv`1536行、`MODULE_STRENGTH_COSINES.csv`384行。

所有96探针grad有限，全部14496个tensor-probe的gradient L2均非零；全网exactzero fraction最大1.46e-8。U0 norm范围.003867… .081432，U500 .000636… .007094，U10000 .000534… .045833，Final .002106… .046612。少数norm-gain层norm到1e-11量级，需考虑参数化和损失尺度，不能只凭某层“小”判死层或以一组大值判爆炸。

### 同t跨强度方向：G1对G4

| RAW U / t | 1 | 10 | 25 | 50 | 75 | 99 |
|---|---|---|---|---|---|---|
| 0 | 0.847117 | 0.875766 | 0.898153 | 0.795474 | 0.696735 | 0.996697 |
| 500 | 0.586817 | -0.018307 | 0.341775 | 0.056553 | -0.278213 | 0.276067 |
| 10000 | 0.507355 | -0.093196 | -0.069006 | 0.644825 | 0.377510 | 0.985825 |
| 45411 | 0.661860 | 0.049718 | 0.681254 | 0.840194 | 0.498164 | 0.991191 |

### 同t跨强度方向：G2对G4

| RAW U / t | 1 | 10 | 25 | 50 | 75 | 99 |
|---|---|---|---|---|---|---|
| 0 | 0.967163 | 0.977933 | 0.984139 | 0.986638 | 0.882046 | 0.998954 |
| 500 | 0.851179 | 0.858948 | 0.867458 | 0.687105 | 0.205153 | 0.632526 |
| 10000 | 0.780044 | -0.150406 | 0.141436 | 0.497712 | 0.708385 | 0.992900 |
| 45411 | 0.957861 | 0.079048 | 0.824077 | 0.937415 | 0.587524 | 0.995626 |

负cos表示该固定batch/state下局部目标方向冲突。U500 G1vsG4在t75为−.278213，U10000在t10/t25为−.093196/−.069006；U10000 G2vsG4在t10为−.150406。Final全部同t整体cos为正，其中t10仅.049718/.07905，接近正交但不能写为负冲突。Final t99强度间近乎同向。故不能把“存在冲突”扩大为“最终所有g始终互斥”或全训练主因。

逐模块揭示局部反例：U500 t75 G1vsG4：downs.0 −.905、mid_attn −.846、time_mlp.1 −.786；Final t10整体正但downs.0 −.012、final_conv −.037，其余多为弱正。模块冲突与全局加权聚合不可混用。

### 跨t方向与梯度贡献

G4 t1vs99：U0 .003271、U500 −.028072、U10000 .048163、Final −.122633。Final t50vs75=.895826，但t50vs99=.015560；早晚t方向差异不能由同t强度一致性替代。

不能把norm占比当方向贡献。对每个固定t定义`g_mix=.25*sum_g g_g`，投影贡献`dot(.25*g_g,g_mix)/||g_mix||²`，四项之和1（一般可为负），由完整Gram直接计算。Final：

| t | G1 | G2 | G3 | G4 |
|---|---|---|---|---|
| 1 | 17.14% | 31.86% | 29.48% | 21.52% |
| 10 | 54.27% | 35.61% | 7.06% | 3.06% |
| 25 | 30.63% | 31.92% | 25.73% | 11.71% |
| 50 | 18.72% | 29.68% | 28.79% | 22.82% |
| 75 | 29.33% | 28.61% | 24.45% | 17.61% |
| 99 | 24.62% | 25.66% | 25.69% | 24.03% |

U0 t1 G4贡献41.36%，Final t10却只有3.06%。该受控状态下轻档占主导有证据，“始终被G4主导”有明确反例；仍不能从4source×6t估计历史真实全分布贡献。Final G4 t99 loss .0001166而gradnorm .043676，t1 loss .0012101而norm .002223，说明loss标量大小不直接等于参数方向或梯度大小。

保存Adam累计m/v重构分量也作为完整向量加入cos矩阵；它采用该已完成step的真实bias correction，无新step。该向量与新探针gradient比较描述历史累计方向与受控局部方向，不能说是真实下步更新或邻checkpoint逐步差。细节见[优化报告](<../../../NOT_INCLUDED.md#ref-8849ab56f3e97a57fc61>)。

### CPU/NPU完整backward对照

运行前另冻结`cpu_backward_parity/PROTOCOL.json`，RAW U0/Final、G4/t50、与NPU完全相同batch4/eps。CPU16线程，官方原始exact nn.GELU，不使用NPU erf patch；NPU使用本实验等价erf exact GELU修复。原native loss容差atol2e-5、rtol2e-4；梯度描述性阈值cos≥.999且relativeL2≤.01。L1在残差零点附近不光滑，触发REVIEW要定位差异，不能事后放宽阈值。

| RAW U | CPU loss | NPU loss | 全梯度cos | relativeL2 | max abs |
|---|---|---|---|---|---|
| 0 | 0.0025404014159 | 0.00254040537402 | 0.9999999880 | 0.00015539 | 4.5628985e-07 |
| 45411 | 0.00109408213757 | 0.0010940770153 | 0.9999994029 | 0.00109748 | 4.2078318e-06 |

两点均PASS；loss误差仅3.96e-9/5.12e-9。全CPU和NPU梯度逐元素可复查，各层比较在`cpu_backward_parity/LAYER_COMPARISON.csv`。这支持这两个真实256²状态前后向数值一致，不能证明全部历史梯度、所有t或整个100步采样轨迹数值无误。

### 资源、退出与可复现

NPU物理3单卡持有共享AB_NPU_3锁、实际npu-smi占用guard，避开已有物理1服务和root其他卡。96探针peak allocated约10.01GiB、reserved约11.84GiB；无optimizer缓存。NPU worker PID622036/start591216415，退出0且同PID身份已不存在；锁实际可重新非阻塞获取/释放后交还root。全部自有CPU/控制器退出详`OWN_ACTORS_EXIT_AUDIT.json`，不会将root后续NPU3诊断误认为本worker。

重现顺序与授权边界：`gradient_probe.py --prepare`冻结协议；已运行证据目录不可覆盖，需另立新目录才可重现。`--run`只读autograd，`--gram`纯CPU完整方向计算；`cpu_backward_parity.py`独立CPU原GELU对照；`gradient_postprocess.py`及`render_audit_figures.py --gradients`只读现存梯度。当前没有未完成probe、没有训练更新、没有自有worker。

[图像原位置：固定全网梯度norm](<../../../NOT_INCLUDED.md#ref-78a4eb9485ba6074801a>)

[图像原位置：跨强度方向](<../../../NOT_INCLUDED.md#ref-14c927447b023cabc9d1>)

[图像原位置：完整固定探针cos矩阵](<../../../NOT_INCLUDED.md#ref-3114bae9195cfe391529>)

### 归因边界

固定控制能证明某些gradient局部方向会冲突、t方向不同、均衡样本不等于均衡参数贡献，也提供反例排除过强断言。它没有重演训练，没有构造新的恢复方法，没有证实这些因素是最终PSNR恶化的唯一或主要原因。原生loss公式的数学审计、RAW/EMA/SDE/meanODE采样和teacher/free-running轨迹证据由其他独立子报告处理。

## 3. Adam foreach/fused实际默认分派补充

原子来源：[ADAM_DISPATCH_AUDIT_ZH.md](<../../../NOT_INCLUDED.md#ref-4624569009a2b05ae8e0>)。

保存参数`foreach=None,fused=None`不说明未使用foreach。当前实际环境为Torch2.7.1+cpu、torch-npu2.7.1.post2，torch_npu已将privateuse1后端注册为`npu`。`_get_foreach_kernels_supported_devices()`当前返回`['cuda','xpu','npu']`，支持类型包含`torch.nn.Parameter`。

独立保存源码与SHA：

- `adam_dispatch/Adam_class.py`，类源码SHA`faea7d447b970d29d15ef6794ff962cb545f82a5bf6d767ee21c7953d8a6f0f4`。
- `adam_dispatch/functional_adam.py:36–46`，两个None时调用`_default_to_fused_or_foreach(...,use_fused=False)`；lr若为Tensor且capturable=False才额外关闭foreach。本实验lr为Python float，differentiable=False、普通非TorchScript执行。
- 同文件`:62–67`，effective foreach=True进入`_multi_tensor_adam`。functional源码SHA`cfe73b1808fe0d354a86bc5dafce06831ed90bced0189f79bae8cd1e32085fd4`；当前完整torch/optim/adam.py文件SHA`78a0aa4a6b182fd586e40f964409cf3a9e2b935a1f8dcb62f31d1a4ca4c2e59a`。
- `adam_dispatch/default_to_fused_or_foreach.py`完整函数SHA`2867120ebdb0605d0726bc8fe1986ecb8f006f87cffa4b628ea9b8e868fea0ee`；其torch/optim/optimizer.py文件SHA`2381e51f3e02b018a6f29bf811656d93cc980bbb387ba868706a16519ea2d906`。

实际一元素探针在物理NPU3：FP32 `torch.nn.Parameter`、differentiable=False、use_fused=False，**只调用helper，返回(False,True)**，即fused=False、foreach=True。结果与当前源码的完整推断一致，见`adam_dispatch/NPU_PARAMETER_HELPER.json`。没有构造optimizer，没有调用optimizer.step或functional Adam，没有NN/forward/backward。

该探针在运行前冻结`adam_dispatch/PROTOCOL.json`，脚本SHA`55cf9889f58850442cf1b555423832139d352b9b3f621836a46410bbce684005`。先取共享AB_NPU_3独占锁，真实npu-smi确认物理3无外部进程才分配一元素；既有物理1服务未动。worker PID851920/start591633086已退出0、真实同身份gone且锁释放，见WORKER_LAUNCH/WORKER_EXIT/EXIT_AUDIT。

可支持的结论是：依当前安装源码、真实NPU参数和已知lr/differentiable参数，当前默认有效dispatch为multi-tensor foreach。**没有历史profiler capture**，不能追认每个历史step实际底层kernel或保证历史package完全不变。保存moment数学重构的Adam分量与真实multi-tensor运算/parameter subtraction可能有FP32舍入差，因此仍不能称为最后一步逐bit实测差值。

此探针为独立新增文件，没有改变已冻结的96梯度或CPU backward脚本，也没有重跑模型。

## 4. 可直接复查的完整证据

所有链接均指向同一独立audit目录中的现有小文件。96份完整FP32梯度、3份Adam分量、2份CPU完整梯度和5000真flow留在远端；此报告不复制这些大数组，逐份远端路径与SHA在相应CSV/JSON。

| 文件 | 内容 |
|---|---|
| [CHECKPOINT_AUDIT.json](<../../../NOT_INCLUDED.md#ref-b39573fd6e10434604b9>) | 7份完整checkpoint、RAW/EMA/Adam state及身份 |
| [PARAMETER_DELTAS.csv](<../../../NOT_INCLUDED.md#ref-44992695b1db340f938e>) | 151参数tensor×7节点的参数统计与相对漂移 |
| [MODULE_DELTAS.csv](<../../../NOT_INCLUDED.md#ref-b1548059550bf0d6912c>) | 模块参数漂移 |
| [ADAM_MOMENTS.csv](<../../../NOT_INCLUDED.md#ref-399d67060ca6eee85147>) | 每参数保存m/v和bias-corrected Adam分量 |
| [PARAMETER_ADAM_SUMMARY.json](<../../../NOT_INCLUDED.md#ref-af1cbdbd593f71efed69>) | 逐tensor漂移分位数和全网Adam分量 |
| [TRAINING_LOG_AUDIT.json](<../../../NOT_INCLUDED.md#ref-84176cfcecff9f7682e4>) | 全部8rank日志和4,359,456呈现index复核 |
| [gradient_probes/PROTOCOL.json](<../../../NOT_INCLUDED.md#ref-b8adef33b2fab120efc3>) | 96只读梯度的冻结源、checkpoint和固定来源 |
| [gradient_probes/GLOBAL_GRADIENTS.csv](<../../../NOT_INCLUDED.md#ref-a312263337bc8e034294>) | 96完整梯度SHA、远端路径和全网统计 |
| [gradient_probes/LAYER_GRADIENTS.csv](<../../../NOT_INCLUDED.md#ref-e6e9520e06a8173cd755>) | 14496条逐参数梯度统计 |
| [gradient_probes/MODULE_GRADIENTS.csv](<../../../NOT_INCLUDED.md#ref-10bec0cdfa89541fa255>) | 1536条模块梯度统计 |
| [gradient_probes/MODULE_STRENGTH_COSINES.csv](<../../../NOT_INCLUDED.md#ref-d4fef04b44957c0ddd2c>) | 384条模块G1/G4、G2/G4方向比较 |
| [gradient_probes/EQUAL_STRENGTH_MIX_PROJECTIONS.csv](<../../../NOT_INCLUDED.md#ref-89dfa8d1fd6fbd7c0113>) | 完整Gram导出的四强度投影贡献 |
| [gradient_probes/PARAMETER_LAYOUT.json](<../../../NOT_INCLUDED.md#ref-137aef1eb7db69a3cf46>) | 完整梯度向量的name/shape/offset/module映射 |
| [gradient_probes/SUMMARY.json](<../../../NOT_INCLUDED.md#ref-9d6968a9f8e1d81ec60e>) | 四checkpoint完整Gram/cos及同t强度摘要 |
| [cpu_backward_parity/PROTOCOL.json](<../../../NOT_INCLUDED.md#ref-fed8cf385e4a01b687c6>) | CPU原GELU与NPU完整backward冻结准则 |
| [cpu_backward_parity/SUMMARY.json](<../../../NOT_INCLUDED.md#ref-c290331708fbecf5d839>) | 两点完整backward loss/梯度对照及SHA |
| [cpu_backward_parity/LAYER_COMPARISON.csv](<../../../NOT_INCLUDED.md#ref-f07f4a8672c07873d249>) | CPU/NPU逐参数梯度差异 |
| [adam_dispatch/SOURCE_AUDIT.json](<../../../NOT_INCLUDED.md#ref-5f7b21fc1e4bc6ed2b61>) | 当前Torch Adam/dispatcher实际源码SHA |
| [adam_dispatch/PROTOCOL.json](<../../../NOT_INCLUDED.md#ref-655252197785ec0b17fc>) | NPU单元素helper冻结边界 |
| [adam_dispatch/NPU_PARAMETER_HELPER.json](<../../../NOT_INCLUDED.md#ref-cde7685c9b1658b488b6>) | 真实NPU Parameter有效foreach/fused结果 |
| [adam_dispatch/EXIT_AUDIT.json](<../../../NOT_INCLUDED.md#ref-b21e4f8b8376723fcd24>) | 小tensorhelper退出与锁释放 |
| [AUDIT_RESOURCE_SUMMARY.json](<../../../NOT_INCLUDED.md#ref-225d173c4eb8fd745b8f>) | 只读诊断资源消耗 |
| [OWN_ACTORS_EXIT_AUDIT.json](<../../../NOT_INCLUDED.md#ref-d37fe82cff23a89c80f9>) | 所有自有CPU/NPU actor真实退出 |
| [FIGURE_QA.json](<../../../NOT_INCLUDED.md#ref-95615ce4ffe6fe9b3659>) | 8图实际目视和图SHA |
| [SUBAUDIT_ACCEPTANCE.json](<../../../NOT_INCLUDED.md#ref-f82805054293539a86f9>) | data/optimizer子审计验收边界 |

四个checkpoint的完整cos矩阵：

- [RAW U0 COSINE.csv](<../../../NOT_INCLUDED.md#ref-12b360b3aad16cbb0d25>)
- [RAW U500 COSINE.csv](<../../../NOT_INCLUDED.md#ref-60c88501f68319969e01>)
- [RAW U10000 COSINE.csv](<../../../NOT_INCLUDED.md#ref-bc88fadba59805f73ad4>)
- [RAW U45411 COSINE.csv](<../../../NOT_INCLUDED.md#ref-f4e18cc4c7a99d12f92c>)

## 5. 文档整合身份

本节仅记录整合所读取的三份原子报告SHA；不是新模型验收。

| 原子报告 | SHA256 |
|---|---|
| [OPTIMIZER_AUDIT_ZH.md](<../../../NOT_INCLUDED.md#ref-8849ab56f3e97a57fc61>) | `060599f7892b558f5795be4097c1870ee4364bcc3b92f953c5e15de35dbe4157` |
| [GRADIENT_AUDIT_ZH.md](<../../../NOT_INCLUDED.md#ref-1eff8c1939978e35b0f4>) | `13a9f5a7251e9da21782fa3c4f95dd62402844c63ffbadf3a6bd23645ffb5fa8` |
| [ADAM_DISPATCH_AUDIT_ZH.md](<../../../NOT_INCLUDED.md#ref-4624569009a2b05ae8e0>) | `515b246129964115eaaa9e6af628ca194ca36fb41fe94df3bc88aabe715e2f73` |


## 用户9.7五项分别判断（综合审计补充）

|问题|回答与证据边界|
|---|---|
|A Optimizer实现是否正确|所查标准Adam构造、DDP缩放/顺序、全部step/state/有限性、保存恢复没有发现错误；不是所有历史kernel/8rank全参数的无限范围证明。|
|B 是否与官方训练recipe一致|不一致。betas/wd/原生loss相同，但B96而非8、lr1e-5 constant而非公开lr1e-4/900k衰减，256输入及EMA主验证也不同。发布权重的全部历史recipe不由公开YAML或strict load证明。|
|C 是否适合从SR迁移本任务|未验证最优性；当前全参配方明确未实现强档恢复。现有正确性检查不能替代同U0单LR或其他单变量短对照。|
|D 梯度冲突/漂移是否有证据|有：151tensor早期均变、Final相对U0全网15.094%漂移，若干G/t探针负cos；但Final同t各G整体cos均正，局部冲突不能扩成全局主因，漂移不等于爆炸。|
|E Optimizer主要问题还是放大目标分歧|主要因果贡献未识别。它实际降低指定matching risk并持续改变参数；方差干预、局部SR功能下降和教师/终态分离提示优化可放大功能漂移，但没有独立optimizer训练对照可把它排为主要根因。|

后续单LR与风险类型控制详见[EXPERIMENT_PLAN.md](<../../../NOT_INCLUDED.md#ref-c4af2d6ac1487a47f49e>)。本补充只整合完成证据，未新建optimizer或执行更新。

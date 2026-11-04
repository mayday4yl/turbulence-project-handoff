# 官方实现与当前实际执行差异审计

上游：`UniDB-SOC/UniDB@1bb63e318891b932c6e641b98729643e46b91e06`。本表依据独立取回同commit Git树与SR文件、实际执行源及日志。此子审计未启动模型推理或训练；仅读取源文件、下载公开原件、核Git对象与做一个CPU张量工具函数探针。最初矩阵保留为REPORT_FIRSTPASS_ZH.md，以下为补证后的修订。

六个vendor文件Git blob均与live commit树一致；这只证明忠实源码，不证明数学或任务适用性。

|功能|官方实现|当前实现|差异|是否影响数学/实验结果|证据|
|---|---|---|---|---|---|
|上游版本|UniDB-SOC/UniDB commit 1bb63e318891b932c6e641b98729643e46b91e06|同commit六份vendor源码|本地blob逐项等于独立下载的commit tree blob|证明源代码对象相同，不证明公式正确|protocol/UPSTREAM_SOURCE.json; official/upstream/GIT_TREE.json|
|实际入口|官方SR train.py→DenoisingModel.optimize_parameters|pipeline.py→torchrun worker.py→DDP(NativeObjective)|自写训练/保存/评价骨架；未实例化官方DenoisingModel|调用相关数学函数，但不能称完整官方训练器复现|pipeline.py:44-70,142-146; worker.py:159-166,182,207-221|
|网络|ConditionalUNet(in_nc=3,out_nc=3,nf64,depth4)|相同151-tensor/137147523参数网络|NPU运行时仅替换1个GELU.forward为原精确erf公式|数学公式意图等价；operator/梯度等价需要数值证据，非SHA保证|core.py:64-86; vendor/unidb_gou/DenoisingUNet_arch.py:19-76|
|Condition|[x_t-LQ_up,LQ_up]与整数扩散时间t|[x_t-y,y]与整数t，无g/xi/prompt|接口相同；LQ_up从x4低分辨率插值，y为同分辨率湍流图|条件语义和任务域实质改变|DenoisingUNet_arch.py:85-99; official SR train.py:238-243; data.py:56-63|
|SR数据|公开默认LR32/GT128、scale4、flip/rot；读入图[0,1]，bicubic上采样条件未clamp、可overshoot|cached256 RGB CHW float32；G1-G4湍流；无在线crop/augment；input允许越界|退化/数据大小/增强分布改变；双方网络condition均与GT同空间分辨率；不能称官方condition严格[0,1]|影响学习分布；strict load不能验证迁移合理性；公开默认YAML不是checkpoint历史训练配置|official SR options train.yml:15-34; LQGT_dataset.py:90-186; deg_utils.py:38-40; data.py:60-63|
|桥超参|lambda_square30,T100,cosine,eps.005,gamma_inv1e-7|core.process硬编码同值|当前均相等；不是DDPM beta schedule；lambda_square输入30被/255作为sigma系数使用|当前/官方参数一致；符号lambda_square不是实际variance30|core.py:103-105; sde_utils.py:74-78,101-146|
|时间|训练每样本randint(1,T)，t1..99；反向100..1；dt由eps与theta_cumsum覆盖|相同官方generate_random_states与reverse_sde|无归一化；tau.squeeze改reshape(-1)保证batch1|B12等价；修复batch1维度角落；100网络输出被忽略|core.py:119-125; sde_utils.py:134,322-375|
|Forward marginal|m_gamma(t)x0+(1-m_gamma)y + f_sigma(t)epsilon|逐调用官方函数|源码相同；m含gamma_inv而f_sigma没有|必须独立检验有限gamma概率一致性；不能以vendor相同视为已通过|sde_utils.py:143-146,190-228,363-375|
|预测均值与target|Euler reverse_sde_step_mean vs reverse_optimum_step/r_mean_1|同一组合，GT正确接clean、condition为degraded|无当前fork公式差异|Euler predicted mean与离散Gaussian条件目标不天然相等；需独立数学审计|official denoising_model.py:127-145; core.py:119-125; sde_utils.py:37-38,202-213,262-269|
|MatchingLoss|unweighted L1逐元素→每样本所有channel/pixel均值→batch均值|相同MatchingLoss(l1,is_weighted=False)|无额外clean/perceptual/recursive loss；官方train.weight读取但也没乘到loss|损失实现忠实不等于目标可保证最终质量|vendor/loss.py:9-29; denoising_model.py:50-53,138-145; core.py:114-125|
|SDE drift实现|训练tensor t用sde_reverse_drift；推理int t用sde_reverse_drift_1|原文件两方法均保留|t1..99代数相同；t100分支特殊|target仍是离散posterior；需对全时刻数字比较|sde_utils.py:254-269|
|SDE sampler|x=condition；t100..1；score=-noise/f_sigma，t100强制score0；x-=drift+sigma_t sqrt(dt)z|worker.evaluate调用同reverse_sde，起始y|与官方采样算法一致；当前用EMA而官方用RAW|t100输出计算但忽略，先生成随机状态；最后t1仍有随机扩散；非精确r_sigma_1采样|sde_utils.py:174-178,271-275,322-337; worker.py:45-68|
|mean-ODE|同100步循环和完整反向SDE drift，仅省略dispersion|诊断直接调用官方reverse_mean_ode|代码级单因素去随机扩散，状态轨迹因此不同|不是probability-flow ODE；也不是已证明的posterior mean；不能只看输出PSNR认定有效恢复|sde_utils.py:177-178,339-354; diagnosis sampler_control.py:35-59|
|Optimizer|公开默认Adam lr1e-4, betas(.9,.99), wd0, MultiStep[300k,500k,600k,700k] gamma.5，900k步|Adam lr1e-5,同betas/wd0，constant，45411步|学习率1/10，取消scheduler，batch12倍，训练量不同|不同优化recipe；不是唯一骨干替换受控实验；公开YAML不证明发布权重实际训练历史|official train.yml:59-73; worker.py:160-166,220|
|EMA|EMA(model,beta.995,update_every10)；requirements未锁版本|同调用；实际ema-pytorch0.3.2|0.3.2还含update_after100、inv_gamma1、power2/3动态warmup，非从第1步恒定.995|原公开训练时实际包版本未知；当前checkpoint保存step/initted但其余默认值依赖环境|official denoising_model.py:118; core.py:128-129,150-159; official/EMA_INSTALLED.py:50-67,177-201|
|验证用的权重|官方训练中验证用online self.model；test加载指定网络文件至self.model，不在test时自动取EMA；save另存G和EMA|主正式验证与选模用EMA；另有Final RAW诊断|相对官方训练期默认验证改变权重分支；发布裸state_dict的历史RAW/EMA来源未知|影响结果；已有RAW子集只说明局部不足以挽回损失，不能当完整RAW终评|denoising_model.py:150-158,189-197; worker.py:195,243|
|评价|PSNR/SSIM用clamp+round uint8、test默认裁4像素；CPU FP32 Output经tensor2img原地clamp后，SR_img再算LPIPS|raw float RGB、无clip/crop/resize；Gaussian RGB SSIM；LPIPS由旧项目模块导入且no clamp|公开脚本与当前协议不同；官方LPIPS不是完全raw输出，纯CPU别名探针已确认|可影响绝对指标；同协议U0/Final配对仍有效；旧metric模块不在RUN_SOURCE源冻结中|official SR test.py:110-150; img_utils.py:136-163; denoising_model.py:163-169; OFFICIAL_TENSOR2IMG_ALIAS_PROBE.json|
|随机数与batch|官方默认seed0、DataParallel/单GPU batch8；验证通常batch1|rank-specific seed843101派生；训练DDP8×12；验证同source4实例并行/每source固定seed|GPU/NPU RNG和batch不同，不能跨后端称同seed同noise|本实验内部配对需验证实际randn张量，不只seed值|worker.py:59-68,165-188; official SR train.yml:29,75|
|运行环境|README为PyTorch/CUDA，requirements未锁torch/ema版本|Ascend910B3/Torch2.7.1+torch-npu2.7.1.post2/FP32/HF32off|共享venv和外部site-packages解析，NPU精度配置及GELU等价补丁|不是封闭可重建环境；当前pkg版本与原记录可比，不证明历史wheel未变化|run_env.sh; protocol/ENVIRONMENT.json; core.py:89-99|
|CLI预算|官方niter900000|formal命令含--steps20，但formal循环仅用deadline/STOP判定|--steps仅probe/bench/pilot生效，formal忽略|不是只训练20步；真实45411日志优先，需报告该易误读接口|pipeline.py:65-70; worker.py:198-202|
|网络upscale元数据|SR options.py设置ConditionalUNet(upscale=4)|core使用默认upscale=1|self.upscale标记not used，forward未访问|此属性没有当前数学影响；实际SR上采样在网络外执行|official options.py:25-30; DenoisingUNet_arch.py:20-23,85-134; deg_utils.py:38-40|

## 最高优先级后续检查

1. `f_sigma`与有限gamma drift/mean的概率一致性，以及Euler predicted-mean与离散posterior target不是同一对象。
2. t100输出忽略、t1继续注噪、采样variance用sigma²dt而不是`r_sigma_1²`，均来自官方源码，不能因此免审。
3. SR x4插值条件与同分辨率湍流是实质任务变化；B96/constant1e-5/256输入/EMA主验证均偏离官方SR recipe。
4. 运行器没有执行vendor/denoising_model或vendor/train，而是调用选定数学模块；必须审查wrapper本身。
5. EMA动态warmup默认与metric外部模块未纳入完整源冻结，不能把旧工程PASS当对整个运行环境无条件证明。

当前结论仅为源码/配置/调用链差异清单；不接受“所有文件相同所以科学方案正确”的推断。

## 实际执行调用链

```text
pipeline.run -> run_env.sh -> torch.distributed.run(8 ranks) -> worker.main
  -> core.initialize -> ConditionalUNet(strict official SR state_dict)
  -> NativeObjective -> DDP -> data.Positions(y,c)
  -> generate_random_states(x0=c,mu=y) -> noise_fn -> noise-to-score
  -> reverse_sde_step_mean -> MatchingLoss vs reverse_optimum_step(...,c)
  -> backward -> Adam.step -> EMA.update
  -> evaluate(EMA.ema_model) -> set_mu(y) -> reverse_sde(y)
  -> external old-project image_metrics
```

本次没有把官方DenoisingModel作为训练器运行。vendor/train.py来自deraining而非SR任务，但它也没有作为本run训练入口执行；实际使用的是共享网络/SDE/loss与自写wrapper。不能因六个vendor文件一致就跳过wrapper检查，也不能把这些文件相同等同于SR完整recipe复现。

## 时间、预测均值与目标均值的证据边界

训练时刻是每样本t=1..99，推理时刻是100..1。t100网络输出被score=0忽略；起始x=y时反向漂移为0，SDE第一步只注随机扰动。t1仍注噪，而GT条件离散posterior的r_sigma_1(1)=0。`SDE.__init__`的dt=1/T又被UniDB重置为`-log(eps)/theta_cumsum[-1]`，不可把最终dt默认为0.01。

预测均值使用Euler drift，监督target使用r_mean_1构造的离散Gaussian条件均值。两者在官方代码共存并不等于同一有限步转移。m/drift含有限gamma而f_sigma没有的概率一致性，仍须数学子审计独立检验。官方注释“maximum likelihood objective”也不能当作Gaussian MLE或完整条件分布学到的证明，实际criterion是无variance加权L1。

## EMA实际默认与保存语义

实际installed ema-pytorch为0.3.2，源文件SHA256为`5813a170affb596072e3cb5872c20971e699bc5e28cfcaa33ae7905f52e82ea2`，原件和环境路径分别保存在EMA_INSTALLED.py、EMA_ENVIRONMENT.json。默认update_after_step=100、inv_gamma=1、power=2/3、min_value=0、karras_beta=False。

update()先递增step，仅旧step可整除10时进入更新；旧step<=100时复制online，之后首次initted复制，再应用逐渐ramp的decay，衰减上限才为0.995。不能把本轮EMA描述为从第1步开始恒定beta0.995。官方requirements未锁EMA版本，故不能据此反推作者发布权重时的默认库实现完全相同。当前checkpoint保存了EMA的network/initted/step/beta/every/update_after，部分其余默认值仍依赖安装环境。

官方训练期验证用online self.model；test加载给定文件到self.model再采样，不自动选择EMA wrapper。保存时另存G与EMA，但重命名发布的gamma_1e7裸state_dict是否本来源于作者RAW或EMA，没有元数据证明，不能从变量名推断。本轮主评价用EMA。已有最终RAW子集对照只证明局部切换不足以挽回损失，不是全节点完整RAW终评。

## 官方评价原地裁剪的独立复核

官方get_current_visuals先将Output变为CPU FP32；SR test将其赋给SR_img，再调用tensor2img。后者的squeeze().float().cpu()在该情形共享原storage，clamp_会原地改动SR_img，之后LPIPS仍使用该SR_img。

OFFICIAL_TENSOR2IMG_ALIAS_PROBE.json保存了纯CPU toy复核：输入[-0.2,1.2]经过官方原函数原地变为[0,1]，最大修改0.2，data pointer未变。未调用任何网络。官方PSNR/SSIM用量化图，而LPIPS用已经裁剪、未量化的浮点图。本轮外部image_metrics明确无clamp，因此与官方绝对指标不可直接混比；但这并不能独自解释本实验同协议U0/Final的退化。

实际外部metric源码另存EXTERNAL_IMAGE_METRICS.py及SHA。其SPEC['loss']仍有旧V5描述，本run NativeObjective并未读取这个字段，不能误判为训练用了旧V5 loss。

## 预训练权重来源与两份SR权重

官方README链接[公开Drive根目录](https://drive.google.com/drive/folders/192X46xuCcPyZrGPDHlUFfug1ggRKyhFn)，其[super-resolution子目录](https://drive.google.com/drive/folders/18x1AceIPM2ymBpNXP8z9jhVR7ws5ibf_)当前列出以下两份文件。本次独立取回原HTML并核SHA，见upstream/DRIVE_PROVENANCE.json。

|文件|Drive ID|字节数|本run|
|---|---|---:|---|
|gamma_1e7.pth|1kIrvZ1X46jFRAYIScverDD3dT1F40Kip|548649183|使用此文件，SHA a6bb76bfabe32b8403d8b112f0ce7876114d830d20b9f4a4119aa860afef64b4|
|gamma_5e5.pth|181F1v3nx2cFBgelajOEtwe0pCwH9b9AP|548649183|没有同数据同预算的迁移对照|

名称所指gamma1e7与公开SR YAML的gamma_inv1e-7一致。这是不同gamma标记，不是不同task prompt或Adapter版本。裸state_dict没有训练超参/数据清单嵌入；所读发布页面也没有作者给出的训练载荷SHA。因此来源、结构和本地/远端文件身份可以核实，精确历史训练recipe、最优权重选择和湍流迁移合理性不能由strict load推出。

## 原任务阳性对照所需的原件

官方LQGTDataset无预生成LR时使用data/util.py的Matlab-style antialiased imresize；当前已完整取回cubic、calculate_weights_indices、imresize原函数。imresize的numpy分支硬编码.cuda()；在本服务器应直接传CPU torch.Tensor CHW/BCHW走原函数，不能把F.interpolate下采样冒称官方退化。预生成DIV2K PNG的8bit量化与on-the-fly浮点LR也应区分。

此外已从[官方DIV2K发布页](https://data.vision.ee.ethz.ch/cvl/DIV2K/)链接的HR.zip和LR_bicubic_X4.zip仅按HTTP206抽取0801.png与0801x4.png两成员（未下载整包），CRC32核验通过。原件与SHA/HTTP headers/ZIP成员位置在upstream/div2k_official_pair/SOURCE.json，HR2040×1356、LR510×339。其对齐crop可以验证原任务部署行为，但一幅crop不等于DIV2K benchmark。

## 复现覆盖缺口

RUN_SOURCE只冻结py文件，INITIAL_EXECUTION_SOURCES.zip也没有run_env.sh。EMA和Torch分别解析到其他共享环境路径，外部旧V5 metric源没有纳入最初冻结清单。现取回EMA/metric源及SHA是当前观测，不能补造作者或历史不可变环境证明。

ENVIRONMENT_COMPARISON.json核对旧ENVIRONMENT与当前运行器采集：executable与torch模块路径、numpy/EMA/torch_npu版本均一致。旧torch package metadata的2.7.1与当前torch.__version__的2.7.1+cpu来自不同报告接口，不能单凭字符串后缀推断历史换包或训练误跑CPU；实际NPU设备与占用须由运行代码/日志独立证明。

本次用裸env import EMA时曾因未加载Ascend库环境出现libhccl缺失；改用stdlib metadata读取源码成功。该新审计读取失败不是正式训练失败证据，也没有造成模型调用或权重改动。

## A/B/C分开判断

- A，是否忠实实现公开loss：代码身份及对应调用链可核，未见当前fork额外改写数学算式；recipe、评估分支与数值后端存在明确改变。
- B，数学目标是否自洽/合理：不因A就判通过；有限gamma协方差、Euler预测与离散target、SDE噪声与posterior variance必须独立推导/量化。
- C，是否适合强湍流：strict load不提供该保证；须由原任务阳性对照、迁移失效、条件依赖和扰动实证支持，不能维护旧解释。

主Git来源：https://github.com/UniDB-SOC/UniDB/tree/1bb63e318891b932c6e641b98729643e46b91e06 。逐文件gitblob URL和SHA见upstream/FETCH_MANIFEST.json；独立live树见upstream/GIT_TREE.json。

## 新诊断运行器 run03 的追加只读复核

原 RUNTIME_REVIEW.json 是模型启动前的 run02 冻结源码快照，现独立保留。run03 在不改科学 PROTOCOL 的前提下增加 sample_id 的 SHA 路径编码、跨 run JSONL 证据读取与已完成 FD/activation 复用。RUNTIME_REVIEW_RUN03.json 重新审阅实际代码，RUNTIME_RUN03_SERVER_HASH_CHECK.json 核对服务器全部新源码与 SOURCE、全部原代码与原冻结清单一致；实际源码原件在 runtime_run03_snapshot/。当前 worker SHA996e86254fdf467f6da3558a9abd4aa86e60e274fb123ad2aa540e5ea76211b4，controller SHA9f4f84898bcecdb1d2b2b942b73bbaf0bf599690332149df154677c86c833c27，科学 PROTOCOL SHA 仍为5c88c9eccb641816b329fc00d965a44bed7fa8c4473b0ef91c0d9894113a57de。

这些修复改变记录/路径/任务复用，不改变网络、桥、loss、输入、种子或训练状态。静态复核不能替代最后全输出的唯一键与SHA审计；activation复用门本身只检查计数，最终CPU验收应另核(case/split/source/G/t/family/module)完整唯一覆盖，而不是为此重跑已完成模型。


---

本顶层交付版完整来源为 `official/REPORT_ZH.md`。原子报告与原始表/源码均保留；文中早期“后续检查”现由 `SAMPLER_AUDIT.md`、`OPTIMIZER_AUDIT.md`、`teacher_noise_probe/REPORT_ZH.md`、`cpu_rollout_parity/REPORT_ZH.md` 完成对应验证。综合因果排序以 `FINAL_FORENSIC_AUDIT.md` 为准，不能只从官方源码相同或某项数学差异推出根因。

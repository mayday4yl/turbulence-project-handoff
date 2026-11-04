# V5 DiffUIR-L：官方基线、反传与短 D0 Pilot 最终报告

日期：2026-10-26，北京时间。独立项目 `v5_diffuir_l_stage1_20261026`。

**DiffUIR-L 已通过 Ascend 910B3 上的 FP32 推理、原生 Loss 真实参数更新和完整三步采样反传；100 次 D0 Pilot 已完成并保存可读回 checkpoint。技术上具备继续开展有界 D0 微调的条件，但当前结果没有证明稳定的湍流恢复收益，不支持直接扩大为长期训练。** 本轮任务完成后已停止，NPU2 无本任务进程。

本次替换的范围是 Clean Restoration 组件：采用 DiffUIR 的像素扩散方法及其 Large U-Net。它同时改变了 ResShift 的去噪网络、表示空间和扩散定义，不能称为“只替换 ResShift 的 U-Net”或“给原 ResShift 换一个 VQ”。旧 V5 D0/A/B 和 ResShift 成果均保留。

## 1. 交付状态与范围

| 项目 | 实际完成情况 | 结论范围 |
|---|---|---|
| ResShift 留存 | 3个原目录、7,321文件、4.5613 GiB，完整 SHA 清单与二次读取核验；33份关键证据副本 | 原地内容归档，不是另建全量异地备份；不再继续 ResShift 正式训练 |
| 官方 Large/权重 | 固定 commit，36,261,187参数，raw及EMA严格核验通过 | 没有默认Base、随机初始化或忽略不匹配键 |
| NPU真实推理 | 固定Val4、FP32、每图3NFE，输出有限，保存全部浮点状态 | zero-shot迁移检查；不是GoPro基准复现 |
| 原生训练反传 | 276/276参数tensor有有限非零梯度，真实Adam更新后全部变化 | 诊断更新单独保存并丢弃；未带入Pilot |
| 三步采样反传 | 两例同噪声前向逐值相同，输入、参数及三个阶段均有有限非零梯度 | 尚未接入四次外层物理递归 |
| 短D0 Pilot | 冻结配置后完成100更新、200样本呈现、保存4份checkpoint | 执行健康；Val50收益不稳定 |
| 保护与回读 | 旧107文件、32 checkpoint元信息、600个数据文件及新官方源码核验通过；最终checkpoint完整读回 | 旧checkpoint为size/mtime/inode比较，未宣称重新全字节哈希 |

## 2. 官方代码、结构和 checkpoint 身份

官方仓库：[iSEE-Laboratory/DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR/tree/c71437ce86ad1b098aa9be4678dc82e8b99649df)，[CVPR 2024论文](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Selective_Hourglass_Mapping_for_Universal_Image_Restoration_Based_on_Diffusion_CVPR_2024_paper.html)。固定 commit：

`c71437ce86ad1b098aa9be4678dc82e8b99649df`

模型显式构造为 `UnetRes(dim=64,dim_mults=(1,2,4,8),channels=3,num_unet=1,condition=True)`，即官方 **64-1248 Large**，各级宽度64/128/256/512；精确36,261,187参数、276可训练tensor。当前固定提交的train/test脚本本身已显式使用Large，不能误报为默认Base；Base为64-1224。网络含卷积、GroupNorm、残差块、LinearAttention与瓶颈Attention；没有VQ、码本或latent encoder/decoder。

官方Large文件夹下载的 `weights/model-300.pt` 为580,824,471字节，SHA256：

`b9d07ab6b32d37d895be65c50e99bbe6b2ae6e32823e7304ec2167bc5f18843f`

checkpoint `step=300000`，含model/opt0/ema/scaler；raw与EMA分别严格验证290个state tensor，无缺键、多键、shape、dtype、数值或有限性问题。标准三步测试对应model-300；model-130属于UDC四步分支，本轮未混用。官方评估使用EMA并调用`init()`重建推理schedule；两分支及重建前后buffer差异分别留证。

**GoPro结论：官方README、训练配置与论文支持“包含GoPro去模糊的五任务通用预训练”，不是GoPro-only专用模型。** 二进制没有单独的训练任务履历字段，也没有原作者完整训练日志；本轮确认发布来源和Large权重匹配，不把它扩大成独立复现其训练历史。

详细映射见[代码审查](<../../../NOT_INCLUDED.md#ref-f32ab97ab0988e0acdd8>)、[raw权重验收](<../../../NOT_INCLUDED.md#ref-b5084840bd337a36231c>)、[EMA权重验收](<../../../NOT_INCLUDED.md#ref-2047f81e1e5f58df51f3>)、[schedule核验](<../../../NOT_INCLUDED.md#ref-804b46220fa75c2860fc>)。

## 3. Condition、原生训练和三步采样

令Clean为C、湍流条件为Y4，归一化为C′=2C−1、Y′=2Y4−1，残差r=Y′−C′。原生训练抽一个随机扩散索引t：

`x_t = C′ + a_t r + b_t ε − d_t Y′`

网络输入是 `cat(x_t,Y′)`，形状 `[B,6,256,256]`，预测RGB残差。时间嵌入实际输入为 `a_t×1000`。原生Loss为 `mean(abs(pred_res−r))`，每个训练样本只调用一次网络；不展开三步采样训练，不加入Clean辅助损失或物理损失。

Condition同时进入前向分布、6通道拼接、采样初始化和更新公式。`['blur']` 是源码接口读取的元数据，当前数值分支没有基于该字符串的task embedding，不能把这个字符串当作新增Condition。

冻结标准配置：timesteps1000、delta_end0.0018、sampling_timesteps3、eta0、sum_scale0.01、pred_res、L1。采样从 `(1−d_999)Y′+0.1ε` 开始，在999→665→332三个索引执行网络；前两步按官方残差/condition系数更新，最后输出clamp后的预测Clean。初始噪声仍随机；eta0不等于确定性。训练噪声系数与采样初始0.1不同是该官方提交行为，本轮未自行改写。

官方采样返回“初始图+三步图”共4个元素，最终取[-1]，实测NFE为3。代码中的两处`no_grad`使原采样无法直接反传；诊断仅在独立包装中临时开放装饰器，保留全部公式、clamp和真实梯度。没有改官方受追踪源码。

| 功能 | 官方科学入口 | 本项目入口/适配职责 |
|---|---|---|
| U-Net、Condition、schedule、Loss、采样 | `upstream/src/model.py` | `scripts/common.py`：显式Large、严格权重、原NPY与设备绑定 |
| 官方训练/测试配置参照 | `upstream/train.py`、`test.py` | `scripts/run_baseline.py`：分阶段权重/推理/梯度验收 |
| 原生Loss更新与三步图反传 | 同一`ResidualDiffusion`类 | `scripts/audit_training.py`、`grad_helpers.py`：观测、同噪声对照和状态恢复 |
| 有界D0优化 | 同一`forward/p_losses` | `scripts/run_pilot.py`：冻结清单、Adam/EMA、有限性检查、日志及checkpoint |
| 完成验收 | 已生成模型与原始资产 | `scripts/verify_completion.py`：CPU回读、哈希及保护核验 |

## 4. 910B3 推理与反传结果

远端为de36，独立根 `/data2/yanglin/v5_diffuir_l_stage1_20261026`；运行前确认物理NPU2空闲。新venv只在新目录安装补充依赖，通过`.pth`只读复用已有Torch/NPU库及Stage1依赖，不是完全隔离容器。torch2.7.1+cpu、torch_npu2.7.1.post2，FP32、HF32关闭。采用官方类的真实NPU算子，未用CPU模型或伪实现替代。没有验证AMP或CUDA数值对齐。

固定原Val4身份和SHA均保留，四张是同一park类别的不同source；输入原FP32 NPY不裁剪、缩放或PNG量化。评价为完整RGB range1、逐图PSNR后平均。

| 固定Val4 | Input | 官方DiffUIR-L EMA zero-shot |
|---|---:|---:|
| PSNR/dB | 17.7892 | **17.5407** |
| SSIM | 0.355331 | **0.350776** |
| MAE | 0.090757 | **0.092212** |

最终RGB整体范围[0.015901,1]，所有输出有限。每张3NFE；首张1.092秒，后三张平均0.1527秒；allocator峰值allocated505.06 MiB、reserved550 MiB。这是单轮同步采样观测，计时不含加载、指标和保存；不是完整吞吐基准或进程总HBM。

原生训练诊断：2微批累积1次Adam更新，276/276参数tensor梯度有限非零，梯度总L2=0.25924；更新后276/276全部变化且有限。该诊断用两张Val图，仅检查链路；随后完整恢复官方raw state，Pilot再从官方raw重新载入，未使用诊断更新后的权重。

完整三步梯度：两例均与同权重、同噪声官方采样逐值相同；参数梯度L2=0.17025/0.24920，输入梯度L2=0.03583/0.03002；三步中每一步网络输出梯度均非零有限。原no_grad输出作为负对照确无计算图。单次三步图的峰值allocated约6.50 GiB，尚不包含四次外层物理递归与其优化器状态。

详见[真实推理/反传报告](<../../../NOT_INCLUDED.md#ref-efe2294622806548d848>)与[Val4对照图](<../../../NOT_INCLUDED.md#ref-ed29d8dda8c1c311707d>)。ResShift图保留其FP16/NFE4设置，图用于观察，不能视为严格匹配训练比较。

## 5. 100更新 D0 Pilot

三项基础gate通过后冻结[配置](<../../../NOT_INCLUDED.md#ref-f8c038a3f284646bbbcd>)与[数据清单](<../../../NOT_INCLUDED.md#ref-ba0c2932fcf28700d186>)，配置SHA为 `1cc0d4a4875086edc769eff7d4335c09e82a009ba81662ef31a53a1478633d23`。

从官方**raw model**重新初始化，fresh Adam与fresh EMA。g固定4，原始Y4→C；原生随机t、q_sample、残差L1保持不变。lr8e−5、betas(.9,.99)、clip1、microbatch1×累积2、FP32、无scheduler、无额外增强。与官方五任务训练相比，有意缩小为单任务有效batch2和100更新，不称完整官方训练复现。

候选训练池50类×2source×4realization=400对；实际呈现200个不同realization，覆盖91source、49类，没有遍历完整候选池。Val50按元数据冻结为50类各1source的r00，和Train按source隔离；固定Val4作为补充面板。Val4与Val50自身重叠1张，不能算作54个独立来源。未使用Test1000或用Val图训练Pilot。

| 更新 | EMA Val50 PSNR | EMA Val50 SSIM |
|---:|---:|---:|
| U0 | 18.73989 | 0.456905 |
| U25（本次best） | **18.74114** | 0.464300 |
| U50 | 18.35820 | 0.460116 |
| U75 | 18.73433 | 0.466522 |
| U100 | 18.48104 | 0.463692 |

Val50输入PSNR为18.95359。best相对同分支U0仅+0.00125 dB，U100为−0.25885 dB；不构成稳定改善。最终raw Val50为18.61176 dB，也低于raw U0的18.73989。固定Val4的raw为17.47459→17.66459 dB、EMA为17.47459→17.17111 dB，显示小面板和权重分支之间差异，不能选其中一个数宣布泛化提升。

这里的U0不同于zero-shot章节的**官方已训练EMA**：Pilot的fresh EMA初始拷自官方raw。EMA依赖保留默认update_after_step=100、update_every=10；短Pilot仍在warmup复制阶段，尚未执行指数滑动平均。实际源码先读取旧step再自增，U100的EMA最后一次复制的是raw U91；U25面板对应raw U21。详见[EMA实际配置](<../../../NOT_INCLUDED.md#ref-99043556d8237ff82247>)和[本次安装依赖源码](<../../../NOT_INCLUDED.md#ref-7bdedd55e54808f935af>)。

100条逐更新日志均有有限Loss、梯度和参数；真实Adam状态全部计数100。200次随机扩散索引包含184个不同t，范围6–998。任务记录21:43:10—21:45:42，152.43秒；其中计时的训练更新合计33.41秒，训练循环含中间评估/保存约109.98秒，训练allocator峰值allocated2.902 GiB/reserved3.303 GiB。时间口径不同，不能把33秒当作完整任务耗时或直接外推多卡正式训练。

保存U0/U25/U50/U100四份完整checkpoint，best指向U25。所有文件尺寸与SHA验证通过；U100模型、276组Adam状态、EMA计数、数据游标200、Python/NumPy/Torch/NPU随机状态成功CPU读回，未额外恢复训练。

大型Pilot checkpoint及完整评估浮点数组保留远端；本地同步日志、状态、评估JSON和全部checkpoint哈希清单。最终模型为：

`/data2/yanglin/v5_diffuir_l_stage1_20261026/runs/d0_pilot_100u/checkpoints/U000100.pt`

其大小725,939,381字节，SHA256 `3c8e91a82502be50e11920a382a1d9f393ba8621288c0c64677fb8ba599b8711`。

最佳checkpoint为同目录`U000025.pt`，SHA256 `055a8b95ccc4660f207a8c021b8184cde4a48db82b055cc5891fbc6ac530347e`。复现best分数须加载该checkpoint的EMA分支（当时拷自raw U21），不是它的raw U25。目录位置及全表见[CHECKPOINTS.json](<../../../NOT_INCLUDED.md#ref-14e0706947d158f47074>)。详见[Pilot报告](<../../../NOT_INCLUDED.md#ref-fea800777290592cbd2f>)、[原始RESULT](<../../../NOT_INCLUDED.md#ref-d25c2ff4dea9f99b3b56>)、[checkpoint回读](<../../../NOT_INCLUDED.md#ref-528caf3f08b5a6b38180>)。

## 6. 可行性结论与下一阶段边界

**正式D0的工程可行性已得到短程支持，效果可行性尚未证实。** 当前没有权重不匹配、FP32算子或原生训练反传的实质阻塞；也没有证据支持把短Pilot的下降归因于架构不适合、扩散定义错误、数据不足或某个单独超参数。只跑了一组短配置和一个训练seed，不能作因果诊断或收敛判断。

如继续正式D0，应先另行冻结训练预算、batch、初始化分支、完整Val1000评估频率与停止规则，并采用同分支U0作为对照；不要直接把本次几乎持平的U25宣布为成熟D0。原生方法仍是后续基线，是否改学习率、初始化分支或训练长度需要作为明确的新实验记录。本次没有自动延长预算或开始大规模训练。

未来四次RGB物理递归可以复用：原V5数据身份、Y4/C配对、raw-ξ及Simulator资产、固定评价清单，和此次通过的DiffUIR训练/采样接口；它们目前只是被保护的可复用资产，尚未集成。DiffUIR的扩散时间与V5物理时间应继续区分。若每次物理恢复包含3次扩散网络调用，则四步外层为12NFE；需要独立验证完整图、显存和优化稳定性，不能由本轮三步梯度连通性直接推出。

本轮未增加ξ Head、物理时间调制或Condition模块；未修改旧V5 D0/A/B；未运行Test1000、AMP、多卡或外层物理B训练。

## 7. 留存与复查入口

- [ResShift归档回执](<../archive/ARCHIVE_RECEIPT_ZH.md>)：全量原地SHA清单、二次核验和旧状态差异。
- [旧资产保护对比](<../../../NOT_INCLUDED.md#ref-fdc161e62320774761f9>)：107受保护文件、32旧checkpoint元信息、600个选用数据文件、新官方27文件及结束后的NPU快照。
- [运行环境](<../evidence/RUNTIME_ENVIRONMENT.json>)、新目录`evidence/pip_freeze.txt`与环境安装日志：实际依赖和安装重试均保留。
- [全部更新日志](<../../../NOT_INCLUDED.md#ref-78598984d4aaebe75bc9>)、[实际训练覆盖](<../../../NOT_INCLUDED.md#ref-2948582ad038ff9872c2>)、[评估历史](<../../../NOT_INCLUDED.md#ref-ddfabfdb8321381e5b2e>)。
- [独立交付复查](<../../../NOT_INCLUDED.md#ref-fd498bf55bcfaae99641>)、[文件交付清单](<../../../NOT_INCLUDED.md#ref-b871f46ef4d7af643073>)。

本项目原生脚本仅供复查；`run_pilot.py`拒绝覆盖已有run目录。任何新训练需新目录与新冻结配置，不能用重新执行现有命令覆盖此次对照。

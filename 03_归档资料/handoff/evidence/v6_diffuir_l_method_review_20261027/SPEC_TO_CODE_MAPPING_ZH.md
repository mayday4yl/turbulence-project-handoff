# V6 spec → code 映射

日期2026-10-27。下列是现有代码映射与缺口，**不是已完成V6实现清单**。方法规范行号以本目录reference的440行原件为准。状态：可复用=已有组件或逻辑；需新增=本次检索范围未发现V6实现；需验收=旧证据不足以覆盖V6新图。

| ID / Spec | 现有代码入口 | 实际行为与V6对应工作 | 状态 |
|---|---|---|---|
| M01 §3.1 Large | [common.py:49](<../../NOT_INCLUDED.md#ref-9cb114c80c9ea4f646eb>) | 显式64-1248、condition、pred_res、1000/3、eta0、sum_scale.01；精确36,261,187是D0主干，不是A/B总参数量 | 可复用 |
| M02 §5 EMA起点 | [common.py:80](<../../NOT_INCLUDED.md#ref-90b1b1f176f549264040>) | raw/EMA290状态严格加载；EMA分支随后init；V6必须把网络参数来源与schedule生命周期分开 | 需新增包装/验收 |
| M03 §3.1 六通道 | [model.py:387](<../../NOT_INCLUDED.md#ref-ef16c389c1227946cf0e>)、[拼接:835](<../../NOT_INCLUDED.md#ref-1da9bfeb320147d60e8d>) | 原生6入3出；禁止额外Y4或time-map通道 | 可复用 |
| M04 §3.2 q_sample | [model.py:1062](<../../NOT_INCLUDED.md#ref-a062c3cd1afc27169c17>) | C′+aR′+bε−dX′，支持显式noise；真实配对才可用 | 可复用 |
| M05 §3.2 residual | [model.py:1080](<../../NOT_INCLUDED.md#ref-f84d19e9cb4ab3d8e61d>) | p_losses给未乘1/2的原生L1列表；V6新组合层必须乘0.5 | 需新增 |
| M06 §3.2 τ抽样 | [model.py:1145](<../../NOT_INCLUDED.md#ref-99d920982bb0c07d1379>) | randint单τ/样本；A/B还须独立传physical_t，不能改τ定义 | 包装需新增 |
| M07 §3.3采样 | [model.py:945](<../../NOT_INCLUDED.md#ref-c2468966dd48c7a79577>) | 初始randn，999/665/332，eta0，last强制False；最终取[-1] | 可复用公式 |
| M08 §3.3真梯度 | [grad_helpers.py](<../../NOT_INCLUDED.md#ref-1cb249a1c1f0adbd4604>) 的 sampling_gradients_enabled | 仅打开sample/ddim_sample装饰器，finally还原；没有STE | 已有单图证据，V6合图需验收 |
| M09 §3.3 clamp | [model.py:880](<../../NOT_INCLUDED.md#ref-d0ccad817c30b96633d1>) | residual与Clean都clamp；训练p_losses未clamp；保留差别 | 可复用；饱和梯度需诊断 |
| M10 §3.4 schedule | [model.py:736](<../../NOT_INCLUDED.md#ref-69f32ff328958ee5e82f>) | init重新赋值CPU buffer；alphas[0]/betas2[0]与构造器不同，checkpoint累计表有微小浮点差异 | 联合生命周期未实现 |
| M11 §4.1时间Adapter | [model.py:411](<../../NOT_INCLUDED.md#ref-0f19e1eb1c1e3f7a9c35>)、[forward:470](<../../NOT_INCLUDED.md#ref-c397c0640e5f0511f77a>) | 256维time_mlp输出是加法接入点；原GELU不改；另加规范SiLU分支 | 需新增 |
| M12 §4.1 ResBlock | [model.py:279](<../../NOT_INCLUDED.md#ref-fe125d6eef6b3f8dfb2b>) | 原时间特征进入Linear256→2C，scale/shift进block1 | 可复用；不可替换扩散时间 |
| M13 §4.2 ξHead | [旧v5_concertormer.py:65](<../../NOT_INCLUDED.md#ref-f5ccab695291d7288666>) | 旧Head输入主干特征，与新concat(X′,Ĉ′)独立U-Net不符；不能迁移旧head权重 | 需新增；调制定义待明确 |
| M14 §2 native raw | [vendor/v4_simulator.py:25](<../../NOT_INCLUDED.md#ref-b6c907bfc3d404051134>)、[effective_state:97](<../v5_v1_simulator_20261020/simulator_v5.py>) | Realization两个字段；36×512²与512²×2严格匹配；g^(5/6)/相关变换/FFT由F执行 | 可复用 |
| M15 §5 D0旧入口 | [training/core.py:93](<../../NOT_INCLUDED.md#ref-a857196686a071cc51c3>) | clean_only分支只有直接Clean MAE，不是V6 D0 | 需新训练组合层 |
| M16 §5旧Pilot | [run_pilot.py:290](<../../NOT_INCLUDED.md#ref-5764b13408bfc6e58853>) | raw初始化；单原生L1；不能续跑充当EMA初始化的V6 | 保留历史，不复用run |
| M17 §6真实A数据 | [alignment_data/plans.py:23](<../../NOT_INCLUDED.md#ref-2f52a50098a9818a0b5c>)、[dataset.py:15](<../../NOT_INCLUDED.md#ref-dece182e118686ae06fe>) | 旧10时间分层缓存、每B48固定12端点36内部；分布思想可参考，更新上限/缓存节点不可直接套V6 | 新loader/采样需实现 |
| M18 §6 A物理项 | [alignment_training/core.py:41](<../../NOT_INCLUDED.md#ref-bffdc033793af6fb8edf>) | 当前g重建当前Yg，.9LC+.1LP；V6改用S3与新H并加Lres | 锚定可复用；合图新增 |
| M19 §7.1外层时间 | [schedules.py](<../../NOT_INCLUDED.md#ref-e810773c05da0b5ddfa9>) 的 sample_training_paths | CPU64独立三点排序、目标dtype下严格端点/次序；非法整行重采样，无jitter | 可复用G4/N4配置 |
| M20 §7.2/7.5 B转移及锚定 | [training/core.py:113](<../../NOT_INCLUDED.md#ref-81746c182b8ec53d1507>) | 4次模型，3次转移，每次F(Ĉ,ξ,4)对原Y4，mean4；替换模型接口并添加TF | 骨架可参考；V6未实现 |
| M21 §7.2终点 | [training/core.py:250](<../../NOT_INCLUDED.md#ref-40d55e3bf0b9d00f0f5b>) | 末次直接Clean，无冗余g0；不能继承infer的no_grad到训练 | 逻辑可复用 |
| M22 §7.3随机流 | [common.py:150](<../../NOT_INCLUDED.md#ref-22aa0f1440ee310992a4>) | 旧Val4每图重置同seed，无source/stage命名；sampler也无显式eps参数 | V6 RNG包装/重放需新增 |
| M23 §7.6 TF | [data_pipeline/dataset.py:38](<../../NOT_INCLUDED.md#ref-081f860be538bb9aa5ff>) | 原dataset只给Clean/Y4/ID，无ξ；不能从旧B batch凭空构造真实TF | 新GT物理数据通路 |
| M24 §7.6同ξ重放 | [data_pipeline/generate.py:64](<../../NOT_INCLUDED.md#ref-58bbd2c413093c75275d>)、[alignment_data/generate.py:108](<../../NOT_INCLUDED.md#ref-ec8f5ac17091cf9548e8>) | 原seed+CPU FP32 B1精确shape重放，并逐字段SHA；只借用函数/校验，不执行旧生成入口 | 可复用重放定义 |
| M25 §7.8 Simulator梯度 | [training/core.py:53](<../../NOT_INCLUDED.md#ref-dc12e9e15a2ebc16ba29>)、[simulator_v5.py:133](<../v5_v1_simulator_20261020/simulator_v5.py>) | FP32 prepare/render保留图；prepared可在同Clean的两个强度复用，不能跨Clean或detach缓存 | 组件可复用，新图需验收 |
| M26 §9平台 | [runtime/environment.py:24](<../../NOT_INCLUDED.md#ref-da5a11b0be5cf80583e0>) 与 [common.py:38](<../../NOT_INCLUDED.md#ref-c7de466a182b26697151>) | 旧Sim internal_format=True、DiffUIR=False；全局设置冲突须独立联合验收；旧环境不改 | 未验收 |
| M27 §9固定PSF后向 | [adjoint.py:14](<../../NOT_INCLUDED.md#ref-a532909a281435b3e3b2>) | 保持原卷积前向、固定核精确线性adjoint；已有证据但V6集成要核对选择和等价性 | 可选已有执行资产 |
| M28 §9全图batch | [bench_worker.py:248](<../../NOT_INCLUDED.md#ref-7eb0a93622b3bfc88b09>) | native与sampling二选一，非联合Loss；optimizer/显存观察框架可参考 | V6 D0/A/B分别新测 |
| M29 §9指标/身份 | [evaluation.py:82](<../../NOT_INCLUDED.md#ref-49ad08fd9258a3db2d1c>)、[training/image_metrics.py:8](<../../NOT_INCLUDED.md#ref-7aef61095a602f8fdc7d>) | 固定Val1000、逐图RGB PSNR、SSIM、LPIPS可复用；需新随机流及完整Step1/4所有指标 | 评价包装新增 |
| M30 工程监护 | [guarded_runner.py:47](<../../NOT_INCLUDED.md#ref-5d08b8635493e612fc4e>) | 最新旧B包装仅观察并暂停，调用原科学unroll，没有改Loss/递归 | 仅参考；不复用旧恢复权限/阈值 |

本地源码与远端核验范围见证据JSON。未运行以上函数，因此本表的“可复用”均不代表已通过V6工程Gate。

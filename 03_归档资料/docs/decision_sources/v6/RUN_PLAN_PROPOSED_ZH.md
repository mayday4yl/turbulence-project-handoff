# UniDB 超分初始化的单帧湍流适配：今晚建议方案

状态：PROPOSED_NOT_STARTED。用户本轮询问训练计划及是否保留原生Loss；本文件为待执行建议，不代表训练授权或已经启动。2026-11-02。

## 目标与方法边界

采用已核官方UniDB-GOU的ConditionalUNet，DIV2K SR/gamma_1e7.pth作为拟定初始化。原生RGB图像条件、时间与随机状态生成、反向一步ML matching目标、gamma_inv=1e-7、T100和原生采样保持。首轮不新增prompt、ControlNet、递归或额外Loss；不沿用D0 Residual L1/SDT/Adam状态。

原生损失在denoising_model.py:134–145为：网络预测noise并转换score，比较reverse_sde_step_mean与reverse_optimum_step；默认MatchingLoss(l1,is_weighted=False)，weight配置1。它是反向一步状态均值的L1匹配，不能称为直接恢复残差L1，也不是直接把最终恢复RGB与GT做L1。

## 拟定训练配置（明确区分官方与本轮建议）

| 项目 | 建议值／来源 |
|---|---|
| 模型 | 官方ConditionalUNet RGB3/nf64/depth4，全参数适配 |
| 初始化 | 官方SR gamma1e7；尚未下载/计算payload SHA/strict load |
| 数据 | 原160K=10000 Clean×4原ξ×G1–G4，每档40K；总体25%各档 |
| 图像 | 保留原256×256配对，条件与GT同尺寸；不使用SR四倍退化；scale1适配属于数据接入变化 |
| 范围 | 核查实际数组范围与官方RGB流程，不套用D0内部2RGB−1；原模拟器小幅越界须明确处理依据，不静默重造/截断数据 |
| Effective batch | 拟定60以保留既有每更新呈现口径；microbatch与acc由实际显存决定，属于我们适配超参数，官方模板batch8 |
| 优化器 | 官方Adam，betas=(.9,.99)，weight_decay按已核配置 |
| LR | 建议迁移初始1e-5，第一阶段固定；这是我们的微调超参数，官方公开模板1e-4。不能把这一选择称为论文完全复现 |
| EMA | 保留官方beta.995/update_every10及所装版本的实际完整EMA状态；每次optimizer update更新一次计数，不按microbatch更新 |
| 训练数值 | 先FP32，后续只作等价设备适配；不得因质量曲线临时更改Loss或采样 |
| 首轮预算 | 最多10000正式optimizer updates；前100作启动健康验收，5000为里程碑，不因达到5000自动停止 |
| 有限阶段结束 | 10000或用户停止后在安全更新边界完整保存并完成终点评估；后续预算单独决定，不默认无限训练 |

10K×60=600000呈现=3.75个160K数据集遍历，仅为预算算术，不是耗时承诺。100步采样为推理成本；常规每microbatch训练只抽一次原生时间并调用一次网络，不展开100步训练。

## 推进与保留训练价值

1. 必要工程Probe：从服务器网络获取所需单一权重，核真实SHA/来源/完整严格加载；核数据来源、配对、Train/Val不交叉与原生范围。运行最小原生梯度、连续/独立恢复数值验收，工程Probe单列并丢弃，不继承到正式run。
2. 正式U0重新从已核官方SR权重开始，建立新的湍流Adam/scheduler/EMA状态，保存U0完整状态及固定验证基线。
3. 同一正式run的前100更新作真实健康验收；通过后自动保持同一网络、优化器、EMA、RNG和数据位置接着训练。正式100更新计入10K预算，不回滚、不重置。普通初期质量升降记录为结果，不作为擅自改Loss的理由。
4. 每500正式更新完整保存并CPU读回，全部保留。U5000/U10000为主要阶段节点；可靠接续基于完整状态，不仅是pth模型参数。
5. 若阶段结果值得继续，后续从该阶段完整checkpoint续训，保持采样位置、随机状态与优化器，不重新加载官方权重开始一遍。短程验收不等于一次性丢弃的训练。

## 验证建议与开销

- 五档G0.5/G1/G2/G3/G4沿用固定Val1000、同来源和固定种子，G0.5不训练。禁止外部测试集选模。
- U0/U5000/U10000（或实际终点）完成五档各1000。每1000更新在预先冻结、按Clean来源选出的各档Val100监控趋势；监控与全量曲线分表，不把500张监控等同5000张完整验证。
- 官方原生T100采样不为速度擅自减步；完整5000张对应500000次denoiser图像评估，必须按实测计入ETA。
- 拟定主曲线统一EMA；官方test默认绑定RAW，此EMA主评测是我们的评测选择，需显式记录。RAW在关键节点补评作核对，二者不拼接为一条曲线；验证必须保存/恢复训练随机状态。
- 主报告共同RGB PSNR/SSIM/LPIPS/MAE与固定同图视觉，保留官方uint8/Y等来源口径作为分表，不混比。训练随机时间目标和完整恢复指标分开解释。
- 最初不急于写死覆盖历史best的指针；完整保留U0/latest/每个完整验证候选及分档Pareto关系。未来需要唯一best时先冻结选模规则，保留G4及弱档代价，不用外部测试挑模型。

## 完整保存接续所需工程包装

官方save_network、固定lastest EMA文件和save_training_state并未构成完整可重复恢复。新工程包装至少保存RAW、EMA完整对象状态（含内部步数/在线副本）、Adam、scheduler、若有scaler、Python/NumPy/TorchCPU/每设备RNG、数据shuffle与sampler/worker随机状态及游标、模型buffer、optimizer update、真实分档呈现、配置/source/data/checkpoint身份。恢复需实做验收；不能将只恢复Adam/scheduler称为完整恢复。该补强不得修改网络/原生Loss/SDE数值定义，Probe与正式分开。

## 证据与未完成项

固定官方commit：1bb63e318891b932c6e641b98729643e46b91e06。源：../recent_pixel_ir_20261101/native_recent/unidb/PINNED_SOURCES.json、../unidb_pretrained_audit_20261102/compatibility/PINNED_TRAIN_CONFIGS.json。

官方代码：https://github.com/UniDB-SOC/UniDB

原文：https://arxiv.org/html/2502.05749

权重目前只有官方目录和HEAD可达证据；尚无载荷校验、strictload、原尺寸运行、NPU梯度或恢复验收。100更新与T100固定验证实测完成前，不能承诺今夜更新数或小时ETA。设备排期需在后续执行时实时核实，不能依据旧占用记录。D0及A/B不恢复，不覆盖历史资产。

本轮0权重下载、0模型执行、0训练，仅计划和静态官方代码核对。

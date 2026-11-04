# UniDB warmup 独立审查补丁：尚未应用、尚未训练

用户本次询问能否继续训练、是否与缺少warmup有关、能否加入。这里提供局部LR改动供审查，不改已收尾的D0源码、配置、权重、STOP或NO_TRAINING_RESUME，不访问远端或启动训练。

## 当前判断

warmup是控制训练初期更新幅度的优化手段，可能有助于保护预训练权重，但本实验还没有warmup对照，不能把它当成已经确定的退化原因。G4的一步目标改善与自身随机采样退化仍是要解释的现象。

另已核查固定commit的官方SR参考配置，其 `warmup_iter: -1`，明确关闭warmup。因此本轮没有漏掉该参考配置里原本启用的warmup步骤；这不证明warmup对湍流微调无用，也不等于拿到了发布权重的全部实际训练日志。
https://raw.githubusercontent.com/UniDB-SOC/UniDB/1bb63e318891b932c6e641b98729643e46b91e06/UniDB-GOU/tasks/sr/options/train.yml

从U45411补一段warmup，检验的是“后续降低/再升高学习率”，不能倒查从U0开始没有warmup的影响。要检验原假设，应从同一官方SR初始化开始建立独立短对照，保持loss、桥、batch、数据顺序、种子、EMA和评价方式一致。

参考：Ma & Yarats, On the adequacy of untuned warmup for adaptive optimization，AAAI 2021：
https://arxiv.org/abs/1910.04209
该文给出2/(1-beta2)的经验warmup时长，本轮beta2=.99对应约200更新。这是提出一个候选值的依据，不证明200步适合湍流，更不证明warmup能修复本次问题。

## 补丁内容

- 新增 `--warmup-steps`，默认0，保持恒定lr行为；新增 `--warmup-start-factor`，默认0.01。
- 例：base lr=1e-5、warmup=200，则第1次更新lr=1e-7，第200次到1e-5，之后恒定。
- 根据“已完成更新数”在optimizer更新前设置实际lr，日志写实际lr。
- checkpoint保存warmup配置、已完成更新数和下一次lr；恢复时验证配置与更新数，不静默重启warmup。
- 不改网络、MatchingLoss、桥、EMA公式、batch、训练数据或推理方法。

`worker_warmup.patch`基于归档worker的固定SHA生成；`warmup_schedule.py`为新增依赖。该patch未应用。归档工程有源码SHA门和停训标记，后续接入应由主任务放入新的独立实验目录并重新验收，不能直接改原归档目录后启动。

## 已验证与尚未验证

`verify_warmup.py`只在CPU验证学习率端点/单调性/默认不变、序列恢复、配置不一致拒绝和全部参数组的lr赋值，不导入torch，不运行模型、不作训练更新。完整修改worker仅作AST/编译检查。

未执行NPU优化器、Adam/EMA/8rank RNG端到端恢复验收，未进行任何质量实验。因此本包不是“工程验收通过、可以自动长训”的声明。

## 如果后续验证

建议先检验一个warmup=200的候选，而非直接延长原配置；观测U500/U1000/U2000附近的固定G1/G4恢复和一步目标。原无warmup分支中能严格配对的已有结果可作为对照，缺失的评价需先检查再决定是否补做。比较时同时看更新数和累计lr预算，避免把“只是学得更慢”误当质量改善。

质量判断需同时报告G4改善与G1退化、原生SDE与确定性对照、固定图和输入基线；loss下降不能单独触发长训。本侧对话没有决定新的训练预算或启动作业。

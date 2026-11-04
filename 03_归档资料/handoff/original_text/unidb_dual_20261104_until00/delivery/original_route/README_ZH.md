# 原路线最终附件：准备入口

本目录只添加收集、核验、绘图与本地打包代码；不修改冻结训练或后处理源码，不运行恢复模型、optimizer、NPU，不下载checkpoint或全部预测。2026-11-03夜间准备完成不代表训练或交付完成。既有21:16全局曲线和G1–G4历史视觉对比保留不动。

最终输出位置：`../delivery/original_route/`。脚本用显式campaign根和bundle中的数据，不依赖旧图脚本的`HERE.parent.parent`项目路径推断。

## 现在可执行的只读预检

```sh
/usr/bin/python3 '/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/delivery_preparation/prepare_original_delivery.py' --dry-run
```

训练、全状态、全5000预测CPU审计尚未完毕时，必须返回exit2，并只在本目录记录`DRY_RUN_*.json`；不得创建delivery目录、采集图片或运行final。`REFUSED_NOT_FINAL_READY`是预期拒绝，不是训练故障。

## 最终证据齐全后才执行

```sh
/usr/bin/python3 '/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/delivery_preparation/prepare_original_delivery.py' --final --python /opt/anaconda3/bin/python
```

此命令会将collector源码通过SSH stdin送到de36，只读核验、传回小bundle，然后本地CPU绘图。它不会启动或重跑模型评价。已有`delivery/original_route/`时拒绝覆盖；应先查看并复用其有效receipt，不能无故重采。

## 必须通过的门槛

1. 当前授权午夜截止已到；两路线`TRAINING_AND_MODEL_VALIDATION_COMPLETE.json`为PASS，controller已退出且其自有子进程退出回执完成。
2. 原线`STATUS=COMPLETE`、final full完成、`TRAINING_STOPPED`与真实LATEST的U/cursor一致。实际停止必须在本次截止之后的完整边界。
3. 原线实际`BEST_FULL.json == BEST.json`，两角色均使用完整EMA五档5000；不自行扩大候选池。历史完整full允许复用，partial和monitor不能冒充full。
4. 真实`STATE_AUDIT_original_FINAL/BEST`必须PASS并绑定实际checkpoint完整文件SHA、RAW151/EMA151/137147523参数、Adam151且step=U、EMA step、8逻辑RNG、cursor=96U。原冻结训练manifest及CPU审计源码和冻结容差必须匹配。
5. `CPU_POSTPROCESS_ACCEPTANCE`与`CPU_FINALIZER_EXIT`须完成；其refs必须逐角色绑定checkpointSHA、state auditSHA、EMA完整digest和SUMMARY SHA。
6. 当前root的`FINAL_OUTPUTS_CPU_AUDIT`必须是本root wrapper产生的纯CPU PASS，不能使用seed_history复制来的旧聚合回执。最终两角色所有5000保存预测须已独立CPU重算PSNR/SSIM/MAE/LPIPS并全SHA验收。collector只验证既有报告/CSV及其SHA，不重算LPIPS或调用任何指标网络。
7. 每个固定小group独立检查original/EMA/U/digest、source/sample/seed/batch4/group_position顺序，输入和GT身份与冻结VAL_MANIFEST一致，selected array逐文件SHA复核。checkpoint完整SHA只在远端原地读，不传输；best=latest时同一checkpoint只重新hash一次。

## 生成内容与科学口径

- `ORIGINAL_TRAIN_GLOBAL.png/.pdf/.csv`：最初D0 U1起至真实最终U，连续完整optimizer-update轨迹；loss是native MatchingLoss L1。保留累计U，拒绝断点、重复或非1e-5的LR。
- `ORIGINAL_VAL_4METRICS_MONITOR100.png/.pdf`、对应CSV：G1/G4各100=同25来源×4。旧D0六个full节点与后续完整full都按同25来源重聚合，不把full均值混成monitor均值。
- `ORIGINAL_VAL_4METRICS_FULL1000.png/.pdf`、对应CSV：完整五档G0.5/1/2/3/4各1000=250来源×4；四指标PSNR↑、SSIM↑、MAE↓、LPIPS↓。G0.5不在训练混合中。
- `ORIGINAL_FINAL_BEST_LATEST_G1/G2/G3/G4.png/.pdf`：各固定两来源首sample，四列Input/最终Best/最终Latest/GT。若Best=Latest，在图中说明两列为同一权重。只显示时clip[0,1]，指标仍为rawfloat。
- `BUNDLE_MANIFEST.json`、`data/FINAL_ORIGINAL_SOURCES.json`、`evidence/final_pipeline/`：保留原始数据路径、SHA、完整角色证据、固定group原件与生成方式。只传固定小数组，最多128MiB bundle，禁止`.pt`和全量预测包。

收集时间只表示附件收集时间。不得将附件生成、CPU PASS或loss下降写成恢复成功、收敛或泛化成功。

## 目视与最终交付仍由root完成

生成后`ORIGINAL_ATTACHMENT_RECEIPT.json`为`ARTIFACTS_GENERATED_ROOT_VISUAL_QA_PENDING`，`final_delivery_complete=false`。root须新目视全部7个PNG（train、monitor四指标、full四指标、G1–G4），将实际PNG SHA写入本附件`VISUAL_QA.json`。还需主任务全进程退出审计、最终报告/CSV/权重备份本地验收；本目录不改自动任务，不自行发送用户通知。

## 自检

```sh
/usr/bin/python3 '/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/delivery_preparation/test_preparation.py'
```

自检仅使用明确标记的临时toy文件，验证拒绝路径、角色绑定、流式tar安全和曲线拼接；不调用恢复/指标网络，不执行最终采集/生成。fixture写在本目录内，运行后自动清除。真实缺完工证据dry-run另有带时间的独立回执。源码SHA见`PREPARATION_SOURCE.json`；该manifest不属于训练冻结源。

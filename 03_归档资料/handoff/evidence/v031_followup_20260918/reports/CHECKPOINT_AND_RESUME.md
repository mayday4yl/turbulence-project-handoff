# 完整状态与恢复入口

两项正式训练已停在1000；本次没有启动1000以后的训练。完整状态在de36，各自只有一个formal目录。

CPU验收入口：`bash code/run_single_env.sh code/verify_boundary.py --checkpoint TASK/formal/checkpoint_001000.pt --output TASK/COMPLETE_1000.json`（将TASK替换为对应任务）。

恢复入口为 `code/train_worker.py --mode train --config configs/TASK_formal.json --output TASK/formal --microbatch MB --resume TASK/formal/checkpoint_NNNNNN.pt --run-until-update 1000`，由 `code/training_controller.py:command` 的四rank torch.distributed.run和现有资源guard启动。T1 MB8，T2 MB1。意外中断时使用同目录最新完整兼容checkpoint，不从初始化重训；当前两项已经到终点，无待恢复工作。

若未来明确授权超过1000，需要带有新授权上限的续训配置和 `--authorize-boundary-extension`，保留原始计划中1000以后floor LR规则以及全部身份校验；不能只改命令行步数。本次未生成或执行该扩展。

- T1_DIRECT: `/data2/yanglin/v031_followup_20260918/T1_DIRECT/formal/checkpoint_001000.pt`；SHA256 `46cf6879e587089d645c8de767b128d48d585cb981bb6f0551c28808f9d7d7ff`。完整raw FP32、AdamW、scheduler、4rank RNG、sampler/counters验收通过。

- T2_FEEDBACK_R: `/data2/yanglin/v031_followup_20260918/T2_FEEDBACK_R/formal/checkpoint_001000.pt`；SHA256 `80cbb6bd56d7c941395a833a56c9389fe1943d712110b495f65c4e6d0d4cae59`。完整raw FP32、AdamW、scheduler、4rank RNG、sampler/counters验收通过。


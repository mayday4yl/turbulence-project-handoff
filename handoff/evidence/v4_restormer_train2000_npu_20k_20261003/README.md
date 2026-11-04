> 最新监控状态（2026-10-05）：用户已回来，Codex本地30分钟自动监控已恢复ACTIVE；四方法健康、完整canonical镜像已验收。见 reports/reconnect_20261005/RECONNECT_REPORT.md。

# Offline monitoring handoff — 2026-10-03

用户离开校园服务器可达网络约两天前，完整保存/自动续训和服务器看护自主重启已实测通过。当前入口：[`handoffs/OFFLINE_TRAINING_HANDOFF_20261003.md`](<../../NOT_INCLUDED.md#ref-b6014317a965013a1ca0>)。证据：`reports/offline_20261003/`。本地断网不停止服务器任务；回校后先查实际进程/健康历史，不能重复启动。所有PID/update仅对应快照。

# Latest execution authority — A100 fresh U0

User explicitly corrected: A100 starts from shared random U0, not NPU checkpoints. See `protocol/A100_FRESH_U0_AUTHORIZATION_20261003.json`. Four separate fresh runs: `runs/a100_fresh_u0_20k` on A100 GPUs3/5 (D0+R08 / R0+L08). CUDA acceptance PASS for all4. Original NPU workers remain safely stopped and results preserved. Checkpoint migration cancelled. Earlier continuation/platform instructions below are historical and superseded.

# Current execution override — 2026-10-03

User explicitly requested stopping NPU and continuing on A100. See `protocol/A100_MIGRATION_AUTHORIZATION_20261003.json` and `reports/migration_a100_20261003/`. NPU workers/controller stopped with full checkpoints D0 U669, R0 U196, R08 U198, L08 U201. Keep NPU stopped. Original scientific specification and historic NPU records stay frozen. A100 has an independent migration execution directory; acceptance and imported checkpoint evidence govern launch. Older platform statements below are historical, superseded.

# V4 Restormer Train2000 / de36 NPU 20k

Authorized on 2026-10-03. Independent new experiment; historical assets read only.

- de36 NPU 0/1/2/3: D0/R0/R08/L08, 20,000 updates each, effective batch 32.
- A100: complete mirrored assets only, no formal training.
- Canonical Places 50-category 2000/250/200 clean split, 4/2/2 realizations, G1-4.
- Single canonical tensor identity; train-only statistics; test locked.
- Backbone reuses pinned D2Turb Restormer candidate and random learned parameters.
- Physics, four-method rollout and L1 objective preserved.

Authority: protocol/SCIENTIFIC_SPEC.json and protocol/AUTHORIZATION.json.
Preparation branches and later training status must be read from their evidence; this README is not a launch/completion claim.

2026-10-05最新监控设置：按用户指令，本地自动核验改为每90分钟，ACTIVE。服务器训练/故障恢复守护频率不变。参见 reports/reconnect_20261005/MONITOR_INTERVAL_90MIN.json。

# V4 Overfit100 controlled training

This is a separate training implementation derived from the preserved `v4_noise_bridge_20260929` correctness implementation. The user authorized this phase in `protocol/AUTHORIZATION.json`; the prior two-step correctness cap is not a cap on newly authorized formal training. Simulator, model, normalization, rollout and serial refinement mathematics are preserved.

Remote root: `/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929` on `10.94.2.1` / `h3c-gpu-server01`.
Python: `/home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python`.
Only physical GPU5/6 (UUIDs in launch_phase.sh) are visible. Shell locks and live GPU guard reject existing occupants. No third GPU or other users' processes are touched.

## Validated preparation

- CPU 24 tests, including prior normalization/refinement contracts.
- Two ranks: D0/R0/R08 real-network and real-simulator gradient comparison to per-sample full-path mean; R0/R08 terminal-only BPTT and detach negative control; exact formal checkpoint model/EMA/optimizer roundtrip and next-batch update continuity; serial inference refinement.
- Temporary validation: 9 global optimizer updates (each replicated on both DDP ranks), kept separate from the prior one-step correctness ledger.
- Benchmark: 12 group/configuration cases, three temporary updates each, total36. Model weights from these checks are never used for formal training.
- Selected B32/micro4/accumulation4 validation: zero optimizer updates; full gradient comparison passed on both ranks.
- Actual distributed evaluator writer is checked before freeze.

All GPU checks use actual 256 RGB, 512 native raw-xi, same P2S assets, FP32, no autocast/TF32. CUDA grid_sample backward remains nondeterministic; no bitwise trajectory equivalence claim. Scalar refinement remains the only production implementation.

## Frozen execution

`protocol/FROZEN.json` is created only after preparation passes and binds code, configuration, data, common initialization, sampler stream, evaluation grouping, and evidence. Default final candidate: world2, B32, micro4 per rank, accumulation4. Always read the actual frozen file.

Training order: R08, R0, D0. Each sees precisely32000 starts, the identical20 shuffled permutations of all1600 starts. No drops, duplicated DDP padding, variable batch, adaptive training length or validation-score hyperparameter selection. The selected divisor batch exactly hits all3200-exposure nodes. Training never refines xi. Main evaluation is raw/EMA, refinement-off; raw last_32k is the primary comparison. After all three main comparisons complete, R08 and R0 each get one raw last_32k val800 serial refinement-on evaluation.

```bash
cd /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929
# Run only a phase that is not already active. Never duplicate tmux/processes.
bash scripts/launch_phase.sh validation
bash scripts/launch_phase.sh benchmark
bash scripts/launch_phase.sh selected
bash scripts/launch_phase.sh evaluator
bash scripts/launch_phase.sh freeze
# This is the authorized formal workflow, starts or resumes from full saved state:
bash scripts/run_formal.sh
```

Use a detached tmux session with output to reports/formal.log. `LATEST.json` commits the full checkpoint. On resume, E/u, model/EMA/optimizer and all ranks' CPU/CUDA/eta RNG are restored. Evaluations are skipped only if protocol and checkpoint hashes match. A failure stops the workflow; it does not silently change batch, precision, refinement or sample budget. Resume events and failed attempts are recorded separately. Successful updates after a checkpoint but before an unexpected crash may be recomputed from that checkpoint; only the recovered final trajectory counts toward exposure totals, and old attempt logs remain.

For clean planned interruption, signal the two worker processes with SIGTERM; they checkpoint after a complete update. Do not kill other processes, reset devices, or terminate the whole server. An evaluation interruption may require replaying that incomplete evaluation.

## Outputs

Comparison root: `/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/runs/v4_overfit100_paired_20260929`.
`STATUS.json` distinguishes training, evaluation, final paired refinement and completion. Each group contains update logs, full checkpoints, metadata for last_32k/best_val and per-start evaluation products. Final CPU summarization emits PAIRED_RESULTS.json, FINAL_REPORT.md, last32k_source_G.csv and learning_curves.png. No test200, train500, historical V3 run, extra seed, extra T0/lambda search, or automatic continuation beyond32000 exposures is included.

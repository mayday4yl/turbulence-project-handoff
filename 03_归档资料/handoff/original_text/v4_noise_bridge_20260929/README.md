# V4 RGB Noise Bridge — implementation and correctness only

This directory is independent of historical V3/V0.31. Authority: the user's 2026-09-29 implementation request and `docs/USER_ALGORITHM.png`. Mapping is in `docs/CODE_MAP.md`. No formal training loop or effect/PSNR Pilot is started by these tools.

The CUDA copy is `/home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_rgb_20260929` on `10.94.2.1` (`h3c-gpu-server01`). It references the existing read-only physical dataset and simulator assets; original V3 source/config/data/checkpoints are unchanged. The original model/simulator sources under `physical_bridge/` are hash-pinned, not modified.

## Implemented interface

```python
stats = FrozenRGBStats.from_artifact('reports/normalization_overfit100.json')
network = NoiseBridgeUNet(stats, **config['network']).to(device)
simulator = PiRNP2SSimulator(config['asset_dir'], **config['physics'], device=device)
rollout = RolloutModule(network, simulator, T0=0.8)
noise_rng = torch.Generator(device=device).manual_seed(recorded_noise_seed)
result = rollout(C, Y, G, noise_generator=noise_rng)
# result['loss_mean'] is the full differentiable path-mean / batch-mean objective.
# A caller may backward/update only under its own explicit training authorization.
network.eval()
result = infer(network, simulator, Y, G, noise_seeds=per_sample_seeds,
               T0=0.8, refinement=True)
```

Modules are `v4.model`, `v4.core`, `v4.inference`, `v4.normalization`, `physical_bridge.simulator`. Independent statistics/data tools are `v4.data` and `v4.normalization`; runtime identity and checkpoint helpers are in `v4.runtime`.

Normalization is explicitly **Overfit100 training population**, 400 trajectories × 4 states = 1600 unique states, one full stored 256×256 view per state. It is not the full500 train population's statistics. Validation/test/clean/rollout states are excluded. Changing the intended training population requires a separate bound statistics artifact.

## Reproducible commands on A100

```bash
cd /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_rgb_20260929
V4_PY=/home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CUDA_VISIBLE_DEVICES= "$V4_PY" -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES= "$V4_PY" -m v4.normalization \
  --dataset-root /home/zhuzhiyu/yanglin/turbulence_ir/shared/data/generated/places_p2s_a100_20260928_g4321_s500_v100_t200_v1 \
  --asset-lock /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/ASSET_LOCK.json \
  --population overfit100 --output reports/normalization_recomputed.json

"$V4_PY" scripts/correctness.py --phase forward
"$V4_PY" scripts/correctness.py --phase bptt
"$V4_PY" scripts/mixed_backward.py
"$V4_PY" scripts/correctness.py --phase refinement
```

CUDA scripts select only the configured GPU5 UUID and reject another compute process there. Run these phases sequentially. They use the full 32,038,377-parameter network, 256px RGB inputs, native 512px raw-xi and unchanged actual simulator, FP32 with TF32/autocast disabled.

`--phase smoke` performs **one** temporary network optimizer step and deletes its temporary checkpoint after model/optimizer/normalization/RNG roundtrip. It is not part of the no-update commands above. The persistent `reports/optimizer_step_ledger.json` enforces a total maximum of two attempted steps for this implementation; the delivered run uses one. No formal checkpoint is produced. Its AdamW configuration is a test fixture, not a proposed formal training protocol.

## Refinement execution and reproducibility boundary

Production `refine_xi` executes each sample separately with its own fresh optimizer, preserving the fixed objective, native two-block reduction, 40 Adam updates and 41 candidates. Optional `audit_updates=True` independently checks each actual update from the same observed gradients in float64, and verifies selected paired candidate hashes; it does not replace simulator gradients or alter the objective.

The original vectorized 40-step raw-xi batch/serial parity test **failed** its preset tolerance. A repeated single-sample run also failed this strict raw-xi tolerance. The unchanged simulator's actual CUDA backward errors in strict deterministic mode specifically at `grid_sampler_2d_backward_cuda`. Same-point gradients verified absence of batch-mean scaling; this does not prove equality of independent 40-step trajectories. The failure remains in `initial_failed_cuda_refinement.json`, `refinement_batch_diagnosis.json`, and `refinement_gradient_diagnosis.json`.

The vectorized solver is private (`_solve_native`) and is not a supported batched production path; it remains solely to reproduce the diagnostic. Serial execution is not a fix for CUDA's repeated-run nondeterminism. Fixed eta seeds reproduce noise, not necessarily optimized raw-xi bit-for-bit. No simulator rewrite, clamp, gradient normalization, new stopping rule, T0/lambda search or tolerance relaxation was used.

#!/usr/bin/env python3
"""Disposable, zero-optimizer coexistence check at the real update500 state.

Uses the unchanged production loader/rollout and the existing DDP comparison.
Inference replays the first eight G4 starts against completed isolated raw500
records. Timing fields are not numerical comparison targets. No new training
samples, checkpoints, solver settings, or optimizer updates are produced.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from datetime import timedelta


def wait_for(path):
    deadline = time.monotonic() + 300
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Coexistence peer readiness timeout: {path}')
        time.sleep(0.2)


def gradient(args, config):
    import torch
    import torch.distributed as dist
    from check_distributed_gradients import (backward_pass, gradient_comparison, forward_comparison,
                                             GRADIENT_RELATIVE_L2_MAX, GRADIENT_COSINE_MIN)
    from train_worker import make_engine, resolved_identity, resume_snapshot
    from v031.data import StartDataset
    from v031.distributed_runtime import configure_local_rank
    from v031.runtime import save_json, sha256, verify_source
    from v031.training import (capture_rank_rng, restore_rank_rng, load_training_checkpoint,
                               compare_states, memory_stats)
    rank = int(os.environ['LOCAL_RANK'])
    runtime = configure_local_rank(config, rank)
    verify_source(config)
    dist.init_process_group('hccl', timeout=timedelta(minutes=10))
    try:
        dataset = StartDataset(config['dataset_root'])
        engine = make_engine(config, dataset, resolved_identity(config, dataset),
                             argparse.Namespace(mode='train', microbatch=1), rank)
        restored = load_training_checkpoint(engine, args.checkpoint, run_until_update=500)
        before = resume_snapshot(engine)
        indices = engine.sampler.next_indices()
        local_indices = indices[rank * 8:(rank + 1) * 8]
        start_rng = capture_rank_rng(engine.device)
        dist.barrier()
        if rank == 0:
            save_json(args.output / 'gradient_ready.json', {'utc_epoch': time.time()})
        wait_for(args.output / 'inference_ready.json')
        dist.barrier()
        started = time.time()
        reference_metrics, reference = backward_pass(engine, local_indices, ddp=False, microbatch=1)
        restore_rank_rng(start_rng, engine.device)
        actual_metrics, actual = backward_pass(engine, local_indices, ddp=True, microbatch=1)
        ended = time.time()
        comparison = gradient_comparison(reference, actual)
        forwards = forward_comparison(reference_metrics, actual_metrics)
        # Return the disposable sampler/RNG to the restored boundary solely to
        # prove every saved training state stayed unchanged by these backward calls.
        engine.sampler.load_state_dict(before['sampler'])
        restore_rank_rng(before['rng'], engine.device)
        unchanged = compare_states(before, resume_snapshot(engine))
        branches = {}
        for branch in ('clean_head', 'zernike_head', 'tilt_head'):
            names = [name for name in reference if name.split('.')[0] == branch]
            if not names:
                raise RuntimeError(f'Missing head gradients: {branch}')
            branches[branch] = gradient_comparison({n: reference[n] for n in names}, {n: actual[n] for n in names})
        checks = [comparison] + list(branches.values())
        numeric = all(c['reference_norm'] > 0 and c['cosine'] >= GRADIENT_COSINE_MIN
                      and c['relative_l2_difference'] <= GRADIENT_RELATIVE_L2_MAX for c in checks)
        passed = (numeric and forwards['passed'] and unchanged['exact']
                  and engine.optimizer_step_calls == 0 and engine.optimizer_step_attempts == 0
                  and engine.completed_updates == 500)
        record = {'status': 'passed' if passed else 'failed', 'rank': rank,
                  'checkpoint_sha256': sha256(args.checkpoint), 'runtime': runtime,
                  'restored': restored, 'gradient': comparison, 'branches': branches,
                  'forward': forwards, 'all_restored_states_unchanged': unchanged['exact'],
                  'optimizer_step_calls': engine.optimizer_step_calls,
                  'optimizer_step_attempts': engine.optimizer_step_attempts,
                  'work_start': started, 'work_end': ended, 'memory': memory_stats(engine.device),
                  'global_start_ids': [dataset.records[i]['start_id'] for i in indices],
                  'thresholds': {'relative_l2_max': GRADIENT_RELATIVE_L2_MAX, 'cosine_min': GRADIENT_COSINE_MIN}}
        save_json(args.output / f'gradient_rank{rank}.json', record)
        if not passed:
            raise RuntimeError('Concurrent full-state/forward/gradient acceptance failed')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def inference(args, config):
    import torch
    import torch_npu  # noqa: F401
    from v031.runtime import build, configure_npu, save_json, state_hash, sha256
    from v031.data import StartDataset
    from v031.evaluation import (_checkpoint_state, rollout_predictions, clean_metrics,
                                 validate_completed_calibration)
    from v031.refinement import synchronize
    from check_distributed_gradients import FORWARD_ATOL, FORWARD_RTOL
    configure_npu(config)
    device = torch.device('npu:0')
    network, simulator = build(config, device)
    state, update = _checkpoint_state(args.checkpoint, config, 'raw')
    if update != 500:
        raise RuntimeError('Probe is bound to checkpoint500')
    network.load_state_dict(state, strict=True)
    del state
    root = Path(__file__).resolve().parent.parent
    calibration = json.loads((root / 'artifacts/calibration_update100/calibration.json').read_text())
    validate_completed_calibration(calibration, config)
    dataset = StartDataset(config['dataset_root'])
    selected = [i for i, row in enumerate(dataset.records) if row['initial_G'] == 4][:8]
    reference_file = root / 'artifacts/evaluation/update_000500/raw/refined/per_start.jsonl'
    references = {r['start_id']: r for r in map(json.loads, reference_file.read_text().splitlines())}
    network_before, simulator_before = state_hash(network), state_hash(simulator)
    save_json(args.output / 'inference_ready.json', {'utc_epoch': time.time()})
    wait_for(args.output / 'gradient_ready.json')
    started = time.time()
    comparisons, differences = [], []
    exact = True
    for index in selected:
        sample = dataset[index]
        target = sample['clean'].unsqueeze(0).to(device)
        observed = sample['observation'].unsqueeze(0).to(device)
        predictions = rollout_predictions(model=network, simulator=simulator, observation=observed,
                                          initial_G=4, mode='refined', solver_config=calibration['selected'])
        synchronize(device)
        original = references[sample['start_id']]
        checked = 0
        for k, prediction in enumerate(predictions):
            prior = original['predictions'][k]
            metrics = clean_metrics(prediction.clean, target)[0]
            pairs = [(metrics['clean_mse'], prior['clean_mse']),
                     (prediction.network_redegradation_mse[0], prior['network_redegradation_mse'])]
            if prediction.refinement:
                current, old = prediction.refinement[0], prior['refinement']
                for key in ('initialization_source', 'iterations', 'gradient_connected', 'finite',
                            'nonzero_zernike_update', 'nonzero_tilt_update'):
                    if current[key] != old[key]:
                        raise RuntimeError(f'Concurrent refinement decision changed: {key}')
                for key in ('previous_candidate_objective', 'network_candidate_objective',
                            'selected_initial_objective', 'best_objective'):
                    if current[key] is None or old[key] is None:
                        if current[key] != old[key]:
                            raise RuntimeError('Refinement candidate presence changed')
                    else:
                        pairs.append((current[key], old[key]))
            for actual, expected in pairs:
                if not (math.isfinite(actual) and math.isfinite(expected)
                        and abs(actual - expected) <= FORWARD_ATOL + FORWARD_RTOL * abs(expected)):
                    raise RuntimeError('Concurrent inference differs from isolated reference')
                exact = exact and actual == expected
                differences.append(abs(actual - expected))
                checked += 1
        comparisons.append({'start_id': sample['start_id'], 'legal_prefixes': len(predictions),
                            'checked_scalar_quantities': checked})
    ended = time.time()
    if network_before != state_hash(network) or simulator_before != state_hash(simulator):
        raise RuntimeError('Inference changed frozen state')
    save_json(args.output / 'inference.json', {'status': 'passed', 'checkpoint_sha256': sha256(args.checkpoint),
              'work_start': started, 'work_end': ended, 'scalar_metrics_exact': exact,
              'max_absolute_metric_difference': max(differences), 'comparisons': comparisons,
              'reference_file': str(reference_file), 'reference_sha256': sha256(reference_file),
              'thresholds': {'atol': FORWARD_ATOL, 'rtol': FORWARD_RTOL},
              'network_unchanged': True, 'simulator_unchanged': True, 'optimizer_updates': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('gradient', 'inference'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    import sys
    sys.path.insert(0, config['base_source'])
    (gradient if args.role == 'gradient' else inference)(args, config)

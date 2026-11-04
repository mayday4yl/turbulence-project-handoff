#!/usr/bin/env python3
"""Materialize the reviewed finite formal protocol; never launch accelerator work.

Requires all paired-pilot evidence and a separate root scientific review. The
formal config is immutable and contains no initialization checkpoint.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    p = args.project.resolve()
    sys.path.insert(0, str(p))
    from v03.common import choose_sample, sha256_file, unique_schedules, verify_base
    report = read(p / 'artifacts/half_lambda_pilot_01_review.json')
    review = read(p / 'artifacts/half_lambda_pilot_01_scientific_review.json')
    if (report['status'] != 'paired_evidence_passed_pending_scientific_review'
            or review.get('method_level_blocker') is not False
            or review.get('reviewed_all_candidates') is not True
            or review.get('pilot_report_sha256') != sha256_file(p / 'artifacts/half_lambda_pilot_01_review.json')):
        raise ValueError('Complete paired evidence and explicit scientific review required')
    selected = report['predeclared_selection']['recommended_lambda_R']
    selected_id = report['predeclared_selection']['recommended_pilot_id']
    config = read(p / 'configs' / f'{selected_id}.json')
    verify_base(config)
    config.update(seed=20260915, sampling_seed=2026091503,
                  purpose='Unique fresh-from-scratch V0.3-half formal run; no pilot weights.')
    config['validation'].update(weight_key='ema', selection='sha256-key-v1')
    config['training'].update(
        maximum_updates=10000, learning_rate=1e-4, final_learning_rate=1e-6,
        warmup_updates=64, ema_decay=.999 ** 4, checkpoint_interval=500,
        milestone_interval=2000, rolling_keep=3, diagnostic_interval=500,
        checkpoint_prepare_at_utc='2026-09-16T03:30:00Z',
        pause_at_utc='2026-09-16T03:50:00Z', yield_at_safe_checkpoint=True)
    for stage, boundary in zip(config['curriculum'], [128/10000, 512/10000, 1.0], strict=True):
        stage['until_fraction'] = boundary
    assert config['training']['lambda_R'] == selected
    assert config['training']['microbatch'] == 2 and config['training']['effective_batch'] == 32
    probabilities = []
    for index, stage in enumerate(config['curriculum']):
        rows = []
        for severity in [.75, 1.5, 2.25, 3., 3.75, 4.5]:
            paths = unique_schedules(config, severity)
            weights = [stage['weights_by_effective_K'][str(len(path)-1)] for path in paths]
            for path, weight in zip(paths, weights, strict=True):
                rows.append({'G': severity, 'schedule': list(path), 'prediction_count': len(path),
                             'conditional_probability': weight/sum(weights),
                             'joint_probability': weight/sum(weights)/6})
        probabilities.append({'stage': index, 'until_fraction': stage['until_fraction'], 'paths': rows,
            'one_shot_probability': sum(r['joint_probability'] for r in rows if r['prediction_count'] == 1),
            'at_least_3_transitions_probability': sum(r['joint_probability'] for r in rows if r['prediction_count'] >= 4),
            'mean_predictions_per_trajectory': sum(r['joint_probability']*r['prediction_count'] for r in rows)})
    if probabilities[-1]['one_shot_probability'] < .40 or probabilities[-1]['at_least_3_transitions_probability'] < .27:
        raise ValueError('Proposed formal exposure floors are not met')

    # Replay only the stateless sampler on CPU. This reads manifest metadata,
    # not image arrays, simulator states, or held-out test samples.
    milestones, path_counts, g_counts, source_trajectories = {}, Counter(), Counter(), set()
    predictions = 0
    for update in range(10000):
        for sample_id in range(update*32, (update+1)*32):
            sample = choose_sample(config, sample_id, update)
            predictions += len(sample['schedule'])
            path_counts[str(tuple(sample['schedule']))] += 1
            g_counts[str(sample['G'])] += 1
            source_trajectories.add(sample['key'])
        if update+1 in [128, 512, 2000, 4000, 6000, 8000, 10000]:
            milestones[str(update+1)] = {'optimizer_updates': update+1,
                'sampled_trajectories': (update+1)*32, 'network_predictions': predictions,
                'trajectory_equivalent_passes': (update+1)*32/4000,
                'distinct_train_trajectory_keys_seen': len(source_trajectories)}
    val_rows = [json.loads(line) for line in (Path(config['dataset_root'])/'manifests/val.jsonl').read_text().splitlines()]
    val_keys = [r['key'] for r in sorted(val_rows, key=lambda r: hashlib.sha256(('v03-val-v1:'+r['key']).encode()).hexdigest())]
    if len(val_keys) != 200 or len(set(val_keys)) != 200:
        raise ValueError('Unexpected validation identity')
    policy = {
        'protocol_frozen': True, 'run_id': 'half_formal_01',
        'config': 'configs/half_formal_01.json', 'config_hash': digest(config),
        'source_commit': args.source_commit, 'guard_memory_gib': 58,
        'fast_validation_interval': 1000, 'fast_validation_trajectories': 16,
        'fast_validation_budget_seconds': 600,
        'full_validation_interval': 2000, 'full_validation_budget_seconds': 5400,
        'validation_weight_key': 'ema', 'validation_G': [.75,1.5,2.25,3.,3.75,4.5],
        'validation_K': [0,1,3,5], 'fast_validation_keys': val_keys[:16],
        'full_validation_keys_sha256': digest(val_keys),
        'early_stop': {'minimum_updates': 4000, 'plateau_intervals': 3,
                       'minimum_improvement_db': .05, 'catastrophic_k0_drop_db': 3.0},
        'continuation': {'primary_updates': 8000, 'maximum_updates': 10000,
                         'lookback_updates': 2000, 'minimum_improvement_db': .05,
                         'k0_preservation_db': .10},
        'checkpoint_selection': {
            'primary': 'maximum absolute macro PSNR, equal six G and equal K=0/1/3/5',
            'k0_preservation_db': .10, 'psnr_tie_db': .03, 'lpips_tie': .005,
            'tie_break': 'lower macro LPIPS; within LPIPS tolerance prefer earlier update',
            'same_subset_and_weight_type_required': True, 'final_selection_requires_full_validation': True,
            'mechanism_diagnostics_choose_checkpoint': False},
        'progressive_gain': {
            'aggregation': 'same checkpoint: equal K=1/3/5 minus K0, then equal six G',
            'minimum_psnr_delta_db': .10, 'minimum_positive_G_count': 4,
            'minimum_per_G_psnr_delta_db': -.20, 'maximum_lpips_delta': .005,
            'minimum_ssim_delta': -.001, 'k0_preservation_required': True},
        'readiness': {
            'declared_training_end_required': True, 'operational_pause_is_not_training_end': True,
            'full_validation_and_selection_required': True, 'progressive_gain_and_k0_gate_required': True,
            'identity_resume_gradient_checks_required': True, 'held_out_test_authorized': False,
            'causal_recursive_only_claim_authorized': False},
        'numerical_recovery': {
            'hyperparameter_changes_within_run': False,
            'rule': 'Stop on nonfinite/divergence; preserve latest complete checkpoint and logs. '
                    'At most one same-config resume for a verified transient engineering interruption, '
                    'after ownership/health checks. No repeating failed numerical update indefinitely. '
                    'Any LR/batch/protocol change requires a new identity; protocol change starts fresh.'},
        'minimum_exposure': {'global_one_shot_all_stages': .40,
                             'final_stage_global_at_least_3_transitions': .27,
                             'final_stage_G4p5_full_K5_conditional': .30},
    }
    pilot_phases = []
    for stage_id, (lo, hi) in enumerate([(5,16), (17,32), (33,64)]):
        values = []
        for row in report['rows']:
            timings = [json.loads(line) for line in (p/'runs'/row['id']/'timing.jsonl').read_text().splitlines()]
            values.extend(r['full_update_wall_seconds'] for r in timings if lo <= r['update'] <= hi and r['update'] not in [32,33,64])
        pilot_phases.append({'stage': stage_id, 'mean_full_update_seconds': statistics.fmean(values),
                             'measured_updates': len(values), 'scope': 'paired pilot same global32 and actual-path distribution; excludes checkpoint/restart steps'})
    pilot_val_seconds = [read(p/'runs'/row['id']/'validation/update-000064/summary.json')['wall_seconds'] for row in report['rows']]
    fast_seconds = statistics.fmean(pilot_val_seconds)*2
    full_seconds = statistics.fmean(pilot_val_seconds)*25
    for count, row in milestones.items():
        n = int(count)
        training_seconds = (min(n,128)*pilot_phases[0]['mean_full_update_seconds']
            + max(0,min(n,512)-128)*pilot_phases[1]['mean_full_update_seconds']
            + max(0,n-512)*pilot_phases[2]['mean_full_update_seconds'])
        validation_seconds = (n//1000)*fast_seconds+(n//2000)*full_seconds
        row.update(estimated_training_hours=training_seconds/3600,
                   estimated_validation_hours=validation_seconds/3600,
                   estimated_total_active_hours=(training_seconds+validation_seconds)/3600,
                   estimated_NPU_hours=4*(training_seconds+validation_seconds)/3600,
                   eta_scope='Active compute estimate; excludes resource waiting, noon pause and later user-authorized downtime. Validation extrapolation may vary with convergence/refinement early stopping.')
    freeze = {
        'identity': 'half_formal_01', 'frozen_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_commit': args.source_commit, 'config_hash': digest(config), 'policy_hash': digest(policy),
        'pilot_report_sha256': sha256_file(p/'artifacts/half_lambda_pilot_01_review.json'),
        'scientific_review_sha256': sha256_file(p/'artifacts/half_lambda_pilot_01_scientific_review.json'),
        'lambda_R': selected, 'source_baseline': config['base_v01_hashes'],
        'dataset_identity': 'places_p2s_bridge_v03_half', 'dataset_manifest_sha256': {
            split:sha256_file(Path(config['dataset_root'])/'manifests'/f'{split}.jsonl') for split in ['train','val','test']},
        'seeds': {'formal': config['seed'], 'sampler': config['sampling_seed']},
        'initialization_checkpoint': None, 'all_training_state_fresh': True,
        'hardware': {'host':'liteserver-de36','physical_cards':[0,1,2,3], 'microbatch_upper_bound':2,
                     'actual_global_paths_per_update':32,'accumulation':'actual length-balanced chunks, usually4-5 per rank'},
        'actual_schedule_probabilities': probabilities, 'milestone_budgets_and_ETA': milestones,
        'full_horizon_G_counts': dict(g_counts), 'full_horizon_schedule_counts': dict(path_counts),
        'pilot_phase_timing': pilot_phases, 'formal_started': False,
        'time_target': 'Today11:30 safe checkpoint,11:50 resume_ready pause; no automatic noon continuation.',
        'budget_rationale': 'Pinned V0.1 configured15000/30000updates atbatch16/full8000 implies30/60trajectory passes. '
                            'Half4000/global32 yields3750/7500 for matching trajectory exposure; use4000minimum, '
                            '8000primary,10000finite conditionalmaximum. Recursive prediction/compute is reported separately.',
        'EMA_rationale': '.999**4 preserves old batch16/full8000 EMA window in dataset-pass units atbatch32/half4000. Not pilot-tuned.',
    }
    for file, document in [('configs/half_formal_01.json',config),
                           ('configs/half_formal_01_policy.json',policy),
                           ('artifacts/half_formal_01_freeze.json',freeze)]:
        if (p/file).exists():
            raise FileExistsError('Formal protocol identity already exists; never overwrite')
    write_new(p/'configs/half_formal_01.json',config)
    write_new(p/'configs/half_formal_01_policy.json',policy)
    write_new(p/'artifacts/half_formal_01_freeze.json',freeze)
    print(json.dumps({'lambda_R':selected, 'config_hash':digest(config), 'milestones':milestones}))


if __name__ == '__main__':
    main()

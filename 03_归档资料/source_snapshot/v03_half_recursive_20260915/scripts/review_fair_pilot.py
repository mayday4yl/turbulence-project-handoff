#!/usr/bin/env python3
"""Audit paired pilot evidence and apply its predeclared validation ranking."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rank_candidates(rows):
    best_k0 = max(r['K0_PSNR'] for r in rows)
    preserved = [r for r in rows if r['K0_PSNR'] >= best_k0 - .1]
    best_score = max(r['macro_G_K_PSNR'] for r in preserved)
    close = [r for r in preserved if r['macro_G_K_PSNR'] >= best_score - .03]
    best_lpips = min(r['macro_G_K_LPIPS'] for r in close)
    tied = [r for r in close if r['macro_G_K_LPIPS'] <= best_lpips + .005]
    chosen = min(tied, key=lambda r: [.1, .05, .2].index(r['lambda_R']))
    return {'recommended_lambda_R': chosen['lambda_R'], 'recommended_pilot_id': chosen['id'],
            'preserved_candidate_ids': [r['id'] for r in preserved],
            'close_score_candidate_ids': [r['id'] for r in close],
            'predeclared_tie_preference_used': len(tied) > 1,
            'basis': 'Absolute validation quality with K0 preservation; never relative gap alone.',
            'statistical_superiority_claimed': False, 'formal_weights_must_be_fresh': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    p = args.project.resolve()
    manifest = read(p / 'artifacts/half_lambda_pilot_01_manifest.json')
    rows, initial_hashes, path_signatures, learning_rates, normalized_configs = [], [], [], [], []
    for variant in manifest['variants']:
        run = Path(variant['run_dir']); config = read(p / variant['config'])
        config['training']['lambda_R'] = None; normalized_configs.append(config)
        identity = read(run / 'identity.json')
        initial_hashes.append(identity['initial_model_hash'])
        metrics = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines()]
        if [r['update'] for r in metrics] != list(range(1, manifest['paired_optimizer_updates'] + 1)):
            raise ValueError('Candidate update budgets are incomplete or duplicated')
        if any(not math.isfinite(float(r[key])) for r in metrics
               for key in ['loss', 'clean_loss', 'reconstruction_loss', 'gradient_norm_before_clip']):
            raise ValueError('Nonfinite training diagnostic')
        audit = manifest['prelaunch_fairness_and_exposure_audit']
        if sum(r['actual_global_paths'] for r in metrics) != audit['expected_sampled_trajectories_per_candidate']:
            raise ValueError('Trajectory exposure mismatch')
        if sum(r['predictions'] for r in metrics) != audit['expected_network_predictions_per_candidate']:
            raise ValueError('Prediction exposure mismatch')
        signatures = []
        observed_paths = collections.Counter()
        for row in metrics:
            ids = sorted(i for rank in row['actual_rank_microbatches'] for chunk in rank['chunks']
                         for i in chunk['real_sample_ids'])
            expected = list(range((row['update'] - 1) * 32, row['update'] * 32))
            if ids != expected:
                raise ValueError('Data order/normalization differs from the fixed global32 protocol')
            observed_paths.update(row['actual_path_counts'])
            signatures.append({'update': row['update'], 'ids': ids, 'paths': row['actual_path_counts']})
        if dict(observed_paths) != audit['actual_schedule_counts']:
            raise ValueError('Actual schedules differ from the predeclared sampler audit')
        path_signatures.append(hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest())
        learning_rates.append([r['learning_rate'] for r in metrics])
        summaries = []
        for update in manifest['validation_updates']:
            s = read(run / 'validation' / f'update-{update:06d}' / 'summary.json')
            if s['status'] != 'complete' or s['split'] != 'val' or s['held_out_test_opened']:
                raise ValueError('Complete validation-only evidence required')
            if s['identity']['keys'] != audit['fixed_validation_keys'] or s['checkpoint_update'] != update:
                raise ValueError('Validation subset/checkpoint mismatch')
            if s['identity']['weight_key'] != 'model' or not s['inference_unchanged']:
                raise ValueError('Pilot inference contract changed')
            if not all(math.isfinite(float(v)) for k in [0, 1, 3, 5]
                       for v in s['per_k'][str(k)]['macro_g'].values()):
                raise ValueError('Nonfinite validation metric')
            summaries.append(s)
        diagnostics = []
        for rank in range(4):
            d = read(run / 'gradient_diagnostics/update-000064' / f'rank-{rank}-results.json')
            if not d['passed'] or not d['weights_unchanged'] or d['optimizer_created'] or d['checkpoint_update'] != 64:
                raise ValueError('Trained checkpoint gradient diagnostic failed')
            diagnostics.append({'rank': rank, 'schedule': d['schedule'], 'sample': d['sample'],
                                'gradient_scale_ratios': d['gradient_scale_ratios']})
        final = summaries[-1]
        rows.append({'id': variant['id'], 'lambda_R': variant['lambda_R'],
                     'K0_PSNR': final['per_k']['0']['macro_g']['psnr'],
                     'macro_G_K_PSNR': final['macro_g_and_k']['psnr'],
                     'macro_G_K_LPIPS': final['macro_g_and_k']['lpips'],
                     'per_k': final['per_k'], 'validation_curve': [
                         {'update': s['checkpoint_update'], 'per_k': s['per_k'],
                          'macro_g_and_k': s['macro_g_and_k']} for s in summaries],
                     'training_first_loss': metrics[0]['loss'], 'training_final_loss': metrics[-1]['loss'],
                     'max_gradient_norm': max(r['gradient_norm_before_clip'] for r in metrics),
                     'diagnostics': diagnostics})
    if len(set(initial_hashes)) != 1 or len(set(path_signatures)) != 1:
        raise ValueError('Initial weights or actual path sequence differ between candidates')
    if not all(c == normalized_configs[0] for c in normalized_configs) or not all(lr == learning_rates[0] for lr in learning_rates):
        raise ValueError('Pilot confounds beyond lambda_R were detected')
    report = {'status': 'paired_evidence_passed_pending_scientific_review', 'rows': rows,
              'fairness': {'identical_initial_model_hash': initial_hashes[0],
                           'identical_actual_sample_path_signature': path_signatures[0],
                           'only_lambda_R_differs': True, 'paired_budgets_complete': True},
              'predeclared_selection': rank_candidates(rows),
              'formal_protocol_frozen': False, 'held_out_test_run': False,
              'interpretation': 'Finite64-update,8-trajectory-validation tuning pilot; not convergence evidence or a final scientific result.'}
    output = p / 'artifacts/half_lambda_pilot_01_review.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['predeclared_selection']))


if __name__ == '__main__':
    main()

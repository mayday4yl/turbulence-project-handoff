#!/usr/bin/env python3
"""Write, but never launch, a finite paired lambda pilot from measured hardware."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--warmup', type=int, required=True)
    parser.add_argument('--validation-trajectories', type=int, required=True)
    parser.add_argument('--maximum-wall-seconds', type=int, required=True)
    parser.add_argument('--not-after-utc', required=True)
    args = parser.parse_args()
    if args.updates < 4 or args.updates % 2 or args.learning_rate <= 0:
        raise ValueError('Positive LR and even paired update budget >=4 required')
    p = args.project.resolve()
    base = json.loads(args.template.read_text())
    if base['workload']['kind'] != 'mixed':
        raise ValueError('Pilot hardware template must be a mixed-path benchmark')
    base['seed'] = 2026091501
    base['sampling_seed'] = 2026091502
    training = base['training']
    training.update(maximum_updates=args.updates, learning_rate=args.learning_rate,
                    final_learning_rate=args.learning_rate, warmup_updates=args.warmup,
                    checkpoint_interval=args.updates // 2, diagnostic_interval=8)
    training.pop('maximum_paths', None)
    training.pop('debug_stall_trace_seconds', None)
    base['curriculum'] = [
        {'until_fraction': .25, 'weights_by_effective_K':
            {'0': .65, '1': .30, '2': .05, '3': .05, '4': .05, '5': .05}},
        {'until_fraction': .5, 'weights_by_effective_K':
            {'0': .40, '1': .30, '2': .15, '3': .15, '4': .15, '5': .15}},
        {'until_fraction': 1., 'weights_by_effective_K':
            {'0': .20, '1': .20, '2': .30, '3': .30, '4': .30, '5': .30}}]
    # The very short pilot assesses the trained model directly. EMA is still
    # built/saved identically, but its mostly-initial weights would obscure a
    # short paired lambda comparison. Formal weight selection remains separate.
    base['validation'].update(weight_key='model', capture_trajectories=1)
    base['purpose'] = 'Paired finite lambda pilot; reset every candidate identically. Never continue these weights into the formal run.'
    variants, steps = [], []
    for coefficient in [.05, .1, .2]:
        name = f'{args.identity}_lambda{str(coefficient).replace(".", "p")}'
        config = copy.deepcopy(base)
        config['training']['lambda_R'] = coefficient
        config_relative = f'configs/{name}.json'
        write(p / config_relative, config)
        run = f'/data2/yanglin/v03_half_recursive_20260915/runs/{name}'
        variants.append({'id': name, 'lambda_R': coefficient, 'config': config_relative,
                         'run_dir': run, 'config_sha256': hashlib.sha256((p / config_relative).read_bytes()).hexdigest()})
        for update in [args.updates // 2, args.updates]:
            worker = ['-m', 'v03.train', '--config', config_relative, '--run-dir', run,
                      '--mode', 'pilot', '--stop-after', str(update)]
            if update == args.updates:
                worker += ['--resume', f'{run}/checkpoints/update-{args.updates // 2:06d}.pt']
            steps.append({'id': f'{name}_train_to_{update:06d}',
                          'timeout_seconds': args.maximum_wall_seconds,
                          'worker_args': worker})
            steps.append({'id': f'{name}_val_{update:06d}',
                          'timeout_seconds': 1200,
                          'worker_args': ['-m', 'v03.evaluate', '--config', config_relative,
                              '--checkpoint', f'{run}/checkpoints/update-{update:06d}.pt',
                              '--output-dir', f'{run}/validation/update-{update:06d}',
                              '--limit-trajectories', str(args.validation_trajectories), '--ks', '0,1,3,5']})
    manifest = {'identity': args.identity, 'status': 'prepared_not_started',
                'hardware_template': str(args.template), 'variants': variants,
                'paired_optimizer_updates': args.updates,
                'paired_sampled_trajectories': args.updates * training['effective_batch'],
                'validation_updates': [args.updates // 2, args.updates],
                'validation_trajectories': args.validation_trajectories,
                'validation_K': [0, 1, 3, 5], 'G': [.75, 1.5, 2.25, 3., 3.75, 4.5],
                'model_seed': base['seed'], 'sampling_seed': base['sampling_seed'],
                'maximum_wall_seconds': args.maximum_wall_seconds,
                'not_after_utc': args.not_after_utc,
                'selection_rule': {'prerequisites': 'All candidates complete equal budgets, identical initial hash and sampled path signatures; finite training and validation.',
                    'k0_preservation': 'Final six-G macro PSNR(K0) must be within0.10dB of the best candidate K0.',
                    'primary': 'Highest final six-G/four-K equally weighted absolute PSNR among preserved candidates.',
                    'close_score_tolerance_db': .03,
                    'close_score_tie': 'Within0.03dB of the best primary score, prefer lower macro LPIPS; if LPIPS within0.005, prefer lambda0.1, then0.05, then0.2.',
                    'review': 'Also inspect paired learning curves, all per-G and per-K metrics, next-state MSE, observation residual, gradient scales and numerical stability. Diagnostics never alone choose the winner.',
                    'inconclusive': 'Do not claim statistical superiority from a short single-seed pilot. Do not expand search for a positive result.'},
                'formal_reset_required': True, 'formal_protocol_frozen': False, 'test_opened': False}
    write(p / f'artifacts/{args.identity}_manifest.json', manifest)
    write(p / f'configs/{args.identity}_queue.json',
          {'id': args.identity + '_queue', 'maximum_wall_seconds': args.maximum_wall_seconds,
           'not_after_utc': args.not_after_utc, 'steps': steps})
    print(json.dumps({'manifest': f'artifacts/{args.identity}_manifest.json', 'steps': len(steps)}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Execute a frozen formal protocol through guarded train/validation segments.

This CPU controller does not select hyperparameters. A reviewed policy must
exist before launch. A noon operational pause cannot be resumed by this script.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Explicit timezone required')
    return parsed.timestamp()


def validation_point(summary):
    if summary['status'] != 'complete' or summary['split'] != 'val' or summary['held_out_test_opened']:
        raise ValueError('Complete validation-only result required')
    metrics = {k: summary['per_k'][str(k)]['macro_g'] for k in [0, 1, 3, 5]}
    if not all(math.isfinite(float(v)) for row in metrics.values() for v in row.values()):
        raise ValueError('Nonfinite validation metric')
    return {'update': summary['checkpoint_update'], 'k0_psnr': metrics[0]['psnr'],
            'score_psnr': sum(metrics[k]['psnr'] for k in metrics) / 4,
            'score_lpips': sum(metrics[k]['lpips'] for k in metrics) / 4,
            'per_k': metrics, 'validation_identity': summary['validation_identity']}


def stop_reason(points, rule):
    """Predeclared rules use only comparable fast-validation samples."""
    if not points:
        return None
    current = points[-1]
    if len(points) > 1:
        best_prior = max(p['k0_psnr'] for p in points[:-1])
        if current['k0_psnr'] < best_prior - rule['catastrophic_k0_drop_db']:
            return 'catastrophic_validation_drop'
    patience = rule['plateau_intervals']
    if current['update'] < rule['minimum_updates'] or len(points) < patience + 1:
        return None
    old_best = max(p['score_psnr'] for p in points[:-patience])
    recent_best = max(p['score_psnr'] for p in points[-patience:])
    if recent_best < old_best + rule['minimum_improvement_db']:
        return 'predeclared_validation_plateau'
    return None


def continuation_reason(points, update, rule):
    """A finite, frozen extension gate; a missing milestone never grants one."""
    if not rule or update != rule['primary_updates']:
        return None
    if not points or points[-1]['update'] != update:
        return 'primary_budget_without_comparable_validation'
    cutoff = update - rule['lookback_updates']
    earlier = [p for p in points if p['update'] <= cutoff]
    recent = [p for p in points if cutoff < p['update'] <= update]
    if not earlier or not recent:
        return 'primary_budget_without_comparable_validation'
    improvement = (max(p['score_psnr'] for p in recent)
                   - max(p['score_psnr'] for p in earlier))
    best_k0 = max(p['k0_psnr'] for p in points)
    if (improvement < rule['minimum_improvement_db']
            or points[-1]['k0_psnr'] < best_k0 - rule['k0_preservation_db']):
        return 'primary_budget_continuation_gate_not_met'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    policy = read(args.policy)
    if policy.get('protocol_frozen') is not True:
        raise ValueError('A frozen reviewed policy is required')
    config_path = project / policy['config']
    config = read(config_path)
    if digest(config) != policy['config_hash']:
        raise ValueError('Frozen config changed')
    if config['workload']['kind'] != 'mixed':
        raise ValueError('Formal training requires mixed physical paths')
    training = config['training']
    if [training.get(k) for k in ['checkpoint_interval', 'milestone_interval', 'rolling_keep']] != [500, 2000, 3]:
        raise ValueError('Formal retention contract changed')
    run = project / 'runs' / policy['run_id']
    control = run / 'controller'
    identity = {'policy_hash': digest(policy), 'config_hash': digest(config)}
    if (control / 'identity.json').exists() and read(control / 'identity.json') != identity:
        raise ValueError('Controller identity changed')
    save(control / 'identity.json', identity)
    frozen_policy = control / 'frozen_policy.json'
    if not frozen_policy.exists():
        save(frozen_policy, policy)
    state = read(control / 'status.json') if (control / 'status.json').exists() else {
        'status': 'starting', 'completed_fast_validation': [], 'completed_full_validation': [],
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'guarded_attempts': []}
    # Stopped protocols require explicit human/agent review, never an implicit restart.
    if state['status'] in ['operationally_paused', 'stopped', 'completed', 'failed']:
        raise ValueError('Controller is terminal; inspect its saved evidence')
    pause_at = timestamp(training['pause_at_utc'])
    max_updates = training['maximum_updates']
    interval = policy['fast_validation_interval']
    if interval != 1000:
        raise ValueError('User fast-validation interval is 1000 updates')

    def guarded(label, worker_args, seconds):
        attempt_id = f'{label}-{time.time_ns()}'
        guard = run / 'controller_guards' / attempt_id
        command = ['python3', str(project / 'scripts/guard.py'), '--run-dir', str(guard),
                   '--timeout-seconds', str(max(30, seconds)), '--memory-limit-gib',
                   str(policy['guard_memory_gib']), '--command', 'bash',
                   str(project / 'scripts/run_env.sh'), '-m', 'torch.distributed.run',
                   '--standalone', '--nproc_per_node=4', *worker_args]
        state['status'] = 'running'; state['active'] = label
        state['guarded_attempts'].append({'id': attempt_id, 'argv': command})
        save(control / 'status.json', state)
        with (control / (attempt_id + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=project, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        state['guarded_attempts'][-1]['returncode'] = result.returncode
        save(control / 'status.json', state)
        if result.returncode:
            raise RuntimeError(f'Guarded phase failed: {label}; inspect preserved logs')

    try:
        while True:
            if (project / 'STOP_REQUESTED').exists():
                state.update(status='stopped', reason='project_stop_requested'); break
            if (run / 'resume_ready.json').exists():
                state.update(status='operationally_paused', resume_ready=read(run / 'resume_ready.json')); break
            latest = read(run / 'latest_checkpoint.json') if (run / 'latest_checkpoint.json').exists() else None
            update = latest['update'] if latest else 0
            if update >= max_updates:
                state.update(status='completed', update=update); break
            if time.time() >= pause_at:
                # Save a resume_ready marker through the trainer at its initial
                # update boundary if the clock expired during validation.
                next_stop = min(max_updates, update + 1)
            else:
                next_stop = min(max_updates, ((update // interval) + 1) * interval)
            worker = ['-m', 'v03.train', '--config', str(config_path), '--run-dir', str(run),
                      '--mode', 'formal', '--stop-after', str(next_stop)]
            if latest:
                worker += ['--resume', latest['path']]
            elif time.time() >= pause_at:
                state.update(status='stopped', reason='deadline_before_fresh_formal_launch'); break
            guarded(f'train-to-{next_stop:06d}', worker, pause_at - time.time() + 240)
            if (run / 'resume_ready.json').exists():
                state.update(status='operationally_paused', resume_ready=read(run / 'resume_ready.json')); break
            latest = read(run / 'latest_checkpoint.json'); update = latest['update']
            checkpoint = latest['path']
            segment = read(run / 'segment_complete.json')
            safe_yield = segment['status'] == 'safe_checkpoint_prepared'
            # Estimate plus margin prevents validation from occupying the final
            # safe-checkpoint window. No scientific K/G coverage is removed.
            # A deliberate11:30 safe yield permits one frozen fast-validation
            # pass before11:50, even when1000updates were not reached. Then
            # resume exactly the same optimizer/curriculum/sampler state.
            validation_deadline = pause_at if safe_yield else timestamp(training['checkpoint_prepare_at_utc'])
            remaining = validation_deadline - time.time()
            pending_stop_reason = None
            if update > 0 and remaining >= policy['fast_validation_budget_seconds'] + 120:
                output = run / 'validation' / f'{"safe" if safe_yield else "fast"}-{update:06d}'
                guarded(f'fast-val-{update:06d}', ['-m', 'v03.evaluate', '--config', str(config_path),
                    '--checkpoint', checkpoint, '--output-dir', str(output),
                    '--limit-trajectories', str(policy['fast_validation_trajectories']), '--ks', '0,1,3,5'],
                    policy['fast_validation_budget_seconds'])
                point = validation_point(read(output / 'summary.json'))
                point['checkpoint'] = checkpoint
                point['operational_safe_validation'] = safe_yield
                state['completed_fast_validation'].append(point)
                # A partial final milestone is reported, but does not count as
                # a regular1000-update interval in plateau patience.
                regular_points = [r for r in state['completed_fast_validation']
                                  if r['update'] % interval == 0 and not r.get('operational_safe_validation')]
                catastrophic_rule = {**policy['early_stop'], 'minimum_updates': 2**63 - 1}
                pending_stop_reason = (stop_reason(state['completed_fast_validation'], catastrophic_rule)
                                       or stop_reason(regular_points, policy['early_stop'])
                                       or continuation_reason(regular_points, update,
                                                              policy.get('continuation')))
                if pending_stop_reason == 'catastrophic_validation_drop':
                    state.update(status='stopped', reason=pending_stop_reason, checkpoint=checkpoint); break
            else:
                state.setdefault('deferred_validation', []).append({'update': update, 'kind': 'fast',
                                                                  'reason': 'safe_checkpoint_window'})
                pending_stop_reason = continuation_reason([], update, policy.get('continuation'))
            remaining = timestamp(training['checkpoint_prepare_at_utc']) - time.time()
            if update % policy['full_validation_interval'] == 0:
                if remaining >= policy['full_validation_budget_seconds'] + 120:
                    output = run / 'validation' / f'full-{update:06d}'
                    guarded(f'full-val-{update:06d}', ['-m', 'v03.evaluate', '--config', str(config_path),
                        '--checkpoint', checkpoint, '--output-dir', str(output), '--ks', '0,1,3,5'],
                        policy['full_validation_budget_seconds'])
                    point = validation_point(read(output / 'summary.json')); point['checkpoint'] = checkpoint
                    state['completed_full_validation'].append(point)
                else:
                    state.setdefault('deferred_validation', []).append({'update': update, 'kind': 'full',
                                                                      'reason': 'safe_checkpoint_window'})
            if pending_stop_reason:
                state.update(status='stopped', reason=pending_stop_reason, checkpoint=checkpoint)
                break
            save(control / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        state['last_updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(control / 'status.json', state)


if __name__ == '__main__':
    main()

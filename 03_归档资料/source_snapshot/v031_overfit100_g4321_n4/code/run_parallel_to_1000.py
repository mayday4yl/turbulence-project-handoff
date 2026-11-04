#!/usr/bin/env python3
"""Checkpoint-bound evaluation alongside unchanged four-rank training.

One unchanged guard owns the complete descendant tree. Only this worker writes
workflow state. The controller waits for resource release before publishing the
final pause or retrying. Numerical/training/evaluation source files stay frozen.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_auto_to_1000 as frozen
from guard_distributed import process_info

ROOT = frozen.ROOT
BASE = ROOT / 'artifacts/parallel_execution'
MANIFEST = ROOT / 'artifacts/PARALLEL_EXECUTION_SHA256.json'
AUTH = ROOT / 'artifacts/PARALLEL_EXECUTION_AUTHORIZATION.json'
CHECKPOINT500 = '66ec73a4f66d6007392944067c62f154140dac91b972d8423690b89d14b87c05'
CALIBRATION = 'edc52ae9eddc8ba98548740ec0b7bc269102812a09f8a9dcd8e5e7247cbb6cba'


def verify():
    cfg = frozen.verify_gates()
    auth = frozen.read(AUTH)
    if (auth.get('max_global_update') != 1000 or auth.get('physical_cards') != [0, 1, 2, 3]
            or auth.get('scope') != 'concurrent_checkpoint_evaluation_and_unchanged_training'):
        raise RuntimeError('Parallel execution authorization mismatch')
    for name, expected in frozen.read(MANIFEST).items():
        if frozen.digest(ROOT / name) != expected:
            raise RuntimeError(f'Parallel delivery hash mismatch: {name}')
    if frozen.digest(frozen.FORMAL / 'checkpoint_000500.pt') != CHECKPOINT500:
        raise RuntimeError('Handoff checkpoint500 hash mismatch')
    if frozen.digest(ROOT / 'artifacts/calibration_update100/calibration.json') != CALIBRATION:
        raise RuntimeError('Frozen calibration hash mismatch')
    if os.environ.get('HCCL_OP_EXPANSION_MODE', 'HOST') != 'HOST':
        raise RuntimeError('Co-residency requires the already-used default HOST communication mode')
    return cfg


def update(**fields):
    fields['controller_pid'] = int(os.environ.get('V031_PARALLEL_CONTROLLER_PID', os.getpid()))
    fields['updated_utc'] = frozen.utc()
    value = frozen.read(frozen.STATUS)
    value.update(fields)
    frozen.write(frozen.STATUS, value)


def checkpoint_path(target):
    if target not in range(500, 1001, 100):
        raise ValueError('Parallel continuation is bounded to checkpoints500..1000')
    return frozen.FORMAL / f'checkpoint_{target:06d}.pt'


def train_command(cfg, target):
    if target not in range(600, 1001, 100):
        raise ValueError('Only frozen100-update segments600..1000 are authorized')
    return ['bash', 'code/run_distributed_env.sh', '-m', 'torch.distributed.run',
            '--standalone', '--nnodes=1', '--nproc_per_node=4', 'code/train_worker.py',
            '--mode', 'train', '--config', 'configs/train_v031.json',
            '--output', str(frozen.FORMAL), '--microbatch', str(cfg['training']['microbatch']),
            '--run-until-update', str(target), '--resume', str(checkpoint_path(target - 100)),
            '--authorize-boundary-extension']


def eval_command(cfg, target, weights):
    if weights not in ('raw', 'ema'):
        raise ValueError('Unknown checkpoint weights')
    return ['bash', 'code/run_env.sh', 'code/calibrate_evaluate.py', '--mode', 'evaluate',
            '--config', 'configs/train_v031.json', '--checkpoint', str(checkpoint_path(target)),
            '--weights', weights, '--manifest', ('start_manifest.jsonl' if target in (500, 1000)
                                                else 'subsets/overfit_16traj_starts.jsonl'),
            '--batch-size', str(cfg['evaluation']['batch_size']), '--output',
            str(ROOT / 'artifacts/evaluation' / f'update_{target:06d}' / weights),
            '--rollout-modes', 'both', '--numeric-config', 'artifacts/calibration_update100/calibration.json']


def verify_evaluation(cfg, target, weights):
    folder = ROOT / 'artifacts/evaluation' / f'update_{target:06d}' / weights
    if not (folder / 'summary.json').exists() or not (folder / 'progress.json').exists():
        return False
    summary, progress = [frozen.read(folder / name) for name in ('summary.json', 'progress.json')]
    if summary.get('status') != 'complete' or progress.get('status') != 'complete':
        return False
    identity = summary['identity']
    cal = frozen.read(ROOT / 'artifacts/calibration_update100/calibration.json')
    count = 1600 if target in (500, 1000) else 64
    manifest = Path(cfg['dataset_root']) / ('start_manifest.jsonl' if count == 1600
                                           else 'subsets/overfit_16traj_starts.jsonl')
    starts = {r['start_id']: r for r in read_lines(manifest)}
    expected = {'global_update': target, 'weights': weights,
                'checkpoint_sha256': frozen.digest(checkpoint_path(target)),
                'manifest_sha256': frozen.digest(manifest),
                'training_identity': frozen.read(ROOT / 'artifacts/train_segment_000500.json')['identity'],
                'calibration_identity': cal['identity'], 'numeric_config': cal['selected'],
                'modes': ['unrefined', 'refined'], 'tie_psnr_db': cfg['evaluation']['tie_psnr_db']}
    if (identity != progress['identity'] or any(identity.get(k) != v for k, v in expected.items())
            or summary.get('start_count_per_mode') != count
            or summary.get('network_unchanged') is not True or summary.get('simulator_unchanged') is not True):
        raise RuntimeError('Evaluation completion identity/state mismatch')
    for mode in ('unrefined', 'refined'):
        rows = read_lines(folder / mode / 'per_start.jsonl')
        if len(rows) != count or len(starts) != count or {r['start_id'] for r in rows} != starts.keys():
            raise RuntimeError('Evaluation coverage is incomplete or duplicated')
        for row in rows:
            source = starts[row['start_id']]
            if (row['initial_G'] != source['initial_G'] or len(row['predictions']) != source['initial_G']
                    or row['mode'] != mode or row['weights'] != weights or row['global_update'] != target):
                raise RuntimeError('Evaluation prefix identity mismatch')
    return True


def read_lines(path, *, allow_incomplete_tail=False):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    if allow_incomplete_tail and lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def verify_segment(target, result):
    if (result.get('global_update') != target or result.get('status') != 'segment_complete'
            or result.get('simulator_unchanged') is not True or result.get('smoke_optimizer_updates') != 0):
        raise RuntimeError('Incomplete training segment evidence')
    sha = frozen.digest(checkpoint_path(target))
    if result.get('checkpoint', {}).get('sha256') != sha:
        raise RuntimeError('Checkpoint publication hash mismatch')
    for rank in range(4):
        rows = read_lines(frozen.FORMAL / f'updates_rank{rank}.jsonl')
        if [r['global_update'] for r in rows] != list(range(1, target + 1)):
            raise RuntimeError('Training logs are not unique/contiguous')
        if any(r.get('finite_loss') is not True or r.get('finite_gradients') is not True
               or r.get('formal') is not True for r in rows):
            raise RuntimeError('Training log has nonfinite or nonformal work')
    return sha


def last_committed():
    target = 500
    for candidate in range(600, 1001, 100):
        proof = ROOT / 'artifacts' / f'train_segment_{candidate:06d}.json'
        if not proof.exists():
            if checkpoint_path(candidate).exists():
                raise RuntimeError('Checkpoint without committed segment proof needs inspection')
            break
        record = frozen.read(proof)
        if (record.get('global_update') != candidate or record.get('status') != 'segment_complete'
                or record.get('checkpoint_sha256') != frozen.digest(checkpoint_path(candidate))
                or record.get('simulator_unchanged') is not True
                or record.get('execution', {}).get('returncode') != 0):
            raise RuntimeError('Previously committed parallel segment proof mismatch')
        target = candidate
    return target


def archive_uncommitted(target, attempt):
    # Called only after the preceding guard proved release of its entire tree.
    paths = [frozen.FORMAL / f'updates_rank{rank}.jsonl' for rank in range(4)]
    if all(p.read_bytes().endswith(b'\n') and
           [r['global_update'] for r in read_lines(p, allow_incomplete_tail=True)] == list(range(1, target + 1))
           for p in paths):
        return
    archive = BASE / f'replayed_suffix_a{attempt:02d}'
    archive.mkdir(parents=True, exist_ok=False)
    evidence = []
    for path in paths:
        shutil.copy2(path, archive / path.name)
        rows = read_lines(path, allow_incomplete_tail=True)
        kept = [r for r in rows if r['global_update'] <= target]
        if [r['global_update'] for r in kept] != list(range(1, target + 1)):
            raise RuntimeError('Committed checkpoint log prefix is incomplete')
        temporary = path.with_suffix('.jsonl.tmp')
        with temporary.open('w') as stream:
            for row in kept:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        evidence.append({'file': path.name, 'original_sha256': frozen.digest(archive / path.name),
                         'retained_updates': target, 'uncommitted_complete_updates': len(rows) - len(kept)})
    frozen.write(archive / 'evidence.json', {'resume_update': target, 'files': evidence})


def spawn(label, argv, attempt, timeout):
    verify()
    folder = BASE / f'attempt_{attempt:02d}' / label
    folder.mkdir(parents=True, exist_ok=False)
    log = (folder / 'worker.log').open('w')
    # No nested resource guards: both siblings descend from the SAME guarded
    # manager, so the existing strict ownership/foreign-process checks apply.
    process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
    launch = {'label': label, 'argv': argv, 'utc': frozen.utc(), 'identity': process_info(process.pid),
              'guard_directory': os.environ['OVERFIT100_RUN_DIR'], 'timeout_seconds': timeout}
    frozen.write(folder / 'launch.json', launch)
    return {'process': process, 'log': log, 'folder': folder, 'launch': launch,
            'started': time.monotonic(), 'timeout': timeout}


def poll(job):
    process = job['process']
    code = process.poll()
    if code is None:
        if time.monotonic() - job['started'] > job['timeout']:
            raise TimeoutError(f"{job['launch']['label']} timeout; outer guard will clean only its tree")
        return False
    job['log'].close()
    result = {**job['launch'], 'returncode': code, 'finished_utc': frozen.utc(),
              'elapsed_seconds': time.monotonic() - job['started']}
    frozen.write(job['folder'] / 'result.json', result)
    job['result'] = result
    if code:
        text = (job['folder'] / 'worker.log').read_text(errors='replace')[-12000:]
        raise RuntimeError(f"{job['launch']['label']} exited {code}: {text}")
    return True


def run_preflight(attempt):
    accepted = BASE / 'COEXISTENCE_ACCEPTANCE.json'
    if accepted.exists() and frozen.read(accepted).get('status') == 'passed':
        return
    output = BASE / f'probe_a{attempt:02d}'
    output.mkdir(parents=True, exist_ok=False)
    common = ['code/check_parallel_workload.py', '--config', 'configs/train_v031.json',
              '--checkpoint', str(checkpoint_path(500)), '--output', str(output)]
    jobs = [spawn('probe_gradient', ['bash', 'code/run_distributed_env.sh', '-m', 'torch.distributed.run',
                                   '--standalone', '--nnodes=1', '--nproc_per_node=4'] + common +
                  ['--role', 'gradient'], attempt, 1800),
            spawn('probe_inference', ['bash', 'code/run_env.sh'] + common + ['--role', 'inference'], attempt, 1800)]
    pending = list(jobs)
    while pending:
        if (ROOT / 'STOP_REQUESTED').exists():
            raise RuntimeError('User/project stop marker present')
        pending = [job for job in pending if not poll(job)]
        time.sleep(1)
    records = [frozen.read(output / f'gradient_rank{i}.json') for i in range(4)]
    inference = frozen.read(output / 'inference.json')
    if any(r.get('status') != 'passed' for r in records + [inference]):
        raise RuntimeError('Concurrent numerical/gradient preflight did not pass')
    overlap = min(records[0]['work_end'], inference['work_end']) - max(records[0]['work_start'], inference['work_start'])
    if overlap < 10:
        raise RuntimeError('Concurrent probe has insufficient measured computation overlap')
    frozen.write(accepted, {'status': 'passed', 'utc': frozen.utc(), 'checkpoint_sha256': CHECKPOINT500,
                           'probe_directory': str(output), 'work_overlap_seconds': overlap,
                           'restoration_optimizer_updates': 0, 'formal_updates_added': 0,
                           'gradient_results': records, 'inference_result': inference})


def worker(attempt):
    cfg = verify()
    update(phase='parallel_coexistence_preflight', status='running', active_training=None, active_evaluation=None)
    run_preflight(attempt)
    committed = last_committed()
    queue = [(u, w) for u in range(500, 1001, 100) for w in ('raw', 'ema')]
    done, reports = set(), set()
    train_job = eval_job = None
    next_target = committed + 100
    while True:
        if (ROOT / 'STOP_REQUESTED').exists():
            raise RuntimeError('User/project stop marker present')
        if train_job and poll(train_job):
            result = frozen.read(frozen.FORMAL / 'results.json')
            sha = verify_segment(next_target, result)
            result.update(checkpoint_sha256=sha, execution=train_job['result'])
            frozen.write(ROOT / 'artifacts' / f'train_segment_{next_target:06d}.json', result)
            committed = next_target
            next_target += 100
            train_job = None
            update(formal_global_update=committed, last_checkpoint=str(checkpoint_path(committed)),
                   last_checkpoint_sha256=sha, training_complete_to_1000=committed == 1000,
                   active_training=None)
        if eval_job and poll(eval_job):
            target, weights = eval_job['key']
            if not verify_evaluation(cfg, target, weights):
                raise RuntimeError('Evaluator exited without complete outputs')
            done.add((target, weights))
            eval_job = None
            update(active_evaluation=None)
        if eval_job is None:
            for target, weights in queue:
                if target > committed:
                    break
                if (target, weights) in done:
                    continue
                if verify_evaluation(cfg, target, weights):
                    done.add((target, weights))
                    continue
                timeout = cfg['orchestration']['full_evaluation_timeout_seconds' if target in (500, 1000)
                                               else 'subset_evaluation_timeout_seconds']
                eval_job = spawn(f'eval_{target:06d}_{weights}', eval_command(cfg, target, weights), attempt, timeout)
                eval_job['key'] = (target, weights)
                update(active_evaluation={'global_update': target, 'weights': weights,
                                          'identity': eval_job['launch']['identity'],
                                          'output': str(ROOT / 'artifacts/evaluation' / f'update_{target:06d}' / weights)})
                break
        if train_job is None:
            # The frozen report already filters rows through its checkpoint.
            # Invoke it BETWEEN training segments so its log bytes cannot change
            # during parsing/hash collection, even if evaluation lagged behind.
            for target in (500, 1000):
                if target not in reports and all((target, w) in done for w in ('raw', 'ema')):
                    subprocess.run([sys.executable, 'code/write_progress_report.py', '--root', str(ROOT),
                                    '--update', str(target)], cwd=ROOT, check=True)
                    reports.add(target)
            if next_target <= 1000:
                train_job = spawn(f'train_{next_target:06d}', train_command(cfg, next_target), attempt,
                                  cfg['orchestration']['train_segment_timeout_seconds'])
                update(active_training={'resume_update': next_target - 100, 'target_update': next_target,
                                        'identity': train_job['launch']['identity']})
        completed_evaluations = [u for u in range(500, 1001, 100) if all((u, w) in done for w in ('raw', 'ema'))]
        update(phase='parallel_training_and_evaluation' if train_job and eval_job else
               'training' if train_job else 'evaluation', status='running',
               completed_evaluation_updates=completed_evaluations,
               last_evaluation_update=max(completed_evaluations, default=400),
               evaluation_complete_at_current_update=committed in completed_evaluations,
               reports_completed=sorted(reports))
        if committed == 1000 and len(done) == len(queue) and reports == {500, 1000}:
            verify()
            frozen.write(BASE / 'COMPLETION.json', {'status': 'complete', 'utc': frozen.utc(),
                         'global_update': 1000, 'evaluations': sorted(done), 'reports': sorted(reports),
                         'checkpoint_sha256': frozen.digest(checkpoint_path(1000)),
                         'allow_automatic_updates_beyond_1000': False})
            return
        time.sleep(5)


def controller(first_attempt):
    lock = (ROOT / '.auto_to_1000.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    BASE.mkdir(parents=True, exist_ok=True)
    own = process_info(os.getpid())
    frozen.write(ROOT / 'artifacts/parallel_to_1000_job.json', {**own, 'created_utc': frozen.utc(),
                 'command': sys.argv, 'log': str(BASE / 'controller.log')})
    os.environ['V031_PARALLEL_CONTROLLER_PID'] = str(os.getpid())
    update(controller_starttime_ticks=own['starttime_ticks'], execution_mode='parallel_checkpoint_evaluation',
           persistent_handoff_job='artifacts/parallel_to_1000_job.json',
           active_guard_directory=None, failure=None, error=None)
    frozen.write(ROOT / 'artifacts/PERSISTENT_HANDOFF_STATUS.json', {
        'status': 'executing_guarded_parallel_auto_to_1000', 'utc': frozen.utc(), 'pid': os.getpid(),
        'starttime_ticks': own['starttime_ticks'], 'job': 'artifacts/parallel_to_1000_job.json',
        'supersedes': 'artifacts/persistent_handoff_03_job.json'})
    try:
        for attempt in range(first_attempt, 4):
            target = last_committed()
            archive_uncommitted(target, attempt)
            run_dir = ROOT / 'runs' / f'stage3_parallel_a{attempt:02d}'
            update(status='running', phase='parallel_preflight', active_guard_directory=str(run_dir))
            result = subprocess.run([sys.executable, 'code/guard_distributed.py', '--run-dir', str(run_dir),
                       '--timeout-seconds', '86400', '--memory-limit-gib', '61', '--command',
                       sys.executable, '-u', 'code/run_parallel_to_1000.py', '--mode', 'worker',
                       '--attempt', str(attempt)], cwd=ROOT, stdin=subprocess.DEVNULL)
            frozen.require_released(run_dir)
            if result.returncode == 0:
                if frozen.read(BASE / 'COMPLETION.json').get('status') != 'complete':
                    raise RuntimeError('Parallel worker lacks final completion evidence')
                audit_env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                                 PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
                subprocess.run([frozen.PYTHON, '-B', 'code/check_parallel_final_checkpoint.py', '1000'],
                               cwd=ROOT, env=audit_env, check=True)
                update(status='PAUSED_AT_1000', phase='complete', formal_global_update=1000,
                       training_complete_to_1000=True, full_evaluation_complete_at_1000=True,
                       active_training=None, active_evaluation=None, active_guard_directory=None,
                       allow_automatic_updates_beyond_1000=False,
                       next_action='Await explicit user authorization for any continuation')
                frozen.write(ROOT / 'artifacts/PERSISTENT_HANDOFF_STATUS.json', {
                    'status': 'PAUSED_AT_1000', 'utc': frozen.utc(), 'pid': os.getpid(),
                    'formal_global_update': 1000, 'allow_automatic_updates_beyond_1000': False})
                return
            kind, details = frozen.failure_kind(run_dir)
            frozen.write(BASE / f'failure_a{attempt:02d}.json', {'kind': kind, 'details': details, 'utc': frozen.utc()})
            if kind != 'infrastructure' or attempt == 3:
                raise RuntimeError(f'Parallel execution requires engineering inspection: {details}')
    except BaseException as exc:
        update(status='blocked', failure=repr(exc), allow_automatic_updates_beyond_1000=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('controller', 'worker'), required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    (controller if args.mode == 'controller' else worker)(args.attempt)

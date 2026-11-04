#!/usr/bin/env python3
"""Small server-side sequential runner for the explicit AUTO_TO_1000 grant.

Every NPU subprocess has its own existing resource guard. No job is launched
before fixed delivery hashes, stage gates and authorization are checked. This
runner never chooses training/refinement hyperparameters and never exceeds 1000.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data2/yanglin/v031_overfit100_g4321_n4')
PYTHON = '/data0/yanglin/envs/physical_bridge/bin/python'
STATUS = ROOT / 'artifacts/WORKFLOW_STATUS.json'
FORMAL = ROOT / 'runs/formal'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def state(**fields):
    saved = read(STATUS)
    saved.update(fields, updated_utc=utc(), controller_pid=os.getpid())
    write(STATUS, saved)


def verify_gates():
    cfg = read(ROOT / 'configs/train_v031.json')
    if cfg.get('mode') != 'AUTO_TO_1000' or cfg['training']['run_until_update'] != 1000:
        raise RuntimeError('This controller is authorized only for AUTO_TO_1000')
    if cfg['training']['allow_automatic_updates_beyond_1000'] is not False:
        raise RuntimeError('Automatic boundary extension is forbidden')
    if any(cfg.get('gates', {}).get(k) is not True for k in
           ['stage1_passed', 'training_config_frozen', 'inference_policy_frozen']):
        raise RuntimeError('Stage gates are not all passed/frozen')
    accepted = read(ROOT / 'artifacts/STAGE1_FINAL_ACCEPTANCE.json')
    if not (ROOT / 'artifacts/STAGE1_STATUS.json').exists():
        raise RuntimeError('Stage1 status missing')
    if read(ROOT / 'artifacts/STAGE1_STATUS.json').get('stage1_status') != 'v031_implementation_acceptance_passed':
        raise RuntimeError('Stage1 final status is not PASS')
    if not accepted:
        raise RuntimeError('Empty Stage1 evidence')
    if read(ROOT / 'artifacts/STAGE2_FINAL_ACCEPTANCE.json').get('status') != 'passed':
        raise RuntimeError('Stage2 final acceptance not passed')
    auth = read(ROOT / 'artifacts/AUTO_TO_1000_AUTHORIZATION.json')
    if auth.get('mode') != 'AUTO_TO_1000':
        raise RuntimeError('Explicit automatic handoff authorization missing')
    ledger = read(ROOT / 'artifacts/STAGE2_SMOKE_LEDGER.json')
    if ledger['reserved_updates'] or ledger['conservative_consumed_updates'] > 60:
        raise RuntimeError('Smoke accounting is unresolved or exceeds budget')
    hashes = read(ROOT / 'artifacts/STAGE2_DELIVERY_SHA256.json')
    for relative, expected in hashes.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen delivery hash changed: {relative}')
    if (ROOT / 'STOP_REQUESTED').exists():
        raise RuntimeError('User/project stop marker present')
    return cfg


def failure_kind(run_dir):
    result = read(run_dir / 'guard_result.json') if (run_dir / 'guard_result.json').exists() else {}
    text = (result.get('reason') or '') + '\n'
    log = run_dir / 'worker.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 128000))
            text += stream.read().decode(errors='replace')
    lowered = text.lower()
    if any(word in lowered for word in ['out of memory', 'guard cap', 'nonfinite', 'non-finite',
                                        'identity mismatch', 'hash mismatch', 'unhealthy', 'is occupied',
                                        'external or unowned', 'disconnected', 'missing trainable gradients']):
        return 'blocked', text[-12000:]
    # Narrow transport/infrastructure retry only. Unknown failures stop for
    # engineering inspection rather than silently changing the experiment.
    if any(word in lowered for word in ['connection reset by peer', 'connection aborted',
                                        'connection closed by peer', 'socket timeout',
                                        'network is unreachable']):
        return 'infrastructure', text[-12000:]
    return 'blocked', text[-12000:]


def require_released(run_dir):
    path = run_dir / 'guard_result.json'
    if not path.exists() or read(path).get('own_npu_processes_released') is not True:
        raise RuntimeError(f'Previous owned process release is unconfirmed: {run_dir}')


def preserve_replayed_suffix(target, attempt):
    """Preserve aborted physical work; canonical logs describe committed state."""
    preceding = max(0, target - 100)
    archive = FORMAL / 'replayed_attempt_logs' / f'before_target_{target:06d}_attempt_{attempt:02d}'
    archive.mkdir(parents=True, exist_ok=False)
    evidence = {'resume_from_global_update': preceding, 'target': target, 'attempt': attempt,
                'reason': 'bounded infrastructure retry from the last complete checkpoint', 'files': []}
    for rank in range(4):
        path = FORMAL / f'updates_rank{rank}.jsonl'
        if not path.exists():
            continue
        shutil.copy2(path, archive / path.name)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        retained = [row for row in rows if row['global_update'] <= preceding]
        replayed = [row for row in rows if row['global_update'] > preceding]
        if [row['global_update'] for row in retained] != list(range(1, preceding + 1)):
            raise RuntimeError('Checkpoint prefix logs are not unique and complete')
        temporary = path.with_suffix('.jsonl.tmp')
        with temporary.open('w') as handle:
            for row in retained:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        evidence['files'].append({'rank': rank, 'original_sha256': digest(archive / path.name),
                                  'retained_rows': len(retained), 'replayed_physical_updates': len(replayed)})
        error = FORMAL / f'error_rank{rank}.json'
        if error.exists():
            shutil.copy2(error, archive / error.name)
    write(archive / 'replay_record.json', evidence)


def guarded(label, command, *, distributed, timeout):
    for attempt in range(1, 4):
        verify_gates()
        run_dir = ROOT / 'runs' / f'{label}_a{attempt:02d}'
        if (run_dir / 'guard_result.json').exists():
            previous = read(run_dir / 'guard_result.json')
            if previous.get('status') == 'completed' and previous.get('own_npu_processes_released') is True:
                return  # Resume the controller without repeating completed work.
            require_released(run_dir)
            kind, details = failure_kind(run_dir)
            if kind != 'infrastructure':
                raise RuntimeError(f'{label} previously failed: {details}')
            continue
        if run_dir.exists():
            raise RuntimeError(f'Incomplete guard directory needs ownership inspection: {run_dir}')
        if attempt > 1 and label == 'stage3_train_000000' and FORMAL.exists():
            if list(FORMAL.glob('*.pt')) or list(FORMAL.glob('updates_rank*.jsonl')):
                raise RuntimeError('Initialization retry found checkpoint/update evidence; inspection required')
            FORMAL.rename(ROOT / 'runs' / f'formal_initialization_failed_before_a{attempt:02d}')
        elif attempt > 1 and label.startswith('stage3_train_'):
            preserve_replayed_suffix(int(label.rsplit('_', 1)[-1]), attempt)
        state(current_stage=3, status='running', phase=label, active_guard_directory=str(run_dir))
        guard = 'guard_distributed.py' if distributed else 'guard.py'
        argv = [sys.executable, str(ROOT / 'code' / guard), '--run-dir', str(run_dir),
                '--timeout-seconds', str(timeout), '--memory-limit-gib', '61' if distributed else '58',
                '--command'] + command
        code = subprocess.call(argv, cwd=ROOT, stdin=subprocess.DEVNULL)
        if code == 0:
            require_released(run_dir)
            return
        require_released(run_dir)
        kind, details = failure_kind(run_dir)
        write(ROOT / 'artifacts' / f'{label}_failure_a{attempt:02d}.json',
              {'phase': label, 'attempt': attempt, 'kind': kind, 'details': details, 'utc': utc()})
        if kind != 'infrastructure':
            raise RuntimeError(f'{label} failed: {details}')
    raise RuntimeError(f'{label}: same infrastructure phase exhausted two automatic retries')


def train(cfg, target):
    checkpoint = FORMAL / f'checkpoint_{target:06d}.pt'
    if checkpoint.exists():
        completed = []
        for path in (ROOT / 'runs').glob(f'stage3_train_{target:06d}_a*/guard_result.json'):
            result = read(path)
            if result.get('status') == 'completed' and result.get('own_npu_processes_released') is True:
                completed.append(path)
        if not completed:
            raise RuntimeError('Existing checkpoint lacks successful guard completion and resource release')
        proof = ROOT / 'artifacts' / f'train_segment_{target:06d}.json'
        result = read(proof) if proof.exists() else read(FORMAL / 'results.json')
        if (result.get('global_update') != target or result.get('simulator_unchanged') is not True
                or result.get('status') not in ['initialization_complete', 'segment_complete']):
            raise RuntimeError('Existing checkpoint lacks successful matching segment evidence')
        expected_hash = result.get('checkpoint_sha256') if proof.exists() else result.get('checkpoint', {}).get('sha256')
        if not expected_hash or expected_hash != digest(checkpoint):
            raise RuntimeError('Completed checkpoint digest changed')
        if not proof.exists():
            result['checkpoint_sha256'] = digest(checkpoint)
            write(proof, result)
        return
    args = ['bash', str(ROOT / 'code/run_distributed_env.sh'), '-m', 'torch.distributed.run',
            '--standalone', '--nnodes=1', '--nproc_per_node=4', 'code/train_worker.py',
            '--mode', 'train', '--config', 'configs/train_v031.json', '--output', str(FORMAL),
            '--microbatch', str(cfg['training']['microbatch']), '--run-until-update', str(target)]
    if target:
        previous = FORMAL / f'checkpoint_{max(0, target-100):06d}.pt'
        if not previous.exists():
            raise RuntimeError(f'Missing preceding complete checkpoint: {previous}')
        args += ['--resume', str(previous), '--authorize-boundary-extension']
    timeout = cfg['orchestration']['train_segment_timeout_seconds']
    guarded(f'stage3_train_{target:06d}', args, distributed=True, timeout=timeout)
    if not checkpoint.exists():
        raise RuntimeError('Successful worker did not publish required complete checkpoint')
    result = read(FORMAL / 'results.json')
    if (result.get('global_update') != target or result.get('simulator_unchanged') is not True
            or result.get('status') not in ['initialization_complete', 'segment_complete']):
        raise RuntimeError('Segment completion evidence failed after checkpoint publication')
    if result.get('checkpoint', {}).get('sha256') != digest(checkpoint):
        raise RuntimeError('Checkpoint bytes differ from the successful worker publication digest')
    result['checkpoint_sha256'] = digest(checkpoint)
    write(ROOT / 'artifacts' / f'train_segment_{target:06d}.json', result)
    state(formal_training_started=target > 0, formal_global_update=target,
          training_complete_to_1000=target == 1000, last_checkpoint=str(checkpoint),
          last_checkpoint_sha256=digest(checkpoint), evaluation_complete_at_current_update=False)


def evaluate(cfg, update, weights):
    output = ROOT / 'artifacts/evaluation' / f'update_{update:06d}' / weights
    modes = ['unrefined'] if update == 0 else ['unrefined', 'refined']
    count = 1600 if update in [500, 1000] else 64
    manifest = 'start_manifest.jsonl' if count == 1600 else 'subsets/overfit_16traj_starts.jsonl'
    args = ['bash', str(ROOT / 'code/run_env.sh'), 'code/calibrate_evaluate.py', '--mode', 'evaluate',
            '--config', 'configs/train_v031.json', '--checkpoint', str(FORMAL / f'checkpoint_{update:06d}.pt'),
            '--weights', weights, '--manifest', manifest, '--batch-size', str(cfg['evaluation']['batch_size']),
            '--output', str(output), '--rollout-modes', 'unrefined' if update == 0 else 'both']
    if update:
        args += ['--numeric-config', 'artifacts/calibration_update100/calibration.json']
    timeout = cfg['orchestration']['full_evaluation_timeout_seconds' if count == 1600 else 'subset_evaluation_timeout_seconds']
    guarded(f'stage3_eval_{update:06d}_{weights}', args, distributed=False, timeout=timeout)
    progress, summary = read(output / 'progress.json'), read(output / 'summary.json')
    if progress.get('status') != 'complete' or summary.get('status') != 'complete':
        raise RuntimeError('Evaluation exited without complete status')
    for mode in modes:
        rows = [json.loads(line) for line in (output / mode / 'per_start.jsonl').read_text().splitlines() if line]
        if len(rows) != count or len({row['start_id'] for row in rows}) != count:
            raise RuntimeError('Incomplete/duplicated evaluation starts')
        if any(len(row['predictions']) != row['initial_G'] for row in rows):
            raise RuntimeError('Evaluation omitted legal prediction prefixes')


def main():
    lock = (ROOT / '.auto_to_1000.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = verify_gates()
    old = read(STATUS)
    if old.get('status') == 'PAUSED_AT_1000':
        return
    state(current_stage=3, status='running', controller_started_utc=utc())
    try:
        train(cfg, 0)
        for weights in ['raw', 'ema']:
            evaluate(cfg, 0, weights)
        for target in range(100, 1001, 100):
            train(cfg, target)
            if target == 100:
                args = ['bash', str(ROOT / 'code/run_env.sh'), 'code/calibrate_evaluate.py',
                        '--mode', 'calibrate', '--config', 'configs/train_v031.json',
                        '--checkpoint', str(FORMAL / 'checkpoint_000100.pt'), '--weights', 'raw',
                        '--policy', 'configs/inference_calibration_policy.json',
                        '--output', 'artifacts/calibration_update100']
                guarded('stage3_calibrate_000100', args, distributed=False,
                        timeout=cfg['orchestration']['calibration_timeout_seconds'])
                calibration = read(ROOT / 'artifacts/calibration_update100/calibration.json')
                if calibration.get('status') != 'complete':
                    raise RuntimeError('Update100 calibration incomplete')
                state(inference_numerical_config='frozen', inference_selected=calibration['selected'])
            for weights in ['raw', 'ema']:
                evaluate(cfg, target, weights)
            state(evaluation_complete_at_current_update=True, last_evaluation_update=target)
            if target in [500, 1000]:
                subprocess.run([sys.executable, str(ROOT / 'code/write_progress_report.py'),
                                '--root', str(ROOT), '--update', str(target)], check=True)
        state(status='PAUSED_AT_1000', phase='complete', formal_global_update=1000,
              training_complete_to_1000=True, full_evaluation_complete_at_1000=True,
              allow_automatic_updates_beyond_1000=False, active_guard_directory=None,
              next_action='Await explicit user authorization for any continuation')
    except BaseException as exc:
        state(status='blocked', failure=repr(exc), allow_automatic_updates_beyond_1000=False)
        raise


if __name__ == '__main__':
    main()

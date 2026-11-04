#!/usr/bin/env python3
"""Run one authorized disposable smoke, with a durable global update budget.

This controller is deliberately sequential. Its ledger reserves the full upper
bound before launching and conservatively charges that bound if a rank vanishes
without final accounting. It never retries or launches formal training.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data2/yanglin/v031_overfit100_g4321_n4')
LEDGER = ROOT / 'artifacts/STAGE2_SMOKE_LEDGER.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--mode', required=True, choices=['memory-smoke', 'performance-smoke', 'resume-check'])
    parser.add_argument('--microbatch', type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument('--warmup-updates', type=int, default=1)
    parser.add_argument('--measure-updates', type=int, required=True)
    args = parser.parse_args()
    if not args.id.replace('_', '').isalnum() or min(args.warmup_updates, args.measure_updates) < 1:
        parser.error('Require an isolated alphanumeric/underscore identity and positive counts')
    reserved = args.warmup_updates + args.measure_updates * (2 if args.mode == 'resume-check' else 1)
    lock = (ROOT / '.smoke_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = read(LEDGER)
    if any(row['id'] == args.id for row in ledger['attempts']):
        raise RuntimeError('Attempt identity already exists; never repeat silently')
    if ledger['conservative_consumed_updates'] + ledger['reserved_updates'] + reserved > 60:
        raise RuntimeError('Global smoke update budget would exceed 60')
    config = ROOT / f'configs/stage2_candidate_mb{args.microbatch}.json'
    output = ROOT / 'artifacts' / args.id
    run = ROOT / 'runs' / args.id
    if output.exists() or run.exists():
        raise RuntimeError('Attempt paths must be fresh')
    worker = ['bash', str(ROOT / 'code/run_distributed_env.sh'), '-m', 'torch.distributed.run',
              '--standalone', '--nnodes=1', '--nproc_per_node=4', 'code/train_worker.py',
              '--mode', args.mode, '--config', str(config), '--output', str(output),
              '--microbatch', str(args.microbatch), '--warmup-updates', str(args.warmup_updates),
              '--measure-updates', str(args.measure_updates)]
    command = [sys.executable, str(ROOT / 'code/guard_distributed.py'), '--run-dir', str(run),
               '--timeout-seconds', '7200', '--memory-limit-gib', '61', '--command'] + worker
    row = dict(vars(args), status='reserved', reserved_updates=reserved, started_utc=now(),
               controller_pid=os.getpid(), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
               command=command, artifact_dir=str(output), run_dir=str(run))
    ledger['attempts'].append(row)
    ledger['reserved_updates'] += reserved
    write(LEDGER, ledger)
    try:
        exit_code = subprocess.call(command, cwd=ROOT, stdin=subprocess.DEVNULL)
    except BaseException as exc:
        row['controller_exception'] = repr(exc)
        write(LEDGER, ledger)  # Reservation remains charged after uncertain interruption.
        raise
    reports = []
    for rank in range(4):
        paths = [output / f'results_rank{rank}.json', output / f'error_rank{rank}.json']
        report = next((read(path) for path in paths if path.exists()), None)
        reports.append(report)
    completed = max([r.get('physical_optimizer_steps_this_invocation', 0) for r in reports if r] + [0])
    attempted = max([r.get('optimizer_step_attempts_this_invocation',
                            r.get('physical_optimizer_steps_this_invocation', 0)) for r in reports if r] + [0])
    guard_status = read(run / 'guard_result.json') if (run / 'guard_result.json').exists() else None
    if all(reports):
        consumed = max(completed, attempted)
    elif guard_status and guard_status.get('launch', {}).get('worker_pid') is None and not (run / 'launch.json').exists():
        consumed = 0
    else:
        consumed = reserved  # Never infer zero work from missing final files.
    if consumed > reserved:
        raise RuntimeError('Worker exceeded reserved upper bound')
    ledger['reserved_updates'] -= reserved
    ledger['actual_completed_updates'] += completed
    ledger['conservative_consumed_updates'] += consumed
    row.update(status='passed' if exit_code == 0 else 'failed', exit_code=exit_code,
               completed_utc=now(), known_completed_updates=completed, conservative_consumed_updates=consumed,
               complete_rank_accounting=all(reports), guard_status=guard_status)
    write(LEDGER, ledger)
    print(json.dumps({key: row[key] for key in ['id', 'status', 'known_completed_updates',
                                               'conservative_consumed_updates']}), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

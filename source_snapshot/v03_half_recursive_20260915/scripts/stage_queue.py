#!/usr/bin/env python3
"""Run a finite, explicit queue of guarded four-card engineering jobs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

PROJECT = Path('/data2/yanglin/v03_half_recursive_20260915')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.queue.read_text())
    out = PROJECT / 'artifacts/runtime' / spec['id']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'status.json').exists():
        raise RuntimeError('Queue already attempted; inspect evidence before creating a new identity')
    started = time.monotonic()
    state = {'id': spec['id'], 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'steps': []}
    save(out/'queue.json', spec)
    try:
        for step in spec['steps']:
            if (PROJECT / 'STOP_REQUESTED').exists():
                raise RuntimeError('Project stop requested; no next stage')
            remaining = spec['maximum_wall_seconds'] - (time.monotonic() - started)
            if spec.get('not_after_utc'):
                deadline = datetime.datetime.fromisoformat(spec['not_after_utc'].replace('Z', '+00:00')).timestamp()
                remaining = min(remaining, deadline - time.time())
            if remaining <= 30:
                raise TimeoutError('Queue total budget exhausted')
            guard_dir = PROJECT / 'runs' / step['id'] / 'guard'
            argv = ['python3', str(PROJECT/'scripts/guard.py'), '--run-dir', str(guard_dir),
                    '--timeout-seconds', str(min(step['timeout_seconds'], remaining - 25)),
                    '--memory-limit-gib', str(step.get('memory_limit_gib', 58)), '--command',
                    'bash', str(PROJECT/'scripts/run_env.sh'), '-m', 'torch.distributed.run',
                    '--standalone', '--nproc_per_node=4', *step['worker_args']]
            state['active_step'] = step['id']
            save(out/'status.json', state)
            step_start = time.monotonic()
            with (out/(step['id']+'.log')).open('w') as log:
                result = subprocess.run(argv, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            row = {'id':step['id'], 'returncode':result.returncode,
                   'elapsed_seconds':time.monotonic()-step_start,
                   'guard_result':str(guard_dir/'guard_result.json')}
            state['steps'].append(row)
            save(out/'status.json',state)
            if result.returncode:
                worker_log = guard_dir/'worker.log'
                tail = worker_log.read_text(errors='replace')[-20000:] if worker_log.exists() else ''
                guard_file = guard_dir/'guard_result.json'
                guard_reason = str(json.loads(guard_file.read_text()).get('reason','')) if guard_file.exists() else ''
                is_oom = any(term in tail.lower() for term in ('out of memory', 'memory allocation', 'memory alloc'))
                is_oom = is_oom or ('own allocation' in guard_reason and 'guard cap' in guard_reason)
                mode_args = step['worker_args']
                benchmark_mode = ('--mode' in mode_args and
                                  mode_args[mode_args.index('--mode') + 1] == 'benchmark')
                guard_state = json.loads(guard_file.read_text()) if guard_file.exists() else {}
                expected_timeout = (step.get('allow_timeout', False) and benchmark_mode and
                                    'Declared worker wall-clock budget exhausted' in guard_reason and
                                    guard_state.get('own_npu_processes_released') is True)
                if expected_timeout:
                    row['accepted_bounded_benchmark_timeout'] = True
                    continue
                if not (step.get('allow_oom',False) and is_oom):
                    raise RuntimeError('Guarded stage failed: '+step['id'])
                row['accepted_benchmark_oom'] = True
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'failed'
        state['error'] = repr(error)
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        save(out/'status.json',state)


if __name__ == '__main__':
    main()

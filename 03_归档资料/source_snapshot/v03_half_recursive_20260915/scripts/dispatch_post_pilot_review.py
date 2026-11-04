#!/usr/bin/env python3
"""Continue predeclared diagnostics immediately after all paired pilots finish.

This CPU watcher never starts formal training or modifies a scientific config.
Every accelerator phase still uses stage_queue and its ownership/health guard.
"""
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

PROJECT = Path('/data2/yanglin/v03_half_recursive_20260915')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    out = PROJECT / 'artifacts/runtime/half_post_pilot_dispatch_01'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise FileExistsError('Post-pilot dispatcher was already attempted')
    state = {'status': 'waiting_for_all_paired_pilots', 'steps': [],
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(out / 'status.json', state)

    def execute(label, command):
        if (PROJECT / 'STOP_REQUESTED').exists():
            raise RuntimeError('Project stop requested')
        state.update(status='running', active=label); save(out / 'status.json', state)
        started = time.monotonic()
        with (out / (label + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        state['steps'].append({'id': label, 'returncode': result.returncode,
                               'elapsed_seconds': time.monotonic() - started})
        save(out / 'status.json', state)
        if result.returncode:
            raise RuntimeError(f'Post-pilot phase failed: {label}; preserve evidence and inspect')

    try:
        deadline = dt.datetime.fromisoformat('2026-09-16T03:15:00+00:00').timestamp()
        while time.time() < deadline:
            if (PROJECT / 'STOP_REQUESTED').exists():
                raise RuntimeError('Project stop requested')
            status = read(PROJECT / 'artifacts/runtime/half_lambda_pilot_01_queue/status.json')
            if status['status'] == 'failed':
                raise RuntimeError('Paired pilot queue failed; no unpaired selection allowed')
            if status['status'] != 'completed':
                time.sleep(15)
                continue
            manifest = read(PROJECT / 'artifacts/half_lambda_pilot_01_manifest.json')
            for candidate in manifest['variants']:
                run = Path(candidate['run_dir'])
                if read(run / 'segment_complete.json')['update'] != 64:
                    raise ValueError('Every candidate must complete64updates')
                for update in [32, 64]:
                    if read(run / 'validation' / f'update-{update:06d}' / 'summary.json')['status'] != 'complete':
                        raise ValueError('Every paired validation must be complete')
            execute('final_checkpoint_gradient_diagnostics', ['python3', str(PROJECT / 'scripts/stage_queue.py'),
                    '--queue', str(PROJECT / 'configs/half_pilot_gradient_queue_01.json')])
            execute('paired_pilot_scientific_evidence_audit', ['python3',
                    str(PROJECT / 'scripts/review_fair_pilot.py'), '--project', str(PROJECT)])
            state['review_artifact'] = str(PROJECT / 'artifacts/half_lambda_pilot_01_review.json')
            # Keep the primary noon checkpoint objective ahead of this extra
            # per-length cost breakdown if diagnostics finish unexpectedly late.
            if time.time() < dt.datetime.fromisoformat('2026-09-16T02:50:00+00:00').timestamp():
                execute('isolated_actual_length_timing', ['python3', str(PROJECT / 'scripts/stage_queue.py'),
                        '--queue', str(PROJECT / 'configs/half_length_benchmark_queue_01.json')])
            else:
                state['length_timing_deferred_reason'] = 'Operational noon checkpoint takes priority after10:50CST'
            state['status'] = 'ready_for_root_protocol_review'
            state['formal_training_started_by_dispatcher'] = False
            break
        else:
            state['status'] = 'expired_without_formal_launch'
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(out / 'status.json', state)


if __name__ == '__main__':
    main()

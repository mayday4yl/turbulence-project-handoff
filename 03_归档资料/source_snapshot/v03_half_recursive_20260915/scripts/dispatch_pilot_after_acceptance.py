#!/usr/bin/env python3
"""CPU-only, finite handoff after the already-running engineering queue."""
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
    out = PROJECT / 'artifacts/runtime/half_pilot_dispatch_01'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise FileExistsError('Dispatcher already attempted')
    state = {'status': 'waiting_for_acceptance', 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(out / 'status.json', state)
    try:
        deadline = dt.datetime.fromisoformat('2026-09-16T02:00:00+00:00').timestamp()
        while time.time() < deadline:
            if (PROJECT / 'STOP_REQUESTED').exists():
                raise RuntimeError('Project stop requested')
            status = read(PROJECT / 'artifacts/runtime/half_engineering_final_queue_04/status.json')
            if status['status'] == 'failed':
                raise RuntimeError('Engineering acceptance failed; root review required')
            if status['status'] != 'completed':
                time.sleep(15)
                continue
            outcomes = {step['id']: step['returncode'] for step in status['steps']}
            for name in ['half_acceptance_safe_yield_04', 'half_acceptance_safe_yield_04_resume']:
                if outcomes.get(name) != 0:
                    raise RuntimeError('Safe-yield/resume acceptance incomplete')
            run = PROJECT / 'runs/half_acceptance_safe_yield_04'
            if read(run / 'segment_complete.json')['update'] != 2:
                raise ValueError('Resume did not complete the two actual updates')
            audits = list((run / 'resume_load_audit').glob('update-000000-rank-*.json'))
            if len(audits) != 4 or not all(read(path)['passed'] for path in audits):
                raise ValueError('Four exact-load resume audits required')
            sync = read(run / 'acceptance/checkpoint_sync_000002.json')
            if not sync['model_equal'] or not sync['ema_equal']:
                raise ValueError('Resumed DDP model/EMA synchronization failed')
            manifest_path = PROJECT / 'artifacts/half_lambda_pilot_01_manifest.json'
            manifest = read(manifest_path)
            if manifest.get('prelaunch_fairness_and_exposure_audit', {}).get('passed') is not True:
                raise ValueError('Paired config/sampling/exposure audit missing')
            if (PROJECT / 'artifacts/runtime/half_lambda_pilot_01_queue/status.json').exists():
                raise FileExistsError('Pilot queue was already attempted')
            manifest['status'] = 'running'
            manifest['started_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            save(manifest_path, manifest)
            state.update(status='pilot_queue_running', pilot_manifest=str(manifest_path))
            save(out / 'status.json', state)
            with (out / 'pilot_queue.log').open('w') as stream:
                result = subprocess.run(['python3', str(PROJECT / 'scripts/stage_queue.py'),
                                         '--queue', str(PROJECT / 'configs/half_lambda_pilot_01_queue.json')],
                                        cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            state.update(status='pilot_queue_completed' if result.returncode == 0 else 'pilot_queue_failed',
                         returncode=result.returncode)
            manifest = read(manifest_path)
            manifest['status'] = 'completed_pending_scientific_review' if result.returncode == 0 else 'failed_preserve_all_evidence'
            save(manifest_path, manifest)
            break
        else:
            state.update(status='expired_without_launch')
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(out / 'status.json', state)


if __name__ == '__main__':
    main()

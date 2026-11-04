#!/usr/bin/env python3
"""Wait for our existing preparation job, freeze passed evidence, run to1000.

This is a persistent sequential handoff, not a second preparation/training job.
The frozen runtime selection and all final gates must pass before any training.
"""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/data2/yanglin/v031_overfit100_g4321_n4')
PROGRESS = ROOT / 'artifacts/PERSISTENT_HANDOFF_STATUS.json'


def write(value):
    value.update(utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid())
    temporary = PROGRESS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(PROGRESS)


def running(identity):
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text()
    except FileNotFoundError:
        return False
    fields = stat[stat.rfind(')') + 2:].split()
    return int(fields[19]) == identity['starttime_ticks'] and fields[0] != 'Z'


def main(preparation_job='artifacts/stage2_preflight_sequence_01_job.json'):
    lock = (ROOT / '.persistent_handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preparation = json.loads((ROOT / preparation_job).read_text())
    scripts = ['code/freeze_stage2.py', 'code/run_auto_to_1000.py', 'code/write_progress_report.py',
               'code/calibrate_evaluate.py', 'code/check_distributed_gradients.py']
    reviewed = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in scripts}
    write({'status': 'waiting_for_existing_stage2', 'preparation_identity': preparation,
           'reviewed_script_hashes': reviewed, 'formal_training_started': False})
    try:
        while running(preparation):
            if (ROOT / 'STOP_REQUESTED').exists():
                raise RuntimeError('Project stop marker prevents further handoff')
            time.sleep(15)
        selected = json.loads((ROOT / 'artifacts/STAGE2_RUNTIME_SELECTION.json').read_text())
        if selected.get('status') != 'passed_waiting_inference_smoke_and_freeze':
            raise RuntimeError('Stage2 did not pass; inspect its existing failure before any further work')
        for relative, expected in reviewed.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Reviewed handoff script changed while waiting: {relative}')
        write({'status': 'freezing_passed_stage2', 'reviewed_script_hashes': reviewed,
               'formal_training_started': False})
        subprocess.run([sys.executable, 'code/freeze_stage2.py'], cwd=ROOT,
                       stdin=subprocess.DEVNULL, check=True)
        write({'status': 'executing_guarded_auto_to_1000', 'reviewed_script_hashes': reviewed,
               'workflow_status': str(ROOT / 'artifacts/WORKFLOW_STATUS.json')})
        subprocess.run([sys.executable, 'code/run_auto_to_1000.py'], cwd=ROOT,
                       stdin=subprocess.DEVNULL, check=True)
        state = json.loads((ROOT / 'artifacts/WORKFLOW_STATUS.json').read_text())
        if state.get('status') != 'PAUSED_AT_1000':
            raise RuntimeError('Automatic run returned without verified completion at1000')
        write({'status': 'PAUSED_AT_1000', 'formal_global_update': 1000,
               'allow_automatic_updates_beyond_1000': False})
    except BaseException as exc:
        write({'status': 'needs_engineering_inspection', 'error': repr(exc),
               'traceback': traceback.format_exc(), 'automatic_retry_here': False,
               'workflow_status': str(ROOT / 'artifacts/WORKFLOW_STATUS.json')})
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-job', default='artifacts/stage2_preflight_sequence_01_job.json')
    main(parser.parse_args().preparation_job)

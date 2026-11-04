#!/usr/bin/env python3
"""One-time, identity-checked replacement of the original sequential controller."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

import run_parallel_to_1000 as parallel
from guard_distributed import process_info, snapshot, check_snapshot

ROOT = parallel.ROOT
frozen = parallel.frozen
EXPECTED = {'controller': (1069441, 187143588), 'handoff': (1035195, 187112042),
            'guard': (1919529, 190400266)}
DIRECTORY = ROOT / 'runs/stage3_eval_000500_ema_a01'


def matched(name, required=True):
    pid, start = EXPECTED[name]
    try:
        info = process_info(pid)
        valid = info['starttime_ticks'] == start and info['state'] != 'Z'
    except FileNotFoundError:
        valid, info = False, None
    if required and not valid:
        raise RuntimeError(f'Prior {name} identity no longer matches; inspect before signalling')
    return info if valid else None


def wait_exited(name, seconds):
    deadline = time.monotonic() + seconds
    while matched(name, required=False):
        if time.monotonic() > deadline:
            raise RuntimeError(f'{name} did not exit; automatic handoff stops')
        time.sleep(0.25)


def main():
    cfg = parallel.verify()
    if not parallel.verify_evaluation(cfg, 500, 'raw'):
        raise RuntimeError('Completed isolated raw500 reference is required')
    saved = frozen.read(frozen.STATUS)
    if (saved['controller_pid'] != EXPECTED['controller'][0] or saved['formal_global_update'] != 500
            or saved['phase'] != 'stage3_eval_000500_ema' or Path(saved['active_guard_directory']) != DIRECTORY):
        raise RuntimeError('Original workflow moved; do not use stale handoff identities')
    identities = {name: matched(name) for name in EXPECTED}
    if (identities['guard']['ppid'] != identities['controller']['pid']
            or identities['controller']['ppid'] != identities['handoff']['pid']):
        raise RuntimeError('Expected prior ancestry changed')
    archive = parallel.BASE / 'original_sequential_handoff'
    archive.mkdir(parents=True, exist_ok=False)
    frozen.write(archive / 'workflow_before.json', saved)
    frozen.write(archive / 'identities.json', identities)
    frozen.write(archive / 'handoff_before.json', frozen.read(ROOT / 'artifacts/PERSISTENT_HANDOFF_STATUS.json'))
    frozen.write(archive / 'evaluation_progress_before.json',
                 frozen.read(ROOT / 'artifacts/evaluation/update_000500/ema/progress.json'))
    frozen.write(archive / 'evaluation_committed_rows_sha256.json', {
        str(path.relative_to(ROOT)): frozen.digest(path)
        for path in (ROOT / 'artifacts/evaluation/update_000500/ema').glob('*/rows/*.json')})
    stopped = False
    try:
        matched('controller')
        os.kill(EXPECTED['controller'][0], signal.SIGSTOP)
        stopped = True
        current = frozen.read(frozen.STATUS)
        if current != saved or matched('guard')['ppid'] != EXPECTED['controller'][0]:
            raise RuntimeError('Controller changed before quiescence')
        # TERM is queued while stopped; CONT lets it terminate without running
        # the next scheduling iteration. The existing guard remains responsible
        # for all worker/compilation descendants and their resource release.
        os.kill(EXPECTED['controller'][0], signal.SIGTERM)
        os.kill(EXPECTED['controller'][0], signal.SIGCONT)
        stopped = False
        wait_exited('controller', 10)
        matched('guard')
        os.kill(EXPECTED['guard'][0], signal.SIGTERM)
        wait_exited('guard', 45)
        wait_exited('handoff', 10)
    finally:
        if stopped and matched('controller', required=False):
            os.kill(EXPECTED['controller'][0], signal.SIGCONT)
    frozen.require_released(DIRECTORY)
    frozen.write(archive / 'guard_result.json', frozen.read(DIRECTORY / 'guard_result.json'))
    frozen.write(archive / 'handoff_after.json', frozen.read(ROOT / 'artifacts/PERSISTENT_HANDOFF_STATUS.json'))
    hashes = frozen.read(archive / 'evaluation_committed_rows_sha256.json')
    if any(frozen.digest(ROOT / name) != expected for name, expected in hashes.items()):
        raise RuntimeError('A committed evaluation row changed during controlled handoff')
    for name in ('.auto_to_1000.lock', '.guard_npu_0.lock', '.persistent_handoff.lock'):
        with (ROOT / name).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw, snap = snapshot()
    check_snapshot(snap)
    (archive / 'pre_parallel_npu_smi.txt').write_text(raw)
    parallel.verify()
    with (parallel.BASE / 'controller.log').open('x') as log:
        worker = subprocess.Popen(['python3', '-u', 'code/run_parallel_to_1000.py', '--mode', 'controller'],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    result = {'status': 'replacement_launched', 'utc': frozen.utc(), 'new_controller': process_info(worker.pid),
              'old_identities': identities, 'prior_owned_npu_resources_released': True,
              'preserved_committed_evaluation_rows': len(hashes),
              'original_failed_guard_reason': 'explicit controlled user-requested orchestration handoff',
              'formal_resume_update': 500, 'max_update': 1000,
              'script_sha256': frozen.digest(__file__)}
    frozen.write(archive / 'HANDOFF.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()

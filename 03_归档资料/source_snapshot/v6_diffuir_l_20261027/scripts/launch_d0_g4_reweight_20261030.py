"""Exclusive one-card guard for the user-authorized finite D0 sampler branches."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from scripts.runtime_utils import ROOT, GIB, snapshot, process_identity, is_same_process, is_descendant, write_json, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--branch', choices=['Control', 'G4_biased'], required=True)
    ap.add_argument('--card', type=int, required=True)
    ap.add_argument('--mode', choices=['probe', 'formal'], required=True)
    ap.add_argument('--end-local', type=int)
    ap.add_argument('--resume')
    a = ap.parse_args()
    assert a.card in [1, 2, 3], 'Only the newly authorized remaining cards'
    ex = ROOT / 'execution_20261030_d0_g4_reweight_5k'
    assert not (ex / 'STOP_REQUEST.json').exists()
    cfg = json.loads((ex / 'CONFIG.json').read_text())
    assert cfg['physical_devices_by_branch'][a.branch] == [a.card]
    lock = (ROOT / 'gates' / f'AB_NPU_{a.card}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = snapshot()
    assert not before['processes'].get(a.card), 'Authorized card currently occupied; never touch a foreign process'
    out = ROOT / 'distributed' / a.run
    out.mkdir(parents=True, exist_ok=False)
    info = before['cards'][a.card]
    limit = dict(before=info, min_free=int(.2 * info['free']) + GIB, allocator_cap=int(.8 * info['free']) - GIB)
    write_json(out / 'SAFETY.json', dict(cards={str(a.card): limit}, physical_devices=[a.card]))
    write_json(out / 'NPU_BEFORE.json', before)
    worker = ROOT / 'scripts/run_d0_g4_reweight_20261030.py'
    cmd = ['bash', str(ROOT / 'scripts/run_env.sh'), str(worker), '--run', a.run,
           '--safety-file', str(out / 'SAFETY.json'), '--branch', a.branch, '--mode', a.mode]
    if a.end_local is not None:
        cmd += ['--end-local', str(a.end_local)]
    if a.resume:
        cmd += ['--resume', a.resume]
    env = os.environ.copy()
    env['V6_DEVICES'] = str(a.card)
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    with (out / 'console.log').open('xb') as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    owner = process_identity(p.pid)
    write_json(out / 'LAUNCH.json', dict(owner, command=cmd, source_sha256=sha(worker),
               launcher_sha256=sha(__file__), started=time.time(), timeout_hours=24 if a.mode == 'formal' else 1))
    start = time.time()
    stopped = None
    stop_time = None
    owned = {}
    while p.poll() is None:
        snap = snapshot()
        with (out / 'HBM_MONITOR.jsonl').open('a') as f:
            f.write(json.dumps({k: v for k, v in snap.items() if k != 'raw'}) + '\n')
        foreign = []
        for proc in snap['processes'].get(a.card, []):
            try:
                if is_descendant(proc['pid'], owner):
                    owned[proc['pid']] = process_identity(proc['pid'])
                elif proc['pid'] not in owned or not is_same_process(owned[proc['pid']]):
                    foreign.append(proc)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if stopped is None and (foreign or snap['cards'][a.card]['free'] < limit['min_free']):
            stopped = dict(reason='physical safety boundary', card=a.card, foreign=foreign, memory=snap['cards'][a.card])
        if stopped is None and time.time() - start > (24 if a.mode == 'formal' else 1) * 3600:
            stopped = dict(reason='explicit engineering runtime bound exceeded')
        if stopped is not None and stop_time is None:
            write_json(out / 'SUPERVISOR_STOP.json', stopped)
            # Signal only the own actor, preserving its DataLoader children for
            # an atomic update-boundary save. Never signal a foreign process.
            rank_file = out / 'RANK_0.json'
            target = json.loads(rank_file.read_text()) if rank_file.exists() else owner
            if is_same_process(target) and is_descendant(target['pid'], owner):
                os.kill(target['pid'], signal.SIGTERM)
            stop_time = time.time()
        if stop_time is not None and time.time() - stop_time > 180 and is_same_process(owner):
            os.killpg(owner['pid'], signal.SIGKILL)
        write_json(out / 'OWNED_WORKERS.json', list(owned.values()))
        time.sleep(2)
    code = p.wait()
    remaining = [r for r in owned.values() if is_same_process(r)]
    assert not remaining, ('Own worker remains', remaining)
    write_json(out / 'NPU_AFTER.json', snapshot())
    write_json(out / 'PROCESS_COMPLETE.json', dict(exit_code=code, seconds=time.time() - start, supervisor_stop=stopped))
    if code or stopped:
        raise RuntimeError((code, stopped))


if __name__ == '__main__':
    main()

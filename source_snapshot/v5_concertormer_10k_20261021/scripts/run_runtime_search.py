"""Launch an explicit list of independent candidates on verified free cards."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.controller import parse_npu_processes, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    a = parser.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    report_path = ROOT / plan['report']
    if report_path.exists():
        raise RuntimeError('Refusing to overwrite a search run')
    candidates = plan['candidates']
    cards = [c['card'] for c in candidates]
    if not 1 <= len(cards) <= 4 or len(set(cards)) != len(cards):
        raise ValueError('Use one candidate per card, at most four')
    snapshot = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
    if any(parse_npu_processes(snapshot, cards).values()):
        raise RuntimeError('A selected card has an existing process; do not disturb it')
    report = {'status': 'RUNNING', 'started_unix': time.time(), 'pid': os.getpid(),
              'formal_optimizer_updates': 0, 'plan': plan, 'resource_snapshot': snapshot, 'jobs': []}
    active = []
    for candidate in candidates:
        path = ROOT / candidate['output']
        if path.exists():
            raise RuntimeError(f'Existing evidence: {path}')
        command = ['bash', 'run_env.sh', 'scripts/benchmark_runtime_candidate.py',
                   '--device', 'npu:0', '--initial', 'weights/adapted_U0.pt',
                   '--dataset', 'data/v5_places_c50_s2000_v250_t250_g4_r4_20261021',
                   '--output', candidate['output'], '--microbatch', str(candidate['microbatch']),
                   '--policy', candidate['policy'], '--method', candidate.get('method', 'R0'),
                   '--precision', candidate.get('precision', 'bf16'),
                   '--warmup', str(candidate.get('warmup', 1)), '--updates', str(candidate.get('updates', 2))]
        if candidate.get('cross_step'):
            command.append('--cross-step')
        if candidate.get('profile'):
            command.append('--profile')
        env = os.environ.copy()
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(candidate['card'])
        log = path.with_suffix('.log').open('xb', buffering=0)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {**candidate, 'pid': child.pid, 'command': command, 'started_unix': time.time(), 'status': 'RUNNING'}
        report['jobs'].append(record)
        active.append((child, record, log))
    atomic_json(report_path, report)
    while active:
        for child, record, log in list(active):
            elapsed = time.time()-record['started_unix']
            if child.poll() is None and elapsed > plan.get('max_candidate_seconds', 900):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=15)
                record['status'] = 'INCOMPLETE_WALL_LIMIT'
            if child.poll() is not None:
                record.update(return_code=child.returncode, finished_unix=time.time())
                if record['status'] == 'RUNNING':
                    record['status'] = 'FINISHED' if child.returncode == 0 else 'FAILED'
                log.close()
                active.remove((child, record, log))
                atomic_json(report_path, report)
        time.sleep(2)
    report.update(status='CANDIDATES_FINISHED', finished_unix=time.time())
    atomic_json(report_path, report)


if __name__ == '__main__':
    main()

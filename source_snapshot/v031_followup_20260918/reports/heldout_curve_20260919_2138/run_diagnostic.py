"""One finite supplementary inference run. Never trains or modifies a checkpoint."""
from pathlib import Path
import sys, json, hashlib, subprocess, time, fcntl, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'code'))
from diagnostic_guard import policy_latency
from guard_shared import snapshot, process_info
from inference_controller import save, digest

def main():
    lock = (HERE / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (HERE / 'COMPLETE.json').exists()
    plan = json.loads((HERE / 'PLAN.json').read_text())
    assert digest(ROOT / 'code/evaluate_followup.py') == plan['worker_sha256']
    assert digest(ROOT / 'shared/heldout32/start_manifest.jsonl') == '50e506c0d57ddcdbab61751425e8b3f2b3ec8e7a4abb83b954f950aa9c0adc60'
    snap = snapshot(range(8))
    save(HERE / 'resource_before.json', snap)
    eligible = []
    timing = {}
    for card in snap['cards']:
        if card['health'] != 'OK' or card['hbm_free_mb'] < 14*1024 or card['aicore_percent'] > 20:
            continue
        ready = True
        for p in card['processes']:
            t = policy_latency(p['pid'])
            timing[str(p['pid'])] = t
            if not t or not t.get('median'):
                ready = False
        if ready:
            eligible.append(card)
    save(HERE / 'external_timing_before.json', timing)
    eligible.sort(key=lambda c: (len(c['processes']), -c['hbm_free_mb']))
    if len(eligible) < 4:
        save(HERE / 'WAIT_RESOURCE.json', {'eligible_cards': [c['card'] for c in eligible], 'time': time.time(), 'reason': 'Need four low-load cards with observable established external latency or no external tasks.'})
        return
    cards = [c['card'] for c in eligible[:4]]
    children = []
    proof = []
    for card, update in zip(cards, (100, 500, 800, 900)):
        jobs = []
        for task in ('T1_DIRECT', 'T2_FEEDBACK_R'):
            cp = ROOT / task / 'formal' / f'checkpoint_{update:06d}.pt'
            job = {'kind': 'evaluation', 'task': task, 'update': update,
                   'checkpoint': str(cp), 'checkpoint_sha256': digest(cp),
                   'dataset_root': str(ROOT / 'shared/heldout32'), 'manifest': 'start_manifest.jsonl',
                   'subset': 'heldout32', 'output': str(HERE / 'evaluation' / task / f'update_{update:06d}_heldout32'),
                   'rank': 0, 'shards': 1}
            jp = HERE / 'jobs' / f'{task}_{update}.json'
            save(jp, job)
            jobs.append(str(jp)); proof.append(job)
        bundle = HERE / 'jobs' / f'bundle_card{card}.json'
        save(bundle, jobs)
        guard = HERE / f'guard_card{card}'
        cmd = ['python3', str(HERE / 'diagnostic_guard.py'), '--cards', str(card), '--role', 'inference',
               '--output', str(guard), '--seconds', '1800', '--own-cap-gib', '8', '--reserve-gib', '6', '--',
               'python3', str(HERE / 'worker_bundle.py'), str(bundle)]
        log = (HERE / f'launcher_card{card}.log').open('w')
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        children.append((p, guard))
        save(HERE / f'guard_identity_card{card}.json', process_info(p.pid))
    save(HERE / 'RUNNING.json', {'cards': cards, 'jobs': proof, 'time': time.time(), 'controller': process_info(__import__('os').getpid())})
    while any(p.poll() is None for p, _ in children):
        time.sleep(3)
    results = [json.loads((g / 'result.json').read_text()) for _, g in children]
    assert all(x['status'] == 'completed' for x in results), results
    for job in proof:
        assert digest(job['checkpoint']) == job['checkpoint_sha256']
        out = Path(job['output'])
        assert (out / 'complete_rank0.json').exists()
        assert len(list(out.glob('*/result.json'))) == 256
    save(HERE / 'COMPLETE.json', {'status': 'complete', 'jobs': proof, 'guards': results, 'time': time.time(), 'new_paths': 2048, 'training_updates': 0})

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(HERE / 'FAILURE.json', {'error': repr(e), 'traceback': traceback.format_exc(), 'time': time.time()})
        raise

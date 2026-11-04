"""Independent synthetic finalizer-identity checks; no Torch or real assets."""
import ast
import hashlib
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/finalize_campaign.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def main():
    tree = ast.parse(SOURCE.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'sha', 'collect'}]
    assert len(functions) == 2
    result = {'status': 'PASS', 'source_sha256': digest(SOURCE), 'model_constructions': 0,
              'optimizer_steps': 0, 'NPU_calls': 0, 'real_checkpoint_or_image_reads': 0}
    with tempfile.TemporaryDirectory(prefix='dual-finalizer-identity-') as temp:
        fixture = Path(temp)
        ns = {'ROOT': fixture, 'Path': Path, 'hashlib': hashlib, 'json': json}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), ns)
        summaries = []
        for branch in ['original', 'improved']:
            out = fixture / 'runs' / branch
            checkpoint = out / 'checkpoints/U00000002.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'synthetic-only; not a Torch checkpoint')
            identity = {'path': str(checkpoint), 'sha256': digest(checkpoint), 'update': 2}
            save(out / 'STATUS.json', {'status': 'COMPLETE', 'final_full_validation_done': True, 'new_updates': 2})
            save(out / 'LATEST.json', identity)
            save(out / 'BEST.json', {'update': 2, 'checkpoint': identity})
            summary = out / 'validation/U00000002_ema_full/SUMMARY.json'
            save(summary, {'status': 'PASS', 'full': True, 'weights': 'EMA', 'count': 5000,
                           'per_g': 1000, 'update': 2, 'branch': branch,
                           'by_g': {g: {} for g in ['0.5', '1', '2', '3', '4']},
                           'parameter_digest': 'synthetic-ema-digest'})
            summaries.append(summary)
            for role in ['FINAL', 'BEST']:
                save(fixture / 'evidence' / f'STATE_AUDIT_{branch}_{role}.json',
                     {'status': 'PASS', 'checkpoint': {**identity, 'EMA': {'full_parameter_digest': 'synthetic-ema-digest'}}})
        refs, updates = ns['collect']()
        assert len(refs) == 4 and len({r['validation'] for r in refs}) == 2
        assert set(updates) == {'original', 'improved'}
        result['same_best_latest_retains_four_roles_but_only_two_unique_audits'] = True

        def must_reject(name, mutate, restore):
            mutate()
            rejected = False
            try:
                ns['collect']()
            except AssertionError:
                rejected = True
            finally:
                restore()
            assert rejected, name
            result[name] = True

        summary = summaries[0]
        original = summary.read_text()
        changed = json.loads(original)
        changed['parameter_digest'] = 'wrong-ema'
        must_reject('wrong_summary_EMA_digest_rejected', lambda: save(summary, changed), lambda: summary.write_text(original))
        checkpoint = fixture / 'runs/original/checkpoints/U00000002.pt'
        original_bytes = checkpoint.read_bytes()
        must_reject('changed_checkpoint_bytes_rejected', lambda: checkpoint.write_bytes(b'wrong-checkpoint'), lambda: checkpoint.write_bytes(original_bytes))
        audit = fixture / 'evidence/STATE_AUDIT_original_FINAL.json'
        original = audit.read_text()
        changed = json.loads(original)
        changed['checkpoint']['update'] = 999
        must_reject('mismatched_CPU_audit_update_rejected', lambda: save(audit, changed), lambda: audit.write_text(original))
        assert len(ns['collect']()[0]) == 4
    output = ROOT / 'evidence/cpu_engineering/FINALIZER_IDENTITY_INDEPENDENT_SELFTEST.json'
    save(output, result)
    print('FINALIZER_IDENTITY_INDEPENDENT_SELFTEST_PASS')


if __name__ == '__main__':
    main()

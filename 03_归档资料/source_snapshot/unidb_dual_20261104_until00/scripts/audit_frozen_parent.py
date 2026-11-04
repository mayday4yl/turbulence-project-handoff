"""Read a naturally stopped18h parent with its unchanged frozen CPU auditor.

Only the output writer and additive JSON metadata reporting are wrapped. The
historical auditor retains its own ROOT, authority, policy and all checks.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'unidb_dual_20261103_until18'
AUDITOR_SHA = '578b36c49e16e8c729e8586f02619e1f38aab9398c6acc0010cdc386befabc43'
POLICY_SHA = '61bc51ce136f314de7cbbf8ea62c0c6dd1def3cce8881e03b9913879f76a7d3e'
SNAPSHOT_KEYS = ('extension_parent', 'extension_parent_cpu_audit',
    'extension_start_update', 'extension_start_cursor', 'extension_start_counts_by_g',
    'extension_start_unix', 'prior_scheduler', 'previous_monitor_done')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT) and not path.is_relative_to(PARENT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', choices=['original', 'improved'], required=True)
    ap.add_argument('--selection', type=Path, required=True)
    args = ap.parse_args()
    assert time.time() >= 1793700000, 'No parent handoff before the natural18h cutoff'
    selection_path = args.selection.resolve()
    assert selection_path.is_relative_to(ROOT)
    selected = json.loads(selection_path.read_text())['parents'][args.branch]
    parent = Path(selected['path']).resolve()
    assert parent.is_relative_to(PARENT / 'runs' / args.branch / 'checkpoints')
    target = ROOT / 'evidence' / ('PARENT_STATE_AUDIT_' + args.branch + '.json')
    if target.exists():
        prior = json.loads(target.read_text())
        assert prior['status'] == 'PASS' and prior['source_sha256'] == AUDITOR_SHA
        assert prior['policy_sha256'] == POLICY_SHA
        assert all(prior['checkpoint'][k] == selected[k] for k in ['path', 'sha256', 'update', 'cursor'])
        assert set(prior['checkpoint']['extension_payload_metadata_snapshot']) == set(SNAPSHOT_KEYS)
        assert sha(parent) == selected['sha256'], 'Previously audited parent bytes changed'
        print(json.dumps({'status': 'PASS_REUSE', 'report': str(target)})); return
    old_source = PARENT / 'audit_state.py'
    frozen = json.loads((PARENT / 'protocol/RUN_SOURCE.json').read_text())['files']
    assert frozen['audit_state.py'] == sha(old_source) == AUDITOR_SHA
    for rel in ['protocol/USER_AUTHORITY.json', 'protocol/AUDIT_STATE_FROZEN_TOLERANCES.json']:
        assert sha(PARENT / rel) == frozen[rel]
    assert sha(PARENT / 'protocol/AUDIT_STATE_FROZEN_TOLERANCES.json') == POLICY_SHA
    latest = json.loads((PARENT / 'runs' / args.branch / 'LATEST.json').read_text())
    assert all(latest[k] == selected[k] for k in ['path', 'sha256', 'update', 'cursor'])
    spec = importlib.util.spec_from_file_location('frozen18_parent_auditor', old_source)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    original_checkpoint = audit.checkpoint

    def checkpoint_with_additive_metadata(path):
        assert Path(path).resolve() == parent
        payload, report = original_checkpoint(path)
        assert payload['phase'] == 'formal'
        assert all(report[k] == selected[k] for k in ['path', 'sha256', 'update', 'cursor'])
        # JSON is the declared metadata carrier; full tensors/RNG were checked
        # by the frozen auditor and are never rewritten by this wrapper.
        snapshot = {k: payload[k] for k in SNAPSHOT_KEYS}
        report['extension_payload_metadata_snapshot'] = json.loads(json.dumps(snapshot, allow_nan=False))
        return payload, report

    def redirect_writer(path, report):
        assert Path(path).resolve() == target and not target.exists()
        report['output_redirect_provenance'] = {
            'writer_redirect_only': True, 'additive_metadata_snapshot_only': True,
            'old_auditor_source': str(old_source), 'old_auditor_sha256': AUDITOR_SHA,
            'old_ROOT_authority_policy_and_checks_unchanged': True,
            'selection': str(selection_path), 'selection_sha256': sha(selection_path),
            'wrapper': str(Path(__file__).resolve()), 'wrapper_sha256': sha(__file__),
            'old_assets_written': False, 'NPU_calls': 0, 'optimizer_objects': 0,
        }
        write(target, report)

    audit.checkpoint = checkpoint_with_additive_metadata
    audit.save_report = redirect_writer
    sys.argv = [str(old_source), '--path', str(parent), '--out', str(target)]
    actor = {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'started_unix': time.time(), 'branch': args.branch, 'parent': selected, 'source_sha256': sha(__file__)}
    write(ROOT / 'evidence' / ('ACTIVE_PARENT_STATE_AUDIT_' + args.branch + '.json'), actor)
    code = 1
    try:
        audit.main()
    except SystemExit as exc:
        code = exc.code or 0
        raise
    finally:
        write(ROOT / 'evidence' / ('PARENT_STATE_AUDIT_EXIT_' + args.branch + '.json'),
            {'actor': actor, 'exit_code': code, 'completed_unix': time.time(),
             'report': str(target), 'report_sha256': sha(target) if target.exists() else None})


if __name__ == '__main__': main()

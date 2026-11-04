"""Wait for natural18h safe saves, audit, scope-clean own actors, launch successor.

This coordinator never requests an early STOP, edits historical assets,
constructs a model/optimizer, or signals an unverified process group.
"""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'unidb_dual_20261103_until18'
RUN_ENV = ROOT.parent / 'unidb_d0_20261102/run_env.sh'
NOT_BEFORE = 1793700000
DEADLINE = 1793721600
BRANCHES = ('original', 'improved')
FORBIDDEN_PIDS = {2938958, 1506794, 1504769}


class NaturalSavePending(AssertionError):
    """Expected transient: old actors have not finished their natural save."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path).resolve(); assert path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def status(stage, **extra):
    write(ROOT / 'evidence/HANDOFF_STATUS.json', {'status': stage, 'unix': time.time(), **extra})


def relative_file(rel):
    p = (ROOT / rel).resolve()
    assert not Path(rel).is_absolute() and p.is_relative_to(ROOT), rel
    return p


def readiness():
    ready = load(ROOT / 'evidence/SUCCESSOR_SOURCE_READY.json')
    assert ready['status'] == 'PASS' and ready['deadline_unix'] == DEADLINE
    assert ready['not_before_unix'] == NOT_BEFORE
    assert ready['static_files'] and ready['run_source_static_files']
    for rel, digest in ready['static_files'].items(): assert sha(relative_file(rel)) == digest, rel
    for rel, digest in ready['run_source_static_files'].items():
        assert ready['static_files'].get(rel) == digest, rel
    for rel in ['training_worker.py', 'controller.py', 'audit_state.py', 'improved_objective.py', 'validation.py',
                'scripts/handoff_after18.py', 'scripts/audit_frozen_parent.py', 'protocol/AUDIT_STATE_FROZEN_TOLERANCES.json']:
        assert rel in ready['static_files'], 'Required fixed source absent: ' + rel
    for rel in ready['required_acceptances']:
        assert rel in ready['static_files'] and load(relative_file(rel))['status'] == 'PASS', rel
    template_path = ROOT / 'evidence/AUTHORITY_TEMPLATE.json'
    if not template_path.exists():
        authority_file = ROOT / 'protocol/USER_AUTHORITY.json'
        assert sha(authority_file) == ready['authority_template_sha256']
        authority = load(authority_file); assert authority['extension_parents'] == {}
        # Byte-for-byte copy with a separate immutable expected SHA, before the
        # only permitted two-field materialization later in this script.
        template_path.write_bytes(authority_file.read_bytes())
    assert sha(template_path) == ready['authority_template_sha256']
    template = load(template_path)
    assert template['deadline_unix'] == DEADLINE and template['previous_deadline_unix'] == NOT_BEFORE
    assert Path(template['previous_campaign_root']).resolve() == PARENT.resolve()
    assert template['parents_pending_natural18h_complete_update_save'] is True
    current = load(ROOT / 'protocol/USER_AUTHORITY.json')
    normalized = dict(current); normalized['extension_parents'] = {}
    normalized['parents_pending_natural18h_complete_update_save'] = True
    assert normalized == template, 'Authority changed beyond approved parent materialization'
    return ready, template


def natural_parent(branch, now=None):
    now = time.time() if now is None else now
    if now < NOT_BEFORE: raise NaturalSavePending('Must not interrupt pre18h training')
    out = PARENT / 'runs' / branch
    stopped = load(out / 'TRAINING_STOPPED.json')
    assert stopped['reason'] == 'deadline' and stopped['deadline_unix'] == NOT_BEFORE
    assert stopped['detected_unix'] >= NOT_BEFORE
    latest = load(out / 'LATEST.json'); p = Path(latest['path']).resolve()
    assert p.is_relative_to(out / 'checkpoints') and p.is_file()
    side = load(p.with_suffix('.json'))
    if side != latest: raise NaturalSavePending('LATEST sidecar publication not complete')
    assert latest['status'] == 'PASS' and latest['full_state'] is True
    assert latest['phase'] == 'formal' and latest['branch'] == branch
    if not all(latest[k] == stopped[k] for k in ['update', 'cursor']):
        raise NaturalSavePending('TRAINING_STOPPED precedes final LATEST publication')
    assert p.stat().st_size == latest['bytes']
    final = load(out / 'FULL_SHA_FINAL.json')
    digests = final['all_rank_full_digests']
    assert final['status'] == 'PASS' and final['sampled'] is False
    assert len(digests) == (3 if branch == 'original' else 5)
    assert len(set(digests)) == 1 and digests[0] == final['parameter_digest_full']
    assert len(digests[0]) == 64
    if final['verified_unix'] < stopped['detected_unix']:
        raise NaturalSavePending('Final full digest publication pending')
    # A FINAL save can reuse an already saved same-update file; a null sidecar
    # parameter digest is therefore valid. The frozen CPU auditor reads all151.
    selected = {k: latest[k] for k in ['path', 'sha256', 'update', 'cursor']}
    selected['cpu_audit_path'] = str(ROOT / 'evidence' / ('PARENT_STATE_AUDIT_' + branch + '.json'))
    return selected


def assert_parent_unchanged(branch, selected):
    current = natural_parent(branch)
    assert current == selected, 'Naturally stopped parent advanced during CPU audit'
    out = PARENT / 'runs' / branch
    for rank in range(3 if branch == 'original' else 5):
        log = out / ('UPDATES_rank' + str(rank) + '.jsonl')
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 262144)); lines = f.read().splitlines()
        last = json.loads(lines[-1])
        assert last['update'] == selected['update'] and last['cursor'] == selected['cursor']
        # rank0 makes the broadcast global boundary decision. Other ranks can
        # enter the same authorized in-flight update a few milliseconds later.
        if rank == 0:
            assert last['started_unix'] < NOT_BEFORE, 'Global optimizer update started after old cutoff'


def bind_parent_audit(branch, selected, audit):
    assert audit['status'] == 'PASS'
    assert audit['source_sha256'] == '578b36c49e16e8c729e8586f02619e1f38aab9398c6acc0010cdc386befabc43'
    assert audit['policy_sha256'] == '61bc51ce136f314de7cbbf8ea62c0c6dd1def3cce8881e03b9913879f76a7d3e'
    checked = audit['checkpoint']
    assert all(checked[k] == selected[k] for k in ['path', 'sha256', 'update', 'cursor'])
    final_path = PARENT / 'runs' / branch / 'FULL_SHA_FINAL.json'
    final = load(final_path)
    digest = checked['RAW']['full_parameter_digest']
    assert final['parameter_digest_full'] == digest, 'CPU RAW bytes differ from terminal full-rank digest'
    assert set(final['all_rank_full_digests']) == {digest}
    return {'status': 'PASS', 'parent': selected, 'CPU_RAW_full_digest': digest,
        'terminal_rank_digest_file': str(final_path), 'terminal_rank_digest_file_sha256': sha(final_path),
        'parent_CPU_audit_sha256': sha(selected['cpu_audit_path']) if Path(selected['cpu_audit_path']).exists() else None}


def proc_snapshot():
    result = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = p.read_text(); f = raw[raw.rfind(')') + 2:].split(); pid = int(p.parent.name)
            result[pid] = {'pid': pid, 'state': f[0], 'ppid': int(f[1]), 'pgid': int(f[2]),
                'sid': int(f[3]), 'start_ticks': f[19],
                'command': (p.parent / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
        except (OSError, ValueError, IndexError): pass
    return result


def same_identity(current, expected):
    return current is not None and str(current['start_ticks']) == str(expected['start_ticks'])


def scope_inventory(processes=None):
    processes = proc_snapshot() if processes is None else processes
    ctrl = load(PARENT / 'evidence/ACTIVE_CONTROLLER.json')
    fin = load(PARENT / 'evidence/ACTIVE_CPU_FINALIZER.json')
    old_training = load(PARENT / 'protocol/RUN_SOURCE.json')['files']
    old_post = load(PARENT / 'protocol/POSTPROCESS_SOURCE.json')['files']
    for actor, rel, frozen in [(ctrl, 'controller.py', old_training),
                               (fin, 'scripts/finalize_campaign.py', old_post)]:
        assert Path(actor['script']).resolve() == (PARENT / rel).resolve()
        assert actor['source_sha256'] == frozen[rel] == sha(PARENT / rel), 'Historical actor source identity differs'
    anchors = [ctrl, fin]; trusted_members = []
    for p in (PARENT / 'evidence').glob('ACTOR_OWNED_*.json'):
        item = load(p); job = item['job']; owner = job.get('controller', {})
        if owner.get('pid') != ctrl['pid'] or str(owner.get('start_ticks')) != str(ctrl['start_ticks']): continue
        anchors.append(job); trusted_members += item.get('members', [])
    for branch in BRANCHES:
        anchors += [load(p) for p in (PARENT / 'runs' / branch).glob('RANK_*.json')]
    owned = {int(a['pid']) for a in anchors + trusted_members if same_identity(processes.get(int(a['pid'])), a)}
    assert not (owned & FORBIDDEN_PIDS), 'External context unexpectedly entered owned set'
    sessions = {processes[pid]['sid'] for pid in owned
                if processes[pid]['sid'] == pid and processes[pid]['pgid'] == pid}
    while True:
        before = set(owned)
        owned |= {pid for pid, q in processes.items() if q['ppid'] in owned}
        owned |= {pid for pid, q in processes.items() if q['sid'] in sessions}
        assert not (owned & FORBIDDEN_PIDS), 'External context found among descendants'
        if before == owned: break
    groups = sorted({processes[pid]['pgid'] for pid in owned})
    assert all(g > 1 for g in groups)
    outsiders = [q for q in processes.values() if q['pgid'] in groups and q['pid'] not in owned]
    assert not outsiders, ('Unowned members in proposed process group', outsiders)
    first = sorted({processes[int(a['pid'])]['pgid'] for a in [ctrl, fin]
                    if same_identity(processes.get(int(a['pid'])), a)})
    return {'unix': time.time(), 'controller': ctrl, 'finalizer': fin,
        'members': {str(pid): processes[pid] for pid in sorted(owned)}, 'groups': groups,
        'first_controller_finalizer_groups': first,
        'then_training_eval_loader_groups': [g for g in groups if g not in first],
        'unowned_members': outsiders, 'historical_actor_source_hashes_verified': True,
        'external_processes_signalled': 0, 'signals_sent': 0}


def verified_group_members(pgid, initial, current):
    members = [q for q in current.values() if q['pgid'] == pgid]
    for q in members:
        expected = initial.get(str(q['pid']))
        assert q['pid'] not in FORBIDDEN_PIDS and expected
        assert all(q[k] == expected[k] for k in ['start_ticks', 'pgid', 'sid']), ('Changed/reused PGID membership', q)
    return members


def remaining(initial, current=None):
    current = proc_snapshot() if current is None else current
    return [q for pid, q in current.items()
            if str(pid) in initial and same_identity(q, initial[str(pid)]) and q['state'] != 'Z']


def stop_owned(proposal):
    initial = proposal['members']; signals = []
    receipt = ROOT / 'evidence/PARENT_ACTOR_HANDOFF.json'
    if receipt.exists():
        assert load(receipt)['status'] == 'PASS_OWN_ACTORS_STOPPED_FOR_EXTENSION'
        assert not remaining(initial), 'Completed handoff has live old actors again; review required'
        return
    proposal_path = ROOT / 'evidence' / ('PARENT_ACTOR_STOP_PROPOSAL_' + str(time.time_ns()) + '.json')
    write(proposal_path, {'status': 'VERIFIED_PROPOSED_OWN_GROUPS_BEFORE_SIGNAL', **proposal})
    for groups in [proposal['first_controller_finalizer_groups'], proposal['then_training_eval_loader_groups']]:
        for sig, seconds in [(signal.SIGTERM, 30), (signal.SIGKILL, 30)]:
            for pgid in groups:
                members = verified_group_members(pgid, initial, proc_snapshot())
                if any(q['state'] != 'Z' for q in members):
                    record = {'pgid': pgid, 'signal': int(sig), 'members': members, 'unix': time.time()}
                    signals.append(record)
                    write(ROOT / 'evidence/PARENT_ACTOR_SIGNALS.json', {'signals': signals, 'proposal': str(proposal_path)})
                    try: os.killpg(pgid, sig)
                    except ProcessLookupError: pass
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if not any(q['pgid'] in groups for q in remaining(initial)): break
                time.sleep(1)
            if not any(q['pgid'] in groups for q in remaining(initial)): break
        assert not any(q['pgid'] in groups for q in remaining(initial)), 'Own PGID failed to exit'
    assert not remaining(initial), 'Own descendants still present'
    write(ROOT / 'evidence/PARENT_ACTOR_HANDOFF.json', {'status': 'PASS_OWN_ACTORS_STOPPED_FOR_EXTENSION',
        'unix': time.time(), 'proposal': str(proposal_path), 'proposal_sha256': sha(proposal_path),
        'signal_records': signals, 'all_verified_own_non_zombie_members_exited': True,
        'external_processes_signalled': 0, 'forbidden_external_PIDs': sorted(FORBIDDEN_PIDS)})


def launch(script, label):
    active = ROOT / 'evidence' / ('ACTIVE_' + label + '.json')
    if active.exists():
        prior = load(active); current = proc_snapshot().get(int(prior['pid']))
        assert prior['script'] == str(ROOT / script) and prior['source_sha256'] == sha(ROOT / script)
        if same_identity(current, prior): return {'reused_alive': True, **prior}
        raise AssertionError(('Prior successor actor exited; root review required, not duplicate launch', label, prior))
    env = os.environ.copy(); env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', PYTHONDONTWRITEBYTECODE='1')
    log = ROOT / 'evidence' / (label.lower() + '_launch_' + str(time.time_ns()) + '.log')
    args = ['bash', str(RUN_ENV), str(ROOT / script)]
    with log.open('wb') as f:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
    identity = proc_snapshot()[process.pid]
    record = {'pid': process.pid, 'start_ticks': identity['start_ticks'], 'pgid': identity['pgid'],
        'command': args, 'log': str(log), 'script': str(ROOT / script), 'source_sha256': sha(ROOT / script),
        'deadline_unix': DEADLINE, 'started_unix': time.time()}
    write(ROOT / 'evidence' / (label + '_LAUNCH.json'), record)
    # Actor overwrites ACTIVE on entry with its own richer identity; this early
    # pin prevents a coordinator retry from duplicating a still-starting actor.
    write(active, record)
    time.sleep(2)
    assert process.poll() is None, ('Successor actor exited during startup', record)
    return record


def main():
    ap = argparse.ArgumentParser(); mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    args = ap.parse_args(); ready, template = readiness()
    if args.prepare:
        print(json.dumps({'status': 'SOURCE_READY_WAITING_NATURAL18H_PARENTS', 'not_before_unix': NOT_BEFORE,
            'deadline_unix': DEADLINE, 'signals_sent': 0, 'optimizer_updates': 0})); return
    lock = (ROOT / 'HANDOFF.lock').open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    me = proc_snapshot()[os.getpid()]
    write(ROOT / 'evidence/ACTIVE_HANDOFF_COORDINATOR.json', {'pid': os.getpid(), 'start_ticks': me['start_ticks'],
        'source_sha256': sha(__file__), 'started_unix': time.time(), 'not_before_unix': NOT_BEFORE})
    try:
        while True:
            assert time.time() < DEADLINE, 'Successor deadline reached before handoff; no update launch allowed'
            readiness()
            try:
                parents = {b: natural_parent(b) for b in BRANCHES}
                break
            except (NaturalSavePending, FileNotFoundError) as exc:
                status('WAITING_NATURAL18H_COMPLETE_SAVES', reason=str(exc), signals_sent=0)
                time.sleep(10)
        selection = ROOT / 'evidence/SELECTED_PARENTS.json'
        selected_report = {'status': 'NATURAL18H_COMPLETE_SAVES_SELECTED', 'unix': time.time(), 'parents': parents}
        if selection.exists(): assert load(selection)['parents'] == parents
        else: write(selection, selected_report)
        bindings = {}
        for b in BRANCHES:
            status('AUDITING_NATURAL18H_PARENT', branch=b, signals_sent=0)
            env = os.environ.copy(); env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', PYTHONDONTWRITEBYTECODE='1')
            log = ROOT / 'evidence' / ('PARENT_STATE_AUDIT_' + b + '.log')
            with log.open('ab') as f:
                result = subprocess.run(['bash', str(RUN_ENV), str(ROOT / 'scripts/audit_frozen_parent.py'),
                    '--branch', b, '--selection', str(selection)], env=env, stdin=subprocess.DEVNULL,
                    stdout=f, stderr=subprocess.STDOUT, timeout=1800)
            assert result.returncode == 0, ('Frozen parent CPU audit failed', b, str(log))
            audit = load(parents[b]['cpu_audit_path']); assert audit['status'] == 'PASS'
            assert all(audit['checkpoint'][k] == parents[b][k] for k in ['path', 'sha256', 'update', 'cursor'])
            assert_parent_unchanged(b, parents[b])
            bindings[b] = bind_parent_audit(b, parents[b], audit)
        ready, template = readiness()
        for b in BRANCHES:
            assert_parent_unchanged(b, parents[b])
            bindings[b] = bind_parent_audit(b, parents[b], load(parents[b]['cpu_audit_path']))
        write(ROOT / 'evidence/PARENT_FINAL_DIGEST_BINDINGS.json', {'status': 'PASS', 'unix': time.time(), 'branches': bindings})
        authority = dict(template); authority['extension_parents'] = parents
        authority['parents_pending_natural18h_complete_update_save'] = False
        write(ROOT / 'protocol/USER_AUTHORITY.json', authority)
        files = dict(ready['run_source_static_files'])
        for rel in ['protocol/USER_AUTHORITY.json'] + ['evidence/PARENT_STATE_AUDIT_' + b + '.json' for b in BRANCHES]:
            files[rel] = sha(ROOT / rel)
        manifest = {'version': 1, 'created_unix': time.time(), 'files': files,
            'scope': 'Midnight continuation from naturally stopped18h full CPU-accepted parents; scientific recipe unchanged',
            'ready_manifest_sha256': sha(ROOT / 'evidence/SUCCESSOR_SOURCE_READY.json')}
        existing = ROOT / 'protocol/RUN_SOURCE.json'
        if existing.exists(): assert load(existing)['files'] == files
        else: write(existing, manifest)
        readiness()
        status('PARENTS_CPU_PASS_SUCCESSOR_FROZEN_READY_FOR_SCOPED_HANDOFF', parents=parents)
        # This is the first possible process signal, strictly after both CPU
        # acceptances and all predefined successor source identities passed.
        stop_owned(scope_inventory())
        assert time.time() < DEADLINE
        controller = launch('controller.py', 'CONTROLLER')
        finalizer = launch('scripts/finalize_campaign.py', 'CPU_FINALIZER')
        status('SUCCESSOR_ACTORS_LAUNCHED_ENGINEERING_GATES_PENDING', parents=parents,
            controller=controller, finalizer=finalizer, direct_optimizer_updates=0)
    except BaseException as exc:
        write(ROOT / 'evidence' / ('HANDOFF_FAILURE_' + str(time.time_ns()) + '.json'),
            {'status': 'FAIL', 'error': repr(exc), 'traceback': traceback.format_exc(), 'unix': time.time()})
        status('HANDOFF_REQUIRES_REVIEW', error=repr(exc)); raise


if __name__ == '__main__': main()

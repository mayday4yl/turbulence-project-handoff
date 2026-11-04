"""CPU mocks only: no real process, signal, network or optimizer operations."""
import hashlib
import importlib.util
import json
import os
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def proc(pid, start, pgid=None, sid=None, ppid=1):
    return {'pid': pid, 'start_ticks': str(start), 'pgid': pgid or pid,
            'sid': sid or pgid or pid, 'ppid': ppid, 'state': 'S'}


def case(name, orphan=False, unknown_orphan=False, outsider=False, external=False,
         reuse_after_term=False, ignore_term=False, reused_anchor=False):
    spec = importlib.util.spec_from_file_location('mock_cleanup_' + name, ROOT / 'controller.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory(prefix='cleanup-mock-') as tmp:
        mod.ROOT = Path(tmp)
        (mod.ROOT / 'evidence').mkdir()
        out = mod.ROOT / 'run'
        out.mkdir()
        (out / 'RANK_0.json').write_text(json.dumps({'pid': 201, 'start_ticks': '1100'}))
        current = {100: proc(100, 1000), 201: proc(201, 1100, ppid=100),
                   211: proc(211, 1200, 201, 201, 201)}
        calls = []
        clock = SimpleNamespace(t=0., serial=0)

        def tick():
            clock.serial += 1
            return clock.serial

        mod.time = SimpleNamespace(time=lambda: 1800. + clock.t,
                                   monotonic=lambda: clock.t, time_ns=tick,
                                   sleep=lambda n: setattr(clock, 't', clock.t + 10))
        mod._proc_snapshot = lambda: {pid: dict(q) for pid, q in current.items()}

        def fake_killpg(pgid, signum):
            calls.append((pgid, int(signum)))
            if reuse_after_term and pgid == 201 and signum == signal.SIGTERM:
                current[211] = proc(211, 9999, 201, 201)
                return
            if ignore_term and pgid == 201 and signum == signal.SIGTERM:
                return
            for pid in [pid for pid, q in current.items() if q['pgid'] == pgid]:
                del current[pid]

        # Deliberately expose no real kill or killpg in the module's os object.
        mod.os = SimpleNamespace(getpid=os.getpid, replace=os.replace, killpg=fake_killpg)

        class FakeProcess:
            pid = 100

            def poll(self):
                return None if 100 in current else 0

            def wait(self, timeout):
                assert 100 not in current
                return 0

        job = {'name': 'mock', 'record': {'pid': 100, 'start_ticks': '1000', 'name': 'mock'},
               'process': FakeProcess(), 'out': out}
        if orphan:
            mod.cache_owned(job)
            current.pop(100)
            current.pop(201)
            current[211]['ppid'] = 1
        if unknown_orphan:
            current.pop(100)
            current.pop(201)
            current[211]['ppid'] = 1
        if outsider:
            current[999] = proc(999, 1300, 201, 999)
        if external:
            current[2938958] = proc(2938958, 1300, 201, 201)
        if reused_anchor:
            current[201] = proc(201, 9999)
            current.pop(211)
        rejected = False
        try:
            mod.terminate_own_group(job, 'mock-only')
        except AssertionError:
            rejected = True
        must_reject = unknown_orphan or outsider or external or reuse_after_term or reused_anchor
        assert rejected == bool(must_reject), (name, rejected, calls)
        if unknown_orphan or outsider or external or reused_anchor:
            assert not calls, (name, calls)
        if reuse_after_term:
            assert (201, int(signal.SIGKILL)) not in calls
        if ignore_term:
            assert (201, int(signal.SIGKILL)) in calls
        if orphan:
            assert (201, int(signal.SIGTERM)) in calls
        assert all(pgid != 2938958 for pgid, _ in calls)
        return {'name': name, 'PASS': True, 'rejected_as_expected': bool(must_reject),
                'mock_signal_calls': calls, 'actual_signals': 0}


def main():
    cases = [case('separate_rank_groups_and_loader'),
             case('cached_orphan_loader_after_root_and_rank_exit', orphan=True),
             case('term_ignored_requires_same_set_kill', ignore_term=True),
             case('uncached_orphan_refused', unknown_orphan=True),
             case('unowned_group_member_refused', outsider=True),
             case('external_context_never_signalled', external=True),
             case('changed_PID_start_after_TERM_refuses_KILL', reuse_after_term=True),
             case('reused_rank_PID_refused', reused_anchor=True)]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    q = {'status': 'PASS', 'controller_sha256': sha(ROOT / 'controller.py'),
         'test_sha256': sha(__file__), 'cases': cases, 'actual_signals': 0,
         'model_objects': 0, 'optimizer_steps': 0, 'NPU_calls': 0,
         'scope': 'Mock /proc and process/signal/time APIs only; no real processes are signalled.'}
    out = ROOT / 'evidence/EXTENSION_CONTROLLER_CLEANUP_MOCK_TESTS.json'
    out.write_text(json.dumps(q, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'cases': len(cases), 'controller_sha256': q['controller_sha256'], 'report': str(out)}))


if __name__ == '__main__':
    main()

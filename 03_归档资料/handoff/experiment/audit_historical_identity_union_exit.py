"""Read-only Linux /proc exit audit across the three authorized UniDB campaigns.

No signals, subprocess launch, checkpoint load, device import or remote writes.
Run remotely through stdin; JSON on stdout is retained locally by the caller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT_NAMES = ['unidb_dual_20261103_until_noon', 'unidb_dual_20261103_until18',
              'unidb_dual_20261104_until00']
PROTECTED_IDENTITIES = {(1506794, '596454736'), (1504769, '596444189')}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_proc(base=Path('/proc')):
    found, errors = {}, []
    for f in base.glob('[0-9]*/stat'):
        try:
            fields = f.read_text().rsplit(')', 1)[1].split()
            pid = int(f.parent.name)
            found[pid] = dict(pid=pid, state=fields[0], ppid=int(fields[1]),
                              pgid=int(fields[2]), sid=int(fields[3]), start_ticks=fields[19])
            try:
                found[pid]['command'] = (f.parent/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except PermissionError:
                found[pid]['command'] = None
                errors.append({'pid': pid, 'reason': 'cmdline_permission_denied'})
            except FileNotFoundError:
                found[pid]['command'] = ''
        except FileNotFoundError:
            pass
        except (OSError, ValueError, IndexError) as e:
            errors.append({'path': str(f), 'reason': str(e)})
    return found, errors


def identities(value):
    if isinstance(value, dict):
        if 'pid' in value and 'start_ticks' in value:
            try:
                pid, start = int(value['pid']), str(value['start_ticks'])
                if pid > 1 and start.isdigit():
                    yield {k: value[k] for k in ['pid', 'start_ticks', 'pgid', 'sid'] if k in value}
            except (ValueError, TypeError):
                pass
        for child in value.values():
            yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def source_files(roots):
    patterns = ['evidence/ACTIVE_*.json', 'evidence/*LAUNCH*.json',
                'evidence/ACTOR_*.json', 'evidence/CLEANUP_VERIFIED_*.json',
                'evidence/PARENT_ACTOR_HANDOFF.json', 'evidence/PARENT_ACTOR_STOP_PROPOSAL_*.json',
                'evidence/owned_process_inventory/*.json',
                'evidence/midpoint_U10000/ACTIVE_CPU_AUDIT.json',
                'runs/*/RANK_*.json', 'gates/*/RANK_*.json',
                'briefing_20261103_1900/ACTIVE_COLLECTOR.json',
                'briefing_20261103_1900/COLLECTOR_LAUNCH.json']
    return sorted({f for root in roots for pattern in patterns for f in root.glob(pattern) if f.is_file()})


def classify(current, records, root_strings, exclude):
    same, gone, zombies = [], [], []
    groups, sessions = set(), set()
    for (pid, start), rec in records.items():
        # Also catch orphan groups whose isolated leader was a recorded actor.
        # These are conservative review candidates, not an ownership assertion.
        groups.add(pid)
        sessions.add(pid)
        for key, target in [('pgid', groups), ('sid', sessions)]:
            if rec.get(key) and int(rec[key]) > 1:
                target.add(int(rec[key]))
        now = current.get(pid)
        if now and now['start_ticks'] == start:
            (zombies if now['state'] == 'Z' else same).append(now)
        else:
            gone.append({'pid': pid, 'start_ticks': start, 'current_start_ticks': now['start_ticks'] if now else None})
    same_ids = {q['pid'] for q in same}
    unexplained = []
    for pid, q in current.items():
        if pid in exclude or (pid, q['start_ticks']) in PROTECTED_IDENTITIES or q['state'] == 'Z' or pid in same_ids:
            continue
        reasons = []
        if q['pgid'] in groups or q['sid'] in sessions:
            reasons.append('recorded_PGID_or_SID_membership_requires_review')
        if any(r in (q.get('command') or '') for r in root_strings):
            reasons.append('campaign_path_in_live_command')
        if reasons:
            unexplained.append({**q, 'reasons': reasons})
    return same, gone, zombies, unexplained


def coverage(roots, sources, records):
    paths = {x['path'] for x in sources}
    checks = {}
    for root in roots:
        for role in ['CONTROLLER', 'CPU_FINALIZER']:
            path = str(root/'evidence'/('ACTIVE_'+role+'.json'))
            checks[root.name+':'+role] = path in paths and any(path in r['sources'] for r in records.values())
        for branch, world in [('original', 3), ('improved', 5)]:
            for rank in range(world):
                path = str(root/'runs'/branch/('RANK_'+str(rank)+'.json'))
                checks[root.name+':'+branch+':rank'+str(rank)] = path in paths and any(path in r['sources'] for r in records.values())
        if root.name.endswith('until_noon'):
            checks[root.name+':historical_groups'] = any(str(root/'evidence/owned_process_inventory')+'/' in s for s in paths)
        else:
            checks[root.name+':owned_groups'] = any(str(root/'evidence/ACTOR_OWNED_') in s for s in paths)
    latest = roots[-1]
    for rel in ['evidence/ACTIVE_HANDOFF_COORDINATOR.json', 'evidence/ACTIVE_STARTUP_AUDIT.json',
                'briefing_20261103_1900/ACTIVE_COLLECTOR.json']:
        path = str(latest/rel)
        checks['current_helper:'+rel] = path in paths and any(path in r['sources'] for r in records.values())
    path = str(roots[0]/'evidence/midpoint_U10000/ACTIVE_CPU_AUDIT.json')
    checks['noon_midpoint_CPU_auditor'] = path in paths and any(path in r['sources'] for r in records.values())
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/data2/yanglin')
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    roots = [Path(args.base)/name for name in ROOT_NAMES]
    sources, records, parse_errors = [], {}, []
    for f in source_files(roots):
        try:
            value = json.loads(f.read_text())
            sources.append({'path': str(f), 'sha256': sha(f)})
            for rec in identities(value):
                pid, start = int(rec['pid']), str(rec['start_ticks'])
                if (pid, start) in PROTECTED_IDENTITIES:
                    continue
                key = (pid, start)
                records.setdefault(key, {'pid': pid, 'start_ticks': start, 'sources': []})
                records[key].update({k: int(rec[k]) for k in ['pgid', 'sid'] if k in rec})
                records[key]['sources'].append(str(f))
        except (OSError, ValueError, TypeError) as e:
            parse_errors.append({'path': str(f), 'error': repr(e)})
    current, proc_errors = read_proc()
    excluded = {os.getpid()}
    pid = os.getpid()
    while pid in current and current[pid]['ppid'] > 1:
        pid = current[pid]['ppid']
        if pid in excluded:
            break
        excluded.add(pid)
    same, gone, zombies, unexplained = classify(current, records, [str(r) for r in roots], excluded)
    requirements, pending = {}, []
    final = roots[-1]
    for rel in ['TRAINING_AND_MODEL_VALIDATION_COMPLETE.json', 'evidence/CONTROLLER_EXIT.json',
                'evidence/CPU_POSTPROCESS_ACCEPTANCE.json', 'evidence/CPU_FINALIZER_EXIT.json']:
        f = final/rel
        if not f.exists():
            pending.append(rel+' missing')
            continue
        q = json.loads(f.read_text())
        requirements[rel] = {'sha256': sha(f), 'record': q}
        if rel.endswith('CONTROLLER_EXIT.json'):
            ok = q.get('all_subprocesses_exited') is True
        elif rel.endswith('CPU_FINALIZER_EXIT.json'):
            ok = q.get('status') == 'CPU_AUDIT_AND_REPORT_PASS_VISUAL_AND_LOCAL_DELIVERY_PENDING'
        else:
            ok = q.get('status') == 'PASS'
        if not ok:
            pending.append(rel+' incomplete')
    coverage_checks = coverage(roots, sources, records)
    enough = len(records) >= 20 and all(coverage_checks.values())
    ok = enough and not same and not unexplained and not parse_errors and not proc_errors
    if args.require_complete:
        ok = ok and not pending
    report = {'status': 'PASS_ALL_OWNED_PROCESSES_EXITED' if ok else 'PENDING_OWNED_PROCESSES_OR_RECEIPTS',
              'scope': 'Three authorized UniDB campaign roots including historical PGID/SID, loaders/resource_tracker and CPU helpers',
              'unix': time.time(), 'final_requested': args.require_complete,
              'all_owned_exited': bool(ok), 'source_records': sources, 'identity_count': len(records),
              'identity_records': list(records.values()), 'required_identity_coverage': enough,
              'identity_coverage_checks': coverage_checks,
              'same_identity_live': same, 'gone_or_pid_reused': gone, 'zombies_not_executing': zombies,
              'unexplained_group_session_or_command_matches': unexplained, 'parse_errors': parse_errors,
              'proc_read_errors': proc_errors, 'excluded_probe_ancestors': sorted(excluded),
              'protected_external_OS_identities': sorted(PROTECTED_IDENTITIES),
              'driver_context_2938958_is_not_an_OS_identity_claim': True,
              'required_completion_receipts': requirements, 'completion_pending': pending,
              'signals_sent': 0, 'model_calls': 0, 'optimizer_updates': 0, 'NPU_calls': 0}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 2



records_list = [{'pid': 2013568, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013569, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013570, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013571, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013572, 'start_ticks': '598538159', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013573, 'start_ticks': '598538160', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013574, 'start_ticks': '598538160', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013575, 'start_ticks': '598538160', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013576, 'start_ticks': '598538160', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013193, 'start_ticks': '598538113', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013194, 'start_ticks': '598538113', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013577, 'start_ticks': '598538160', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013578, 'start_ticks': '598538160', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013579, 'start_ticks': '598538161', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013580, 'start_ticks': '598538161', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013328, 'start_ticks': '598538123', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013329, 'start_ticks': '598538123', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013466, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013339, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013340, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013341, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013342, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013343, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013344, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013345, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013346, 'start_ticks': '598538128', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013347, 'start_ticks': '598538129', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013467, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013468, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013469, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013470, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013471, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013472, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013473, 'start_ticks': '598538147', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2015276, 'start_ticks': '598540785', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2015277, 'start_ticks': '598540849', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2015281, 'start_ticks': '598540861', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013492, 'start_ticks': '598538150', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013244, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2015294, 'start_ticks': '598541759', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2015296, 'start_ticks': '598541953', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013249, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2015297, 'start_ticks': '598541978', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013254, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013256, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013258, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013261, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013263, 'start_ticks': '598538118', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013136, 'start_ticks': '598538109', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013267, 'start_ticks': '598538119', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013268, 'start_ticks': '598538119', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013269, 'start_ticks': '598538119', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012692, 'sid': 2012692}, {'pid': 2013670, 'start_ticks': '598538194', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013671, 'start_ticks': '598538194', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013672, 'start_ticks': '598538195', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013673, 'start_ticks': '598538195', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013674, 'start_ticks': '598538195', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013675, 'start_ticks': '598538195', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013676, 'start_ticks': '598538195', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013677, 'start_ticks': '598538195', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013553, 'start_ticks': '598538154', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013555, 'start_ticks': '598538155', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012691, 'sid': 2012691}, {'pid': 2013564, 'start_ticks': '598538157', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013565, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013566, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}, {'pid': 2013567, 'start_ticks': '598538158', 'sources': ['/data2/yanglin/unidb_dual_20261104_until00/evidence/ACTOR_OWNED_2012492_598536680.json'], 'pgid': 2012690, 'sid': 2012690}]
info = {'earlier_audit_sha256': '55aa079a5382324b4960e9636eef1769bf9b1b258d756e6e1bad3fdd4b6ac25c', 'final_audit_sha256': 'e1817f0e9752d9ae31fc2fc0c66a8a4b6ee68834ad126d0105e0303b252122e8', 'earlier_identity_count': 534, 'final_identity_count': 468, 'additional_historical_count': 66, 'union_count': 534}
current, errors = read_proc()
exclude={os.getpid()};pid=os.getpid()
while pid in current and current[pid]['ppid']>1:
 pid=current[pid]['ppid']
 if pid in exclude:break
 exclude.add(pid)
records={(q['pid'],str(q['start_ticks'])):q for q in records_list}
same,gone,zombies,unexplained=classify(current,records,['/data2/yanglin/'+r for r in ROOT_NAMES],exclude)
ok=not same and not zombies and not unexplained and not errors and len(gone)==len(records_list)
report={'status':'PASS_HISTORICAL_IDENTITY_UNION_EXIT' if ok else 'FAIL','unix':time.time(),**info,'historical_records':records_list,'same_identity_live':same,'gone_or_pid_reused':gone,'zombies':zombies,'unexplained_matches':unexplained,'proc_read_errors':errors,'signals_sent':0,'model_calls':0,'scope':'Read current /proc for the 66 original loader/resource_tracker identities no longer present in final ACTOR_OWNED snapshot; final 468 plus these 66 cover historical union 534'}
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if ok else 2)

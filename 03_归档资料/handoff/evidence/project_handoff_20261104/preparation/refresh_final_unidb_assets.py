"""Refresh final UniDB asset roles from accepted small receipts, never payloads.

Only writes ASSET_REGISTER.csv, asset_evidence.json, OPERATIONS_GUIDE_ZH.md and
an explicit refresh receipt in this preparation directory. Original asset
builder and remote plan/results remain untouched. Four accepted adapter hashes
must be supplied together; callers cannot silently reuse an older exit audit.
"""
from pathlib import Path
import argparse
import copy
import csv
import datetime
import hashlib
import io
import json


HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
SNAPSHOT = HERE / 'asset_preparation_snapshot_20261104'
CAMPAIGN = BASE / 'unidb_dual_20261104_until00'
EXPECTED = {
    'original:latest': (67180, '0381ae3435d2939a2b8b4fbc215c3f78ac6c3775f254bead31dee3e9cdb78e98'),
    'original:best': (67180, '0381ae3435d2939a2b8b4fbc215c3f78ac6c3775f254bead31dee3e9cdb78e98'),
    'improved:latest': (40848, '11a19694f8fc2f4f343e97fab6d3a326759253dbb8eb6b68d2275e144e2cbfc9'),
    'improved:best': (10000, '728c30638ee41259c6302b53979f23e115e618c08dca7c2156db8f15e1105a75'),
}
PROTECTED = ['build_asset_register.py', 'REMOTE_PATH_CHECK_PLAN.json',
             'REMOTE_PATH_CHECK_RESULT.json', 'REMOTE_PATH_CHECK_RESULT_A100.json']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def small_json(path):
    path = Path(path)
    assert path.suffix == '.json', ('Refuse nonmetadata input', str(path))
    assert path.stat().st_size < 8_000_000, ('Metadata limit', str(path))
    data = path.read_bytes()
    return json.loads(data), {'path': str(path.resolve()), 'relative_path': str(path.resolve().relative_to(BASE)),
                              'sha256': digest(data), 'bytes': len(data), 'local_exists': True}


def local_metadata(remote):
    prefix = '/data2/yanglin/'
    assert remote.startswith(prefix)
    path = BASE / remote[len(prefix):]
    assert path.suffix == '.json'
    assert path.resolve().is_relative_to(BASE)
    return path


def identities(q):
    return {k: (v['update'], v['sha256']) for k, v in q['identities'].items()}


def stat_payload(path):
    # Do not open, read, torch.load or hash any checkpoint payload.
    p = Path(path)
    assert p.resolve().is_relative_to(BASE) and p.suffix == '.pt' and p.is_file()
    s = p.stat()
    return {'path': str(p), 'state': 'PRESENT', 'kind': 'file', 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--exit-sha256', required=True)
    parser.add_argument('--backup-sha256', required=True)
    parser.add_argument('--validation-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    for value in [args.receipt_sha256, args.exit_sha256, args.backup_sha256, args.validation_sha256]:
        assert len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    snap_manifest, snap_ident = small_json(SNAPSHOT / 'SNAPSHOT_MANIFEST.json')
    assert snap_manifest['original_rows'] == 77 and len(snap_manifest['files']) == 7
    for source in snap_manifest['files']:
        p = SNAPSHOT / source['relative_path']
        assert p.stat().st_size == source['bytes'] and digest(p.read_bytes()) == source['sha256']
    protected_before = {name: digest((HERE / name).read_bytes()) for name in PROTECTED}
    for name in PROTECTED:
        assert protected_before[name] == digest((SNAPSHOT / name).read_bytes()), name
    old_csv = list(csv.DictReader(io.StringIO((SNAPSHOT / 'ASSET_REGISTER.csv').read_text())))
    assert len(old_csv) == 77 and len({r['asset_id'] for r in old_csv}) == 77
    fields = list(old_csv[0])
    prior, _ = small_json(SNAPSHOT / 'asset_evidence.json')
    assert len(prior['assets']) == 77 and prior['asset_count'] == 77
    evidence = copy.deepcopy(prior)
    sources = evidence['sources']
    # Historical source facts remain fixed. Refuse rather than silently rewrite
    # an old row if a mutable source changed since preparation.
    for ident, source in sources.items():
        p = Path(source['path'])
        assert p.is_file() and digest(p.read_bytes()) == source['sha256'], ('Historical source changed', ident)

    def register(ident, path, expected_sha=None):
        value, identity = small_json(path)
        if expected_sha is not None:
            assert identity['sha256'] == expected_sha, (ident, 'Receipt hash mismatch')
        if ident in sources:
            assert sources[ident]['sha256'] == identity['sha256'], ident
        sources[ident] = identity
        return value

    receipt = register('UNIDB_FINAL_DELIVERY_RECEIPT', CAMPAIGN / 'delivery/experiment_full_delivery_receipt.json', args.receipt_sha256)
    exit_audit = register('UNIDB_FINAL_OWNED_EXIT', CAMPAIGN / 'delivery/all_owned_process_exit_audit.json', args.exit_sha256)
    backup = register('UNIDB_FINAL_UNIQUE_BACKUP', CAMPAIGN / 'delivery/unique_checkpoint_backup_audit.json', args.backup_sha256)
    validation = register('UNIDB_FINAL_ADAPTER_VALIDATION', CAMPAIGN / 'delivery/EXPERIMENT_FINAL_RECEIPT_VALIDATION.json', args.validation_sha256)
    assert receipt['status'] == backup['status'] == exit_audit['status'] == 'PASS'
    assert receipt['final'] is backup['final'] is exit_audit['final'] is True
    assert identities(receipt) == identities(backup) == identities(validation) == EXPECTED
    assert validation['status'] == 'PASS_EXPLICIT_NATIVE_FINAL_RECEIPT'
    assert receipt['all_four_checkpoint_state_roles_bound'] is True
    assert receipt['actual_final_unique_checkpoint_backups_accepted'] is True
    assert receipt['all_owned_processes_exited'] is True
    assert exit_audit['all_owned_exited'] is True and exit_audit['live_owned_actors'] == []
    assert exit_audit['external_signals_sent'] == 0
    # Final union must include the separately checked historical loader actors,
    # rather than the earlier468-only snapshot. Named identity bindings and the
    # adapter validator retain the underlying audit evidence.
    assert exit_audit.get('identity_count', 0) >= 534, 'Historical-loader exit supplement not yet bound'
    assert set(exit_audit['campaign_roots']) == {'unidb_dual_20261103_until_noon', 'unidb_dual_20261103_until18', 'unidb_dual_20261104_until00'}
    assert len(backup['files']) == backup['unique_file_count'] == 3 and backup['logical_role_count'] == 4
    assert backup['no_duplicate_download'] is True and backup['old_U10000_complete_byte_SHA_reused'] is True

    adapter_paths = {Path(sources[k]['path']).name: sources[k] for k in [
        'UNIDB_FINAL_DELIVERY_RECEIPT', 'UNIDB_FINAL_OWNED_EXIT', 'UNIDB_FINAL_UNIQUE_BACKUP']}
    for binding in validation['adapters'].values():
        actual = adapter_paths[Path(binding['path']).name]
        assert binding['path'] == actual['path'] and binding['sha256'] == actual['sha256']
    for binding in receipt['evidence_bindings']:
        if Path(binding['path']).name in adapter_paths:
            assert adapter_paths[Path(binding['path']).name]['sha256'] == binding['sha256']
    cpu_binding = next(b for b in backup['evidence_bindings'] if Path(b['path']).name == 'CPU_POSTPROCESS_ACCEPTANCE.json')
    cpu = register('UNIDB_FINAL_CPU_POSTPROCESS', Path(cpu_binding['path']), cpu_binding['sha256'])
    assert cpu['status'] == 'PASS' and len(cpu['refs']) == 4
    refs = {f"{r['branch']}:{r['role']}": r for r in cpu['refs']}
    assert set(refs) == set(EXPECTED)
    for role, ref in refs.items():
        assert (ref['update'], ref['checkpoint_sha256']) == EXPECTED[role]
        state = register('UNIDB_FINAL_STATE_' + role.replace(':', '_').upper(), local_metadata(ref['checkpoint_CPU_audit']), ref['checkpoint_CPU_audit_sha256'])
        ck = state['checkpoint']
        assert state['status'] == ck['status'] == 'PASS'
        assert ck['path'] == ref['checkpoint_path'] and ck['sha256'] == ref['checkpoint_sha256']
        assert ck['update'] == ref['update'] and ck['branch'] == ref['branch']
        assert ck['EMA']['full_parameter_digest'] == ref['checkpoint_EMA_full_digest']
        summary_id = 'UNIDB_FINAL_SUMMARY_' + ref['branch'].upper() + '_U' + str(ref['update'])
        summary = register(summary_id, local_metadata(ref['summary']), ref['summary_sha256'])
        assert summary['status'] == 'PASS' and summary['full'] is True and summary['weights'] == 'EMA'
        assert summary['count'] == 5000 and summary['per_g'] == 1000
        assert summary['branch'] == ref['branch'] and summary['update'] == ref['update']
        assert summary['parameter_digest'] == ref['checkpoint_EMA_full_digest']

    rows = copy.deepcopy(old_csv)
    assets = copy.deepcopy(prior['assets'])
    row_map = {r['asset_id']: r for r in rows}
    asset_map = {r['asset_id']: r for r in assets}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    final_source = sources['UNIDB_FINAL_DELIVERY_RECEIPT']
    root_row = row_map['UNIDB_MIDNIGHT_ROOT']
    root_row.update(local_evidence_path=final_source['path'], source_sha256=final_source['sha256'],
        verification_scope='Final SHA-bound experimental receipt, complete-state/EMA full5000 references and whole-campaign exit acceptance; refresh itself smallmetadata plus localstat only',
        remote_current_check='FINAL_ACCEPTANCE_RECEIPTS_JOINED_NO_NEW_REMOTE_CALL',
        notes='Training stopped atmidnight originalU67180/improvedU40848; final best originalU67180/improvedU10000. Fourroles/threeunique complete checkpoints locally byte-backed. RootQA/CPU/exit accepted; wholeproject document/package delivery separate.')
    row_map['UNIDB_PRIOR_NOON_BACKUP']['notes'] += '; Historical fourcandidate record retained. ImprovedU10000 exactsameSHA is now reused as finalimproved full-best; no extra download. See three final checkpoint rows.'
    row_map['UNIDB_18_DELIVERY_ABSENT']['notes'] += '; Historical declaredpath observation only, not a currentfinal backup gap: final accepted threeunique checkpoints are stored at the new explicitly listed paths.'
    for ident in ['UNIDB_MIDNIGHT_ROOT', 'UNIDB_PRIOR_NOON_BACKUP', 'UNIDB_18_DELIVERY_ABSENT']:
        asset_map[ident].update(row_map[ident])
    asset_map['UNIDB_MIDNIGHT_ROOT']['evidence_id'] = 'UNIDB_FINAL_DELIVERY_RECEIPT'
    asset_map['UNIDB_18_DELIVERY_ABSENT']['historical_snapshot_only'] = True

    final_stats = []
    seen_roles = set()
    ids = {('original', 67180): 'UNIDB_FINAL_ORIGINAL_U67180',
           ('improved', 40848): 'UNIDB_FINAL_IMPROVED_LATEST_U40848',
           ('improved', 10000): 'UNIDB_FINAL_IMPROVED_BEST_U10000'}
    for item in backup['files']:
        assert item['status'] == 'PASS_BYTE_IDENTICAL_FINAL'
        assert item['sha256'] == item['source_sha256']
        st = stat_payload(item['local_path'])
        assert st['bytes'] == item['bytes'] and st['mtime_ns'] == item['local_mtime_ns']
        byte_receipt = item['source_byte_receipt']
        register('UNIDB_FINAL_BYTES_' + item['branch'].upper() + '_U' + str(item['update']), Path(byte_receipt['path']), byte_receipt['sha256'])
        for role in item['roles']:
            role_key = item['branch'] + ':' + role
            assert role_key not in seen_roles
            seen_roles.add(role_key)
            ref = refs[role_key]
            assert (item['update'], item['sha256']) == EXPECTED[role_key]
            assert item['source_path'] == ref['checkpoint_path']
        roles = '+'.join(item['roles'])
        aid = ids[(item['branch'], item['update'])]
        note = f"Finalaccepted {item['branch']} {roles}; completeRAW/Adam/EMA/RNG/cursor/schedule state. EMA full5000 identity bound via CPU refs; localfullbyteSHA accepted in source receipt and reused, thisrefresh stat only."
        if item['update'] == 10000:
            note += ' Exact oldnoon U10000 candidate reused as finalimproved full-best; not duplicated, not latestU40848.'
        r = {k: '' for k in fields}
        r.update(asset_id=aid, category='accepted final complete checkpoint backup', version_scope='UniDB midnight final',
            host='liteserver-de36', canonical_path=item['source_path'], local_payload_path=st['path'],
            local_payload_state=st['state'], local_payload_kind=st['kind'], local_payload_bytes=str(st['bytes']),
            local_evidence_path=sources['UNIDB_FINAL_UNIQUE_BACKUP']['path'], local_evidence_exists='True',
            source_sha256=sources['UNIDB_FINAL_UNIQUE_BACKUP']['sha256'], historical_payload_sha256=item['sha256'],
            verification_scope='Final complete-state and role acceptance plus previously fullbyteSHA verified localbackup; thisrefresh local existence/size/mtime only, no payload read',
            split_or_population='EMA fiveG eachVal1000/native100stepSDE; best selection from declared completefull pool',
            test_policy='Val, not new Test; no new evaluation or inference in assetrefresh',
            protection_or_dependency='Preserve exact complete checkpoint and referenced source/Adam/EMA/RNG/cursor/scheduler identities; no resume/cleanup authorized',
            remote_current_check='FINAL_NATIVE_CPU_METADATA_BOUND_NO_NEW_REMOTE_CALL', notes=note)
        rows.append(r)
        assets.append({**r, 'local_stat': st, 'evidence_id': 'UNIDB_FINAL_UNIQUE_BACKUP',
                       'accepted_final_roles': item['roles'], 'branch': item['branch'], 'update': item['update']})
        final_stats.append({'asset_id': aid, 'source_path': item['source_path'], 'update': item['update'],
                            'branch': item['branch'], 'roles': item['roles'], 'sha256_reported_by_accepted_receipts': item['sha256'],
                            'source_byte_receipt': byte_receipt, 'local_stat': st, 'new_payload_reads': 0})
    assert seen_roles == set(EXPECTED) and len(rows) == len(assets) == 80
    assert sum(st['local_stat']['bytes'] for st in final_stats) == backup['total_unique_bytes'] == 6584061917
    changed_old_ids = {before['asset_id'] for before, after in zip(old_csv, rows[:77]) if before != after}
    assert changed_old_ids == {'UNIDB_MIDNIGHT_ROOT', 'UNIDB_PRIOR_NOON_BACKUP', 'UNIDB_18_DELIVERY_ABSENT'}
    assert len({r['asset_id'] for r in rows}) == 80
    register('ASSET_PRE_FINAL_SNAPSHOT', SNAPSHOT / 'SNAPSHOT_MANIFEST.json', snap_ident['sha256'])

    guide = (SNAPSHOT / 'OPERATIONS_GUIDE_ZH.md').read_text()
    guide += '\n\n## 10. 最终补充：午夜实验已验收的三份完整权重\n\n'
    guide += f'本补充生成于 `{now}`。前文及原77条资产的stat、旧noon候选与旧18路径缺失是准备期历史观察，保留原口径；本节补充后来通过的实际终评、完整状态/EMA身份、保存预测CPU复算、root目视、备份和全部自有进程退出证据，不改写旧receipt的pending字段。实验内容完成不代表总交接文档/资料包或用户消息已交付。\n\n'
    guide += '| 最终角色 | Update | 本地完整权重位置 | 接手含义 |\n|---|---:|---|---|\n'
    for item in backup['files']:
        guide += f"| {item['branch']} {'＋'.join(item['roles'])} | {item['update']} | `{item['local_path']}` | 完整RAW/Adam/EMA/RNG/cursor/scheduler；完整字节SHA复用已验receipt，本刷新只stat |\n"
    guide += '\n原线best=latest U67180；改进latest U40848，full-best U10000来自旧noon。四个逻辑角色只需要三份unique文件，共6,584,061,917字节。U10000复用旧候选，不重新下载或复制大checkpoint。质量报告与best/latest角色不能相互替代，也不能把monitor U3000当full-best。\n\n'
    guide += f"全部自有退出证据绑定 `{exit_audit['identity_count']}` 个精确身份及三段controller/rank/loader/CPU/helper范围，外部signals=0；这不是按历史PID猜测，也不对drivercontext推断OS存活。\n\n"
    guide += '只读入口：当前实验 `delivery/experiment_full_delivery_receipt.json`、`delivery/all_owned_process_exit_audit.json`、`delivery/unique_checkpoint_backup_audit.json` 与 `delivery/EXPERIMENT_FINAL_RECEIPT_VALIDATION.json`；其绝对路径/SHA在asset_evidence.json的新来源与final_unidb_update中。无需运行旧launcher、重新算模型、重复CPU指标或恢复训练。\n\n'
    guide += '本次使用 `refresh_final_unidb_assets.py` 从 `asset_preparation_snapshot_20261104` 增量刷新。旧 `build_asset_register.py` 仅作历史生成器；**不要为最终状态重跑旧builder**，它会覆盖远端计划并破坏已完成stat补证的planSHA绑定。原REMOTE_PATH_CHECK_PLAN及两份RESULT字节保持不变，A100连接未核局限不因本次UniDB终评而消失。\n'

    original_limit = evidence['limitations']
    evidence['historical_limitations'] = copy.deepcopy(original_limit)
    evidence['historical_quality_review'] = copy.deepcopy(evidence['quality_review'])
    evidence['historical_generation_script'] = copy.deepcopy(evidence['generation_script'])
    evidence['historical_generation_script']['preserved_copy'] = str(SNAPSHOT / 'build_asset_register.py')
    evidence.update(status='COMPLETE_LOCAL_ASSET_REGISTER_WITH_ACCEPTED_FINAL_UNIDB_SUPPLEMENT',
        prepared_UTC=now, assets=assets, asset_count=80, source_count=len(sources),
        generation_script={'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes()),
                           'behavior': 'Incremental snapshot-based stdlib localmetadata and checkpointstat only; no oldbuilder, remoteplan/result mutation, payload read or model/device call'},
        final_unidb_update={'status': 'PASS_FINAL_NATIVE_RECEIPTS_BOUND_LOCAL_STAT_ONLY', 'captured_UTC': now,
            'identities': receipt['identities'], 'adapters': {name: {k: sources[name][k] for k in ['path', 'sha256', 'bytes']} for name in [
                'UNIDB_FINAL_DELIVERY_RECEIPT', 'UNIDB_FINAL_OWNED_EXIT', 'UNIDB_FINAL_UNIQUE_BACKUP', 'UNIDB_FINAL_ADAPTER_VALIDATION']},
            'unique_checkpoint_local_stats': final_stats, 'unique_file_count': 3, 'logical_role_count': 4,
            'total_unique_bytes': backup['total_unique_bytes'], 'identity_count': exit_audit['identity_count'],
            'all_owned_exited': True, 'external_signals_sent': 0, 'no_new_payload_reads': True,
            'parent_experiment_acceptance_only_not_whole_handoff_delivered': True},
        limitations=['Originalhistorical asset and environment gaps remain; originalmetadata provenance is preserved in snapshot and historical_limitations.',
                     'Final UniDB fullstate/predictionCPU/visual/backup/exit acceptance is reused from accepted SHA-bound receipts. Thisrefresh only smallmetadata SHA and exactlocal file stat.',
                     'Remote stat plan/results unchanged; their A100connection failure remains unresolved. No fresh remote presence, environment execution or quota acceptance implied.',
                     'Wholeproject finaldocument/render/package and actual user-facing handoff are separate root acceptance steps.'],
        quality_review={'status': 'PASS_FINAL_INCREMENTAL_METADATA_JOIN', 'asset_rows': 80, 'historical_rows_retained': 77,
            'historical_rows_unchanged': 74, 'historical_ids_with_explicit_supplement': sorted(changed_old_ids),
            'source_count': len(sources), 'three_unique_local_checkpoint_stats_match': True,
            'four_roles_joined_to_CPU_state_and_EMA_full5000_refs': True,
            'whole_campaign_exit_identity_count': exit_audit['identity_count'],
            'protected_remote_plan_results_and_old_builder_unchanged': True,
            'no_duplicate_asset_ids': True, 'no_payload_model_optimizer_device_or_network_calls': True})
    csv_out = io.StringIO(newline='')
    writer = csv.DictWriter(csv_out, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    csv_bytes = csv_out.getvalue().encode()
    guide_bytes = guide.encode()
    evidence['outputs']['ASSET_REGISTER.csv'] = {'path': str(HERE / 'ASSET_REGISTER.csv'), 'sha256': digest(csv_bytes), 'bytes': len(csv_bytes)}
    evidence['outputs']['OPERATIONS_GUIDE_ZH.md'] = {'path': str(HERE / 'OPERATIONS_GUIDE_ZH.md'), 'sha256': digest(guide_bytes), 'bytes': len(guide_bytes)}
    for name in PROTECTED:
        assert digest((HERE / name).read_bytes()) == protected_before[name], name
    result = {'status': 'PASS_CHECK_ONLY' if args.check_only else 'PASS_FINAL_INCREMENTAL_ASSET_REFRESH',
              'asset_count': len(rows), 'source_count': len(sources), 'protected_SHA': protected_before,
              'final_unique_files': 3, 'final_roles': 4, 'checkpoint_payload_reads': 0, 'network_calls': 0,
              'generation_script_sha256': digest(Path(__file__).read_bytes())}
    if not args.check_only:
        (HERE / 'ASSET_REGISTER.csv').write_bytes(csv_bytes)
        (HERE / 'OPERATIONS_GUIDE_ZH.md').write_bytes(guide_bytes)
        (HERE / 'asset_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
        result['outputs'] = {name: {'sha256': digest((HERE / name).read_bytes()), 'bytes': (HERE / name).stat().st_size}
                             for name in ['ASSET_REGISTER.csv', 'OPERATIONS_GUIDE_ZH.md', 'asset_evidence.json']}
        (HERE / 'FINAL_UNIDB_ASSET_REFRESH.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

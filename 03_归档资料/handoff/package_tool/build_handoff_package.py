#!/usr/bin/env python3
"""Small, local, portable handoff package builder. No network or model imports.

--plan and --dry-run only write this new preparation directory. --final creates
delivery only after explicit final receipt adapters AND native evidence pass.
Never loads, copies or rehashes checkpoint/data/prediction payloads.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import urllib.parse
import zipfile

HERE = Path(__file__).resolve().parent
PREP = HERE.parent
HANDOFF = PREP.parent
PROJECT = HANDOFF.parent
CAMPAIGN = PROJECT / 'unidb_dual_20261104_until00'
DEADLINE = 1793721600
MAX_FILE = 32 << 20
MAX_TOTAL = 128 << 20
MAX_FILES = 2000
TEXT = {'.md', '.json', '.csv', '.txt', '.yaml', '.yml', '.py', '.sh'}
EXPLICIT_FINAL = {'.docx', '.pdf', '.png', '.jpg', '.jpeg'}
PAYLOAD = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npy', '.npz', '.pkl', '.pickle', '.mp4', '.avi', '.mov', '.tar', '.gz', '.zip'}
PAYLOAD_DIRS = {'checkpoints', 'predictions', 'prediction', 'arrays', 'raw_xi'}
PRIVATE_DIRS = {'.git', '.ssh', '.aws', '.codex', '.agents', 'node_modules', 'site-packages', '__pycache__'}
SLOTS = {
    'experiment_full_delivery_receipt': 'unidb_midnight_full_delivery_v1',
    'all_owned_process_exit_audit': 'all_campaign_owned_process_exit_v1',
    'unique_checkpoint_backup_audit': 'actual_final_unique_checkpoint_backup_v1',
    'handoff_final_document_visual_qa': 'handoff_final_document_visual_qa_v1',
}
ORIGINAL_PNGS = {'ORIGINAL_TRAIN_GLOBAL.png', 'ORIGINAL_VAL_4METRICS_MONITOR100.png',
                 'ORIGINAL_VAL_4METRICS_FULL1000.png'} | {
                     'ORIGINAL_FINAL_BEST_LATEST_G' + g + '.png' for g in ['1', '2', '3', '4']}
NATIVE_SMALL = ['report/REPORT_STATUS.json', 'report/FIGURE_MANIFEST.json', 'report/VISUAL_QA.json',
                'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json', 'evidence/CONTROLLER_EXIT.json',
                'evidence/CPU_POSTPROCESS_ACCEPTANCE.json', 'evidence/CPU_FINALIZER_EXIT.json',
                'evidence/FINAL_OUTPUTS_CPU_AUDIT.json'] + [
                    'evidence/STATE_AUDIT_' + branch + '_' + role + '.json'
                    for branch in ['original', 'improved'] for role in ['FINAL', 'BEST']]
LINK = re.compile(r'(!?)\[([^\]]*)\]\((<[^>]*>|[^)\n]*)\)')
CREDENTIAL = re.compile(r'''(?ix)(?:\b(?:password|passwd|client[_-]?secret)\b\s*[:=]\s*["']([^"'\r\n]+)["']|\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token)\b\s*[:=]\s*["']([A-Za-z0-9_./+=-]{16,})["']|\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{20,})''')
REGISTERS = ['VERSION_REGISTER.csv', 'EXPERIMENT_REGISTER.csv', 'DATA_REGISTER.csv', 'CODE_ENV_REGISTER.csv', 'WEIGHT_REGISTER.csv', 'ASSET_REGISTER.csv', 'EVIDENCE_REGISTER.csv']


class Refused(ValueError):
    pass


def need(value, message):
    if not value:
        raise Refused(message)


def jload(path):
    p = Path(path)
    need(p.is_file(), '缺少JSON证据：' + str(p))
    need(p.stat().st_size <= MAX_FILE, 'JSON证据超出小文件限制：' + str(p))
    return json.loads(p.read_text())


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    p = Path(path)
    need(p.stat().st_size <= MAX_FILE, '拒绝读取大文件内容：' + str(p))
    return digest_bytes(p.read_bytes())


def project_path(raw, *, project=PROJECT, must_exist=True):
    """Reject traversal and every symlink component, including in-tree links."""
    p = Path(raw)
    need(not any(ord(x) < 32 or ord(x) == 127 for x in str(raw)), '路径含控制字符')
    need('..' not in p.parts, '拒绝路径穿越：' + str(raw))
    p = p if p.is_absolute() else project / p
    try:
        rel = p.relative_to(project)
    except ValueError:
        raise Refused('路径不在项目根：' + str(raw))
    need(not (set(rel.parts) & PRIVATE_DIRS) and p.name not in {'.env', 'auth.json', 'credentials.json', 'id_rsa', 'id_ed25519'},
         '拒绝凭据/私有运行目录：' + str(raw))
    cursor = project
    for part in rel.parts:
        cursor = cursor / part
        need(not cursor.is_symlink(), '拒绝符号链接资产：' + str(cursor))
    need(p.resolve().is_relative_to(project.resolve()), '解析路径越出项目根')
    if must_exist:
        need(p.is_file(), '本地小文件不存在：' + str(p))
    return p


def destination(raw):
    need(isinstance(raw, str) and raw and '\\' not in raw
         and not any(ord(x) < 32 or ord(x) == 127 for x in raw),
         '非法包内路径')
    p = PurePosixPath(raw)
    need(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts,
         '拒绝包内绝对/穿越路径：' + raw)
    need(str(p) == raw and raw not in {'README.md', 'SOURCE_MAP.csv', 'MANIFEST.json', 'NOT_INCLUDED.md'},
         '包内路径冲突或非规范：' + raw)
    need(p.parts[0] != 'original_text', 'original_text命名空间仅供工具保留原文，不能由来源指定')
    return raw


def copy_policy(path, explicit=False):
    if path.suffix.lower() in PAYLOAD or set(x.lower() for x in path.parts) & PAYLOAD_DIRS:
        return 'PAYLOAD_OR_BULK_PREDICTION_EXCLUDED'
    if path.suffix.lower() not in TEXT | (EXPLICIT_FINAL if explicit else set()):
        return 'UNSUPPORTED_SMALL_FILE_TYPE'
    if path.stat().st_size > MAX_FILE:
        return 'OVER_32MiB_SMALL_FILE_LIMIT'
    return None


def credential_lines(path):
    if path.suffix.lower() not in TEXT:
        return []
    # Never return or log matched values. This is a conservative screening gate,
    # not a claim of exhaustive secret detection.
    found = []
    for i, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        match = CREDENTIAL.search(line)
        if not match:
            continue
        value = next((x for x in match.groups() if x is not None), '')
        if value.startswith(('<', '${')) or value.upper().startswith(('YOUR_', 'REPLACE_', 'PLACEHOLDER')):
            continue
        found.append(i)
    return found


def local_from_remote(raw):
    """Only translate the known user's campaign roots; never read arbitrary /data."""
    p = Path(raw)
    prefix = Path('/data2/yanglin')
    if p.is_absolute() and p.is_relative_to(prefix):
        p = PROJECT / p.relative_to(prefix)
    return project_path(p)


def config_file(path=None):
    p = project_path(path or HERE / 'PACKAGE_FINAL_INPUTS.json')
    need(p.suffix.lower() == '.json' and copy_policy(p) is None,
         '打包输入配置只允许小JSON，拒绝读取payload或批量目录')
    return p, jload(p)


def bound_json(reference):
    need(isinstance(reference, dict), '证据逻辑槽必须为path/sha256对象')
    p = project_path(reference.get('path', ''))
    need(p.suffix.lower() == '.json' and copy_policy(p) is None,
         '证据逻辑槽只允许小JSON元数据；拒绝payload/批量目录：' + str(p))
    need(re.fullmatch(r'[0-9a-f]{64}', reference.get('sha256', '')),
         '证据逻辑槽缺少完整SHA：' + str(p))
    need(sha(p) == reference['sha256'], '逻辑槽证据SHA不符：' + str(p))
    return p, jload(p)


def file_binding(item, *, copy_allowed=True):
    p = project_path(item.get('path', ''))
    need(copy_policy(p, explicit=True) is None if copy_allowed else True,
         '不允许作为便携附件：' + str(p))
    need(p.stat().st_size == item.get('bytes') and sha(p) == item.get('sha256'),
         '小文件size/SHA绑定失败：' + str(p))
    return p


def require_adapter(name, cfg):
    p, q = bound_json(cfg.get('receipts', {}).get(name, {}))
    need(q.get('schema') == SLOTS[name] and q.get('scope') == SLOTS[name]
         and q.get('status') == 'PASS' and q.get('final') is True,
         name + '不是指定scope/schema的真实FINAL PASS；DRAFT或通用status不可替代')
    need(q.get('evidence_bindings'), name + '缺少原始验收证据绑定')
    for r in q['evidence_bindings']:
        bound_json(r)
    return p, q


def checked_document_qa(docqa):
    need(docqa.get('reviewer') == 'root' and docqa.get('draft') is False
         and docqa.get('page_count', 0) > 0
         and docqa.get('pages_reviewed') == list(range(1, docqa['page_count'] + 1)), '主文档FINAL逐页root目视不完整')
    artifact_records = docqa.get('artifacts', [])
    artifacts = [file_binding(q) for q in artifact_records]
    need(len(artifacts) == len(set(artifacts)), '最终文档QA重复artifact path，不能重复page1冒充全部页')
    page_indices = [q.get('page_index') for q, path in zip(artifact_records, artifacts) if path.suffix == '.png']
    need(all(type(i) is int for i in page_indices) and sorted(page_indices) == list(range(1, docqa['page_count'] + 1)),
         '最终PNG必须有唯一明确page_index=1..page_count')
    need(any(p.suffix == '.docx' for p in artifacts) and any(p.suffix == '.pdf' for p in artifacts)
         and sum(p.suffix == '.png' for p in artifacts) == docqa['page_count']
         and not any('draft' in p.name.lower() for p in artifacts), '主文档必须绑定FINAL DOCX/PDF/全部页PNG；拒绝DRAFT_QA')
    markdown = file_binding(docqa.get('source_markdown', {}))
    need(markdown.suffix == '.md' and 'draft' not in markdown.name.lower(), '最终主文稿SHA未绑定或仍使用DRAFT主文稿')
    need(not any(x in markdown.read_text() for x in ['本稿尚不能作为最终交付', '最终评价和CPU验收仍在进行', '以上最终接收项目尚未全部完成'])
         and '准备稿' not in '\n'.join(markdown.read_text().splitlines()[:5]), '最终主稿仍保留本次准备态/终评未完文字')
    return artifacts, markdown


def final_content_refresh(cfg, identities):
    path, q = bound_json(cfg.get('final_handoff_refresh', {}))
    need(q.get('schema') == 'handoff_final_content_refresh_v1'
         and q.get('scope') == 'handoff_final_content_refresh_v1'
         and q.get('status') == 'PASS' and q.get('final') is True
         and q.get('identities') == identities, '最终内容刷新未绑定真实四角色身份')
    history = file_binding(q.get('current_unidb_history', {}))
    evidence = file_binding(q.get('current_unidb_evidence', {}))
    state = file_binding(q.get('current_handoff_state', {}))
    e, s = jload(evidence), jload(state)
    need(e.get('current_experiment_final_acceptance') is True and e.get('gaps') == []
         and e.get('document_sha256') == sha(history), '当前UniDB evidence仍未最终验收/有待补项或文稿SHA未刷新')
    need(s.get('status') in {'COMPLETE', 'FINAL_EXPERIMENT_DELIVERY_COMPLETE'}
         and s.get('handover_not_complete') is False
         and s.get('campaign') == CAMPAIGN.name, '当前handoff状态仍是pending，不能将旧准备快照打包为current')
    stale = ['最终质量、CPU验收与交付尚待补齐', '当前最终评价进行中', '最终评价与验收未完成',
             '本次最终收尾中的实验', '完整checkpoint张量和最终保存预测的CPU验收仍待']
    need(not any(x in history.read_text() for x in stale), '当前UniDB叙述仍含本次终评未完的准备态段落')
    need(set(q.get('registers', {})) == set(REGISTERS), '最终内容刷新缺少七注册表的明确绑定')
    registers = {name: file_binding(value) for name, value in q['registers'].items()}
    with registers['EXPERIMENT_REGISTER.csv'].open() as f:
        current = [r for r in csv.DictReader(f)
                   if r.get('version_id') == 'version:UniDB' and r.get('status_scope', '').startswith('CURRENT_')]
    need(current and all(r.get('current_final_acceptance') == 'True'
                         and 'PENDING' not in r.get('status_scope', '').upper()
                         and 'PENDING' not in r.get('scientific_or_execution_status', '').upper() for r in current),
         '最终EXPERIMENT_REGISTER当前UniDB行仍为False/PENDING；旧快照必须标历史')
    consolidation = file_binding(q.get('consolidation_acceptance', {}))
    c = jload(consolidation)
    need(c.get('status') in {'PASS', 'PASS_WITH_EXPLICIT_UNFILLED_FIELDS'}
         and c.get('current_UniDB_final_acceptance') is True and c.get('errors') == [], '整合验收仍未包含当前UniDB最终结果')
    need((set(REGISTERS) - {'ASSET_REGISTER.csv'}) <= set(c.get('outputs', {})), '整合验收缺少六张原生CSV输出绑定')
    for name, r in c.get('outputs', {}).items():
        if name in registers:
            need(r.get('sha256') == sha(registers[name]) and r.get('bytes') == registers[name].stat().st_size,
                 '最终注册表与整合验收size/SHA不符：' + name)
    return {'path': str(path), 'sha256': sha(path), 'receipt': q,
            'required_files': [str(path), str(history), str(evidence), str(state), str(consolidation)]
            + [str(p) for p in registers.values()]}


def final_readiness(cfg):
    """Read local receipt mirrors only. No large SHA/proc/SSH/checkpoint loads."""
    errors = []
    proof = {'required_packaging_files': []}
    if cfg.get('status') != 'READY_FOR_PACKAGE':
        errors.append('PACKAGE_FINAL_INPUTS仍为准备态，尚未绑定真实最终收据')
    for slot in SLOTS:
        try:
            p, q = require_adapter(slot, cfg)
            proof[slot] = {'path': str(p), 'sha256': sha(p), 'receipt': q}
            proof['required_packaging_files'] += [str(p)] + [str(project_path(x['path'])) for x in q['evidence_bindings']]
        except (Refused, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    try:
        report = jload(CAMPAIGN / 'report/REPORT_STATUS.json')
        need(report.get('status') == 'FINAL_READY' and report.get('final_requested') is True
             and report.get('final_requirements_pass') is True and report.get('visual_qa_pass') is True
             and report.get('pending') == [], '实验REPORT_STATUS尚非全部原生门槛FINAL_READY')
        for k in ['new_restoration_network_calls', 'metric_network_calls', 'optimizer_updates_by_reporter']:
            need(report.get(k) == 0, 'reporter零新调用字段不符：' + k)
        complete = jload(CAMPAIGN / 'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json')
        need(complete.get('status') == 'PASS' and set(complete.get('branches', {})) == {'original', 'improved'}
             and all(q.get('status') == 'PASS' for q in complete['branches'].values()), '两路线完整模型评价尚未PASS')
        cx = jload(CAMPAIGN / 'evidence/CONTROLLER_EXIT.json')
        need(cx.get('all_subprocesses_exited') is True and cx['unix'] >= complete['unix'], '控制器/子进程退出未验收')
        cpu = jload(CAMPAIGN / 'evidence/CPU_POSTPROCESS_ACCEPTANCE.json')
        fx = jload(CAMPAIGN / 'evidence/CPU_FINALIZER_EXIT.json')
        need(cpu.get('status') == 'PASS' and fx.get('actor') == cpu.get('actor')
             and fx.get('status') == 'CPU_AUDIT_AND_REPORT_PASS_VISUAL_AND_LOCAL_DELIVERY_PENDING'
             and fx['unix'] >= cpu['completed_unix'], 'CPU收尾器尚未验收并退出')
        need(all(cpu.get(k) == 0 for k in ['restoration_model_calls', 'new_optimizer_updates', 'NPU_jobs']), 'CPU收尾零新恢复调用不符')
        aud = jload(CAMPAIGN / 'evidence/FINAL_OUTPUTS_CPU_AUDIT.json')
        need(aud.get('status') == 'PASS' and aud.get('pure_CPU') is True
             and aud.get('restoration_model_constructed') is False
             and all(aud.get(k) == 0 for k in ['restoration_network_calls', 'NPU_calls', 'optimizer_steps']), '全部保存预测CPU审计未PASS')
        refs = cpu.get('refs', [])
        need({(q['branch'], q['role']) for q in refs} == {(b, r) for b in ['original', 'improved'] for r in ['best', 'latest']}
             and len(refs) == 4, 'CPU收尾未绑定两线best/latest四逻辑角色')
        identities = {}
        for ref in refs:
            key = ref['branch'] + ':' + ref['role']
            state_path = local_from_remote(ref['checkpoint_CPU_audit'])
            summary_path = local_from_remote(ref['summary'])
            proof['required_packaging_files'] += [str(state_path), str(summary_path)]
            need(sha(state_path) == ref['checkpoint_CPU_audit_sha256'] and sha(summary_path) == ref['summary_sha256'], '最终state/SUMMARY小文件SHA不符：' + key)
            state, summary = jload(state_path), jload(summary_path)
            cp = state.get('checkpoint', {})
            need(state.get('status') == 'PASS' and cp.get('status') == 'PASS'
                 and cp.get('sha256') == ref['checkpoint_sha256'] and cp.get('update') == ref['update']
                 and cp.get('branch') == ref['branch']
                 and cp.get('EMA', {}).get('full_parameter_digest') == ref['checkpoint_EMA_full_digest']
                 == summary.get('parameter_digest'), '完整state与EMA/SUMMARY身份不符：' + key)
            need(state.get('source_sha256') == sha(CAMPAIGN / 'audit_state.py')
                 and all(state.get(k) == 0 for k in ['model_objects', 'optimizer_objects', 'NPU_calls']), '完整state审计来源或只读契约不符：' + key)
            for weights in ['RAW', 'EMA']:
                need(cp.get(weights, {}).get('tensors') == 151 and cp[weights].get('parameters') == 137147523
                     and cp[weights].get('all_FP32_finite') is True, '完整151张量FP32状态审计缺失：' + key)
            need(cp.get('Adam', {}).get('states') == 151 and cp['Adam'].get('step_all') == ref['update']
                 and cp['Adam'].get('all_FP32_finite') is True and cp.get('EMA_step') == ref['update']
                 and cp.get('cursor') == 96 * ref['update']
                 and cp.get('RNG', {}).get('banks') == (8 if ref['branch'] == 'original' else 5)
                 and cp['RNG'].get('structure_complete') is True, '完整Adam/EMA/RNG/cursor不符：' + key)
            need(summary.get('status') == 'PASS' and summary.get('full') is True
                 and summary.get('branch') == ref['branch'] and summary.get('update') == ref['update']
                 and summary.get('weights') == 'EMA' and summary.get('count') == 5000
                 and summary.get('per_g') == 1000 and summary.get('sources_per_g') == 250
                 and set(summary.get('by_g', {})) == {'0.5', '1', '2', '3', '4'}
                 and summary.get('nfe_per_image') == 100 and summary.get('sampler') == 'native_reverse_sde'
                 and summary.get('image_clip_crop_resize') is False and summary.get('test_used') is False,
                 '最终SUMMARY并非完整原生五档5000：' + key)
            need(aud.get('audited_summaries', {}).get(ref['summary']) == ref['summary_sha256'], 'CPU审计未绑定实际最终SUMMARY：' + key)
            results = [r for r in aud.get('validations', []) if r.get('summary_path') == ref['summary']]
            need(len(results) == 1, 'CPU完整预测审计结果缺失/重复：' + key)
            result = results[0]
            need(result.get('status') == 'PASS' and result.get('images') == 5000
                 and result.get('CPU_LPIPS_images') == 5000 and result.get('all_prediction_input_clean_SHA_verified') is True
                 and result.get('failure_counts') == {}, 'CPU全5000 SHA/四指标覆盖未通过：' + key)
            identities[key] = {'update': ref['update'], 'sha256': ref['checkpoint_sha256']}
        fm = jload(CAMPAIGN / 'report/FIGURE_MANIFEST.json')
        qa = jload(CAMPAIGN / 'report/VISUAL_QA.json')
        need(qa.get('status') == 'PASS', '最终实验图目视QA缺失')
        for name, expected in fm.get('figures', {}).items():
            if name.endswith('.png'):
                need(qa.get('figures', {}).get(name) == expected and sha(CAMPAIGN / 'figures' / name) == expected, '最终实验PNG/目视SHA不符：' + name)
                proof['required_packaging_files'].append(str(CAMPAIGN / 'figures' / name))
        need(sum(n.endswith('.png') for n in fm.get('figures', {})) >= 3, '缺最终三张实验图')
        proof['native_identities'] = identities
        proof['native_CPU_exit_unix'] = fx['unix']
        proof['required_packaging_files'] += [str(CAMPAIGN / p) for p in NATIVE_SMALL]
        original = CAMPAIGN / 'delivery/original_route'
        original_receipt = jload(original / 'ORIGINAL_ATTACHMENT_RECEIPT.json')
        original_qa = jload(original / 'VISUAL_QA.json')
        need(original_receipt.get('status') == 'ARTIFACTS_GENERATED_ROOT_VISUAL_QA_PENDING'
             and original_receipt.get('final_update') == identities['original:latest']['update']
             and original_receipt.get('full_best_update') == identities['original:best']['update']
             and all(original_receipt.get(k) == 0 for k in ['new_restoration_model_calls', 'optimizer_calls', 'NPU_calls']),
             '仅原路线最终附件身份或零新模型调用不符')
        need(original_qa.get('status') == 'PASS' and set(original_qa.get('figures', {})) == ORIGINAL_PNGS,
             '仅原路线最终7张新PNG缺root逐图目视SHA')
        for name in ORIGINAL_PNGS:
            need(original_receipt['outputs'].get(name) == original_qa['figures'][name] == sha(original / name),
                 '仅原路线附件PNG与实际最终QA身份不符：' + name)
            proof['required_packaging_files'].append(str(original / name))
        proof['required_packaging_files'] += [str(original / p) for p in ['ORIGINAL_ATTACHMENT_RECEIPT.json', 'VISUAL_QA.json']]
    except (OSError, Refused, ValueError, KeyError, TypeError) as exc:
        errors.append('实验原生证据：' + str(exc))
    try:
        need(all(k in proof for k in SLOTS) and 'native_identities' in proof, '先完成实验原生和四类逻辑槽证据')
        delivery = proof['experiment_full_delivery_receipt']['receipt']
        need(delivery.get('campaign') == CAMPAIGN.name and delivery.get('deadline_unix') == DEADLINE
             and delivery.get('identities') == proof['native_identities'], '实验最终交付adapter身份未精确对应真实角色')
        exitq = proof['all_owned_process_exit_audit']['receipt']
        need(exitq.get('all_owned_exited') is True and exitq.get('live_owned_actors') == []
             and exitq.get('external_signals_sent') == 0 and exitq.get('checked_unix', 0) >= proof['native_CPU_exit_unix']
             and set(exitq.get('campaign_roots', [])) == {'unidb_dual_20261103_until_noon', 'unidb_dual_20261103_until18', CAMPAIGN.name}, '全部历史/当前自有PGID/loader/CPU退出审计不完整')
        backups = proof['unique_checkpoint_backup_audit']['receipt']
        seen = {}
        for item in backups.get('files', []):
            p = project_path(item['local_path'])
            need(p.suffix.lower() in {'.pt', '.pth'} and p.stat().st_size == item['bytes']
                 and p.stat().st_mtime_ns == item['local_mtime_ns'] and item.get('status') == 'PASS_BYTE_IDENTICAL_FINAL'
                 and item['sha256'] == item['source_sha256'] and item.get('SHA_verified_unix', 0) > 0
                 and item.get('final_role_binding_verified_unix', 0) > DEADLINE, '真正最终权重备份字节验收缺失或文件已变')
            for role in item['roles']:
                key = item['branch'] + ':' + role
                need(key not in seen, '重复最终备份角色')
                seen[key] = {'update': item['update'], 'sha256': item['sha256']}
        need(seen == proof['native_identities'], '备份不是实际最终unique best/latest；旧候选不能替代')
        docqa = proof['handoff_final_document_visual_qa']['receipt']
        artifacts, markdown = checked_document_qa(docqa)
        proof['final_document_paths'] = [str(p) for p in artifacts]
        proof['required_packaging_files'] += [str(p) for p in artifacts] + [str(markdown)]
    except (OSError, Refused, ValueError, KeyError, TypeError) as exc:
        errors.append('最终交接/退出/备份/文档QA：' + str(exc))
    try:
        need('native_identities' in proof, '尚无真实最终身份用于内容刷新核对')
        refresh = final_content_refresh(cfg, proof['native_identities'])
        proof['final_handoff_refresh'] = refresh
        proof['required_packaging_files'] += refresh['required_files']
    except (OSError, Refused, ValueError, KeyError, TypeError) as exc:
        errors.append('最终叙述/注册表刷新：' + str(exc))
    return {'status': 'READY' if not errors else 'NOT_READY', 'errors': errors, 'proof': proof}


def collect_plan(cfg):
    try:
        refresh_path, refresh = bound_json(cfg.get('final_handoff_refresh', {}))
    except (Refused, ValueError, KeyError, TypeError):
        refresh_path, refresh = None, {}
    registry = project_path(refresh.get('registers', {}).get('EVIDENCE_REGISTER.csv', {}).get('path', PREP / 'EVIDENCE_REGISTER.csv'))
    entries, omitted, errors = {}, [], []
    preferred = {}
    try:
        _, docqa = bound_json(cfg.get('receipts', {}).get('handoff_final_document_visual_qa', {}))
        for item in docqa.get('artifacts', []) + ([docqa['source_markdown']] if docqa.get('source_markdown') else []):
            p = project_path(item['path'])
            preferred[str(p)] = ('document_pages/' if p.suffix == '.png' else 'documents/') + p.name
    except (Refused, OSError, ValueError, KeyError, TypeError):
        pass
    for item in cfg.get('extra_small_assets', []) + cfg.get('final_documents', []):
        try:
            path = project_path(item['path'])
            want = destination(item['destination'])
            need(str(path) not in preferred or preferred[str(path)] == want, '显式附件与FINAL QA的稳定包内路径冲突')
            preferred[str(path)] = want
        except (Refused, OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))

    def add(raw, dest=None, expected=None, explicit=False, required=True):
        try:
            p = project_path(raw)
            reason = copy_policy(p, explicit)
            if reason:
                omitted.append({'original_path': str(p), 'recorded_sha256': expected or '', 'reason': reason, 'bytes': p.stat().st_size})
                return
            h = sha(p)
            need(not expected or h == expected, '来源SHA已变化，先刷新注册表：' + str(p))
            lines = credential_lines(p)
            need(not lines, '疑似凭据，仅报告位置不输出值：' + str(p) + ' lines=' + str(lines) + ' count=' + str(len(lines)))
            rel = str(p.relative_to(PROJECT))
            d = destination(dest or preferred.get(str(p)) or 'evidence/' + rel)
            need(Path(d).suffix.lower() == p.suffix.lower(), '不允许改后缀伪装来源文件类型')
            if str(p) in entries:
                need(entries[str(p)]['destination'] == d or dest is None, '同来源被指定两个包内路径')
                return
            need(not any(x['destination'] == d for x in entries.values()), '两个来源冲突到同包内路径')
            entries[str(p)] = {'original_path': str(p), 'project_relative_path': rel, 'destination': d, 'source_sha256': h, 'bytes': p.stat().st_size}
        except (OSError, Refused, ValueError) as exc:
            if required:
                errors.append(str(exc))
            else:
                omitted.append({'original_path': str(raw), 'reason': str(exc)})

    histories = ['legacy_v1_v3.md', 'v4_history.md', 'v5_history.md', 'v6_history.md', 'unidb_history.md', 'extra_branches.md', 'OPERATIONS_GUIDE_ZH.md']
    for name in histories:
        override = refresh.get('current_unidb_history', {}) if name == 'unidb_history.md' else {}
        add(override.get('path', PREP / name), 'history/' + name, override.get('sha256'))
    for name in REGISTERS:
        override = refresh.get('registers', {}).get(name, {})
        add(override.get('path', PREP / name), 'registers/' + name, override.get('sha256'))
    for name in ['legacy_evidence.json', 'v4_evidence.json', 'v5_evidence.json', 'v6_evidence.json', 'unidb_evidence.json', 'extra_evidence.json', 'asset_evidence.json', 'CONSOLIDATION_ACCEPTANCE.json', 'UNIDB_PENDING_FINAL.json']:
        override = refresh.get('current_unidb_evidence', {}) if name == 'unidb_evidence.json' else refresh.get('consolidation_acceptance', {}) if name == 'CONSOLIDATION_ACCEPTANCE.json' else {}
        prefix = 'historical_preparation/' if name == 'UNIDB_PENDING_FINAL.json' else 'provenance/'
        add(override.get('path', PREP / name), prefix + name, override.get('sha256'))
    if refresh_path:
        add(refresh_path)
        if refresh.get('current_handoff_state'):
            v = refresh['current_handoff_state'];add(v['path'], 'provenance/UNIDB_FINAL_HANDOFF_STATE.json', v.get('sha256'))
    with registry.open() as f:
        for row in csv.DictReader(f):
            # Existing copies above are deliberately assigned friendly stable paths.
            p = Path(row['canonical_path'])
            dest = entries.get(str(p), {}).get('destination')
            replaced = {'unidb_history.md': refresh.get('current_unidb_history', {}),
                        'unidb_evidence.json': refresh.get('current_unidb_evidence', {}),
                        'CONSOLIDATION_ACCEPTANCE.json': refresh.get('consolidation_acceptance', {})}
            replaced.update({name: ref for name, ref in refresh.get('registers', {}).items()})
            if (p.parent == PREP and p.name in replaced and replaced[p.name].get('path')
                    and project_path(replaced[p.name]['path']) != p):
                dest = 'historical_preparation/' + p.name
            add(p, dest=dest, expected=row.get('actual_sha256'))
    # Known local small mirrors are candidates, never completion declarations.
    for rel in NATIVE_SMALL:
        path = CAMPAIGN / rel
        if path.is_file():
            add(path)
    cpu_path = CAMPAIGN / 'evidence/CPU_POSTPROCESS_ACCEPTANCE.json'
    if cpu_path.is_file():
        for ref in jload(cpu_path).get('refs', []):
            for key in ['checkpoint_CPU_audit', 'summary']:
                try:
                    add(local_from_remote(ref[key]))
                except (Refused, KeyError) as exc:
                    errors.append('实际final ref小文件未同步：' + str(exc))
    for base in [CAMPAIGN / 'report', CAMPAIGN / 'delivery/original_route']:
        if base.is_dir():
            for path in sorted(base.iterdir()):
                if path.is_file() and path.suffix.lower() in {'.md', '.csv', '.json', '.pdf'}:
                    add(path, explicit=True)
    for name in ['CAMPAIGN_DYNAMICS.png', 'FIXED_G4_BEST.png', 'FIXED_G4_LATEST.png']:
        path = CAMPAIGN / 'figures' / name
        if path.is_file():
            add(path, explicit=True)
    for name in sorted(ORIGINAL_PNGS) + ['ORIGINAL_ATTACHMENT_RECEIPT.json', 'VISUAL_QA.json']:
        path = CAMPAIGN / 'delivery/original_route' / name
        if path.is_file():
            add(path, explicit=True)
    for slot in SLOTS:
        try:
            path, q = bound_json(cfg.get('receipts', {}).get(slot, {}))
            add(path)
            for item in q.get('evidence_bindings', []):
                add(item['path'], expected=item.get('sha256'))
            if slot == 'handoff_final_document_visual_qa':
                for item in q.get('artifacts', []) + ([q['source_markdown']] if q.get('source_markdown') else []):
                    path = Path(item['path'])
                    prefix = 'document_pages/' if path.suffix == '.png' else 'documents/'
                    add(item['path'], prefix + path.name, item.get('sha256'), explicit=True)
        except (Refused, KeyError, TypeError):
            # Missing adapters are clearly reported by final_readiness below.
            pass
    for item in cfg.get('extra_small_assets', []):
        add(item['path'], item['destination'], item.get('sha256'), explicit=True)
    for item in cfg.get('final_documents', []):
        add(item['path'], item['destination'], item.get('sha256'), explicit=True)
    total = sum(x['bytes'] for x in entries.values())
    if total > MAX_TOTAL or len(entries) > MAX_FILES:
        errors.append('便携包超过128MiB/2000文件上限')
    return {'schema': 'handoff_package_plan_v1', 'status': 'PLAN_ONLY_NOT_DELIVERY',
            'entries': sorted(entries.values(), key=lambda x: x['destination']), 'omitted': omitted,
            'errors': errors, 'input_bytes': total, 'file_count': len(entries),
            'registry_sha256': sha(registry), 'historical_pending_snapshot': jload(PREP / 'UNIDB_PENDING_FINAL.json'),
            'final_content_refresh_configured': refresh_path is not None,
            'limits': {'file_bytes': MAX_FILE, 'total_bytes': MAX_TOTAL, 'files': MAX_FILES},
            'payloads_included': False, 'network_model_optimizer_NPU_calls': 0}


def portable_markdown(text, source, dest, mapping, excluded):
    def replace(match):
        image, label, raw = match.groups()
        target = raw.strip('<>')
        parsed = urllib.parse.urlsplit(target)
        if parsed.scheme in {'http', 'https', 'mailto'} or target.startswith('#'):
            return match.group(0)
        pathpart = urllib.parse.unquote(parsed.path)
        pathpart = re.sub(r':\d+$', '', pathpart)
        p = Path(pathpart)
        p = p if p.is_absolute() else source.parent / p
        canonical = str(p.resolve())
        if canonical in mapping:
            relative = os.path.relpath(mapping[canonical], str(PurePosixPath(dest).parent)).replace(os.sep, '/')
            suffix = ('#' + parsed.fragment) if parsed.fragment else ''
            return image + '[' + label + '](<' + relative + suffix + '>)'
        key = digest_bytes(target.encode())[:20]
        excluded[key] = {'original_reference': target, 'from_source': str(source), 'reason': 'Not part of selected portable small-file scope; consult source map/original asset register'}
        relative = os.path.relpath('NOT_INCLUDED.md', str(PurePosixPath(dest).parent)).replace(os.sep, '/')
        return '[' + ('图像原位置：' if image else '') + label + '](<' + relative + '#ref-' + key + '>)'
    return LINK.sub(replace, text)


def check_links(stage):
    checked = 0
    for p in stage.rglob('*.md'):
        if p.relative_to(stage).parts[0] == 'original_text':
            # Exact historical bytes are preserved separately; portable views
            # above carry rewritten and validated local links.
            continue
        for _, _, raw in LINK.findall(p.read_text()):
            target = raw.strip('<>')
            parsed = urllib.parse.urlsplit(target)
            if parsed.scheme in {'http', 'https', 'mailto'} or target.startswith('#'):
                continue
            need(not parsed.scheme and not target.startswith('/'), '便携MD仍有绝对或私有协议链接：' + target)
            q = (p.parent / urllib.parse.unquote(parsed.path)).resolve()
            need(q.is_relative_to(stage.resolve()) and q.is_file(), '包内链接缺失/越界：' + target)
            checked += 1
    return checked


def make_package(plan, readiness):
    need(readiness['status'] == 'READY' and not plan['errors'], '最终门槛或文件计划尚未通过')
    included = {x['original_path'] for x in plan['entries']}
    need(set(readiness['proof']['required_packaging_files']) <= included,
         '计划未包含全部实际最终收据/4角色state与SUMMARY/最终文档或7张原路线图')
    out = project_path(HANDOFF / 'delivery/portable_handoff', must_exist=False)
    archive = project_path(out.parent / 'portable_handoff.zip', must_exist=False)
    need(not out.exists() and not archive.exists(), '已存在成品，禁止覆盖或重复打包')
    stage = Path(tempfile.mkdtemp(prefix='accepted_package_stage_', dir=HERE))
    mapping = {x['original_path']: x['destination'] for x in plan['entries']}
    excluded, rows = {}, []
    for item in plan['entries']:
        p = project_path(item['original_path'])
        raw = p.read_bytes()
        need(digest_bytes(raw) == item['source_sha256'] and len(raw) == item['bytes'], '复制期间来源发生变化：' + str(p))
        original_copy = ''
        if p.suffix.lower() == '.md':
            original_copy = 'original_text/' + item['project_relative_path']
            archival = stage / original_copy
            archival.parent.mkdir(parents=True, exist_ok=True)
            archival.write_bytes(raw)
            raw = portable_markdown(raw.decode(), p, item['destination'], mapping, excluded).encode()
        target = stage / item['destination']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        rows.append({**item, 'package_sha256': digest_bytes(raw), 'package_bytes': len(raw), 'markdown_links_rewritten': p.suffix.lower() == '.md', 'original_copy_destination': original_copy})
    note = '# 未包含的原始资产与引用\n\n本便携包不含任何大checkpoint、训练数据、raw ξ、完整预测、视频或完整环境。原路径和历史SHA只是索引，非已下载或当前远端存在证明。\n\n'
    for item in plan['omitted']:
        note += '- `' + item['original_path'] + '`：' + item['reason'] + '\n'
    for key, item in sorted(excluded.items()):
        note += '\n<a id="ref-' + key + '"></a>\n\n- 原引用：`' + item['original_reference'] + '`；来源：`' + item['from_source'] + '`。未作为便携附件复制。\n'
    (stage / 'NOT_INCLUDED.md').write_text(note)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader();writer.writerows(rows)
    (stage / 'SOURCE_MAP.csv').write_text(stream.getvalue())
    document_links = '\n'.join('- [' + PurePosixPath(x['destination']).name + '](<' + x['destination'] + '>)' for x in rows if x['destination'].startswith('documents/'))
    (stage / 'README.md').write_text('# 湍流项目便携交接资料\n\n这是最终验收后生成的小文件包。**不包含大权重、原始数据、raw ξ、全量预测或环境镜像。**\n\n[原位置与包内文件映射](SOURCE_MAP.csv) · [未包含资产](NOT_INCLUDED.md)\n\n' + document_links + '\n\n- [实验注册表](registers/EXPERIMENT_REGISTER.csv)\n- [数据注册表](registers/DATA_REGISTER.csv)\n- [权重注册表](registers/WEIGHT_REGISTER.csv)\n- [操作与复核说明](history/OPERATIONS_GUIDE_ZH.md)\n- [当前UniDB最终状态](provenance/UNIDB_FINAL_HANDOFF_STATE.json)\n\nhistorical_preparation只保留当时准备快照，内含pending不代表当前仍未完成；当前以最终状态和已刷新的history/register为准。CSV/JSON中的绝对路径保留用于溯源，不代表已移交payload；Markdown本地链接仅在可读副本改写，原始MD字节另存original_text并保留原SHA。原文中的旧链接仅作历史证据，不承诺portable。旧脚本仅存为证据，不自动执行。\n')
    link_count = check_links(stage)
    manifest = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage / 'MANIFEST.json').write_text(json.dumps({'schema': 'portable_handoff_manifest_v1', 'scope': 'Small artifacts only; manifest excludes its own bytes', 'files': manifest, 'source_count': len(rows), 'checked_local_markdown_links': link_count, 'payloads_included': False}, ensure_ascii=False, indent=2) + '\n')
    need(sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()) <= MAX_TOTAL, '转换后包超过128MiB')
    need(sum(p.is_file() for p in stage.rglob('*')) <= MAX_FILES, '含原文副本的最终文件数超过2000')
    for name, rec in manifest.items():
        p = stage / name
        need(p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256'], '包内回读SHA不符')
    staged_archive = stage.with_suffix('.zip')
    with zipfile.ZipFile(staged_archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                z.write(p, arcname='portable_handoff/' + str(p.relative_to(stage)))
    with zipfile.ZipFile(staged_archive) as z:
        need(z.testzip() is None, 'ZIP CRC验证失败')
    zip_digest = hashlib.sha256()
    with staged_archive.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            zip_digest.update(block)
    out.parent.mkdir(exist_ok=True)
    project_path(out, must_exist=False)
    project_path(archive, must_exist=False)
    stage.rename(out)
    staged_archive.rename(archive)
    return {'status': 'PORTABLE_SMALL_PACKAGE_CREATED_NOT_EXTERNAL_TRANSMISSION', 'directory': str(out), 'zip': str(archive), 'zip_sha256': zip_digest.hexdigest(), 'zip_bytes': archive.stat().st_size, 'manifest_sha256': sha(out / 'MANIFEST.json'), 'checked_links': link_count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    for name in ['plan', 'dry-run', 'final']:
        group.add_argument('--' + name, action='store_true')
    p.add_argument('--inputs', help='Explicit final receipt adapter config, inside project')
    args = p.parse_args()
    delivery_before = (HANDOFF / 'delivery/portable_handoff').exists()
    try:
        path, cfg = config_file(args.inputs)
        plan = collect_plan(cfg)
        ready = final_readiness(cfg)
        if args.plan or args.dry_run:
            receipt = {'mode': 'plan' if args.plan else 'dry-run', 'generated_local': dt.datetime.now().astimezone().isoformat(), 'inputs_path': str(path), 'inputs_sha256': sha(path), 'plan': plan, 'readiness': ready, 'delivery_directory_created': False, 'new_model_or_network_calls': 0}
            dest = HERE / ('PACKAGING_PLAN.json' if args.plan else 'PACKAGING_DRY_RUN.json')
            dest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({'status': 'PLAN_ONLY' if args.plan else ready['status'], 'files': plan['file_count'], 'input_bytes': plan['input_bytes'], 'omitted': len(plan['omitted']), 'plan_errors': plan['errors'], 'missing_final_evidence': ready['errors'], 'receipt': str(dest), 'delivery_directory_created': False}, ensure_ascii=False))
            return 0 if args.plan else (0 if ready['status'] == 'READY' and not plan['errors'] else 2)
        result = make_package(plan, ready)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, Refused, ValueError, KeyError, TypeError, csv.Error) as exc:
        created = not delivery_before and (HANDOFF / 'delivery/portable_handoff').exists()
        print(json.dumps({'status': 'REFUSED', 'reason': str(exc), 'delivery_created_by_this_call': created,
                          'partial_outputs_preserved_if_present': created}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

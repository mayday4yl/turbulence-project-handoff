"""Strict final experimental metadata gate for the handoff registers.
Copied readable gate statements from package helper SHA41ed23a8, retaining its
native CPU4-role/SUMMARY/Adam/RNG/full-output/figure/exit/backup requirements.
No document QA or package publication: those are downstream independent gates.
Only local small JSON/figures/source SHA; large checkpoints are stat-only.
"""
import hashlib
import importlib.util
from pathlib import Path
PACKAGE_SOURCE_SHA256 = '41ed23a8e75e88ffaf091e47ea2c6d5023e743e515cbef76389c036d9b557fd9'
SOURCE_DERIVATION = 'native final_readiness first try and upstream adapter/exit/backup portion of its second try; exact audited source only'
SLOTS = ('experiment_full_delivery_receipt', 'all_owned_process_exit_audit', 'unique_checkpoint_backup_audit')

def validate(preparation, registry):
    prep = Path(preparation).resolve()
    package_source = prep / 'package_preparation/build_handoff_package.py'
    source_sha = hashlib.sha256(package_source.read_bytes()).hexdigest()
    if source_sha != PACKAGE_SOURCE_SHA256:
        raise ValueError('Reviewed package metadata validator source changed; independent review required')
    spec = importlib.util.spec_from_file_location('_handoff_readonly_package_gate', package_source)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    need, sha, jload = tool.need, tool.sha, tool.jload
    project_path, local_from_remote = tool.project_path, tool.local_from_remote
    CAMPAIGN, DEADLINE = tool.CAMPAIGN, tool.DEADLINE
    NATIVE_SMALL, ORIGINAL_PNGS = tool.NATIVE_SMALL, tool.ORIGINAL_PNGS
    need(registry.get('current_experiment_final_acceptance') is True,
         'A final receipt gate requires explicit final request; false remains preparation')
    receipt = registry.get('final_experiment_receipt')
    related_receipts = registry.get('final_related_receipts', {})
    need(set(related_receipts) == set(SLOTS[1:]), 'Final registers require both exact owned-exit and actual-unique-backup adapter bindings')
    cfg = {'status':'READY_FOR_PACKAGE', 'receipts':{'experiment_full_delivery_receipt':receipt, **related_receipts}}
    proof = {'required_packaging_files':[]}
    for slot in SLOTS:
        path, q = tool.require_adapter(slot, cfg)
        proof[slot] = {'path':str(path), 'sha256':sha(path), 'receipt':q}
        proof['required_packaging_files'] += [str(path)] + [str(project_path(item['path'])) for item in q['evidence_bindings']]
    report = jload(CAMPAIGN / 'report/REPORT_STATUS.json')
    need(report.get('status') == 'FINAL_READY' and report.get('final_requested') is True and (report.get('final_requirements_pass') is True) and (report.get('visual_qa_pass') is True) and (report.get('pending') == []), '实验REPORT_STATUS尚非全部原生门槛FINAL_READY')
    for k in ['new_restoration_network_calls', 'metric_network_calls', 'optimizer_updates_by_reporter']:
        need(report.get(k) == 0, 'reporter零新调用字段不符：' + k)
    complete = jload(CAMPAIGN / 'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json')
    need(complete.get('status') == 'PASS' and set(complete.get('branches', {})) == {'original', 'improved'} and all((q.get('status') == 'PASS' for q in complete['branches'].values())), '两路线完整模型评价尚未PASS')
    cx = jload(CAMPAIGN / 'evidence/CONTROLLER_EXIT.json')
    need(cx.get('all_subprocesses_exited') is True and cx['unix'] >= complete['unix'], '控制器/子进程退出未验收')
    cpu = jload(CAMPAIGN / 'evidence/CPU_POSTPROCESS_ACCEPTANCE.json')
    fx = jload(CAMPAIGN / 'evidence/CPU_FINALIZER_EXIT.json')
    need(cpu.get('status') == 'PASS' and fx.get('actor') == cpu.get('actor') and (fx.get('status') == 'CPU_AUDIT_AND_REPORT_PASS_VISUAL_AND_LOCAL_DELIVERY_PENDING') and (fx['unix'] >= cpu['completed_unix']), 'CPU收尾器尚未验收并退出')
    need(all((cpu.get(k) == 0 for k in ['restoration_model_calls', 'new_optimizer_updates', 'NPU_jobs'])), 'CPU收尾零新恢复调用不符')
    aud = jload(CAMPAIGN / 'evidence/FINAL_OUTPUTS_CPU_AUDIT.json')
    need(aud.get('status') == 'PASS' and aud.get('pure_CPU') is True and (aud.get('restoration_model_constructed') is False) and all((aud.get(k) == 0 for k in ['restoration_network_calls', 'NPU_calls', 'optimizer_steps'])), '全部保存预测CPU审计未PASS')
    refs = cpu.get('refs', [])
    need({(q['branch'], q['role']) for q in refs} == {(b, r) for b in ['original', 'improved'] for r in ['best', 'latest']} and len(refs) == 4, 'CPU收尾未绑定两线best/latest四逻辑角色')
    identities = {}
    for ref in refs:
        key = ref['branch'] + ':' + ref['role']
        state_path = local_from_remote(ref['checkpoint_CPU_audit'])
        summary_path = local_from_remote(ref['summary'])
        proof['required_packaging_files'] += [str(state_path), str(summary_path)]
        need(sha(state_path) == ref['checkpoint_CPU_audit_sha256'] and sha(summary_path) == ref['summary_sha256'], '最终state/SUMMARY小文件SHA不符：' + key)
        (state, summary) = (jload(state_path), jload(summary_path))
        cp = state.get('checkpoint', {})
        need(state.get('status') == 'PASS' and cp.get('status') == 'PASS' and (cp.get('sha256') == ref['checkpoint_sha256']) and (cp.get('update') == ref['update']) and (cp.get('branch') == ref['branch']) and (cp.get('EMA', {}).get('full_parameter_digest') == ref['checkpoint_EMA_full_digest'] == summary.get('parameter_digest')), '完整state与EMA/SUMMARY身份不符：' + key)
        need(state.get('source_sha256') == sha(CAMPAIGN / 'audit_state.py') and all((state.get(k) == 0 for k in ['model_objects', 'optimizer_objects', 'NPU_calls'])), '完整state审计来源或只读契约不符：' + key)
        for weights in ['RAW', 'EMA']:
            need(cp.get(weights, {}).get('tensors') == 151 and cp[weights].get('parameters') == 137147523 and (cp[weights].get('all_FP32_finite') is True), '完整151张量FP32状态审计缺失：' + key)
        need(cp.get('Adam', {}).get('states') == 151 and cp['Adam'].get('step_all') == ref['update'] and (cp['Adam'].get('all_FP32_finite') is True) and (cp.get('EMA_step') == ref['update']) and (cp.get('cursor') == 96 * ref['update']) and (cp.get('RNG', {}).get('banks') == (8 if ref['branch'] == 'original' else 5)) and (cp['RNG'].get('structure_complete') is True), '完整Adam/EMA/RNG/cursor不符：' + key)
        need(summary.get('status') == 'PASS' and summary.get('full') is True and (summary.get('branch') == ref['branch']) and (summary.get('update') == ref['update']) and (summary.get('weights') == 'EMA') and (summary.get('count') == 5000) and (summary.get('per_g') == 1000) and (summary.get('sources_per_g') == 250) and (set(summary.get('by_g', {})) == {'0.5', '1', '2', '3', '4'}) and (summary.get('nfe_per_image') == 100) and (summary.get('sampler') == 'native_reverse_sde') and (summary.get('image_clip_crop_resize') is False) and (summary.get('test_used') is False), '最终SUMMARY并非完整原生五档5000：' + key)
        need(aud.get('audited_summaries', {}).get(ref['summary']) == ref['summary_sha256'], 'CPU审计未绑定实际最终SUMMARY：' + key)
        results = [r for r in aud.get('validations', []) if r.get('summary_path') == ref['summary']]
        need(len(results) == 1, 'CPU完整预测审计结果缺失/重复：' + key)
        result = results[0]
        need(result.get('status') == 'PASS' and result.get('images') == 5000 and (result.get('CPU_LPIPS_images') == 5000) and (result.get('all_prediction_input_clean_SHA_verified') is True) and (result.get('failure_counts') == {}), 'CPU全5000 SHA/四指标覆盖未通过：' + key)
        identities[key] = {'update': ref['update'], 'sha256': ref['checkpoint_sha256']}
    fm = jload(CAMPAIGN / 'report/FIGURE_MANIFEST.json')
    qa = jload(CAMPAIGN / 'report/VISUAL_QA.json')
    need(qa.get('status') == 'PASS', '最终实验图目视QA缺失')
    for (name, expected) in fm.get('figures', {}).items():
        if name.endswith('.png'):
            need(qa.get('figures', {}).get(name) == expected and sha(CAMPAIGN / 'figures' / name) == expected, '最终实验PNG/目视SHA不符：' + name)
            proof['required_packaging_files'].append(str(CAMPAIGN / 'figures' / name))
    need(sum((n.endswith('.png') for n in fm.get('figures', {}))) >= 3, '缺最终三张实验图')
    proof['native_identities'] = identities
    proof['native_CPU_exit_unix'] = fx['unix']
    proof['required_packaging_files'] += [str(CAMPAIGN / p) for p in NATIVE_SMALL]
    original = CAMPAIGN / 'delivery/original_route'
    original_receipt = jload(original / 'ORIGINAL_ATTACHMENT_RECEIPT.json')
    original_qa = jload(original / 'VISUAL_QA.json')
    need(original_receipt.get('status') == 'ARTIFACTS_GENERATED_ROOT_VISUAL_QA_PENDING' and original_receipt.get('final_update') == identities['original:latest']['update'] and (original_receipt.get('full_best_update') == identities['original:best']['update']) and all((original_receipt.get(k) == 0 for k in ['new_restoration_model_calls', 'optimizer_calls', 'NPU_calls'])), '仅原路线最终附件身份或零新模型调用不符')
    need(original_qa.get('status') == 'PASS' and set(original_qa.get('figures', {})) == ORIGINAL_PNGS, '仅原路线最终7张新PNG缺root逐图目视SHA')
    for name in ORIGINAL_PNGS:
        need(original_receipt['outputs'].get(name) == original_qa['figures'][name] == sha(original / name), '仅原路线附件PNG与实际最终QA身份不符：' + name)
        proof['required_packaging_files'].append(str(original / name))
    proof['required_packaging_files'] += [str(original / p) for p in ['ORIGINAL_ATTACHMENT_RECEIPT.json', 'VISUAL_QA.json']]
    need(all((k in proof for k in SLOTS)) and 'native_identities' in proof, '先完成实验原生和四类逻辑槽证据')
    delivery = proof['experiment_full_delivery_receipt']['receipt']
    need(delivery.get('campaign') == CAMPAIGN.name and delivery.get('deadline_unix') == DEADLINE and (delivery.get('identities') == proof['native_identities']), '实验最终交付adapter身份未精确对应真实角色')
    exitq = proof['all_owned_process_exit_audit']['receipt']
    need(exitq.get('all_owned_exited') is True and exitq.get('live_owned_actors') == [] and (exitq.get('external_signals_sent') == 0) and (exitq.get('checked_unix', 0) >= proof['native_CPU_exit_unix']) and (set(exitq.get('campaign_roots', [])) == {'unidb_dual_20261103_until_noon', 'unidb_dual_20261103_until18', CAMPAIGN.name}), '全部历史/当前自有PGID/loader/CPU退出审计不完整')
    backups = proof['unique_checkpoint_backup_audit']['receipt']
    seen = {}
    for item in backups.get('files', []):
        p = project_path(item['local_path'])
        need(p.suffix.lower() in {'.pt', '.pth'} and p.stat().st_size == item['bytes'] and (p.stat().st_mtime_ns == item['local_mtime_ns']) and (item.get('status') == 'PASS_BYTE_IDENTICAL_FINAL') and (item['sha256'] == item['source_sha256']) and (item.get('SHA_verified_unix', 0) > 0) and (item.get('final_role_binding_verified_unix', 0) > DEADLINE), '真正最终权重备份字节验收缺失或文件已变')
        for role in item['roles']:
            key = item['branch'] + ':' + role
            need(key not in seen, '重复最终备份角色')
            seen[key] = {'update': item['update'], 'sha256': item['sha256']}
    need(seen == proof['native_identities'], '备份不是实际最终unique best/latest；旧候选不能替代')

    files = []
    for raw in sorted(set(proof['required_packaging_files'])):
        path = project_path(raw)
        files.append({'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path),
                      'kind':'fixed_final_figure' if path.suffix.lower() == '.png' else 'final_metadata'})
    return {'status':'PASS_EXPLICIT_NATIVE_FINAL_RECEIPT', 'identities':proof['native_identities'],
            'native_CPU_exit_unix':proof['native_CPU_exit_unix'], 'files':files,
            'package_gate_source':{'path':str(package_source),'sha256':source_sha},
            'adapters':{slot:{'path':proof[slot]['path'],'sha256':proof[slot]['sha256']} for slot in SLOTS},
            'network_calls':0, 'model_calls':0, 'optimizer_updates':0, 'checkpoint_payload_reads':0,
            'scope':'Experimental native final acceptance only; final handoff document/package delivery remain separate'}

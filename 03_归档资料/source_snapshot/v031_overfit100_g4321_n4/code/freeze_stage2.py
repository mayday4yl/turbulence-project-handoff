#!/usr/bin/env python3
"""Freeze only a fully passed Stage 2; this script never launches training."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path('/data2/yanglin/v031_overfit100_g4321_n4')


def read(relative):
    return json.loads((ROOT / relative).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(relative, value):
    path = ROOT / relative
    text = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f'Refuse to overwrite a different frozen artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def checked_evidence(record):
    path = Path(record['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Evidence path leaves this project')
    require(digest(path) == record['sha256'], f'Evidence hash changed: {path}')
    return json.loads(path.read_text())


def released_guard(name):
    guard = read(f'runs/{name}/guard_result.json')
    require(guard['status'] == 'completed' and guard['own_npu_processes_released'] is True,
            f'Unreleased/failed acceptance worker: {name}')


def main():
    require(not (ROOT / 'runs/formal').exists(), 'Formal run already exists; do not refreeze')
    selected = read('artifacts/STAGE2_RUNTIME_SELECTION.json')
    require(selected['status'] == 'passed_waiting_inference_smoke_and_freeze', 'Runtime selection is incomplete')
    ledger = read('artifacts/STAGE2_SMOKE_LEDGER.json')
    require(checked_evidence(selected['ledger']) == ledger, 'Selection ledger identity changed')
    require(ledger['reserved_updates'] == 0 and ledger['conservative_consumed_updates'] <= 60,
            'Unresolved or over-budget smoke accounting')
    require(read('artifacts/STAGE1_STATUS.json')['stage1_status'] == 'v031_implementation_acceptance_passed',
            'Stage 1 must pass first')
    require(read('artifacts/AUTO_TO_1000_AUTHORIZATION.json')['mode'] == 'AUTO_TO_1000', 'Handoff not authorized')
    inferred = read('artifacts/stage2_inference_batch1_01/smoke.json')
    require(inferred['status'] == 'passed' and inferred['comparison_passed']
            and inferred['network_simulator_unchanged'] and inferred['full_refined_status'] == 'passed'
            and inferred['batch_size'] == 1, 'Canonical single-image inference not accepted')
    initial_gradient = selected.get('initial_ddp_normalization', selected['selected_ddp_normalization'])
    for name in ['stage2_inference_batch1_01', Path(initial_gradient['path']).parent.name]:
        released_guard(name)
    cpu_log = (ROOT / 'artifacts/stage2_cpu_tests_final.log').read_text()
    require(re.search(r'Ran 57 tests in .*\n\nOK\s*$', cpu_log), 'Final 57 CPU tests have not passed')
    controller = read('artifacts/STAGE2_CONTROLLER_TESTS.json')
    require(controller['status'] == 'passed' and controller['tests_run'] == 16, 'Controller regression tests missing')
    for relative, expected in controller['source_hashes'].items():
        require(digest(ROOT / relative) == expected, f'Controller changed after tests: {relative}')
    mb = selected['selected_microbatch']
    cfg = read(f'configs/stage2_candidate_mb{mb}.json')
    require(digest(ROOT / f'configs/stage2_candidate_mb{mb}.json') == selected['candidate_configs'][str(mb)]['sha256'],
            'Selected candidate config changed')
    gradient_record = selected['selected_ddp_normalization']
    gradient = checked_evidence(gradient_record)
    require(gradient['status'] == 'passed', 'Selected DDP normalization not passed')
    require(len(gradient['rank_results']) == 4, 'Selected gradient evidence lacks four ranks')
    for rank, row in enumerate(gradient['rank_results']):
        require(row['status'] == 'passed' and row['rank'] == rank and row['microbatch'] == mb
                and row['config_sha256'] == digest(ROOT / f'configs/stage2_candidate_mb{mb}.json')
                and row['optimizer_step_attempts'] == row['optimizer_step_calls'] == 0
                and row['network_unchanged'] and row['simulator_unchanged'], 'Selected rank gradient evidence failed')
        require(row.get('reference_microbatch', 1) == 1, 'Gradient reference must use canonical microbatch one')
    require(mb == 1 or 'guard' in gradient_record, 'Selected multi-image gradient guard evidence missing')
    if 'guard' in gradient_record:
        guard = checked_evidence(gradient_record['guard'])
        require(guard['status'] == 'completed' and guard['own_npu_processes_released'] is True,
                'Selected-microbatch gradient guard did not finish cleanly')
    resume_record = selected['selected_resume_check']
    resume = checked_evidence(resume_record)
    resume_policy = read('configs/resume_acceptance_policy.json')
    require(resume_policy['criterion'] == 'exact_restore_then_bounded_fp32_reexecution_v1'
            and resume_policy['continuation_atol'] == 2**-23
            and resume_policy['continuation_rtol'] == 8 * 2**-23
            and resume_policy['continuation_group_relative_l2_max'] == 8 * 2**-23,
            'Resume numerical acceptance policy changed')
    resume_tests = read('artifacts/STAGE2_RESUME_REEXECUTION_TESTS.json')
    require(resume_tests['status'] == 'passed' and resume_tests['tests_run'] == 8
            and resume_tests['worker_sha256'] == digest(ROOT / 'code/train_worker.py'),
            'Resume checker tests do not cover final bytes')
    require(resume['status'] == 'passed' and len(resume['rank_results']) == 4, 'Selected resume evidence missing')
    released_guard(Path(resume_record['path']).parent.name)
    for rank, row in enumerate(resume['rank_results']):
        require(row['rank'] == rank and row['microbatch'] == mb and row['status'] == 'passed'
                and row['resume_boundary_parity']['exact'] is True
                and row['resume_boundary_parity']['checked_before_any_restored_optimizer_attempt'] is True
                and row['resume_parity']['passed'] is True
                and row['resume_parity']['identical_suffix_start_ids'] is True,
                'Selected complete-state resume failed on one or more ranks')
        parity = row['resume_parity']
        require(parity['criterion'] == resume_policy['criterion']
                and parity['numerical_tolerance'] == {'atol': 2**-23, 'rtol': 8 * 2**-23,
                                                       'group_relative_l2_max': 8 * 2**-23}
                and parity['failed_acceptance_fields'] == []
                and set(parity['group_relative_l2']) == set(resume_policy['independent_relative_l2_groups'])
                and all(v is not None and math.isfinite(v) and 0 <= v <= 8 * 2**-23
                        for v in parity['group_relative_l2'].values()),
                'Selected reexecution does not meet the declared FP32 bounds')
    method_hashes = gradient['rank_results'][0]['identity']['actual_code_sha256']
    for relative, expected in method_hashes.items():
        require(digest(ROOT / relative) == expected, f'Method code changed after acceptance: {relative}')
    subset = read('artifacts/stage2_frozen_subset_metadata.json')
    policy = read('configs/inference_calibration_policy.json')
    require(inferred['policy'] == policy, 'Inference smoke policy differs from frozen calibration policy')
    require(len(policy['start_ids']) == 12 and len(policy['trajectory_ids']) == 4,
            'Fixed inference calibration sample identities missing')
    require(subset['start_manifest_sha256'] == '606347e24c03d669306011b86621d237df16101191b9dbd78ddf8a9d3e87aed2',
            'Fixed 64-start subset changed')
    median = selected['selected_median_seconds_per_update']
    solver_seconds = statistics.median([x['individual_solver_seconds'] for x in inferred['batch_vs_individual']])
    # Real measurements at ten xi steps bound estimates for the fixed candidate
    # budgets. Timing is an estimate, not a promise about trained iterates.
    per_iteration = solver_seconds / 10
    two_path_seconds = inferred['full_refined_two_G4_seconds']
    path_non_solver = max(0, two_path_seconds - 6 * solver_seconds) / 2
    fixed_unrefined_two_weights = 2 * 64 * path_non_solver
    fixed_refined_range = [2 * 64 * path_non_solver + 2 * 96 * n * per_iteration for n in [10, 40]]
    full_unrefined_two_weights = 2 * 1600 * path_non_solver
    full_refined_range = [2 * 1600 * path_non_solver + 2 * 2400 * n * per_iteration for n in [10, 40]]
    timing = {'training_1000_updates_seconds': 1000 * median,
              'training_median_seconds_per_update': median,
              'calibration_24_probes_3_LR_40_steps_seconds': 24 * 3 * 40 * per_iteration,
              'calibration_integration_12_starts_seconds_range': [24 * n * per_iteration + 12 * path_non_solver for n in [10, 40]],
              'fixed64_both_weights_unrefined_seconds': fixed_unrefined_two_weights,
              'fixed64_both_weights_refined_seconds_range': fixed_refined_range,
              'full1600_both_weights_unrefined_seconds': full_unrefined_two_weights,
              'full1600_both_weights_refined_seconds_range': full_refined_range,
              'inference_measurement_seconds_per_iteration': per_iteration,
              'estimation_scope': 'Measured original-initialization single-image FP32 10-step solver and full G4 paths. Ranges reflect only fixed10/40 budgets; setup/checkpoint/hash/preview overhead and trained-iterate variation are additional.'}
    cfg['stage'] = 3
    for key in ['acceptance', 'sample_selection', 'formal_training_started', 'restoration_optimizer_updates']:
        cfg.pop(key, None)
    cfg['gates'] = {key: True for key in ['stage1_passed', 'training_config_frozen', 'inference_policy_frozen']}
    cfg['identity'] = {**cfg['identity'], 'scope': 'formal_seed_restart_after_disposable_smokes',
                       'runtime_selection_sha256': digest(ROOT / 'artifacts/STAGE2_RUNTIME_SELECTION.json'),
                       'inference_policy_sha256': digest(ROOT / 'configs/inference_calibration_policy.json')}
    cfg['code_hashes'] = method_hashes
    cfg['precision']['selected'] = 'fp32'
    cfg['precision']['bf16_exclusion'] = 'Stage1 global gradient cosine 0.8705482464 below0.99'
    cfg['evaluation'] = {'batch_size': 1, 'tie_psnr_db': 1e-6,
                         'fixed_preview_start_ids': subset['first_trajectory_preview_start_ids'],
                         'fixed64_manifest': 'subsets/overfit_16traj_starts.jsonl',
                         'full_manifest': 'start_manifest.jsonl', 'modes': ['unrefined', 'refined'],
                         'weights': ['raw', 'ema'], 'initial_update0_modes': ['unrefined'],
                         'batch2_exclusion': 'native xi comparison failed at unchanged rtol1e-4/atol1e-5',
                         'calibrate_once_at_update': 100}
    cfg['resume_acceptance'] = resume_policy
    cfg['orchestration'] = {'train_segment_timeout_seconds': max(3600, math.ceil(100 * median * 3 + 1800)),
                            'calibration_timeout_seconds': 7200,
                            'subset_evaluation_timeout_seconds': 7200,
                            'full_evaluation_timeout_seconds': 43200,
                            'infrastructure_retry_limit': 2, 'automatic_stop_update': 1000,
                            'separate_process_for_every_evaluation': True}
    transport_repair = ROOT / 'artifacts/STAGE2_HCCL_TIMEOUT_REPAIR.json'
    if transport_repair.exists():
        repair = json.loads(transport_repair.read_text())
        require(repair['status'] == 'verified_by_completed_runtime_selection', 'Transport repair is unvalidated')
        checked_evidence(repair['retest'])
        require(checked_evidence(repair['runtime_selection']) == selected, 'Repair selection identity changed')
        require(digest(ROOT / repair['run_environment_script']) == repair['run_environment_script_sha256'],
                'Repaired distributed launch environment changed')
        cfg['orchestration']['hccl_connect_timeout_seconds'] = repair['change']['HCCL_CONNECT_TIMEOUT']
    write('configs/train_v031.json', cfg)
    write('configs/inference_v031.json', {'status': 'frozen_calibrate_once_at_update100',
          'policy': str(ROOT / 'configs/inference_calibration_policy.json'),
          'policy_sha256': digest(ROOT / 'configs/inference_calibration_policy.json'),
          'numerical_config_authority_after_update100': str(ROOT / 'artifacts/calibration_update100/calibration.json'),
          'evaluation_batch_size': 1, 'precision': 'fp32', 'no_user_choice_pending': True})
    write('artifacts/STAGE2_TIMING_ESTIMATES.json', timing)
    protocol = '''# V0.31 冻结训练与评估协议

本协议依据统一补充和用户已激活的 AUTO_TO_1000 授权。科学定义沿用 Stage 1：四种真实起点，完整预测状态递归；每步 clean MSE 与原始 Y/G 重退化 MSE；路径内平均再对32条路径平均；lambda_R=0.2；完整 BPTT；训练不做 xi refinement。

恢复网络从 seed20260916 和原初始化重新建立，全部1600起点均匀洗牌、无放回。每轮50updates，共20轮到1000。AdamW peak1e-4、betas(0.9,0.999)、eps1e-8；bias/normalization不衰减，其余1e-4；所有组同一学习率。完成累积和四卡归约后 global norm clip1。EMA0.996，每步更新参数、复制 buffers。FP32主参数、优化器、EMA、simulator和loss；BF16因Stage1梯度parity失败被排除。

50updates线性warmup，之后cosine到update1000的1e-5，之后显式hold。此次自动终点1000，不重启schedule。当前唯一分批配置见train_v031.json和下方报告；设备仅de36物理0–3。没有开放超参数搜索。

每100updates完整保存network、EMA、AdamW、scheduler、四rank RNG、global sampler、计数与身份。初始checkpoint0只做固定64起点unrefined基线。update100用固定raw权重校准一次：固定4trajectory的G4/3/2共12起点，unrefined非末步共24probe；同一probe固定clean/Y/G/初始xi，不读取GT。固定Adam betas(0.9,0.999)/eps1e-8/无WD，LR[1e-3,1e-2,1e-1]，预算[10,20,40]复用40步前缀；每transition重置状态；完整两块xi同LR，不clamp/归一化/投影/clip/提前停止。返回包括初始化的最佳原始观测残差迭代。

各候选必须在24probe均finite且梯度连接，按平均相对残差下降A保留A>=0.95A_best，选累计实测最快，3%耗时tie优先少步再小LR；最多按已记录合格排序尝试3组真实完整桥集成。正式refined每次固定当前clean重新比较网络xi与前步refined xi，逐图整对选择，tie继承previous，选中者只优化一次，末步/one-shot不refine。低信号单列。数值校准不使用clean质量，不证明图像质量最优。

update100校准后所有G/raw/EMA/后续checkpoint共用一组数值。每100固定64起点、500和1000全1600（包含固定64，不重复计算），raw/EMA × unrefined/refined分开；每个G记录全部合法K，第一步为同一原始Y/G的one-shot。逐步网络预测xi的原始观测MSE与refinement候选/前后objective分列。原始float RGB全图mean MSE和data_range1单图PSNR后平均，不clamp指标；tie阈值1e-6dB，G1不算多步增益。只为预览clamp，样本ID在看效果前固定。

评估逐图执行，独立进程不恢复或污染训练RNG/sampler/optimizer。batch2的native xi数值对照失败，原阈值和失败记录保留；batch1两样本重复求解完全一致且完整路径通过。评估batch与训练分批独立。

服务器入口code/run_auto_to_1000.py按0→100校准→每100诊断→500全评估与报告→1000全评估与报告顺序执行。每个子进程由原资源守护器检查空闲/健康/归属并限定时长，只清理本任务已证明归属的进程；持久化日志/PID/进度并支持客户端断开。相同基础设施故障最多重试2次，先确认资源已释放；保留中断重放日志，正式连续日志只代表完整checkpoint接续轨迹。未授权资源、身份冲突、core gradient问题、正式nonfinite/确定性OOM或需改科学方法时停止报告。

1000训练和完整评估均完成才标PAUSED_AT_1000，不因此宣称收敛或泛化。之后须有新的明确授权，将冻结配置复制为续训配置并加入匹配的新run_until_update及continuation_authorization，然后从完整checkpoint_001000.pt用--resume和--authorize-boundary-extension继续；保留原schedule/code/数值身份，不重新初始化。当前入口始终禁止自动超过1000。
'''
    (ROOT / 'artifacts/V031_TRAINING_PROTOCOL.md').write_text(protocol)
    lines = ['# V0.31 Stage 2 验收与冻结报告', '',
             '**状态：PASS，已冻结唯一训练配置和 update100 一次校准规则。正式更新尚为0。**', '',
             f'FP32；world_size=4；microbatch/card={mb}；accumulation={8//mb}；effective batch=32。',
             f'实测 steady-state median={median:.6f}秒/update。', '',
             '| microbatch | 状态 | 最长路径峰值 allocated GiB | mixed-G measured秒/update |',
             '|---:|---|---:|---|']
    for key, candidate in selected['candidates'].items():
        peak = candidate.get('memory_peak_allocated_bytes')
        lines.append(f"| {key} | {candidate['status']} {candidate.get('exclusion_reason','')} | {peak/2**30 if peak is not None else '未完成'} | {candidate.get('measured_seconds',[])} |")
    lines += ['', '所有实际候选均使用相同独立初始化和相同相应样本流。显存覆盖完整G4/K4、AdamW状态和EMA；排名使用mixed-G完整updates。具体排除理由、冷启动、测量波动、复测与选择过程见STAGE2_RUNTIME_SELECTION.json。没有使用smoke图像质量作选择。', '',
              f"Smoke已完成{ledger['actual_completed_updates']}次；保守计费{ledger['conservative_consumed_updates']}/60；未决预留0。全部smoke状态被排除出正式初始化。", '',
              '最终57项CPU测试与16项流程模拟检查通过。真实四卡归一化、梯度连接和state不变检查通过；选中分批的真实完整checkpoint立即恢复与继续轨迹验收见selected_resume_check。记录了原始失败/排除项，不把失败改成PASS。', '',
              'Inference采用batch1，固定两G4样本重复xi/目标值完全一致、完整四步路径通过。batch2的zernike最大绝对偏差约9.42e-5，未通过原rtol1e-4/atol1e-5；目标值接近不替代xi门槛。该工程选择不改变网络或simulator数学，也不宣称backend root cause已定位。', '',
              '校准固定ID：' + ', '.join(policy['trajectory_ids']), '',
              '唯一待运行的数值选择是已冻结的update100校准程序，当前没有待用户决定的scientific choice。全部loss/rollout/full-BPTT/数据不变。', '',
              '耗时分别估计（秒；初始化/检查点/额外I/O需另计）：', '',
              '```json', json.dumps(timing, indent=2, ensure_ascii=False), '```', '',
              '启动入口：`python3 code/run_auto_to_1000.py`。只有完整Stage2验收、固定文件hash和用户授权全部满足才启动。服务器进程真实启动后记录PID/starttime与日志；本报告本身不宣称已启动。', '',
              '完整冻结协议：`artifacts/V031_TRAINING_PROTOCOL.md`；配置：`configs/train_v031.json`、`configs/inference_v031.json`、`configs/inference_calibration_policy.json`。']
    (ROOT / 'artifacts/V031_STAGE2_REPORT.md').write_text('\n'.join(lines) + '\n')
    with (ROOT / 'artifacts/V031_STAGE2_REPORT.md').open('a') as handle:
        handle.write('\n续训验收口径：重载完整状态后、任何新 optimizer attempt 之前仍要求所有状态逐位相同；后续 LR、optimizer step、EMA计数、RNG、sampler和曝光计数也逐位相同。原 NPU 浮点重算 bitwise FAIL 保留，不能改写为通过。后续 FP32 权重及 AdamW moments 另按已声明 epsilon 界限和分组 relative L2 界限通过数值验收，见 resume_acceptance_policy.json 与 STAGE2_RESUME_ACCEPTANCE_CLARIFICATION.md；本报告不承诺 NPU 后续算术逐位复现。\n')
    if transport_repair.exists():
        with (ROOT / 'artifacts/V031_STAGE2_REPORT.md').open('a') as handle:
            handle.write('\nHCCL 工程修复：首次 mixed-G microbatch4 建链超时后，固定 HCCL_CONNECT_TIMEOUT=3600秒；只延长建链等待，算术和验收阈值不变。失败和保守预算计费保留，后续真实复验已通过，详见 STAGE2_HCCL_TIMEOUT_REPAIR.json 和 runtime selection recovery。\n')
    prior_acceptance = ROOT / 'artifacts/STAGE2_FINAL_ACCEPTANCE.json'
    recorded_utc = (json.loads(prior_acceptance.read_text())['recorded_utc'] if prior_acceptance.exists()
                    else dt.datetime.now(dt.timezone.utc).isoformat())
    acceptance = {'status': 'passed', 'recorded_utc': recorded_utc,
                  'stage1': 'passed', 'training_config': 'frozen',
                  'inference_policy': 'frozen_calibrate_once_at_update100', 'unresolved_scientific_choices': [],
                  'formal_training_started': False, 'formal_global_update': 0,
                  'train_config_sha256': digest(ROOT / 'configs/train_v031.json'),
                  'inference_policy_sha256': digest(ROOT / 'configs/inference_calibration_policy.json'),
                  'runtime_selection_sha256': digest(ROOT / 'artifacts/STAGE2_RUNTIME_SELECTION.json'),
                  'method_code_hashes': method_hashes, 'smoke_ledger': ledger,
                  'inference_batch': 1, 'backend_root_cause_proven': False}
    write('artifacts/STAGE2_FINAL_ACCEPTANCE.json', acceptance)
    paths = list((ROOT / 'code').glob('*.py')) + list((ROOT / 'code').glob('*.sh'))
    paths += list((ROOT / 'code/v031').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    paths += list((ROOT / 'configs').glob('*.json'))
    if transport_repair.exists():
        paths += [transport_repair, ROOT / 'docs/STAGE2_HCCL_TIMEOUT_REPAIR.md']
    paths += [ROOT / name for name in ['docs/STAGE2_RUNTIME_SELECTION_PROTOCOL.md',
              'docs/STAGE2_CROSS_MICROBATCH_GRADIENT_ADDENDUM.md',
              'docs/STAGE2_RESUME_ACCEPTANCE_CLARIFICATION.md',
              'artifacts/STAGE2_RESUME_REEXECUTION_TESTS.json',
              'artifacts/STAGE2_FINAL_ACCEPTANCE.json', 'artifacts/V031_STAGE2_REPORT.md',
              'artifacts/V031_TRAINING_PROTOCOL.md', 'artifacts/STAGE2_RUNTIME_SELECTION.json',
              'artifacts/STAGE2_CONTROLLER_TESTS.json', 'artifacts/STAGE2_TIMING_ESTIMATES.json',
              'artifacts/STAGE2_SMOKE_LEDGER.json', 'artifacts/STAGE1_FINAL_ACCEPTANCE.json',
              'artifacts/STAGE1_DELIVERY_SHA256.json', 'artifacts/AUTO_TO_1000_AUTHORIZATION.json']]
    write('artifacts/STAGE2_DELIVERY_SHA256.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))})
    status = read('artifacts/WORKFLOW_STATUS.json')
    status.update(status='ready_for_automatic_stage3', current_stage=2, stage2='passed',
                  training_config='frozen', inference_policy='frozen_calibrate_once_at_update100',
                  inference_batch1='passed', native_four_card_gradient='passed_final_selected_configuration',
                  smoke_optimizer_updates=ledger['conservative_consumed_updates'], formal_global_update=0,
                  formal_training_started=False, run_until_update=1000, allow_automatic_updates_beyond_1000=False)
    (ROOT / 'artifacts/WORKFLOW_STATUS.json').write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'stage2_passed_and_frozen', 'microbatch': mb, 'formal_updates': 0}))


if __name__ == '__main__':
    main()

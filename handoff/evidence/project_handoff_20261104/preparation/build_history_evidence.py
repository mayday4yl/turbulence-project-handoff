#!/usr/bin/env python3
"""Build V4/V5/V6 handoff provenance from existing small local reports only.

No third-party imports, network, subprocess, accelerator, checkpoint load or writes
outside this new preparation directory. Run --verify to recheck recorded sources.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
MAX_SOURCE_BYTES = 8 * 1024 * 1024


def branch(key, kind, status, date, sources, **facts):
    return dict(id=key, kind=kind, historical_status=status,
                historical_date=date, sources=sources.split(), facts=facts)


V4_SOURCES = {
    'implementation': 'v4_noise_bridge_20260929/DELIVERY_REPORT.md',
    'asset_readme': 'v4_noise_bridge_20260929/README.md',
    'initial_formal': 'v4_noise_bridge_training_20260929/results/FINAL_REPORT_ZH.md',
    'initial_acceptance': 'v4_noise_bridge_training_20260929/reports/final_completion_audit.json',
    'metrics_supplement': 'v4_metrics_supplement_20260929/results/FINAL_REPORT_ZH.md',
    'overfit5_full': 'v4_overfit5_d0_40ep_20260929/results/report/REPORT_ZH.md',
    'overfit5_clean': 'v4_overfit5_cleanonly_d0_40ep_20260929/results/report/REPORT_ZH.md',
    'cancelled': 'v4_overfit5_cleanonly_resume200_20260929/README.md',
    'overfit5_plateau': 'v4_overfit5_cleanonly_resume_plateau_20260929/results/report/REPORT_ZH.md',
    'overfit5_full_resume': 'v4_overfit5_full_d0_resume_20260929/results/report/REPORT_ZH.md',
    'one_source': 'v4_one_source16_cleanonly_d0_20260929/results/report/REPORT_ZH.md',
    'one_pair': 'v4_single_pair_cleanonly_400u_20260929/results/report/REPORT_ZH.md',
    'fixed_operator': 'v4_fixed_physics_operator_20260929/results/report/REPORT_ZH.md',
    'blind_ambiguity': 'v4_blind_ambiguity_20260929/results/report/REPORT_ZH.md',
    'train500': 'v4_train500_matched_v2_perf1_eval5kfull/reports/U05000_READOUT.md',
    'pirn': 'v4_pirn_backbone_matched4_20261001/reports/SUMMARY.md',
    'swinir': 'v4_swinir_l1_train500_2k_20261001/reports/U2000_DELIVERY_REVIEW.md',
    'restormer_candidate': 'v4_restormer_candidate_20261002/README.md',
    'restormer_formal': 'v4_restormer_train2000_npu_20k_20261003/README.md',
    'restormer_matched': 'v4_nafnet_sidd116m_train8000_20261006/reports/OLD_U5000_FULLVAL_SUMMARY.md',
    'nafnet_candidate': 'v4_nafnet_sidd116m_candidate_20261005/README.md',
    'nafnet_adaptation': 'v4_nafnet_sidd116m_candidate_20261005/ADAPTATION.md',
    'nafnet_candidate_acceptance': 'v4_nafnet_sidd116m_candidate_20261005/reports/pretrained/ACCEPTANCE_SUMMARY.json',
    'nafnet_pretrained_identity': 'v4_nafnet_sidd116m_candidate_20261005/protocol/PRETRAINED_SOURCE.json',
    'nafnet_formal': 'v4_nafnet_sidd116m_train8000_20261006/README.md',
    'nafnet_phase1': 'v4_nafnet_sidd116m_train8000_20261006/reports/PHASE1_FINAL_20261012/U20000_FOUR_METHOD_RESULTS.md',
    'nafnet_phase1_acceptance': 'v4_nafnet_sidd116m_train8000_20261006/reports/PHASE1_FINAL_20261012/FINAL_ACCEPTANCE.json',
    'nafnet_rev4': 'v4_nafnet_sidd116m_train8000_20261006/deliverables/U20000_FOUR_METHOD_RESULTS_20261016_REV4/README_ZH.md',
    'nafnet_stop': 'v4_nafnet_train15000_20261013/reports/user_stop_20261018/停止与结果保全说明.md',
    'nafnet_stop_acceptance': 'v4_nafnet_train15000_20261013/reports/user_stop_20261018/USER_STOP_ACCEPTANCE.json',
    'nafnet_preservation': 'v4_nafnet_train15000_20261013/reports/user_stop_20261018/FINAL_PRESERVATION_ACCEPTANCE.json',
    'continuous_initial': 'v4_followup_pilot_train15000_20261013/PILOT_REPORT.md',
    'continuous_final': 'v4_followup_pilot_lr5e6_attempt2_20261014/PILOT_REPORT.md',
    'rawfield': 'de36_g4_raw_vs_field_12h_20261015/reports/FINAL_CPU_AUDIT/FINAL_REPORT_ZH.md',
    'rawfield_checkpoint': 'de36_g4_raw_vs_field_12h_20261015/reports/FINAL_CPU_AUDIT/CHECKPOINT_AUDIT.json',
    'rawfield_delivery': 'de36_g4_raw_vs_field_12h_20261015/reports/FINAL_CPU_AUDIT/DELIVERY_COMPLETE.json',
    'archive_oct02': 'storage_archives/A100_cleanup_20261002/CLEANUP_RECEIPT.md',
    'archive_oct25': 'output/a100_cleanup_100g_review_20261025/execution/CLEANUP_RECEIPT.md',
    'archive_oct25_acceptance': 'output/a100_cleanup_100g_review_20261025/execution/ARCHIVE_ACCEPTANCE.json',
    'archive_oct25_restore': 'output/a100_cleanup_100g_review_20261025/execution/RESTORE_README.md',
}

V4_BRANCHES = [
    branch('noise_bridge', 'implementation_gate', 'ENGINEERING_ACCEPTED_WITH_SERIAL_REFINEMENT_LIMITS', '2026-09-29', 'implementation asset_readme', formal_updates=0, temporary_updates=1, parameters=32038377, strict_refinement_determinism_pass=False),
    branch('overfit100', 'formal', 'D0_R0_R08_COMPLETED', '2026-09-29', 'initial_formal initial_acceptance metrics_supplement', each_updates=1000, each_exposures=32000, validation_starts=800, validation_clean_sources=100, raw_off_psnr={'Input':20.5394,'D0':19.1983,'R0':18.8805,'R08':19.1895}, all_models_above_input=False),
    branch('overfit5_full', 'fitting_diagnostic', 'COMPLETED', '2026-09-29', 'overfit5_full', updates=400, train_starts=80, train_psnr=22.773847, val64_psnr=17.534017),
    branch('overfit5_clean', 'fitting_diagnostic', 'COMPLETED', '2026-09-29', 'overfit5_clean', updates=400, train_psnr=22.590230, val64_psnr=17.549631),
    branch('overfit5_resume200_cancelled', 'preparation', 'CANCELLED_NOT_UPLOADED', '2026-09-29', 'cancelled', updates=0),
    branch('overfit5_clean_plateau', 'fitting_diagnostic', 'HIT_CAP_NOT_CONFIRMED_PLATEAU', '2026-09-29', 'overfit5_plateau', total_updates=2000, train_psnr=31.037437, val64_psnr=15.744348),
    branch('overfit5_full_resume', 'fitting_diagnostic', 'COMPLETED', '2026-09-29', 'overfit5_full_resume', total_updates=2000, train_psnr=30.670328, val64_psnr=15.803111),
    branch('one_source16', 'fitting_diagnostic', 'COMPLETED', '2026-09-29', 'one_source', updates=400, train_psnr=30.560403, val64_psnr=11.952586),
    branch('single_pair', 'fitting_diagnostic', 'COMPLETED_NO_HELDOUT_VAL', '2026-09-29', 'one_pair', updates=400, input_psnr=19.440704, train_pair_psnr=34.693485),
    branch('fixed_operator', 'identifiability_diagnostic', 'COMPLETED_STRUCTURAL_NONINJECTIVITY_EVIDENCE', '2026-09-29', 'fixed_operator', full_svd=False, preregistered_nearzero_witness_found=False),
    branch('blind_ambiguity', 'identifiability_diagnostic', 'LOCAL_C_XI_COMPENSATION_WITNESSES', '2026-09-29', 'blind_ambiguity', network_updates=0, diagnostic_variable_updates=21600, formal_restoration_test=False),
    branch('unet_train500', 'formal_stage_readout', 'U5000_STAGE_VERIFIED_LATER_FULL_END_NOT_VERIFIED', '2026-10-01', 'train500', each_updates=5000, each_exposures=160000, raw_off_val800_psnr={'D0':19.6670,'R0':19.4339,'R08':19.5665,'L08':19.3841,'Input':20.5394}),
    branch('pirn_backbone', 'candidate_gate', 'FOUR_METHOD_GATES_COMPLETE_NOT_FORMAL_TRAIN', '2026-10-01', 'pirn', optimizer_updates=0, official_pirn_full_method=False, parameters=17611761),
    branch('swinir_2k', 'formal', 'FOUR_METHOD_U2000_DELIVERED', '2026-10-02', 'swinir', each_updates=2000, each_exposures=64000, raw_off_val800_psnr={'D0':20.473245,'R0':20.443722,'R08':20.419548,'L08':20.400306,'Input':20.539433}),
    branch('restormer_candidate', 'candidate_gate', 'BOUNDED_CORRECTNESS_GATE_ONLY', '2026-10-02', 'restormer_candidate', parameters=27032525, optimizer_updates=0, r0_cuda_gate_cpu_saved_tensor_offload=True),
    branch('restormer_platform_migration', 'formal_then_fresh_restart', 'NPU_USER_STOP_A100_NEW_RANDOM_U0', '2026-10-03', 'restormer_formal restormer_matched', npu_stop_updates={'D0':669,'R0':196,'R08':198,'L08':201}, a100_from_npu_checkpoint=False, a100_main_closeout_update=5000, val2000_psnr={'D0':22.7106,'R0':22.7497,'R08':22.7013,'L08':22.7123}),
    branch('nafnet_candidate', 'candidate_gate', 'OFFICIAL_SIDD_BODY_ADAPTATION_ACCEPTED', '2026-10-06', 'nafnet_candidate nafnet_adaptation nafnet_candidate_acceptance nafnet_pretrained_identity', official_parameters=115982915, adapted_parameters=117767209, full_official_residual_mapping_unchanged=False),
    branch('nafnet_train8000', 'formal', 'FOUR_U20000_AND_104_EVALS_COMPLETE_REV4_SUPPLEMENT', '2026-10-16', 'nafnet_formal nafnet_phase1 nafnet_phase1_acceptance nafnet_rev4', train_clean_sources=8000, val_clean_sources=1000, full_val_starts=8000, each_updates=20000, each_exposures=640000, psnr={'D0':23.09077535,'R0':22.91951237,'R08':22.73627484,'L08':22.81135343}, test_used=False),
    branch('nafnet_train15000', 'formal_user_stop', 'ALL_SAFE_SAVED_NOT_ALL_U40000', '2026-10-18', 'nafnet_stop nafnet_stop_acceptance nafnet_preservation', global_update={'D0':40000,'R0':40000,'R08':32891,'L08':21559}, actual_stop_point_full_eval_available={'D0':True,'R0':True,'R08':False,'L08':False}, local_stop_checkpoint_backup_recorded_bytes=7541233596, auto_resume=False),
    branch('continuous_delta_initial', 'pilot', 'NO_MATCHED_COMMON_MILESTONE_AT_REPORT', '2026-10-13', 'continuous_initial'),
    branch('continuous_delta_attempt2', 'pilot', 'MATCHED_U1000_NEGATIVE_PSNR_RESULT', '2026-10-14', 'continuous_final', each_updates=1000, val_starts=400, D0_psnr=18.753920, CONT_DELTA_R0_psnr=18.421053, delta_ci95=[-.40078325,-.27556037]),
    branch('raw_field', 'formal_pilot', 'MATCHED_U1000_FULL_VAL64_CPU_DELIVERY_COMPLETE', '2026-10-15', 'rawfield rawfield_checkpoint rawfield_delivery', each_updates=1000, train_clean_sources=128, val_clean_sources=32, val_starts=64, raw_psnr=19.836685, field_psnr=19.827981, input_psnr=20.133437, delta_ci95=[-.01755358,-.00024447], all_metric_field_worse=False, full_training_generalization_claim=False),
    branch('a100_archives', 'archive_then_delete', 'HISTORICAL_HASH_VERIFIED_ARCHIVES', '2026-10-25', 'archive_oct02 archive_oct25 archive_oct25_acceptance archive_oct25_restore', oct02_checkpoint_files=93, oct25_files=126050, oct25_remote_archive='/data2/yanglin/storage_archives/A100_cleanup_20261025_train15000_data/files/', current_remote_existence_rechecked=False),
]

V5_SOURCES = {
    'method': 'v5_v1_simulator_20261020/docs/METHOD_SPEC_ZH.md',
    'simulator_acceptance': 'v5_v1_simulator_20261020/acceptance_20261021/RESEARCH_ACCEPTANCE_ZH.md',
    'a100_gate': 'v5_v1_simulator_a100_20261021/a100_reports/A100_GATE_REPORT_ZH.md',
    'backbone_review': 'v5_backbone_review_20261021/BACKBONE_COMPARISON_ZH.md',
    'a100_benchmark': 'v5_concertormer_a100_benchmark_20261021/README.md',
    'psf_optimization': 'v5_concertormer_10k_20261021/optimizations/psf_adjoint_20261021/REPORT_ZH.md',
    'formal_execution': 'v5_concertormer_10k_20261021/reports/CURRENT_EXECUTION_ZH.md',
    'formal_final': 'v5_concertormer_10k_20261021/reports/alignment_A6k_B4k_20261022/FINAL_REPORT_ZH.md',
    'formal_results': 'v5_concertormer_10k_20261021/reports/alignment_A6k_B4k_20261022/FINAL_RESULTS.json',
    'unified_val': 'v5_concertormer_10k_20261021/reports/alignment_A6k_B4k_20261022/UNIFIED_VAL1000_FINAL_ZH.md',
    'unified_val_json': 'v5_concertormer_10k_20261021/reports/alignment_A6k_B4k_20261022/UNIFIED_VAL1000_FINAL.json',
    'stepwise_replay': 'v5_concertormer_10k_20261021/reports/three_stage_stepwise_20261027/MANIFEST.json',
    'resshift_stage1': 'v5_resshift_stage1_20261026/reports/FINAL_REPORT_ZH.md',
    'resshift_grad': 'v5_resshift_grad_vq_audit_20261026/reports/FINAL_REPORT_ZH.md',
    'resshift_codec': 'v5_resshift_vq_repeat4_20261026/reports/FINAL_REPORT_ZH.md',
    'resshift_archive': 'v5_diffuir_l_stage1_20261026/archive/ARCHIVE_RECEIPT_ZH.md',
    'diffuir_stage1': 'v5_diffuir_l_stage1_20261026/reports/FINAL_REPORT_ZH.md',
    'diffuir_benchmark': 'v5_diffuir_l_benchmark_20261026/BENCHMARK_REPORT_ZH.md',
    'diffuir_benchmark_results': 'v5_diffuir_l_benchmark_20261026/RESULTS.json',
}

V5_BRANCHES = [
    branch('simulator', 'research_acceptance', 'ACCEPTED_WITH_KNOWN_LIMITS_ORIGINAL_GATE_NOT_ALL_PASSED', '2026-10-21', 'method simulator_acceptance a100_gate', formal_training_in_gate=False, nearzero_threshold=.002, nearzero_actual_max=.002176, full_gate_pass=False),
    branch('backbone_review', 'static_candidate_review', 'CPU_STATIC_ONLY', '2026-10-21', 'backbone_review', models_run=0, checkpoints_downloaded=0, training_updates=0),
    branch('a100_preparation', 'benchmark_preparation', 'GPU_BENCHMARK_NOT_RUN_AT_REPORT', '2026-10-21', 'a100_benchmark', formal_training=False),
    branch('psf_input_gradient', 'equivalent_engineering_optimization', 'BOUNDED_MATH_GRADIENT_MULTICARD_PASS', '2026-10-21', 'psf_optimization', original_frozen_sources_changed=False, four_card_seconds_before_after={'D0':[21.45,9.60],'R0':[87.28,39.09]}, quality_claim=False),
    branch('concertormer_D0_A_B', 'formal', 'ALL_AUTHORIZED_ALIGNMENT_COMPLETE_EXITED', '2026-10-26 00:32 Asia/Shanghai', 'formal_execution formal_final formal_results unified_val unified_val_json', updates={'D0':10000,'A':6000,'B':4000}, best_updates={'D0':8250,'A':5000,'B':2750}, fixed_val_clean_sources=250, fixed_val_starts=1000, best_psnr={'D0':21.44350572,'A':20.156652,'B':21.736105}, input_psnr=18.795937524, B_minus_D0_psnr=.292599187, B_C4_minus_C1_psnr=-.105346, B_complete_checkpoint_sha256_from_original_receipt='5b1f2ff98e0cd2541899689cd6d7827a3b3f644465fc8b52ac0701f4405f21ed', retained_run_hours=91.19, workflow_wall_hours=97.10, test_used=False, refinement_on_executed=False),
    branch('concertormer_stepwise_replay', 'post_training_illustrative_replay', 'FIXED4TRAIN_AND4VAL_BEST_CHECKPOINT_REPLAY_COMPLETE', '2026-10-27', 'stepwise_replay', training_updates=0, examples=8, training_time_snapshots=False, full_val_metrics=False),
    branch('resshift_stage1', 'candidate_zero_shot', 'OFFICIAL_BASELINE_VAL4_ONLY_COMPLETED', '2026-10-26', 'resshift_stage1', formal_updates=0, val_images=4, input_psnr=17.7892, model_psnr=16.9712, full_val1000=False),
    branch('resshift_grad_vq', 'gradient_representation_diagnostic', 'FP32_PASS_FP16_OPERATOR_BLOCKER_RETAINED', '2026-10-26', 'resshift_grad', optimizer_updates=0, fp32_denoising_steps=4, fixed_true_xi_physical_stages_tested=2, full_four_physical_stages_predicted_xi_tested=False, clean_codec_val_images=250, clean_codec_psnr=28.7253),
    branch('resshift_codec_repeat', 'codec_only_diagnostic', 'FOUR_PASSES_ALL250_PSNR_DECLINE', '2026-10-26', 'resshift_codec', codec_calls=1000, unet_calls=0, optimizer_updates=0, psnr_by_pass=[28.7253,27.4151,26.0999,24.9492], full_restoration_claim=False),
    branch('resshift_archive', 'archive_in_place', 'HASH_MANIFEST_ACCEPTED_NOT_EXTRA_FULL_BACKUP', '2026-10-26 21:27 Asia/Shanghai', 'resshift_archive', files=7321, bytes=4897644957, source_directories_retained_in_place=True),
    branch('diffuir_L_stage1', 'candidate_plus_short_pilot', '100_UPDATE_NATIVE_RESIDUAL_PILOT_COMPLETE_EXITED', '2026-10-26', 'diffuir_stage1', official_parameters=36261187, formal_pilot_updates=100, actual_pairs=200, clean_sources=91, classes=49, val_images=50, ema_psnr={'U0':18.73989,'U25':18.74114,'U100':18.48104}, input_psnr=18.95359, mature_ema=False, best_checkpoint_sha256_from_original_receipt='055a8b95ccc4660f207a8c021b8184cde4a48db82b055cc5891fbc6ac530347e', physical_outer_recursion=False),
    branch('diffuir_benchmark', 'engineering_only', '12_RUNS_PASS_6_ZERO_UPDATE_ENV_FAILS_PRESERVED', '2026-10-26', 'diffuir_benchmark diffuir_benchmark_results', successful_runs=12, initial_zero_update_environment_failures=6, temporary_updates_discarded=144, formal_updates=0, native_measured_max_microbatch=16, native_seconds_per_update=.7095, three_step_backward_measured_max_batch=8, three_step_backward_seconds=1.0407, larger_batches_not_tested=True, hardware_hard_limit_claim=False),
]

V6_BASE = 'v6_diffuir_l_20261027/'
V6_REL_SOURCES = {
    'method': '../v6_diffuir_l_method_review_20261027/METHOD_REVIEW_ZH.md',
    'mapping': '../v6_diffuir_l_method_review_20261027/SPEC_TO_CODE_MAPPING_ZH.md',
    'initial_d0': 'V6_D0_A_B_FINAL_REPORT_ZH.md',
    'initial_results': 'reports/V6_D0_A_B_FINAL_RESULTS.json',
    'continuation': 'reports/D0_CONTINUATION_FINAL_REPORT_ZH.md',
    'A_complete': 'reports/A_COMPLETION_REPORT_ZH.md',
    'B_preflight': 'reports/B_PREFLIGHT_PILOT_AND_STARTUP_REPORT_ZH.md',
    'B_status': 'execution_20261027_five_npu/STATUS.json',
    'B_pause': 'execution_20261028_d0_until_noon/B_PAUSE_ACCEPTANCE.json',
    'B_best': 'distributed/B_five_npu_formal10000_01/BEST.json',
    'B_best_full': 'distributed/B_five_npu_formal10000_01/validation/U007000_1000/SUMMARY.json',
    'B_latest_full': 'distributed/B_five_npu_formal10000_01/validation/U009000_1000/SUMMARY.json',
    'B_result': 'distributed/B_five_npu_formal10000_01/RESULT.json',
    'joint_pause': 'execution_20261028_d0_until_noon/USER_PAUSE_REPORT_ZH.md',
    'residual_final': 'execution_20261029_d0_user_stop/D0_OFFICIAL_RESIDUAL_FINAL_REPORT_ZH.md',
    'residual_acceptance': 'execution_20261029_d0_user_stop/FINAL_ACCEPTANCE.json',
    'residual_cpu_state': 'execution_20261029_d0_user_stop/CHECKPOINT_CPU_AUDIT.json',
    'mixed_final': 'execution_20261029_d0_multistrength160k/final_20261030_1800/D0_MULTISTRENGTH_REPORT_ZH.md',
    'mixed_acceptance': 'execution_20261029_d0_multistrength160k/final_20261030_1800/FINAL_ACCEPTANCE.json',
    'mixed_delivery': 'execution_20261029_d0_multistrength160k/final_20261030_1800/LOCAL_FINAL_DELIVERY_ACCEPTANCE.json',
    'reweight_final': 'execution_20261030_d0_g4_reweight_5k/final_20261031_0255/D0_G4_REWEIGHT_FINAL_REPORT_ZH.md',
    'reweight_acceptance': 'execution_20261030_d0_g4_reweight_5k/final_20261031_0255/FINAL_ACCEPTANCE.json',
    'reweight_supplement': 'execution_20261030_d0_g4_reweight_5k/final_20261031_0255/FINAL_DELIVERY_SUPPLEMENT_20261031_ZH.md',
    'external95544': 'external_eval_20261029_otis_atd/OTIS_ATD_EVALUATION_REPORT_ZH.md',
    'external95544_acceptance': 'external_eval_20261029_otis_atd/FINAL_ACCEPTANCE.json',
    'official13': 'diagnostics/official_five_tasks_20261030/all13_final_delivery_20261031_0650/FINAL_REPORT_ZH.md',
    'official13_acceptance': 'diagnostics/official_five_tasks_20261030/all13_final_delivery_20261031_0650/FINAL_ACCEPTANCE.json',
    'external6': 'diagnostics/external_singleframe_20261031/HANDOFF_CURRENT_ZH.md',
    'external6_delivery': 'diagnostics/external_singleframe_20261031/FINAL_DELIVERY.json',
    'external_final': 'diagnostics/external_best_final_singleframe_20261101/CURRENT_HANDOFF_V3_ZH.md',
    'external_final_delivery': 'diagnostics/external_best_final_singleframe_20261101/CURRENT_DELIVERY.json',
    'atsyn_acceptance': 'diagnostics/external_atsyn_singleframe_20261101/FINAL_DELIVERY_ACCEPTANCE.json',
    'dependency': 'diagnostics/D0_diffusion_dependency_20261027_1955/D0_DIFFUSION_DEPENDENCY_REPORT_ZH.md',
    'mechanism': 'diagnostics/diffusion_mechanism_audit_20261028/V6_DIFFUSION_MECHANISM_DIAGNOSIS_ZH.md',
    'mechanism_theory': 'diagnostics/diffusion_mechanism_audit_20261028/DIFFUIR_THEORY_AND_CODE_AUDIT_ZH.md',
    'train_val': 'diagnostics/d0_u14927_train_val_generalization_20261028/TRAIN_VAL_GENERALIZATION_REPORT_ZH.md',
    'strength_eval': 'diagnostics/d0_u95544_g1_g4_val1000_20261029/G1_G4_VAL1000_REPORT_ZH.md',
    'loss_eval': 'diagnostics/d0_train_val_loss_20261029/TRAIN_VAL_LOSS_ZH.md',
    'strength_preview': 'diagnostics/strength_preview_g025_g4_20261029/PREVIEW_REPORT_ZH.md',
    'data_inventory': 'diagnostics/data_strength_inventory_20261029/DATA_STRENGTHS_ZH.md',
    'post_meeting': 'diagnostics/post_meeting_audit_20261101/AUDIT_SUMMARY.json',
    'global_curve': 'diagnostics/d0_global_train_val_review_20261031/AUDIT_AND_COVERAGE.json',
    'selection_v1': 'diagnostics/pixel_diffusion_selection_20261101/FINAL_RESEARCH_ACCEPTANCE.json',
    'selection_v2': 'diagnostics/pixel_diffusion_selection_v2_20261101/RESEARCH_REPORT_ZH.md',
    'selection_v2_acceptance': 'diagnostics/pixel_diffusion_selection_v2_20261101/FINAL_RESEARCH_ACCEPTANCE.json',
    'recent_ir': 'diagnostics/recent_pixel_ir_20261101/REPORT_ZH.md',
    'recent_ir_acceptance': 'diagnostics/recent_pixel_ir_20261101/RESEARCH_ACCEPTANCE.json',
    'nonres_expansion': 'diagnostics/pixel_nonres_topconf_expansion_20261101/REPORT_ZH.md',
    'deblur_selection': 'diagnostics/pixel_deblur_selection_20261101/REPORT_ZH.md',
    'pixeldit': 'diagnostics/pixeldit_fit_20261101/REPORT_ZH.md',
    'latent_fit': 'diagnostics/deblurdiff_diffbir_fit_20261101/FIT_REPORT_ZH.md',
}
V6_SOURCES = {k: str(Path(V6_BASE, v)) for k, v in V6_REL_SOURCES.items()}
V6_SOURCES.update({
    'turbnet': 'turbnet_zeroshot_20261028/reports/FINAL_REPORT_ZH.md',
    'turbwnet': 'turbwnet_zeroshot_20261028/reports/FINAL_REPORT_ZH.md',
})

V6_BRANCHES = [
    branch('method_review', 'static_review', 'METHOD_AND_CODE_MAPPING_COMPLETE_NOT_TRAIN', '2026-10-27', 'method mapping', optimizer_updates=0),
    branch('initial_joint_D0', 'formal', 'U10000_COMPLETE', '2026-10-27', 'initial_d0 initial_results', updates=10000, exposures=600000, effective_batch=60, train_nfe=4, inference_nfe=3, validation_psnr=21.03393849, validation_lpips=.53120907, official_u0_lpips=.51510603),
    branch('joint_D0_continuation', 'formal', 'U13334_COMPLETE_THEN_U14927_USER_PAUSE', '2026-10-28', 'continuation joint_pause', initial_extra_updates=3334, initial_extra_exposures=200040, U13334_psnr=21.05311102, next_extra_updates=1593, pause_update=14927, pause_full_val_evaluated_in_later_branch=True),
    branch('A', 'formal', '6000_UPDATES_COMPLETE_EXITED', '2026-10-28', 'A_complete', updates=6000, actual_exposures=259280, world_size_changed_from_to=[6,5], best_update=1000, best_C4_psnr=19.86120922, best_C1_psnr=20.99742496, latest_C4_psnr=19.79950879),
    branch('B', 'formal_user_pause', 'U9500_SAFE_SAVED_EXITED_NOT_U10000', '2026-10-28', 'B_preflight B_status B_pause B_best B_best_full B_latest_full B_result', updates=9500, cursor=95000, terminal_U9500_full_summary_found=False, best_update=7000, best_psnr=21.04583810240762, best_ssim=.5487442090092839, best_lpips=.5329662312120199, best_mae=.05630583086428803, best_C4_minus_C1_psnr=-.03826110927160897, latest_full_update=9000, latest_full_psnr=21.055674913514768, best_is_strict_psnr_max=False, checkpoint_sha256_from_original_receipt='e49abaae1e487e63e73a3c3151b8f5585bd4945f9a15fde103ffc1bd572f7337', auto_resume=False),
    branch('official_residual_D0', 'formal_user_stop', 'U95544_FINAL_CPU_FULL_VAL_AND_EXIT_COMPLETE', '2026-10-29', 'residual_final residual_acceptance residual_cpu_state', parent_update=14927, final_update=95544, effective_new_updates=80617, exposures=4837020, discarded_unsaved_updates=420, discarded_unsaved_exposures=25200, from_official_u0=False, train_nfe=1, inference_nfe=3, psnr=21.16514236838766, ssim=.5562608945579739, lpips=.4976828113496304, mae=.055116699808241854, checkpoint_sha256_from_original_receipt='1fd16df15c64cc496bda89ccb32fd6a9413c9de8e59942fc13cbb6592e757cc3'),
    branch('mixed_strength160k', 'formal', 'U157505_SAFE_STOP_FULL_FIVE_G_CPU_LOCAL_DELIVERY_COMPLETE', '2026-10-30 18:00 Asia/Shanghai', 'mixed_final mixed_acceptance mixed_delivery', parent_update=95544, final_update=157505, new_updates=61961, exposures=3717660, train_clean_sources=10000, starts=160000, training_G=[1,2,3,4], validation_G=[.5,1,2,3,4], val_starts_each_G=1000, latest_psnr={'0.5':29.92958440,'1':27.82736964,'2':24.81663086,'3':22.62914854,'4':20.89072573}, best_update=109544, best_G4_psnr=20.98248545, G4_latest_minus_parent=-.27441664, final_G4_protection_floor_pass=False, latest_sha256_from_original_receipt='9e6b66e6a46434dc86b327dfce6be04d78aa1de61449d7791e282c62d52e0029', best_sha256_from_original_receipt='5d7c57408af1a296f940997406c4bb2b2e876b9100ae20cd30ce2690a0d693a8'),
    branch('Control_G4biased', 'matched_finite_formal', 'BOTH5000_COMPLETE_EXTENSION_CRITERION_FAILED_EXTRA0', '2026-10-31 02:55 Asia/Shanghai', 'reweight_final reweight_acceptance reweight_supplement', common_parent=157505, each_updates=5000, each_exposures=300000, global_update=162505, each_checkpoints=13, control_psnr=[30.15297247,28.13339141,25.08305824,22.76327627,20.88218708], biased_psnr=[30.04724353,27.97340765,24.90353741,22.65574966,20.96989413], G_order=[.5,1,2,3,4], G4_biased_minus_control=.08770705, source_ci95=[.07842587,.09601288], biased_3k_to5k_G4_gain=.01862229, extension_required_recent_gain=.02, extension_pass=False, extra_updates=0, AB_resumed=False, control_sha256_from_original_receipt='50a7162ba7c821e5e91ef79197218405dea3743b9bda955a50120537bb1027fb', biased_sha256_from_original_receipt='2dd267d35428c92433576bd5fae78a1af60efcb0f3e5bda7cda4e29f9341ba7d'),
    branch('turbnet_candidate', 'asset_review', 'STOPPED_MISSING_OFFICIAL_WEIGHTS', '2026-10-28', 'turbnet', model_runs=0, optimizer_updates=0, performance_failure_established=False),
    branch('turbwnet_initial', 'candidate_asset_cpu_smoke', 'USER_STOP_AFTER_CPU_FORWARD_NOT_NPU_FULL_VAL', '2026-10-29', 'turbwnet', cpu_weighted_forward_pass=True, npu_full_val=False, optimizer_updates=0, later_separate_OTIS4328_completed=True),
    branch('OTIS_ATD_U95544', 'external_zero_shot', 'COMPLETE_OUTPUTS_METRICS_AND_EXIT', '2026-10-29', 'external95544 external95544_acceptance', OTIS_restores=5195, OTIS_mask_GT_frames=4328, ATD_noGT_frames=190, sequence_macro_input_psnr=10.372096, sequence_macro_model_psnr=8.515455, psnr_improved_frames=0, metric_protocol='gray_luma_mask_not_main_RGB_Val1000'),
    branch('official13_forgetting', 'external_zero_shot', '13_OF13_FULL_SUBSETS_COMPLETE', '2026-10-31 06:50 Asia/Shanghai', 'official13 official13_acceptance', paired_inputs=43321, restores=86642, UNet_calls=259926, training_updates=0, all13_D0_Y_PSNR_lower_and_RGB_LPIPS_worse=True, degradation_training_phase_causally_identified=False),
    branch('external6groups', 'external_zero_shot_delivery', 'SIX_GROUPS_AND_OFFLINE_PACKAGE_COMPLETE_BLOCKED_SOURCES_EXPLICIT', '2026-11-01 02:40 Asia/Shanghai', 'external6 external6_delivery', base_inputs=5222, extra_Door=3, three_D0_restores=15675, D0_UNet_calls=47025, additional_TurbWNet_OTIS=4328, files=760, complete_final_text400_OCR=False),
    branch('ATSyn999_best_final', 'external_zero_shot', 'FULL999_IDENTITY_OUTPUT_METRIC_EXIT_VISUAL_LOCAL_SYNC_ACCEPTED', '2026-11-01', 'external_final external_final_delivery atsyn_acceptance', original_test_scenes=1000, excluded_near_duplicate_scenes=1, evaluated=999, fixed_singleframe='turb24.png', paper_multiframe_exact_reproduction=False, PSNR=[21.38151965,21.47726880,21.58389958], SSIM=[.58283467,.59024354,.59554888], LPIPS=[.55090750,.52245069,.49099741], MAE=[.05836579,.05688721,.05629226], order=['Input','best109544','final157505'], final_text400_OCR=False),
    branch('diffusion_dependency', 'read_only_model_diagnostic', 'VAL50_DIAGNOSTIC_COMPLETE', '2026-10-27', 'dependency', independent_clean_sources=50, seeds_each=4, optimizer_updates=0, D0_C3_minus_C1_psnr=.223054, condition_only_strict_hypothesis_supported=False, generative_posterior_proved=False),
    branch('diffusion_mechanism', 'read_only_CPU_model_diagnostic', 'COMPLETED_WITH_OFFICIAL_FORWARD_INIT_DISTRIBUTION_LIMIT', '2026-10-28', 'mechanism mechanism_theory', clean_sources=12, models=3, NFE_compared=[1,2,3,4,6], optimizer_updates=0, new_NPU_calls=0, formal_train_code_changed=False, full_posterior_consistency_established=False),
    branch('train_val_generalization', 'read_only_model_diagnostic', 'FIXED_TRAIN300_VS_VAL250_COMPLETE_NOT_WHOLE_TRAIN', '2026-10-28', 'train_val', train_clean_sources=300, train_starts=1200, val_clean_sources=250, val_starts=1000, train_minus_val_psnr=-.140351, proves_no_overfit=False, proves_undertraining=False),
    branch('U95544_strength', 'read_only_model_diagnostic', 'FOUR_G_EACH1000_COMPLETE', '2026-10-29', 'strength_eval', optimizer_updates=0, new_restores=4000, G1_G2_PSNR_change_from_input=[-6.8537,-3.9315], rationale_for_multistrength_not_loss_bug_proof=True),
    branch('residual_train_val_loss', 'read_only_model_diagnostic', 'U95544_FULL_VAL_RESIDUAL_SINGLE_POINT_COMPLETE', '2026-10-29', 'loss_eval', optimizer_updates=0, val_starts=1000, train_last1000_raw_L1=.089182109, val_EMA_native_L1=.087682003, val_EMA_S3_clean_MAE=.055116700, full_historical_val_residual_curve_exists=False),
    branch('strength_preview_inventory', 'physics_only_and_asset_inventory', 'PREVIEW192_DEGRADATIONS_PLUS32_CLEAN_COMPLETE', '2026-10-29', 'strength_preview data_inventory', restoration_calls=0, training_updates=0, independent_sources=32, preview_G=[.25,.5,1,2,3,4], whole_val_quality=False),
    branch('post_meeting_audit', 'read_only_forensic', 'NO_CONFIRMED_NEW_BUG_IN_AUDITED_SCOPE_WITH_SCIENTIFIC_LIMITS', '2026-11-01', 'post_meeting global_curve', cached_observations_rehashed_in_original_audit=5000, physical_truth_validation=False),
    branch('candidate_research', 'literature_and_static_source_review', 'BOUNDED_SOURCE_METADATA_RESEARCH_COMPLETE_NO_MODEL_TEST', '2026-11-01', 'selection_v1 selection_v2 selection_v2_acceptance recent_ir recent_ir_acceptance nonres_expansion deblur_selection pixeldit latent_fit', candidate_model_runs=0, training_updates=0, checkpoint_payloads_downloaded_in_research=0, all_candidates_strict_loaded=False, priority_is_quality_ranking=False),
]


def digest(path):
    size = path.stat().st_size
    if size > MAX_SOURCE_BYTES:
        raise RuntimeError(f'Refuse large report input ({size} bytes): {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(key, rel):
    path = (PROJECT / rel).resolve()
    path.relative_to(PROJECT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in ('.md', '.json', '.csv', '.txt'):
        raise RuntimeError(f'Refuse payload file type: {path}')
    return dict(id=key, path=str(path.relative_to(PROJECT)),
                absolute_path=str(path), bytes=path.stat().st_size,
                sha256=digest(path), local_report_rehashed=True,
                remote_asset_revalidated=False)


def build(version, sources, branches):
    # Resolve every source before writing any report. A missing source fails closed.
    refs = [source_record(k, p) for k, p in sources.items()]
    for item in branches:
        assert item['sources'] and set(item['sources']) <= sources.keys(), item['id']
    document = HERE / f'{version}_history.md'
    data = dict(
        schema='project_handoff.history_evidence.v1',
        version=version,
        prepared_at_local=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        scope='Historical local evidence synthesis; not live remote/process/model revalidation',
        project_root=str(PROJECT),
        history_document=dict(path=str(document.relative_to(PROJECT)), sha256=digest(document), bytes=document.stat().st_size),
        generator=dict(path=str(Path(__file__).resolve().relative_to(PROJECT)), sha256=digest(Path(__file__))),
        execution=dict(network_calls=0, accelerator_calls=0, restoration_model_calls=0,
                       optimizer_updates=0, checkpoint_payload_loads=0,
                       source_asset_writes=0, temporary_or_final_training_started=False),
        checkpoint_SHA_policy='All checkpoint hashes in facts are recorded original acceptance values; payloads not rehashed by this preparation',
        historical_status_policy='Dates refer to source reports; old PID/RUNNING/resource snapshots are not current facts. Later final/stop receipts take precedence.',
        metric_policy='Keep source population, checkpoint/EMA, native sampler, strength and metric protocol distinct; no cross-protocol ranking or partial-to-full substitution',
        sources=refs, branches=branches,
        limits=['No current remote file/process/disk existence audit',
                'No independent recalculation of historical model predictions in this task',
                'No permission to resume old runs or delete retained/archive assets',
                'UniDB final experiment and full-project final handoff remain parent-task responsibilities'],
    )
    out = HERE / f'{version}_evidence.json'
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return dict(version=version, sources=len(refs), branches=len(branches), path=str(out), sha256=digest(out))


def verify(version):
    out = HERE / f'{version}_evidence.json'
    data = json.loads(out.read_text())
    checked = 0
    for item in data['sources'] + [data['history_document'], data['generator']]:
        path = PROJECT / item['path']
        assert path.is_file(), str(path)
        assert digest(path) == item['sha256'], str(path)
        checked += 1
    source_ids = {s['id'] for s in data['sources']}
    assert len(source_ids) == len(data['sources'])
    assert len({b['id'] for b in data['branches']}) == len(data['branches'])
    for b in data['branches']:
        assert set(b['sources']) <= source_ids, b['id']
    return dict(version=version, status='PASS', checked_report_files=checked,
                branches=len(data['branches']), evidence_sha256=digest(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        result = [verify(v) for v in ('v4', 'v5', 'v6')]
    else:
        result = [build('v4', V4_SOURCES, V4_BRANCHES),
                  build('v5', V5_SOURCES, V5_BRANCHES),
                  build('v6', V6_SOURCES, V6_BRANCHES)]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

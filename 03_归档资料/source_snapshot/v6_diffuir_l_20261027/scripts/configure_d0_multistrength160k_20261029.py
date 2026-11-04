"""Seal the complete executor only after the latest human steering is implemented."""
import ast,json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from scripts.runtime_utils import ROOT,sha,write_json
EX=ROOT/'execution_20261029_d0_multistrength160k'
def main():
    assert not (EX/'STAGE2_READY.json').exists()
    cfg={'stage':'D0','experiment_id':'V6_D0_U95544_resume_multistrength160k','updates':1000000000,
        'updates_field_is_technical_upper_bound_not_budget':True,'microbatch':5,'accumulation':3,'world_size':4,'effective_batch':60,
        'precision':'FP32','HF32':False,'amp':False,'gradient_checkpointing':False,'seed':737001,'ema_beta':.995,'global_clip':1.,
        'loss':'official_normalized_raw_residual_L1','residual_multiplier':1.,'clean_training_branch':False,
        'training_nfe_per_sample':1,'sampling_nfe':3,'native_diffusion_times':1000,'continuation_lr':2e-6,
        'physical_devices':[4,5,6,7],'training_counts_by_g':{str(g):40000 for g in [1,2,3,4]},'train_pairs':160000,
        'train_clean_sources':10000,'realizations_per_source':4,'validation_g':[.5,1,2,3,4],
        'checkpoint_every_new_updates':1000,'full_validation_every_new_updates':2000,
        'parent_update':95544,'old_data_cursor':5732640,'initial_new_manifest_cursor':0,
        'noon_review_local':'2026-10-30T12:00:00+08:00',
        'noon_review_unix':datetime(2026,10,30,12,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp(),
        'automatic_noon_stop':False,'training_deadline_unix':None,'stop_request':str(EX/'STOP_REQUEST.json'),
        'best_G4_PSNR_floor':21.16514236838766-.2,'best_scope':'five_g_equal_macro_with_G4_retention',
        'best_macro_PSNR_tie':.02,'test_used':False,'initialization_label':'joint_U14927_then_native_G4_U95544_full_state_to_multistrength'}
    write_json(EX/'CONFIG.json',cfg);write_json(EX/'CONFIG_MANIFEST.json',{'config_sha256':sha(EX/'CONFIG.json')})
    # All pre-existing scientific sources remain frozen, including upstream code.
    old=json.load(open(ROOT/'execution_20261029_d0_until18/SOURCE_MANIFEST.json'))['files']
    for n,h in old.items():assert sha(ROOT/n)==h,('Historical source drift',n)
    files=dict(old)
    names=['scripts/prepare_d0_multistrength_20261029.py','scripts/prepare_d0_multistrength160k_20261029.py',
        'scripts/generate_d0_multistrength160k_20261029.py','scripts/d0_multistrength160k_pipeline_20261029.py',
        'scripts/multistrength_support_20261029.py','scripts/multistrength_order_20261029.py',
        'scripts/run_d0_multistrength_20261029.py','scripts/audit_d0_multistrength_20261029.py',
        'scripts/configure_d0_multistrength160k_20261029.py','scripts/multistrength_gallery_20261029.py',
        'authority/USER_D0_MULTISTRENGTH160K_UPDATES_20261029.md',str((EX/'RUN_PLAN_ZH.md').relative_to(ROOT))]
    for n in names:
        if n.endswith('.py'):ast.parse((ROOT/n).read_text())
        files[n]=sha(ROOT/n)
    write_json(EX/'SOURCE_MANIFEST.json',{'files':files,'old_frozen_files_unchanged':len(old),'new_source_count':len(names),
        'scientific_core_unchanged':['method_impl/d0_official_residual.py','method_impl/d0.py','upstream/src/model.py','simulator_v5.py']})
    from scripts.multistrength_support_20261029 import select_best
    def candidate(step,g4,psnr,lpips):return {'update':step,'validation':{'by_g':{'4':{'psnr':g4}},'means':{'psnr':psnr,'lpips':lpips,'ssim':.5}}}
    parent=candidate(95544,21.16514236838766,18,.5)
    # A huge macro gain is ineligible if G4 retention fails; eligible gain is used.
    assert select_best([parent,candidate(97544,20.9,30,.1)])['update']==95544
    assert select_best([parent,candidate(97544,21.,19,.4)])['update']==97544
    assert select_best([parent,candidate(97544,21.,18.01,.4)])['update']==97544
    write_json(EX/'CPU_LOGIC_GATE.json',{'status':'PASS','G4_hard_retention_filter':True,'macro_PSNR_LPIPS_tie':True,
        'syntax_checked':True,'no_model_executed':True,'no_training_updates':True})
    policy_path=ROOT/'gates/EXECUTION_POLICY_CURRENT.json';old_policy=json.load(open(policy_path))
    write_json(EX/'POLICY_BEFORE_AUTHORIZED_TRANSITION.json',old_policy)
    policy={**old_policy,'status':'AUTHORIZED_D0_MULTISTRENGTH160K_PREPARATION',
        'authority':'authority/USER_D0_MULTISTRENGTH160K_UPDATES_20261029.md','active_execution':EX.name,
        'allow_d0_continuation':True,'auto_resume_forbidden':False,'allow_ab_formal_training':False,
        'allow_ab_benchmark':False,'allow_ab_pilot':False,'stop_request':str(EX/'STOP_REQUEST.json'),
        'deadline_unix':None,'normal_training_not_before_unix':None,'automatic_noon_stop':False,
        'noon_review_unix':cfg['noon_review_unix'],'checkpoint_every':1000,'full_validation_every':2000,
        'training_counts_by_g':cfg['training_counts_by_g'],'train_pairs':160000,'old_stop_requests_preserved':True}
    write_json(policy_path,policy)
    write_json(EX/'STAGE2_READY.json',{'status':'READY_FOR_ACTUAL_GATES_NOT_PREPASSED','unix':time.time(),
        'config_sha256':sha(EX/'CONFIG.json'),'source_manifest_sha256':sha(EX/'SOURCE_MANIFEST.json'),
        'human_authorized_automatic_progression':True,'scientific_gates_not_yet_executed_not_marked_PASS':True})
    print('Executor sealed; waiting for actual generation/data/resume gates',flush=True)
if __name__=='__main__':main()

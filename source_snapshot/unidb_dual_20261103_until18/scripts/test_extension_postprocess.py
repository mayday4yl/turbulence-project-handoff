"""Pure CPU fixture/AST tests; no torch, restoration/metric network or optimizer.
Artifacts remain under the new campaign's evidence/postprocess_cpu_fixture.
"""
import ast,base64,copy,hashlib,json,math,os,shutil,subprocess,sys,time
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
REAL=Path(__file__).resolve().parents[1]
BASE=REAL/'evidence/postprocess_cpu_fixture'
if BASE.exists():
 assert BASE.resolve().is_relative_to((REAL/'evidence').resolve())
 shutil.rmtree(BASE)
BASE.mkdir(parents=True,exist_ok=True)
ROOT=BASE/'unidb_dual_20261103_until18';PREVIOUS=BASE/'unidb_dual_20261103_until_noon'
for root in [ROOT,PREVIOUS]:
 for sub in ['scripts','protocol','evidence','runs/original/validation','runs/improved/validation','gates','figures','report']:(root/sub).mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(q,indent=2)+'\n')
def jsonl(p,rr):Path(p).write_text(''.join(json.dumps(q)+'\n' for q in rr))
checks=[]
def check(label,ok):assert ok,label;checks.append(label)
def rejected(label,fn):
 try:fn()
 except (AssertionError,FileNotFoundError,KeyError):checks.append(label);return
 raise AssertionError('Expected rejection '+label)

def functions(path,ns):
 tree=ast.parse(Path(path).read_text())
 for node in tree.body:
  if isinstance(node,ast.FunctionDef):exec(compile(ast.Module([node],type_ignores=[]),str(path),'exec'),ns)
 return ns

authority={'deadline_unix':1793700000,'previous_deadline_unix':1793678400,'previous_campaign_root':str(PREVIOUS),
 'extension_parents':{},'allocation':{'original_continue':[0,1,2],'improved':[3,4,5,6,7]}}
metrics=lambda v:{'psnr':v,'ssim':.5,'mae':.1,'lpips':.4}
def summary(branch,u,full=True,value=20):
 return {'status':'PASS','branch':branch,'update':u,'weights':'EMA','full':full,'count':5000 if full else 200,
  'per_g':1000 if full else 100,'sources_per_g':250 if full else 25,'image_clip_crop_resize':False,
  'parameter_digest':f'EMA-{branch}-{u}','by_g':{g:metrics(value) for g in (['0.5','1','2','3','4'] if full else ['1','4'])}}
for branch,start,best_u in [('original',55760,55411),('improved',20000,10000)]:
 base=45411 if branch=='original' else 0
 authority['extension_parents'][branch]={'path':str(PREVIOUS/'runs'/branch/'checkpoints'/f'U{start:08d}.pt'),
  'sha256':'parent-sha','update':start,'cursor':96*start}
 def row(u,epoch):
  started=1793678300.+u%10 if epoch=='old' else 1793678301.+(49.22606926759084 if branch=='original' else 55.025858016808826)*60+(u-start-1)
  return {'branch':branch,'update':u,'new_updates':u-base,'cursor':96*u,'loss':.1,'lr':1e-5 if branch=='original' else 2e-7,
   'seconds':1.,'started_unix':started,'completed_unix':started+1,'grad_norm':1.,'grad_norm_after_clip':1.}
 jsonl(PREVIOUS/'runs'/branch/'UPDATES_rank0.jsonl',[row(start-1,'old'),row(start,'old')])
 jsonl(ROOT/'runs'/branch/'UPDATES_rank0.jsonl',[row(u,'new') for u in range(start+1,start+4)])
 for root,u in [(PREVIOUS,best_u),(ROOT,start+3)]:
  cp=root/'runs'/branch/'checkpoints'/f'U{u:08d}.pt';cp.parent.mkdir(exist_ok=True);cp.write_bytes(f'CPU-checkpoint-fixture-{branch}-{u}'.encode())
  ident={'path':str(cp),'sha256':sha(cp),'update':u,'cursor':96*u}
  if root==PREVIOUS:
   q=summary(branch,u,value=22);write(root/'runs'/branch/'validation'/f'U{u:08d}_ema_full'/'SUMMARY.json',q)
   best={'full':True,'weights':'EMA','selection_role':'full_validation','update':u,'checkpoint':ident,'summary':q}
   write(ROOT/'runs'/branch/'BEST_FULL.json',best);write(ROOT/'runs'/branch/'BEST.json',best)
   link=ROOT/'runs'/branch/'validation'/f'U{u:08d}_ema_full'
   if not link.exists():link.symlink_to(root/'runs'/branch/'validation'/link.name,target_is_directory=True)
   write(ROOT/'evidence'/f'STATE_AUDIT_{branch}_BEST.json',{'status':'PASS','checkpoint':{**ident,'EMA':{'full_parameter_digest':q['parameter_digest']}}})
  else:
   write(ROOT/'runs'/branch/'LATEST.json',ident)
   write(ROOT/'runs'/branch/'STATUS.json',{'status':'TRAINING','update':u,'new_updates':u-base,'final_full_validation_done':False})
   write(ROOT/'evidence'/f'STATE_AUDIT_{branch}_FINAL.json',{'status':'PASS','checkpoint':{**ident,'EMA':{'full_parameter_digest':f'EMA-{branch}-{u}'}}})
 # Incomplete frozen noon-parent full directories exist but have no SUMMARY.
 partial=PREVIOUS/'runs'/branch/'validation'/f'U{start:08d}_ema_full';partial.mkdir(exist_ok=True)
 write(partial/'PROGRESS_rank0.json',{'groups':10})
 write(ROOT/'runs'/branch/'CONFIG.json',{'branch':branch,'phase':'formal'})
 for epoch,root in [('until_noon',PREVIOUS),('until18_extension',ROOT)]:
  for suffix,n in [('initial',2 if epoch=='until_noon' else 1),('restart',1)]:
   folder=root/'gates'/(branch+'_'+suffix);folder.mkdir(exist_ok=True)
   write(folder/'RESULT.json',{'status':'PASS','branch':branch,'ranks':[{'completed_updates_this_process':n,'world':3 if branch=='original' else 5}]})
write(ROOT/'protocol/USER_AUTHORITY.json',authority)
write(ROOT/'protocol/MONITOR_SELECTION.json',{'sources':['source0','source1']})
write(ROOT/'protocol/SHARED_CONTEXTS.json',{'permitted_contexts':[{'card':0,'kind':'observed_external_training_context'},{'card':2,'kind':'observed_external_training_context'}]})
historical_validation=PREVIOUS/'runs/improved/validation/U00010000_ema_full'
historical_record={'status':'PASS','signature':'historical-U10000-signature','validation_dir':str(historical_validation),
 'summary_sha256':sha(historical_validation/'SUMMARY.json'),'CPU_metric_forward_batches':1250,
 'CPU_metric_forward_batches_this_invocation':1250,'reused_completed_audit':False}
historical_aggregate={'status':'PASS','protocol':{'auditor_sha256':'legacy'},'validations':[historical_record,copy.deepcopy(historical_record)],
 'CPU_metric_model_constructions':1,'CPU_metric_forward_batches_this_invocation':1250,
 'audited_summaries':{str(historical_validation/'SUMMARY.json'):historical_record['summary_sha256']}}
write(PREVIOUS/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',historical_aggregate)
write(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',historical_aggregate)
shutil.copy2(REAL/'scripts/audit_campaign_saved_outputs.py',ROOT/'scripts/audit_campaign_saved_outputs.py')

ns={'argparse':__import__('argparse'),'Path':Path,'hashlib':hashlib,'json':json,'os':os,'sys':sys,'time':time,'math':math,'np':np,'csv':__import__('csv'),
 'defaultdict':defaultdict,'datetime':datetime,'timedelta':timedelta,'timezone':timezone,
 'ROOT':ROOT,'PREVIOUS':PREVIOUS,'OLD':BASE/'d0','OUT':ROOT/'report','FIG':ROOT/'figures','INPUTS':{},'WARNINGS':[],
 'BRANCHES':['original','improved'],'COLORS':{'original':'blue','improved':'orange'},'METRICS':['psnr','ssim','mae','lpips'],
 'GS':['0.5','1','2','3','4'],'DEADLINE':1793700000,'__file__':str(REAL/'scripts/make_campaign_report.py')}
functions(REAL/'scripts/make_campaign_report.py',ns)
for b in ['original','improved']:
 line=ns['collect_branch'](b);start=authority['extension_parents'][b]['update']
 check(b+'_merge_five_log_rows',len(line['training'])==5)
 check(b+'_retained_count_not_row_count',line['retained_update']==start+3 and line['retained_update']-line['base']!=len(line['training']))
 check(b+'_symlink_old_full_deduplicated',len(line['validations'])==1)
 check(b+'_old_full_can_remain_best',line['best_full_summary']['campaign_stage']=='until_noon')
 check(b+'_partial_noon_parent_is_not_full_candidate',all(q['update']!=start for q in line['validations']))
 check(b+'_no_latest_full_from_partial',line['latest_full_summary'] is None)
gates=ns['engineering_work']()
check('extension_four_gates_four_updates_384_presentations',sum(r['completed_updates_this_gate_process'] for r in gates if r['campaign_stage']=='until18_extension')==4 and sum(r['recorded_presentations'] for r in gates if r['campaign_stage']=='until18_extension')==384)
check('old_gate_work_separate',sum(r['completed_updates_this_gate_process'] for r in gates if r['campaign_stage']=='until_noon')==6)
# Actual main runs with CPU-only placeholder rendering in this isolated fixture;
# this tests report control/data flow, not figure appearance.
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
ns['fixed_selection']=lambda:({'status':'PENDING','sources':['source0','source1']},[])
ns['old_baselines']=lambda sources:({}, {}, [], {})
def render(*args):
 for name in ['CAMPAIGN_DYNAMICS.png','CAMPAIGN_DYNAMICS.pdf']:(ROOT/'figures'/name).write_bytes(PNG)
def fixed(role,*args):
 for ext in ['png','pdf']:(ROOT/'figures'/('FIXED_G4_'+role.upper()+'.'+ext)).write_bytes(PNG)
 return {'role':role,'selected_validation':{},'assets':[]}
ns['training_curves']=render;ns['fixed_grid']=fixed
oldargv=sys.argv[:];sys.argv=['make_campaign_report.py','--final']
try:
 try:ns['main']()
 except SystemExit as exc:check('final_report_refuses_unfinished_pipeline_exit2',exc.code==2)
 else:raise AssertionError('--final accepted unfinished pipeline')
finally:sys.argv=oldargv
status=json.loads((ROOT/'report/REPORT_STATUS.json').read_text())
check('final_requirement_false_no_final_report',not status['final_requirements_pass'] and not (ROOT/'report/FINAL_REPORT_ZH.md').exists())
check('campaign_vs_extension_counts_explicit',all(r['extension_new_updates']==3 and r['committed_new_updates']==authority['extension_parents'][r['branch']]['update']-r['initial_update']+3 for r in status['branches']))
check('canonical_summary_keys_in_required_manifest',all(str(Path(p).resolve())==p for p in json.loads((ROOT/'report/REPORT_MANIFEST.json').read_text())['required_final_summary_sha256']))
check('handoff_update_intervals_not_idle',all(abs(r['recorded_update_gap_minutes']-({'original':49.22606926759084,'improved':55.025858016808826}[r['branch']]))<1e-8 for r in status['branches']))
check('completed_log_U_and_saved_checkpoint_U_separate_fields',all('last_completed_log_update' in r and 'latest_checkpoint_update' in r for r in status['branches']))
cp=status['cpu_cost_provenance']
check('copied_historical_aggregate_not_current_invocation',cp['copied_historical_aggregate_not_new_invocation'] and not cp['current']['invocation_receipt_present'])
check('historical_signature_dedup_1250_not_2500',cp['historical']['unique_validation_signatures']==1 and cp['historical']['CPU_metric_forward_batches']==1250)
check('reporter_zero_metric_restoration_model_calls',cp['reporter']['CPU_metric_model_constructions']==0 and cp['reporter']['reporter_metric_model_calls']==cp['reporter']['reporter_restoration_model_calls']==0)
body=(ROOT/'report/STAGE_REPORT_ZH.md').read_text()
check('report_shared_context_and_gap_scope_disclosed','卡0、2' in body and '不能当作纯空闲' in body and '具体百分比' in body)
check('stage_binding_worded_as_final_requirement','最终要求每条指针' in body and '当前是否满足以pending' in body)
check('copied1250_never_called_current_final_audit','最终CPU指标后处理：' not in body and '历史已完成CPU审计来源' in body and '当前尚无新extension wrapper' in body)
# A real new-wrapper receipt reuses the historical signature with zero fresh
# forward work; duplicate new signatures never double the reported cost.
current_record={**historical_record,'CPU_metric_forward_batches_this_invocation':0,'reused_completed_audit':True}
new_record={**historical_record,'signature':'new-final-signature','CPU_metric_forward_batches_this_invocation':1250,
 'validation_dir':str(ROOT/'runs/improved/validation/U00030000_ema_full'),'summary_sha256':'new-summary'}
new_aggregate={'status':'PASS','protocol':{'wrapper_sha256':sha(ROOT/'scripts/audit_campaign_saved_outputs.py'),'out_root':str(ROOT)},
 'validations':[current_record,new_record,copy.deepcopy(new_record)],'CPU_metric_model_constructions':1,'CPU_metric_forward_batches_this_invocation':1250}
write(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',new_aggregate)
current=ns['cpu_cost_provenance'](ns['js'](ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json'))
check('current_new_wrapper_invocation_distinct_from_history',current['current']['invocation_receipt_present'])
check('current_signature_dedup_and_cache_zero_fresh_forward',current['current']['unique_validation_signatures']==2 and current['current']['CPU_metric_forward_batches']==1250 and current['current']['cache_reused_unique_signatures']==1)
check('cached_reference_1250_not_added_to_current_cost',next(r for r in current['validation_rows'] if r['validation_signature']=='historical-U10000-signature')['current_invocation_metric_forward_batches']==0)
old_source=REAL/'evidence/report_provenance_repair_20261103/make_campaign_report.py'
if old_source.exists():
 old_text=old_source.read_text();new_text=(REAL/'scripts/make_campaign_report.py').read_text()
 for name in ['summary','fixed_selection','prediction_rows','old_baselines','training_curves','fixed_grid']:
  node=lambda source:ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  check(name+'_unchanged_by_provenance_repair',node(old_text)==node(new_text))
 start=' required={};pending=[]';end=' ready=not pending'
 old_gate=old_text[old_text.index(start):old_text.index(end)]
 new_gate=new_text[new_text.index(start):new_text.index(end)]
 new_gate=new_gate[:new_gate.index(' cpu_provenance=cpu_cost_provenance(cpu)')]
 check('final_CPU_state_and_summary_gate_unchanged',old_gate==new_gate)

# Regression: root-local monitor markers must not erase a stronger historical
# monitor; the existing full-best/image choice must remain untouched.
write(PREVIOUS/'runs/improved/validation/U00003000_ema_monitor/SUMMARY.json',summary('improved',3000,full=False,value=18.61586053227808))
write(ROOT/'runs/improved/validation/U00021000_ema_monitor/SUMMARY.json',summary('improved',21000,full=False,value=13.7777))
write(PREVIOUS/'runs/improved/BEST_MONITOR.json',{'update':3000})
write(ROOT/'runs/improved/BEST_MONITOR.json',{'update':21000})
monitor_regression=ns['collect_branch']('improved')
check('historical_U3000_monitor_beats_extension_U21000_without_changing_full_or_image_role',
 monitor_regression['campaign_monitor_best_summary']['update']==3000 and monitor_regression['extension_monitor_best']['update']==21000
 and monitor_regression['extension_monitor_best_summary']['update']==21000
 and monitor_regression['best_full_summary']['update']==monitor_regression['best_image_summary']['update']==10000)
finalns={'argparse':__import__('argparse'),'Path':Path,'hashlib':hashlib,'json':json,'ROOT':ROOT,'DEADLINE':1793700000}
functions(REAL/'scripts/finalize_campaign.py',finalns)
rejected('finalizer_collect_rejects_training_not_complete',finalns['collect'])
for b in ['original','improved']:
 u=authority['extension_parents'][b]['update']+3;base=45411 if b=='original' else 0
 write(ROOT/'runs'/b/'STATUS.json',{'status':'COMPLETE','update':u,'new_updates':u-base,'extension_new_updates':3,'extension_new_presentations':288,'final_full_validation_done':True})
 write(ROOT/'runs'/b/'validation'/f'U{u:08d}_ema_full'/'SUMMARY.json',summary(b,u,value=19))
refs,updates=finalns['collect']()
check('finalizer_accepts_old_best_new_latest_canonical_refs',len(refs)==4 and sum(r['campaign_stage']=='until_noon' for r in refs)==2 and all(str(Path(r['summary']).resolve())==r['summary'] for r in refs))
check('finalizer_extension_counts_not_cumulative',all(r['extension_new_updates']==3 for r in updates.values()))
# Execute the real wrapper self-test through a byte-identical legacy source copy.
legacy=REAL.parent/'unidb_dual_20261103_until_noon/scripts/audit_saved_outputs.py'
shutil.copy2(legacy,PREVIOUS/'scripts/audit_saved_outputs.py')
wrapper=REAL/'scripts/audit_campaign_saved_outputs.py';shutil.copy2(wrapper,ROOT/'scripts/audit_campaign_saved_outputs.py')
result=subprocess.run([sys.executable,str(ROOT/'scripts/audit_campaign_saved_outputs.py'),'--self-test','--out',str(ROOT/'evidence/WRAPPER_SELFTEST.json')],capture_output=True,text=True)
check('wrapper_real_primary_algorithm_selftest_zero_models',result.returncode==0)
w=json.loads((ROOT/'evidence/WRAPPER_SELFTEST.json').read_text())
check('wrapper_legacy_SHA_and_zero_model_calls',w['legacy_auditor_sha256']=='a9c684bf42fdab2a1cf34670d717d7d3d1e4dc21bccc5b7f63bc18beaddf3950' and w['metric_or_restoration_models_constructed']==0)
report={'status':'PASS','checks':checks,'count':len(checks),'scope':'AST bookkeeping and isolated CPU fixtures; no torch, restoration or metric network, real optimizer or accelerator. Plot appearance requires actual matplotlib rendering/root QA.',
 'sources':{name:sha(REAL/'scripts'/name) for name in ['make_campaign_report.py','finalize_campaign.py','audit_campaign_saved_outputs.py']},
 'legacy_auditor_sha256':sha(legacy),'NPU_calls':0,'restoration_network_calls':0,'metric_network_calls':0,'optimizer_updates':0}
write(REAL/'evidence/EXTENSION_POSTPROCESS_CPU_TESTS.json',report)
print(json.dumps(report,indent=2))

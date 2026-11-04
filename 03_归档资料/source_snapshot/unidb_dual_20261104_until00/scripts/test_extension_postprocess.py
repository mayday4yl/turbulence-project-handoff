"""Pure stdlib, isolated three-stage bookkeeping/state-binding regression tests.
No imports of torch, numpy, restoration/metric networks; no optimizer/accelerator.
Actual matplotlib rendering and original auditor self-test are separate live checks.
"""
import ast,argparse,base64,copy,csv,hashlib,json,math,os,shutil,sys,time
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
REAL=Path(__file__).resolve().parents[1];BASE=REAL/'evidence/postprocess_cpu_fixture'
if BASE.exists():
 assert BASE.resolve().is_relative_to((REAL/'evidence').resolve());shutil.rmtree(BASE)
roots=[BASE/n for n in ['unidb_dual_20261103_until_noon','unidb_dual_20261103_until18','unidb_dual_20261104_until00']]
NOON,PREVIOUS,ROOT=roots
for r in roots:
 for d in ['scripts','protocol','evidence','runs/original/validation','runs/improved/validation','gates','figures','report']:(r/d).mkdir(parents=True,exist_ok=True)
def write(p,q):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(q,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(q)+'\n' for q in rows))
def functions(path,ns):
 for n in ast.parse(Path(path).read_text()).body:
  if isinstance(n,ast.FunctionDef):exec(compile(ast.Module([n],type_ignores=[]),str(path),'exec'),ns)
 return ns
checks=[]
def check(label,ok):assert ok,label;checks.append(label)
def reject(label,fn):
 try:fn()
 except (AssertionError,FileNotFoundError,KeyError):checks.append(label);return
 raise AssertionError('Expected rejection '+label)
auths=[]
for i,r in enumerate(roots):
 a={'deadline_unix':[1793678400,1793700000,1793721600][i],'previous_campaign_root':str(roots[max(0,i-1)]),'history_campaign_roots':[str(x) for x in roots[:i]],'extension_parents':{},'allocation':{'original_continue':[0,1,2],'improved':[3,4,5,6,7]}}
 auths.append(a)
def summary(b,u,v=20,full=True):
 return {'status':'PASS','branch':b,'update':u,'weights':'EMA','full':full,'count':5000 if full else 200,'per_g':1000 if full else 100,'sources_per_g':250 if full else 25,'image_clip_crop_resize':False,'parameter_digest':f'{b}-{u}','by_g':{g:{'psnr':v,'ssim':.5,'mae':.1,'lpips':.4} for g in (['0.5','1','2','3','4'] if full else ['1','4'])}}
idents={}
for b,base in [('original',45411),('improved',0)]:
 idents[b]={}
 for i,r in enumerate(roots):
  start=base+[0,3,5][i];end=base+[3,5,7][i]
  auths[i]['extension_parents'][b]={'path':str(roots[max(0,i-1)]/'runs'/b/'checkpoints'/f'U{start:08d}.pt'),'sha256':'parent','update':start,'cursor':96*start}
  rr=[{'branch':b,'update':u,'new_updates':u-base,'cursor':96*u,'loss':.1,'lr':1e-5 if b=='original' else 2e-7,'seconds':1.,'started_unix':1793678000+i*100+u-base,'completed_unix':1793678001+i*100+u-base,'grad_norm':1.,'grad_norm_after_clip':1.} for u in range(start+1,end+1)]
  jsonl(r/'runs'/b/'UPDATES_rank0.jsonl',rr)
  for suffix,n in [('initial',2 if i==0 else 1),('restart',1)]:
   write(r/'gates'/(b+'_'+suffix)/'RESULT.json',{'status':'PASS','branch':b,'ranks':[{'completed_updates_this_process':n,'world':3 if b=='original' else 5}]})
  u=base+[1,4,7][i];cp=r/'runs'/b/'checkpoints'/f'U{u:08d}.pt';cp.parent.mkdir(exist_ok=True);cp.write_bytes(f'{b}-{u}'.encode());ident={'path':str(cp),'sha256':sha(cp),'update':u,'cursor':96*u};idents[b][i]=ident
  if i<2:write(r/'runs'/b/'validation'/f'U{u:08d}_ema_full'/'SUMMARY.json',summary(b,u,22-i))
  if i==0:
   best={'full':True,'weights':'EMA','selection_role':'full_validation','update':u,'checkpoint':ident}
   for target in roots[1:]:
    write(target/'runs'/b/'BEST_FULL.json',best);write(target/'runs'/b/'BEST.json',best)
    (target/'runs'/b/'validation'/f'U{u:08d}_ema_full').symlink_to(r/'runs'/b/'validation'/f'U{u:08d}_ema_full',target_is_directory=True)
  if i==1:
   (ROOT/'runs'/b/'validation'/f'U{u:08d}_ema_full').symlink_to(r/'runs'/b/'validation'/f'U{u:08d}_ema_full',target_is_directory=True)
   write(r/'runs'/b/'validation'/f'U{end:08d}_ema_full'/'PROGRESS_rank0.json',{'groups':4})
  if i==2:
   write(r/'runs'/b/'CONFIG.json',{'branch':b,'phase':'formal'});write(r/'runs'/b/'LATEST.json',ident);write(r/'runs'/b/'STATUS.json',{'status':'TRAINING','update':u,'new_updates':u-base})
   for role,k in [('BEST',0),('FINAL',2)]:
    bind=idents[b][k];write(ROOT/'evidence'/f'STATE_AUDIT_{b}_{role}.json',{'status':'PASS','checkpoint':{**bind,'EMA':{'full_parameter_digest':f'{b}-{bind["update"]}'}}})
for r,a in zip(roots,auths):write(r/'protocol/USER_AUTHORITY.json',a)
write(ROOT/'protocol/MONITOR_SELECTION.json',{'sources':['s0','s1']});write(ROOT/'protocol/SHARED_CONTEXTS.json',{'permitted_contexts':[{'card':0},{'card':2}]})
for r,u,v in [(NOON,2,18.61586),(PREVIOUS,5,13.77),(ROOT,6,13.75)]:write(r/'runs/improved/validation'/f'U{u:08d}_ema_monitor'/'SUMMARY.json',summary('improved',u,v,False))
write(ROOT/'runs/improved/BEST_MONITOR.json',{'update':6})
for n in ['make_campaign_report.py','audit_campaign_saved_outputs.py','finalize_campaign.py']:shutil.copy2(REAL/'scripts'/n,ROOT/'scripts'/n)
ns={'argparse':argparse,'Path':Path,'hashlib':hashlib,'json':json,'os':os,'sys':sys,'time':time,'math':math,'csv':csv,'defaultdict':defaultdict,'datetime':datetime,'timedelta':timedelta,'timezone':timezone,'ROOT':ROOT,'PREVIOUS':PREVIOUS,'OLD':BASE/'d0','OUT':ROOT/'report','FIG':ROOT/'figures','INPUTS':{},'WARNINGS':[],'BRANCHES':['original','improved'],'COLORS':{'original':'blue','improved':'orange'},'METRICS':['psnr','ssim','mae','lpips'],'GS':['0.5','1','2','3','4'],'DEADLINE':1793721600,'__file__':str(REAL/'scripts/make_campaign_report.py')}
functions(REAL/'scripts/make_campaign_report.py',ns)
for b in ['original','improved']:
 q=ns['collect_branch'](b);check(b+'_all_three_stages_seven_unique_committed_updates',len(q['training'])==7 and len({x['update'] for x in q['training']})==7)
 check(b+'_canonical_full_symlinks_deduplicated',len([x for x in q['validations'] if x['full']])==2)
 check(b+'_historical_best_preserved',q['best_full_summary']['campaign_stage']=='until_noon')
 check(b+'_partial_direct_parent_excluded',all(x['update']!=auths[2]['extension_parents'][b]['update'] or not x['full'] for x in q['validations']))
 check(b+'_no_latest_full_before_completion',q['latest_full_summary'] is None)
 check(b+'_latest_handoff_gap_not_first_handoff',ns['update_handoff_gap'](q['training'])['prior_last_completed_update']==auths[2]['extension_parents'][b]['update'])
check('historical_monitor_winner_and_current_marker_separate',ns['collect_branch']('improved')['campaign_monitor_best_summary']['update']==2 and ns['collect_branch']('improved')['extension_monitor_best']['update']==6)
check('both_handoff_intervals_per_route_preserved',len(ns['handoff_intervals']({b:ns['collect_branch'](b) for b in ['original','improved']}))==4)
g=ns['engineering_work']();check('three_stage_gate_work_6_plus4_plus4_not_copied',sum(x['completed_updates_this_gate_process'] for x in g)==14 and sum(x['completed_updates_this_gate_process'] for x in g if x['campaign_stage']=='until00_extension')==4)
# Duplicate same aggregate copied into the18 root must not add historical cost.
histpath=NOON/'runs/improved/validation/U00000001_ema_full';h={'status':'PASS','signature':'old','validation_dir':str(histpath),'summary_sha256':sha(histpath/'SUMMARY.json'),'CPU_metric_forward_batches':1250,'CPU_metric_forward_batches_this_invocation':1250,'reused_completed_audit':False}
history={'status':'PASS','protocol':{'auditor_sha256':'legacy'},'validations':[h,copy.deepcopy(h)],'CPU_metric_model_constructions':1}
for r in roots:write(r/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',history)
p=ns['cpu_cost_provenance'](ns['js'](ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json'))
check('copied_historical_aggregate_not_new_invocation',p['copied_historical_aggregate_not_new_invocation'] and not p['current']['invocation_receipt_present'])
check('noon_copy_at18_cost_dedup_signature_and_receipt',p['historical']['unique_validation_signatures']==1 and p['historical']['CPU_metric_forward_batches']==1250 and p['historical']['CPU_metric_model_constructions']==1)
# A genuine18-stage invocation may have one cache hit and one new validation;
# count real fresh work once, not cached reference totals or a copied receipt.
prior_new={**h,'signature':'stage18-new','validation_dir':str(PREVIOUS/'runs/improved/validation/U00000004_ema_full'),'summary_sha256':'stage18-summary','CPU_metric_forward_batches_this_invocation':1250}
prior_receipt={'status':'PASS','protocol':{'out_root':str(PREVIOUS)},'validations':[{**h,'CPU_metric_forward_batches_this_invocation':0,'reused_completed_audit':True},prior_new,copy.deepcopy(prior_new)],'CPU_metric_model_constructions':1}
write(PREVIOUS/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',prior_receipt)
p=ns['cpu_cost_provenance'](ns['js'](ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json'))
check('genuine18_receipt_adds_only_its_fresh_work_and_one_model',p['historical']['unique_validation_signatures']==2 and p['historical']['CPU_metric_forward_batches']==2500 and p['historical']['CPU_metric_model_constructions']==2)
new={**h,'signature':'new','CPU_metric_forward_batches_this_invocation':1250,'validation_dir':str(ROOT/'runs/improved/validation/U00000007_ema_full'),'summary_sha256':'new'}
receipt={'status':'PASS','protocol':{'wrapper_sha256':sha(ROOT/'scripts/audit_campaign_saved_outputs.py'),'out_root':str(ROOT)},'validations':[{**h,'CPU_metric_forward_batches_this_invocation':0,'reused_completed_audit':True},new,copy.deepcopy(new)],'CPU_metric_model_constructions':1}
write(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json',receipt);p=ns['cpu_cost_provenance'](ns['js'](ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json'))
check('current_receipt_dedup_and_cache_zero_new_forwards',p['current']['unique_validation_signatures']==2 and p['current']['CPU_metric_forward_batches']==1250 and p['current']['cache_reused_unique_signatures']==1)
# Exercise real report control flow using placeholder bytes, not actual plotting.
ns['fixed_selection']=lambda:({'status':'PENDING','sources':['s0','s1']},[]);ns['old_baselines']=lambda sources:({}, {}, [], {})
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
def curves(*args):
 for e in ['png','pdf']:(ROOT/'figures'/('CAMPAIGN_DYNAMICS.'+e)).write_bytes(PNG)
def fixed(role,*args):
 for e in ['png','pdf']:(ROOT/'figures'/('FIXED_G4_'+role.upper()+'.'+e)).write_bytes(PNG)
 return {'role':role,'selected_validation':{},'assets':[]}
ns['training_curves']=curves;ns['fixed_grid']=fixed
argv=sys.argv[:];sys.argv=['make_campaign_report.py','--final']
try:
 try:ns['main']()
 except SystemExit as e:check('unfinished_final_actual_exit2',e.code==2)
 else:raise AssertionError('unfinished --final was accepted')
finally:sys.argv=argv
q=json.loads((ROOT/'report/REPORT_STATUS.json').read_text());check('stage_current_segment_two_updates_campaign_seven',all(x['committed_new_updates']==7 and x['extension_new_updates']==2 for x in q['branches']))
check('stage_three_roots_report_no_final_claim',q['status']=='IN_PROGRESS' and not (ROOT/'report/FINAL_REPORT_ZH.md').exists())
fn={'argparse':argparse,'Path':Path,'hashlib':hashlib,'json':json,'ROOT':ROOT,'DEADLINE':1793721600};functions(REAL/'scripts/finalize_campaign.py',fn)
reject('finalizer_rejects_live_training',fn['collect'])
for b,base in [('original',45411),('improved',0)]:
 u=base+7;write(ROOT/'runs'/b/'STATUS.json',{'status':'COMPLETE','update':u,'new_updates':7,'extension_new_updates':2,'extension_new_presentations':192,'final_full_validation_done':True});write(ROOT/'runs'/b/'validation'/f'U{u:08d}_ema_full'/'SUMMARY.json',summary(b,u,19))
refs,updates=fn['collect']();check('finalizer_accepts_oldest_best_and_new_latest_binding',len(refs)==4 and len({r['summary'] for r in refs})==4)
check('finalizer_current_extension_counts_not_whole_campaign',all(x['campaign_new_updates']==7 and x['extension_new_updates']==2 for x in updates.values()))
bindpath=ROOT/'evidence/STATE_AUDIT_improved_FINAL.json';bind=json.loads(bindpath.read_text());broken=copy.deepcopy(bind);broken['checkpoint']['EMA']['full_parameter_digest']='wrong';write(bindpath,broken);reject('finalizer_rejects_parameter_digest_mismatch',fn['collect']);write(bindpath,bind)
# Wrapper approval and cache bookkeeping: unchanged real auditor revalidates allSHA.
wn={'Path':Path,'json':json,'ROOT':ROOT,'hashlib':hashlib,'shutil':shutil};functions(REAL/'scripts/audit_campaign_saved_outputs.py',wn)
own,approved=wn['approved_roots']();check('wrapper_all_three_roots_authorized',own==ROOT.resolve() and approved==[NOON.resolve(),PREVIOUS.resolve()])
val=NOON/'runs/improved/validation/U00000001_ema_full';prefix=ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT_fixture';csvpath=NOON/'evidence/cache.csv';csvpath.write_text('fixture')
cache={**h,'per_image_CSV':str(csvpath),'per_image_CSV_sha256':sha(csvpath)}
for r in [NOON,PREVIOUS]:write(r/'evidence'/(prefix.name+'_RESULT.json'),cache)
wn['seed_completed_cache'](val.resolve(),prefix,approved);check('wrapper_seed_history_cache_result_only_inside_newroot',prefix.with_name(prefix.name+'_RESULT.json').is_file())
# Methodology functions remain byte-AST equivalent to frozen18 report.
old=REAL.parent/'unidb_dual_20261103_until18/scripts/make_campaign_report.py';oldtree=ast.parse(old.read_text());newtree=ast.parse((REAL/'scripts/make_campaign_report.py').read_text())
for name in ['summary','fixed_selection','prediction_rows','old_baselines','load_asset','fixed_grid']:
 node=lambda tr:ast.dump(next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
 check(name+'_scientific_population_or_fixed_selection_unchanged',node(oldtree)==node(newtree))
report={'status':'PASS','count':len(checks),'checks':checks,'scope':'stdlib AST extraction, three-stage isolated fixtures, actual report --final rejection, canonical dedup, source binding; placeholder image bytes are not visual QA. No numpy/torch/model/optimizer/accelerator imports.','sources':{n:sha(REAL/'scripts'/n) for n in ['make_campaign_report.py','finalize_campaign.py','audit_campaign_saved_outputs.py']},'restoration_network_calls':0,'metric_network_calls':0,'optimizer_updates':0,'NPU_calls':0}
write(REAL/'evidence/EXTENSION_POSTPROCESS_CPU_TESTS.json',report);print(json.dumps(report,indent=2))

"""Pure Python metadata/authority and frozen historical-delegation checks.
No Torch, model, optimizer, or tensor checkpoint is imported or constructed.
"""
import ast,copy,hashlib,json,math,shutil,time
from pathlib import Path
from types import SimpleNamespace
REAL=Path(__file__).resolve().parents[1];BASE=REAL/'evidence/midnight_auditor_cpu_fixture';ROOT=BASE/'new';PREV=BASE/'previous18';NOON=BASE/'noon'
for p in [ROOT/'protocol',ROOT/'evidence',PREV/'protocol',NOON/'protocol']:p.mkdir(parents=True,exist_ok=True)
POLICY=REAL/'protocol/AUDIT_STATE_FROZEN_TOLERANCES.json';shutil.copy2(POLICY,ROOT/'protocol'/POLICY.name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(q,indent=2)+'\n')
class NoTensor:pass
class NoArray:pass
ns={'Path':Path,'ROOT':ROOT,'POLICY_PATH':ROOT/'protocol'/POLICY.name,'json':json,'math':math,'hashlib':hashlib,'copy':copy,
 'torch':SimpleNamespace(Tensor=NoTensor),'np':SimpleNamespace(integer=int,floating=float,ndarray=NoArray)}
funcs=['sha','ensure','integer','finite_number','exact','count_map','campaign_context','verified_extension_parent_audit','extension_metadata','historical_checkpoint_auditor']
t=ast.parse((REAL/'audit_state.py').read_text())
for n in t.body:
 if isinstance(n,ast.FunctionDef) and n.name in funcs:exec(compile(ast.Module([n],type_ignores=[]),str(REAL/'audit_state.py'),'exec'),ns)
AUTH={'deadline_unix':1793721600,'previous_deadline_unix':1793700000,'native_schedule_deadline_unix':1793678400,
 'extension_handover_authorized_unix':1793698100,'previous_campaign_root':str(PREV),'history_campaign_roots':[str(NOON),str(PREV)],
 'extension_lrs':{'original':1e-5,'improved':2e-7},'extension_parents':{}}
SPEC={'version':'epsilon_opt_mse_wallclock_cosine_v1','initial_lr':2e-7,'floor_lr':2e-7,'peak_lr':2e-6,'warmup_updates':200}
for b,u in [('original',61000),('improved',30000)]:
 e={'path':str(PREV/'runs'/b/'checkpoints'/f'U{u:08d}.pt'),'sha256':'b'*64,'update':u,'cursor':96*u};AUTH['extension_parents'][b]=e
 noon_sched={'deadline_unix':1793678400,'start_unix':1793650000.,'warmup_end_unix':1793650200. if b=='improved' else None,'schedule_spec':SPEC}
 prior={**noon_sched,'deadline_unix':1793700000,'schedule_deadline_unix':1793678400,'training_stop_deadline_unix':1793700000,'extension_lr':AUTH['extension_lrs'][b],
  'type':'constant' if b=='original' else 'extension_constant_floor','current_lr':AUTH['extension_lrs'][b]}
 snap={'extension_parent':{'history':'noon','update':55760 if b=='original' else 20000},'extension_parent_cpu_audit':{'history':'old-CPU-acceptance','sha256':'c'*64},
  'extension_start_update':55760 if b=='original' else 20000,'extension_start_cursor':96*(55760 if b=='original' else 20000),
  'extension_start_counts_by_g':{'1':1,'2':2,'3':3,'4':4},'extension_start_unix':1793681268.,'prior_scheduler':noon_sched,'previous_monitor_done':['old-key']}
 report={'status':'PASS','policy_sha256':sha(POLICY),'checkpoint':{**e,'counts_by_g':{str(g):24*u for g in range(1,5)},'scheduler':prior,'Adam':{'lr':AUTH['extension_lrs'][b]},'extension_payload_metadata_snapshot':snap}}
 write(ROOT/'evidence'/f'PARENT_STATE_AUDIT_{b}.json',report)
write(ROOT/'protocol/USER_AUTHORITY.json',AUTH)
CASES=[]
def fixture(b,n):
 e=AUTH['extension_parents'][b];f=ROOT/'evidence'/f'PARENT_STATE_AUDIT_{b}.json';p=json.loads(f.read_text())['checkpoint'];s=copy.deepcopy(p['scheduler']);s.update(deadline_unix=1793721600,training_stop_deadline_unix=1793721600,completed_updates=e['update']+n,next_update=e['update']+n+1)
 c=copy.deepcopy(p['counts_by_g']);c['1']+=96*n;u=e['update']+n
 q={'format':'UNIDB_DUAL_FULL_STATE_V1','branch':b,'update':u,'cursor':e['cursor']+96*n,'initial_update':45411 if b=='original' else 0,'new_updates':u-(45411 if b=='original' else 0),
  'config':{'deadline_unix':1793721600,'schedule_deadline_unix':1793678400},'scheduler':s,'counts_by_g':c,'prior_scheduler':copy.deepcopy(p['scheduler']),
  'prior_extension_metadata':copy.deepcopy(p['extension_payload_metadata_snapshot']),'extension_parent':copy.deepcopy(e),'extension_parent_cpu_audit':{'path':str(f),'sha256':sha(f)},
  'extension_start_update':e['update'],'extension_start_cursor':e['cursor'],'extension_start_counts_by_g':copy.deepcopy(p['counts_by_g']),
  'extension_new_updates':n,'extension_new_presentations':96*n,'extension_start_unix':1793698200.}
 return q,ROOT/'gates'/b/f'U{u:08d}.pt',AUTH['extension_lrs'][b]
def accept(label,q,path,lr):ns['extension_metadata'](q,path,lr);CASES.append(label)
def reject(label,q,path,lr):
 try:ns['extension_metadata'](q,path,lr)
 except (AssertionError,KeyError,TypeError,ValueError):CASES.append(label);return
 raise AssertionError('Must reject '+label)
for b in ['original','improved']:
 for n in [0,1,2,100]:q,path,lr=fixture(b,n);accept(b+f'_accept_{n}_with_earlier_than18_authorized_handover',q,path,lr)
 q,path,lr=fixture(b,1)
 integer_keys=copy.deepcopy(q);integer_keys['prior_extension_metadata']['extension_start_counts_by_g']={int(k):v for k,v in integer_keys['prior_extension_metadata']['extension_start_counts_by_g'].items()}
 accept(b+'_accept_original_integer_strength_keys_in_JSON_snapshot',integer_keys,path,lr)
 changes=[('stop_deadline',lambda x:x['scheduler'].__setitem__('deadline_unix',1793700000)),('native_schedule_deadline',lambda x:x['scheduler'].__setitem__('schedule_deadline_unix',1793700000)),
 ('config_schedule_deadline',lambda x:x['config'].__setitem__('schedule_deadline_unix',1793700000)),('warmup_reset',lambda x:x['scheduler'].__setitem__('warmup_end_unix',1793701000)),
 ('start_reset',lambda x:x['scheduler'].__setitem__('start_unix',1793701000)),('prior18_changed',lambda x:x['prior_scheduler'].__setitem__('deadline_unix',1793678400)),
 ('lost_noon_history',lambda x:x['prior_extension_metadata'].pop('prior_scheduler')),('changed_ancestor_base',lambda x:x['prior_extension_metadata'].__setitem__('extension_start_update',999)),
 ('parent_sha',lambda x:x['extension_parent'].__setitem__('sha256','0'*64)),('parent_cursor',lambda x:x['extension_parent'].__setitem__('cursor',0)),
 ('parent_audit_sha',lambda x:x['extension_parent_cpu_audit'].__setitem__('sha256','0'*64)),('start_reanchor',lambda x:x.__setitem__('extension_start_update',x['extension_start_update']+1)),
 ('cursor_accounting',lambda x:x.__setitem__('extension_new_presentations',0)),('initial_reset',lambda x:x.__setitem__('initial_update',x['extension_start_update'])),
 ('start_before_user_authority',lambda x:x.__setitem__('extension_start_unix',1793698099)),('start_after_midnight',lambda x:x.__setitem__('extension_start_unix',1793721600))]
 for label,fn in changes:x=copy.deepcopy(q);fn(x);reject(b+'_reject_'+label,x,path,lr)
 reject(b+'_reject_wrong_LR',q,path,lr*1.1)
 reject(b+'_reject_outside_roots',q,BASE/'unauthorized.pt',lr)
# Historical delegation: frozen fake auditor proves module __file__/ROOT stay in
# its own root; actual full Tensor check remains the unmodified historical code.
for root in [NOON,PREV]:
 source=root/'audit_state.py';source.write_text("from pathlib import Path\nROOT=Path(__file__).resolve().parent\ndef checkpoint(path):\n assert Path(path).is_relative_to(ROOT)\n return {'branch':'improved','update':10000,'cursor':960000},{'status':'PASS','own_root':str(ROOT)}\n")
 shutil.copy2(POLICY,root/'protocol'/POLICY.name);write(root/'protocol/USER_AUTHORITY.json',{'root':str(root)})
 write(root/'protocol/RUN_SOURCE.json',{'files':{'audit_state.py':sha(source),'protocol/USER_AUTHORITY.json':sha(root/'protocol/USER_AUTHORITY.json')}})
 result=ns['historical_checkpoint_auditor'](root/'runs/improved/checkpoints/U00010000.pt')
 assert result[1]['own_root']==str(root) and result[1]['historical_frozen_auditor']['tolerances_unchanged'];CASES.append('historical_'+root.name+'_delegates_original_ROOT_and_tolerances')
 source.write_text(source.read_text()+'# unauthorized mutation\n')
 try:ns['historical_checkpoint_auditor'](root/'runs/improved/checkpoints/U00010000.pt')
 except AssertionError:CASES.append('historical_'+root.name+'_reject_changed_frozen_auditor')
 else:raise AssertionError('Historical source mutation accepted')
assert ns['historical_checkpoint_auditor'](BASE/'unknown/checkpoint.pt') is None;CASES.append('historical_unapproved_root_not_delegated')
assert sha(POLICY)=='61bc51ce136f314de7cbbf8ea62c0c6dd1def3cce8881e03b9913879f76a7d3e';CASES.append('frozen_tolerances_exact_SHA')
result={'status':'PASS','count':len(CASES),'checks':CASES,'scope':'Pure Python AST metadata/delegation fixtures; zero Torch imports, checkpoint tensors, model, optimizer, or NPU calls; actual new gate checkpoints still require full CPU audit.',
 'auditor_sha256':sha(REAL/'audit_state.py'),'test_sha256':sha(__file__),'policy_sha256':sha(POLICY),'unix':time.time()}
write(REAL/'evidence/EXTENSION_AUDITOR_CPU_TESTS.json',result);print(json.dumps(result,indent=2))

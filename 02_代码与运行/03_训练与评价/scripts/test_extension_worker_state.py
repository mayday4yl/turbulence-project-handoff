"""CPU bookkeeping test: AST-extract helpers, use tiny plain-Python holders.
No torch import, restoration network, real optimizer, accelerator or training.
The .pt fixtures contain Python pickle only and live below evidence/worker_cpu_state_semantics.
"""
import ast
import copy
import hashlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT_REAL=Path(__file__).resolve().parents[1]
WORKER=ROOT_REAL/'training_worker.py'
OLD_WORKER=ROOT_REAL.parent/'unidb_dual_20261103_until_noon/training_worker.py'
BASE=ROOT_REAL/'evidence/worker_cpu_state_semantics'
BASE.mkdir(parents=True,exist_ok=True)
NEW=BASE/'new';OLD=BASE/'previous'
(NEW/'evidence').mkdir(parents=True,exist_ok=True);OLD.mkdir(exist_ok=True)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def tiny_bank(i):return {'torch':('torch',i),'npu':('npu',i),'numpy':('numpy',i),'python':('python',i)}
class Holder:
    def load_state_dict(self,payload,strict=True):self.value=copy.deepcopy(payload)
    def state_dict(self):return copy.deepcopy(self.value)
class AdamHolder:
    """Stores dictionaries only; never creates or executes torch.optim."""
    def load_state_dict(self,payload):
        self.value=copy.deepcopy(payload);self.state=self.value['state'];self.param_groups=self.value['param_groups']
    def state_dict(self):return copy.deepcopy(self.value)
class FakeDist:
    def all_gather_object(self,out,local):out[:]=copy.deepcopy(self.gathered)
    def barrier(self):pass
class FakeTorch:
    @staticmethod
    def load(path,map_location,weights_only):
        assert map_location=='cpu' and weights_only is False
        return pickle.loads(Path(path).read_bytes())
    @staticmethod
    def save(payload,file):pickle.dump(payload,file,protocol=4)

restored=[]
def restore_rng(bank,device):restored.append(copy.deepcopy(bank))
def restore_ema(holder,value):holder.value=copy.deepcopy(value);holder.step=value['step']
D=FakeDist()
namespace={'Path':Path,'copy':copy,'Counter':Counter,'math':math,'json':json,'ROOT':NEW,
 'FORMAT':'UNIDB_DUAL_FULL_STATE_V1','TRAINING_DEADLINE':1793721600,'PREVIOUS_TRAINING_DEADLINE':1793700000,'SCHEDULE_DEADLINE':1793678400,
 'torch':FakeTorch,'sha':digest,'rng_restore':restore_rng,'ema_restore':restore_ema,
 'cpu_tree':copy.deepcopy,'ema_save':lambda x:copy.deepcopy(x.value),'source_checks':lambda:{'mock':True},
 'capture':lambda device:copy.deepcopy(restored[-1]),'dist':D,'broadcast':lambda x,rank:x,
 'write':write,'full_consensus':lambda *a:'mock-full-digest','SEED':843101,
 'os':__import__('os'),'time':__import__('time')}
tree=ast.parse(WORKER.read_text())
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTENSION_KEYS' for t in node.targets):
        exec(compile(ast.Module([node],type_ignores=[]),str(WORKER),'exec'),namespace)
    if isinstance(node,ast.FunctionDef) and node.name in ['extension_progress','validate_payload_structure','approved_extension_parent','save_checkpoint','load_resume']:
        exec(compile(ast.Module([node],type_ignores=[]),str(WORKER),'exec'),namespace)
# Save imports this immutable specification; supply a plain module, no torch import.
import sys
sys.modules['improved_objective']=SimpleNamespace(SCHEDULE_SPEC={'test':'unchanged-historical-spec'})

checks=[]
def check(label,condition):
    assert condition,label
    checks.append(label)
def rejected(label,call):
    try:call()
    except (AssertionError,FileNotFoundError,KeyError,ValueError):checks.append(label);return
    raise AssertionError('Expected rejection: '+label)

authority={'deadline_unix':1793721600,'previous_deadline_unix':1793700000,'native_schedule_deadline_unix':1793678400,
 'previous_campaign_root':str(OLD),'extension_lrs':{'original':1e-5,'improved':2e-7},'extension_parents':{}}
payloads={};paths={};configs={}
for branch,u in [('original',61000),('improved',30000)]:
    world=3 if branch=='original' else 5
    cfg={'branch':branch,'phase':'formal','physical_world_size':world,'effective_batch':96,
         'data_seed':843101,'precision':'FP32_HF32off','objective':'native_mean_matching_L1' if branch=='original' else 'epsilon_opt_MSE',
         'gamma_inv':1e-7,'T':100}
    configs[branch]=cfg
    cursor=96*u
    payload={'format':namespace['FORMAT'],'branch':branch,'phase':'formal','update':u,'initial_update':45411 if branch=='original' else 0,
      'new_updates':u-(45411 if branch=='original' else 0),'cursor':cursor,'counts_by_g':{str(g):cursor//4 for g in range(1,5)},
      'network':{'weight':[.125,-.25]},'optimizer':{'state':{0:{'step':u,'exp_avg':[.1],'exp_avg_sq':[.01]}},
        'param_groups':[{'lr':1e-5 if branch=='original' else 2e-7,'betas':(.9,.99),'eps':1e-8,'weight_decay':0}]},
      'ema':{'step':u,'initted':True,'ema_model':{'weight':[.126,-.249]}},
      'rng_by_logical_rank':[tiny_bank(i) for i in range(8)] if branch=='original' else None,
      'rng_by_rank':[tiny_bank(100+i) for i in range(5)] if branch=='improved' else None,
      'physical_world_size':world,'logical_world_size':8 if branch=='original' else None,
      'loader_generator_states':[('loader',r) for r in range(world)],
      'data_positions':{'next_global_position':cursor},'scaler':None,'config':cfg,
      'scheduler':{'start_unix':1793650000.25,'deadline_unix':1793700000,'schedule_deadline_unix':1793678400,'warmup_end_unix':1793650200.5 if branch=='improved' else None,
        'type':'extension_constant_floor' if branch=='improved' else 'constant','current_lr':1e-5 if branch=='original' else 2e-7},
      'parent':{'lineage':'official_U0' if branch=='improved' else 'D0_U45411'},'monitor_done':[f'{u}:EMA:1'],'last_completed_unix':1793678400.1}
    payload.update(extension_parent={'old_parent':'noon','update':55760 if branch=='original' else 20000,'cursor':96*(55760 if branch=='original' else 20000)},
      extension_parent_cpu_audit={'old_path':'read-only18parentaudit','sha256':'f'*64},
      extension_start_update=55760 if branch=='original' else 20000, extension_start_cursor=96*(55760 if branch=='original' else 20000),
      extension_start_counts_by_g={'1':1,'2':2,'3':3,'4':4},extension_start_unix=1793681200,
      prior_scheduler={'deadline_unix':1793678400,'start_unix':1793650000.25,'warmup_end_unix':1793650200.5 if branch=='improved' else None},
      previous_monitor_done=['noon-monitor'])
    payload['scheduler']['schedule_spec']={'test':'unchanged-historical-spec'}
    path=OLD/(branch+'.pt');path.write_bytes(pickle.dumps(payload,protocol=4))
    parent={'path':str(path),'sha256':digest(path),'update':u,'cursor':cursor}
    authority['extension_parents'][branch]=parent
    write(path.with_suffix('.json'),{'sha256':digest(path),'full_state':True})
    write(NEW/'evidence'/f'PARENT_STATE_AUDIT_{branch}.json',{'status':'PASS','checkpoint':copy.deepcopy(parent)})
    payloads[branch]=payload;paths[branch]=path

for branch,payload in payloads.items():
    cfg=configs[branch];world=cfg['physical_world_size']
    for rank in range(world):
        net,adam,ema=Holder(),AdamHolder(),Holder()
        state,banks,loader=namespace['load_resume'](paths[branch],net,adam,ema,'mock_cpu',rank,world,cfg,authority)
        check(branch+f'_rank{rank}_RAW_Adam_EMA_exact',net.value==payload['network'] and adam.state_dict()==payload['optimizer'] and ema.value==payload['ema'])
        check(branch+f'_rank{rank}_loader_exact',loader==payload['loader_generator_states'][rank])
        check(branch+f'_rank{rank}_historical_anchors',state['deadline_unix']==1793721600 and state['schedule_deadline_unix']==1793678400 and state['prior_scheduler']==payload['scheduler'] and state['start_unix']==payload['scheduler']['start_unix'] and state['warmup_end_unix']==payload['scheduler']['warmup_end_unix'])
        if branch=='original':
            check(f'original_rank{rank}_logical_bank_exact',banks=={j:payload['rng_by_logical_rank'][j] for j in range(rank,8,world)})
            check(f'original_rank{rank}_ambient_bank_restored',restored[-1]==banks[min(banks)])
        else:check(f'improved_rank{rank}_physical_bank_exact',restored[-1]==payload['rng_by_rank'][rank])
        check(branch+f'_rank{rank}_all_prior_extension_metadata_exact',state['prior_extension_metadata']=={k:payload[k] for k in namespace['EXTENSION_KEYS'] if k in payload})
        check(branch+f'_rank{rank}_no_old_monitor_path_reuse',not state['monitor_done'] and state['previous_monitor_done']==payload['monitor_done'])
    # Execute the actual save helper at rank0 using a predetermined multi-rank gather.
    net,adam,ema=Holder(),AdamHolder(),Holder()
    state,banks,loader=namespace['load_resume'](paths[branch],net,adam,ema,'mock_cpu',0,world,cfg,authority)
    state['extension_start_unix']=1793700100.5
    D.gathered=[{'rng':{j:copy.deepcopy(payload['rng_by_logical_rank'][j]) for j in range(r,8,world)} if branch=='original' else copy.deepcopy(payload['rng_by_rank'][r]),
                 'loader_generator':SimpleNamespace(clone=lambda:None)} for r in range(world)]
    # list entries must preserve independently saved generator bytes, plain tuples here.
    for r,row in enumerate(D.gathered):row['loader_generator']=payload['loader_generator_states'][r]
    class GeneratorHolder:
        def get_state(self):return SimpleNamespace(clone=lambda:payload['loader_generator_states'][0])
    out=NEW/'runs'/branch;out.mkdir(parents=True,exist_ok=True)
    result=namespace['save_checkpoint'](out,state,net,adam,ema,'mock_cpu',0,world,cfg,banks,GeneratorHolder(),label='START',check_full=True)
    start=pickle.loads(Path(result['path']).read_bytes())
    check(branch+'_START_parent_LR_preserved',start['optimizer']['param_groups']==payload['optimizer']['param_groups'])
    check(branch+'_START_two_clock_and_immutable_prior',start['scheduler']['deadline_unix']==1793721600 and start['scheduler']['schedule_deadline_unix']==1793678400 and start['prior_scheduler']==payload['scheduler'])
    check(branch+'_START_zero_extension_count',start['extension_new_updates']==start['extension_new_presentations']==0)
    # Directly edit a synthetic payload as a bookkeeping fixture, not an optimizer update.
    own=copy.deepcopy(start);delta=7;own['update']+=delta;own['new_updates']+=delta;own['cursor']+=96*delta
    own['data_positions']['next_global_position']=own['cursor'];own['extension_new_updates']=delta;own['extension_new_presentations']=96*delta
    own['counts_by_g']['1']+=96*delta;own['optimizer']['state'][0]['step']=own['update'];own['ema']['step']=own['update']
    own['optimizer']['param_groups'][0]['lr']=authority['extension_lrs'][branch]
    own['scheduler'].update(completed_updates=own['update'],next_update=own['update']+1,current_lr=authority['extension_lrs'][branch])
    own['monitor_done']=[f"{own['update']}:EMA:0"]
    own_path=out/'checkpoints'/f"U{own['update']:08d}.pt";own_path.write_bytes(pickle.dumps(own,protocol=4))
    write(own_path.with_suffix('.json'),{'sha256':digest(own_path),'full_state':True})
    recovered,rb,rl=namespace['load_resume'](own_path,Holder(),AdamHolder(),Holder(),'mock_cpu',0,world,cfg,authority)
    check(branch+'_own_resume_preserves_every_extension_anchor',all(recovered[k]==state[k] for k in namespace['EXTENSION_KEYS']))
    check(branch+'_own_resume_preserves_new_monitor_keys',recovered['monitor_done']==set(own['monitor_done']))
    check(branch+'_own_resume_counts_cumulative',recovered['update']==payload['update']+delta and namespace['extension_progress'](recovered)['extension_new_presentations']==96*delta)
    # Negative identities: each must be rejected before producing a continuation.
    broken=copy.deepcopy(authority);broken['extension_parents'][branch]['sha256']='0'*64
    rejected(branch+'_reject_wrong_parent_SHA',lambda:namespace['load_resume'](paths[branch],Holder(),AdamHolder(),Holder(),'mock_cpu',0,world,cfg,broken))
    broken=copy.deepcopy(authority);broken['extension_parents'][branch]['cursor']+=96
    rejected(branch+'_reject_wrong_parent_cursor',lambda:namespace['load_resume'](paths[branch],Holder(),AdamHolder(),Holder(),'mock_cpu',0,world,cfg,broken))
    wrong_path=OLD/('other_'+branch+'.pt');wrong_path.write_bytes(paths[branch].read_bytes())
    rejected(branch+'_reject_unapproved_parent_path',lambda:namespace['load_resume'](wrong_path,Holder(),AdamHolder(),Holder(),'mock_cpu',0,world,cfg,authority))
    audit=NEW/'evidence'/f'PARENT_STATE_AUDIT_{branch}.json';audit_value=json.loads(audit.read_text());audit_value['status']='FAIL';write(audit,audit_value)
    rejected(branch+'_reject_failed_parent_CPU_audit',lambda:namespace['load_resume'](paths[branch],Holder(),AdamHolder(),Holder(),'mock_cpu',0,world,cfg,authority))
    audit_value['status']='PASS';write(audit,audit_value)
    gated=copy.deepcopy(own);gated['phase']='gate';own_path.write_bytes(pickle.dumps(gated,protocol=4));write(own_path.with_suffix('.json'),{'sha256':digest(own_path),'full_state':True})
    rejected(branch+'_reject_formal_inheriting_gate',lambda:namespace['load_resume'](own_path,Holder(),AdamHolder(),Holder(),'mock_cpu',0,world,cfg,authority))

# Verify the actual two terminal STATUS dictionary expressions against the
# finalizer contract, including eval-only recovery without any optimizer step.
terminal=[]
for node in ast.walk(ast.parse(WORKER.read_text())):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='write' and len(node.args)==2 and isinstance(node.args[1],ast.Dict):
        keys=[k.value if isinstance(k,ast.Constant) else None for k in node.args[1].keys]
        vals=[v.value if isinstance(v,ast.Constant) else None for v in node.args[1].values]
        if 'status' in keys and vals[keys.index('status')]=='COMPLETE' and 'final_full_validation_done' in keys:
            terminal.append(node.args[1])
check('exactly_formal_and_eval_only_terminal_STATUS_contracts',len(terminal)==2)
for branch,base,u in [('original',61000,61007),('improved',30000,30007)]:
    st={'update':u,'initial_update':45411 if branch=='original' else 0,'extension_start_update':base,'extension_start_cursor':96*base,'cursor':96*u}
    rows=[eval(compile(ast.Expression(expr),str(WORKER),'eval'),{**namespace,'state':st,'reason':'absolute_deadline','args':SimpleNamespace(phase='formal')}) for expr in terminal]
    required=['update','new_updates','extension_start_update','extension_start_cursor','extension_new_updates','extension_new_presentations','final_full_validation_done']
    check(branch+'_eval_only_and_formal_complete_STATUS_same_progress',all(all(k in q for k in required) for q in rows) and all(rows[0][k]==rows[1][k] for k in required))
    check(branch+'_terminal_extension96_count_contract',all(q['extension_new_updates']==7 and q['extension_new_presentations']==672 for q in rows))
    evaluation=next(q for q in rows if q.get('evaluation_only'))
    check(branch+'_eval_only_contract_zero_optimizer_updates',evaluation['new_optimizer_updates_this_process']==0)

old=OLD_WORKER.read_text();new=WORKER.read_text()
for name in ['layout_for_rank','positions_for_layout','evaluate_isolated','stop_decision']:
    def body(source):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    check(name+'_AST_unchanged',body(old)==body(new))
check('forward_DDP_weight_backward_gradient_Adam_EMA_block_byte_identical',old[old.index('            optimizer.zero_grad('):old.index("            if args.branch=='improved' and state['update']==200:")]==new[new.index('            optimizer.zero_grad('):new.index('            dist.all_reduce(loss_sum);')])
check('no_scheduler_function_call_or_method_reseed','lr_for_update(' not in new and 'manual_seed(' not in new and 'np.random.seed(' not in new and 'random.seed(' not in new)
check('no_startup_monitor',"label='START',check_full=True)\n            # The existing parent monitors" in new)
check('cumulative_monitor_cadence_retained','full=new>0 and new%10000==0' in new and 'new%args.monitor_every==0' in new)
result={'status':'PASS','scope':'Pure Python AST/bookkeeping fixture only; no torch, restoration network, real optimizer or accelerator imported/executed',
        'checks':checks,'count':len(checks),'worker_sha256':digest(WORKER),'baseline_worker_sha256':digest(OLD_WORKER)}
write(ROOT_REAL/'evidence/EXTENSION_WORKER_CPU_SELFTEST.json',result)
print(json.dumps(result,indent=2))

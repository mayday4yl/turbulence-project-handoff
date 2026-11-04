"""Independent CPU audit of midnight formal START and first3 global updates.

No restoration/metric network or optimizer object is constructed. Checkpoint
loads use CPU only. This helper is deliberately outside the training freeze.
"""
from pathlib import Path
import argparse,copy,fcntl,gc,hashlib,json,math,os,struct,sys,time,traceback
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['ASCEND_RT_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/formal_startup_audit'
DEADLINE=1793721600
BRANCHES=('original','improved')
ANCESTOR_KEYS=('extension_parent','extension_parent_cpu_audit','extension_start_update',
 'extension_start_cursor','extension_start_counts_by_g','extension_start_unix','prior_scheduler','previous_monitor_done')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
 path=Path(path).resolve();assert path.is_relative_to(ROOT)
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def emit(event,**extra):
 q={'unix':time.time(),'event':event,**extra};line=json.dumps(q,ensure_ascii=False,allow_nan=False)
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'AUDIT.log').open('a') as f:f.write(line+'\n')
 print(line,flush=True)

def ensure(ok,message):
 if not ok:raise AssertionError(message)


def exact(a,b,label,stats):
 if isinstance(a,torch.Tensor):
  ensure(isinstance(b,torch.Tensor) and a.device.type==b.device.type=='cpu',label+' CPU tensor mismatch')
  ensure(a.dtype==b.dtype and a.shape==b.shape,label+' tensor schema mismatch')
  aa=a.detach().contiguous().reshape(-1).view(torch.uint8);bb=b.detach().contiguous().reshape(-1).view(torch.uint8)
  ensure(torch.equal(aa,bb),label+' tensor element bytes differ')
  stats['tensor_count']+=1;stats['tensor_bytes_compared']+=aa.numel();return
 if isinstance(a,np.ndarray):
  ensure(isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label+' NumPy bytes differ')
  stats['numpy_arrays']+=1;return
 if isinstance(a,dict):
  ensure(isinstance(b,dict) and list(a)==list(b),label+' dict keys/order differ')
  for key in a:exact(a[key],b[key],label+'.'+str(key),stats)
 elif isinstance(a,(tuple,list)):
  ensure(type(a) is type(b) and len(a)==len(b),label+' sequence schema differs')
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'.'+str(i),stats)
 elif isinstance(a,float):
  ensure(isinstance(b,float) and struct.pack('>d',a)==struct.pack('>d',b),label+' float bits differ')
 else:ensure(type(a) is type(b) and a==b,label+' scalar differs')


def category(a,b,label):
 stats={'tensor_count':0,'tensor_bytes_compared':0,'numpy_arrays':0};exact(a,b,label,stats)
 return {'status':'PASS','bitwise_complete_equal':True,**stats}


def proc_identity(pid):
 raw=(Path('/proc')/str(pid)/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 return {'pid':int(pid),'start_ticks':f[19],'state':f[0]}


def frozen_sources():
 p=ROOT/'protocol/RUN_SOURCE.json';manifest=read(p)
 required={'training_worker.py','controller.py','audit_state.py','protocol/USER_AUTHORITY.json',
  'evidence/PARENT_STATE_AUDIT_original.json','evidence/PARENT_STATE_AUDIT_improved.json'}
 ensure(required.issubset(manifest['files']),'Required startup source/parent reports absent from RUN_SOURCE')
 for rel,digest in manifest['files'].items():
  f=(ROOT/rel).resolve();ensure(f.is_relative_to(ROOT),'Frozen path escapes root')
  ensure(sha(f)==digest,'Frozen startup source drift: '+rel)
 return {'path':str(p),'sha256':sha(p),'files_checked':len(manifest['files'])}


def prefix_rows(path,n=3):
 rows=[];h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for line in f:
   if not line.strip():continue
   if not line.endswith(b'\n'):break
   rows.append(json.loads(line));h.update(line)
   if len(rows)==n:break
 return rows,h.hexdigest()


def new_failures():
 for active_name,pattern in [('ACTIVE_HANDOFF_COORDINATOR.json','HANDOFF_FAILURE_*.json'),('ACTIVE_CONTROLLER.json','CONTROLLER_FAILURE*.json')]:
  p=ROOT/'evidence'/active_name
  if not p.exists():continue
  started=read(p).get('started_unix',float('inf'))
  for failure in (ROOT/'evidence').glob(pattern):
   if failure.stat().st_mtime>=started:raise RuntimeError('Current pipeline failure: '+str(failure)+' '+str(read(failure).get('error')))


def ready_snapshot():
 new_failures();authority=read(ROOT/'protocol/USER_AUTHORITY.json')
 ensure(authority['deadline_unix']==DEADLINE,'Not authorized midnight campaign')
 ensure(authority['previous_deadline_unix']==1793700000 and authority['native_schedule_deadline_unix']==1793678400,'Three-clock authority mismatch')
 missing=[]
 if authority.get('parents_pending_natural18h_complete_update_save') or set(authority['extension_parents'])!=set(BRANCHES):
  return None,['authority parents not materialized']
 gate=ROOT/'evidence/ENGINEERING_ACCEPTANCE.json'
 if not gate.is_file():missing.append(str(gate))
 else:
  acceptance=read(gate);ensure(acceptance['status']=='PASS' and acceptance['gate_weights_discarded'] is True,'Actual engineering acceptance did not discard gates')
  ensure(acceptance['formal_parents']==authority['extension_parents'],'Gate/formal parent authority differs')
 for branch,parent in authority['extension_parents'].items():
  out=ROOT/'runs'/branch;start=out/'checkpoints'/f"U{parent['update']:08d}.pt"
  for p in [start,start.with_suffix('.json'),out/'FULL_SHA_START.json',out/'CONFIG.json']:
   if not p.is_file():missing.append(str(p))
  for rank in range(3 if branch=='original' else 5):
   log=out/f'UPDATES_rank{rank}.jsonl';actor=out/f'RANK_{rank}.json'
   if not actor.is_file():missing.append(str(actor))
   if not log.is_file() or len(prefix_rows(log)[0])<3:missing.append(str(log)+' first3 complete rows')
 if missing:return None,missing
 return (authority,read(gate)),[]


def wait_ready(wait,timeout):
 start=time.monotonic()
 while True:
  try:snapshot,missing=ready_snapshot()
  except FileNotFoundError as exc:snapshot=None;missing=[str(exc)]
  if snapshot:return snapshot
  write(OUT/'WAIT_STATUS.json',{'status':'WAITING_FORMAL_START_AND_FIRST3_UPDATES','unix':time.time(),'missing':missing,'seconds':time.monotonic()-start,'new_model_optimizer_NPU_calls':0})
  if not wait:return None
  ensure(time.monotonic()-start<timeout and time.time()<DEADLINE,'Timed out before complete formal startup proof')
  time.sleep(5)


def network_digest(net):
 h=hashlib.sha256()
 for name,value in net.items():
  h.update(name.encode());h.update(value.detach().cpu().contiguous().reshape(-1).numpy().tobytes())
 return h.hexdigest()


def compare_branch(branch,approved,authority,auditor):
 started=time.time();parent=Path(approved['path']).resolve();run=ROOT/'runs'/branch
 start=run/'checkpoints'/f"U{approved['update']:08d}.pt";side=read(start.with_suffix('.json'))
 ensure(parent.is_relative_to(Path(authority['previous_campaign_root']).resolve()/'runs'/branch/'checkpoints'),'Parent path not authorized previous branch')
 ensure(side['status']=='PASS' and side['full_state'] is True and side['phase']=='formal','START sidecar not complete formal state')
 parent_sha=sha(parent);start_sha=sha(start)
 ensure(parent_sha==approved['sha256'] and start_sha==side['sha256'],'Actual parent/START bytes differ from SHA')
 parent_audit_path=Path(approved['cpu_audit_path']).resolve();ensure(parent_audit_path.is_relative_to(ROOT),'Parent acceptance escapes root')
 acceptance=read(parent_audit_path);ensure(acceptance['status']=='PASS','Parent complete CPU report not PASS')
 ensure(all(acceptance['checkpoint'][k]==approved[k] for k in ['path','sha256','update','cursor']),'Parent CPU acceptance identity mismatch')
 emit('LOADING_PARENT_AND_START_CPU',branch=branch,parent_update=approved['update'])
 p=torch.load(parent,map_location='cpu',weights_only=False);s=torch.load(start,map_location='cpu',weights_only=False)
 ensure(p['branch']==s['branch']==branch and p['phase']==s['phase']=='formal','Wrong branch/formal phase')
 ensure(p['update']==s['update']==approved['update'] and p['cursor']==s['cursor']==approved['cursor'],'Not exact zero-update START')
 ensure(s['extension_new_updates']==s['extension_new_presentations']==0,'Discarded gate updates leaked into formal START')
 ensure(s['physical_world_size']==p['physical_world_size']==(3 if branch=='original' else 5),'Physical world changed')
 ensure(len(s['network'])==151 and sum(x.numel() for x in s['network'].values())==137147523,'Network incomplete')
 ensure(len(s['optimizer']['state'])==151,'Adam151 states incomplete')
 report={'status':'PASS','branch':branch,'update':s['update'],'cursor':s['cursor'],'parent':{'path':str(parent),'sha256':parent_sha},
  'START':{'path':str(start),'sha256':start_sha,'bytes':start.stat().st_size},'categories':{}}
 for label,key in [('RAW151','network'),('Adam151_all_groups_steps_moments','optimizer'),('EMA151_and_all_state','ema'),('counts','counts_by_g'),('original_D0_or_SR_lineage','parent'),('data_positions','data_positions')]:
  report['categories'][label]=category(p[key],s[key],label)
 for field in ['initial_update','new_updates','logical_world_size','physical_world_size','scaler']:
  report['categories'][field]=category(p[field],s[field],field)
 banks='rng_by_logical_rank' if branch=='original' else 'rng_by_rank';n=8 if branch=='original' else 5
 ensure(len(p[banks])==len(s[banks])==n,'Method RNG banks incomplete')
 report['categories']['all_method_RNG']=category(p[banks],s[banks],'method_RNG')
 report['method_RNG_banks']=n
 report['categories']['prior18_scheduler_complete']=category(p['scheduler'],s['prior_scheduler'],'prior_scheduler')
 report['categories']['prior18_extension_metadata_complete']=category({k:p[k] for k in ANCESTOR_KEYS},s['prior_extension_metadata'],'prior_extension_metadata')
 report['categories']['previous_monitor_history']=category(p.get('monitor_done',[]),s['previous_monitor_done'],'previous_monitor_done')
 ensure(s['monitor_done']==[],'New campaign reused old monitor completion cache')
 report['three_clock_metadata']=auditor.extension_metadata(s,start,s['optimizer']['param_groups'][0]['lr'])
 ensure(s['config']['deadline_unix']==DEADLINE and s['config']['schedule_deadline_unix']==1793678400,'START config clock mismatch')
 ensure(s['source_identity']['new_manifests']['protocol/RUN_SOURCE.json']==sha(ROOT/'protocol/RUN_SOURCE.json'),'START not bound to active frozen source')
 loader=[];world=s['physical_world_size']
 ensure(len(p['loader_generator_states'])==len(s['loader_generator_states'])==world,'Loader generators incomplete')
 for rank,(a,b) in enumerate(zip(p['loader_generator_states'],s['loader_generator_states'])):
  generator=torch.Generator(device='cpu');generator.set_state(a.clone())
  seed=int(torch.empty((),dtype=torch.int64).random_(generator=generator));expected=generator.get_state()
  ensure(torch.equal(expected,b),'Dedicated loader differs from exactly one legal iterator base_seed advance, rank'+str(rank))
  loader.append({'rank':rank,'one_iterator_base_seed_advance_exact':True,'base_seed':seed,
   'parent_state_sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest(),'START_state_sha256':hashlib.sha256(b.numpy().tobytes()).hexdigest()})
 report['loader_generators']=loader
 marker=run/'FULL_SHA_START.json';m=read(marker);digest=network_digest(s['network'])
 ensure(m['status']=='PASS' and m['sampled'] is False and len(m['all_rank_full_digests'])==world,'START full-rank consensus incomplete')
 ensure(set(m['all_rank_full_digests'])=={digest} and m['parameter_digest_full']==s['full_parameter_digest']==digest,'All-rank START consensus not bound to actual saved RAW bytes')
 ensure(acceptance['checkpoint']['RAW']['full_parameter_digest']==digest,'START RAW digest differs from accepted parent')
 report['all_rank_RAW_full_consensus']={'path':str(marker),'sha256':sha(marker),'full_digest':digest,'ranks':world}
 report['parent_LR']=p['optimizer']['param_groups'][0]['lr'];report['START_LR']=s['optimizer']['param_groups'][0]['lr']
 report['parent_counts_by_g']=p['counts_by_g'];report['seconds']=time.time()-started
 del p,s;gc.collect();emit('COMPLETE_PARENT_START_BITWISE_PASS',branch=branch,seconds=report['seconds'])
 return report


def validate_prefix_rows(branch,approved,config,rows,deadline=DEADLINE,n=3):
 world=3 if branch=='original' else 5;ensure(set(rows)==set(range(world)),'Prefix ranks incomplete')
 ensure(config['branch']==branch and config['phase']=='formal' and config['effective_batch']==96,'Formal prefix config mismatch')
 ensure(config['deadline_unix']==deadline and config['schedule_deadline_unix']==1793678400,'Prefix config clock mismatch')
 prefix=[]
 for i in range(n):
  u=approved['update']+i+1;pre=approved['cursor']+i*96;positions=[];loss=None;counts=None;completed=None
  for rank in range(world):
   r=rows[rank][i];ensure(r['branch']==branch and r['rank']==rank,'Prefix branch/rank mismatch')
   ensure(r['update']==u and r['cursor']==pre+96 and r['new_updates']==u-(45411 if branch=='original' else 0),'Prefix global update/cursor mismatch')
   ensure(r['extension_start_update']==approved['update'] and r['extension_start_cursor']==approved['cursor'] and r['extension_new_updates']==i+1 and r['extension_new_presentations']==96*(i+1),'Prefix extension counters mismatch')
   lr=1e-5 if branch=='original' else 2e-7;ensure(r['lr']==lr,'Prefix LR changed')
   ensure(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and math.isfinite(r['grad_norm_after_clip']),'Prefix nonfinite loss/gradient')
   ensure(r['started_unix']<deadline and r['completed_unix']>=r['started_unix'],'Prefix training clock invalid')
   if loss is None:loss=r['loss'];completed=r['completed_unix'];counts=r['counts_by_g']
   ensure(r['loss']==loss and r['completed_unix']==completed and r['counts_by_g']==counts,'Prefix global loss/completion/Gcounts differ across ranks')
   ensure(sum(counts.values())==r['cursor'],'Prefix global strength counts do not sum to cursor')
   layout=config['physical_micro_layout'][rank];expected=[pre+x['offset']+j for x in layout for j in range(x['n'])]
   ensure(r['positions']==expected and len(expected)==r['real_presentations_local'],'Prefix local positions mismatch')
   ensure(len(r['microbatches'])==len(layout)==r['real_microbatches'],'Prefix micro count mismatch')
   for micro,item in zip(r['microbatches'],layout):
    expected_micro=[pre+item['offset']+j for j in range(item['n'])]
    ensure(micro['positions']==expected_micro and micro['n']==item['n'],'Prefix micro positions mismatch')
    ensure(len(micro['tau'])==item['n'] and all(1<=t<=99 for t in micro['tau']),'Prefix native timestep range changed')
    ensure(micro['gradient_scale_before_DDP_mean']==(3/8 if branch=='original' else item['n']*5/96),'Prefix DDP sample weighting changed')
   if branch=='improved':ensure(r['grad_norm_after_clip']<=1.00001,'Prefix clip exceeds native threshold')
   positions+=r['positions']
  ensure(len(positions)==len(set(positions))==96 and sorted(positions)==list(range(pre,pre+96)),'Global batch is not exactly96 unique real positions')
  prefix.append({'update':u,'extension_update':i+1,'cursor':pre+96,'global_real_presentations':96,'unique_positions':96,'global_loss':loss,'all_rank_LR':lr,'completed_unix':completed})
 return prefix


def process_and_prefix(branch,approved):
 run=ROOT/'runs'/branch;world=3 if branch=='original' else 5;config=read(run/'CONFIG.json');ranks=[];rows={}
 for rank in range(world):
  path=run/f'RANK_{rank}.json';actor=read(path)
  ensure(actor['branch']==branch and actor['phase']=='formal' and actor['rank']==rank and actor['world']==world,'Rank metadata mismatch')
  current=proc_identity(actor['pid']);ensure(current['state']!='Z' and str(actor['start_ticks'])==current['start_ticks'],'Formal rank PID/start differs or exited')
  argv=[x.decode(errors='replace') for x in (Path('/proc')/str(actor['pid'])/'cmdline').read_bytes().split(b'\0') if x]
  ensure(str(ROOT/'training_worker.py') in argv and '--eval-only' not in argv and '--phase' in argv and argv[argv.index('--phase')+1]=='formal','Actual command is not formal worker')
  log=run/f'UPDATES_rank{rank}.jsonl';records,digest=prefix_rows(log);ensure(len(records)==3,'Not3 complete rank updates')
  ensure(records[0]['started_unix']>=path.stat().st_mtime,'Prefix belongs to an older rank incarnation; independent recovery review required')
  rows[rank]=records;ranks.append({'rank':rank,**current,'actual_command':argv,'rank_record_sha256':sha(path),'prefix_file':str(log),'prefix_rows':3,'prefix_sha256':digest})
 prefix=validate_prefix_rows(branch,approved,config,rows)
 return {'status':'PASS','physical_ranks':ranks,'first_complete_global_updates':prefix,'prefix_global_positions_exact96':True}


def self_test():
 checks=[]
 def good(name,fn):fn();checks.append(name)
 def bad(name,fn):
  try:fn()
  except (AssertionError,KeyError,ValueError,TypeError):checks.append(name);return
  raise AssertionError('Must reject '+name)
 for branch,base in [('original',62000),('improved',30000)]:
  world=3 if branch=='original' else 5;approved={'update':base,'cursor':96*base};layout=[]
  for rank in range(world):
   if branch=='original':layout.append([{'lid':lid,'offset':lid*12,'n':12} for lid in range(rank,8,world)])
   else:
    sizes=[20,19,19,19,19];offset=sum(sizes[:rank]);layout.append([{'lid':None,'offset':offset,'n':10},{'lid':None,'offset':offset+10,'n':sizes[rank]-10}])
  config={'branch':branch,'phase':'formal','effective_batch':96,'deadline_unix':DEADLINE,'schedule_deadline_unix':1793678400,'physical_micro_layout':layout};rows={r:[] for r in range(world)}
  for i in range(3):
   pre=approved['cursor']+96*i;u=base+i+1
   for rank in range(world):
    micros=[{'positions':[pre+x['offset']+j for j in range(x['n'])],'n':x['n'],'tau':[55]*x['n'],'gradient_scale_before_DDP_mean':3/8 if branch=='original' else x['n']*5/96} for x in layout[rank]]
    positions=[j for x in micros for j in x['positions']]
    rows[rank].append({'branch':branch,'rank':rank,'update':u,'cursor':pre+96,'new_updates':u-(45411 if branch=='original' else 0),'extension_start_update':base,'extension_start_cursor':96*base,'extension_new_updates':i+1,'extension_new_presentations':96*(i+1),
     'lr':1e-5 if branch=='original' else 2e-7,'loss':.1,'grad_norm':.2,'grad_norm_after_clip':.2,'started_unix':1793701000+i*5,'completed_unix':1793701004+i*5,
     'counts_by_g':{'1':(pre+96)//4,'2':(pre+96)//4,'3':(pre+96)//4,'4':(pre+96)//4},'positions':positions,'real_presentations_local':len(positions),'microbatches':micros,'real_microbatches':len(micros)})
  good(branch+'_three96_batches_valid',lambda:validate_prefix_rows(branch,approved,config,rows))
  for label,change in [('position',lambda q:q[0][0]['positions'].__setitem__(0,-1)),('LR',lambda q:q[0][0].__setitem__('lr',1.1e-5)),('cursor',lambda q:q[0][0].__setitem__('cursor',0)),
   ('step',lambda q:q[0][0].__setitem__('update',0)),('loss_sync',lambda q:q[1][0].__setitem__('loss',.2)),('counts',lambda q:q[0][0]['counts_by_g'].__setitem__('1',0)),
   ('after_deadline',lambda q:q[0][0].__setitem__('started_unix',DEADLINE)),('ancestor_base',lambda q:q[0][0].__setitem__('extension_start_update',0)),
   ('DDP_scale',lambda q:q[0][0]['microbatches'][0].__setitem__('gradient_scale_before_DDP_mean',1.)),('tau_boundary',lambda q:q[0][0]['microbatches'][0]['tau'].__setitem__(0,100))]:
   q=copy.deepcopy(rows);change(q);bad(branch+'_reject_'+label,lambda:validate_prefix_rows(branch,approved,config,q))
  q=copy.deepcopy(rows);q.pop(world-1);bad(branch+'_reject_missing_rank',lambda:validate_prefix_rows(branch,approved,config,q))
  if branch=='improved':
   q=copy.deepcopy(rows);q[0][0]['grad_norm_after_clip']=1.1;bad(branch+'_reject_clip',lambda:validate_prefix_rows(branch,approved,config,q))
 # Execute the actual main AST with only filesystem/identity/argument mocks:
 # reuse and refusal must both overwrite a stale EXIT with this exact actor.
 import ast,tempfile
 from types import SimpleNamespace
 main_ast=next(n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
 class MockParser:
  def add_argument(self,*a,**k):pass
  def parse_args(self):return SimpleNamespace(self_test=False,wait=False,wait_seconds=1)
 OUT.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='lifecycle_fixture_',dir=OUT) as temp:
  for reuse_ok in [True,False]:
   fake_root=Path(temp)/('reuse' if reuse_ok else 'refuse');fake_out=fake_root/'evidence/formal_startup_audit'
   fake_out.mkdir(parents=True);(fake_root/'protocol').mkdir()
   authority=fake_root/'protocol/USER_AUTHORITY.json';authority.write_text('{"mock_authority":true}\n')
   parent=fake_root/'parent_fixture.pt';start=fake_root/'START_fixture.pt';parent.write_bytes(b'fixture_parent_no_tensor');start.write_bytes(b'fixture_START_no_tensor')
   receipt=fake_root/'evidence/FORMAL_EXTENSION_STARTUP_ACCEPTANCE.json'
   receipt.write_text(json.dumps({'status':'PASS','auditor_source_sha256':sha(__file__) if reuse_ok else '0'*64,'authority_sha256':sha(authority),
    'branches':[{'parent':{'path':str(parent),'sha256':sha(parent)},'START':{'path':str(start),'sha256':sha(start)}}]})+'\n')
   exit_file=fake_root/'evidence/STARTUP_AUDIT_EXIT.json';exit_file.write_text('{"actor":{"pid":1,"start_ticks":"old"},"completed_unix":1}\n')
   writes=[]
   def fixture_write(path,value):
    path=Path(path).resolve();ensure(path.is_relative_to(fake_root),'Lifecycle write escapes fixture')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');writes.append(str(path))
   namespace={'__file__':str(Path(__file__).resolve()),'ROOT':fake_root,'OUT':fake_out,'Path':Path,'argparse':SimpleNamespace(ArgumentParser=MockParser),
    'fcntl':fcntl,'os':os,'json':json,'time':time,'traceback':traceback,'sha':sha,'read':read,'write':fixture_write,'ensure':ensure,
    'proc_identity':lambda pid:{'pid':pid,'start_ticks':'fixture_current','state':'S'},'emit':lambda *a,**k:None}
   exec(compile(ast.Module([main_ast],type_ignores=[]),str(__file__),'exec'),namespace)
   rejected=False
   try:namespace['main']()
   except AssertionError:rejected=True
   ensure(rejected is not reuse_ok,'Lifecycle reuse/refusal path not executed as intended')
   actual_actor=read(fake_root/'evidence/ACTIVE_STARTUP_AUDIT.json');actual_exit=read(exit_file)
   ensure(actual_exit['actor']==actual_actor and actual_actor['pid']==os.getpid() and actual_actor['start_ticks']=='fixture_current','Lifecycle EXIT did not match current actor')
   ensure(actual_exit['completed_unix']>=actual_actor['started_unix'] and str(exit_file) in writes,'Lifecycle stale EXIT was not rewritten')
   checks.append('prior_PASS_reuse_writes_current_ACTOR_EXIT' if reuse_ok else 'prior_PASS_validation_refusal_writes_current_ACTOR_EXIT')
 result={'status':'PASS','checks':len(checks),'names':checks,'scope':'Pure stdlib actual prefix-validator schema/positive/negative plus actual-main lifecycle AST fixtures; no Torch import, checkpoint load, model, optimizer or NPU call. Actual complete bitwise verification remains pending.',
  'source_sha256':sha(__file__),'unix':time.time()}
 write(ROOT/'evidence/FORMAL_STARTUP_HELPER_SELFTEST.json',result);print(json.dumps(result,indent=2))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');ap.add_argument('--wait-seconds',type=int,default=10800);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 if args.self_test:self_test();return
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'AUDIT.lock').open('a+')
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print(json.dumps({'status':'ALREADY_RUNNING_NO_DUPLICATE','active':read(ROOT/'evidence/ACTIVE_STARTUP_AUDIT.json') if (ROOT/'evidence/ACTIVE_STARTUP_AUDIT.json').exists() else None}));return
 current=proc_identity(os.getpid());actor={**current,'script':str(Path(__file__).resolve()),'source_sha256':sha(__file__),'started_unix':time.time(),'wait':args.wait,'wait_limit_seconds':args.wait_seconds,'model_objects':0,'optimizer_objects':0,'NPU_calls':0}
 write(ROOT/'evidence/ACTIVE_STARTUP_AUDIT.json',actor);started=time.time();receipt=ROOT/'evidence/FORMAL_EXTENSION_STARTUP_ACCEPTANCE.json'
 try:
  if receipt.exists():
   old=read(receipt);ensure(old['status']=='PASS' and old['auditor_source_sha256']==sha(__file__),'Prior startup audit not reusable; root review required')
   ensure(old['authority_sha256']==sha(ROOT/'protocol/USER_AUTHORITY.json'),'Authority changed after startup acceptance')
   for branch in old['branches']:
    for key in ['parent','START']:ensure(sha(branch[key]['path'])==branch[key]['sha256'],'Previously accepted checkpoint bytes changed')
   emit('REUSE_COMPLETE_STARTUP_ACCEPTANCE_NO_NEW_LOADS',receipt_sha256=sha(receipt));return
  emit('STARTUP_AUDIT_STARTED',actor=actor)
  protocol={'status':'ARMED','actor':actor,'exact_comparison':'CPU tensor element bytes, float bits and dictionary/sequence schema; no numerical allclose',
   'checkpoint_deserialization_plan':4,'loader_scope':'Each dedicated CPU Generator advances exactly once for new iterator base_seed. All method RNG banks remain exact.',
   'new_model_optimizer_NPU_calls':0,'source_sha256':sha(__file__)}
  write(OUT/'PROTOCOL.json',protocol)
  ready=wait_ready(args.wait,args.wait_seconds)
  if ready is None:emit('NOT_READY_NO_CHECKPOINT_LOADS');raise SystemExit(2)
  authority,engineering=ready;freeze=frozen_sources()
  global torch,np
  import torch
  import numpy as np
  torch.set_num_threads(4);torch.set_num_interop_threads(1)
  sys.path.insert(0,str(ROOT));import audit_state as auditor
  reports=[compare_branch(b,authority['extension_parents'][b],authority,auditor) for b in BRANCHES]
  write(OUT/'FULL_START_COMPARISONS.json',{'status':'PASS','branches':reports,'checkpoint_deserializations':4,'new_model_optimizer_NPU_calls':0})
  proofs={b:process_and_prefix(b,authority['extension_parents'][b]) for b in BRANCHES}
  pids=[q['pid'] for proof in proofs.values() for q in proof['physical_ranks']];ensure(len(pids)==len(set(pids))==8,'Not8 distinct formal rank actors')
  ensure(frozen_sources()==freeze,'Source freeze changed during startup comparison')
  for proof in proofs.values():
   for actor_rank in proof['physical_ranks']:
    live=proc_identity(actor_rank['pid']);ensure(live['start_ticks']==actor_rank['start_ticks'] and live['state']!='Z','Rank identity changed during startup proof')
  result={'status':'PASS','scope':'Actual midnight formal START full-state preservation and first3-update engineering acceptance; no restoration quality/convergence/generalization or midnight completion claim.',
   'branches':reports,'formal_prefix_and_processes':proofs,'formal_real_processes_verified':8,'checkpoint_deserializations':4,
   'engineering_acceptance':{'path':str(ROOT/'evidence/ENGINEERING_ACCEPTANCE.json'),'sha256':sha(ROOT/'evidence/ENGINEERING_ACCEPTANCE.json'),'status':engineering['status']},
   'source_freeze':freeze,'new_model_objects':0,'new_optimizer_objects':0,'new_optimizer_steps':0,'NPU_calls':0,
   'auditor_source_sha256':sha(__file__),'authority_sha256':sha(ROOT/'protocol/USER_AUTHORITY.json'),'training_worker_sha256':sha(ROOT/'training_worker.py'),
   'state_auditor_sha256':sha(ROOT/'audit_state.py'),'controller_sha256':sha(ROOT/'controller.py'),'seconds':time.time()-started,'verified_unix':time.time()}
  write(receipt,result);write(OUT/'STATUS.json',{'status':'PASS','unix':time.time(),'receipt_sha256':sha(receipt),'checkpoint_deserializations':4})
  emit('FORMAL_EXTENSION_STARTUP_CPU_ACCEPTANCE_PASS',seconds=result['seconds'],receipt_sha256=sha(receipt))
 except SystemExit:raise
 except BaseException as exc:
  failure={'status':'FAIL','actor':actor,'error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.time()-started,'new_model_optimizer_NPU_calls':0,'unix':time.time()}
  write(OUT/('FAILURE_'+str(time.time_ns())+'.json'),failure);write(OUT/'STATUS.json',failure);emit('STARTUP_REVIEW_FAILURE',error=repr(exc));raise
 finally:
  write(ROOT/'evidence/STARTUP_AUDIT_EXIT.json',{'actor':actor,'completed_unix':time.time(),'PASS_receipt_present':receipt.exists(),'model_objects':0,'optimizer_objects':0,'NPU_calls':0})


if __name__=='__main__':main()

"""CPU engineering invariants, independent of the mandatory full NAFNet GPU gates."""
from pathlib import Path
import sys,copy,random,time,ast,json,re
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from runlib.common import *
import torch,numpy as np
from runlib.data import Stream,eta_batch
from runlib.engine import step
from runlib.checkpoints import save,restore
from runlib.fixtures import tree_equal
for folder in ['runlib','scripts','data_pipeline']:
 for p in (R/folder).glob('*.py'):ast.parse(p.read_text(),filename=str(p))
torch.set_num_threads(2);torch.manual_seed(8)
class Fake:
 pass
fake=Fake();fake.train=[{'start_id':f'{g}_{d}_{i}','source_id':f'{d}_{i//16}','G':g,'domain':d} for i in range(16000) for g in [1,2,3,4] for d in ['Places365','LSDIR']]
s32=Stream(fake,32);s64=Stream(fake,64)
assert s32.identity==s64.identity
for epoch in [0,1,4]:
 a=[r['start_id'] for u in range(epoch*4000,(epoch+1)*4000) for r in s32.rows(u)]
 b=[r['start_id'] for u in range(epoch*2000,(epoch+1)*2000) for r in s64.rows(u)]
 assert a==b and len(set(a))==128000
rr=s32.rows(0);e1=eta_batch(rr,'cpu',0);e2=torch.cat([eta_batch(rr[i:i+3],'cpu',0) for i in range(0,32,3)])
assert torch.equal(e1,e2) and not torch.equal(e1,eta_batch(rr,'cpu',1));del e1,e2
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.w=torch.nn.Parameter(torch.tensor([2.,-1.,.5]));self.stats=torch.nn.Module();self.stats.register_buffer('mu',torch.tensor([.2,.3,.4]))
class ToyStore:
 def batch(self,rows,device):
  x=torch.tensor([r['G']+(.1 if r['domain']=='LSDIR' else -.1) for r in rows])[:,None,None,None].expand(-1,3,2,2)
  return x*.1,x,torch.tensor([r['G'] for r in rows],dtype=torch.float32)
class ToyObjective:
 def __init__(self,m):self.m=m
 def __call__(self,C,Y,G,eta):
  pred=Y*self.m.w[None,:,None,None]+eta[:,:,:2,:2]*.03
  lc=(pred-C).abs().flatten(1).mean(1);lr=(pred*2-Y).abs().flatten(1).mean(1);loss=lc+.2*lr
  return {'path_clean':lc,'path_reconstruction':lr,'path_losses':loss,'loss_mean':loss.mean()}
torch.cuda.synchronize=lambda *a,**k:None;torch.cuda.max_memory_allocated=lambda:0;torch.cuda.max_memory_reserved=lambda:0
c={'effective_batch':32,'peak_lr':1e-4,'max_updates':20000,'precision':'fp32'};out=[]
for micro in [1,8,32]:
 m=Toy();ema=copy.deepcopy(m).requires_grad_(False);o=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);obj=ToyObjective(m)
 metrics=step(m,obj,o,ToyStore(),rr,c,1,micro,ema,namespace='fixture');out.append((m.w.detach().clone(),metrics))
 assert metrics['pre_clip_grad_norm']>1 and abs(metrics['post_clip_grad_norm']-1)<1e-6
for w,metrics in out[1:]:assert torch.allclose(w,out[0][0],rtol=1e-6,atol=1e-7) and abs(metrics['total_loss']-out[0][1]['total_loss'])<1e-6
D=R/'reports/engineering_fixtures'/str(time.time_ns());D.mkdir(parents=True)
identity={'purpose':'CPU_TOY_ENGINE_FIXTURE_NOT_SCIENCE_ACCEPTANCE'}
save(D/'full_U00001.pt',m,ema,o,'D0',c,s32,1,32,identity)
expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(ema.state_dict()),copy.deepcopy(o.state_dict()))
nextids=s32.state(1)['next_sample_ids'];step(m,obj,o,ToyStore(),s32.rows(1),c,2,32,ema,namespace='fixture');after=m.w.detach().clone()
u,payload=restore(D/'full_U00001.pt',m,ema,o,'D0',c,s32,identity);assert u==1 and tree_equal(m.state_dict(),expected[0]) and tree_equal(ema.state_dict(),expected[1]) and tree_equal(o.state_dict(),expected[2]) and s32.state(1)['next_sample_ids']==nextids
step(m,obj,o,ToyStore(),s32.rows(1),c,2,32,ema,namespace='fixture');assert torch.equal(m.w,after)
assert lr_at(500,c)==1e-4 and abs(lr_at(20000,c)-1e-6)<1e-15
c64={**c,'effective_batch':64,'max_updates':10000};assert all(abs(lr_at(i,c64)-lr_at(2*i,c))<1e-15 for i in [1,250,251,10000])
sample='| 0       0                 | 3266918       | python | 20238 |\n| 3       0                 | 3729323 | python | 50000 |'
busy={int(m.group(1)) for m in re.finditer(r'^\|\s+([0-7])\s+0\s*\|\s*(\d+)\s*\|',sample,re.M)};assert busy=={0,3}
from runlib import resources
import os
actual=resources.inventory;actual_legacy=resources.legacy_ready;resources.legacy_ready=lambda gpu:True;uid=os.getuid()
inv={'account':'zhuzhiyu','account_uid':uid,'gpus':{i:{'used_MiB':0,'total_MiB':81920} for i in range(8)},'processes':[{'gpu':3,'pid':11,'uid':uid},{'gpu':5,'pid':22,'uid':uid}]}
resources.inventory=lambda:inv
resources.check(3,allowed_copids=[11]);rejected=False
try:resources.check(2)
except AssertionError:rejected=True
assert rejected
inv['processes'].append({'gpu':1,'pid':33,'uid':None});rejected=False
try:resources.check(3,allowed_copids=[11])
except RuntimeError as e:rejected='UNKNOWN_PID' in str(e)
assert rejected
inv['processes']=[{'gpu':5,'pid':22,'uid':uid},{'gpu':3,'pid':11,'uid':uid+1}];rejected=False
try:resources.check(3)
except RuntimeError as e:rejected='GPU_OCCUPIED' in str(e)
assert rejected;resources.inventory=actual;resources.legacy_ready=actual_legacy
orphan=D/'orphan_U00002.pt';torch.save({'incomplete_transaction_fixture':True},orphan)
save(orphan,m,ema,o,'D0',c,s32,2,32,identity);assert len(list((D/'uncommitted_transactions').glob('*.uncommitted')))==1
rejected=False
try:restore(orphan,m,ema,o,'D0',c,s32,{'wrong':'identity'})
except AssertionError:rejected=True
assert rejected
atomic(R/'reports/CPU_ENGINE_ACCEPTANCE.json',{'status':'PASS','full_epoch_universal_stream_B32_B64':True,'eta_microbatch_invariant':True,'effective_gradient_weighting_and_once_only_clip':True,'checkpoint_state_and_next_update_toy_exact':True,'orphan_transaction_preserved':True,'wrong_checkpoint_identity_rejected':True,'LR_vs_exposure_equivalence':True,'NPU_process_table_parser':True,'account_two_GPU_and_unknown_ownership_gate':True,'unrelated_GPU_occupant_protected':True,'code_syntax':True,'not_full_NAFNet_GPU_acceptance':True,'source_hashes':source_hashes(),'time':time.time()})
print('CPU_ENGINE_ACCEPTANCE PASS; GPU gates still required')

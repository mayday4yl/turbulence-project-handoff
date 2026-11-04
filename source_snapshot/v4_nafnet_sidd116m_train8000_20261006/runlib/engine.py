import time,torch,math
from .common import *
from .data import eta_batch

def gradnorm(params):
 values=[p.grad.detach().float().square().sum() for p in params if p.grad is not None]
 return torch.sqrt(torch.stack(values).sum()) if values else torch.tensor(0.)
def step(model,obj,opt,store,rows,c,u,micro,ema=None,peak_probe=False,namespace='formal',audit=False):
 B=c['effective_batch'];assert len(rows)==B and B%micro==0
 rows=sorted(rows,key=lambda r:r['G']);device=next(model.parameters()).device
 torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True);losses=[]
 for i in range(0,B,micro):
  rr=rows[i:i+micro];C,Y,G=store.batch(rr,device);r=obj(C,Y,G,eta=eta_batch(rr,device,epoch=((u-1)*B)//128000,namespace=namespace))
  (r['loss_mean']*(len(rr)/B)).backward();losses.append(torch.stack([r[k].detach() for k in ['path_clean','path_reconstruction','path_losses']],1));del r,C,Y,G
 missing=[n for n,p in model.named_parameters() if p.grad is None];assert not missing,missing
 branches={}
 if audit:
  b={'backbone':[],'clean_head':[],'xi_head':[],'conditioning':[]}
  for n,p in model.named_parameters():
   if n.startswith(('severity.','condition_film.')) or '.film.' in n:kind='conditioning'
   elif n.startswith('clean_head.'):kind='clean_head'
   elif n.split('.')[0] in model._OFFICIAL_SHARED_MODULES:kind='backbone'
   else:kind='xi_head'
   b[kind].append(p)
  branches={k+'_grad_norm':float(gradnorm(v)) for k,v in b.items()}
 before=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);after=gradnorm(model.parameters())
 means=torch.cat(losses).float().mean(0);assert torch.isfinite(means).all(),'nonfinite loss'
 lr=c['peak_lr'] if peak_probe else lr_at(u,c)
 for group in opt.param_groups:group['lr']=lr
 opt.step()
 if audit:
  assert all(torch.isfinite(p).all() for p in model.parameters()),'nonfinite model state'
  assert all(torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)),'nonfinite optimizer state'
 if ema is not None:
  with torch.no_grad():
   for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.004)
   for p,q in zip(ema.buffers(),model.buffers()):p.copy_(q)
 torch.cuda.synchronize();dt=time.monotonic()-t
 return {'L_C':float(means[0]),'L_R':float(means[1]),'total_loss':float(means[2]),'pre_clip_grad_norm':float(before),'post_clip_grad_norm':float(after),'clipped':float(before)>1,'LR':lr,'update_seconds':dt,'starts_per_second':B/dt,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'nonfinite_count':0,**branches}

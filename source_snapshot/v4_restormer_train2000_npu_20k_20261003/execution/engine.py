import time
import torch
from .common import lr_at
from .data import eta_batch
from npu_runtime.backend import synchronize
from v4.training import update_ema


def step(model, objective, optimizer, scaler, store, rows, update, microbatch, ema=None, phase=None):
    if len(rows)!=32 or microbatch not in (1,2,4,8,16,32): raise ValueError('Effective batch must be32')
    if any(sum(r['G']==g for r in rows)!=8 for g in (1,2,3,4)): raise ValueError('Unbalanced G starts')
    # Keep the selected effective batch/cursor intact while grouping equal-depth
    # samples. All layers are per-image and eta is keyed by start, so only the
    # permitted floating-point gradient reduction order changes. This reduces
    # padded recursive work and CANN specialization to stable active batch sizes.
    rows=sorted(rows,key=lambda r:r['G'])
    phase={} if phase is None else phase;phase['optimizer_started']=False
    device=next(model.parameters()).device;start=time.monotonic()
    optimizer.zero_grad(set_to_none=True);values=[]
    for offset in range(0,32,microbatch):
        part=rows[offset:offset+microbatch];C,Y,G=store.batch(part,device)
        result=objective(C,Y,G,eta=eta_batch(part,device,epoch=(update-1)//1000+1))
        scaler.scale(result['loss_mean']*(len(part)/32)).backward()
        values.append(torch.stack([result[k].detach() for k in ('path_clean','path_reconstruction','path_losses')],1))
        del C,Y,G,result
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing: raise RuntimeError('Missing trainable gradients: '+repr(missing))
    scaler.unscale_(optimizer)
    try:
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
    except RuntimeError as exc:
        if scaler.is_enabled() and scaler.get_scale()>1 and 'non-finite' in str(exc).lower():
            # No optimizer mutation yet. Retry identical keyed B32 at a smaller scale.
            scaler.update(new_scale=max(1.,scaler.get_scale()/2))
            return step(model,objective,optimizer,scaler,store,rows,update,microbatch,ema,phase)
        raise
    means=torch.cat(values).float().cpu().double().mean(0)
    if not torch.isfinite(means).all(): raise FloatingPointError('Nonfinite path loss')
    for pg in optimizer.param_groups:pg['lr']=lr_at(update)
    old_scale=scaler.get_scale()
    phase['optimizer_started']=True
    scaler.step(optimizer);scaler.update()
    if scaler.get_scale()<old_scale:raise FloatingPointError('AMP skipped optimizer update')
    if ema is not None:update_ema(ema,model,32)
    synchronize(device)
    return {'L_C':float(means[0]),'L_R':float(means[1]),'total_loss':float(means[2]),
            'weighted_L_R':.2*float(means[1]),'gradient_norm':float(gradnorm),
            'LR':lr_at(update),'update_seconds':time.monotonic()-start}

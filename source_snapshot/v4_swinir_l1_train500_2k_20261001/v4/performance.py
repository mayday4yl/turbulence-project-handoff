"""Execution-only tuning. Same ordered B32 loss and FP32 graph as parent."""
import numpy as np
import torch
from .matched import eta_batch


def backward32(objective, store, rows, microbatch, epoch):
    if len(rows)!=32 or microbatch not in (1,2):
        raise ValueError('Expected ordered B32 and registered microbatch')
    if any(sum(r['G']==g for r in rows)!=8 for g in range(1,5)):
        raise ValueError('Unbalanced G batch')
    device=next(objective.network.parameters()).device
    values=[]
    for offset in range(0,32,microbatch):
        micro=rows[offset:offset+microbatch]
        C,Y,G=store.batch(micro,device)
        result=objective(C,Y,G,eta=eta_batch(micro,device,epoch))
        if not torch.isfinite(result['path_losses']).all():
            raise FloatingPointError('Nonfinite training loss')
        (result['loss_mean']*(len(micro)/32)).backward()
        values.append(torch.stack([result['path_clean'].detach(),
                                  result['path_reconstruction'].detach(),
                                  result['path_losses'].detach()],dim=1))
        del C,Y,G,result
    # Preserve per-start order and float64 CPU mean, with one transfer per B32.
    return torch.cat(values).cpu().numpy().astype(np.float64).mean(0).tolist()

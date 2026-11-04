"""I1 first-hop diagnostics. GT is confined to this diagnostic module."""
import math,time
from pathlib import Path
import numpy as np
import torch
from .followup_refinement import clone_xi,blocks,Objectives
from .training import synchronize

def scalar_metrics(x,target):
    mse=float((x-target).square().mean())
    return {'MSE':mse,'PSNR':None if mse==0 else -10*math.log10(mse),'zero_MSE':mse==0}

def direction(gj,gq,d):
    a=torch.cat([x.detach().cpu().double().flatten() for x in gj])
    b=torch.cat([x.detach().cpu().double().flatten() for x in gq])
    delta=torch.cat([x.detach().cpu().double().flatten() for x in d])
    na,nb=float(a.norm()),float(b.norm())
    return {'J_gradient_L2':na,'Q_gradient_L2':nb,'cosine':float(a.dot(b))/(na*nb) if na*nb else None,
            'grad_Q_dot_first_Adam_proposal':float(b.dot(delta))}

def diagnose(*,model,clean,sim,data,index,include_direction,arrays=None,save_full_xi=False):
    item=data[index];row=data.records[index]
    y=item['observation'].unsqueeze(0).to('npu:0');gt=item['clean'].unsqueeze(0).to('npu:0')
    if int(item['G'])!=4:raise ValueError('First-hop diagnostics must use G4')
    t=data.trajectories[row['trajectory_id']]
    weak=data._array(t['x3_ref'],(3,256,256)).unsqueeze(0).to('npu:0')
    with torch.no_grad():p=model(y,y.new_tensor([4.]))
    anchor=clone_xi(p.realization);c=p.clean.detach();obj=Objectives(sim,clean,c,y,4,3,anchor)
    obj.calls['network']+=1
    with torch.no_grad():z0=obj.F(c,anchor,3);u0=obj.U(anchor)
    result={'start_id':row['start_id'],'source_id':row['source_id'],'trajectory_id':row['trajectory_id'],
            'scope':'first-hop diagnostic only; GT never generates a solver update','runs':{}}
    if arrays is not None:
        arrays.mkdir(parents=True,exist_ok=True)
        np.savez(arrays/'fixed.npz',c=c.cpu().numpy(),Ua=u0.cpu().numpy(),GT=gt.cpu().numpy(),Y=y.cpu().numpy())
    if save_full_xi:
        np.savez(arrays/'xi_initial.npz',zernike=anchor.epsilon_zernike.cpu().numpy(),tilt=anchor.epsilon_tilt.cpu().numpy())
    for lr,which,n in ((.01,'both',40),(.001,'both',40),(.01,'zernike',10),(.01,'tilt',10)):
        label=f'{which}_lr{lr:g}'
        xi=clone_xi(anchor,True);all_blocks=blocks(xi)
        chosen=all_blocks if which=='both' else [all_blocks[0 if which=='zernike' else 1]]
        other=all_blocks[1 if which=='zernike' else 0] if which!='both' else None
        if other is not None:other.requires_grad_(False)
        optimizer=torch.optim.Adam(chosen,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
        best=clone_xi(anchor)
        with torch.no_grad():best_j=float(obj.objective(anchor,'R'))
        best_step=0;trace=[];first_d=None
        started=time.perf_counter()
        for step in range(n+1):
            with torch.no_grad():current_j=float(obj.objective(xi,'R'))
            if not math.isfinite(current_j):raise FloatingPointError('Nonfinite diagnostic J')
            if current_j<best_j:best,best_j,best_step=clone_xi(xi),current_j,step
            entry={'n':step,'J_current':current_j,'J_best':best_j,'best_inner_step':best_step}
            if step in (0,1,5,10,20,40):
                for policy,value in (('current',xi),('best',best)):
                    with torch.no_grad():z=obj.F(c,value,3);u=obj.U(value)
                    entry[policy]={'Z_change_RMS':float((z-z0).square().mean().sqrt()),
                        'U_change_RMS':float((u-u0).square().mean().sqrt()),
                        'U_vs_GT':scalar_metrics(u,gt),'Z_vs_true_X3':scalar_metrics(z,weak)}
                    if arrays is not None and step in (0,1,10,40):
                        np.savez(arrays/f'{label}_{policy}_n{step}.npz',next_state=z.cpu().numpy(),next_clean=u.cpu().numpy())
            if step==n:
                trace.append(entry);break
            optimizer.zero_grad(set_to_none=True)
            j=obj.objective(xi,'R');g=torch.autograd.grad(j,chosen,create_graph=False)
            obj.calls['VJP']+=1
            before=[b.detach().clone() for b in all_blocks]
            for block,grad in zip(chosen,g):
                if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite diagnostic gradient')
                block.grad=grad
            entry['gradient']=[None if b.grad is None else {'L2':float(b.grad.norm()),
                'RMS':float(b.grad.square().mean().sqrt())} for b in all_blocks]
            optimizer.step()
            updates=[b.detach()-old for b,old in zip(all_blocks,before)]
            if step==0 and lr==.01 and which=='both':first_d=[d.clone() for d in updates]
            entry['actual_update']=[{'L2':float(d.norm()),'RMS':float(d.square().mean().sqrt())} for d in updates]
            entry['relative_initial_RMS']=[float((b.detach()-a).square().mean().sqrt()) for b,a in zip(all_blocks,blocks(anchor))]
            trace.append(entry)
        synchronize(y.device)
        result['runs'][label]={'trace':trace,'seconds':time.perf_counter()-started,'best_inner_step':best_step}
        if save_full_xi and which=='both' and lr==.01:
            np.savez(arrays/'xi_final_current.npz',zernike=xi.epsilon_zernike.detach().cpu().numpy(),tilt=xi.epsilon_tilt.detach().cpu().numpy())
            np.savez(arrays/'xi_final_best.npz',zernike=best.epsilon_zernike.cpu().numpy(),tilt=best.epsilon_tilt.cpu().numpy())
        if include_direction and which=='both' and lr==.01:
            x=clone_xi(anchor,True)
            j=obj.objective(x,'R');gj=torch.autograd.grad(j,blocks(x),create_graph=False)
            q=(obj.U(x)-gt).square().mean();gq=torch.autograd.grad(q,blocks(x),create_graph=False)
            obj.calls['VJP']+=2
            proposal=clone_xi(anchor)
            with torch.no_grad():
                for b,d in zip(blocks(proposal),first_d):b.add_(d)
                after_q=float((obj.U(proposal)-gt).square().mean())
            result['direction']={'all':direction(gj,gq,first_d),
                'zernike':direction(gj[:1],gq[:1],first_d[:1]),
                'tilt':direction(gj[1:],gq[1:],first_d[1:]),
                'actual_Q_before':float(q.detach()),'actual_Q_after_first_Adam':after_q,
                'actual_Q_change':after_q-float(q.detach())}
    result['calls']=obj.calls;result['status']='complete'
    return result

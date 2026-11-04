"""Finite I1 solvers. This module has no GT/quality-selection input."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math, time
import torch
from physical_bridge.simulator import Realization
from .training import synchronize


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    learning_rate: float = 0.0
    iterations: int = 0
    radius: float | None = None


def candidates():
    return [Candidate('U0','U0'),Candidate('R_legacy','R',0.01,40)]+[
        Candidate(f'R_weak_lr{lr:g}_n{n}','R',lr,n)
        for lr in (0.001,0.01) for n in (1,5,10)]+[
        Candidate(f'{family}_r{r:g}',family,0.01,10,r)
        for family in ('P','L') for r in (0.002,0.005,0.01)]


def clone_xi(xi, grad=False):
    return Realization(xi.epsilon_zernike.detach().clone().requires_grad_(grad),
                       xi.epsilon_tilt.detach().clone().requires_grad_(grad))


def blocks(xi):return [xi.epsilon_zernike,xi.epsilon_tilt]


class Objectives:
    """Fixed c/Y/G/s and branch-local Ua; H preserves direct and indirect VJPs."""
    def __init__(self, simulator, clean_network, c, y, G, s, anchor):
        self.simulator,self.network=simulator,clean_network
        if any(p.requires_grad for m in (simulator,clean_network) for p in m.parameters()):
            raise ValueError('Inference model and simulator parameters must be frozen')
        if c.shape[0]!=1:raise ValueError('Native xi solver is single-sample')
        self.c,self.y=c.detach(),y.detach()
        self.G,self.s=int(G),int(s)
        self.calls={'network':0,'F':0,'VJP':0}
        with torch.no_grad():self.ua=self.U(anchor).detach()

    def F(self,c,xi,g):
        self.calls['F']+=1
        # Prepare the ACTUAL c argument inside render. In H it depends on xi.
        return self.simulator.render(c,xi,g).output

    def U(self,xi):
        z=self.F(self.c,xi,self.s)
        self.calls['network']+=1
        return self.network(z,z.new_tensor([self.s]))

    def objective(self,xi,family):
        if family=='L':
            u=self.U(xi)
            return (self.F(u,xi,self.G)-self.y).square().mean()
        return (self.F(self.c,xi,self.G)-self.y).square().mean()

    def values(self,xi):
        with torch.no_grad():
            j=(self.F(self.c,xi,self.G)-self.y).square().mean()
            u=self.U(xi)
            d=(u-self.ua).square().mean()
            h=(self.F(u,xi,self.G)-self.y).square().mean()
        return {'J':float(j),'H':float(h),'D':float(d)}


def finite(values):return all(math.isfinite(x) for x in values.values())


def solve(*, candidate, objectives, anchor, previous=None):
    """Adam proposals + declared backtracking; choose best h, never best clean."""
    if candidate not in candidates() or candidate.family=='U0':
        raise ValueError('Only the twelve declared new solvers and legacy baseline are legal')
    started=time.perf_counter()
    protected=candidate.family in ('P','L')
    key='H' if candidate.family=='L' else 'J'
    limit=(candidate.radius**2*(1+1e-6)+1e-12) if protected else math.inf
    anchor_values=objectives.values(anchor)
    if not finite(anchor_values):raise FloatingPointError('Nonfinite anchor')
    if anchor_values['D']>limit:raise RuntimeError('Network anchor is not feasible')
    initial,values,source=anchor,anchor_values,'network'
    previous_values=None
    if previous is not None:
        previous_values=objectives.values(previous)
        if not finite(previous_values):raise FloatingPointError('Nonfinite previous candidate')
        if previous_values['D']<=limit and previous_values[key]<=values[key]:
            initial,values,source=previous,previous_values,'previous_refined'
    current=clone_xi(initial,True)
    optimizer=torch.optim.Adam(blocks(current),lr=candidate.learning_rate,
        betas=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
    best,best_values,best_step=clone_xi(initial),dict(values),0
    trace=[{'step':0,'accepted':True,'values':dict(values),'best_values':dict(values),
            'best_inner_step':0,'initialization_source':source,
            'anchor_values':anchor_values,'previous_values':previous_values}]
    accepted=0
    for step in range(1,candidate.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        h=objectives.objective(current,candidate.family)
        if not torch.isfinite(h):raise FloatingPointError('Nonfinite objective at accepted point')
        gradients=torch.autograd.grad(h,blocks(current),create_graph=False,allow_unused=False)
        objectives.calls['VJP']+=1
        before=[x.detach().clone() for x in blocks(current)]
        grad_stats=[]
        for p,g in zip(blocks(current),gradients):
            if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite xi gradient')
            p.grad=g
            grad_stats.append({'L2':float(g.norm()),'RMS':float(g.square().mean().sqrt())})
        optimizer.step()  # Advance moments ONCE; alpha checks never touch Adam state.
        proposal=[p.detach()-old for p,old in zip(blocks(current),before)]
        attempts=[]; accepted_values=None; chosen_alpha=None
        for alpha in ((1.,.5,.25,.125,.0625,.03125) if protected else (1.,)):
            with torch.no_grad():
                for p,old,delta in zip(blocks(current),before,proposal):p.copy_(old+alpha*delta)
            trial_finite=all(bool(torch.isfinite(p).all()) for p in blocks(current))
            trial=objectives.values(current) if trial_finite else {'J':math.nan,'H':math.nan,'D':math.nan}
            good=trial_finite and finite(trial) and trial['D']<=limit
            if protected:good=good and trial[key]<=values[key]
            attempts.append({'alpha':alpha,'finite':finite(trial),
                             'values':{k:v if math.isfinite(v) else None for k,v in trial.items()},
                             'accepted':bool(good)})
            if good:
                accepted_values,chosen_alpha=trial,alpha
                break
            if not protected:raise FloatingPointError('Nonfinite unconstrained Adam proposal')
        if accepted_values is None:
            trace.append({'step':step,'accepted':False,'gradient':grad_stats,'trials':attempts,
                          'best_values':best_values,'best_inner_step':best_step,'termination':'all_alphas_rejected'})
            break
        accepted+=1;values=accepted_values
        updates=[p.detach()-old for p,old in zip(blocks(current),before)]
        shifts=[p.detach()-a for p,a in zip(blocks(current),blocks(anchor))]
        if values[key]<best_values[key]:
            best,best_values,best_step=clone_xi(current),dict(values),step
        trace.append({'step':step,'accepted':True,'alpha':chosen_alpha,'values':dict(values),
            'best_values':dict(best_values),'best_inner_step':best_step,'gradient':grad_stats,
            'update':[{'L2':float(d.norm()),'RMS':float(d.square().mean().sqrt())} for d in updates],
            'shift_from_anchor_RMS':[float(d.square().mean().sqrt()) for d in shifts],
            'trials':attempts})
    synchronize(anchor.epsilon_zernike.device)
    result_delta=[a-b for a,b in zip(blocks(best),blocks(initial))]
    changed=any(bool(x.ne(0).any()) for x in result_delta)
    if protected and best_values['D']>limit:raise RuntimeError('Returned xi violates output protection')
    return {'xi':best,'trace':trace,'candidate':asdict(candidate),'calls':dict(objectives.calls),
            'seconds':time.perf_counter()-started,'best_step':best_step,'best_values':best_values,
            'accepted_proposals':accepted,'zero_update':not changed,'fallback':protected and not changed,
            'initialization_source':source,'allowed_D':limit if protected else None}

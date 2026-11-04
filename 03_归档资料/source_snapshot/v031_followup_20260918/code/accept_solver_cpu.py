"""Return/initialization/backtracking edge cases, no restoration-model updates."""
from pathlib import Path
import math
import torch
from physical_bridge.simulator import Realization
from v031.followup_refinement import solve,candidates,blocks
from v031.runtime import save_json,sha256

ROOT=Path(__file__).resolve().parents[1]
def xi(x):return Realization(torch.tensor([x],dtype=torch.float32),torch.tensor([x],dtype=torch.float32))
class Objective:
    def __init__(self,nonfinite_trials=False):self.calls={'network':0,'F':0,'VJP':0};self.nonfinite=nonfinite_trials
    def objective(self,x,family):return sum(((b+1) if family=='L' else (b-1)).square().mean() for b in blocks(x))/2
    def values(self,x):
        vals={'J':float(self.objective(x,'P')),'H':float(self.objective(x,'L')),'D':float(sum(b.square().mean() for b in blocks(x))/2)}
        if self.nonfinite and vals['D']>0:vals['D']=math.nan
        return vals

def main():
    p=next(c for c in candidates() if c.name=='P_r0.002');l=next(c for c in candidates() if c.name=='L_r0.002')
    records={}
    for name,c,previous,obj in [('infeasible_previous',p,xi(.1),Objective()),('feasible_previous',p,xi(.001),Objective()),
        ('tie_previous',p,xi(0),Objective()),('H_initialization',l,xi(.001),Objective()),('nonfinite_trials',p,None,Objective(True))]:
        result=solve(candidate=c,objectives=obj,anchor=xi(0),previous=previous)
        assert result['best_values']['D']<=result['allowed_D']
        if name in ('infeasible_previous','H_initialization'):assert result['initialization_source']=='network'
        if name in ('feasible_previous','tie_previous'):assert result['initialization_source']=='previous_refined'
        if name=='nonfinite_trials':
            assert result['fallback'] and result['best_step']==0 and result['accepted_proposals']==0
            assert len(result['trace'][-1]['trials'])==6
        if name=='infeasible_previous':assert result['trace'][1]['alpha']==.125
        accepted=[t for t in result['trace'] if t['accepted']]
        key='H' if c.family=='L' else 'J'
        assert all(a['values'][key]>=b['values'][key] for a,b in zip(accepted,accepted[1:]))
        assert all(t['values']['D']<=result['allowed_D'] for t in accepted)
        result.pop('xi');records[name]=result
    save_json(ROOT/'acceptance/solver_cpu_edges.json',{'status':'passed','solver_sha256':sha256(ROOT/'code/v031/followup_refinement.py'),
        'restoration_optimizer_updates':0,'cases':records})

if __name__=='__main__':main()

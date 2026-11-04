"""Meaningful CPU checks for grids, delta identity and complete path gradients."""
from lib import *
import torch
from types import SimpleNamespace

def main():
    c={'seed':20261013}
    for u in range(100):
        for sid in ['a','b','c']:
            g=grid_for(sid,u,c);assert len(g) in [2,3,4] and g[0]==1 and g[-1]==0
            assert g==grid_for(sid,u,c)
    class Net(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.3));self.earlier=[]
        def forward(self,x,condition):
            clean=x[:,:3]*self.w+condition[:,0,None,None,None]*.02
            clean.retain_grad();self.earlier.append(clean)
            return SimpleNamespace(clean=clean,realization=self.w)
    class Sim:
        def prepare_image(self,x):return x
        def render_prepared(self,p,xi,g):
            g=torch.as_tensor(g).reshape(-1,1,1,1)
            return SimpleNamespace(output=p*(1+xi*g))
    C=torch.ones(2,3,4,4)*.4;Y=torch.ones_like(C)*.6;model=Net();sim=Sim()
    r=objective(model,sim,C,Y,[[1.,.6,0.],[1.,.3,0.]])
    traced=objective(model,sim,C,Y,[[1.,.6,0.],[1.,.3,0.]],trace=True)
    assert torch.equal(r['final'],traced['final']) and torch.equal(r['loss'],traced['loss']),'Telemetry changed scientific output'
    assert torch.equal(torch.autograd.grad(r['loss'],model.w,retain_graph=True)[0],torch.autograd.grad(traced['loss'],model.w,retain_graph=True)[0]),'Telemetry changed gradients'
    r['final'].square().mean().backward()
    assert model.earlier[0].grad is not None and model.earlier[0].grad.abs().sum()>0, 'Transition detached'
    p=Y*.4;xi=torch.tensor(.3);F=lambda g:sim.render_prepared(p,xi,g).output
    assert torch.equal(Y+(F(4)-F(4)),Y)
    q=Y.clone()
    for x,y in [(4.,3.7),(3.7,2.2),(2.2,1.)]:q=q+(F(y)-F(x))
    assert torch.allclose(q,Y+(F(1)-F(4)),atol=1e-7)
    from timebox_policy import make_timebox
    box=make_timebox({'timebox_policy':{'training_seconds':18000,'gpu_seconds':20400,'total_seconds':21600}},956.,1791940000.)
    assert box['final_deadline_server']-box['first_gpu_gate_start_server']==21600
    assert box['training_deadline_server']<box['gpu_deadline_server']<box['final_deadline_server']
    assert box['original_timebox_unchanged']
    print('PASS deterministic grid, zero step, telescoping, earlier-stage full-BPTT gradient, unchanged telemetry outputs and new six-hour budget')
if __name__=='__main__':main()

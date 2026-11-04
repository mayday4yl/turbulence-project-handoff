"""Candidate Y/G-only inference, inherited method dispatch; no training entrypoint."""
import torch
from v4.core import noise_schedule, validate_inputs
from v4.inference import infer as physical_infer
from .factory import group_spec

def infer_group(network, simulator, observation, severities, *, group, noise_seeds,
                refinement=False, retain_trace=False, audit_refinement=False):
    spec=group_spec(group); G=validate_inputs(observation,severities)
    if network.training or simulator.training: raise ValueError('Inference requires eval mode')
    if len(noise_seeds)!=len(G): raise ValueError('One eta seed per start is required')
    if refinement and group in ('D0','L08'):
        raise ValueError('Refinement is N/A for D0/L08 under the current protocol')
    if group in ('R0','R08'):
        if not refinement:
            return physical_infer(network,simulator,observation,G,noise_seeds=noise_seeds,
                                  T0=spec['T0'],refinement=False,retain_trace=retain_trace)
        parts=[physical_infer(network,simulator,observation[i:i+1],G[i:i+1],
                              noise_seeds=[noise_seeds[i]],T0=spec['T0'],refinement=True,
                              retain_trace=retain_trace,audit_refinement=audit_refinement)
               for i in range(len(G))]
        return {'clean':torch.cat([r['clean'] for r in parts]),'eta':torch.cat([r['eta'] for r in parts]),
                'trace_by_sample':[r['trace'] for r in parts],
                'refinements':[v for r in parts for v in r['refinements']],
                'prediction_count':sum(r['prediction_count'] for r in parts),
                'transition_count':sum(r['transition_count'] for r in parts),
                'refinement_sample_count':sum(r['refinement_sample_count'] for r in parts),
                'refinement_seconds':sum(r['refinement_seconds'] for r in parts),
                'refinement_execution':'per_sample_serial'}
    with torch.no_grad(),torch.autocast(device_type=observation.device.type,enabled=False):
        if group=='D0':
            pred=network(torch.cat([observation,observation],dim=1),torch.stack([G,torch.zeros_like(G)],dim=1))
            return {'clean':pred.clean,'prediction_count':len(G),'transition_count':0,
                    'refinement_sample_count':0,'refinement_seconds':0.,'refinements':[],'trace':[]}
        eta=torch.cat([torch.randn((1,*observation.shape[1:]),device='cpu',dtype=observation.dtype,
                                   generator=torch.Generator(device='cpu').manual_seed(int(seed))).to(observation.device)
                       for seed in noise_seeds])
        active=torch.arange(len(G),device=G.device); state=observation; finals=[None]*len(G); trace=[]
        for step in range(int(G.max())):
            original_G,Y=G[active],observation[active]
            g,t=original_G-step,noise_schedule(original_G,step,spec['T0'])
            Z=network.stats.mix(state,eta[active],t)
            pred=network(torch.cat([Z,Y],dim=1),torch.stack([g,t],dim=1))
            for j,index in enumerate(active.tolist()):
                if g[j]==1: finals[index]=pred.clean[j:j+1]
            continuing=torch.nonzero(g>1,as_tuple=False).flatten()
            if retain_trace: trace.append({'step':step+1,'active':active.tolist(),'g':g.tolist(),'t':t.tolist()})
            if len(continuing):
                a=noise_schedule(original_G[continuing],step+1,1.).reshape(-1,1,1,1)
                state=(1-a)*pred.clean[continuing]+a*Y[continuing]
                active=active[continuing]
        return {'clean':torch.cat(finals),'eta':eta,'trace':trace,'prediction_count':int(G.sum()),
                'transition_count':int((G-1).sum()),'refinement_sample_count':0,
                'refinement_seconds':0.,'refinements':[]}

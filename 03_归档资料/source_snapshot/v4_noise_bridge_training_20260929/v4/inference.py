"""Known-G inference. Refinement only between legal predictions."""
import torch
from physical_bridge.simulator import Realization
from .core import validate_inputs, noise_schedule
from .refinement import refine_xi


def infer(network, simulator, observation, severities, *, noise_seeds, T0=.8,
          refinement=True, retain_trace=False, audit_refinement=False):
    G = validate_inputs(observation, severities)
    if network.training or simulator.training:
        raise ValueError('Inference requires eval mode')
    if any(p.requires_grad for p in simulator.parameters()):
        raise ValueError('Simulator must be frozen')
    if len(noise_seeds) != len(G):
        raise ValueError('Record one eta seed per sample')
    eta = torch.cat([torch.randn((1, *observation.shape[1:]), device=observation.device, dtype=observation.dtype,
                                 generator=torch.Generator(device=observation.device).manual_seed(int(seed)))
                     for seed in noise_seeds])
    active = torch.arange(len(G), device=G.device)
    state = observation
    finals, trace, refinement_records = [None] * len(G), [], []
    for step in range(int(G.max())):
        with torch.no_grad(), torch.autocast(device_type=observation.device.type, enabled=False):
            original_g, original_y = G[active], observation[active]
            g, t = original_g - step, noise_schedule(original_g, step, T0)
            z = network.stats.mix(state, eta[active], t)
            pred = network(torch.cat((z, original_y), 1), torch.stack((g, t), 1))
            continuing = torch.nonzero(g > 1, as_tuple=False).flatten()
            for j, idx in enumerate(active.tolist()):
                if g[j] == 1:
                    finals[idx] = pred.clean[j:j+1]
        record = {'step':step+1, 'active':active.tolist(), 'g':g.tolist(), 't':t.tolist(),
                  'refined_original_indices':active[continuing].tolist() if refinement else []}
        if len(continuing):
            c = pred.clean[continuing]
            xi = Realization(pred.epsilon_zernike[continuing], pred.epsilon_tilt[continuing])
            if refinement:
                result = refine_xi(simulator=simulator, clean=c, xi_hat=xi,
                                   observation=original_y[continuing], original_g=original_g[continuing], network=network,
                                   audit_updates=audit_refinement)
                xi = result.realization
                refinement_records.append(result)
            with torch.no_grad(), torch.autocast(device_type=observation.device.type, enabled=False):
                state = simulator.render(c, xi, g[continuing]-1).output
            active = active[continuing]
        if retain_trace:
            trace.append(record)
    return {'clean':torch.cat(finals), 'eta':eta, 'trace':trace, 'refinements':refinement_records,
            'prediction_count':int(G.sum()), 'transition_count':int((G-1).sum()),
            'refinement_sample_count':int((G-1).sum()) if refinement else 0}

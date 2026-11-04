"""V4 predicted physical rollout, complete paths and path-first loss reduction."""
import torch
from torch import nn
from physical_bridge.simulator import Realization


def mse(a, b):
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] != 3:
        raise ValueError('Expected equally shaped BCHW RGB tensors')
    return (a.float() - b.float()).square().flatten(1).mean(1)


def validate_inputs(y, G):
    if y.ndim != 4 or y.shape[1] != 3 or y.dtype != torch.float32 or not y.shape[0]:
        raise ValueError('Expected nonempty FP32 RGB batch')
    if G.numel() != y.shape[0] or not torch.isfinite(y).all():
        raise ValueError('Invalid batch metadata or nonfinite observation')
    G = G.to(device=y.device, dtype=torch.float32).reshape(-1)
    if not ((G == 1) | (G == 2) | (G == 3) | (G == 4)).all():
        raise ValueError('Original G must be 1,2,3,4')
    return G


def noise_schedule(K, step, T0):
    if not 0 <= T0 <= 1:
        raise ValueError('T0 must be in [0,1]')
    return torch.where(K > 1, T0 * (K - step - 1) / (K - 1).clamp_min(1), 0.)


def sample_eta(y, generator):
    if generator is None:
        raise ValueError('Use an independent noise Generator, separate from model/data RNG')
    # One independent normal tensor per batch sample; never expanded from one image.
    return torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator)


class RolloutModule(nn.Module):
    def __init__(self, network, simulator, T0=0.8):
        super().__init__()
        if any(p.requires_grad for p in simulator.parameters()):
            raise ValueError('Simulator parameters must be frozen')
        if not 0 <= T0 <= 1:
            raise ValueError('Invalid T0')
        self.network, self.simulator, self.T0 = network, simulator, float(T0)
        self.simulator.eval()

    def train(self, mode=True):
        super().train(mode)
        self.simulator.eval()
        return self

    def forward(self, clean, observation, severities, *, noise_generator=None,
                eta=None, retain_intermediates=False):
        G = validate_inputs(observation, severities)
        if clean.shape != observation.shape or clean.device != observation.device or clean.dtype != torch.float32:
            raise ValueError('Clean and observation must share shape/device and be FP32')
        if not torch.isfinite(clean).all():
            raise ValueError('Nonfinite clean')
        eta = sample_eta(observation, noise_generator) if eta is None else eta
        if eta.shape != observation.shape or eta.device != observation.device or eta.dtype != torch.float32 or not torch.isfinite(eta).all():
            raise ValueError('Invalid independent path noise')
        batch = len(G)
        active = torch.arange(batch, device=G.device)
        state = observation
        sums_c = [clean.new_zeros(()) for _ in range(batch)]
        sums_r = [clean.new_zeros(()) for _ in range(batch)]
        finals = [None] * batch
        trace = []
        with torch.autocast(device_type=clean.device.type, enabled=False):
            for step in range(int(G.max().item())):
                original_g = G.index_select(0, active)
                original_y = observation.index_select(0, active)
                g = original_g - step
                t = noise_schedule(original_g, step, self.T0)
                path_eta = eta.index_select(0, active)
                z = self.network.stats.mix(state, path_eta, t)
                pred = self.network(torch.cat((z, original_y), 1), torch.stack((g, t), 1))
                prepared = self.simulator.prepare_image(pred.clean)
                reconstruction = self.simulator.render_prepared(prepared, pred.realization, original_g).output
                lc = mse(pred.clean, clean.index_select(0, active))
                lr = mse(reconstruction, original_y)
                for j, idx in enumerate(active.tolist()):
                    sums_c[idx] = sums_c[idx] + lc[j]
                    sums_r[idx] = sums_r[idx] + lr[j]
                    if g[j] == 1:
                        finals[idx] = pred.clean[j:j+1]
                continuing = torch.nonzero(g > 1, as_tuple=False).flatten()
                next_state = None
                if len(continuing):
                    next_prepared = type(prepared)(prepared.image_mean.index_select(0, continuing),
                                                 prepared.dictionary_images.index_select(0, continuing))
                    xi = Realization(pred.epsilon_zernike.index_select(0, continuing),
                                     pred.epsilon_tilt.index_select(0, continuing))
                    next_state = self.simulator.render_prepared(next_prepared, xi, g.index_select(0, continuing)-1).output
                if retain_intermediates:
                    row = {'step': step+1, 'active': active.clone(), 'G': original_g,
                           'g': g, 't': t, 'Y': original_y, 'eta': path_eta, 'state': state,
                           'Z': z, 'clean': pred.clean, 'epsilon_zernike': pred.epsilon_zernike,
                           'epsilon_tilt': pred.epsilon_tilt, 'next_state': next_state,
                           'lc': lc, 'lr': lr}
                    for x in row.values():
                        if torch.is_tensor(x) and x.requires_grad:
                            x.retain_grad()
                    trace.append(row)
                if next_state is not None:
                    state = next_state
                    active = active.index_select(0, continuing)
        path_clean = torch.stack(sums_c) / G
        path_reconstruction = torch.stack(sums_r) / G
        path_losses = path_clean + .2 * path_reconstruction
        return {'loss_mean': path_losses.mean(), 'path_losses': path_losses,
                'path_clean': path_clean, 'path_reconstruction': path_reconstruction,
                'final_clean': torch.cat(finals), 'trace': trace, 'eta': eta,
                'prediction_count': int(G.sum()), 'transition_count': int((G-1).sum()),
                'observation_render_count': int(G.sum()), 'training_refinement_count': 0}

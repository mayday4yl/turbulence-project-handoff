"""Fixed improved UniDB recipe: epsilon_opt MSE; native bridge/network unchanged.

This module constructs no optimizer and starts no training. ``core.process`` must
resolve to the frozen original D0 core (the worker sets its import path).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as functional

from core import process


SCHEDULE_SPEC = {
    'version': 'epsilon_opt_mse_wallclock_cosine_v1',
    'initial_lr': 2e-7,
    'peak_lr': 2e-6,
    'floor_lr': 2e-7,
    'warmup_updates': 200,
    'warmup': 'linear in completed optimizer updates; update1=initial, update200=peak',
    'after_warmup': 'cosine over [persisted actual warmup_end_unix, absolute deadline]',
}


class ImprovedObjective(torch.nn.Module):
    """Batch-mean epsilon_opt squared error, with one native network call.

    For active native times 1..99, mu_hat = mu0 - c_t * noise and
    epsilon_opt = (mu0 - mu_star) / c_t. Thus this criterion is mathematically
    mean(((mu_hat - mu_star) / c_t)**2), not regression to the sampled epsilon.
    """

    def __init__(self, net, device):
        super().__init__()
        self.network = net
        self.sde = process(device)
        self.sde.set_model(net)
        if self.sde.T != 100:
            raise ValueError('The fixed recipe requires the original T=100 bridge')
        self.last_diagnostics = {}

    def forward(self, degraded, clean):
        tau, state = self.sde.generate_random_states(x0=clean, mu=degraded)
        # Native generation draws CPU randint(1,100) independently per image.
        # Validate before transfer/network call so t100 can never divide by zero.
        if tau.numel() != degraded.shape[0] or not bool(((tau >= 1) & (tau < 100)).all()):
            raise ValueError('Improved objective permits one native time per image in 1..99 only')
        tau = tau.to(degraded.device)
        noise = self.sde.noise_fn(state, tau.reshape(-1))
        with torch.no_grad():
            mu0 = self.sde.reverse_sde_step_mean(state, torch.zeros_like(state), tau)
            mu_star = self.sde.reverse_optimum_step(state, clean, tau)
            c_t = self.sde.sigmas[tau].square() * self.sde.dt / self.sde.f_sigma(tau)
            epsilon_opt = (mu0 - mu_star) / c_t
        loss = functional.mse_loss(noise, epsilon_opt.detach(), reduction='mean')
        # Detached device scalars avoid forced host synchronization each microbatch.
        # The worker may aggregate these for logging; no extra network forward.
        with torch.no_grad():
            self.last_diagnostics = {
                'epsilon_opt_mse': loss.detach(),
                'native_mu_l1': (mu0 - c_t * noise.detach() - mu_star).abs().mean(),
                'c_t_min': c_t.min(),
                'c_t_max': c_t.max(),
            }
        return loss, tau.detach()


def lr_for_update(next_update, start_unix, deadline, warmup_end_unix=None, *, now_unix=None):
    """Return LR for a 1-based *optimizer update*, with no hidden state.

    The worker checkpoints update count/start/deadline/warmup_end/SCHEDULE_SPEC.
    It sets warmup_end_unix once, immediately after optimizer update200 completes,
    and preserves it on resume. ``now_unix`` only injects a clock for tests.
    Deadline stopping belongs to the worker; this function cannot authorize a step.
    """
    if isinstance(next_update, bool) or int(next_update) != next_update or next_update < 1:
        raise ValueError('next_update must be a positive 1-based optimizer-update integer')
    next_update = int(next_update)
    start_unix, deadline = float(start_unix), float(deadline)
    now = time.time() if now_unix is None else float(now_unix)
    if not all(math.isfinite(v) for v in (start_unix, deadline, now)) or deadline <= start_unix:
        raise ValueError('Finite start < absolute deadline and finite current clock are required')
    initial, peak, floor = (SCHEDULE_SPEC[k] for k in ('initial_lr', 'peak_lr', 'floor_lr'))
    warmup = SCHEDULE_SPEC['warmup_updates']
    if next_update <= warmup:
        if warmup_end_unix is not None:
            raise ValueError('A completed warmup timestamp conflicts with an unfinished update count')
        if now >= deadline:
            return floor
        return initial + (peak - initial) * (next_update - 1) / (warmup - 1)
    if warmup_end_unix is None:
        raise ValueError('Resume after update200 requires persisted warmup_end_unix; never repeat warmup')
    end = float(warmup_end_unix)
    if not math.isfinite(end) or end < start_unix:
        raise ValueError('warmup_end_unix must be finite and at or after start_unix')
    if now >= deadline:
        return floor
    if end >= deadline:
        raise ValueError('No post-warmup training interval remains before the deadline')
    phase = min(1., max(0., (now - end) / (deadline - end)))
    return floor + .5 * (peak - floor) * (1. + math.cos(math.pi * phase))

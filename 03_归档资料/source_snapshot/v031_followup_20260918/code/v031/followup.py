"""Exactly the two followup training definitions; no old method is modified."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import nn
from .core import full_image_mse, severity_path

CLEAN_MODULES = ('severity', 'input', 'encoder', 'downsample', 'middle',
                 'clean_decoder', 'clean_head')


class CleanOnly(nn.Module):
    """Reuse the corresponding modules of an already initialized full network."""
    def __init__(self, full):
        super().__init__()
        for name in CLEAN_MODULES:
            self.add_module(name, getattr(full, name))
        self.followup_task = 'T1_DIRECT'

    def forward(self, image, severity):
        embedding = self.severity(severity)
        value = self.input(image)
        skips = []
        for index, level in enumerate(self.encoder):
            value = level(value, embedding)
            skips.append(value)
            if index < len(self.downsample):
                value = self.downsample[index](value)
        for block in self.middle:
            value = block(value, embedding)
        return torch.sigmoid(self.clean_head(self.clean_decoder(value, skips, embedding))).float()


class RolloutModule(nn.Module):
    def __init__(self, network, simulator, lambda_R=0.2, *, network_bfloat16=False):
        super().__init__()
        if network_bfloat16:
            raise ValueError('Followup is FP32 only')
        self.task = network.followup_task
        if self.task not in ('T1_DIRECT', 'T2_FEEDBACK_R'):
            raise ValueError('Unknown task')
        self.network, self.simulator = network, simulator
        if any(p.requires_grad for p in simulator.parameters()):
            raise ValueError('Simulator parameters must be frozen')
        if any(p.dtype != torch.float32 for p in network.parameters()):
            raise ValueError('FP32 parameters required')
        self.lambda_R = 0.0 if self.task == 'T1_DIRECT' else 0.2
        self.simulator.eval()

    def train(self, mode=True):
        super().train(mode)
        self.simulator.eval()
        return self

    def forward(self, clean, observation, severities, schedules: Sequence | None=None,
                *, retain_intermediates=False, collect_step_metrics=False):
        if clean.shape != observation.shape or clean.ndim != 4 or clean.shape[1] != 3:
            raise ValueError('Expected equal RGB BCHW images')
        if clean.dtype != torch.float32 or observation.dtype != torch.float32:
            raise ValueError('Authoritative FP32 arrays required')
        original_g = severities.to(clean.device, dtype=torch.float32).reshape(-1)
        tails = [severity_path(g) for g in original_g.tolist()]
        if schedules is not None and [tuple(x) for x in schedules] != tails:
            raise ValueError('Metadata must retain authoritative original tails')
        paths = [(path[0],) for path in tails] if self.task == 'T1_DIRECT' else tails
        lengths = clean.new_tensor([len(x) for x in paths])
        batch = len(paths)
        if batch != clean.shape[0] or not batch:
            raise ValueError('Nonempty batch, one severity per sample required')
        clean_sums = [clean.new_zeros(()) for _ in paths]
        r_sums = [clean.new_zeros(()) for _ in paths]
        trace, metrics = [], []
        state, active = observation, list(range(batch))
        render_calls = 0
        with torch.autocast(device_type=clean.device.type, enabled=False):
            for k in range(max(map(len, paths))):
                idx = torch.tensor(active, device=clean.device, dtype=torch.long)
                gs = clean.new_tensor([paths[i][k] for i in active])
                prediction = self.network(state, gs)
                c = prediction if self.task == 'T1_DIRECT' else prediction.clean
                lc = full_image_mse(c, clean.index_select(0, idx))
                lr = torch.zeros_like(lc)
                xi, reconstructed = None, None
                if self.task == 'T2_FEEDBACK_R':
                    xi = prediction.realization
                    reconstructed = self.simulator.render(c, xi, original_g.index_select(0, idx)).output
                    lr = full_image_mse(reconstructed, observation.index_select(0, idx))
                    render_calls += 1
                for j, i in enumerate(active):
                    clean_sums[i] = clean_sums[i] + lc[j]
                    r_sums[i] = r_sums[i] + lr[j]
                continuing = [j for j,i in enumerate(active) if k+1 < len(paths[i])]
                # The actual feedback tensor stays connected to every earlier clean.
                next_state = c.index_select(0, torch.tensor(continuing, device=c.device)) if continuing else None
                if collect_step_metrics:
                    metrics.append({'step':k+1, 'active_path_indices':tuple(active),
                                    'current_g':gs.detach(), 'clean_losses':lc.detach(),
                                    'reconstruction_losses':lr.detach()})
                if retain_intermediates:
                    trace.append({'step':k, 'input':state, 'clean':c,
                                  'epsilon_zernike':None if xi is None else xi.epsilon_zernike,
                                  'epsilon_tilt':None if xi is None else xi.epsilon_tilt,
                                  'next_state':next_state, 'clean_losses':lc,
                                  'reconstruction_losses':lr, 'reconstructed':reconstructed,
                                  'current_g':gs, 'original_g':original_g.index_select(0,idx),
                                  'original_observation':observation.index_select(0,idx),
                                  'active_path_indices':tuple(active)})
                if continuing:
                    state = next_state
                    active = [active[j] for j in continuing]
            pc, pr = torch.stack(clean_sums)/lengths, torch.stack(r_sums)/lengths
            losses = pc+self.lambda_R*pr
        return {'loss_sum':losses.sum(), 'loss_mean':losses.mean(), 'path_losses':losses,
                'path_clean':pc, 'path_reconstruction':pr, 'clean_loss_sum':pc.sum(),
                'reconstruction_loss_sum':pr.sum(), 'path_count':batch,
                'prediction_count':sum(map(len,paths)),
                'logical_render_count':0 if self.task=='T1_DIRECT' else sum(map(len,paths)),
                'batched_render_call_count':render_calls, 'training_refinement_count':0,
                'trace':trace, 'step_metrics':metrics}

"""Observe detached scalar losses and the original clipping return value.

No loss replacement, gradient replacement, skipped sample, extra optimizer step,
or changed random draw is permitted. A failed guard makes every rank raise.
"""
from pathlib import Path
import json
import math
import os
import time

import torch
import torch.distributed as dist


class GradientGuardError(FloatingPointError):
    pass


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


class PauseOnlyGuard:
    def __init__(self, policy, output_dir, *, seed, world_size, accumulation_steps, rank):
        if policy.get('schema') != 'v5_pause_only_gradient_guard_v1':
            raise ValueError('Unknown guard policy')
        if policy.get('action') != 'abort_all_ranks_before_optimizer_no_skip':
            raise ValueError('Guard may only abort; it cannot change optimization')
        for key in ('max_micro_total_loss', 'max_micro_clean_mae', 'max_micro_observation_mae',
                    'max_preclip_gradient_norm'):
            value = policy[key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Explicit positive finite safety thresholds required')
        self.policy = policy
        self.folder = Path(output_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.seed, self.world, self.accum, self.rank = seed, world_size, accumulation_steps, rank
        self.update, self.rows, self.norm_window = None, [], []

    def _write(self, event, **values):
        path = self.folder / f'guard_rank{self.rank}.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(safe({'event': event, 'unix_time': time.time(),
                'rank': self.rank, 'update': self.update, 'pid': os.getpid(), **values}), allow_nan=False) + '\n')
            stream.flush()
            if event == 'gradient_guard_abort':
                os.fsync(stream.fileno())

    def _consensus(self, local_bad, device, *, point, diagnostics):
        flag = torch.tensor([int(local_bad)], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            from alignment_training.state import UpdateBatchSampler
            sampler = UpdateBatchSampler(8000, seed=self.seed, microbatch=2,
                accumulation_steps=self.accum, world_size=self.world, rank=self.rank,
                start_update=self.update, max_updates=self.update)
            history = [{**row, 'dataset_indices': sampler.indices_for(self.update, row['micro'])}
                       for row in self.rows]
            self._write('gradient_guard_abort', point=point, local_bad=bool(local_bad),
                        diagnostics=diagnostics, micro_history=history,
                        optimizer_step_executed=False, sample_skipped=False,
                        threshold_semantics='Operational pause criterion, not a scientific acceptance bound')
            raise GradientGuardError(f'gradient guard abort at {point}; no optimizer update; rank={self.rank}')

    def observe_result(self, result, *, path_seed, device):
        delta = path_seed - self.seed
        update_index, remainder = divmod(delta, self.accum * self.world)
        micro, expected_rank = divmod(remainder, self.world)
        if expected_rank != self.rank or update_index < 0:
            raise RuntimeError('Guard path identity differs from formal runner')
        update = update_index + 1
        if update != self.update:
            self.update, self.rows = update, []
        if micro != len(self.rows):
            raise RuntimeError('Guard micro sequence is not contiguous')
        scalars = [result.loss, result.clean_loss, result.observation_loss,
                   *result.stage_clean_losses, *result.stage_observation_losses]
        values = torch.stack([x.detach() for x in scalars]).float().cpu().tolist()
        row = {'micro': micro, 'path_seed': path_seed, 'loss': values[0],
               'clean_mae': values[1], 'observation_mae': values[2],
               'stage_clean_mae': values[3:7], 'stage_observation_mae': values[7:11]}
        self.rows.append(row)
        bad = (any(not math.isfinite(x) for x in values)
               or values[0] > self.policy['max_micro_total_loss']
               or values[1] > self.policy['max_micro_clean_mae']
               or values[2] > self.policy['max_micro_observation_mae'])
        # Synchronize all ranks before any rank enters this micro's backward.
        self._consensus(bad, device, point='micro_forward_before_backward', diagnostics=row)
        return result

    def observe_clipped_norm(self, value, *, device):
        norm = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        if len(self.rows) != self.accum:
            raise RuntimeError('Guard clipping occurred before all microbatches')
        bad = not math.isfinite(norm) or norm > self.policy['max_preclip_gradient_norm']
        self._consensus(bad, device, point='original_gradient_clip_before_optimizer',
                        diagnostics={'preclip_gradient_norm': norm})
        self.norm_window.append((self.update, norm))
        if self.update % 10 == 0:
            self._write('guard_window_checked', update_start=self.norm_window[0][0],
                updates_checked=len(self.norm_window),
                preclip_gradient_norm_max=max(x[1] for x in self.norm_window),
                optimizer_step_executed='not_yet_for_current_update',
                gradients_modified_by_guard=False)
            self.norm_window.clear()
        return value

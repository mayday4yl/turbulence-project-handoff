"""Run the unchanged scientific trainer with detached pause-only observations."""
import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from alignment_training import runner
from alignment_training.authorization import require_alignment_authorization
from alignment_training.state import canonical_hash, file_sha256
from .guard import PauseOnlyGuard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--guard-policy', required=True)
    parser.add_argument('--resume-acceptance', required=True)
    parser.add_argument('--factory', default='alignment_training.factories:training_components')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    config = runner.RunConfig(**json.loads(Path(args.config).read_text()))
    config.validate()
    require_alignment_authorization(config)
    policy = json.loads(Path(args.guard_policy).read_text())
    acceptance = json.loads(Path(args.resume_acceptance).read_text())
    if config.method != 'B' or not config.resume_checkpoint:
        raise RuntimeError('This extension supports only explicit B full-state resume')
    if (policy.get('status') != 'ACTIVE_PAUSE_ONLY_MONITOR' or
            acceptance.get('status') != 'ACCEPTED_FOR_GUARDED_RESUME' or
            acceptance.get('scientific_config_hash') != canonical_hash(runner.scientific_config(config)) or
            acceptance.get('guard_policy_sha256') != file_sha256(args.guard_policy) or
            acceptance.get('method_changed') is not False or
            acceptance.get('optimizer_scheduler_rng_sampler_reset') is not False):
        raise RuntimeError('Root-cause/recovery acceptance is missing or does not bind this run')
    module, name = args.factory.split(':', 1)
    components = getattr(importlib.import_module(module), name)(config, args.device)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    guard = PauseOnlyGuard(policy, Path(config.output_dir) / 'gradient_guard',
        seed=config.seed, world_size=config.world_size,
        accumulation_steps=config.accumulation_steps, rank=rank)
    original_unroll = runner.unroll_training
    original_clip = torch.nn.utils.clip_grad_norm_

    def observed_unroll(*pos, **kwargs):
        result = original_unroll(*pos, **kwargs)
        return guard.observe_result(result, path_seed=kwargs['path_seed'], device=components['device'])

    def observed_clip(*pos, **kwargs):
        # This is precisely the original operation, arguments and return tensor.
        try:
            value = original_clip(*pos, **kwargs)
        except RuntimeError as exc:
            # A rank-local nonfinite exception must not leave other ranks
            # waiting at the subsequent guard collective. This matches that
            # collective and fails all ranks, without an optimizer step.
            guard._consensus(True, components['device'],
                point='original_gradient_clip_exception_before_optimizer',
                diagnostics={'original_exception': repr(exc)})
            raise
        return guard.observe_clipped_norm(value, device=components['device'])

    runner.unroll_training = observed_unroll
    torch.nn.utils.clip_grad_norm_ = observed_clip
    try:
        runner.train(config, **components)
    finally:
        runner.unroll_training = original_unroll
        torch.nn.utils.clip_grad_norm_ = original_clip


if __name__ == '__main__':
    main()

"""CPU-only guard correctness tests. Never imports torch_npu or starts training."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .guard import PauseOnlyGuard, GradientGuardError


POLICY = {'schema': 'v5_pause_only_gradient_guard_v1',
    'action': 'abort_all_ranks_before_optimizer_no_skip',
    'max_micro_total_loss': 1., 'max_micro_clean_mae': 1.,
    'max_micro_observation_mae': 10., 'max_preclip_gradient_norm': 1000.}


def result(loss, clean=None, observation=None):
    clean = loss if clean is None else clean
    observation = loss if observation is None else observation
    return SimpleNamespace(loss=loss, clean_loss=clean, observation_loss=observation,
        stage_clean_losses=[clean] * 4, stage_observation_losses=[observation] * 4)


def test_identity(folder):
    torch.manual_seed(12345)
    initial = torch.nn.Linear(3, 2)
    initial.weight.data.mul_(.01)
    initial.bias.data.zero_()
    x = torch.randn(4, 2, 3)
    records = []
    for enabled in (False, True):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
        guard = PauseOnlyGuard(POLICY, folder, seed=20261022, world_size=6, accumulation_steps=4, rank=0)
        torch.manual_seed(554)
        before_rng = torch.get_rng_state().clone()
        losses = []
        for micro in range(4):
            loss = model(x[micro]).square().mean()
            item = result(loss)
            if enabled:
                assert guard.observe_result(item, path_seed=20261022 + 6*micro, device='cpu') is item
            (loss / 4).backward()
            losses.append(loss.detach().clone())
        grads_before = [p.grad.clone() for p in model.parameters()]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        grads_after = [p.grad.clone() for p in model.parameters()]
        if enabled:
            assert guard.observe_clipped_norm(norm, device='cpu') is norm
        optimizer.step()
        assert torch.equal(before_rng, torch.get_rng_state())
        records.append((losses, grads_before, grads_after, [p.detach().clone() for p in model.parameters()]))
    assert all(torch.equal(a, b) for left, right in zip(*records) for a, b in zip(left, right))


def test_loss_abort(folder):
    tests = [(2., .1, .1), (.1, 2., .1), (.1, .1, 11.), (float('nan'), .1, .1)]
    for index, values in enumerate(tests):
        guard = PauseOnlyGuard(POLICY, folder / str(index), seed=20261022, world_size=6, accumulation_steps=4, rank=0)
        parameter = torch.tensor(.1, requires_grad=True)
        try:
            guard.observe_result(result(*[parameter * 0 + v for v in values]), path_seed=20261022, device='cpu')
        except GradientGuardError:
            assert parameter.grad is None
            continue
        raise AssertionError('Expected pre-backward abort')


def test_gradient_abort(folder):
    guard = PauseOnlyGuard(POLICY, folder, seed=20261022, world_size=6, accumulation_steps=4, rank=0)
    for micro in range(4):
        guard.observe_result(result(torch.tensor(.1)), path_seed=20261022+micro*6, device='cpu')
    try:
        guard.observe_clipped_norm(torch.tensor(1001.), device='cpu')
    except GradientGuardError:
        return
    raise AssertionError('Expected pre-optimizer abort')


def test_guard_dominates_stale_timeout(folder):
    from .resume_controller import (guard_offsets, guard_abort_preempts_retry,
                                    GuardAbortNoRetry)
    from alignment_runtime.controller import transient_failure
    folder.mkdir(parents=True)
    log = folder / 'guard_rank0.jsonl'
    log.write_text(json.dumps({'event': 'gradient_guard_abort', 'update': 999})+'\n')
    offsets = guard_offsets(folder)
    timeout = RuntimeError('old log: HCCL timeout, connection reset by peer')
    assert transient_failure(str(timeout))
    # A historical guard event does not reclassify a new ordinary transient.
    guard_abort_preempts_retry(folder, offsets, folder/'no_receipt.json',
                              stage='B', original_error=timeout)
    assert not (folder/'no_receipt.json').exists()
    with log.open('a') as stream:
        stream.write(json.dumps({'event': 'gradient_guard_abort', 'update': 1001})+'\n')
    try:
        guard_abort_preempts_retry(folder, offsets, folder/'receipt.json',
                                  stage='B', original_error=timeout)
    except GuardAbortNoRetry as exc:
        assert not isinstance(exc, RuntimeError)
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['automatic_retry_allowed'] is False
        assert len(receipt['new_guard_aborts']) == 1
        assert receipt['new_guard_aborts'][0]['record']['update'] == 1001
        return
    raise AssertionError('Current guard abort must preempt old transient text')


def ddp_worker(rank, init_file, folder):
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    guard = PauseOnlyGuard(POLICY, folder, seed=20261022, world_size=2, accumulation_steps=4, rank=rank)
    parameter = torch.tensor(.1, requires_grad=True)
    aborted = False
    try:
        guard.observe_result(result(parameter * 0 + (2. if rank == 1 else .1)),
                             path_seed=20261022+rank, device='cpu')
    except GradientGuardError:
        aborted = True
    assert aborted and parameter.grad is None
    (Path(folder) / f'rank{rank}_PASS').write_text('No backward or optimizer update executed.\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='v5_guard_cpu_') as temporary:
        folder = Path(temporary)
        test_identity(folder / 'identity')
        test_loss_abort(folder / 'loss')
        test_gradient_abort(folder / 'gradient')
        test_guard_dominates_stale_timeout(folder / 'controller')
        mp.spawn(ddp_worker, args=(str(folder / 'gloo_init'), str(folder / 'ddp')), nprocs=2, join=True)
        assert all((folder / 'ddp' / f'rank{rank}_PASS').exists() for rank in range(2))
    report = {'status': 'PASS', 'scope': 'CPU-only isolated pause guard, not NPU science acceptance',
        'passing_path_loss_full_gradient_clip_parameter_update_exact': True,
        'global_rng_unchanged': True, 'each_loss_ceiling_aborts_before_backward': True,
        'nonfinite_aborts_before_backward': True,
        'huge_gradient_aborts_before_optimizer': True,
        'two_rank_gloo_single_rank_injection_both_abort_before_backward': True,
        'current_guard_abort_preempts_stale_timeout_and_cannot_enter_RuntimeError_retry': True,
        'old_guard_event_does_not_change_ordinary_transient_policy': True,
        'official_training_started': False, 'npu_used': False}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

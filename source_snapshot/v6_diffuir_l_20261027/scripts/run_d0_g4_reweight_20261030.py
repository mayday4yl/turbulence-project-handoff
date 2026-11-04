"""Finite single-card actor with four original logical RNG/data/seed slots."""
import argparse
import copy
import json
import os
import signal
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from grad_helpers import capture_rng, observe_network, state_digest
from method_impl.d0 import ImmediateEMA, derive_seed
from method_impl.d0_official_residual import D0OfficialResidualRestorer
from method_impl.d0_g4_reweight_20261030 import (BASE, EX, OLD_EX, config, freeze, load_pool_and_lookup,
    ReweightedOrder, SerialDataset, restore_virtual, capture_virtual, logical_slot, branch_identity,
    image_pair, parent_s3_probe)
from method_impl.training_io import new_snapshot, restore_snapshot, atomic_torch_save, cpu_tree
from scripts.audit_ddp_resume import equal_tree
from scripts.runtime_utils import ROOT, sha, write_json, process_identity

STOP = False


def stopping(sig, frame):
    global STOP
    STOP = True


def serial_gradient_reference(model, pool, lookup, cfg, branch, rngs, device, out):
    """Compare 12-micro mean with mean of four three-micro logical-rank grads."""
    order = ReweightedOrder(pool, lookup, branch, cfg)
    batches = []
    for slot in range(12):
        indices = [order.at(cfg['parent_cursor'] + slot * 5 + j) for j in range(5)]
        pairs = [image_pair(pool, i) for i in indices]
        batches.append((torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])))
    def backward(slot, divisor, states):
        micro, vr = logical_slot(slot)
        restore_virtual(states, vr, device)
        x, c = [v.to(device) for v in batches[slot]]
        loss = model(x, c, seed=derive_seed(cfg['seed'], BASE + 1, micro, vr))['loss']
        (loss / divisor).backward()
        capture_virtual(states, vr, device)
    model.zero_grad(set_to_none=True)
    states = copy.deepcopy(rngs)
    for slot in range(12):
        backward(slot, 12, states)
    serial = [p.grad.detach().cpu().clone() for p in model.parameters()]
    reference = [torch.zeros_like(g) for g in serial]
    for vr in range(4):
        model.zero_grad(set_to_none=True)
        states = copy.deepcopy(rngs)
        for micro in range(3):
            backward(micro * 4 + vr, 3, states)
        for result, p in zip(reference, model.parameters()):
            result.add_(p.grad.detach().cpu(), alpha=.25)
    maximum = max(float((a - b).abs().max()) for a, b in zip(serial, reference))
    numerator = sum(float((a.double() - b.double()).square().sum()) for a, b in zip(serial, reference))
    denominator = sum(float(b.double().square().sum()) for b in reference)
    relative = (numerator / max(denominator, 1e-30)) ** .5
    assert maximum <= 1e-6 and relative <= 1e-5, (maximum, relative)
    model.zero_grad(set_to_none=True)
    write_json(out / 'GRADIENT_MEAN_EQUIVALENCE.json', dict(status='PASS', maxabs=maximum,
        relative_L2=relative, serial_microbatch=5, serial_accumulation=12, effective_batch=60,
        reference='mean of four logical-rank means of three native microbatches',
        actual_four_card_HCCL_bitwise_reproduction_claimed=False, optimizer_updates=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--safety-file', required=True)
    ap.add_argument('--branch', required=True, choices=['Control', 'G4_biased'])
    ap.add_argument('--mode', required=True, choices=['probe', 'formal'])
    ap.add_argument('--end-local', type=int)
    ap.add_argument('--resume')
    a = ap.parse_args()
    import torch_npu
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    cfg = config()
    freeze()
    safety = json.loads(Path(a.safety_file).read_text())
    assert len(safety['physical_devices']) == 1
    physical = safety['physical_devices'][0]
    assert [physical] == cfg['physical_devices_by_branch'][a.branch]
    limit = safety['cards'][str(physical)]
    free, total = torch.npu.mem_get_info()
    torch.npu.set_per_process_memory_fraction(limit['allocator_cap'] / total)
    out = ROOT / 'distributed' / a.run
    write_json(out / 'RANK_0.json', dict(process_identity(os.getpid()), physical_device=physical))
    policy = json.loads((ROOT / 'gates/EXECUTION_POLICY_CURRENT.json').read_text())
    assert policy['active_execution'] == EX.name and policy['allow_d0_g4_reweight']
    assert not policy['allow_ab_formal_training']
    assert json.loads((EX / 'PARENT_AND_DATA_GATE.json').read_text())['status'] == 'PASS'
    if a.mode == 'formal':
        assert json.loads((EX / 'SMOKE_RESUME_GATE.json').read_text())['status'] == 'PASS'
        assert a.resume is None, 'Formal starts only from the real common parent, never Probe'
    pool, lookup, data_identity = load_pool_and_lookup()
    parent = torch.load(cfg['parent_path'], map_location='cpu', weights_only=False)
    assert sha(cfg['parent_path']) == cfg['parent_sha256'] and parent['update'] == BASE
    assert parent['cursor'] == cfg['parent_cursor'] and len(parent['rng_by_rank']) == 4
    model = D0OfficialResidualRestorer(ROOT.parent / 'v5_diffuir_l_stage1_20261026/weights/model-300.pt').to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-6, betas=(.9, .99), weight_decay=0)
    ema = ImmediateEMA(model.network)
    payload = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else parent
    if a.resume:
        assert a.mode == 'probe' and payload['config']['experiment'] == cfg['experiment']
        assert payload['branch'] == a.branch
    restore_snapshot(payload, model, opt, ema)
    assert equal_tree(opt.state_dict(), payload['optimizer']) and equal_tree(ema.state_dict(), payload['ema'])
    assert state_digest(model.network.state_dict()) == state_digest(payload['network'])
    assert payload['scaler'] is None
    assert all(int(v['step']) == payload['update'] for v in opt.state_dict()['state'].values())
    assert all(g['lr'] == 2e-6 and tuple(g['betas']) == (.9, .99) and g['weight_decay'] == 0 for g in opt.param_groups)
    canonical_rng = copy.deepcopy(payload['rng_by_rank'])
    for vr in range(4):
        restore_virtual(canonical_rng, vr, device, verify=True)
    local_update = payload['update'] - BASE
    cursor = payload['cursor']
    assert cursor == cfg['parent_cursor'] + local_update * 60
    identity = branch_identity(parent, cfg, a.branch, physical)
    branch_cfg = dict(cfg, branch=a.branch, updates=BASE + (10000 if a.branch == 'G4_biased' else 5000))
    cp = out / 'checkpoints'
    cp.mkdir(exist_ok=True)
    counts = Counter({int(k): v for k, v in payload.get('branch_strength_presentations', {}).items()})
    base_counts = Counter({int(k): v for k, v in payload.get('branch_original_strength_presentations', {}).items()})
    write_json(out / 'RESUME_INHERITANCE.json', dict(status='PASS', update=payload['update'], cursor=cursor,
        common_parent_sha256=cfg['parent_sha256'], raw_EMA_Adam_native_sampling_buffers_exact=True,
        scaler=None, constant_lr=2e-6, four_logical_rng_slots_restored=True,
        physical_device=physical, physical_world_size=1, microbatch=5, accumulation=12, effective_batch=60,
        actual_four_card_bitwise_resume_claimed=False, formal_from_probe=False))
    write_json(out / 'CONFIG.json', branch_cfg)
    write_json(out / 'IDENTITY.json', identity)
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    if a.mode == 'probe' and not a.resume:
        parent_s3_probe(model, ema, device, out / 'PARENT_S3_EQUIVALENCE.json')
        serial_gradient_reference(model, pool, lookup, cfg, a.branch, canonical_rng, device, out)
        restore_snapshot(payload, model, opt, ema)
        canonical_rng = copy.deepcopy(payload['rng_by_rank'])
    target = a.end_local if a.mode == 'probe' else (10000 if a.branch == 'G4_biased' else 5000)
    assert target is not None and target > local_update
    ds = SerialDataset(pool, lookup, cfg, a.branch, cursor, (target - local_update) * 60)
    generator = torch.Generator(device='cpu').manual_seed(derive_seed(cfg['seed'], 'serial_loader_workers'))
    loader = DataLoader(ds, batch_size=5, shuffle=False, num_workers=2, multiprocessing_context='spawn',
                        prefetch_factor=2, persistent_workers=True, generator=generator)
    iterator = iter(loader)
    order = ReweightedOrder(pool, lookup, a.branch, cfg)
    parent_scheduler = copy.deepcopy(parent['scheduler'])

    def save(u):
        step = BASE + u
        value = new_snapshot(model, opt, ema, update=step, cursor=cfg['parent_cursor'] + u * 60,
                             rng=copy.deepcopy(canonical_rng), config=branch_cfg, identity=identity)
        value['scheduler'] = parent_scheduler if u == 0 else dict(type='G4_reweight_constant_LR_continuation',
            last_update=step, constant_lr=2e-6, parent_scheduler=parent_scheduler)
        value.update(parent_scheduler=parent_scheduler, branch=a.branch, branch_updates=u,
            branch_presentations=u * 60, branch_strength_presentations=dict(counts),
            branch_original_strength_presentations=dict(base_counts),
            parent_checkpoint_sha256=cfg['parent_sha256'], parent_update=BASE,
            parent_manifest_cursor=cfg['parent_cursor'], stage_updates=u, stage_presentations=u * 60,
            total_presentations_lineage=step * 60, previous_data_cursor=parent.get('previous_data_cursor'),
            parent_config=parent['config'], parent_identity=parent['identity'],
            rng_layout='four original logical ranks; four device vectors retained per logical rank')
        path = cp / f'U{step:06d}.pt'
        if path.exists():
            existing = torch.load(path, map_location='cpu', weights_only=False)
            assert equal_tree(existing, value), 'Existing complete update must not be overwritten'
            digest = sha(path)
        else:
            digest = atomic_torch_save(path, value)
        check = torch.load(path, map_location='cpu', weights_only=False)
        assert equal_tree(check, value), 'Full-state CPU readback'
        assert len(check['optimizer']['state']) == 276
        assert all(int(s['step']) == step for s in check['optimizer']['state'].values())
        write_json(out / 'LAST.json', dict(path=str(path), sha256=digest, update=step,
            branch_updates=u, cursor=value['cursor'], cpu_readback=True))
        return path

    def validate(u):
        from scripts.multistrength_support_20261029 import evaluate_rank, summarize
        opt.zero_grad(set_to_none=True)
        torch.npu.empty_cache()
        saved = capture_rng(device)
        try:
            location = out / 'validation' / f'u{u:06d}_five_g_1000'
            evaluate_rank(model, ema, device, location, 0, 1, True)
            summary = summarize(location, 1, True, BASE + u)
            summary.update(branch=a.branch, branch_updates=u)
            write_json(location / 'SUMMARY.json', summary)
        finally:
            from grad_helpers import restore_rng
            restore_rng(saved, device)
            model.train()
        write_json(out / f'VALIDATION_u{u:06d}.json', dict(status='PASS', path=str(location),
            count=5000, actual_S3=True, raw_metrics_unchanged=True, fresh_execution=True))

    save(local_update)
    started = time.time()
    torch.npu.reset_peak_memory_stats()
    completed = local_update
    try:
        for u in range(local_update + 1, target + 1):
            if STOP or Path(cfg['stop_request']).exists():
                break
            begin = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            positions, original_indices, chosen_indices, gs, original_gs, taus, seeds = [], [], [], [], [], [], []
            activation = []
            for slot in range(12):
                x, c, indices, pp = next(iterator)
                micro, vr = logical_slot(slot)
                restore_virtual(canonical_rng, vr, device)
                x, c = x.to(device), c.to(device)
                positions += pp.tolist()
                chosen_indices += indices.tolist()
                pairs = [order.pair(p) for p in pp.tolist()]
                assert [j for _, j in pairs] == indices.tolist()
                original_indices += [i for i, _ in pairs]
                original_gs += [int(pool[i]['g']) for i, _ in pairs]
                gs += [int(pool[j]['g']) for _, j in pairs]
                seed = derive_seed(cfg['seed'], BASE + u, micro, vr)
                seeds.append(seed)
                audit = (u <= 2 or u % 50 == 0) and slot == 0
                if audit:
                    with observe_network(model.native, retain=True) as (calls, outputs):
                        result = model(x, c, seed=seed)
                    assert len(calls) == len(outputs) == 1
                    manual = (outputs[0] - ((2 * x - 1) - (2 * c - 1))).abs().mean()
                    assert torch.allclose(result['loss'], manual, atol=3e-8, rtol=1e-6)
                else:
                    result = model(x, c, seed=seed)
                assert bool(torch.isfinite(result['loss']))
                (result['loss'] / 12).backward()
                total_loss += result['loss'].detach() / 12
                taus += result['tau'].tolist()
                if audit:
                    assert outputs[0].grad is not None and bool(torch.isfinite(outputs[0].grad).all())
                    assert bool(outputs[0].grad.abs().max() > 0)
                    activation.append(dict(nfe=1, finite=True, nonzero=True,
                        manual_L1_abs=float((manual - result['loss']).abs()), grad_l2=float(outputs[0].grad.norm())))
                    outputs.clear()
                    del manual
                capture_virtual(canonical_rng, vr, device)
                del x, c, result
            assert positions == list(range(cfg['parent_cursor'] + (u - 1) * 60, cfg['parent_cursor'] + u * 60))
            assert all(p.grad is not None for p in model.parameters())
            assert bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            assert bool(norm > 0) and all(group['lr'] == 2e-6 for group in opt.param_groups)
            opt.step()
            ema.update(model.network)
            assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
            assert bool(torch.stack([torch.isfinite(p).all() for p in ema.model.parameters()]).all())
            counts.update(gs)
            base_counts.update(original_gs)
            completed = u
            torch.npu.synchronize()
            elapsed = time.perf_counter() - begin
            free, total = torch.npu.mem_get_info()
            assert free >= limit['min_free']
            record = dict(update=BASE + u, branch_update=u, branch=a.branch, cursor=cfg['parent_cursor'] + u * 60,
                loss=float(total_loss), residual_loss=float(total_loss), grad_norm=float(norm), lr=2e-6,
                ema_updates=ema.updates, training_nfe_per_sample=1, sampling_training_calls=0,
                microbatch=5, accumulation=12, effective_batch=60, virtual_rank_count=4,
                actual_global_positions=positions, original_pool_indices=original_indices,
                chosen_pool_indices=chosen_indices, original_g_values=original_gs, g_values=gs,
                tau=taus, logical_micro_seeds=seeds, denoiser_output_gradient_audit=activation,
                seconds=elapsed, images_per_second=60 / elapsed, finite=True,
                peak_allocated=torch.npu.max_memory_allocated(), peak_reserved=torch.npu.max_memory_reserved(), device_free=free)
            with (out / 'UPDATES.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
                f.flush()
            if u % 10 == 0 or u <= 2:
                write_json(out / 'STATUS.json', dict(status='RUNNING', unix=time.time(), branch=a.branch,
                    branch_update=u, update=BASE + u, loss=record['loss'], lr=2e-6, actual_g_presentations=dict(counts)))
            if a.mode == 'probe' or u <= 2 or u % 500 == 0 or u in [1000, 3000, 5000, 6000, 8000, 10000] or STOP:
                save(u)
            if a.mode == 'formal' and u in [1000, 3000, 5000, 6000, 8000, 10000] and not STOP:
                validate(u)
            if a.mode == 'formal' and a.branch == 'G4_biased' and u == 5000 and not STOP:
                decision = EX / 'EXTENSION_DECISION.json'
                write_json(out / 'STATUS.json', dict(status='WAITING_FOR_MATCHED_5K_EXTENSION_DECISION', branch_update=u, unix=time.time()))
                deadline = time.time() + 12 * 3600
                while not decision.exists() and not STOP and not Path(cfg['stop_request']).exists():
                    assert time.time() < deadline, 'Bounded matched Control decision wait exceeded'
                    time.sleep(5)
                if STOP or Path(cfg['stop_request']).exists():
                    break
                d = json.loads(decision.read_text())
                assert d['status'] == 'PASS_FROZEN_RULE_DECISION'
                if not d['extend_G4_biased']:
                    break
            if STOP or Path(cfg['stop_request']).exists():
                break
        save(completed)
        write_json(out / 'RESULT.json', dict(status='PASS', branch=a.branch, mode=a.mode,
            completed_branch_updates=completed, completed_updates=BASE + completed,
            full_checkpoint_saved=True, native_single_step_loss=True, effective_batch=60,
            actual_g_presentations=dict(counts), seconds=time.time() - started,
            user_or_resource_stopped=STOP or Path(cfg['stop_request']).exists(), temporary_weights_used_for_formal=False))
    except BaseException as exc:
        write_json(out / 'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), last_completed_branch_update=completed))
        raise


if __name__ == '__main__':
    main()

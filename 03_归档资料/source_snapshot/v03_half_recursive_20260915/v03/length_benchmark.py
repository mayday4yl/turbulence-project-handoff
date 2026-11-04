"""Finite four-card timings for six legal actual prediction counts.

Engineering diagnostic only: original schedule builder and simulator, real
train trajectories, no saved weights or held-out data. Three updates per case;
the first is excluded from steady timing. Main mixed-path selection is separate.
"""
import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .common import atomic_json, build_model, build_simulator, choose_sample, seed_all, verify_base
from .core import RolloutModule, ddp_loss_scale
from .train import EMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_hashes = verify_base(config)
    from physical_bridge.data import BridgeArchive, evenly_spaced_schedule
    import torch_npu  # noqa: F401
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('Exactly four authorized ranks required')
    device = torch.device(f'npu:{int(os.environ["LOCAL_RANK"])}')
    torch.npu.set_device(device); torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(4)
    dist.init_process_group('hccl', timeout=dt.timedelta(minutes=10))
    output = args.output_dir
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'summary.json').exists():
            raise FileExistsError('Length benchmark already completed')
    dist.barrier()
    simulator = build_simulator(config, device)
    archive = BridgeArchive(config['dataset_root'], 'train')
    rows = []
    for severity, requested_k in [(0.75, 0), (1.5, 1), (2.25, 3), (3., 3), (3.75, 5), (4.5, 5)]:
        schedule = tuple(evenly_spaced_schedule(severity, requested_k))
        seed_all(config['seed'])
        network = build_model(config, device)
        module = RolloutModule(network, simulator, config['training']['lambda_R'], network_bfloat16=True)
        wrapped = DistributedDataParallel(module, device_ids=[device.index], broadcast_buffers=False)
        module.train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=config['training']['learning_rate'],
                                      betas=(.9, .99), weight_decay=1e-4)
        ema = EMA(network, config['training']['ema_decay'])
        for update in range(1, 4):
            torch.npu.synchronize(device); dist.barrier()
            torch.npu.reset_peak_memory_stats(device)
            started = time.monotonic()
            ids = [(update - 1) * 8 + rank * 2 + i for i in range(2)]
            records = [archive.records[choose_sample(config, sample, 0)['record_index']] for sample in ids]
            clean = torch.stack([archive.clean(record) for record in records]).to(device)
            observation = torch.stack([archive.state(record, severity) for record in records]).to(device)
            strengths = torch.full((2,), severity, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            result = wrapped(clean, observation, strengths, [schedule, schedule])
            loss = result['loss_sum'] * ddp_loss_scale(8, world)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); ema.update(network)
            torch.npu.synchronize(device)
            elapsed = torch.tensor(time.monotonic() - started, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            peak = torch.tensor([torch.npu.max_memory_allocated(device), torch.npu.max_memory_reserved(device)],
                                dtype=torch.int64, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            row = {'G': severity, 'requested_K': requested_k, 'schedule': list(schedule),
                   'actual_predictions_per_path': len(schedule), 'update': update,
                   'physical_NPU_count': world, 'per_device_microbatch': 2, 'global_paths': 8,
                   'gradient_accumulation': 1, 'seconds': float(elapsed.item()),
                   'allocated_peak_bytes': int(peak[0].item()), 'reserved_peak_bytes': int(peak[1].item()),
                   'gradient_norm': float(gradient.item())}
            rows.append(row)
            if rank == 0:
                atomic_json(output / 'progress.json', row)
                print(json.dumps(row), flush=True)
            del result, loss, clean, observation, strengths
        del wrapped, module, network, optimizer, ema
        gc.collect(); torch.npu.empty_cache(); dist.barrier()
    archive.store.close()
    if verify_base(config) != source_hashes:
        raise ValueError('V0.1 source authority changed')
    if rank == 0:
        by_length = {}
        for length in range(1, 7):
            selected = [row for row in rows if row['actual_predictions_per_path'] == length]
            steady = selected[1:]
            seconds = sum(row['seconds'] for row in steady)
            by_length[str(length)] = {'schedule': selected[0]['schedule'], 'G': selected[0]['G'],
                'requested_K': selected[0]['requested_K'], 'mean_update_seconds': seconds / 2,
                'trajectories_per_second': 16 / seconds, 'network_predictions_per_second': 16 * length / seconds,
                'optimizer_updates_per_second': 2 / seconds, 'first_update_seconds': selected[0]['seconds'],
                'allocated_peak_bytes': max(row['allocated_peak_bytes'] for row in selected),
                'reserved_peak_bytes': max(row['reserved_peak_bytes'] for row in selected)}
        atomic_json(output / 'summary.json', {'status': 'completed', 'by_actual_prediction_count': by_length,
            'physical_NPU_count': 4, 'per_device_microbatch': 2, 'global_paths_per_update': 8,
            'total_sampled_paths': 144, 'total_network_predictions': 504,
            'scope': 'Isolated legal length fixtures, global8. Do not substitute for the global32 mixed-path throughput benchmark.',
            'timing_scope': 'Data read/stack/transfer, network, original simulator, backward, DDP synchronization, clip, optimizer and EMA. Firstupdate excluded from steadystate; startup/firstshape compilation not an isolated compiler measurement.',
            'source_hashes': source_hashes, 'weights_saved': False, 'held_out_test_run': False, 'updates': rows})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

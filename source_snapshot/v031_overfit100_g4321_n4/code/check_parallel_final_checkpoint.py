#!/usr/bin/env python3
"""CPU-only complete-state audit after the parallel guard releases its resources."""
import datetime
import hashlib
import json
import math
import pathlib
import sys

import torch

root = pathlib.Path('/data2/yanglin/v031_overfit100_g4321_n4')
update = int(sys.argv[1])
assert update == 1000

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

workflow = json.loads((root / 'artifacts/WORKFLOW_STATUS.json').read_text())
assert workflow['formal_global_update'] == update
assert workflow['run_until_update'] == 1000
assert not workflow['allow_automatic_updates_beyond_1000']
path = root / 'runs/formal' / f'checkpoint_{update:06d}.pt'
payload = torch.load(path, map_location='cpu', weights_only=True)
required = ['network', 'ema', 'optimizer', 'scheduler', 'rng_by_rank', 'sampler', 'identity', 'counts']
assert all(key in payload for key in required)
assert payload['format'] == 'v031_distributed_update_boundary_v1'
assert payload['completed_updates'] == payload['global_update'] == payload['ema']['updates'] == update
assert payload['sampler']['consumed'] == payload['counts']['paths'] == update * 32
assert payload['formal_training'] and payload['smoke_optimizer_updates'] == 0
assert payload['world_size'] == 4 and payload['microbatch'] == 1 and payload['effective_batch'] == 32
assert len(payload['rng_by_rank']) == 4
assert all(x['torch_npu_local'].numel() > 0 for x in payload['rng_by_rank'])
states = payload['optimizer']['state']
parameter_ids = {p for group in payload['optimizer']['param_groups'] for p in group['params']}
assert set(states) == parameter_ids
steps = sorted({int(value['step'].item()) for value in states.values()})
assert steps == [update]

def inspect_tensors(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        return 1
    if isinstance(value, dict):
        return sum(inspect_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(inspect_tensors(v) for v in value)
    return 0

tensor_counts = {key: inspect_tensors(payload[key]) for key in ['network', 'ema', 'optimizer']}
schedule = payload['scheduler']
assert schedule['completed_updates'] == update and schedule['next_update_index'] == update + 1
def learning_rate(n):
    if n <= 50:
        return 1e-4 * n / 50
    if n >= 1000:
        return 1e-5
    return 1e-5 + (1e-4 - 1e-5) * (1 + math.cos(math.pi * (n - 50) / 950)) / 2
assert abs(schedule['last_used_lr'] - learning_rate(update)) < 1e-15
assert abs(schedule['next_lr'] - learning_rate(update + 1)) < 1e-15
checkpoint_sha256 = digest(path)
assert checkpoint_sha256 == workflow['last_checkpoint_sha256']
manifest = json.loads((root / 'artifacts/STAGE2_DELIVERY_SHA256.json').read_text())
mismatches = [rel for rel, expected in manifest.items() if not (root / rel).is_file() or digest(root / rel) != expected]
assert not mismatches, mismatches
calibration_check = json.loads((root / 'artifacts/INFERENCE_UPDATE100_FREEZE_CHECK.json').read_text())
assert digest(root / 'artifacts/calibration_update100/calibration.json') == calibration_check['calibration_file_sha256']
segment = json.loads((root / 'artifacts/train_segment_001000.json').read_text())
assert segment['execution']['returncode'] == 0
assert segment['checkpoint_sha256'] == checkpoint_sha256
guard = json.loads((pathlib.Path(workflow['active_guard_directory']) / 'guard_result.json').read_text())
assert guard['status'] == 'completed' and guard['returncode'] == 0 and guard['own_npu_processes_released']
out = {
    'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'passed', 'checkpoint': str(path), 'sha256': checkpoint_sha256,
    'bytes': path.stat().st_size, 'global_update': update, 'optimizer_steps': steps,
    'optimizer_parameter_states': len(states), 'ema_updates': payload['ema']['updates'],
    'sampler_consumed': payload['sampler']['consumed'], 'rng_rank_count': len(payload['rng_by_rank']),
    'scheduler': schedule, 'required_complete_state_fields': required,
    'checked_tensor_counts': tensor_counts, 'all_floating_tensors_fp32_and_finite': True,
    'frozen_stage2_files_verified': len(manifest), 'inference_calibration_unchanged': True,
    'training_segment_exited_successfully': True,
    'parallel_guard_completed_and_resources_released': True,
    'run_until_update': 1000, 'allow_automatic_updates_beyond_1000': False,
}
(root / 'artifacts' / f'CHECKPOINT_{update:06d}_STATE_CHECK.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))

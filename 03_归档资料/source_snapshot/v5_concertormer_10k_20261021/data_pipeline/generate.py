"""Generate only new G=4 V5 FP32 observations with per-source dynamic locks."""
import argparse
import fcntl
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import SIMULATOR_SHA256, atomic, digest, npy_bytes, rows, sha, verify_array, write_json


def load_runtime(simulator_root, device, cache_root):
    simulator_root = Path(simulator_root)
    assert sha(simulator_root / 'simulator_v5.py') == SIMULATOR_SHA256
    torch.set_num_threads(2)
    sys.path.insert(0, str(simulator_root))
    # Match the accepted simulator's explicitly configured NPU execution path.
    if device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(device)
        torch.npu.init()
        torch.npu.set_compile_mode(jit_compile=False)
        cache = Path(cache_root)
        cache.mkdir(parents=True, exist_ok=True)
        torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype',
                             'NPU_FUZZY_COMPILE_BLACKLIST': 'Conv2DBackpropInput',
                             'ACL_OP_COMPILER_CACHE_MODE': 'enable',
                             'ACL_OP_COMPILER_CACHE_DIR': str(cache)})
        torch.npu.config.allow_internal_format = True
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        if hasattr(torch.npu, 'aclnn'):
            torch.npu.aclnn.allow_hf32 = False
    from simulator_v5 import build_simulator, Realization
    sim = build_simulator(device=device)
    sim.eval()
    for parameter in sim.parameters():
        parameter.requires_grad_(False)
    # Check exact all-enabled physical semantics and FP32 buffers.
    assert sim.smooth_psf and sim.pixel_center_grid and sim.tilt_fft == 'legacy_full'
    assert sim.mu.dtype == torch.float32
    identity = {'simulator_sha256': sha(simulator_root / 'simulator_v5.py'),
                'vendor_simulator_sha256': sha(simulator_root / 'vendor/v4_simulator.py'),
                'asset_sha256': {p.name: sha(p) for p in sorted((simulator_root / 'assets').iterdir()) if p.is_file()},
                'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'backend': device.split(':')[0], 'physics_dtype': 'float32',
                'hf32': False, 'jit_compile': False if device.startswith('npu') else None,
                'npu_precision': 'must_keep_origin_dtype' if device.startswith('npu') else None,
                'raw_rng': 'independent CPU torch.Generator.manual_seed per kind; torch.randn(dtype=float32) exact B=1 shape',
                'raw_shapes': {'epsilon_zernike': [1, 36, 512, 512], 'epsilon_tilt': [1, 512, 512, 2]},
                'G': 4.0, 'clamp': False, 'smooth_psf': True, 'pixel_center_grid': True,
                'tilt_fft': 'legacy_full', 'observation_dtype': 'float32'}
    if device.startswith('npu'):
        identity['torch_npu'] = str(torch_npu.__version__)
    return sim, Realization, identity


def raw_noise(row):
    tensors = []
    for seed_key, shape in [('seed_zernike', (1, 36, 512, 512)), ('seed_tilt', (1, 512, 512, 2))]:
        generator = torch.Generator(device='cpu').manual_seed(row[seed_key])
        tensors.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    hashes = {'epsilon_zernike': digest(tensors[0].numpy().tobytes()),
              'epsilon_tilt': digest(tensors[1].numpy().tobytes())}
    return tensors, hashes


def sync(device):
    if device.startswith('npu'):
        torch.npu.synchronize(device)


def generate(a):
    root = Path(a.root)
    freeze = json.loads((root / 'SOURCE_FREEZE.json').read_text())
    assert sha(root / 'sample_plan.jsonl') == freeze['sample_plan_sha256']
    assert sha(root / 'source_manifest.jsonl') == freeze['source_manifest_sha256']
    samples = rows(root / 'sample_plan.jsonl')
    sources = rows(root / 'source_manifest.jsonl')
    sim, Realization, identity = load_runtime(a.simulator_root, a.device, root.parent.parent / 'cache/data_generation' / a.device.replace(':', '_'))
    identity.update({'source_freeze_sha256': sha(root / 'SOURCE_FREEZE.json'),
                     'generator_file_sha256': sha(Path(__file__)), 'raw_rng_helper_file_sha256': sha(Path(__file__).with_name('common.py'))})
    write_json(root / 'GENERATION_IDENTITY.json', identity)
    identity_hash = sha(root / 'GENERATION_IDENTITY.json')
    by_source = {source['source_id']: [] for source in sources}
    for row in samples:
        by_source[row['source_id']].append(row)
    started = time.perf_counter()
    made = resumed = locked_elsewhere = 0
    for source in sources:
        if a.splits and source['split'] not in a.splits:
            continue
        lock_path = root / 'locks' / (source['source_id'] + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked_elsewhere += 1
                continue
            pending = []
            for row in by_source[source['source_id']]:
                record_path = root / row['record_ref']
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    assert record['generation_identity_sha256'] == identity_hash
                    assert record['sample_plan_row'] == row
                    verify_array(root / row['observation_ref'], record['observation_sha256'])
                    resumed += 1
                else:
                    pending.append(row)
            if not pending:
                continue
            clean = verify_array(root / source['clean']['ref'], source['clean']['sha256'])
            with torch.no_grad():
                image = torch.from_numpy(clean[None]).to(a.device)
                prepared = sim.prepare_image(image)
                for row in pending:
                    sample_started = time.perf_counter()
                    raw, raw_hashes = raw_noise(row)
                    xi = Realization(*(tensor.to(a.device) for tensor in raw))
                    # No clamp, no extra noise, no old Y, and no precision conversion.
                    result = sim.render_prepared(prepared, xi, 4.0)
                    sync(a.device)
                    output = result.output[0].cpu().numpy()
                    if output.dtype != np.float32 or output.shape != (3, 256, 256) or not np.isfinite(output).all():
                        raise ValueError(f'Nonfinite/invalid observation at {row["sample_id"]}')
                    encoded = npy_bytes(output)
                    # A prior interrupted write has no record; require same bytes on replay.
                    atomic(root / row['observation_ref'], encoded)
                    record = {'sample_plan_row': row, 'generation_identity_sha256': identity_hash,
                              'observation_sha256': digest(encoded), 'observation_pixel_sha256': digest(output.tobytes()),
                              'raw_sha256': raw_hashes, 'observation_min': float(output.min()),
                              'observation_max': float(output.max()), 'finite': True,
                              'elapsed_seconds': time.perf_counter() - sample_started,
                              'device': a.device, 'host': socket.gethostname(), 'pid': os.getpid()}
                    write_json(root / row['record_ref'], record)
                    made += 1
                    del raw, xi, result, output
                    if a.max_samples and made >= a.max_samples:
                        break
                del prepared, image
            progress = {'status': 'GENERATING', 'device': a.device, 'made': made, 'resumed': resumed,
                        'locked_elsewhere': locked_elsewhere, 'elapsed_seconds': time.perf_counter() - started,
                        'last_source': source['source_id'], 'pid': os.getpid()}
            write_json(root / 'workers' / f'{a.device.replace(":", "_")}.json', progress, frozen=False)
            if made and made % 100 == 0:
                print(json.dumps(progress), flush=True)
            if a.max_samples and made >= a.max_samples:
                break
    result = {'status': 'WORKER_FINISHED', 'device': a.device, 'made': made, 'resumed': resumed,
              'locked_elsewhere': locked_elsewhere, 'elapsed_seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.npu.max_memory_allocated(a.device) if a.device.startswith('npu') else None}
    write_json(root / 'workers' / f'{a.device.replace(":", "_")}.json', result, frozen=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--simulator-root', default='/data2/yanglin/v5_v1_simulator_20261020')
    parser.add_argument('--device', required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=0, help='Optional bounded generation; never creates READY by itself')
    generate(parser.parse_args())

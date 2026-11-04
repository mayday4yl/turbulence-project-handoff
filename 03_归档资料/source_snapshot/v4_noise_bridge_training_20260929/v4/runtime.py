"""Explicit CUDA FP32 runtime, provenance and temporary checkpoint roundtrip."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def verify_imports(root, config):
    import physical_bridge.model as base_model
    import physical_bridge.simulator as base_simulator
    import v4.model as model
    import v4.core as core
    root = Path(root).resolve()
    modules = [base_model, base_simulator, model, core]
    actual = {}
    for module in modules:
        p = Path(module.__file__).resolve()
        if not p.is_relative_to(root):
            raise RuntimeError(f'Wrong import source: {module.__name__}: {p}')
        actual[module.__name__] = {'path':str(p),'sha256':sha256(p)}
    for rel, expected in json.loads((root/'configs/base_source_identity.json').read_text()).items():
        if sha256(root/rel) != expected:
            raise RuntimeError(f'Frozen V3 snapshot mismatch: {rel}')
    for name, expected in config['asset_sha256'].items():
        if sha256(Path(config['asset_dir'])/name) != expected:
            raise RuntimeError(f'Simulator asset mismatch: {name}')
    return actual


def gpu_guard(uuid):
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    conflicts = [row for row in rows if uuid in row and int(row.split(',')[0]) != os.getpid()]
    if conflicts:
        raise RuntimeError(f'GPU acquired by another process: {conflicts}')
    return rows


def configure_cuda(seed):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly selected CUDA GPU required')


def checkpoint_payload(model, optimizer, noise_generator, step_count, config):
    return {'purpose':'TEMPORARY_CORRECTNESS_ONLY_NOT_FORMAL_CHECKPOINT',
            'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
            'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
            'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all(),
            'noise_rng':noise_generator.get_state(), 'optimizer_steps':step_count,
            'config':config}


def restore_checkpoint(payload, model, optimizer, noise_generator):
    if payload['purpose'] != 'TEMPORARY_CORRECTNESS_ONLY_NOT_FORMAL_CHECKPOINT':
        raise ValueError('Wrong checkpoint purpose')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng']])
    noise_generator.set_state(payload['noise_rng'].cpu())

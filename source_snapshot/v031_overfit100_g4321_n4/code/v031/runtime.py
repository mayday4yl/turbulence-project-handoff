"""Stage-one provenance helpers. No training or optimizer updates."""
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if hasattr(torch, 'npu'):
        torch.npu.manual_seed_all(seed)


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_source(config):
    import physical_bridge
    root = Path(config['base_source'])
    if not Path(physical_bridge.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Wrong physical_bridge import')
    hashes = {name: sha256(root/name) for name in config['base_hashes']}
    if hashes != config['base_hashes']:
        raise RuntimeError('Pinned source hash mismatch')
    return hashes


def build(config, device):
    from physical_bridge.model import DualHeadUNet
    from physical_bridge.simulator import PiRNP2SSimulator
    verify_source(config)
    seed_all(config['seed'])
    model = DualHeadUNet(**config['network']).to(device)
    simulator = PiRNP2SSimulator(config['asset_dir'], **config['physics'], device=device)
    simulator.eval().requires_grad_(False)
    return model, simulator


def configure_npu(config):
    """Explicit execution/precision switches; leaves all simulator math intact."""
    runtime = config['npu_execution']
    for name,key in [('TASK_QUEUE_ENABLE','task_queue_enable'),('ASCEND_LAUNCH_BLOCKING','launch_blocking')]:
        if os.environ.get(name) != str(runtime[key]):
            raise RuntimeError(f'{name} must be set to {runtime[key]} before importing torch_npu; use run_env.sh')
    torch.npu.set_device('npu:0')
    # Apply options after lazy initialization, which can set backend defaults.
    torch.npu.init()
    torch.npu.set_compile_mode(jit_compile=runtime['jit_compile'])
    # The 910B/CANN dynamic Conv2DBackpropInput tiler rejects the 33x33
    # dictionary kernel in the real recurrent graph. Compile only that low-level
    # operator via its JIT route; global JIT stays disabled for complex FFTs.
    # This is an execution choice, not a change to simulator mathematics or VJP.
    torch.npu.set_option({
        'ACL_PRECISION_MODE':runtime['precision_mode'],
        'NPU_FUZZY_COMPILE_BLACKLIST':','.join(runtime['force_jit_ops']),
    })
    torch.npu.config.allow_internal_format=runtime['allow_internal_format']
    torch.npu.conv.allow_hf32=runtime['conv_hf32']
    torch.npu.matmul.allow_hf32=runtime['matmul_hf32']
    if hasattr(torch.npu,'aclnn'):
        torch.npu.aclnn.allow_hf32=runtime['aclnn_hf32']
    return dict(runtime)

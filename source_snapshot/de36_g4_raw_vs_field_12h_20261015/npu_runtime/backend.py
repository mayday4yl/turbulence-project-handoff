"""Device setup and resume state. No simulator or method mathematics live here."""
import contextlib
import os
import random
from pathlib import Path
import numpy as np
import torch


def configure(device_index=2, precision='bf16'):
    if isinstance(device_index, bool) or device_index not in (2,3):
        raise ValueError('Only de36 physical NPU 2,3 are authorized')
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0,1,2,3')
    if visible != '0,1,2,3':
        raise RuntimeError('Set ASCEND_RT_VISIBLE_DEVICES=0,1,2,3; device_index is physical')
    os.environ.setdefault('TASK_QUEUE_ENABLE', '0')
    os.environ.setdefault('ASCEND_LAUNCH_BLOCKING', '0')
    import torch_npu
    torch.npu.set_device(device_index)
    torch.npu.init()
    torch.npu.set_compile_mode(jit_compile=False)
    compiler_cache = Path(os.environ.get('V4_NPU_KERNEL_CACHE',
                             str(Path(__file__).resolve().parents[1] / 'cache' / 'npu_kernels')))
    compiler_cache.mkdir(parents=True, exist_ok=True)
    torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype',
                          'NPU_FUZZY_COMPILE_BLACKLIST':'Conv2DBackpropInput',
                          'ACL_OP_COMPILER_CACHE_MODE':'enable',
                          'ACL_OP_COMPILER_CACHE_DIR':str(compiler_cache)})
    torch.npu.config.allow_internal_format = True
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    if hasattr(torch.npu, 'aclnn'):
        torch.npu.aclnn.allow_hf32 = False
    if precision not in ('bf16','fp16','fp32'):
        raise ValueError('Unknown precision')
    if precision == 'bf16' and not torch.npu.is_bf16_supported():
        raise RuntimeError('BF16 unavailable; select explicitly accepted FP16/FP32 configuration')
    return {'device':f'npu:{device_index}', 'precision':precision,
            'torch':str(torch.__version__), 'torch_npu':str(torch_npu.__version__),
            'device_properties':str(torch.npu.get_device_properties(device_index)),
            'jit_compile':False, 'conv_backward_jit':True, 'internal_format':True,
            'physics_dtype':'float32', 'hf32':False,
            'compiler_cache_mode':'enable', 'compiler_cache_dir':str(compiler_cache)}


def autocast(device, precision='bf16'):
    device = torch.device(device)
    if precision == 'fp32' or device.type == 'cpu':
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype={'bf16':torch.bfloat16,'fp16':torch.float16}[precision])


def make_scaler(device='npu:0', precision='bf16'):
    device = torch.device(device)
    enabled = precision == 'fp16'
    if device.type == 'npu':
        return torch.npu.amp.GradScaler(enabled=enabled)
    return torch.amp.GradScaler(device.type, enabled=enabled)


def synchronize(device='npu:0'):
    device = torch.device(device)
    if device.type == 'npu': torch.npu.synchronize(device)
    elif device.type == 'cuda': torch.cuda.synchronize(device)


def empty_cache(device='npu:0'):
    device = torch.device(device)
    if device.type == 'npu': torch.npu.empty_cache()
    elif device.type == 'cuda': torch.cuda.empty_cache()


def capture_rng(noise_generator=None, device='npu:0'):
    device = torch.device(device)
    result = {'python':random.getstate(), 'numpy':np.random.get_state(),
              'cpu':torch.get_rng_state(), 'device_type':device.type}
    if device.type == 'npu': result['accelerator'] = torch.npu.get_rng_state(device).cpu()
    elif device.type == 'cuda': result['accelerator'] = torch.cuda.get_rng_state(device).cpu()
    if noise_generator is not None: result['eta'] = noise_generator.get_state().cpu()
    return result


def restore_rng(state, noise_generator=None, device='npu:0'):
    device = torch.device(device)
    if state['device_type'] != device.type: raise ValueError('Resume backend mismatch')
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if device.type == 'npu': torch.npu.set_rng_state(state['accelerator'].cpu(), device)
    elif device.type == 'cuda': torch.cuda.set_rng_state(state['accelerator'].cpu(), device)
    if noise_generator is not None: noise_generator.set_state(state['eta'].cpu())

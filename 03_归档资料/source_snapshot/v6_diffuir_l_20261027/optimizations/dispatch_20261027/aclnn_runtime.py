"""Isolated ACLNN candidate: exact PSF adjoint removes the large-kernel input-backward requirement."""
from pathlib import Path
import torch
from scripts.runtime_utils import ROOT

def configure(device='npu:0'):
    import torch_npu
    torch.npu.set_device(device);torch.npu.init();torch.npu.set_compile_mode(jit_compile=False)
    cache=ROOT/'cache/ab_dispatch_aclnn_kernels';cache.mkdir(parents=True,exist_ok=True)
    torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype',
        'ACL_OP_COMPILER_CACHE_MODE':'enable','ACL_OP_COMPILER_CACHE_DIR':str(cache)})
    torch.npu.config.allow_internal_format=True
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
    if hasattr(torch.npu,'aclnn'):torch.npu.aclnn.allow_hf32=False
    return {'dtype':'FP32','hf32':False,'internal_format':True,'jit_compile':False,
        'conv_backward_jit_exception':None,'explicit_cpu_fallback':False}

"""Runtime selection. CUDA formal training requires a future explicit authorization."""
import torch
from npu_runtime.backend import configure


def setup(index,precision,backend='npu'):
    if backend=='npu':return configure(index,precision)
    if backend!='cuda':raise ValueError('Unsupported formal backend')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(index)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 unsupported; acceptance needed')
    return {'device':f'cuda:{index}','precision':precision,'torch':str(torch.__version__),
            'cuda':torch.version.cuda,'device_properties':str(torch.cuda.get_device_properties(index)),
            'physics_dtype':'float32','TF32':False,'backend':'cuda'}

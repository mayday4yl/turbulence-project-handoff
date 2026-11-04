"""Isolated FP32 primitive dispatch diagnostic; never selects formal settings."""
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from runtime.environment import configure, make_simulator, synchronize, memory_stats
from training.state import atomic_json

out = ROOT/'reports/probe_conv_dispatch_native_fresh.json'
if out.exists():
    raise RuntimeError('Keep prior evidence')
report = {'status': 'RUNNING', 'scope': 'isolated accepted Simulator FP32 primitive, dispatch diagnostic only',
          'formal_optimizer_updates': 0, 'started_unix': time.time()}
atomic_json(report, out)
try:
    report['base_environment'] = configure('npu:0', conv_backward_jit=False)
    # Only this isolated process changes operator compilation dispatch. Physics,
    # source, tensors and dtype remain unchanged; production entry is untouched.
    report['diagnostic_override'] = {'NPU_FUZZY_COMPILE_BLACKLIST': 'NEVER_SET_IN_THIS_PROCESS', 'jit_compile': False,
                                     'simulator_dtype': 'float32', 'hf32': False}
    sim = make_simulator('npu:0').eval()
    for param in sim.parameters():
        param.requires_grad_(False)
    torch.manual_seed(20261021)
    image = torch.rand(2,3,256,256).to('npu:0').requires_grad_()
    report['measurements'] = []
    for repeat in range(2):
        image.grad = None
        synchronize('npu:0'); start = time.perf_counter()
        prepared = sim.prepare_image(image)
        loss = prepared.image_mean.square().mean()+prepared.dictionary_images.square().mean()
        synchronize('npu:0'); forward = time.perf_counter()-start
        start = time.perf_counter()
        loss.backward()
        synchronize('npu:0'); backward = time.perf_counter()-start
        if not torch.isfinite(image.grad).all().item():
            raise FloatingPointError('Nonfinite primitive gradient')
        report['measurements'].append({'repeat': repeat, 'forward_seconds': forward,
            'backward_seconds': backward, 'gradient_norm': image.grad.norm().item(), **memory_stats('npu:0')})
        del prepared, loss
        atomic_json(report, out)
    report['status'] = 'SUPPORTED_PRIMITIVE_ONLY_NUMERICAL_PARITY_PENDING'
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
report['finished_unix'] = time.time()
atomic_json(report, out)
print(json.dumps(report), flush=True)

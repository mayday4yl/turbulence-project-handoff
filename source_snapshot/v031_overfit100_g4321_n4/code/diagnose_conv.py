"""Finite execution-kernel diagnostic; no model, simulator rewrite, or updates."""
import argparse
import json
from pathlib import Path
import traceback
import torch
import torch_npu
from v031.runtime import save_json
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['strict_nd','strict_jit'],required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output)
torch.npu.set_device('npu:0')
torch.npu.set_compile_mode(jit_compile=a.mode=='strict_jit')
torch.npu.config.allow_internal_format=a.mode=='strict_jit'
torch.npu.conv.allow_hf32=False
torch.npu.matmul.allow_hf32=False
if hasattr(torch.npu,'aclnn'):torch.npu.aclnn.allow_hf32=False
r={'mode':a.mode,'zero_optimizer_updates':True,'cases':[],'source':Path(__file__).read_text()}
try:
    from v031.checkpoint import capture_rng_state,restore_rng_state
    state=capture_rng_state(include_npu=True)
    ref=torch.randn(64,device='npu:0')
    restore_rng_state(state)
    replay=torch.randn(64,device='npu:0')
    r['real_npu_rng_exact_replay']=bool(torch.equal(ref,replay))
    r['settings']={'internal_format_declared':a.mode=='strict_jit',
       'conv_hf32':torch.npu.conv.allow_hf32,'matmul_hf32':torch.npu.matmul.allow_hf32,
       'jit_compile':not torch.npu.is_jit_compile_false()}
    save_json(out,r)
    for channels in (1,100):
        x=torch.randn(3,1,288,288,device='npu:0',requires_grad=True)
        w=torch.randn(channels,1,33,33,device='npu:0')
        y=torch.nn.functional.conv2d(x,w)
        y.square().mean().backward()
        g=x.grad
        torch.npu.synchronize()
        r['cases'].append({'kernel':[channels,1,33,33],'output_shape':list(y.shape),'grad_norm':float(g.norm().cpu()),'finite':bool(torch.isfinite(g).all().cpu())})
        save_json(out,r);print(json.dumps(r['cases'][-1]),flush=True)
    r['passed']=True
except BaseException as exc:
    r.update({'passed':False,'exception':repr(exc),'traceback':traceback.format_exc()})
    raise
finally:save_json(out,r)

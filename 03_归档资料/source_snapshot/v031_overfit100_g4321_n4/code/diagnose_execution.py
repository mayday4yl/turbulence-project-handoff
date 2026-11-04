"""One bounded backend experiment; explicit precision, no optimizer or updates."""
import argparse
import json
from pathlib import Path
import time
import traceback
import torch
import torch_npu
from v031.runtime import build, configure_npu, save_json
from v031.checkpoint import capture_rng_state
from v031.core import RolloutModule
from v031.data import StartDataset

p=argparse.ArgumentParser()
p.add_argument('--initialization',choices=['eager','deferred_jit'],required=True)
p.add_argument('--case',choices=['random','prepared','recursive'],required=True)
p.add_argument('--base-grad-format',action='store_true')
p.add_argument('--force-jit-conv',action='store_true')
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=root/a.output
if out.exists():raise FileExistsError(out)
cfg=json.loads((root/'configs/stage1_acceptance.json').read_text())
r={'arguments':vars(a),'source':Path(__file__).read_text(),'configuration':cfg,
   'restoration_optimizer_updates':0,'events':[]}
def event(name,**details):
    row={'event':name,'time':time.time(),**details}
    r['events'].append(row);save_json(out,r);print(json.dumps(row),flush=True)
def settings():
    return {'initialized':torch.npu.is_initialized(),
      **{name:str(torch_npu._C._npu_getOption(name)) for name in ['jitCompile','ALLOW_INTERNAL_FORMAT','ALLOW_CONV_HF32','ALLOW_MATMUL_HF32','ACL_PRECISION_MODE','ACL_OP_SELECT_IMPL_MODE','NPU_FUZZY_COMPILE_BLACKLIST']}}
def tensor_info(t):
    return {'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),'format':int(torch_npu.get_npu_format(t)),'contiguous':t.is_contiguous()}
try:
    torch.set_num_threads(4)
    if a.initialization=='eager':configure_npu(cfg)
    else:
        torch.npu.set_device('npu:0')
        torch.npu.set_compile_mode(jit_compile=True)
        torch.npu.config.allow_internal_format=True
        torch.npu.conv.allow_hf32=False
        torch.npu.matmul.allow_hf32=False
        if hasattr(torch.npu,'aclnn'):torch.npu.aclnn.allow_hf32=False
    if a.force_jit_conv:
        torch.npu.set_option({'NPU_FUZZY_COMPILE_BLACKLIST':'Conv2DBackpropInput'})
    event('before_rng_capture',settings=settings())
    state=capture_rng_state(include_npu=True)
    event('after_rng_capture',settings=settings(),api_jit_compile_false=bool(torch.npu.is_jit_compile_false()))
    assert not torch.npu.conv.allow_hf32 and not torch.npu.matmul.allow_hf32
    if a.case=='random':
        for channels in (1,100):
            x=torch.randn(3,1,288,288,device='npu:0',requires_grad=True)
            w=torch.randn(channels,1,33,33,device='npu:0')
            y=torch.nn.functional.conv2d(x,w)
            event('forward',channels=channels,input=tensor_info(x),weight=tensor_info(w),output=tensor_info(y))
            y.register_hook(lambda g: (event('incoming_gradient',tensor=tensor_info(g)) or g))
            y.square().mean().backward()
            torch.npu.synchronize()
            event('backward',channels=channels,finite=bool(torch.isfinite(x.grad).all().cpu()),norm=float(x.grad.norm().cpu()))
            del x,w,y
    elif a.case=='prepared':
        model,sim=build(cfg,torch.device('npu:0'))
        event('after_build',settings=settings(),mu=tensor_info(sim.mu),basis=tensor_info(sim.basis_psf))
        for branch in ('mean','dictionary'):
            x=torch.rand(1,3,256,256,device='npu:0',requires_grad=True)
            prepared=sim.prepare_image(x)
            y=prepared.image_mean if branch=='mean' else prepared.dictionary_images
            event('forward',branch=branch,input=tensor_info(x),output=tensor_info(y))
            y.register_hook(lambda g: (event('incoming_gradient',tensor=tensor_info(g)) or g))
            y.square().mean().backward()
            torch.npu.synchronize()
            event('backward',branch=branch,finite=bool(torch.isfinite(x.grad).all().cpu()),norm=float(x.grad.norm().cpu()))
            del x,prepared,y
    else:
        original_conv=torch.nn.functional.conv2d
        counter=[0]
        def traced_conv(x,w,*args,**kwargs):
            y=original_conv(x,w,*args,**kwargs)
            if tuple(w.shape[-2:])==(33,33):
                counter[0]+=1;index=counter[0]
                event('simulator_convolution',index=index,input=tensor_info(x),weight=tensor_info(w),output=tensor_info(y))
                if y.requires_grad:
                    def incoming(g):
                        event('simulator_convolution_gradient',index=index,tensor=tensor_info(g),settings=settings())
                        return torch_npu.npu_format_cast(g.contiguous(),0) if a.base_grad_format else g
                    y.register_hook(incoming)
            return y
        torch.nn.functional.conv2d=traced_conv
        model,sim=build(cfg,torch.device('npu:0'));model.train()
        dataset=StartDataset(cfg['dataset_root'],cfg['diagnostic_subset'])
        sample=next(dataset[i] for i,x in enumerate(dataset.records) if x['initial_G']==4)
        C,Y,G=(sample[key].unsqueeze(0).to('npu:0') for key in ('clean','observation','G'))
        rollout=RolloutModule(model,sim,.2)
        result=rollout(C,Y,G,retain_intermediates=True)
        event('recursive_backward_start',settings=settings())
        result['trace'][-1]['clean_losses'].mean().backward()
        torch.npu.synchronize()
        event('recursive_backward_complete',norms=[{key:float(t[key].grad.norm().cpu()) if t[key].grad is not None else None for key in ('clean','epsilon_zernike','epsilon_tilt')} for t in result['trace'][:-1]])
    r['passed']=True
except BaseException as e:
    r.update({'passed':False,'exception':repr(e),'traceback':traceback.format_exc()})
    raise
finally:save_json(out,r)

"""Bounded identity-gradient layout diagnostic; never trains or changes physics."""
import json
from pathlib import Path
import traceback
import torch
import torch_npu
from v031.runtime import build,configure_npu,save_json
from v031.core import RolloutModule
from v031.data import StartDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/layout_01.json'
if OUT.exists():raise FileExistsError(OUT)
cfg=json.loads((ROOT/'configs/stage1_acceptance.json').read_text())
r={'zero_optimizer_updates':True,'hooks':[],'source':Path(__file__).read_text()}
try:
    configure_npu(cfg)
    model,sim=build(cfg,torch.device('npu:0'))
    model.train()
    r['buffer_formats']={n:{'shape':list(t.shape),'stride':list(t.stride()),'format':torch_npu.get_npu_format(t)} for n,t in [('mu',sim.mu),('basis_psf',sim.basis_psf)]}
    original=sim.prepare_image
    def prepare(image):
        prepared=original(image)
        for name,value in [('mean',prepared.image_mean),('dictionary',prepared.dictionary_images)]:
            def identity_layout(grad,name=name):
                info={'branch':name,'shape':list(grad.shape),'stride':list(grad.stride()),'format':torch_npu.get_npu_format(grad),'dtype':str(grad.dtype)}
                converted=torch_npu.npu_format_cast(grad.contiguous(),0)
                info.update({'converted_stride':list(converted.stride()),'converted_format':torch_npu.get_npu_format(converted)})
                r['hooks'].append(info);save_json(OUT,r)
                return converted
            value.register_hook(identity_layout)
        return prepared
    sim.prepare_image=prepare
    dataset=StartDataset(cfg['dataset_root'],cfg['diagnostic_subset'])
    sample=next(dataset[i] for i,x in enumerate(dataset.records) if x['initial_G']==4)
    C,Y,G=(sample[key].unsqueeze(0).to('npu:0') for key in ('clean','observation','G'))
    out=RolloutModule(model,sim,.2)(C,Y,G,retain_intermediates=True)
    out['trace'][-1]['clean_losses'].mean().backward()
    torch.npu.synchronize()
    r['native_gradient_norms']=[{key:float(t[key].grad.norm().cpu()) if t[key].grad is not None else None for key in ('clean','epsilon_zernike','epsilon_tilt')} for t in out['trace'][:-1]]
    r['passed']=all(v is not None and v>0 for row in r['native_gradient_norms'] for v in row.values())
except BaseException as e:
    r.update({'passed':False,'exception':repr(e),'traceback':traceback.format_exc()})
    raise
finally:save_json(OUT,r)

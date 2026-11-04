"""Read only new paired checkpoints on CPU; no NPU initialization."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['ASCEND_RT_VISIBLE_DEVICES']='2,3'
from pathlib import Path
import json,hashlib,torch,math
# Explicitly register the NPU pickle CPU map adapter before the FIRST torch.load.
import torch_npu
R=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
frozen=json.loads((R/'protocol/FROZEN.json').read_text());warm=torch.load(R/'runs/common_warm.pt',map_location='cpu',weights_only=False);assert sha(R/'runs/common_warm.pt')==frozen['common_warm_sha256'];reports={}
assert all(x.device.type=='cpu' for x in warm['model'].values())
assert not torch.npu.is_initialized(), 'CPU deserialization must not initialize an NPU'
for arm in ['Raw','Field']:
 for u in [0,500,1000]:
  p=R/f'runs/{arm}/U{u:05d}.pt';j=torch.load(p,map_location='cpu',weights_only=False);assert j['update']==u and j['arm']==arm and j['config']['G']==4 and j['frozen_sha256']==sha(R/'protocol/FROZEN.json')
  model=j['model'];opt=j['optimizer'];assert all(x.device.type=='cpu' and torch.isfinite(x).all() for x in model.values())
  count=sum(x.numel() for n,x in model.items() if n not in ['alpha_scale','u_scale']);assert count==651417
  for state in opt['state'].values():
   for value in state.values():
    if torch.is_tensor(value):assert value.device.type=='cpu' and torch.isfinite(value).all()
  steps=[int(x['step']) for x in opt['state'].values()];assert (not steps if u==0 else min(steps)==max(steps)==u)
  if u==0:
   assert all(torch.equal(model[n],x) for n,x in warm['model'].items())
   for name in ['zernike_head.weight','zernike_head.bias','tilt_head.weight','tilt_head.bias']:assert torch.count_nonzero(model[name])==0
  reports[f'{arm}_U{u:05d}']={'sha256':sha(p),'update':u,'parameter_count_excluding_field_scale_buffers':count,'model_finite':True,'optimizer_finite':True,'optimizer_state_count':len(steps),'optimizer_step_min':min(steps) if steps else None,'optimizer_step_max':max(steps) if steps else None,'common_warm_exactly_copied':True if u==0 else 'checked_at_U0','lr':opt['param_groups'][0]['lr']}
  del j,model,opt
assert not torch.npu.is_initialized(), 'No NPU context allowed in CPU audit'
(OUT/'CHECKPOINT_AUDIT.json').write_text(json.dumps({'status':'PASS','execution_device':'CPU_only','torch':str(torch.__version__),'checkpoints':reports},indent=2)+'\n');print('CPU checkpoint audit PASS')

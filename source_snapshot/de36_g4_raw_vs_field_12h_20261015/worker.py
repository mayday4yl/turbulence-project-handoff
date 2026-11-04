from common import *
import argparse,math,signal,traceback,gc,statistics,random
import numpy as np
import torch
from npu_runtime.backend import configure,synchronize,capture_rng
from physical_bridge.simulator import PiRNP2SSimulator,Realization
from physical_bridge.metrics import per_sample_psnr as psnr,per_sample_ssim as ssim
from v4.refinement import module_hash
from core import PilotUNet,Prediction,rollout,optimizer,parameter_counts,render_prediction,render_field,l1
from data import Store
from refine_adapter import refine_stage

P=argparse.ArgumentParser();P.add_argument('phase',choices=['accept','warm','bench','train']);P.add_argument('--arm',choices=['Raw','Field'],default='Raw');P.add_argument('--device',type=int,required=True);a=P.parse_args()
DEVICE=None
LPIPS=None

def load_lpips():
 global LPIPS
 if LPIPS is None:
  cache=Path('/data2/yanglin/v4_restormer_train2000_npu_20k_20261003/assets/torch')
  weight=cache/'hub/checkpoints/alexnet-owt-7be5be79.pth'
  assert weight.is_file(),'No LPIPS download permitted'
  os.environ['TORCH_HOME']=str(cache)
  from physical_bridge.metrics import LPIPSMetric
  LPIPS=LPIPSMetric(DEVICE,net='alex')
  save_json(ROOT/f'reports/LPIPS_{a.phase}_{a.arm}.json',{'net':'alex','weight_path':str(weight),'weight_sha256':digest(weight),'download':False})
 return LPIPS

def save_pt(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
 torch.save(payload,temp);os.replace(temp,path)

def setup():
 global DEVICE
 env=configure(a.device,'fp32');DEVICE=torch.device(env['device']);torch.set_num_threads(4)
 torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);random.seed(CONFIG['seed'])
 save_json(ROOT/f'reports/ENV_{a.phase}_{a.arm}.json',env)
 return PiRNP2SSimulator(CONFIG['asset_dir'],device=DEVICE,**CONFIG['physics'])

def scales():return json.loads((ROOT/'protocol/FIELD_SCALES.json').read_text())
def model(arm,warm=True):
 torch.manual_seed(CONFIG['seed']);m=PilotUNet(arm,scales() if arm=='Field' else None)
 if warm:
  p=torch.load(ROOT/'runs/common_warm.pt',map_location='cpu',weights_only=False)
  mismatch=m.load_state_dict(p['model'],strict=False)
  assert not mismatch.unexpected_keys and set(mismatch.missing_keys)==({'alpha_scale','u_scale'} if arm=='Field' else set())
 return m.to(DEVICE)
def finite(x):return bool(torch.isfinite(x).all())
def grad_stat(x):
 assert x is not None and finite(x)
 return {'l2':float(x.norm()),'max_abs':float(x.abs().max()),'nonzero_fraction':float((x!=0).float().mean()),'finite':True}
def delta(x,y):
 d=(x-y).abs();return {'mean_abs':float(d.mean()),'max_abs':float(d.max())}
def ckpt(m,opt,u,path):
 save_pt(path,{'model':m.state_dict(),'optimizer':opt.state_dict(),'update':u,'arm':m.arm,'config':CONFIG,'rng':capture_rng(device=DEVICE),'epoch':time.time(),'frozen_sha256':digest(ROOT/'protocol/FROZEN.json') if (ROOT/'protocol/FROZEN.json').exists() else None})

def accept(sim,store):
 report={'rows':[],'scale_realizations':[],'renderer_quirks':{'tilt_irfft2':'literal real-pair input; dim=(1,2), s=(512,512)','warp':'divide by image_size-1 then align_corners=False; preserved','p2s':'literal fc2 reuse; preserved'},'physics':sim.protocol()}
 # Existing training-side xi only, no generated trajectories or GT state supervision.
 sz=torch.zeros(36,dtype=torch.float64);su=torch.zeros(2,dtype=torch.float64)
 with torch.no_grad():
  for r in store.rows['train'][::2][:32]:
   check_deadline(9000)
   xi=Realization(store.array(r,'epsilon_zernike').to(DEVICE),store.array(r,'epsilon_tilt').to(DEVICE));state=sim.effective_state(xi,1.)
   sz+=state.zernike.square().mean((0,2,3)).cpu().double();su+=state.displacement.square().mean((0,1,2)).cpu().double()
   report['scale_realizations'].append({'trajectory_id':r['trajectory_id'],'z_sha':r['arrays']['epsilon_zernike']['sha256'],'t_sha':r['arrays']['epsilon_tilt']['sha256']})
  scale={'alpha_rms':(sz/32).sqrt().clamp_min(1e-6).tolist(),'u_rms':(su/32).sqrt().clamp_min(1e-6).tolist(),'n':32,'floor':1e-6,'provenance':report['scale_realizations'],'coordinate':'g_ref=1 before P2S / before grid normalization'}
  save_json(ROOT/'protocol/FIELD_SCALES.json',scale)
  for r in store.rows['train'][::2][:2]:
   c,y=store.batch(r,DEVICE);xi=Realization(store.array(r,'epsilon_zernike').to(DEVICE),store.array(r,'epsilon_tilt').to(DEVICE));ref=sim.effective_state(xi,1.);prep=sim.prepare_image(c)
   for g in (1,2,3,4):
    raw=sim.render_prepared(prep,xi,float(g));field=render_field(sim,prep,ref.zernike,ref.displacement,float(g))
    row={'trajectory_id':r['trajectory_id'],'g':g,'alpha':delta(raw.effective_state.zernike,field.effective_state.zernike),'u':delta(raw.effective_state.displacement,field.effective_state.displacement),'blur':delta(raw.blur_only,field.blur_only),'output':delta(raw.output,field.output)}
    report['rows'].append(row);save_json(ROOT/'reports/RENDER_PARITY.json',report)
    for k in ('blur','output'):
     assert row[k]['mean_abs']<=1e-5 and row[k]['max_abs']<=1e-3,row
   out=ROOT/'figures'/f"parity_{r['trajectory_id']}.npz";np.savez(out,clean=c.cpu().numpy(),raw=raw.output.cpu().numpy(),field=field.output.cpu().numpy())
  raw_m=model('Raw',False);field_m=model('Field',False)
  source=raw_m.state_dict();missing=field_m.load_state_dict(source,strict=False);assert set(missing.missing_keys)=={'alpha_scale','u_scale'}
  c,y=store.batch(store.rows['train'][0],DEVICE);cond=y.new_tensor([[4.,0.]])
  pr=raw_m(torch.cat([y,y],1),cond);pf=field_m(torch.cat([y,y],1),cond)
  assert torch.equal(pr.clean,pf.clean)
  for pp in (pr,pf):assert int(torch.count_nonzero(pp.alpha_or_z))==0 and int(torch.count_nonzero(pp.u_or_t))==0
  rr=render_prediction(sim,sim.prepare_image(pr.clean),pr,'Raw',4.);rf=render_prediction(sim,sim.prepare_image(pf.clean),pf,'Field',4.)
  report['zero_initialization']=delta(rr.output,rf.output);assert report['zero_initialization']['max_abs']==0
  # Existing accepted family adapter must exactly match our Raw forward.
  from v4.model import NoiseBridgeUNet
  original=NoiseBridgeUNet(None,**CONFIG['network']).to(DEVICE);original.load_state_dict(source)
  po=original(torch.cat([y,y],1),cond)
  assert torch.equal(po.clean,pr.clean) and torch.equal(po.epsilon_zernike,pr.alpha_or_z) and torch.equal(po.epsilon_tilt,pr.u_or_t)
  report['raw_forward_equals_existing_family']=True
  report['parameters']={'Raw':parameter_counts(raw_m),'Field':parameter_counts(field_m)}
 report.update(status='PASS',epoch=time.time(),physics_parameters_frozen=all(not p.requires_grad for p in sim.parameters()),source_hashes={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*.py')})
 save_json(ROOT/'reports/RENDER_PARITY.json',report)

def warm(store):
 m=model('Raw',False);opt=optimizer(m);start=time.monotonic();rows=[]
 for u in range(CONFIG['warm_updates']):
  check_deadline(9000);opt.zero_grad(set_to_none=True);loss_value=0;t=time.monotonic()
  for r in store.training_rows(u,'warm'):
   c,y=store.batch(r,DEVICE);p=m.clean_only(torch.cat([y,y],1),y.new_tensor([[4.,0.]]));loss=l1(p,c).mean()/4;loss.backward();loss_value+=float(loss.detach())
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();synchronize(DEVICE)
  if (u+1)%50==0:
   append_json(ROOT/'logs/warm.jsonl',{'update':u+1,'LC':loss_value,'seconds':time.monotonic()-t,'elapsed':time.monotonic()-start});print('warm',u+1,loss_value,flush=True)
 for name in ('zernike_head','tilt_head'):
  assert all(torch.count_nonzero(p)==0 for p in getattr(m,name).parameters())
 save_pt(ROOT/'runs/common_warm.pt',{'model':m.state_dict(),'warm_updates':CONFIG['warm_updates'],'optimizer_discarded':True,'common_model_hash':module_hash(m),'seconds':time.monotonic()-start})
 save_json(ROOT/'reports/COMMON_INITIALIZATION.json',{'sha256':digest(ROOT/'runs/common_warm.pt'),'updates':CONFIG['warm_updates'],'seconds':time.monotonic()-start,'shared_clean_condition_copied_exactly':True,'state_last_layers_zero':True,'formal_optimizer_reset':True,'not_D0_baseline':True})

def update(m,sim,store,opt,u,phase='formal',lr=None):
 check_deadline(1800);start=time.monotonic();synchronize(DEVICE);opt.zero_grad(set_to_none=True)
 if lr is not None:
  for group in opt.param_groups:group['lr']=lr
 lc=np.zeros(4);lr_values=np.zeros(4)
 for r in store.training_rows(u,phase):
  c,y=store.batch(r,DEVICE);out=rollout(m,sim,c,y);(out['loss']/4).backward()
  lc+=out['lc'].detach().cpu().numpy().reshape(4)/4;lr_values+=out['lr'].detach().cpu().numpy().reshape(4)/4
  del out,c,y
 norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();synchronize(DEVICE)
 return {'update':u+1,'LC':lc.tolist(),'LR':lr_values.tolist(),'loss':float(np.mean(lc+.2*lr_values)),'seconds':time.monotonic()-start,'gradient_norm_before_clip':float(norm),'lr':opt.param_groups[0]['lr'],'peak_allocated_bytes':torch.npu.max_memory_allocated(DEVICE),'peak_reserved_bytes':torch.npu.max_memory_reserved(DEVICE)}

def bench(sim,store):
 m=model(a.arm);m.train();opt=optimizer(m);c,y=store.batch(store.rows['train'][0],DEVICE)
 out=rollout(m,sim,c,y,True);terminal=l1(out['predictions'][-1].clean,c).mean();terminal.backward()
 first=out['predictions'][0];grads={n:grad_stat(x.grad) for n,x in zip(['first_clean','first_alpha_or_z','first_u_or_t'],first)}
 assert all(s['l2']>0 for s in grads.values()),grads
 grads['z_head_weight']=grad_stat(m.zernike_head.weight.grad);grads['tilt_head_weight']=grad_stat(m.tilt_head.weight.grad)
 assert all(not p.requires_grad and p.grad is None for p in sim.parameters())
 save_json(ROOT/f'reports/GRAD_{a.arm}.json',{'status':'PASS','source':'terminal_C4_L1_only','three_physical_transitions':True,'no_detach':True,'gradients':grads,'raw_tilt_sparse_zero_directions_allowed':True})
 del out,terminal,first,c,y;m.zero_grad(set_to_none=True);gc.collect();torch.npu.empty_cache();torch.npu.reset_peak_memory_stats(DEVICE)
 times=[]
 for u in range(5):
  row=update(m,sim,store,opt,u,'benchmark');row['timed']=u>=2;times.append(row);print('bench',a.arm,row,flush=True)
  save_json(ROOT/f'reports/BENCH_{a.arm}.json',{'status':'RUNNING','updates':times})
 # All timing states are discarded; load common initialization for inference timing.
 del m,opt;gc.collect();torch.npu.empty_cache();m=model(a.arm);m.eval()
 c,y=store.batch(store.rows['train'][0],DEVICE)
 with torch.no_grad():p=m(torch.cat([y,y],1),y.new_tensor([[4.,0.]]))
 refined,ref=refine_stage(m,sim,p,y)
 load_lpips();start=time.monotonic()
 with torch.no_grad():metrics,_=evaluate_one(m,sim,c,y,False)
 synchronize(DEVICE);off_seconds=time.monotonic()-start
 report={'status':'PASS','updates':times,'median_update_seconds':statistics.median(x['seconds'] for x in times[2:]),'max_update_seconds':max(x['seconds'] for x in times[2:]),'refinement_stage_seconds':ref['seconds'],'off_input_seconds':off_seconds,'discarded_test_updates':True,'training_side_refinement_timing':ref,'parameters':parameter_counts(m)}
 save_json(ROOT/f'reports/BENCH_{a.arm}.json',report)

def evaluate_one(m,sim,c,y,on):
 state=y;stages=[];images={'clean':c.detach().cpu().numpy(),'Y4':y.detach().cpu().numpy()};refs=[]
 for k,g in enumerate((4,3,2,1)):
  with torch.no_grad():
   p=m(torch.cat([state,y],1),y.new_tensor([[float(g),0.]]));prep=sim.prepare_image(p.clean);back=render_prediction(sim,prep,p,m.arm,4.)
   stage={'stage':k+1,'g':g,'psnr':float(psnr(p.clean,c)),'ssim':float(ssim(p.clean,c)),'LC':float(l1(p.clean,c)),'LR':float(l1(back.output,y)),
    'u_rms_g4':float(back.effective_state.displacement.square().mean().sqrt()),'u_max_abs_g4':float(back.effective_state.displacement.abs().max()),'grid_oob_g4':float((back.flow.abs()>1).any(-1).float().mean())}
   images[f'C{k+1}']=p.clean.detach().cpu().numpy()
   if g==1:stage['lpips']=float(load_lpips()(p.clean,c))
  if g>1:
   if on:
    p,ref=refine_stage(m,sim,p,y);ref['stage']=k+1;refs.append(ref)
   with torch.no_grad():
    transition=render_prediction(sim,prep,p,m.arm,float(g-1));state=transition.output;images[f'Y{g-1}_hat']=state.detach().cpu().numpy()
    if on:
     refs[-1].update(u_rms_next=float(transition.effective_state.displacement.square().mean().sqrt()),u_max_abs_next=float(transition.effective_state.displacement.abs().max()),grid_oob_next=float((transition.flow.abs()>1).any(-1).float().mean()))
  stages.append(stage)
 return {'stages':stages,'refinement':refs,'final_psnr':stages[-1]['psnr'],'final_ssim':stages[-1]['ssim'],'final_lpips':stages[-1]['lpips'],'stage_gain':stages[-1]['psnr']-stages[0]['psnr']},images

def evaluate(m,sim,store,u,on=False):
 frozen=json.loads((ROOT/'protocol/FROZEN.json').read_text());assert frozen['validation_metrics_seen_before_lock'] is False
 m.eval();rows=store.rows['val'];subset=frozen['refinement_trajectory_ids']
 if on:rows=[r for r in rows if r['trajectory_id'] in subset]
 dest=ROOT/f'runs/{a.arm}/eval_U{u:05d}_{"ON" if on else "OFF"}';dest.mkdir(parents=True,exist_ok=True);start=time.monotonic();results=[]
 for i,r in enumerate(rows):
  check_deadline(900);c,y=store.batch(r,DEVICE);synchronize(DEVICE);t=time.monotonic()
  row,images=evaluate_one(m,sim,c,y,on);synchronize(DEVICE);row.update(trajectory_id=r['trajectory_id'],source_id=r['source_id'],update=u,arm=m.arm,on=on,seconds=time.monotonic()-t)
  save_json(dest/f'{r["trajectory_id"]}.json',row);results.append(row)
  if r['trajectory_id'] in subset:np.savez(dest/f'{r["trajectory_id"]}.npz',**images)
 save_json(dest/'SUMMARY.json',{'count':len(rows),'seconds':time.monotonic()-start,'mean_final_psnr':float(np.mean([r['final_psnr'] for r in results])),'mean_final_ssim':float(np.mean([r['final_ssim'] for r in results])),'rows':[r['trajectory_id'] for r in rows]})
 m.train()

def train(sim,store):
 frozen=json.loads((ROOT/'protocol/FROZEN.json').read_text());n=frozen['N'];m=model(a.arm);opt=optimizer(m);u=0;start=time.monotonic()
 ckpt(m,opt,0,ROOT/f'runs/{a.arm}/U00000.pt');evaluate(m,sim,store,0)
 try:
  for u0 in range(n):
   check_deadline(frozen['post_training_reserve_seconds'])
   lr=1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*u0/max(1,n-1)))
   row=update(m,sim,store,opt,u0,lr=lr);u=u0+1
   append_json(ROOT/f'logs/{a.arm}_train.jsonl',row)
   if u%50==0 or u in (n//2,n):
    ckpt(m,opt,u,ROOT/f'runs/{a.arm}/U{u:05d}.pt');print('train',a.arm,u,row['loss'],flush=True)
   save_json(ROOT/f'runs/{a.arm}/STATUS.json',{'status':'TRAINING','update':u,'N':n,'epoch':time.time(),'elapsed_seconds':time.monotonic()-start})
   if u in (n//2,n):evaluate(m,sim,store,u)
  evaluate(m,sim,store,n,True)
  save_json(ROOT/f'runs/{a.arm}/STATUS.json',{'status':'COMPLETE','update':n,'N':n,'epoch':time.time(),'elapsed_seconds':time.monotonic()-start})
 except BaseException:
  # Complete optimizer boundary only; partial update may exist and must not be labeled comparable.
  save_json(ROOT/f'runs/{a.arm}/STOPPED.json',{'last_completed_update':u,'epoch':time.time(),'error':traceback.format_exc(),'checkpoint_policy':'Use only previously atomically saved common nodes; no partial update relabeling'})
  raise

def main():
 if a.phase=='train':
  frozen=json.loads((ROOT/'protocol/FROZEN.json').read_text())
  for name,expected in frozen['source_hashes'].items():assert digest(ROOT/name)==expected,name
  assert digest(ROOT/'runs/common_warm.pt')==frozen['common_warm_sha256']
  assert digest(ROOT/'protocol/SAMPLES.json')==frozen['sample_manifest_sha256']
  assert digest(ROOT/'protocol/FIELD_SCALES.json')==frozen['scale_sha256']
 check_deadline(1800);store=Store()
 if a.phase=='warm':
  configure(a.device,'fp32')
  global DEVICE
  DEVICE=torch.device(f'npu:{a.device}');torch.set_num_threads(4);warm(store)
 else:
  sim=setup()
  {'accept':accept,'bench':bench,'train':train}[a.phase](sim,store)
 print('PHASE_PASS',a.phase,a.arm,flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  save_json(ROOT/f'reports/FAILURE_{a.phase}_{a.arm}.json',{'epoch':time.time(),'traceback':traceback.format_exc()});raise

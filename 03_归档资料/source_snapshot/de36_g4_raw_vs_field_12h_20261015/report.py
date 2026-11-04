"""CPU-only report from committed per-input records; never runs a model."""
from common import *
import csv,collections
import numpy as np

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def paired(left,right):
 assert set(left)==set(right)
 by=collections.defaultdict(list)
 for k in left:by[left[k]['source_id']].append([left[k]['final_psnr']-right[k]['final_psnr'],left[k]['final_ssim']-right[k]['final_ssim']])
 per={s:np.mean(v,axis=0).tolist() for s,v in by.items()};x=np.array(list(per.values()))
 return {'source_count':len(x),'input_count':len(left),'mean_delta_psnr':float(x[:,0].mean()),'median_delta_psnr':float(np.median(x[:,0])),'win_rate_psnr':float((x[:,0]>0).mean()),'mean_delta_ssim':float(x[:,1].mean()),'per_source':per}

def records(arm,u,on):
 d=ROOT/f'runs/{arm}/eval_U{u:05d}_{"ON" if on else "OFF"}'
 if not (d/'SUMMARY.json').exists():return {}
 return {p.stem:read(p) for p in d.glob('*.json') if p.name!='SUMMARY.json'}

def main():
 frozen=read(ROOT/'protocol/FROZEN.json',{});asset=read(ROOT/'reports/ASSET_AUDIT.json',{});parity=read(ROOT/'reports/RENDER_PARITY.json',{})
 acceptance={a:{'gradient':read(ROOT/f'reports/GRAD_{a}.json'),'benchmark':read(ROOT/f'reports/BENCH_{a}.json')} for a in ('Raw','Field')}
 save_json(ROOT/'reports/GRAD_AND_BENCH.json',acceptance)
 text=['# G4 Raw versus Field: bounded paired experiment','',f"Frozen common updates: {frozen.get('N','not locked')}. Main population: val32 sources x2 realizations, G4 only. Refinement subset is separate.",'','No claim follows from re-degradation loss or refinement J reduction alone.','']
 summary={'common_nodes':{},'refinement_subset':{},'scope':'single seed, small G4-only pilot; no Test','frozen':frozen}
 allrows=[]
 for u in frozen.get('common_nodes',[]):
  arms={a:records(a,u,False) for a in ('Raw','Field')}
  if not all(len(v)==64 for v in arms.values()):continue
  node={}
  for arm,rr in arms.items():
   node[arm]={'PSNR':float(np.mean([r['final_psnr'] for r in rr.values()])),'SSIM':float(np.mean([r['final_ssim'] for r in rr.values()])),
    'LPIPS':float(np.mean([r['final_lpips'] for r in rr.values()])),
    'stage_PSNR':np.mean([[s['psnr'] for s in r['stages']] for r in rr.values()],axis=0).tolist(),
    'stage_LC':np.mean([[s['LC'] for s in r['stages']] for r in rr.values()],axis=0).tolist(),
    'stage_LR':np.mean([[s['LR'] for s in r['stages']] for r in rr.values()],axis=0).tolist()}
   for r in rr.values():allrows.append(r)
  node['Field_minus_Raw']=paired(arms['Field'],arms['Raw']);summary['common_nodes'][str(u)]=node
  text.append(f"U{u}: Raw PSNR {node['Raw']['PSNR']:.6f}, Field {node['Field']['PSNR']:.6f}; Field-Raw {node['Field_minus_Raw']['mean_delta_psnr']:+.6f} dB.")
 n=frozen.get('N');subset=frozen.get('refinement_trajectory_ids',[])
 if n:
  outputs={f'{a}-{on}':records(a,n,on=='ON') for a in ('Raw','Field') for on in ('OFF','ON')}
  outputs={k:{sid:r for sid,r in rr.items() if sid in subset} for k,rr in outputs.items()}
  if all(len(rr)==len(subset) and len(rr)>0 for rr in outputs.values()):
   for name,(l,r) in {'Field-Raw_OFF':('Field-OFF','Raw-OFF'),'Field-Raw_ON':('Field-ON','Raw-ON'),'Raw_ON-OFF':('Raw-ON','Raw-OFF'),'Field_ON-OFF':('Field-ON','Field-OFF')}.items():
    result=paired(outputs[l],outputs[r]);summary['refinement_subset'][name]=result;text.append(f"Subset {name}: {result['mean_delta_psnr']:+.6f} dB; {result['source_count']} sources; win rate {result['win_rate_psnr']:.3f}.")
   for k,rr in outputs.items():
    if k.endswith('-ON'):
     allrows.extend(rr.values())
     for sid,r in rr.items():
      off=outputs[k.replace('-ON','-OFF')][sid]
      if r['final_psnr']<off['final_psnr']:text.append(f"Negative refinement result: {k} {sid}: PSNR change {r['final_psnr']-off['final_psnr']:+.6f} dB despite selected stage J never increasing.")
   # Fixed four-source panels. Display clipping applies only to PNG, not metrics or npz.
   from PIL import Image,ImageDraw
   for sid in subset[:4]:
    cols=[]
    for key in ['Raw-OFF','Raw-ON','Field-OFF','Field-ON']:
     arm,on=key.split('-');path=ROOT/f'runs/{arm}/eval_U{n:05d}_{on}/{sid}.npz'
     z=np.load(path);col=Image.new('RGB',(256,6*280),'white');draw=ImageDraw.Draw(col)
     for j,label in enumerate(['clean','Y4','C1','C2','C3','C4']):
      x=np.clip(z[label][0].transpose(1,2,0),0,1);col.paste(Image.fromarray((x*255).round().astype(np.uint8)),(0,j*280+24));draw.text((5,j*280+5),f'{key} {label}',fill='black')
     cols.append(col)
    panel=Image.new('RGB',(1024,1680),'white')
    for j,col in enumerate(cols):panel.paste(col,(j*256,0))
    panel.save(ROOT/f'figures/{sid}_four_outputs.png')
 lifecycle=[json.loads(l) for l in (ROOT/'logs/lifecycle.jsonl').read_text().splitlines()] if (ROOT/'logs/lifecycle.jsonl').exists() else []
 finish=[r for r in lifecycle if r['event']=='finish'];summary['completed_phase_npu_card_hours']=sum(r['end_epoch']-r['start_epoch'] for r in finish)/3600
 summary['failed_or_active_phase_card_time_not_included']=len([r for r in lifecycle if r['event']=='start'])!=len(finish)
 summary['acceptance']=acceptance;summary['render_parity_status']=parity.get('status','not passed')
 save_json(ROOT/'reports/RESULTS.json',summary)
 if allrows:
  fields=['arm','on','update','source_id','trajectory_id','final_psnr','final_ssim','final_lpips','stage_gain','seconds']+[f'{m}{i}' for i in range(1,5) for m in ['PSNR','SSIM','LC','LR']]
  with (ROOT/'reports/per_input_metrics.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for r in allrows:
    row={k:r[k] for k in fields if k in r}
    for i,s in enumerate(r['stages'],1):
     for m in ['PSNR','SSIM','LC','LR']:row[f'{m}{i}']=s[m.lower() if m in ('PSNR','SSIM') else m]
    writer.writerow(row)
 text.extend(['',f"Completed phase NPU card-hours: {summary['completed_phase_npu_card_hours']:.4f}.",f"Render parity: {summary['render_parity_status']}.",'','All original simulator coordinate/FFT conventions were preserved. Field and Raw normalized proximity do not imply equal physical regularization strength.'])
 if not summary['common_nodes']:text.append('No completed paired formal validation node. Engineering evidence only; no restoration conclusion.')
 (ROOT/'reports/RESULT_REPORT.md').write_text('\n'.join(text)+'\n')
 (ROOT/'reports/ASSET_AND_IMPL_AUDIT.md').write_text('# Asset and implementation audit\n\n'+json.dumps(asset,indent=2)+'\n\nOnly the independent adapter/core/controller files are new. Exact original model/simulator/refinement hashes are in ASSET_AUDIT.json.\n')
if __name__=='__main__':main()

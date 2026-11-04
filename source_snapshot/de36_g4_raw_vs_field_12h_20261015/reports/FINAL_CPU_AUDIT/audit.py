"""Independent CPU aggregation/audit. No model execution or training changes."""
from pathlib import Path
import json,statistics as st,math,hashlib,csv,random,zipfile,time
import numpy as np
R=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def mean(x):return st.fmean(x)
def write_csv(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paired(left,right):
 assert set(left)==set(right);by={}
 for sid in left:
  assert left[sid]['source_id']==right[sid]['source_id'];source=left[sid]['source_id'];by.setdefault(source,[]).append({k:left[sid][k]-right[sid][k] for k in ['final_psnr','final_ssim','final_lpips','stage_gain']})
 source={s:{k:mean(x[k] for x in rr) for k in rr[0]} for s,rr in by.items()};v=[r['final_psnr'] for r in source.values()]
 rng=random.Random(20261015);bs=sorted(mean(rng.choices(v,k=len(v))) for _ in range(10000))
 return {'source_count':len(v),'input_count':len(left),'mean_PSNR_delta':mean(v),'median_PSNR_delta':st.median(v),'PSNR_wins':sum(x>0 for x in v),'PSNR_win_rate':mean(x>0 for x in v),'mean_SSIM_delta':mean(r['final_ssim'] for r in source.values()),'mean_LPIPS_delta':mean(r['final_lpips'] for r in source.values()),'mean_stage_gain_delta':mean(r['stage_gain'] for r in source.values()),'source_bootstrap95_PSNR_mean':[bs[249],bs[9749]],'per_source':source}

def main():
 frozen=read(R/'protocol/FROZEN.json');samples=read(R/'protocol/SAMPLES.json')['val'];ids=[r['trajectory_id'] for r in samples];expected={r['trajectory_id']:r['source_id'] for r in samples};subset=frozen['refinement_trajectory_ids'];assert len(ids)==64 and len(subset)==8
 data={};allrows=[];refrows=[];summary={'status':'RUNNING','N':1000,'main_population':'val32sources x2 existing realizations, G4, OFF','refinement_population':'8 different sources x1 existing realization, final U1000 only','nodes':{},'subset':{},'checkpoint_identities':{},'refinement':{},'runtime':{},'fixed_image_PSNR_checks':[]}
 for u in [0,500,1000]:
  node={}
  for arm in ['Raw','Field']:
   on_values=[False,True] if u==1000 else [False]
   for on in on_values:
    mode='ON' if on else 'OFF';d=R/f'runs/{arm}/eval_U{u:05d}_{mode}';s=read(d/'SUMMARY.json');wanted=subset if on else ids
    rows={p.stem:read(p) for p in d.glob('val_*.json')};assert set(rows)==set(wanted) and s['count']==len(wanted) and set(s['rows'])==set(wanted)
    for sid,r in rows.items():
     assert r['trajectory_id']==sid and r['source_id']==expected[sid] and r['update']==u and r['arm']==arm and r['on'] is on
     assert [v['g'] for v in r['stages']]==[4,3,2,1]
     assert all(math.isfinite(r[k]) for k in ['final_psnr','final_ssim','final_lpips','stage_gain'])
     assert len(r['refinement'])==(3 if on else 0)
     row={'update':u,'arm':arm,'mode':mode,'trajectory_id':sid,'source_id':r['source_id'],'final_psnr':r['final_psnr'],'final_ssim':r['final_ssim'],'final_lpips':r['final_lpips'],'C4_minus_C1':r['stage_gain'],'seconds':r['seconds']}
     for j,v in enumerate(r['stages'],1):
      for metric in ['psnr','ssim','LC','LR']:row[f'C{j}_{metric}']=v[metric]
     allrows.append(row)
     if on:
      for k,v in enumerate(r['refinement'],1):
       opt=v['optimizer'];cand=v['candidates'];trace=v['trace'];idx=v['best_iteration']
       assert v['stage']==k and len(cand)==41 and len(trace)==40 and opt['steps']==40 and opt['lr']==.01 and opt['lambda_ref']==.01
       assert opt['betas']==[.9,.999] and opt['eps']==1e-8 and opt['weight_decay']==0 and opt['state_entries_at_start']==0 and opt['state_steps_at_end']==[40,40] and opt['execution']=='per_sample_serial'
       assert all(v['invariance']['unchanged'].values()) and all(t['finite'] for t in trace)
       assert idx==min(range(41),key=lambda i:cand[i]['total'][0])
       assert abs(v['J_before']-cand[0]['total'][0])<1e-12 and abs(v['J_after']-cand[idx]['total'][0])<1e-12 and v['J_after']<=v['J_before']+1e-12
       for c in cand:assert math.isclose(c['total'][0],c['observation'][0]+.01*c['proximity'][0],abs_tol=3e-9,rel_tol=3e-6)
       assert math.isclose(cand[idx]['proximity'][0],v['stay_alpha_or_z_contribution']+v['stay_u_or_t_contribution'],abs_tol=2e-7,rel_tol=1e-5)
       refrows.append({'arm':arm,'trajectory_id':sid,'source_id':r['source_id'],'stage':k,'seconds':v['seconds'],'J_before':v['J_before'],'J_after':v['J_after'],'best_iteration':idx,'obs_before':cand[0]['observation'][0],'obs_after':cand[idx]['observation'][0],'stay_alpha_or_z':v['stay_alpha_or_z_contribution'],'stay_u_or_t':v['stay_u_or_t_contribution'],'change_rms_alpha_or_z':v['normalized_change_rms_alpha_or_z'],'change_rms_u_or_t':v['normalized_change_rms_u_or_t'],'u_rms_next':v['u_rms_next'],'u_max_abs_next':v['u_max_abs_next'],'grid_oob_next':v['grid_oob_next'],'optimizer_instance_id':opt['instance_id'],'coordinate_system':v['coordinate_system']})
     if u==1000 and sid in subset:
      p=d/f'{sid}.npz';assert p.is_file()
      with np.load(p,allow_pickle=False) as z:
       assert all(z[n].shape==(1,3,256,256) and np.isfinite(z[n]).all() for n in ['clean','Y4','C1','C2','C3','C4','Y3_hat','Y2_hat','Y1_hat'])
       for j,v in enumerate(r['stages'],1):
        mse=np.square(z[f'C{j}'].astype(np.float64)-z['clean'].astype(np.float64)).mean();ps=-10*math.log10(max(mse,1e-12));error=abs(ps-v['psnr']);assert error<1e-4,(arm,mode,sid,j,error)
        summary['fixed_image_PSNR_checks'].append({'arm':arm,'mode':mode,'trajectory_id':sid,'stage':j,'abs_error_dB':error})
    stats={k:mean(rows[sid][k] for sid in wanted) for k in ['final_psnr','final_ssim','final_lpips','stage_gain','seconds']}
    assert abs(stats['final_psnr']-s['mean_final_psnr'])<1e-12 and abs(stats['final_ssim']-s['mean_final_ssim'])<1e-12
    for metric in ['psnr','ssim','LC','LR']:stats['stage_'+metric]=[mean(rows[sid]['stages'][j][metric] for sid in wanted) for j in range(4)]
    stats['adjacent_PSNR_gain']=[stats['stage_psnr'][j+1]-stats['stage_psnr'][j] for j in range(3)]
    stats['path_LC']=mean(stats['stage_LC']);stats['path_LR']=mean(stats['stage_LR'])
    data[(u,arm,on)]=rows
    if not on:node[arm]=stats
    else:summary['subset'][arm+'-ON']=stats
   cp=R/f'runs/{arm}/U{u:05d}.pt'
   with zipfile.ZipFile(cp) as z:assert z.testzip() is None
   summary['checkpoint_identities'][f'{arm}_U{u:05d}']={'bytes':cp.stat().st_size,'sha256':sha(cp)}
  node['Field_minus_Raw']=paired(data[(u,'Field',False)],data[(u,'Raw',False)]);summary['nodes'][str(u)]=node
 outputs={f'{a}-{m}':{sid:r for sid,r in data[(1000,a,m=='ON')].items() if sid in subset} for a in ['Raw','Field'] for m in ['OFF','ON']}
 for a in ['Raw','Field']:
  rr=outputs[a+'-OFF'];summary['subset'][a+'-OFF']={k:mean(r[k] for r in rr.values()) for k in ['final_psnr','final_ssim','final_lpips','stage_gain','seconds']}
 for name,left,right in [('Field-Raw_OFF','Field-OFF','Raw-OFF'),('Field-Raw_ON','Field-ON','Raw-ON'),('Raw_ON-OFF','Raw-ON','Raw-OFF'),('Field_ON-OFF','Field-ON','Field-OFF')]:summary['subset'][name]=paired(outputs[left],outputs[right])
 pair_rows=[]
 for scope,x in [(f'U{u}_full64_Field-Raw',summary['nodes'][str(u)]['Field_minus_Raw']) for u in [0,500,1000]]+[(k,summary['subset'][k]) for k in ['Field-Raw_OFF','Field-Raw_ON','Raw_ON-OFF','Field_ON-OFF']]:
  for source,p in x['per_source'].items():pair_rows.append({'comparison':scope,'source_id':source,**p})
 for arm in ['Raw','Field']:
  rr=[v for v in refrows if v['arm']==arm];assert len(rr)==24 and len({x['optimizer_instance_id'] for x in rr})==24
  negative=[sid for sid in subset if outputs[arm+'-ON'][sid]['final_psnr']<outputs[arm+'-OFF'][sid]['final_psnr']]
  summary['refinement'][arm]={'stages':24,'mean_J_before':mean(x['J_before'] for x in rr),'mean_J_after':mean(x['J_after'] for x in rr),'sum_optimization_seconds':sum(x['seconds'] for x in rr),'mean_best_iteration':mean(x['best_iteration'] for x in rr),'best_iteration_counts':{str(i):sum(x['best_iteration']==i for x in rr) for i in sorted({x['best_iteration'] for x in rr})},'PSNR_negative_sources':negative,'PSNR_negative_count':len(negative),'per_branch_stay_mean':{k:mean(x[k] for x in rr) for k in ['stay_alpha_or_z','stay_u_or_t']},'per_branch_change_rms_mean':{k:mean(x[k] for x in rr) for k in ['change_rms_alpha_or_z','change_rms_u_or_t']},'mean_grid_oob_next':mean(x['grid_oob_next'] for x in rr),'max_u_abs_next':max(x['u_max_abs_next'] for x in rr)}
  log=[json.loads(x) for x in (R/f'logs/{arm}_train.jsonl').read_text().splitlines()];assert len(log)==1000 and [x['update'] for x in log]==list(range(1,1001))
  assert all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm_before_clip']) for x in log)
  runtime={'update_count':1000,'synchronized_training_seconds':sum(x['seconds'] for x in log),'median_update_seconds':st.median(x['seconds'] for x in log),'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in log),'peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in log),'last100_minibatch_stage_LC':[mean(x['LC'][j] for x in log[-100:]) for j in range(4)],'last100_minibatch_stage_LR':[mean(x['LR'][j] for x in log[-100:]) for j in range(4)],'last100_minibatch_loss':mean(x['loss'] for x in log[-100:])}
  runtime['last100_minibatch_path_LC']=mean(runtime['last100_minibatch_stage_LC']);runtime['last100_minibatch_path_LR']=mean(runtime['last100_minibatch_stage_LR']);summary['runtime'][arm]=runtime
 exits=read(R/'reports/TRAIN_PROCESS_EXIT_BOUNDS.json');assert {x['arm'] for x in exits.values()}=={'Raw','Field'}
 for x in exits.values():summary['runtime'][x['arm']]['train_and_eval_card_hours_bounds']=[(x['last_observed_live_epoch']-x['start_epoch'])/3600,(x['first_observed_exit_epoch']-x['start_epoch'])/3600]
 lifecycle=[json.loads(x) for x in (R/'logs/lifecycle.jsonl').read_text().splitlines()];starts={x['pid']:x for x in lifecycle if x['event']=='start'};ends={x['pid']:x for x in lifecycle if x['event']=='finish'};prep_bounds=[]
 for pid,x in starts.items():
  if x['phase']=='train':continue
  if pid not in ends:
   low=read(R/'reports/attempt1_environment/FAILURE_accept_Raw.json')['epoch'];high=read(R/'reports/attempt1_environment/CONTROLLER_FAILURE.json')['epoch']
  else:
   high=ends[pid]['end_epoch']
   if x['phase']=='accept':low=read(R/'reports/RENDER_PARITY.json')['epoch']
   elif x['phase']=='warm':low=(R/'reports/COMMON_INITIALIZATION.json').stat().st_mtime
   else:low=(R/f"reports/BENCH_{x['arm']}.json").stat().st_mtime
  assert x['start_epoch']<=low<=high
  prep_bounds.append({'phase':x['phase'],'arm':x['arm'],'pid':pid,'card_hours_bounds':[(low-x['start_epoch'])/3600,(high-x['start_epoch'])/3600]})
 summary['preformal_card_time_process_intervals']=prep_bounds
 summary['preformal_card_hours_bounds']=[sum(x['card_hours_bounds'][i] for x in prep_bounds) for i in [0,1]]
 summary['total_card_hours_bounds']=[sum(x['train_and_eval_card_hours_bounds'][i] for x in summary['runtime'].values())+summary['preformal_card_hours_bounds'][i] for i in [0,1]]
 summary['card_time_definition']='Own NPU-worker process windows, including cold compilation/loading; bounded by final artifact creation and observed process exit. Does not count Field idle peer-wait after exit.'
 status=read(R/'reports/STATUS.json');clock=read(R/'reports/EXPERIMENT_CLOCK.json');assert status['status']=='COMPLETE' and status['epoch']<clock['deadline_epoch'];summary['experiment_wall_seconds']=status['wall_seconds'];summary['cpu_delivery_audit_epoch']=time.time();summary['checkpoint_warm_sha256']=sha(R/'runs/common_warm.pt');assert summary['checkpoint_warm_sha256']==frozen['common_warm_sha256']
 summary['max_fixed_image_PSNR_recompute_error_dB']=max(x['abs_error_dB'] for x in summary['fixed_image_PSNR_checks']);summary['checkpoint_audit_NPU0_diagnostic']=read(R/'reports/FINAL_CPU_AUDIT/CHECKPOINT_AUDIT_REPAIR.json')
 summary['total_card_hours_including_diagnostic_bounds']=[summary['total_card_hours_bounds'][0],summary['total_card_hours_bounds'][1]+summary['checkpoint_audit_NPU0_diagnostic']['diagnostic_NPU0_card_time_bound_seconds'][1]/3600]
 cp_audit=read(R/'reports/FINAL_CPU_AUDIT/CHECKPOINT_AUDIT.json');assert cp_audit['status']=='PASS'
 for key,x in summary['checkpoint_identities'].items():assert x['sha256']==cp_audit['checkpoints'][key]['sha256'],key
 assert all(sha(R/name)==h for name,h in frozen['source_hashes'].items())
 summary['frozen_source_hashes_verified']=True;summary['status']='PASS'
 (OUT/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');write_csv('per_input_metrics.csv',allrows);write_csv('per_source_paired.csv',pair_rows);write_csv('refinement_stage_records.csv',refrows)
 print(json.dumps({'status':summary['status'],'main_final':summary['nodes']['1000'],'subset':{k:v for k,v in summary['subset'].items() if '-' in k and k in ['Raw-OFF','Raw-ON','Field-OFF','Field-ON']},'paired_subset':{k:{q:v for q,v in summary['subset'][k].items() if q!='per_source'} for k in ['Field-Raw_OFF','Field-Raw_ON','Raw_ON-OFF','Field_ON-OFF']},'refinement':summary['refinement'],'runtime':summary['runtime'],'experiment_wall_seconds':summary['experiment_wall_seconds']},indent=2))
if __name__=='__main__':main()

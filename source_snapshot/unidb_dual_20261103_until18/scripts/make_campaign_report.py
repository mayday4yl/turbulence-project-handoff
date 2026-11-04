"""Read-only CPU report from saved campaign logs/metrics/images. No torch/model.

Default: provisional stage report. --final requires completed best/latest full
summaries for both routes, model-completion receipt, and a bound independent CPU
metric audit (evidence/FINAL_OUTPUTS_CPU_AUDIT.json, audited_summaries path->SHA).
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,os,sys,time
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'cache/matplotlib_report'))
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'unidb_d0_20261102'
PREVIOUS=ROOT.parent/'unidb_dual_20261103_until_noon';DEADLINE=1793700000
OUT=ROOT/'report';FIG=ROOT/'figures';INPUTS={};WARNINGS=[]
BRANCHES=['original','improved'];COLORS={'original':'#1f77b4','improved':'#d95f02'}
METRICS=['psnr','ssim','mae','lpips'];GS=['0.5','1','2','3','4']

def digest(data):return hashlib.sha256(data).hexdigest()
def bytes_read(path):
 path=Path(path);raw=path.read_bytes();INPUTS[str(path.resolve())]={'sha256':digest(raw),'bytes_read':len(raw),'snapshot_unix':time.time()};return raw

def js(path):
 path=Path(path)
 return json.loads(bytes_read(path)) if path.exists() else None

def lines(path):
 path=Path(path)
 if not path.exists():return []
 raw=bytes_read(path);text=raw.decode();parts=text.splitlines()
 if text and not text.endswith('\n'):parts=parts[:-1];WARNINGS.append('Ignored in-flight incomplete last JSONL line: '+str(path))
 return [json.loads(x) for x in parts if x.strip()]

def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def csvout(name,rows):
 path=OUT/name;fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or ['status']);w.writeheader();w.writerows(rows)

def gkey(g):return f'{float(g):g}'
def fmt(x):
 if x is None:return '待完成'
 if isinstance(x,float):return f'{x:.6g}'
 return str(x)
def table(rows,cols):
 s='| '+' | '.join(v for k,v in cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'
 return s+''.join('| '+' | '.join(fmt(r.get(k)) for k,v in cols)+' |\n' for r in rows)

def summary(path,new=False):
 q=js(path)
 if q is None:return None
 if q.get('status')!='PASS':WARNINGS.append('Not a completed summary: '+str(path));return None
 assert q['weights'] in ['EMA','RAW'] and q['image_clip_crop_resize'] is False
 if q['full']:assert q['count']==5000 and q['per_g']==1000 and {gkey(g) for g in q['by_g']}==set(GS)
 elif new:assert q['count']==200 and q['per_g']==100 and {gkey(g) for g in q['by_g']}=={'1','4'}
 q={**q,'_path':str(Path(path).resolve()),'_sha256':INPUTS[str(Path(path).resolve())]['sha256']}
 return q

def prediction_rows(directory):
 directory=Path(directory);records=[]
 # Completed rank logs are canonical. Group files also allow fixed figures while a
 # whole new validation is still running, but only explicitly completed summaries
 # are used for metric curves or checkpoint selection.
 for p in sorted(directory.glob('rank*.jsonl')):records+=lines(p)
 if not records:
  for p in sorted((directory/'groups').glob('G*/*.json')):
   q=js(p);records+=q.get('records',[])
 seen={}
 for r in records:
  key=(r['sample_id'],gkey(r['g']))
  if key in seen:assert seen[key].get('prediction_sha256')==r.get('prediction_sha256'),'Conflicting saved prediction record'
  seen[key]=r
 return list(seen.values())

def fixed_selection():
 mp=ROOT/'protocol/MONITOR_SELECTION.json'
 if not mp.exists():mp=OLD/'protocol/MONITOR_SELECTION.json'
 mon=js(mp);assert mon and len(mon['sources'])>=2
 sources=mon['sources'][:2];gate=js(OLD/'evidence/DATA_GATE.json');manifest=Path(gate['val_manifest'])
 if not manifest.exists():
  WARNINGS.append('Validation manifest unavailable on this host; fixed images pending.');return {'status':'PENDING','sources':sources,'g':4,'monitor_path':str(mp)},[]
 rows=lines(manifest);assert INPUTS[str(manifest.resolve())]['sha256']==gate['val_manifest_sha256']
 selected=[]
 for source in sources:
  row=sorted([r for r in rows if r['source_id']==source],key=lambda r:r['sample_id'])[0]
  selected.append({'source_id':source,'sample_id':row['sample_id'],'g':4,'input_path':row['states']['4']['path'],'input_sha256':row['states']['4']['sha256'],'clean_path':row['clean_path'],'clean_sha256':row['clean_sha256']})
 path=OUT/'FIXED_IMAGE_SELECTION.json';existing=js(path)
 q={'status':'PASS','rule':'MONITOR_SELECTION.sources first2 in declared order, G4, first lexical sample_id per source; never chosen by model result','sources':sources,'g':4,'monitor_path':str(mp.resolve()),'monitor_sha256':INPUTS[str(mp.resolve())]['sha256'],'val_manifest_sha256':gate['val_manifest_sha256'],'selected':selected}
 if existing:
  assert existing['sources']==sources and existing['selected']==selected,'Predeclared fixed-image selection changed'
  q=existing
 else:write(path,q)
 return q,selected

def local_iso(value):
 return datetime.fromtimestamp(float(value),timezone(timedelta(hours=8))).isoformat(timespec='seconds') if value is not None else None


def update_handoff_gap(training):
 old=[r for r in training if r['campaign_stage']=='until_noon']
 new=[r for r in training if r['campaign_stage']=='until18_extension']
 end=max(old,key=lambda r:r['update']) if old else None
 start=min(new,key=lambda r:r['update']) if new else None
 seconds=float(start['started_unix'])-float(end['completed_unix']) if end and start else None
 return {'noon_last_completed_update':end['update'] if end else None,
  'noon_last_completed_unix':end['completed_unix'] if end else None,
  'extension_first_started_update':start['update'] if start else None,
  'extension_first_update_started_unix':start['started_unix'] if start else None,
  'noon_last_completed_local_iso':local_iso(end['completed_unix']) if end else None,
  'extension_first_update_started_local_iso':local_iso(start['started_unix']) if start else None,
  'recorded_update_gap_seconds':seconds,'recorded_update_gap_minutes':seconds/60 if seconds is not None else None,
  'recorded_update_gap_scope':'Last noon committed update completed to first extension committed update started; includes old validation, handoff, engineering gates and initialization; not pure idle time.'}


def cpu_cost_provenance(cpu):
 """Separate historical evidence from a real new-wrapper invocation and reporter.

 Signatures deduplicate validation evidence; a copied historical aggregate never
 becomes a new invocation. Cached total-reference batches are never added as
 fresh work: only current per-validation *_this_invocation fields count there.
 """
 history_path=PREVIOUS/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json';history=js(history_path)
 wrapper=ROOT/'scripts/audit_campaign_saved_outputs.py'
 protocol=cpu.get('protocol',{}) if cpu else {}
 current=bool(cpu and cpu.get('status')=='PASS' and wrapper.is_file()
  and protocol.get('wrapper_sha256')==digest(bytes_read(wrapper))
  and Path(protocol.get('out_root','/')).resolve()==ROOT.resolve())
 def unique(q):
  found={}
  for r in q.get('validations',[]) if q and q.get('status')=='PASS' else []:
   if r.get('status')!='PASS' or not r.get('signature'):continue
   key=r['signature']
   if key in found:
    assert found[key]['summary_sha256']==r['summary_sha256'],'Audit signature conflicts across summaries'
    continue
   found[key]=r
  return found
 historical=unique(history);now=unique(cpu) if current else {};rows=[]
 hp=str(history_path.resolve());cp=str((ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json').resolve())
 hh=INPUTS.get(hp,{}).get('sha256');ch=INPUTS.get(cp,{}).get('sha256')
 for signature in sorted(set(historical)|set(now)):
  h=historical.get(signature);n=now.get(signature);r=h or n
  path=r.get('validation_dir',r.get('validation',''));name=Path(path).name
  fresh=int(n.get('CPU_metric_forward_batches_this_invocation',0)) if n else 0
  if n and n.get('reused_completed_audit'):assert fresh==0,'Cache reuse cannot add metric forwards'
  rows.append({'validation_signature':signature,'validation':path,'branch':Path(path).parent.parent.name,
   'update':int(name.split('_')[0][1:]) if name.startswith('U') else None,
   'summary_sha256':r['summary_sha256'],'historical_aggregate':hp if h else None,'historical_aggregate_sha256':hh if h else None,
   'historical_recorded_metric_forward_batches':int(h.get('CPU_metric_forward_batches',h.get('CPU_metric_forward_batches_this_invocation',0))) if h else 0,
   'current_wrapper_aggregate':cp if n else None,'current_wrapper_aggregate_sha256':ch if n else None,
   'current_invocation_metric_forward_batches':fresh,'current_invocation_reused_cache':bool(n and n.get('reused_completed_audit')),
   'reporter_metric_model_calls':0,'reporter_restoration_model_calls':0})
 scopes=[{'scope':'historical_noon_completed_audit','source_aggregate':hp,'source_sha256':hh,
  'unique_validation_signatures':len(historical),'CPU_metric_forward_batches':sum(r['historical_recorded_metric_forward_batches'] for r in rows),
  'CPU_metric_model_constructions':history.get('CPU_metric_model_constructions') if history and history.get('status')=='PASS' else None,
  'reporter_model_calls':0,'cost_role':'Historical evidence only; copying/cache does not create extension work.'},
 {'scope':'current_extension_finalizer_invocation','source_aggregate':cp if current else None,'source_sha256':ch if current else None,
  'invocation_receipt_present':current,'unique_validation_signatures':len(now),
  'CPU_metric_forward_batches':sum(r['current_invocation_metric_forward_batches'] for r in rows) if current else None,
  'CPU_metric_model_constructions':cpu.get('CPU_metric_model_constructions') if current else None,
  'cache_reused_unique_signatures':sum(bool(r['current_invocation_reused_cache']) for r in rows),
  'reporter_model_calls':0,'cost_role':'Actual new-wrapper receipt only; per-signature this_invocation forwards, caches add zero.'},
 {'scope':'this_reporter_invocation','CPU_metric_forward_batches':0,'CPU_metric_model_constructions':0,
  'reporter_metric_model_calls':0,'reporter_restoration_model_calls':0,'optimizer_updates':0,'NPU_calls':0}]
 return {'historical':scopes[0],'current':scopes[1],'reporter':scopes[2],'validation_rows':rows,'scope_rows':scopes,
  'copied_historical_aggregate_not_new_invocation':bool(cpu and cpu.get('status')=='PASS' and not current),
  'deduplication':'Exact validation signature within each scope; copied/reference batch totals are never treated as new forwards.'}


def collect_branch(branch):
 run=ROOT/'runs'/branch;previous=PREVIOUS/'runs'/branch;base=45411 if branch=='original' else 0
 authority=js(ROOT/'protocol/USER_AUTHORITY.json');parent=authority['extension_parents'][branch];extension_base=parent['update']
 cfg=js(run/'CONFIG.json');status=js(run/'STATUS.json');latest=js(run/'LATEST.json')
 if cfg:assert cfg['branch']==branch and cfg['phase']=='formal','Gate metadata must not be reported as formal training'
 best=js(run/'BEST_FULL.json') or js(run/'BEST.json') or js(previous/'BEST_FULL.json')
 extension_monitor=js(run/'BEST_MONITOR.json')
 monitor=extension_monitor or js(previous/'BEST_MONITOR.json')
 by={}
 for epoch,folder in [('until_noon',previous),('until18_extension',run)]:
  for r in lines(folder/'UPDATES_rank0.jsonl'):
   assert r['branch']==branch and r['update']>=base and r['new_updates']==r['update']-base
   u=int(r['update']);assert u not in by,'Duplicate committed update across noon/extension rank0 logs'
   assert (u<=extension_base if epoch=='until_noon' else u>extension_base),'Log crosses frozen extension parent'
   by[u]={**r,'campaign_stage':epoch,'log_path':str((folder/'UPDATES_rank0.jsonl').resolve())}
 slim=[]
 for u,r in sorted(by.items()):
  slim.append({k:r.get(k) for k in ['branch','update','new_updates','cursor','campaign_stage','log_path','extension_start_update','extension_new_updates','extension_new_presentations','started_unix','completed_unix','seconds','loss','lr','grad_norm','grad_norm_after_clip']})
 complete=[];seen_paths=set();seen_keys={}
 for folder in [previous,run]:
  for path in sorted((folder/'validation').glob('*/SUMMARY.json')):
   canonical=str(path.resolve())
   if canonical in seen_paths:continue
   seen_paths.add(canonical);q=summary(path,new=True)
   if q is None:continue
   assert q['branch']==branch
   key=(q['update'],q['weights'],q['full'])
   if key in seen_keys:
    assert q['_sha256']==seen_keys[key]['_sha256'],'Conflicting completed validation identity'
    continue
   q['campaign_stage']='until18_extension' if Path(q['_path']).is_relative_to(ROOT.resolve()) else 'until_noon'
   seen_keys[key]=q;complete.append(q)
 monitor_candidates=[q for q in complete if q['weights']=='EMA' and not q['full']]
 campaign_monitor=max(monitor_candidates,key=lambda q:(q['by_g']['4']['psnr'],-q['by_g']['4']['lpips'],-q['update'])) if monitor_candidates else None
 def locate(update,full=None):
  choices=[q for q in complete if q['weights']=='EMA' and q['update']==update and (full is None or q['full']==full)]
  return sorted(choices,key=lambda q:q['full'],reverse=True)[0] if choices else None
 extension_monitor_summary=locate(extension_monitor['update'],False) if extension_monitor else None
 bestfull=None
 if best:
  assert best['full'] is True and best['weights']=='EMA' and best['selection_role']=='full_validation'
  checkpoint=Path(best['checkpoint']['path']).resolve()
  allowed=[run.resolve(),previous.resolve(),Path(authority['previous_campaign_root']).resolve()/'runs'/branch]
  assert any(checkpoint.is_relative_to(x) for x in allowed),'Best points outside the continuous 3+5 campaign line'
  bestfull=locate(best['update'],True)
 latestfull=locate(latest['update'],True) if latest else None
 newest=sorted([q for q in complete if q['weights']=='EMA'],key=lambda q:(q['update'],q['full']),reverse=True)
 latest_available=(locate(latest['update']) if latest else None) or (newest[0] if newest else None)
 best_image=bestfull or (locate(monitor['update'],False) if monitor else None) or latest_available
 if bestfull:
  candidates=[q for q in complete if q['weights']=='EMA' and q['full']]
  winner=max(candidates,key=lambda q:(q['by_g']['4']['psnr'],-q['by_g']['4']['lpips'],-q['update']))
  assert winner['update']==bestfull['update'],'BEST_FULL is not best among continuous campaign completed full summaries'
 retained=max([extension_base]+[r['update'] for r in slim]+([latest['update']] if latest else []))
 if status and status.get('status')=='COMPLETE':
  assert latest and latest['update']==retained,'Complete line checkpoint/log update mismatch'
  assert sorted(by)==list(range(base+1,retained+1)),'Final continuous campaign update log has missing/gapped updates'
 return {'branch':branch,'run':str(run),'previous_run':str(previous),'base':base,'extension_base':extension_base,
  'extension_parent':parent,'retained_update':retained,'config':cfg,'status':status,'latest':latest,'best':best,'monitor_best':monitor,
  'training':slim,'validations':complete,'best_full_summary':bestfull,'latest_full_summary':latestfull,
  'latest_available_summary':latest_available,'best_image_summary':best_image,
  'campaign_monitor_best_summary':campaign_monitor,'extension_monitor_best':extension_monitor,'extension_monitor_best_summary':extension_monitor_summary}


def recovery_work(branch,run,retained_update):
 records=[]
 for path in sorted(Path(run).glob('UNCOMMITTED_RECOVERY_rank0*.json')):
  q=js(path);rr=q['records'];assert isinstance(rr,list)
  assert len({r['update'] for r in rr})==len(rr),'Duplicate update within one recovery event'
  assert all(r['branch']==branch for r in rr),'Recovery file belongs to a different route'
  records.append({'branch':branch,'recovery_file':str(path.resolve()),'discarded_completed_updates':len(rr),'extra_recorded_presentations':96*len(rr),
   'first_update':min((r['update'] for r in rr),default=None),'last_update':max((r['update'] for r in rr),default=None),
   'update_numbers_reached_again_by_current_retained_trajectory':sum(r['update']<=retained_update for r in rr),
   'scope':'Each discarded completed execution counted once per recovery file; repeated discard of same update in another event is additional compute, not an effective new update.'})
 return records


def engineering_work():
 out=[]
 for epoch,root in [('until_noon',PREVIOUS),('until18_extension',ROOT)]:
  for folder in sorted((root/'gates').glob('*')):
   if not folder.is_dir():continue
   q=js(folder/'RESULT.json');rr=lines(folder/'UPDATES_rank0.jsonl')
   if q and q.get('status')=='PASS':
    r=q['ranks'][0];n=int(r['completed_updates_this_process']);branch=q['branch'];world=r['world'];state='PASS_GATE_ONLY'
   elif rr:
    n=len(rr);branch=rr[0]['branch'];rank=js(folder/'RANK_0.json');world=rank.get('world') if rank else None;state='PARTIAL_GATE_LOG_ONLY'
   else:continue
   out.append({'campaign_stage':epoch,'gate':folder.name,'branch':branch,'physical_NPUs':world,
    'completed_updates_this_gate_process':n,'recorded_presentations':96*n,'status':state,'carried_into_formal':False,
    'scope':'Noon initial2+resume1 and original8 reference are separate discarded work; extension initial1+resume1 per route totals4updates/384presentations when all four pass. Unrecorded in-flight work is excluded.'})
 return out


def old_baselines(monitor_sources):
 run=OLD/'runs/formal_until_20261103_noon';out={};raw={};matched=[];input_all=defaultdict(list)
 for label,u in [('U0',0),('parentFinal',45411)]:
  p=run/'validation'/f'U{u:08d}_full'/'SUMMARY.json';s=summary(p)
  rr=prediction_rows(p.parent) if s else [];out[label]=s;raw[label]=rr
  for g in ['1','4']:
   r=[x for x in rr if gkey(x['g'])==g and x['source_id'] in monitor_sources]
   if r:
    assert len(r)==100 and len({x['source_id'] for x in r})==25
    matched.append({'baseline':label,'g':g,'count':len(r),'sources':25,**{m:float(np.mean([x['metrics'][m] for x in r])) for m in METRICS}})
  if label=='U0':
   for r in rr:input_all[gkey(r['g'])].append(r['input_metrics'])
 inp={g:{m:float(np.mean([r[m] for r in values])) for m in METRICS} for g,values in input_all.items()}
 return out,raw,matched,inp

def training_curves(branches,matched):
 fig,axs=plt.subplots(2,4,figsize=(18,8))
 for col,b in enumerate(BRANCHES):
  rr=branches[b]['training'];x=[r['new_updates'] for r in rr];color=COLORS[b]
  ax=axs[0,col];ax.plot(x,[r['loss'] for r in rr],color=color,alpha=.3,lw=.5,label='Raw committed loss')
  if len(rr)>=10:
   n=min(100,len(rr));ax.plot(x[n-1:],np.convolve([r['loss'] for r in rr],np.ones(n)/n,'valid'),color=color,lw=1.1,label=f'Trailing mean{n}')
  ax.set_title(b+(' native mean L1' if b=='original' else ' epsilon_opt MSE'));ax.set_ylabel('Own objective loss (different units)')
  if rr:ax.legend(fontsize=7)
  ax=axs[1,col];ax.plot(x,[r['lr'] for r in rr],color=color);ax.set_title(b+' learning rate');ax.set_ylabel('LR')
  if rr:ax.set_yscale('log')
  ax=axs[1,col+2];ax.plot(x,[r['grad_norm'] for r in rr],color=color,label='Before clip');ax.plot(x,[r['grad_norm_after_clip'] for r in rr],color=color,ls='--',label='After clip')
  ax.set_title(b+' gradient norm');ax.set_ylabel('Norm in own objective units')
  if rr:ax.legend(fontsize=7);ax.set_yscale('log')
 for col,g in [(2,'1'),(3,'4')]:
  ax=axs[0,col]
  for b in BRANCHES:
   for full,marker in [(False,'o'),(True,'D')]:
    r=sorted([q for q in branches[b]['validations'] if q['weights']=='EMA' and q['full']==full],key=lambda q:q['update'])
    if r:ax.plot([q['update']-branches[b]['base'] for q in r],[q['by_g'][g]['psnr'] for q in r],marker=marker,ls='-' if not full else 'None',markersize=3,color=COLORS[b],label=b+(' full1000' if full else ' monitor100'))
  for r in matched:
   if r['g']==g:ax.axhline(r['psnr'],ls=':' if r['baseline']=='U0' else '--',c='gray',lw=.8,label=r['baseline']+' matched monitor')
  ax.set_title('G'+g+' native EMA PSNR');ax.set_ylabel('PSNR to GT (dB)')
  if ax.lines:ax.legend(fontsize=6,ncol=1)
 for ax in axs.flat:
  ax.set_xlabel('Added optimizer updates since original campaign base');ax.grid(alpha=.2)
  if not any(len(line.get_xdata())>0 for line in ax.lines):ax.text(.5,.5,'Waiting for formal data',ha='center',va='center',transform=ax.transAxes)
 for col,b in enumerate(BRANCHES):
  boundary=branches[b]['extension_base']-branches[b]['base']
  for ax in [axs[0,col],axs[1,col],axs[1,col+2]]:ax.axvline(boundary,color='black',ls=':',lw=.8,label='Noon extension boundary')
 for ax in [axs[0,2],axs[0,3]]:
  for b in BRANCHES:ax.axvline(branches[b]['extension_base']-branches[b]['base'],color=COLORS[b],ls=':',lw=.6)
 fig.suptitle('Continuous noon +18:00 campaign; dotted verticals=extension starts; loss/gradient units differ',fontsize=12);fig.tight_layout()
 fig.savefig(FIG/'CAMPAIGN_DYNAMICS.png',dpi=170,bbox_inches='tight');fig.savefig(FIG/'CAMPAIGN_DYNAMICS.pdf',bbox_inches='tight');plt.close(fig)

def load_asset(path,expected,assets):
 path=Path(path)
 if not path.exists():return None
 blob=bytes_read(path);actual=digest(blob)
 if expected:assert actual==expected,'Fixed asset SHA mismatch: '+str(path)
 arr=np.load(io.BytesIO(blob),allow_pickle=False);assert arr.shape==(3,256,256) and arr.dtype==np.float32 and np.isfinite(arr).all()
 assets.append({'path':str(path),'sha256':actual,'display_clamp_01':True,'raw_min':float(arr.min()),'raw_max':float(arr.max()),'raw_outside_01_fraction':float(np.mean((arr<0)|(arr>1)))})
 return arr

def fixed_grid(role,selected,branches,old_records):
 assets=[];fig,axes=plt.subplots(max(1,len(selected)),6,figsize=(18,3.8*max(1,len(selected))),squeeze=False)
 records={name:{(r['sample_id'],gkey(r['g'])):r for r in rr} for name,rr in old_records.items()}
 chosen={}
 for b in BRANCHES:
  q=branches[b]['best_image_summary'] if role=='best' else branches[b]['latest_available_summary'];chosen[b]=q
  rr=prediction_rows(Path(q['_path']).parent) if q else [];records[b]={(r['sample_id'],gkey(r['g'])):r for r in rr}
 for i,s in enumerate(selected):
  key=(s['sample_id'],'4');paths=[('Input',s['input_path'],s['input_sha256'],None)]
  for name,title in [('U0','Original U0 EMA'),('parentFinal','Parent Final EMA'),('original','Original route EMA'),('improved','Improved route EMA')]:
   r=records[name].get(key);q=chosen.get(name);suffix=('\nU'+str(q['update'])+(' full' if q['full'] else ' monitor')+'\n'+q.get('campaign_stage','')) if q else ''
   paths.append((title+suffix,r.get('prediction') if r else None,r.get('prediction_sha256') if r else None,r))
  paths.append(('GT',s['clean_path'],s['clean_sha256'],None))
  for j,(title,path,expected,r) in enumerate(paths):
   ax=axes[i,j];a=load_asset(path,expected,assets) if path else None
   if a is None:ax.set_facecolor('#eeeeee');ax.text(.5,.5,'Saved output pending\n(no model called)',ha='center',va='center',transform=ax.transAxes,fontsize=9)
   else:ax.imshow(np.clip(np.moveaxis(a,0,2),0,1),interpolation='nearest')
   if r:title+='\nPSNR '+f"{r['metrics']['psnr']:.3f}"+' dB'
   ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
   if j==0:ax.set_ylabel(s['source_id']+'\n'+s['sample_id'],fontsize=8)
 if not selected:
  for ax in axes.flat:ax.axis('off')
  axes[0,0].text(0.,.5,'Fixed source manifest/assets unavailable on this host; no images selected by results.',fontsize=11,transform=axes[0,0].transAxes)
 fig.suptitle('Fixed G4 '+role+' available saved EMA predictions | Display clamp[0,1] ONLY; raw metrics have no clip/crop/resize',fontsize=11);fig.tight_layout()
 name='FIXED_G4_'+role.upper();fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 return {'role':role,'selected_validation':{b:({'path':q['_path'],'update':q['update'],'full':q['full'],'campaign_stage':q.get('campaign_stage')} if q else None) for b,q in chosen.items()},'assets':assets,'scope':'Stage fallback is a completed monitor candidate, explicitly labeled; final best/latest must resolve full5000 summaries.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
 started=time.time();authority=js(ROOT/'protocol/USER_AUTHORITY.json');assert authority['deadline_unix']==DEADLINE
 global PREVIOUS
 configured=Path(authority['previous_campaign_root']).resolve()
 if configured.exists():PREVIOUS=configured
 root_status=js(ROOT/'STATUS.json');shared_contexts=js(ROOT/'protocol/SHARED_CONTEXTS.json');selection,selected=fixed_selection();monitor_path=ROOT/'protocol/MONITOR_SELECTION.json'
 if not monitor_path.exists():monitor_path=OLD/'protocol/MONITOR_SELECTION.json'
 monitor=js(monitor_path);branches={b:collect_branch(b) for b in BRANCHES};baselines,old_records,matched,inputs=old_baselines(set(monitor['sources']))
 updates=[r for b in BRANCHES for r in branches[b]['training']];vr=[];quality=[];activity=[];recovery=[];gate_work=engineering_work()
 actor_events=[{**r,'campaign_stage':epoch} for epoch,root in [('until_noon',PREVIOUS),('until18_extension',ROOT)] for r in lines(root/'evidence/ACTOR_EVENTS.jsonl')]
 for b in BRANCHES:
  line=branches[b];last=line['training'][-1] if line['training'] else None
  retained=line['retained_update'];extension_new=retained-line['extension_base'];campaign_new=retained-line['base']
  replay=[]
  for epoch,folder in [('until_noon',line['previous_run']),('until18_extension',line['run'])]:
   replay.extend({**r,'campaign_stage':epoch} for r in recovery_work(b,folder,retained))
  recovery.extend(replay)
  stopped=js(Path(line['run'])/'TRAINING_STOPPED.json');actors=[r for r in actor_events if r.get('event')=='launched' and r.get('phase')=='formal' and r.get('branch')==b]
  extension_actors=[r for r in actors if r['campaign_stage']=='until18_extension']
  activity.append({'branch':b,'physical_NPUs':len(authority['allocation']['original_continue' if b=='original' else 'improved']),'initial_update':line['base'],
   'committed_new_updates':campaign_new,'new_presentations':96*campaign_new,'last_global_update':retained,
   'last_completed_log_update':last['update'] if last else None,**update_handoff_gap(line['training']),
   'noon_parent_update':line['extension_base'],'before_extension_updates':line['extension_base']-line['base'],
   'extension_new_updates':extension_new,'extension_new_presentations':96*extension_new,
   'recorded_committed_rows':len(line['training']),'extension_recorded_rows':sum(r['campaign_stage']=='until18_extension' for r in line['training']),
   'first_formal_actor_start_unix':min(r['started_unix'] for r in actors) if actors else None,
   'extension_first_actor_start_unix':min(r['started_unix'] for r in extension_actors) if extension_actors else None,
   'training_stop_detection_unix':stopped.get('detected_unix') if stopped else None,'final_evaluation_completed_unix':line['status'].get('completed_unix') if line['status'] else None,
   'extra_discarded_completed_updates':sum(r['discarded_completed_updates'] for r in replay),'extra_recorded_presentations':sum(r['extra_recorded_presentations'] for r in replay),'recovery_record_files':len(replay),
   'extension_extra_discarded_updates':sum(r['discarded_completed_updates'] for r in replay if r['campaign_stage']=='until18_extension'),
   'formal_actor_attempts':len(actors),'extension_actor_attempts':len(extension_actors),'configured_cards':','.join(map(str,authority['allocation']['original_continue' if b=='original' else 'improved'])),'last_completed_unix':last['completed_unix'] if last else None,
   'rank0_recorded_update_seconds':sum(r['seconds'] for r in line['training']),
   'extension_rank0_recorded_update_seconds':sum(r['seconds'] for r in line['training'] if r['campaign_stage']=='until18_extension'),
   'worker_status':line['status'].get('status') if line['status'] else 'EXTENSION_NOT_STARTED','latest_checkpoint_update':line['latest']['update'] if line['latest'] else None,
   'best_full_update':line['best_full_summary']['update'] if line['best_full_summary'] else None,
   'best_full_stage':line['best_full_summary']['campaign_stage'] if line['best_full_summary'] else None,
   'best_monitor_update':line['campaign_monitor_best_summary']['update'] if line['campaign_monitor_best_summary'] else None,
   'best_monitor_g4_psnr':line['campaign_monitor_best_summary']['by_g']['4']['psnr'] if line['campaign_monitor_best_summary'] else None,
   'best_monitor_summary':line['campaign_monitor_best_summary']['_path'] if line['campaign_monitor_best_summary'] else None,
   'best_monitor_summary_sha256':line['campaign_monitor_best_summary']['_sha256'] if line['campaign_monitor_best_summary'] else None,
   'extension_best_monitor_update':line['extension_monitor_best']['update'] if line['extension_monitor_best'] else None,
   'extension_best_monitor_g4_psnr':line['extension_monitor_best_summary']['by_g']['4']['psnr'] if line['extension_monitor_best_summary'] else None})
  for q in line['validations']:
   for g,m in q['by_g'].items():vr.append({'branch':b,'update':q['update'],'new_updates':q['update']-line['base'],'campaign_stage':q['campaign_stage'],'weights':q['weights'],'full':q['full'],'g':gkey(g),'per_g':q['per_g'],'sources_per_g':q['sources_per_g'],'summary':q['_path'],'summary_sha256':q['_sha256'],**m})
  for role,key in [('best_full','best_full_summary'),('latest_full','latest_full_summary')]:
   q=line[key]
   if q:
    for g,m in q['by_g'].items():quality.append({'line':b,'role':role,'campaign_stage':q['campaign_stage'],'update':q['update'],'new_updates':q['update']-line['base'],'g':gkey(g),'count':1000,'sources':250,**m,'summary':q['_path'],'summary_sha256':q['_sha256']})
 for item in activity:
  for key in ['first_formal_actor_start_unix','extension_first_actor_start_unix','training_stop_detection_unix','final_evaluation_completed_unix']:
   item[key.replace('_unix','_local_iso')]=local_iso(item.get(key))
 for name,q in baselines.items():
  if q:
   for g,m in q['by_g'].items():quality.append({'line':name,'role':'external_fixed_baseline','campaign_stage':'外部基线（不适用）','update':q['update'],'new_updates':None,'g':gkey(g),'count':1000,'sources':250,**m,'summary':q['_path'],'summary_sha256':q['_sha256']})
 for g,m in inputs.items():quality.append({'line':'Input','role':'input_baseline','campaign_stage':'不适用','update':'不适用','new_updates':'不适用','g':g,'count':1000,'sources':250,**m})
 ref={(r['line'],r['g']):r for r in quality if r['role'] in ['external_fixed_baseline','input_baseline']}
 for r in quality:
  for label in ['Input','U0','parentFinal']:
   if (label,r['g']) in ref:r['delta_psnr_vs_'+label]=r['psnr']-ref[(label,r['g'])]['psnr']
 csvout('TRAINING_UPDATES.csv',updates);csvout('VALIDATION_HISTORY.csv',vr);csvout('FULL_QUALITY_COMPARISON.csv',quality);csvout('MATCHED_MONITOR_BASELINES.csv',matched);csvout('ACTIVITY_AND_COST.csv',activity);csvout('RECOVERY_REPLAY_WORK.csv',recovery);csvout('ENGINEERING_GATE_WORK.csv',gate_work)
 # Every final model must have its own complete evaluated checkpoint. Old U0 and
 # parentFinal remain separate baselines and are never inserted into new BEST.
 required={};pending=[]
 for b in BRANCHES:
  line=branches[b]
  if not line['status'] or line['status'].get('status')!='COMPLETE' or not line['status'].get('final_full_validation_done'):pending.append(b+' worker/full evaluation not complete')
  for key in ['best_full_summary','latest_full_summary']:
   q=line[key]
   if not q:pending.append(b+' '+key+' missing')
   else:
    required[q['_path']]=q['_sha256']
    role='FINAL' if key=='latest_full_summary' else 'BEST'
    state_audit=js(ROOT/'evidence'/f'STATE_AUDIT_{b}_{role}.json')
    identity=line['latest'] if role=='FINAL' else line['best']['checkpoint']
    if not state_audit or state_audit.get('status')!='PASS':pending.append(b+' '+role+' complete-state CPU audit missing')
    else:
     bound=state_audit['checkpoint']
     if (Path(bound['path']).resolve()!=Path(identity['path']).resolve() or bound['sha256']!=identity['sha256'] or bound['update']!=q['update'] or bound['EMA']['full_parameter_digest']!=q['parameter_digest']):pending.append(b+' '+role+' state/SUMMARY EMA identity mismatch')
 model_done=js(ROOT/'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json')
 if not model_done or model_done.get('status')!='PASS':pending.append('controller model-validation completion receipt missing')
 cpu=js(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json')
 if not cpu or cpu.get('status')!='PASS':pending.append('independent saved-output CPU audit not PASS')
 else:
  audited={str(Path(p).resolve()):h for p,h in cpu.get('audited_summaries',{}).items()}
  for p,h in required.items():
   if audited.get(p)!=h:pending.append('CPU audit not bound to actual summary '+p)
 cpu_provenance=cpu_cost_provenance(cpu)
 csvout('CPU_POSTPROCESS_WORK.csv',cpu_provenance['scope_rows']);csvout('CPU_AUDIT_VALIDATION_PROVENANCE.csv',cpu_provenance['validation_rows'])
 ready=not pending
 # Cache only figures. A later QA/report-only refresh keeps already accepted PNG
 # and PDF bytes unchanged when the underlying data/signatures are unchanged.
 fig_sig=digest(json.dumps({'script':digest(Path(__file__).read_bytes()),'updates':updates,'validation':vr,'matched':matched,'selection':selection,
  'selected':{b:{k:(line[k]['_sha256'] if line[k] else None) for k in ['best_image_summary','latest_available_summary']} for b,line in branches.items()}},sort_keys=True).encode())
 fm=js(OUT/'FIGURE_MANIFEST.json');reuse=bool(fm and fm.get('input_signature')==fig_sig and all((FIG/name).exists() and digest((FIG/name).read_bytes())==h for name,h in fm.get('figures',{}).items()))
 if reuse:
  for audit in fm['fixed_image_audits']:
   for asset in audit['assets']:
    path=Path(asset['path'])
    if not path.exists() or digest(bytes_read(path))!=asset['sha256']:reuse=False;break
 if reuse:fixed=fm['fixed_image_audits']
 else:
  training_curves(branches,matched);fixed=[fixed_grid(role,selected,branches,old_records) for role in ['best','latest']]
  fm={'input_signature':fig_sig,'figures':{p.name:digest(p.read_bytes()) for p in sorted(FIG.glob('*')) if p.name.startswith(('CAMPAIGN_DYNAMICS','FIXED_G4_')) and p.suffix in ['.png','.pdf']},'fixed_image_audits':fixed,'display_only_clamp':True,'visual_QA_by_root_required':True};write(OUT/'FIGURE_MANIFEST.json',fm)
 if args.final and ready:
  assert len(selected)==2 and all(len(q['assets'])==12 for q in fixed),'Complete fixed six-column images are required for final report'
 qa=js(OUT/'VISUAL_QA.json');visual_ok=bool(qa and qa.get('status')=='PASS' and all(qa.get('figures',{}).get(n)==h for n,h in fm['figures'].items() if n.endswith('.png')))
 phase=('FINAL_CPU_AUDITED_VISUAL_QA_PENDING' if not visual_ok else 'FINAL_READY') if args.final and ready else ('STAGE_DATA_READY_FOR_FINAL_REPORT' if ready else 'IN_PROGRESS')
 now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds');text='# UniDB 两路线'+('最终' if args.final and ready else '阶段')+'报告\n\n'
 text+=f"生成于{now}。状态：**{phase}**。原续训线累计新增{activity[0]['committed_new_updates']}次更新、改进线累计新增{activity[1]['committed_new_updates']}次更新；中午后延长段分别新增{activity[0]['extension_new_updates']}、{activity[1]['extension_new_updates']}次更新。训练截止2026-11-03 18:00 Asia/Shanghai，截止后可完成最终评价。工程gate、CPU数学PASS或小monitor不代表正式训练完成、收敛或泛化成功。\n\n"
 if root_status:text+='当前根执行状态：`'+str(root_status.get('status'))+'`。\n\n'
 if pending:text+='尚待：'+'；'.join(pending)+'。\n\n'
 text+='## 两条路线与实际工作量\n\n'
 text+='原线从父Final RAW U45411及Adam/EMA/8逻辑RNG继续，原native mean-L1、lr1e-5；3张物理卡模拟8条逻辑B12流。改进线从官方完整SR U0重新初始化，fresh Adam，epsilon_opt MSE、LR2e-7→2e-6 warmup200后wall-clock cosine→2e-7、global clip1.0；5卡局部B20/19/19/19/19。中午后由各自完整父状态U55760/U20000继续，原LR保持1e-5、改进LR保持2e-7，不重新热身、不回升LR，完整RAW/Adam/EMA/RNG/cursor及旧日程锚点继承。两线均全参数、B96、FP32/HF32off、原网络/桥、无task prompt/Adapter/raw-xi/A/B或物理递归。\n\n'
 text+='这是起点、objective、schedule和资源都不同的实用综合recipe比较，不是单变量因果ablation或吞吐benchmark。原L1没有被证明实现错误；epsilon_opt目标不是实际forward epsilon，loss为mean((noise−epsilon_opt)²)，不再重复除c²。\n\n'
 text+=table(activity,[('branch','路线'),('physical_NPUs','卡'),('initial_update','起始U'),('committed_new_updates','3+5累计新增更新'),('new_presentations','累计呈现'),('extension_new_updates','中午后新增'),('extension_new_presentations','延长段呈现'),('latest_checkpoint_update','最近完整保存checkpoint U'),('last_completed_log_update','本快照已完成日志U'),('best_full_update','full best U'),('best_monitor_update','全campaign监控最优U'),('extension_best_monitor_update','延长段监控最优U'),('worker_status','状态')])+'\n'
 text+=table(activity,[('branch','路线'),('first_formal_actor_start_local_iso','首正式actor启动(北京时间)'),('training_stop_detection_local_iso','停训检测(北京时间)'),('final_evaluation_completed_local_iso','终评完成(北京时间)'),('formal_actor_attempts','两阶段正式/恢复actor数')])+'\n'
 if shared_contexts:
  cards=sorted({int(r['card']) for r in shared_contexts.get('permitted_contexts',[]) if r.get('kind')=='observed_external_training_context'})
  if cards:text+='资源共享披露：`protocol/SHARED_CONTEXTS.json`记录卡'+('、'.join(map(str,cards)))+'与外部训练任务共享；3+5是设备分配数，不代表独占算力。共享负载会影响实际时延，现有记录不能将时延变化因果归于具体任务或估算具体百分比；未停止外部任务。\n\n'
 text+='交接更新间隔按旧noon最后一条committed更新完成→新extension首条committed更新开始计算；包含旧验证、资源交接、工程门与初始化，不能当作纯空闲时间：\n\n'
 text+=table(activity,[('branch','路线'),('noon_last_completed_update','noon最后完成U'),('noon_last_completed_local_iso','完成(北京时间)'),('extension_first_started_update','extension首更新U'),('extension_first_update_started_local_iso','开始(北京时间)'),('recorded_update_gap_minutes','记录更新间隔(min)')])+'\n'
 text+='；'.join(r['branch']+f"约{r['recorded_update_gap_minutes']:.2f}分钟" for r in activity if r['recorded_update_gap_minutes'] is not None)+'。该间隔不是纯空闲损失。\n\n'
 text+='两线累计新增更新是当前有效轨迹global update−原始base（45411/0）；中午后新增是global update−各自noon父（55760/20000）。连续曲线合并旧noon和新extension的committed日志，以完整父与global update计数，不把新日志行数冒充累计训练量；恢复回放不会被重复计为有效新更新。以下额外计算只来自各线UNCOMMITTED_RECOVERY_rank0*.json中已完成但未保留的记录；同一update若在不同恢复事件再次丢弃，按不同已发生执行计额外工作，未记录的in-flight耗费无法完整重建。无此类文件时明确为0。\n\n'
 text+=table(activity,[('branch','路线'),('committed_new_updates','有效轨迹新增更新'),('new_presentations','有效新增呈现'),('extra_discarded_completed_updates','额外已完成未保留更新'),('extra_recorded_presentations','额外已记录呈现'),('recovery_record_files','恢复记录文件')])+'\n'
 text+='工程门均丢弃：旧noon的2更新门/1更新恢复门及原8卡reference单列；新延长段每线1更新门+1更新恢复门，四门合计4更新/384样本呈现，均不带入正式训练。以下按各进程实际完成量计，不用累计new_updates代替该进程次数：\n\n'
 if gate_work:text+=table(gate_work,[('campaign_stage','阶段'),('gate','工程门'),('branch','路线'),('physical_NPUs','卡'),('completed_updates_this_gate_process','该进程更新'),('recorded_presentations','该门呈现'),('status','范围')])+'\n'
 else:text+='尚无完成的工程门结果。\n\n'
 text+='累计/延长段呈现分别=96×对应有效更新，不是新增独立图。ACTIVITY_AND_COST.csv中的seconds是rank0已记录更新段之和，包含范围有限，**不是CPU核时、完整分布式critical time或GPU利用小时**；验证/保存/工程gate、恢复与空档另由controller actor记录。不能由1号卡driver context推断某OS进程存活或被终止。\n\n'
 text+='## 训练与固定验证曲线\n\n'
 text+='两种loss和梯度norm量纲不同，分图只看各自动态，不能横比绝对值或以loss下降宣称恢复改善。EMA monitor每G100图=25来源×4ξ；完整验证每G1000图=250来源×4ξ，使用不同marker。旧U0/父Final水平线仅在同25来源重聚合的monitor子集可得时绘制，未把完整5000均值冒充monitor基线。\n\n![动态](../figures/CAMPAIGN_DYNAMICS.png)\n\n'
 text+='## 完整质量结果\n\n'
 text+='**best覆盖同一条3+5路线从noon到18点所有已完成5000全量验证的checkpoint，按G4 PSNR最大、LPIPS再择优选出。** 历史已完成full保留可选best角色，可能仍指向旧noon根；最终要求每条指针与完整state/EMA digest/SUMMARY SHA完成绑定；当前是否满足以pending及验收状态为准。旧U0/父Final只作外部基线，不纳入best。noon父U55760/U20000在交接时未完成full，其partial现场保留，不作为full候选。全campaign监控最优U由两根目录已完成EMA monitor合并后按G4 PSNR、LPIPS、早U规则选出；延长段监控最优U仅为当前extension的BEST_MONITOR marker，二者分列，不写回worker marker。监控最优未必做过完整full验证，不能当作full-best或泛化成功，也不增加既定full候选池。两个role指向同一验证时不算重复模型证据。\n\n'
 if any(r['line'] in BRANCHES for r in quality):text+=table(quality,[('line','模型/路线'),('role','角色'),('campaign_stage','阶段'),('update','U'),('g','G'),('psnr','PSNR↑'),('ssim','SSIM↑'),('mae','MAE↓'),('lpips','LPIPS↓'),('delta_psnr_vs_Input','ΔInput dB'),('delta_psnr_vs_U0','ΔU0 dB'),('delta_psnr_vs_parentFinal','Δ父Final dB')])+'\n'
 else:text+='两条新路线尚无完成的全量结果。旧U0/父Final已审计基线另存FULL_QUALITY_COMPARISON.csv，不能替代本轮结果。\n\n'
 raw_latest=[]
 for b in BRANCHES:
  candidates=[q for q in branches[b]['validations'] if q['weights']=='RAW']
  if candidates:
   q=max(candidates,key=lambda q:q['update'])
   raw_latest.extend({'branch':b,'update':q['update'],'campaign_stage':q['campaign_stage'],'g':gkey(g),'count':q['per_g'],'sources':q['sources_per_g'],**m} for g,m in q['by_g'].items())
 if raw_latest:
  text+='最新已完成RAW monitor另列（每G100图，不与EMA full混称）：\n\n'+table(raw_latest,[('branch','路线'),('update','U'),('campaign_stage','阶段'),('g','G'),('count','图数'),('psnr','PSNR↑'),('ssim','SSIM↑'),('mae','MAE↓'),('lpips','LPIPS↓')])+'\n'
 text+='所有主结果为原生100步native reverse SDE、相同来源/seed、raw float RGB、无clip/crop/resize；RAW小monitor与EMA完整结果分开。LPIPS改善可能与PSNR/SSIM下降并存，逐项报告。验证隔离并恢复训练RNG由worker/独立门检查，本报告不加载权重、构造恢复网络或重新算神经指标。完整性和恢复目标是否达到是不同结论。\n\n'
 text+='## 预声明固定图\n\n'
 text+='固定源为MONITOR_SELECTION中前2来源的G4、各按sample_id排序取首ξ，不按输出挑图；记录于FIXED_IMAGE_SELECTION.json。每列Input、旧U0、父Final、新original、新improved、GT。阶段图仅用已有已完成验证；可能显示明确标注的monitor候选，空白列表示尚无保存结果，不调用模型补图。图像仅显示时clamp[0,1]；指标仍为未经裁剪原浮点值，SHA/min/max/越界比例存FIGURE_MANIFEST。\n\n![固定best](../figures/FIXED_G4_BEST.png)\n\n![固定latest](../figures/FIXED_G4_LATEST.png)\n\n'
 for q in fixed:text+='图'+q['role']+'所用新验证：'+json.dumps(q['selected_validation'],ensure_ascii=False)+'。\n\n'
 text+='## 验收、范围与限制\n\n'
 hist=cpu_provenance['historical'];invocation=cpu_provenance['current']
 text+='本报告生成器本次恢复网络调用=0、指标网络调用=0、指标模型构造=0、optimizer更新=0；只读已有日志、指标和预测。\n\n'
 if hist['unique_validation_signatures']:
  text+='历史已完成CPU审计来源：`'+hist['source_aggregate']+'`，SHA `'+str(hist['source_sha256'])+'`。按validation signature去重后'+str(hist['unique_validation_signatures'])+'套验证，历史记录指标forward '+str(hist['CPU_metric_forward_batches'])+'个batch、指标模型构造'+str(hist['CPU_metric_model_constructions'])+'次；这些是旧noon已发生的审计成本，不是本次reporter调用或延长段新增开销。\n\n'
 if invocation['invocation_receipt_present']:
  text+='本次最终CPU审计实际invocation来源：`'+invocation['source_aggregate']+'`，SHA `'+str(invocation['source_sha256'])+'`。按signature去重后新指标forward '+str(invocation['CPU_metric_forward_batches'])+'个batch、指标模型构造'+str(invocation['CPU_metric_model_constructions'])+'次，复用历史cache '+str(invocation['cache_reused_unique_signatures'])+'套；命中cache仅重核SHA，新增指标forward=0，未把历史reference batch数重复相加。\n\n'
 else:
  text+='当前尚无新extension wrapper的最终CPU审计invocation收据；复制的历史aggregate/cache不计作本次最终审计或延长段新增成本。最终wrapper会在完成后提供实际调用收据，历史/本次/本reporter分列于CPU_POSTPROCESS_WORK.csv，逐signature来源见CPU_AUDIT_VALIDATION_PROVENANCE.csv。\n\n'
 text+=f"独立CPU指标审计绑定状态：{'PASS且SUMMARY SHA匹配' if ready else '尚未满足完整交付条件'}；固定图目视状态：{'已通过匹配PNG SHA的root QA' if visual_ok else '待root/heartbeat对本次新图SHA重新目视确认'}。--final仅在两线best/latest完整full、controller完成及CPU保存输出审计PASS并绑定实际SUMMARY后允许生成最终报告。\n\n"
 text+='G1–G4为同模拟器训练强度，G0.5仅作较弱强度验证；这里是内容留出验证，未建立真实外部湍流或Test泛化。18点后评价可以继续；18点后不得再开始新的训练更新，在途更新于完整边界保存收尾。改进延长段固定floor2e-7，属于明确披露的复合日程，不能把延长结果单归于loss或LR。3+5资源分配服务于实用训练机会，不证明某配方算力效率或单因果优越。旧父实验与封存forensic audit没有被本报告改写。\n\n'
 if WARNINGS:text+='当前读取备注：'+'；'.join(WARNINGS)+'。\n\n'
 command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]];text+='生成命令：`'+' '.join(command)+'`。输入文件snapshot SHA、脚本SHA、输出SHA见REPORT_MANIFEST.json；活动日志SHA对应实际读取的完整字节前缀，随后追加不会追溯改变本次CSV。\n'
 name='FINAL_REPORT_ZH.md' if args.final and ready else 'STAGE_REPORT_ZH.md';(OUT/name).write_text(text);(OUT/'REPORT_ZH.md').write_text(text)
 result={'status':phase,'final_requested':args.final,'final_requirements_pass':ready,'pending':pending,'root_status':root_status,'branches':activity,'report':str(OUT/name),'fixed_sources':selection.get('sources'),'new_restoration_network_calls':0,'metric_network_calls':0,'optimizer_updates_by_reporter':0,'cpu_cost_provenance':cpu_provenance,'shared_contexts_source':str((ROOT/'protocol/SHARED_CONTEXTS.json').resolve()) if shared_contexts else None,'figure_reused':reuse,'visual_qa_pass':visual_ok,'generated_unix':time.time()};write(OUT/'REPORT_STATUS.json',result)
 outputs={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in list(OUT.glob('*.csv'))+list(OUT.glob('*.md'))+list(FIG.glob('CAMPAIGN_DYNAMICS.*'))+list(FIG.glob('FIXED_G4_*.*'))+[OUT/'FIGURE_MANIFEST.json',OUT/'REPORT_STATUS.json'] if p.is_file()}
 write(OUT/'REPORT_MANIFEST.json',{'status':phase,'source_sha256':digest(Path(__file__).read_bytes()),'command_argv':command,'inputs':INPUTS,'output_sha256':outputs,'required_final_summary_sha256':required,'elapsed_seconds':time.time()-started,'calls':{'restoration_network':0,'metric_network':0,'optimizer_updates':0}})
 print(json.dumps({'status':phase,'report':str(OUT/name),'final_pending':pending,'figures':len(fm['figures']),'new_model_calls':0},ensure_ascii=False,indent=2))
 if args.final and not ready:raise SystemExit(2)

if __name__=='__main__':main()

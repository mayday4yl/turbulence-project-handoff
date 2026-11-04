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

def collect_branch(branch):
 run=ROOT/'runs'/branch;base=45411 if branch=='original' else 0
 cfg=js(run/'CONFIG.json');status=js(run/'STATUS.json');latest=js(run/'LATEST.json')
 if cfg:assert cfg['branch']==branch and cfg['phase']=='formal','Gate metadata must not be reported as formal training'
 best=js(run/'BEST_FULL.json') or js(run/'BEST.json');monitor=js(run/'BEST_MONITOR.json')
 rr=lines(run/'UPDATES_rank0.jsonl');by={}
 for r in rr:
  assert r['branch']==branch and r['update']>=base and r['new_updates']==r['update']-base
  u=int(r['update']);assert u not in by,'Duplicate committed update in rank0 log';by[u]=r
 slim=[]
 for u,r in sorted(by.items()):
  slim.append({k:r.get(k) for k in ['branch','update','new_updates','cursor','started_unix','completed_unix','seconds','loss','lr','grad_norm','grad_norm_after_clip']})
 complete=[]
 for path in sorted((run/'validation').glob('*/SUMMARY.json')):
  q=summary(path,new=True)
  if q is not None:
   assert q['branch']==branch;complete.append(q)
 def locate(update,full=None):
  choices=[q for q in complete if q['weights']=='EMA' and q['update']==update and (full is None or q['full']==full)]
  return sorted(choices,key=lambda q:q['full'],reverse=True)[0] if choices else None
 bestfull=None
 if best:
  assert best['full'] is True and best['weights']=='EMA' and best['selection_role']=='full_validation'
  assert Path(best['checkpoint']['path']).resolve().is_relative_to(run.resolve()),'New best points outside its own campaign line'
  bestfull=locate(best['update'],True)
 latestfull=locate(latest['update'],True) if latest else None
 newest=sorted([q for q in complete if q['weights']=='EMA'],key=lambda q:(q['update'],q['full']),reverse=True)
 latest_available=(locate(latest['update']) if latest else None) or (newest[0] if newest else None)
 best_image=bestfull or (locate(monitor['update'],False) if monitor else None) or latest_available
 if bestfull:
  candidates=[q for q in complete if q['weights']=='EMA' and q['full']]
  winner=max(candidates,key=lambda q:(q['by_g']['4']['psnr'],-q['by_g']['4']['lpips'],-q['update']))
  assert winner['update']==bestfull['update'],'BEST_FULL is not best among completed campaign full summaries'
 return {'branch':branch,'run':str(run),'base':base,'config':cfg,'status':status,'latest':latest,'best':best,'monitor_best':monitor,'training':slim,'validations':complete,'best_full_summary':bestfull,'latest_full_summary':latestfull,'latest_available_summary':latest_available,'best_image_summary':best_image}

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
 for folder in sorted((ROOT/'gates').glob('*')):
  if not folder.is_dir():continue
  q=js(folder/'RESULT.json');rr=lines(folder/'UPDATES_rank0.jsonl')
  if q and q.get('status')=='PASS':
   r=q['ranks'][0];n=int(r['completed_updates_this_process']);branch=q['branch'];world=r['world'];state='PASS_GATE_ONLY'
  elif rr:
   n=len(rr);branch=rr[0]['branch'];rank=js(folder/'RANK_0.json');world=rank.get('world') if rank else None;state='PARTIAL_GATE_LOG_ONLY'
  else:continue
  out.append({'gate':folder.name,'branch':branch,'physical_NPUs':world,'completed_updates_this_gate_process':n,'recorded_presentations':96*n,'status':state,'carried_into_formal':False,
   'scope':'2-step initial gate and1-step resume gate are discarded separately; original8 reference is additional engineering work. Unrecorded in-flight failure work cannot be reconstructed here.'})
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
  ax.set_xlabel('Added optimizer updates in this campaign');ax.grid(alpha=.2)
  if not any(len(line.get_xdata())>0 for line in ax.lines):ax.text(.5,.5,'Waiting for formal data',ha='center',va='center',transform=ax.transAxes)
 fig.suptitle('Stage/final campaign dynamics: monitor and full populations are separate; loss/gradient units differ',fontsize=12);fig.tight_layout()
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
  for name,title in [('U0','Original U0 EMA'),('parentFinal','Parent Final EMA'),('original','New original EMA'),('improved','New improved EMA')]:
   r=records[name].get(key);q=chosen.get(name);suffix=('\nU'+str(q['update'])+(' full' if q['full'] else ' monitor')) if q else ''
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
 return {'role':role,'selected_validation':{b:({'path':q['_path'],'update':q['update'],'full':q['full']} if q else None) for b,q in chosen.items()},'assets':assets,'scope':'Stage fallback is a completed monitor candidate, explicitly labeled; final best/latest must resolve full5000 summaries.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
 started=time.time();authority=js(ROOT/'protocol/USER_AUTHORITY.json');root_status=js(ROOT/'STATUS.json');selection,selected=fixed_selection();monitor_path=ROOT/'protocol/MONITOR_SELECTION.json'
 if not monitor_path.exists():monitor_path=OLD/'protocol/MONITOR_SELECTION.json'
 monitor=js(monitor_path);branches={b:collect_branch(b) for b in BRANCHES};baselines,old_records,matched,inputs=old_baselines(set(monitor['sources']))
 updates=[r for b in BRANCHES for r in branches[b]['training']];vr=[];quality=[];activity=[];recovery=[];gate_work=engineering_work()
 actor_events=lines(ROOT/'evidence/ACTOR_EVENTS.jsonl')
 for b in BRANCHES:
  line=branches[b];last=line['training'][-1] if line['training'] else None
  replay=recovery_work(b,line['run'],last['update'] if last else line['base']);recovery.extend(replay)
  stopped=js(Path(line['run'])/'TRAINING_STOPPED.json');actors=[r for r in actor_events if r.get('event')=='launched' and r.get('phase')=='formal' and r.get('branch')==b]
  activity.append({'branch':b,'physical_NPUs':len(authority['allocation']['original_continue' if b=='original' else 'improved']),'initial_update':line['base'],
   'committed_new_updates':last['new_updates'] if last else 0,'new_presentations':96*last['new_updates'] if last else 0,'last_global_update':last['update'] if last else None,
   'first_formal_actor_start_unix':min(r['started_unix'] for r in actors) if actors else None,'training_stop_detection_unix':stopped.get('detected_unix') if stopped else None,'final_evaluation_completed_unix':line['status'].get('completed_unix') if line['status'] else None,
   'extra_discarded_completed_updates':sum(r['discarded_completed_updates'] for r in replay),'extra_recorded_presentations':sum(r['extra_recorded_presentations'] for r in replay),'recovery_record_files':len(replay),
   'formal_actor_attempts':len(actors),'configured_cards':','.join(map(str,authority['allocation']['original_continue' if b=='original' else 'improved'])),'last_completed_unix':last['completed_unix'] if last else None,'rank0_recorded_update_seconds':sum(r['seconds'] for r in line['training']),
   'worker_status':line['status'].get('status') if line['status'] else 'FORMAL_NOT_STARTED','latest_checkpoint_update':line['latest']['update'] if line['latest'] else None,
   'best_full_update':line['best_full_summary']['update'] if line['best_full_summary'] else None,'best_monitor_update':line['monitor_best']['update'] if line['monitor_best'] else None})
  for q in line['validations']:
   for g,m in q['by_g'].items():vr.append({'branch':b,'update':q['update'],'new_updates':q['update']-line['base'],'weights':q['weights'],'full':q['full'],'g':gkey(g),'per_g':q['per_g'],'sources_per_g':q['sources_per_g'],'summary':q['_path'],'summary_sha256':q['_sha256'],**m})
  for role,key in [('best_full','best_full_summary'),('latest_full','latest_full_summary')]:
   q=line[key]
   if q:
    for g,m in q['by_g'].items():quality.append({'line':b,'role':role,'update':q['update'],'new_updates':q['update']-line['base'],'g':gkey(g),'count':1000,'sources':250,**m,'summary':q['_path'],'summary_sha256':q['_sha256']})
 for name,q in baselines.items():
  if q:
   for g,m in q['by_g'].items():quality.append({'line':name,'role':'external_fixed_baseline','update':q['update'],'new_updates':None,'g':gkey(g),'count':1000,'sources':250,**m,'summary':q['_path'],'summary_sha256':q['_sha256']})
 for g,m in inputs.items():quality.append({'line':'Input','role':'input_baseline','update':None,'new_updates':None,'g':g,'count':1000,'sources':250,**m})
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
   else:required[q['_path']]=q['_sha256']
 model_done=js(ROOT/'TRAINING_AND_MODEL_VALIDATION_COMPLETE.json')
 if not model_done or model_done.get('status')!='PASS':pending.append('controller model-validation completion receipt missing')
 cpu=js(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json')
 if not cpu or cpu.get('status')!='PASS':pending.append('independent saved-output CPU audit not PASS')
 else:
  audited=cpu.get('audited_summaries',{})
  for p,h in required.items():
   if audited.get(p)!=h:pending.append('CPU audit not bound to actual summary '+p)
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
 text+=f"生成于{now}。状态：**{phase}**。原续训线新增{activity[0]['committed_new_updates']}次更新、改进线新增{activity[1]['committed_new_updates']}次更新；截止2026-11-03 12:00 Asia/Shanghai。工程gate、CPU数学PASS或小monitor不代表正式训练完成、收敛或泛化成功。\n\n"
 if root_status:text+='当前根执行状态：`'+str(root_status.get('status'))+'`。\n\n'
 if pending:text+='尚待：'+'；'.join(pending)+'。\n\n'
 text+='## 两条路线与实际工作量\n\n'
 text+='原线从父Final RAW U45411及Adam/EMA/8逻辑RNG继续，原native mean-L1、lr1e-5；3张物理卡模拟8条逻辑B12流。改进线从官方完整SR U0重新初始化，fresh Adam，epsilon_opt MSE、LR2e-7→2e-6 warmup200后wall-clock cosine→2e-7、global clip1.0；5卡局部B20/19/19/19/19。两线均全参数、B96、FP32/HF32off、原网络/桥、无task prompt/Adapter/raw-xi/A/B或物理递归。\n\n'
 text+='这是起点、objective、schedule和资源都不同的实用综合recipe比较，不是单变量因果ablation或吞吐benchmark。原L1没有被证明实现错误；epsilon_opt目标不是实际forward epsilon，loss为mean((noise−epsilon_opt)²)，不再重复除c²。\n\n'
 text+=table(activity,[('branch','路线'),('physical_NPUs','卡'),('initial_update','起始U'),('committed_new_updates','本轮新增更新'),('new_presentations','本轮呈现'),('latest_checkpoint_update','latest U'),('best_full_update','full best U'),('best_monitor_update','monitor best U'),('worker_status','状态')])+'\n'
 text+=table(activity,[('branch','路线'),('first_formal_actor_start_unix','首正式actor启动Unix'),('training_stop_detection_unix','停训检测Unix'),('final_evaluation_completed_unix','终评完成Unix'),('formal_actor_attempts','正式/恢复actor数')])+'\n'
 text+='主线新增更新是当前有效轨迹global update−base；恢复回放不会被重复计为有效新更新。以下额外计算只来自各线UNCOMMITTED_RECOVERY_rank0*.json中已完成但未保留的记录；同一update若在不同恢复事件再次丢弃，按不同已发生执行计额外工作，未记录的in-flight耗费无法完整重建。无此类文件时明确为0。\n\n'
 text+=table(activity,[('branch','路线'),('committed_new_updates','有效轨迹新增更新'),('new_presentations','有效新增呈现'),('extra_discarded_completed_updates','额外已完成未保留更新'),('extra_recorded_presentations','额外已记录呈现'),('recovery_record_files','恢复记录文件')])+'\n'
 text+='工程门及其恢复检查均不带入正式训练，2更新门与1更新恢复门按每个gate进程分别计数，原8卡reference另列，不能把gate的new_updates_total=3误当恢复进程又做3次：\n\n'
 if gate_work:text+=table(gate_work,[('gate','工程门'),('branch','路线'),('physical_NPUs','卡'),('completed_updates_this_gate_process','该进程更新'),('recorded_presentations','该门呈现'),('status','范围')])+'\n'
 else:text+='尚无完成的工程门结果。\n\n'
 text+='新增呈现=96×本轮更新，不是新增独立图。ACTIVITY_AND_COST.csv中的seconds是rank0已记录更新段之和，包含范围有限，**不是CPU核时、完整分布式critical time或GPU利用小时**；验证/保存/工程gate、恢复与空档另由controller actor记录。不能由1号卡driver context推断某OS进程存活或被终止。\n\n'
 text+='## 训练与固定验证曲线\n\n'
 text+='两种loss和梯度norm量纲不同，分图只看各自动态，不能横比绝对值或以loss下降宣称恢复改善。EMA monitor每G100图=25来源×4ξ；完整验证每G1000图=250来源×4ξ，使用不同marker。旧U0/父Final水平线仅在同25来源重聚合的monitor子集可得时绘制，未把完整5000均值冒充monitor基线。\n\n![动态](../figures/CAMPAIGN_DYNAMICS.png)\n\n'
 text+='## 完整质量结果\n\n'
 text+='**新best仅限各自本轮已完成5000全量验证的checkpoint，按G4 PSNR最大、LPIPS再择优选出。** 旧U0/父Final只作同协议外部基线，未纳入新best。若本轮只有最终一套full，则best=latest合理；BEST_MONITOR独立列出，不能称全量best。两个role指向同一验证时不算重复模型证据。\n\n'
 if any(r['line'] in BRANCHES for r in quality):text+=table(quality,[('line','模型/路线'),('role','角色'),('update','U'),('g','G'),('psnr','PSNR↑'),('ssim','SSIM↑'),('mae','MAE↓'),('lpips','LPIPS↓'),('delta_psnr_vs_Input','ΔInput dB'),('delta_psnr_vs_U0','ΔU0 dB'),('delta_psnr_vs_parentFinal','Δ父Final dB')])+'\n'
 else:text+='两条新路线尚无完成的全量结果。旧U0/父Final已审计基线另存FULL_QUALITY_COMPARISON.csv，不能替代本轮结果。\n\n'
 text+='所有主结果为原生100步native reverse SDE、相同来源/seed、raw float RGB、无clip/crop/resize；RAW小monitor与EMA完整结果分开。LPIPS改善可能与PSNR/SSIM下降并存，逐项报告。验证隔离并恢复训练RNG由worker/独立门检查，本报告不加载权重、构造恢复网络或重新算神经指标。完整性和恢复目标是否达到是不同结论。\n\n'
 text+='## 预声明固定图\n\n'
 text+='固定源为MONITOR_SELECTION中前2来源的G4、各按sample_id排序取首ξ，不按输出挑图；记录于FIXED_IMAGE_SELECTION.json。每列Input、旧U0、父Final、新original、新improved、GT。阶段图仅用已有已完成验证；可能显示明确标注的monitor候选，空白列表示尚无保存结果，不调用模型补图。图像仅显示时clamp[0,1]；指标仍为未经裁剪原浮点值，SHA/min/max/越界比例存FIGURE_MANIFEST。\n\n![固定best](../figures/FIXED_G4_BEST.png)\n\n![固定latest](../figures/FIXED_G4_LATEST.png)\n\n'
 for q in fixed:text+='图'+q['role']+'所用新验证：'+json.dumps(q['selected_validation'],ensure_ascii=False)+'。\n\n'
 text+='## 验收、范围与限制\n\n'
 text+=f"独立CPU指标审计绑定状态：{'PASS且SUMMARY SHA匹配' if ready else '尚未满足完整交付条件'}；固定图目视状态：{'已通过匹配PNG SHA的root QA' if visual_ok else '待root/heartbeat目视确认'}。--final仅在两线best/latest完整full、controller完成及CPU保存输出审计PASS并绑定实际SUMMARY后允许生成最终报告。\n\n"
 text+='G1–G4为同模拟器训练强度，G0.5仅作较弱强度验证；这里是内容留出验证，未建立真实外部湍流或Test泛化。3+5资源分配服务于实用训练机会，不证明某配方算力效率或单因果优越。旧父实验与封存forensic audit没有被本报告改写。\n\n'
 if WARNINGS:text+='当前读取备注：'+'；'.join(WARNINGS)+'。\n\n'
 command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]];text+='生成命令：`'+' '.join(command)+'`。输入文件snapshot SHA、脚本SHA、输出SHA见REPORT_MANIFEST.json；活动日志SHA对应实际读取的完整字节前缀，随后追加不会追溯改变本次CSV。\n'
 name='FINAL_REPORT_ZH.md' if args.final and ready else 'STAGE_REPORT_ZH.md';(OUT/name).write_text(text);(OUT/'REPORT_ZH.md').write_text(text)
 result={'status':phase,'final_requested':args.final,'final_requirements_pass':ready,'pending':pending,'root_status':root_status,'branches':activity,'report':str(OUT/name),'fixed_sources':selection.get('sources'),'new_restoration_network_calls':0,'metric_network_calls':0,'optimizer_updates_by_reporter':0,'figure_reused':reuse,'visual_qa_pass':visual_ok,'generated_unix':time.time()};write(OUT/'REPORT_STATUS.json',result)
 outputs={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in list(OUT.glob('*.csv'))+list(OUT.glob('*.md'))+list(FIG.glob('CAMPAIGN_DYNAMICS.*'))+list(FIG.glob('FIXED_G4_*.*'))+[OUT/'FIGURE_MANIFEST.json',OUT/'REPORT_STATUS.json'] if p.is_file()}
 write(OUT/'REPORT_MANIFEST.json',{'status':phase,'source_sha256':digest(Path(__file__).read_bytes()),'command_argv':command,'inputs':INPUTS,'output_sha256':outputs,'required_final_summary_sha256':required,'elapsed_seconds':time.time()-started,'calls':{'restoration_network':0,'metric_network':0,'optimizer_updates':0}})
 print(json.dumps({'status':phase,'report':str(OUT/name),'final_pending':pending,'figures':len(fm['figures']),'new_model_calls':0},ensure_ascii=False,indent=2))
 if args.final and not ready:raise SystemExit(2)

if __name__=='__main__':main()

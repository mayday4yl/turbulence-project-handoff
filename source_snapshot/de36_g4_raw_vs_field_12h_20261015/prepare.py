"""CPU-only manifest freeze and asset audit; never opens Test arrays."""
from common import *
import collections,subprocess,importlib.metadata

def main():
 d=Path(CONFIG['dataset_root']);old=Path('/data2/yanglin/v4_restormer_train2000_npu_20k_20261003')
 assert d.is_dir() and old.is_dir() and (d/'DATA_READY.json').is_file()
 rows=[]
 for line in (d/'source_manifest.jsonl').open():
  if '"split": "test"' in line:continue
  r=json.loads(line)
  if r['split'] in ('train','val'):rows.append(r)
 selected={}
 for split,n in [('train',128),('val',32)]:
  buckets=collections.defaultdict(list)
  for r in rows:
   if r['split']==split:buckets[str(r.get('category',r.get('category_id','unknown')))].append(r)
  for b in buckets.values():b.sort(key=lambda r:seed_for(f"{CONFIG['seed']}|{split}|{r['source_id']}"))
  order=sorted(buckets,key=lambda k:seed_for(f"{CONFIG['seed']}|category|{k}"));chosen=[]
  while len(chosen)<n:
   for k in order:
    if buckets[k] and len(chosen)<n:chosen.append(buckets[k].pop(0))
  selected[split]=chosen
 ids={s:{r['source_id'] for r in rr} for s,rr in selected.items()};assert not ids['train']&ids['val']
 trajectories={'train':[],'val':[]}
 for line in (d/'trajectory_manifest.jsonl').open():
  if '"split": "test"' in line:continue
  r=json.loads(line);s=r['split']
  if s in ids and r['source_id'] in ids[s] and r['plan']['realization_index'] in (0,1):trajectories[s].append(r)
 for s in trajectories:
  rank={r['source_id']:i for i,r in enumerate(selected[s])}
  trajectories[s].sort(key=lambda r:(rank[r['source_id']],r['plan']['realization_index']))
  assert len(trajectories[s])==2*len(selected[s])
  for r in trajectories[s]:
   assert r['arrays']['clean']['shape']==[3,256,256] and r['arrays']['X_4']['shape']==[3,256,256]
 save_json(ROOT/'protocol/SAMPLES.json',trajectories)
 save_json(ROOT/'protocol/SOURCES.json',selected)
 refinement=[r['trajectory_id'] for r in trajectories['val'][::2][:8]]
 save_json(ROOT/'protocol/REFINEMENT_SUBSET_CANDIDATES.json',{'ordered_8_distinct_sources':refinement,'fallback_4':refinement[:4],'selection_before_val_metrics':True})
 files={str(d/f):digest(d/f) for f in ['DATA_READY.json','SOURCE_FREEZE.json','source_manifest.jsonl','trajectory_manifest.jsonl']}
 for f in ['I0_I2.pt','P2S_model.pt','dictionary.pt']:
  p=Path(CONFIG['asset_dir'])/f;files[str(p)]=digest(p)
 originals={}
 for f in ['physical_bridge/model.py','physical_bridge/simulator.py','v4/model.py','v4/refinement.py']:
  originals[f]={'remote_original':str(old/f),'original_sha256':digest(old/f),'independent_sha256':digest(ROOT/f)}
  assert originals[f]['original_sha256']==originals[f]['independent_sha256']
 versions={}
 for pkg in ['torch','torch-npu','numpy']:
  try:versions[pkg]=importlib.metadata.version(pkg)
  except importlib.metadata.PackageNotFoundError:versions[pkg]='missing'
 cann={str(p):p.read_text(errors='replace') for p in Path('/usr/local/Ascend').glob('**/ascend_toolkit_install.info')}
 checkpoints=[{'path':str(p),'bytes':p.stat().st_size} for p in old.glob('runs/formal_20k/*/checkpoints/*.pt')]
 save_json(ROOT/'reports/ASSET_AUDIT.json',{'epoch':time.time(),'hashes':files,'reused_sources':originals,'versions':versions,'CANN':cann,'available_historical_checkpoints_not_loaded':checkpoints,'npu_smi':subprocess.check_output(['npu-smi','info'],text=True),'data_counts':{s:len(x) for s,x in trajectories.items()},'test_arrays_opened':False,'import_root':str(ROOT)})
 print('CPU_PREPARATION_PASS',flush=True)
if __name__=='__main__':main()

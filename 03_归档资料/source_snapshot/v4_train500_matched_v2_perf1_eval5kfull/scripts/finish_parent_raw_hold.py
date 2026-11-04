"""Finish active RAW U0 evaluations, preserve checkpoints, stop superseded EMA work.
Only operates the recorded parent attempt, before any optimizer update.
"""
import hashlib,json,os,pathlib,signal,time
ROOT=pathlib.Path(__file__).resolve().parents[1];PARENT=ROOT.parent/'v4_train500_matched_v2'
RUN=ROOT.parents[1]/'runs/v4_train500_matched_v2'

def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def owned(pid,script):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return False
 tokens=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 cwd=(p/'cwd').resolve()
 expected=PARENT/'scripts'/script
 matches=any((pathlib.Path(t) if pathlib.Path(t).is_absolute() else cwd/t).resolve()==expected.resolve() for t in tokens[1:] if not t.startswith('-'))
 assert matches and cwd==PARENT.resolve(),(pid,'Unexpected process identity')
 return True

def main():
 ids=json.loads((RUN/'PIDS.json').read_text());record={'purpose':'USER_AUTHORIZED_EVALUATION_SCHEDULE_TRANSITION','parent_attempt':ids['attempt'],
    'parent_frozen_sha256':digest(PARENT/'protocol/FROZEN.json'),'time':time.time(),'workers':{},'optimizer_updates':0,
    'note':'Completed RAW evaluation retained. Already completed EMA retained. Incomplete EMA preserved as incomplete, never marked successful. Resume from verified complete U0 checkpoint.'}
 out=ROOT/'reports/PARENT_RAW_HOLD.json'
 # Stop only the parent coordinator's dispatch/health loop while individual RAW evals finish.
 if owned(ids['coordinator'],'run_formal.py'):os.kill(ids['coordinator'],signal.SIGSTOP)
 for g,pid in ids['workers'].items():
  steps=RUN/g/'train_steps.jsonl';assert not steps.exists() or not steps.read_text().strip()
  latest=json.loads((RUN/g/'LATEST.json').read_text());assert latest['update']==0
  ck=RUN/g/latest['path'];assert digest(ck)==latest['sha256']
  record['workers'][g]={'pid':pid,'checkpoint':str(ck),'checkpoint_sha256':latest['sha256'],'status':'WAITING_RAW_COMMIT'}
 write(out,record)
 remaining=set(ids['workers'])
 while remaining:
  if (PARENT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Parent failure, do not bypass')
  for g in list(remaining):
   base=RUN/g/'evaluations/U00000/raw'
   required=[base/pop/'COMMIT.json' for pop in ['train8000','val800','train_monitor400']]
   if not all(p.exists() for p in required):continue
   for commit in required:
    data=json.loads(commit.read_text())
    for rel,h in data['file_sha256'].items():assert digest(commit.parent/rel)==h
   pid=ids['workers'][g]
   if owned(pid,'train_worker.py'):
    os.kill(pid,signal.SIGSTOP);time.sleep(.5)
    # Inference is read-only; complete RAW commits and resumable U0 state verified above.
    record['workers'][g]['incomplete_EMA']=[str(p.relative_to(RUN/g)) for p in (RUN/g/'evaluations/U00000/EMA').glob('*.partial')]
    os.kill(pid,signal.SIGKILL)
   record['workers'][g].update(status='ENGINEERING_STOPPED_AFTER_RAW_U0',raw_complete=True,time=time.time())
   write(RUN/g/'STATUS.json',{'status':'ENGINEERING_STOPPED_AFTER_RAW_U0','update':0,'exposures':0,
      'checkpoint_verified':True,'raw_evaluation_complete':True,'reason':'Authorized schedule revision and separately authorized engineering tuning'})
   remaining.remove(g);write(out,record)
  time.sleep(1)
 if owned(ids['coordinator'],'run_formal.py'):os.kill(ids['coordinator'],signal.SIGKILL)
 record['status']='PASS_COMPLETE_RAW_U0_SAFE_RESUME';record['finished_at']=time.time();write(out,record)

if __name__=='__main__':main()

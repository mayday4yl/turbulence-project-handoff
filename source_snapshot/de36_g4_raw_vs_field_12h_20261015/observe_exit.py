"""Read-only process observer for accurate per-arm card-time end bounds."""
from pathlib import Path
import json,time,os,fcntl
ROOT=Path(__file__).resolve().parent
lock=(ROOT/'reports/exit_observer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
clock=json.loads((ROOT/'reports/EXPERIMENT_CLOCK.json').read_text());tracked={};last_seen={};done={}
while time.time()<clock['deadline_epoch']+60:
 workers=json.loads((ROOT/'reports/WORKERS.json').read_text())
 for w in workers:
  if w['phase']!='train':continue
  key=str(w['pid']);tracked[key]=w
  if key in done:continue
  p=Path('/proc')/key;live=False
  try:
   st=p.joinpath('stat').read_text().split();live=st[21]==w['start_ticks'] and st[2]!='Z'
  except FileNotFoundError:pass
  now=time.time()
  if live:last_seen[key]=now
  elif key in last_seen:done[key]={**w,'last_observed_live_epoch':last_seen[key],'first_observed_exit_epoch':now,'end_uncertainty_seconds':now-last_seen[key],'phase_hours_upper_bound':(now-w['start_epoch'])/3600}
 if done:
  path=ROOT/'reports/TRAIN_PROCESS_EXIT_BOUNDS.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(done,indent=2));os.replace(tmp,path)
 if len(done)==2:break
 time.sleep(5)

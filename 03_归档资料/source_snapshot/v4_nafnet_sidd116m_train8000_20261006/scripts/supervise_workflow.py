"""Detached CPU supervisor. No accelerator initialization and no legacy restarts."""
from pathlib import Path
import sys,os,time,fcntl,subprocess
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from runlib.common import *
from runlib.launch import alive
lock=(R/'reports/supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
statefile=R/'reports/SUPERVISOR.json';state=read(statefile) if statefile.exists() else {'retries':0}
token=str(R/'scripts/workflow_controller.py')
while True:
 if (R/'reports/PHASE1_COMPLETE.json').exists() or (R/'protocol/STOP_FORMAL.json').exists():break
 status=read(R/'reports/WORKFLOW_STATUS.json') if (R/'reports/WORKFLOW_STATUS.json').exists() else {}
 if status.get('status')=='PREFLIGHT_BLOCKED':break
 if not state.get('pid') or not alive(state['pid'],token):
  if state.get('pid'):
   state['retries']+=1
   if state['retries']>3:atomic(statefile,{**state,'status':'SUPERVISOR_RETRY_EXHAUSTED','time':time.time()});break
  with (R/'logs/workflow_controller.log').open('ab') as log:
   env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
   p=subprocess.Popen([sys.executable,'-u',str(R/'scripts/workflow_controller.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
  state.update(pid=p.pid,supervisor_pid=os.getpid(),time=time.time(),status='MONITORING');atomic(statefile,state)
 time.sleep(30)

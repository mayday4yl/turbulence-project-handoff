"""Independent hard wall limit, PID identity and own-directory checks."""
from common import *
import signal
while time.time()<experiment_deadline()-30:
 p=ROOT/'reports/STATUS.json'
 if p.exists() and json.loads(p.read_text()).get('status') in ('COMPLETE','BLOCKED'):raise SystemExit(0)
 time.sleep(min(30,max(1,experiment_deadline()-30-time.time())))
items=json.loads((ROOT/'reports/WORKERS.json').read_text()) if (ROOT/'reports/WORKERS.json').exists() else []
items.append(json.loads((ROOT/'reports/CONTROLLER.json').read_text()))
receipt=[]
for item in items:
 try:
  pid=item['pid'];proc=Path(f'/proc/{pid}')
  if proc.joinpath('stat').read_text().split()[21]!=item['start_ticks']:continue
  if proc.joinpath('cwd').resolve()!=ROOT:continue
  os.killpg(pid,signal.SIGTERM);receipt.append(pid)
 except (FileNotFoundError,ProcessLookupError):pass
save_json(ROOT/'reports/HARD_DEADLINE_STOP.json',{'epoch':time.time(),'terminated_own_groups':receipt,'budget_seconds':43200})
time.sleep(10)
for pid in receipt:
 try:os.killpg(pid,signal.SIGKILL)
 except ProcessLookupError:pass
save_json(ROOT/'reports/STATUS.json',{'status':'STOPPED_AT_12H','epoch':time.time(),'no_autoresume':True})

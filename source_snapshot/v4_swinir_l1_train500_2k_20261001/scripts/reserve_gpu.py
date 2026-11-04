"""User-authorized bounded memory reservation; no fake compute or optimizer updates."""
import json,os,sys,time,signal,subprocess
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];gpu=int(sys.argv[1]);uuid=sys.argv[2]
assert gpu in (2,3) and os.environ['CUDA_VISIBLE_DEVICES']==uuid
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines():
    if line.strip()==uuid:raise RuntimeError('Foreign GPU process; refusing reservation')
free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=70000,free
end=time.time()+3600
stop=False
def finish(*args):
    global stop
    stop=True
signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
allocation=torch.empty(48*1024**3,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
p=root/'reports'/f'GPU{gpu}_RESERVATION.json'
def report(status):
    p.write_text(json.dumps({'status':status,'pid':os.getpid(),'physical_gpu':gpu,'gpu_uuid':uuid,'reserved_GiB':48,'expires_at':end,'time':time.time(),'purpose':'USER_REQUESTED_PREPARATION_RESERVATION_NO_COMPUTE','optimizer_updates':0},indent=2)+'\n')
report('RESERVED');print('RESERVED',gpu,os.getpid(),flush=True)
while not stop and time.time()<end and not (root/'reports/RELEASE_RESERVATION').exists():time.sleep(1)
del allocation;torch.cuda.empty_cache();report('RELEASED');print('RELEASED',gpu,flush=True)

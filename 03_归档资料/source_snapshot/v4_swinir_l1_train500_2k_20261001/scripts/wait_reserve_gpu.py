"""Bounded, user-authorized local GPU waiter. No compute loops or optimizer updates."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];reports=root/'reports'
uuids={2:'GPU-e49c0ca8-8eac-7cec-4ecb-228af638edfb',3:'GPU-f3a8ea0e-dc60-515a-6afa-785da8eca151'}
uuid=uuids[a.gpu];assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
lock=(reports/f'GPU{a.gpu}_WAITER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stop=False

def halt(*args):
 global stop
 stop=True
signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
begin=time.time();deadline=begin+3600;allocation=None

def state(status,**extra):
 out=reports/f'GPU{a.gpu}_WAITER.json';tmp=out.with_suffix('.tmp')
 tmp.write_text(json.dumps({'status':status,'pid':os.getpid(),'gpu':a.gpu,'uuid':uuid,'start_time':begin,'deadline':deadline,'time':time.time(),'optimizer_updates':0,'purpose':'AUTHORIZED_TEMPORARY_RESOURCE_PREPARATION',**extra},indent=2)+'\n');tmp.replace(out)
 print(status,json.dumps(extra),flush=True)

def foreign():
 lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 return [int(l.split(',')[0]) for l in lines if l.split(',')[1].strip()==uuid and int(l.split(',')[0])!=os.getpid()]

state('READY_CPU_ONLY_WAITING_FOR_IDLE',poll_seconds=1,torch_imported=True)
try:
 last=None
 while not stop and time.time()<deadline and not (reports/'RELEASE_RESERVATION').exists():
  others=foreign()
  if others:
   if others!=last:state('WAITING_FOREIGN_PROCESS',foreign_pids=others);last=others
   time.sleep(1);continue
  free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free<70000:time.sleep(1);continue
  if foreign():continue
  allocation=torch.empty(48*1024**3,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
  others=foreign()
  if others:
   del allocation;allocation=None;torch.cuda.empty_cache()
   state('RELEASED_CONCURRENT_FOREIGN_START',foreign_pids=others)
   # Fully destroy CUDA context by exiting; never retain our context as a false reservation.
   break
  state('RESERVED',reserved_GiB=48,allocated_GiB=torch.cuda.memory_allocated()/2**30)
  while not stop and time.time()<deadline and not (reports/'RELEASE_RESERVATION').exists():time.sleep(.5)
  break
finally:
 if allocation is not None:del allocation;torch.cuda.empty_cache()
 state('EXITED',released=True)

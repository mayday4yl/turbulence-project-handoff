"""Warm CPU coordinator: reserve at most two genuinely idle GPUs; never touch foreign jobs."""
import argparse,fcntl,json,multiprocessing as mp,os,signal,subprocess,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';STOP=False

def put(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def signal_stop(*args):
 global STOP
 STOP=True

def apps():
 result={}
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
  pid,uuid=[s.strip() for s in line.split(',')];result.setdefault(uuid,[]).append(int(pid))
 return result

def hold(gpu,uuid,conn,max_seconds):
 global STOP
 STOP=False;signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
 os.environ['CUDA_VISIBLE_DEVICES']=uuid
 allocation=None;start=time.monotonic()
 try:
  if apps().get(uuid):raise RuntimeError('Foreign process appeared before allocation')
  allocation=torch.empty(48*1024**3,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
  if any(pid!=os.getpid() for pid in apps().get(uuid,[])):raise RuntimeError('Foreign process raced allocation; yielding our reservation')
  conn.send({'status':'RESERVED','pid':os.getpid(),'gpu':gpu,'uuid':uuid,'reserved_GiB':48,'optimizer_updates':0})
  while not STOP and time.monotonic()-start<max_seconds and not (R/'RELEASE_RESERVATION').exists():time.sleep(.2)
 except BaseException as e:conn.send({'status':'RESERVATION_REJECTED','gpu':gpu,'uuid':uuid,'error':repr(e)})
 finally:
  if allocation is not None:del allocation;torch.cuda.empty_cache()
  conn.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--until-midnight-seconds',type=float,required=True);a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']='';assert not torch.cuda.is_initialized()
 lock=(R/'RESOURCE_WAITER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
 start=time.monotonic();deadline=start+a.until_midnight_seconds;hard_end=deadline+1800
 active={};events=[];priority=[2,3,0,1,4,7,5,6];last=None
 def record(status,inventory):
  data={'status':status,'pid':os.getpid(),'time':time.time(),'elapsed_seconds':time.monotonic()-start,'seconds_until_user_midnight':max(0,deadline-time.monotonic()),'deadline_user_timezone':'Asia/Shanghai','maximum_our_GPUs':2,'torch_preimported_CPU_only':True,'reservations':{str(g):v['info'] for g,v in active.items()},'inventory':inventory,'optimizer_updates':0,'events':events[-30:]}
  put(R/'RESOURCE_WAITER.json',data)
 try:
  while not STOP and time.monotonic()<hard_end and not (R/'RELEASE_RESERVATION').exists():
   control=R/'RESOURCE_WAITER_CONTROL.json'
   if control.exists():
    deadline=float(json.loads(control.read_text())['server_monotonic_deadline']);hard_end=deadline+1800
   lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();ap=apps();inventory={}
   for l in lines:
    idx,uuid,free,used,util=[s.strip() for s in l.split(',')]
    inventory[int(idx)]={'uuid':uuid,'free_MiB':int(free),'used_MiB':int(used),'util_percent':int(util),'pids':ap.get(uuid,[])}
   for g,v in list(active.items()):
    if v['conn'].poll():
     try:v['info']=v['conn'].recv();events.append(v['info'])
     except EOFError:pass
    if not v['process'].is_alive():
     v['process'].join();v['conn'].close();del active[g]
   if time.monotonic()<deadline:
    for g in priority:
     item=inventory[g]
     if len(active)>=2:break
     if g in active or item['pids'] or item['free_MiB']<70000 or item['util_percent']>5:continue
     parent,child=mp.get_context('fork').Pipe();process=mp.get_context('fork').Process(target=hold,args=(g,item['uuid'],child,hard_end-time.monotonic()))
     process.start();child.close();active[g]={'process':process,'conn':parent,'info':{'status':'ACQUIRING','gpu':g,'pid':process.pid,'uuid':item['uuid']}}
   status='TWO_GPUS_RESERVED' if len(active)==2 and all(v['info']['status']=='RESERVED' for v in active.values()) else 'MIDNIGHT_FALLBACK_REVIEW_DUE' if time.monotonic()>=deadline else 'MONITORING_ALL_EIGHT_GPUS'
   record(status,inventory)
   if status!=last:print(status,flush=True);last=status
   time.sleep(1)
 finally:
  for v in active.values():
   if v['process'].is_alive():v['process'].terminate()
  for v in active.values():v['process'].join(timeout=15)
  record('EXITED_ALL_OUR_RESERVATIONS_RELEASED',{})
if __name__=='__main__':main()

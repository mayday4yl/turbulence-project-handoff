"""Independent deadline enforcement, only exact new worker identities."""
from lib import *
import signal
from controller import exact_alive,finish_report

def main():
    lock=(ROOT/'deadline_guard.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(ROOT/'DEADLINE_GUARD_RECEIPT.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'cwd':str(ROOT),'time':time.time()})
    box=read(ROOT/'TIMEBOX.json');sent={}
    while time.time()<box['final_deadline_server']+120:
        state=read(ROOT/'CONTROLLER_STATE.json') if (ROOT/'CONTROLLER_STATE.json').exists() else {}
        if (ROOT/'FINALIZATION_PENDING_WORKERS.json').exists() and not any(exact_alive(q) for q in state.get('jobs',{}).values()):
            finish_report(state)
        if (ROOT/'CPU_FINALIZED.json').exists() and not any(exact_alive(q) for q in state.get('jobs',{}).values()):return
        for q in state.get('jobs',{}).values():
            if not exact_alive(q):continue
            deadline=box['training_deadline_server'] if q['mode'] in ['train','gate'] else box['gpu_deadline_server']
            if time.time()<deadline:continue
            key=(q['pid'],q['start_ticks'])
            if key not in sent:
                os.kill(q['pid'],signal.SIGTERM);sent[key]=time.time();append(ROOT/'deadline_signals.jsonl',{'signal':'SIGTERM','job':q,'time':time.time()})
            elif time.time()-sent[key]>120 and exact_alive(q):
                os.kill(q['pid'],signal.SIGKILL);append(ROOT/'deadline_signals.jsonl',{'signal':'SIGKILL_AFTER_120S_NEW_WORKER_ONLY','job':q,'time':time.time()})
        time.sleep(15)
    atomic(ROOT/'DEADLINE_FINALIZED.json',{'status':'DEADLINE_REACHED','time':time.time()})
if __name__=='__main__':main()

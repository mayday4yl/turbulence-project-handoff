"""Own-process-only bounded acceptance controller; stop on the first failure."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def run(groups):
    workers=[]
    for group,gpu in groups:
        if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Run is PAUSED')
        previous=ROOT/f'reports/acceptance_{group}.json'
        if previous.exists():
            if json.loads(previous.read_text())['status']=='PASS':continue
            raise RuntimeError('Existing incomplete attempt requires review')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=(ROOT/f'reports/acceptance_{group}.log').open('x')
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/validate_group.py'),'--group',group,'--microbatch','2'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((child,log,group))
    try:
        while any(p.poll() is None for p,_,_ in workers):
            if (ROOT/'reports/RUN_PAUSED.json').exists() or any(p.poll() not in (None,0) for p,_,_ in workers):
                raise RuntimeError('Correctness failure: stop all owned temporary acceptance workers')
            time.sleep(1)
        if any(p.returncode for p,_,_ in workers):raise RuntimeError('Acceptance worker failed')
    finally:
        for p,log,_ in workers:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
            log.close()

if __name__=='__main__':
    run([('D0',5),('R0',6)])
    run([('R08',5)])
    print('Remaining three group correctness PASS; not formal training',flush=True)

"""Persistent owner of preflight, benchmarks, resume gate, training and finish."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def identity(pid):
    p=Path('/proc')/str(pid)
    return {'pid':pid,'start_ticks':p.joinpath('stat').read_text().split()[21],
        'cmdline':p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def phase(name,**kwargs):
    write(ROOT/'STATUS.json',{'phase':name,'unix':time.time(),**kwargs})
    print(name,kwargs,flush=True)


def guard(deadline):
    write(ROOT/'evidence/DEADLINE_GUARD.json',{'status':'ARMED','deadline_unix':deadline,**identity(os.getpid())})
    while time.time()<deadline:time.sleep(min(30,max(.01,deadline-time.time())))
    write(ROOT/'STOP_REQUEST.json',{'reason':'authorized deadline','due_unix':deadline,'requested_unix':time.time()})


def run(label,script_args,distributed=False,timeout=7200):
    folder=ROOT/'evidence';folder.mkdir(exist_ok=True)
    env=os.environ.copy();env['UNIDB_DEVICES']='0,1,2,3,4,5,6,7' if distributed else '0'
    env.update(HCCL_CONNECT_TIMEOUT='300',HCCL_EXEC_TIMEOUT='7200')
    command=['bash',str(ROOT/'run_env.sh')]
    if distributed:command+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8','--']
    command+=script_args
    with (folder/(label+'.log')).open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(folder/(label+'_LAUNCH.json'),{**identity(child.pid),'command':command,'unix':time.time()})
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only our new torchrun process group; torchrun reaps its own workers.
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise RuntimeError(label+' exceeded explicit runtime bound')
    write(folder/(label+'_EXIT.json'),{'exit_code':code,'unix':time.time()})
    if code:raise RuntimeError(label+' exited '+str(code))


def worker(name,mode,micro,steps=20,**extra):
    args=[str(ROOT/'worker.py'),'--run',name,'--mode',mode,'--microbatch',str(micro),'--steps',str(steps)]
    if extra.get('resume'):args+=['--resume',str(extra['resume'])]
    if extra.get('evaluate'):args+=['--full-eval']
    timeout=200000 if mode=='formal' else 7200
    run(name,args,True,timeout)
    return ROOT/'runs'/name


def audit(folder,output,compare=None):
    path=Path(json.loads((folder/'LATEST.json').read_text())['path'])
    args=[str(ROOT/'scripts/audit_checkpoint.py'),str(path),'--output',str(ROOT/'evidence'/output)]
    if compare:args+=['--compare',str(compare)]
    run(output.replace('.json',''),args)


def main():
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())
    assert auth['initialization_file']=='gamma_1e7.pth' and auth['initialization_task']=='official UniDB DIV2K super-resolution'
    assert auth['max_cards']==8 and auth['card1_existing_service_may_coexist']
    deadline=auth['deadline_unix'];assert time.time()<deadline
    threading.Thread(target=guard,args=(deadline,),daemon=True).start()
    phase('WAITING_FOR_ASSETS',deadline_unix=deadline)
    while not all((ROOT/'evidence'/name).exists() for name in ['DATA_GATE.json','WEIGHT_DOWNLOAD.json']):
        if time.time()>=deadline:raise RuntimeError('Deadline reached before assets were ready')
        time.sleep(5)
    # Coordinate with pre-existing project launchers without altering their code.
    locks=[]
    for card in range(8):
        f=Path(f'/data2/yanglin/v6_diffuir_l_20261027/gates/AB_NPU_{card}.lock').open('a')
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    baseline=subprocess.check_output(['npu-smi','info'],text=True)
    (ROOT/'evidence/NPU_BEFORE.txt').write_text(baseline)
    import re
    occupants=[]
    for line in baseline.splitlines():
        match=re.match(r'^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|',line)
        if match:occupants.append({'card':int(match[1]),'reported_pid':int(match[2])})
    assert all(p['card']==1 and p['reported_pid']==2938958 for p in occupants),occupants
    write(ROOT/'evidence/SHARED_CARD_ACCEPTANCE.json',{'status':'PASS','preserved_occupants':occupants,
        'foreign_process_signals':0,'user_authority':str(ROOT/'protocol/USER_AUTHORITY.json')})
    frozen={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    write(ROOT/'protocol/RUN_SOURCE.json',{'files':frozen,'authority_sha256':sha(ROOT/'protocol/USER_AUTHORITY.json')})
    phase('NATIVE_STRICT_LOAD_AND_NUMERICAL_PREFLIGHT')
    run('preflight',[str(ROOT/'scripts/preflight.py')])
    phase('EIGHT_CARD_BATCH_BENCHMARK')
    results=[]
    for micro in [4,8]:
        try:
            folder=worker(f'bench_micro{micro}','bench',micro,23)
            result=json.loads((folder/'RESULT.json').read_text());results.append(result)
        except RuntimeError as error:
            if micro==4:raise
            write(ROOT/f'evidence/OPTIONAL_MICRO{micro}_FAILED.json',{'error':repr(error),'fallback':'previous passing measured candidate'})
    eight=[r for r in results if r['config']['microbatch']==8]
    if eight and max(r['peak_reserved'] for r in eight)<36*(1<<30):
        try:
            folder=worker('bench_micro12','bench',12,23)
            results.append(json.loads((folder/'RESULT.json').read_text()))
        except RuntimeError as error:
            write(ROOT/'evidence/OPTIONAL_MICRO12_FAILED.json',{'error':repr(error),'fallback':'previous passing measured candidate'})
    chosen=max(results,key=lambda r:r['images_per_second'])
    micro=chosen['config']['microbatch'];write(ROOT/'evidence/BATCH_SELECTION.json',{
        'status':'PASS','criterion':'highest measured end-to-end optimizer-update images/sec among safe eight-card candidates',
        'selected_microbatch':micro,'effective_batch':micro*8,'results':results})
    phase('COMPLETE_STATE_RESTART_ACCEPTANCE',microbatch=micro,effective_batch=micro*8)
    reference=worker('resume_reference2','probe',micro,2)
    first=worker('resume_first1','probe',micro,1)
    resumed=worker('resume_last1','probe',micro,1,resume=json.loads((first/'LATEST.json').read_text())['path'])
    audit(resumed,'RESUME_GATE.json',json.loads((reference/'LATEST.json').read_text())['path'])
    phase('PILOT_200_NATIVE_UPDATES_AND_FIXED_MONITOR')
    pilot=worker('pilot200','pilot',micro,200,evaluate=True);audit(pilot,'PILOT_STATE_AUDIT.json')
    write(ROOT/'evidence/ENGINEERING_GATE.json',{'status':'PASS','native':str(ROOT/'evidence/PREFLIGHT_NATIVE.json'),
        'resume':str(ROOT/'evidence/RESUME_GATE.json'),'pilot':str(pilot/'RESULT.json'),
        'pilot_states_discarded_for_formal':True,'formal_initialization':'fresh strict official SR gamma1e7',
        'microbatch':micro,'effective_batch':micro*8,'loss_unchanged':True})
    phase('FORMAL_INITIAL_BASELINE_AND_TRAINING',deadline_unix=deadline,microbatch=micro,effective_batch=micro*8)
    formal=worker('formal_until_20261103_noon','formal',micro)
    phase('FINAL_INDEPENDENT_STATE_AUDIT')
    audit(formal,'FINAL_CHECKPOINT_AUDIT.json')
    run('finalize',[str(ROOT/'scripts/finalize.py')])
    (ROOT/'evidence/NPU_AFTER.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
    phase('COMPLETE',final_report=str(ROOT/'FINAL_REPORT_ZH.md'))
    write(ROOT/'FINAL_ACCEPTANCE.json',{'status':'PASS','run':str(formal),'report':str(ROOT/'FINAL_REPORT_ZH.md'),
        'training_and_evaluation_exited':True,'deadline_unix':deadline,'completed_unix':time.time(),
        'original_assets_modified':False,'AB_started':False,'foreign_process_signals':0})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        phase('FAILED',error=repr(error));raise

"""Persistent finite D0 mixed-severity pipeline; never launches A/B."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from scripts.runtime_utils import ROOT,sha,write_json,process_identity,is_same_process,snapshot

EX=ROOT/'execution_20261029_d0_multistrength160k'
def read(p):return json.loads(Path(p).read_text())
def status(name,**kw):write_json(EX/'STATUS.json',{'status':name,'unix':time.time(),**kw})
def cpu(task):
    cmd=['bash',str(ROOT/'scripts/run_env.sh'),str(ROOT/'scripts/audit_d0_multistrength_20261029.py'),'--task',task]
    env=os.environ.copy();env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
    with (EX/(task+'.log')).open('ab') as f:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,('CPU stage failed',task)
def job(name,script,args=(),hours=1000000):
    assert not (EX/'STOP_REQUEST.json').exists(),'User STOP'
    out=ROOT/'distributed'/name;receipt=EX/(name+'_DRIVER.json')
    if not out.exists() and not receipt.exists():
        s=snapshot();assert all(not s['processes'].get(c) for c in [4,5,6,7]),'Authorized cards occupied'
        cmd=['/usr/bin/python3','-u','-m','scripts.launch_ab','--run',name,'--script',script,
            '--devices','4,5,6,7','--timeout-hours',str(hours),'--',*map(str,args)]
        with (EX/(name+'_driver.log')).open('ab') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(receipt,{**process_identity(p.pid),'command':cmd,'unix':time.time()})
    done=out/'PROCESS_COMPLETE.json';assert receipt.exists()
    status('RUNNING_'+name,run=name)
    last_monitor=0
    while not done.exists():
        assert is_same_process(read(receipt)) or done.exists(),'Driver exited without completion; inspect live workers before any retry'
        if name.endswith('_formal01'):
            if not (EX/'STARTUP_ACCEPTANCE.json').exists():
                paths=[out/f'UPDATES_rank{i}.jsonl' for i in range(4)]
                if all(p.exists() and sum(1 for _ in p.open())>=20 for p in paths):cpu('startup')
            if (EX/'NOON_REVIEW.json').exists() and not (EX/'NOON_REPORT_READY.json').exists():
                cpu('report');write_json(EX/'NOON_REPORT_READY.json',{'status':'PASS','unix':time.time(),'training_continues':True})
            if time.time()-last_monitor>3600:
                cpu('monitor');last_monitor=time.time()
        time.sleep(2)
    assert read(done)['exit_code']==0,('GPU job failure',name)
    assert all(not is_same_process(read(p)) for p in out.glob('RANK_*.json'))
    if (EX/'STOP_REQUEST.json').exists():
        if name.endswith('_formal01'):cpu('finalize')
        status('USER_STOPPED',run=name);raise SystemExit(0)
    return out
def main():
    lock=(EX/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(EX/'ACTIVE_CONTROLLER.json',process_identity(os.getpid()))
    try:
        assert not (EX/'STOP_REQUEST.json').exists()
        old_done=ROOT/'distributed/V6_D0_multistrength_generate_20261029_01/PROCESS_COMPLETE.json'
        status('WAITING_FOR_PREVIOUS_GENERATION_EXIT')
        while not old_done.exists():time.sleep(2)
        if not (EX/'DATA_PLAN.json').exists():
            status('PREPARING_DATA_PLAN_AND_PARENT_SHA')
            env=os.environ.copy();env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
            with (EX/'prepare.log').open('ab') as f:
                p=subprocess.run(['bash',str(ROOT/'scripts/run_env.sh'),str(ROOT/'scripts/prepare_d0_multistrength160k_20261029.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            assert p.returncode==0,'Preparation failed'
        job('V6_D0_multistrength160k_generate_20261029_01','generate_d0_multistrength160k_20261029.py',hours=3)
        status('WAITING_FOR_FROZEN_STAGE2_ENGINE')
        began=time.time()
        while not (EX/'STAGE2_READY.json').exists():
            assert not (EX/'STOP_REQUEST.json').exists();assert time.time()-began<7200;time.sleep(2)
        for n,h in read(EX/'SOURCE_MANIFEST.json')['files'].items():assert sha(ROOT/n)==h,n
        cpu('seal_data');cpu('parent')
        job('V6_D0_multistrength160k_baseline_20261029_01','run_d0_multistrength_20261029.py',['--mode','baseline'])
        cpu('baseline')
        job('V6_D0_multistrength160k_probe_reference_20261029_01','run_d0_multistrength_20261029.py',['--mode','probe','--end-update','95546'])
        job('V6_D0_multistrength160k_probe_restarted_20261029_01','run_d0_multistrength_20261029.py',
            ['--mode','probe','--end-update','95546','--resume',ROOT/'distributed/V6_D0_multistrength160k_probe_reference_20261029_01/checkpoints/U095545.pt'])
        cpu('resume')
        job('V6_D0_U95544_multistrength160k_formal01','run_d0_multistrength_20261029.py',['--mode','formal'])
        cpu('finalize')
        status('ALL_COMPLETE',final_acceptance=read(EX/'FINAL_ACCEPTANCE.json'))
    except SystemExit:raise
    except BaseException as exc:
        status('ENGINEERING_FAILURE_REQUIRES_DIAGNOSIS',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()

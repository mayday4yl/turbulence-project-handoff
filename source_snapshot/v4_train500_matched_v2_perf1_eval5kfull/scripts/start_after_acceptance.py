"""One-shot authorized launcher: all selected gates, immutable freeze, start."""
import json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from v4.runtime import save_json


def main():
    required=['EXECUTION_REVISION_ACCEPTANCE.json','RESOURCE_CPU_ACCEPTANCE.json','REPORT_REVISION_ACCEPTANCE.json']
    start=time.monotonic()
    while not all((ROOT/'reports'/name).exists() for name in required):
        if (ROOT/'reports/RUN_PAUSED.json').exists() or (ROOT/'reports/SELECTED_EXECUTION_GATE_FAILURE.json').exists():
            raise RuntimeError('Acceptance failed; no automatic bypass')
        if time.monotonic()-start>1800:raise TimeoutError('Selected acceptance incomplete; no launch')
        time.sleep(2)
    for name in required:assert json.loads((ROOT/'reports'/name).read_text())['status']=='PASS'
    # Wait for acceptance children to exit; their context must not overlap training.
    uuids=json.loads((ROOT/'protocol/FROZEN_PARENT.json').read_text())['gpu_uuids']
    until=time.monotonic()+60
    while True:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if not any(u in apps for u in uuids.values()):break
        if time.monotonic()>until:raise RuntimeError('Reserved GPU still occupied')
        time.sleep(1)
    assert not (ROOT/'protocol/FROZEN.json').exists(),'Already frozen: inspect, never duplicate launch'
    subprocess.run([sys.executable,str(ROOT/'scripts/freeze_execution_revision.py')],cwd=ROOT,check=True)
    session='v4_train500_perf1_formal'
    assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
    cmd='cd '+shlex.quote(str(ROOT))+' && '+shlex.quote(sys.executable)+' -u scripts/run_formal_performance.py > reports/formal_controller.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
    save_json(ROOT/'reports/FORMAL_DISPATCH.json',{'status':'DISPATCHED_AFTER_SELECTED_GATES_PASS','tmux':session,
        'time':time.time(),'note':'Dispatch is not proof of completed updates; inspect run PIDS and train_steps'})
    print('Formal coordinator dispatched after selected gates PASS',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save_json(ROOT/'reports/LAUNCH_BLOCKED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise

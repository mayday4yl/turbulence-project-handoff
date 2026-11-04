"""One authorized transition from serial evaluation to disjoint four-card work."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from sharded_eval_support import ROOT, BASE, CANON, verify, frozen
from guard_distributed import process_info, snapshot, check_snapshot

EXPECTED_CONTROLLER=(1965131,190531678)


def alive(identity):
    try:
        actual=process_info(identity['pid'])
        return actual['state']!='Z' and actual['starttime_ticks']==identity['starttime_ticks']
    except FileNotFoundError:
        return False


def wait_exit(identity,seconds):
    end=time.monotonic()+seconds
    while alive(identity):
        if time.monotonic()>end:
            raise RuntimeError('Owned process did not exit after controlled handoff')
        time.sleep(.2)


def main():
    verify()
    w=frozen.read(frozen.STATUS)
    if w['formal_global_update']!=1000 or not w['training_complete_to_1000'] or w.get('active_training'):
        raise RuntimeError('Evaluation-only handoff refuses active training')
    if (w['controller_pid'],w['controller_starttime_ticks'])!=EXPECTED_CONTROLLER:
        raise RuntimeError('Prior controller identity changed')
    directory=Path(w['active_guard_directory'])
    if directory!=ROOT/'runs/stage3_parallel_a01':
        raise RuntimeError('Prior guard directory differs')
    launch=frozen.read(directory/'launch.json')
    controller=process_info(w['controller_pid']);guard=process_info(launch['guard_pid'])
    if not alive(controller) or not alive(guard) or guard['ppid']!=controller['pid']:
        raise RuntimeError('Prior guard/controller ancestry differs')
    if controller['starttime_ticks']!=EXPECTED_CONTROLLER[1]:
        raise RuntimeError('Controller PID was reused')
    archive=BASE/'prior_serial_handoff'
    archive.mkdir(parents=True,exist_ok=False)
    frozen.write(archive/'workflow_before.json',w)
    frozen.write(archive/'identities.json',{'controller':controller,'guard':guard})
    shutil.copy2(ROOT/'artifacts/PERSISTENT_HANDOFF_STATUS.json',archive/'handoff_before.json')
    # Stop the scheduling parent first. The original guard remains solely
    # responsible for cleaning its proven worker/compiler descendant tree.
    if not alive(controller):
        raise RuntimeError('Controller exited before its identity-checked signal')
    os.kill(controller['pid'],signal.SIGTERM)
    wait_exit(controller,10)
    if not alive(guard):
        raise RuntimeError('Guard exited unexpectedly before controlled cleanup')
    os.kill(guard['pid'],signal.SIGTERM)
    wait_exit(guard,50)
    frozen.require_released(directory)
    shutil.copy2(directory/'guard_result.json',archive/'guard_result.json')
    preserved={}
    for weights in ['raw','ema']:
        folder=CANON/weights
        for name in ['progress.json','summary.json']:
            if (folder/name).exists():
                shutil.copy2(folder/name,archive/(weights+'_'+name))
        for path in folder.glob('*/rows/*.json'):
            preserved[str(path.relative_to(ROOT))]=frozen.digest(path)
    frozen.write(archive/'preserved_rows_sha256.json',preserved)
    for name in ['.auto_to_1000.lock','.guard_npu_0.lock']:
        with (ROOT/name).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw,snap=snapshot();check_snapshot(snap)
    (archive/'pre_sharded_npu_smi.txt').write_text(raw)
    verify()
    with (BASE/'controller.log').open('x') as stream:
        p=subprocess.Popen(['python3','-u','code/run_sharded_evaluation.py','--mode','controller'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    frozen.write(archive/'HANDOFF.json',{'status':'replacement_launched','utc':frozen.utc(),
        'new_controller':process_info(p.pid),'old_controller':controller,'old_guard':guard,
        'preserved_committed_rows':len(preserved),'old_owned_resources_released':True,
        'old_guard_failure_reason':'explicit user-requested serial-to-four-card evaluation handoff',
        'training_updates_added':0,'physical_cards':[0,1,2,3]})
    print('Four-card evaluation controller launched; existing committed rows retained.',flush=True)


if __name__=='__main__':
    main()

"""Persistent finite two-branch executor. No A/B, no ratio search."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from scripts.runtime_utils import ROOT, write_json, process_identity, is_same_process

EX = ROOT / 'execution_20261030_d0_g4_reweight_5k'
PREFIX = 'V6_D0_G4_reweight_20261030'


def state(name, **kw):
    write_json(EX / 'STATUS.json', dict(status=name, unix=time.time(), A_B_started=False, **kw))


def cpu(task):
    env = os.environ.copy()
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    env['V6_DEVICES'] = '2,3'
    cmd = ['bash', str(ROOT / 'scripts/run_env.sh'), str(ROOT / 'scripts/audit_d0_g4_reweight_v3_20261030.py'), '--task', task]
    with (EX / (task + '.log')).open('ab') as log:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, ('Independent CPU gate failed', task)


def launch(branch, phase, *, resume=None):
    cfg = json.loads((EX / 'CONFIG.json').read_text())
    name = f'{PREFIX}_{branch}_{phase}01'
    out = ROOT / 'distributed' / name
    receipt = EX / (name + '_DRIVER.json')
    assert not out.exists() and not receipt.exists(), 'Inspect previous live job; never duplicate launch'
    assert not (EX / 'STOP_REQUEST.json').exists()
    card = cfg['physical_devices_by_branch'][branch][0]
    cmd = ['/usr/bin/python3', '-u', '-m', 'scripts.launch_d0_g4_reweight_20261030',
           '--run', name, '--branch', branch, '--card', str(card), '--mode', 'formal' if phase == 'formal' else 'probe']
    if phase != 'formal':
        cmd += ['--end-local', '2']
    if resume is not None:
        cmd += ['--resume', str(resume)]
    env = os.environ.copy()
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    with (EX / (name + '_driver.log')).open('xb') as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(p.pid)
    write_json(receipt, dict(identity, command=cmd, branch=branch, phase=phase))
    return dict(path=out, receipt=receipt, identity=identity)


def wait(jobs):
    while True:
        done = True
        for job in jobs:
            result = job['path'] / 'PROCESS_COMPLETE.json'
            if result.exists():
                value = json.loads(result.read_text())
                assert value['exit_code'] == 0 and value['supervisor_stop'] is None, (str(result), value)
            else:
                assert is_same_process(job['identity']), ('Driver exited; preserve and inspect workers', str(job['path']))
                done = False
        if done:
            return
        time.sleep(2)


def main():
    lock = (EX / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(EX / 'ACTIVE_CONTROLLER.json', process_identity(os.getpid()))
    try:
        import hashlib
        for manifest in ['repair_20261030_resume_numerics/SOURCE_MANIFEST.json','repair_20261030_startup_identity/SOURCE_MANIFEST.json']:
            for name, expected in json.loads((EX / manifest).read_text())['files'].items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        assert not is_same_process(json.loads((EX / 'PIPELINE_LAUNCH02.json').read_text())), 'Previous CPU controller remains active'
        assert json.loads((EX / 'SMOKE_RESUME_GATE.json').read_text())['status'] == 'PASS'
        jobs=[]
        for branch in ['Control','G4_biased']:
            name=f'{PREFIX}_{branch}_formal01';out=ROOT/'distributed'/name;receipt=EX/(name+'_DRIVER.json')
            driver=json.loads(receipt.read_text());rank=json.loads((out/'RANK_0.json').read_text())
            assert is_same_process(driver) and is_same_process(rank), 'Inspect actual existing actor before adoption'
            jobs.append(dict(path=out,receipt=receipt,identity=driver))
        state('CPU_ADOPTION_OF_EXISTING_HEALTHY_FORMAL_ACTORS_NO_RESTART')
        startup_done = False
        extension_done = False
        while True:
            all_done = True
            for job in jobs:
                complete = job['path'] / 'PROCESS_COMPLETE.json'
                if complete.exists():
                    r = json.loads(complete.read_text())
                    assert r['exit_code'] == 0 and r['supervisor_stop'] is None, (str(complete), r)
                else:
                    assert is_same_process(job['identity']), ('Formal driver exited without receipt', str(job['path']))
                    all_done = False
            if not startup_done:
                logs = [job['path'] / 'UPDATES.jsonl' for job in jobs]
                if all(p.exists() and sum(1 for _ in p.open()) >= 20 for p in logs):
                    cpu('startup')
                    startup_done = True
            if not extension_done:
                vals = [job['path'] / 'validation/u005000_five_g_1000/SUMMARY.json' for job in jobs]
                if all(p.exists() for p in vals):
                    cpu('extension')
                    extension_done = True
            progress = {}
            for job in jobs:
                p = job['path'] / 'STATUS.json'
                if p.exists():
                    progress[job['path'].name] = json.loads(p.read_text())
            state('FORMAL_RUNNING_OR_AWAITING_FIXED_VALIDATION', branches=progress,
                  startup_acceptance_completed=startup_done, extension_decision_completed=extension_done)
            if all_done:
                break
            time.sleep(5)
        assert startup_done and extension_done
        for branch, job in zip(['Control', 'G4_biased'], jobs):
            result = json.loads((job['path'] / 'RESULT.json').read_text())
            if branch == 'Control':
                assert result['completed_branch_updates'] == 5000
            else:
                decision = json.loads((EX / 'EXTENSION_DECISION.json').read_text())
                assert result['completed_branch_updates'] == (10000 if decision['extend_G4_biased'] else 5000)
        state('CPU_FINAL_REPORT_AND_PAIRED_AUDIT')
        env = os.environ.copy()
        env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        with (EX / 'finalize.log').open('ab') as log:
            r = subprocess.run(['bash', str(ROOT / 'scripts/run_env.sh'),
                str(ROOT / 'scripts/finalize_d0_g4_reweight_20261030.py')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert r.returncode == 0
        state('ALL_FINITE_BRANCH_BUDGETS_COMPLETE_REPORT_READY', final_acceptance='FINAL_ACCEPTANCE.json')
    except BaseException as exc:
        state('ENGINEERING_FAILURE_PRESERVED_INSPECT_LIVE_JOBS_BEFORE_RETRY', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

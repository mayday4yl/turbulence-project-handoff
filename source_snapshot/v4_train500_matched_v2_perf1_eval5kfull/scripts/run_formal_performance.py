"""Single-GPU jobs with boundary relocation and complete common-stage delivery."""
import fcntl,json,os,signal,subprocess,sys,time,traceback,uuid,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from v4.runtime import save_json,sha256
from v4.stages import GROUPS,NODES,read_control,late_stop,verify_budget_scope
from v4.scheduling import relocation_candidate
from train_worker import verify_frozen
from stage_report import report
from v4.eta import eta_report


def main():
    frozen=verify_frozen();run=Path(frozen['results_root']);active={};pending={}
    selected=json.loads((ROOT/'protocol/RESOURCE_SCHEDULE.json').read_text())['selected']
    handle=(run/'COORDINATOR.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Explicit pause')
    attempt='perf1_'+uuid.uuid4().hex;coordbase=run/'coordination'/attempt;coordbase.mkdir(parents=True)
    def event(kind,**values):
        with (run/'execution_events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':kind,'execution_hash':frozen['execution_hash'],**values})+'\n')
    def pids():
        save_json(run/'PIDS.json',{'attempt':attempt,'coordinator':os.getpid(),
            'workers':{g:v['p'].pid for g,v in active.items()},'gpu_mapping':{g:v['gpu'] for g,v in active.items()},
            'coordination':str(coordbase),'execution_hash':frozen['execution_hash']})
    def launch(g,gpu,members,coord):
        log=(run/g/f'worker_{attempt}_{uuid.uuid4().hex[:8]}.log').open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_HOME=str(ROOT/'assets/torch'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/train_worker.py'),'--group',g,'--gpu',str(gpu),
           '--members',','.join(members),'--coordination',str(coord)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[g]={'p':p,'gpu':gpu,'log':log};event('worker_launch',group=g,gpu=gpu,pid=p.pid);pids()
    def guard():
        if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Numerical/correctness/explicit pause')
        owned={v['p'].pid for v in active.values()}
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        for line in lines:
            pid,gpu=[v.strip() for v in line.split(',')]
            if gpu in frozen['gpu_uuids'].values() and int(pid) not in owned:raise RuntimeError('Foreign process on reserved GPU')
    def auxiliary(node):
        # All four training workers exited at the stage boundary: one evaluator per GPU.
        assert not active
        for g,gpu in [('R08',5),('R0',6)]:
            out=run/'refinement_ablation'/f'U{node:05d}'/g
            if (out/'COMMIT.json').exists():continue
            out.mkdir(parents=True,exist_ok=True);log=(out/f'worker_{attempt}.log').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/refinement_ablation.py'),'--group',g,'--update',str(node)],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[g]={'p':p,'gpu':gpu,'log':log};pids()
        while active:
            guard()
            for g,v in list(active.items()):
                code=v['p'].poll()
                if code is not None:
                    if code:raise RuntimeError('Auxiliary inference failure')
                    v['log'].close();del active[g];pids()
            time.sleep(2)
    def endpoint_backfill(node):
        assert node in (5000,7500) and not active
        for pair in [[('D0',5),('R0',6)],[('R08',5),('L08',6)]]:
            for g,gpu in pair:
                log=(run/g/f'endpoint_backfill_U{node}_{attempt}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate_endpoint_train.py'),'--group',g,'--update',str(node)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[g]={'p':child,'gpu':gpu,'log':log};pids()
            while active:
                guard()
                for g,v in list(active.items()):
                    code=v['p'].poll()
                    if code is not None:
                        if code:raise RuntimeError('Endpoint backfill failed')
                        v['log'].close();del active[g];pids()
                time.sleep(2)
    def endpoint_supplement(node):
        # Preserve previously released stage bytes and disclose any later work.
        out=run/'stage_deliveries'/f'U{node:05d}_user_endpoint_fulltrain'
        if (out/'MANIFEST.json').exists():return
        out.mkdir(parents=True,exist_ok=True)
        for g in GROUPS:
            source=run/g/'evaluations'/f'U{node:05d}'/'raw/train8000'
            commit=json.loads((source/'COMMIT.json').read_text())
            for name,digest in commit['file_sha256'].items():assert sha256(source/name)==digest
            destination=out/g
            if not destination.exists():shutil.copytree(source,destination)
        save_json(out/'MANIFEST.json',{'status':'USER_DECIDED_COMMON_BUDGET_TRUNCATION','comparison_update':node,
            'original_stage_release':str(run/'stages'/f'U{node:05d}'/'RELEASE.json'),
            'actual_training_counts':{g:json.loads((run/g/'STATUS.json').read_text()) for g in GROUPS},
            'excess_training_preserved':True,'sufficiency_trigger':'NOT_APPLICABLE_REQUIRES_FULL_10K_CURVE',
            'file_sha256':{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}})
    try:
        guard()
        for node in NODES:
            stage=run/'stages'/f'U{node:05d}'
            if (stage/'RELEASE.json').exists():
                release=json.loads((stage/'RELEASE.json').read_text())
                if release['action']=='stop':return
                continue
            need=[g for g in GROUPS if not (stage/f'{g}.READY.json').exists()]
            queues={5:[],6:[]}
            if selected['compute_scheduling']=='solo_queue':
                queues={gpu:[g for g in selected['queues'][str(gpu)] if g in need] for gpu in (5,6)}
                for gpu in (5,6):
                    if queues[gpu]:
                        group=queues[gpu].pop(0);launch(group,gpu,[group],coordbase/f'U{node:05d}_{group}')
            else:
                for gpu,members in [(5,[g for g in ['R08','D0'] if g in need]),(6,[g for g in ['R0','L08'] if g in need])]:
                    for g in members:launch(g,gpu,members,coordbase/f'U{node:05d}')
            last_eta=last_guard=0.
            while active:
                if time.monotonic()-last_guard>5:guard();last_guard=time.monotonic()
                for g,v in list(active.items()):
                    code=v['p'].poll()
                    if code is None:continue
                    if code:raise RuntimeError(f'{g} worker failed')
                    status=json.loads((run/g/'STATUS.json').read_text());v['log'].close();del active[g];pids()
                    event('worker_boundary_exit',group=g,gpu=v['gpu'],status=status)
                    if status['status']=='ENGINEERING_RELOCATION_BOUNDARY':
                        dest=pending.pop(g)
                        if not late_stop(read_control(ROOT/'protocol/CONTROL.json'),run/'stages'):
                            launch(g,dest,[g],coordbase/f'relocated_{g}_{uuid.uuid4().hex[:8]}')
                    elif status['status']=='STAGE_READY_RELEASE_GPU' and queues[v['gpu']]:
                        next_group=queues[v['gpu']].pop(0)
                        if not late_stop(read_control(ROOT/'protocol/CONTROL.json'),run/'stages'):
                            launch(next_group,v['gpu'],[next_group],coordbase/f'U{node:05d}_{next_group}')
                    elif status['status'] not in ['STAGE_READY_RELEASE_GPU','USER_STOPPED_AT_BOUNDARY']:
                        raise RuntimeError('Unexpected worker exit '+str(status))
                c=read_control(ROOT/'protocol/CONTROL.json')
                if selected['dynamic_relocation'] and not pending and not late_stop(c,run/'stages'):
                    updates={g:json.loads((run/g/'STATUS.json').read_text()).get('update',0) if (run/g/'STATUS.json').exists() else 0 for g in active}
                    candidate=relocation_candidate({g:v['gpu'] for g,v in active.items()},updates)
                    if candidate:
                        g,dest=candidate
                        ready_file=run/g/'EXECUTION_READY.json'
                        if not ready_file.exists() or json.loads(ready_file.read_text())['pid']!=active[g]['p'].pid:
                            time.sleep(1);continue
                        pending[g]=dest
                        event('relocation_requested',group=g,source_gpu=active[g]['gpu'],destination_gpu=dest,update=updates[g])
                        os.kill(active[g]['p'].pid,signal.SIGUSR1)
                if time.monotonic()-last_eta>60:eta_report(run);last_eta=time.monotonic()
                time.sleep(1)
            ready=[stage/f'{g}.READY.json' for g in GROUPS]
            if not all(p.exists() for p in ready):
                if late_stop(read_control(ROOT/'protocol/CONTROL.json'),run/'stages'):
                    target=read_control(ROOT/'protocol/CONTROL.json')['stop_at_common_update']
                    if target in (5000,7500) and (run/'stages'/f'U{target:05d}'/'RELEASE.json').exists():
                        endpoint_backfill(target);endpoint_supplement(target)
                    save_json(run/'STATUS.json',{'status':'USER_STOPPED_PRESERVE_ACTUAL_COUNTS','node':node,'actual':{g:json.loads((run/g/'STATUS.json').read_text()) for g in GROUPS}});return
                raise RuntimeError('All processes exited without complete common stage')
            for p in ready:
                row=json.loads(p.read_text());assert row['update']==node
                assert sha256(row['checkpoint_path'])==row['checkpoint_sha256']
                assert sha256(row['evaluation_commit'])==row['evaluation_commit_sha256']
            verify_budget_scope(ROOT)
            save_json(stage/'REPORT_STATUS.json',{'status':'AUXILIARY_REFINEMENT_INFERENCE','time':time.time()})
            auxiliary(node)
            control_before_delivery=read_control(ROOT/'protocol/CONTROL.json')
            if node<10000 and (control_before_delivery['stop_at_common_update']==node or control_before_delivery['stop_at_next_boundary']):
                endpoint_backfill(node)
            save_json(stage/'REPORT_STATUS.json',{'status':'BUILDING_COMPLETE_DELIVERY','time':time.time()})
            manifest=report(run,node);c=read_control(ROOT/'protocol/CONTROL.json')
            action='stop' if node==10000 or c['stop_at_common_update']==node or c['stop_at_next_boundary'] else 'continue'
            # A user stop may arrive while the report is being rendered.
            if action=='stop' and node<10000:
                if any(not (run/g/'evaluations'/f'U{node:05d}'/'raw/train8000/COMMIT.json').exists() for g in GROUPS):
                    endpoint_backfill(node);endpoint_supplement(node)
            release={'update':node,'all_four_complete':True,'delivery_manifest':str(manifest),'delivery_manifest_sha256':sha256(manifest),
                     'action':action,'time':time.time(),'execution_hash':frozen['execution_hash'],
                     'reason':'USER_DECIDED_COMMON_BUDGET_TRUNCATION' if action=='stop' and node<10000 else 'DEFAULT_COMMON_PROTOCOL'}
            save_json(stage/'RELEASE.json',release);save_json(stage/'REPORT_STATUS.json',{'status':'COMPLETE','action':action,'time':time.time()})
            save_json(run/'LATEST_MILESTONE.json',release);event('COMMON_STAGE_DELIVERED',**release)
            if action=='stop':
                save_json(run/'STATUS.json',{'status':'COMPLETE_U10000_ALL_DELIVERIES' if node==10000 else 'USER_DECIDED_COMMON_BUDGET_TRUNCATION',
                                           'update':node,'no_auto_extension':True,'refinement_auxiliary_complete':True});return
    except BaseException as e:
        if not (ROOT/'reports/RUN_PAUSED.json').exists():save_json(ROOT/'reports/RUN_PAUSED.json',{'gate':'performance_formal_coordinator','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()})
        raise
    finally:
        for v in active.values():
            if v['p'].poll() is None:v['p'].terminate()
        end=time.monotonic()+120
        while any(v['p'].poll() is None for v in active.values()) and time.monotonic()<end:time.sleep(.5)
        for v in active.values():
            if v['p'].poll() is None:v['p'].kill();v['p'].wait()
            v['log'].close()
        pids();fcntl.flock(handle,fcntl.LOCK_UN);handle.close()

if __name__=='__main__':main()

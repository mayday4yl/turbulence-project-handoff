"""Two independent GPU pairs, common stage barriers, no automatic budget extension."""
import fcntl,json,os,signal,subprocess,sys,time,traceback,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from v4.runtime import save_json,sha256
from v4.stages import GROUPS,NODES,read_control
from train_worker import verify_frozen
from stage_report import report,rules
from v4.eta import eta_report

def main():
    frozen=verify_frozen();run=Path(frozen['results_root']);run.mkdir(parents=True,exist_ok=True)
    lock=(run/'COORDINATOR.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for g in GROUPS:(run/g).mkdir(exist_ok=True)
    active={};attempt=uuid.uuid4().hex;cards=list(frozen['mapping']);mapping={g:int(card) for card,gs in frozen['mapping'].items() for g in gs}
    def pids():save_json(run/'PIDS.json',{'coordinator':os.getpid(),'workers':{g:v['p'].pid for g,v in active.items()},'gpu_mapping':{g:v['gpu'] for g,v in active.items()},'attempt':attempt,'execution_hash':frozen['execution_hash']})
    def guard():
        if (ROOT/'reports/RUN_PAUSED.json').exists():raise RuntimeError('Global pause requested')
        control=read_control(ROOT/'protocol/CONTROL.json')
        if control['stop_at_next_boundary']:
            for v in active.values():
                if v['p'].poll() is None:v['p'].terminate()
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        free={int(l.split(',')[0]):int(l.split(',')[1]) for l in lines}
        if any(free[int(c)]<frozen['minimum_free_MiB'] for c in cards):raise RuntimeError('GPU memory headroom below frozen safety floor')
        if frozen['resource_mode']=='exclusive':
            owned={v['p'].pid for v in active.values()}
            for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
                pid,gpu=[x.strip() for x in l.split(',')]
                if gpu in frozen['gpu_uuids'].values() and int(pid) not in owned:raise RuntimeError('Foreign process appeared on selected exclusive GPU')
    def launch(group,gpu,script,extra):
        log=(run/group/f'{script}_{attempt}_{time.time_ns()}.log').open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=frozen['gpu_uuids'][str(gpu)],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCH_HOME=str(ROOT/'assets/torch'))
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script),'--group',group,*extra],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        active[group]={'p':p,'gpu':gpu,'log':log};pids()
    def wait(training=False):
        last_eta=0
        while active:
            guard()
            for g,v in list(active.items()):
                code=v['p'].poll()
                if code is None:continue
                v['log'].close();del active[g];pids()
                if code:raise RuntimeError(f'{g} child exited {code}')
                if training:
                    status=json.loads((run/g/'STATUS.json').read_text())['status']
                    if status not in ('STAGE_READY_RELEASE_GPU','USER_STOPPED_AT_BOUNDARY'):raise RuntimeError('Unexpected training worker exit '+status)
            if training and time.monotonic()-last_eta>60:eta_report(run,frozen);last_eta=time.monotonic()
            time.sleep(1)
    try:
        guard();pids()
        for node in NODES:
            stage=run/'stages'/f'U{node:05d}'
            if (stage/'RELEASE.json').exists():
                release=json.loads((stage/'RELEASE.json').read_text())
                assert sha256(release['delivery_manifest'])==release['delivery_manifest_sha256']
                if release['action']=='stop':return
                continue
            if read_control(ROOT/'protocol/CONTROL.json')['stop_at_next_boundary']:return
            for card,cohort in frozen['mapping'].items():
                members=[g for g in cohort if not (stage/f'{g}.READY.json').exists()]
                for group in members:launch(group,int(card),'train_worker.py',['--gpu',card,'--members',','.join(members),'--coordination',str(run/'coordination'/attempt/f'U{node:05d}')])
            wait(training=True)
            if read_control(ROOT/'protocol/CONTROL.json')['stop_at_next_boundary']:
                save_json(run/'STATUS.json',{'status':'USER_STOPPED_PRESERVE_ACTUAL_COUNTS'});return
            for g in GROUPS:
                ready=json.loads((stage/f'{g}.READY.json').read_text());assert ready['update']==node
                assert sha256(ready['checkpoint_path'])==ready['checkpoint_sha256']
                assert sha256(ready['evaluation_commit'])==ready['evaluation_commit_sha256']
            save_json(stage/'REPORT_STATUS.json',{'status':'REFINEMENT_ON_INFERENCE_ONLY','update':node})
            for g,card in [('R08',cards[0]),('R0',cards[1])]:launch(g,int(card),'refinement_ablation.py',['--update',str(node)])
            wait()
            if node==2000:
                choices={}
                for g in GROUPS:
                    scores={u:json.loads((run/g/'evaluations'/f'U{u:05d}'/'raw/val800/aggregate.json').read_text())['overall']['psnr'] for u in range(0,2001,250)}
                    choices[g]=rules.best_update(scores,list(scores),1e-6)
                save_json(run/'FINAL_BEST_SELECTION.json',{'criterion':'RAW_OFF_val800_mean_PSNR','tie_tolerance_dB':1e-6,'updates':choices,'refinement_excluded':True})
                for pair in [('D0','R0'),('R08','L08')]:
                    for g,card in zip(pair,cards):
                        if choices[g] not in NODES:launch(g,int(card),'evaluate_endpoint_train.py',['--update',str(choices[g])])
                    wait()
            save_json(stage/'REPORT_STATUS.json',{'status':'BUILDING_COMPLETE_CPU_DELIVERY','update':node})
            manifest=report(run,node)
            control=read_control(ROOT/'protocol/CONTROL.json');action='stop' if node==2000 or control['stop_at_next_boundary'] or control['stop_at_common_update']==node else 'continue'
            release={'update':node,'all_four_complete':True,'delivery_manifest':str(manifest),'delivery_manifest_sha256':sha256(manifest),'action':action,'time':time.time(),'execution_hash':frozen['execution_hash'],'reason':'PRESET_BUDGET_ENDPOINT' if node==2000 else 'COMMON_STAGE_DELIVERED'}
            save_json(stage/'RELEASE.json',release);save_json(stage/'REPORT_STATUS.json',{'status':'COMPLETE','action':action});save_json(run/'LATEST_MILESTONE.json',release)
            if action=='stop':
                save_json(run/'STATUS.json',{'status':'COMPLETE_U02000_ALL_DELIVERIES' if node==2000 else 'USER_STOPPED_AFTER_COMMON_STAGE','update':node,'no_automatic_extension':True});return
    except BaseException as e:
        save_json(ROOT/'reports/RUN_PAUSED.json',{'gate':'formal_coordinator','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise
    finally:
        for v in active.values():
            if v['p'].poll() is None:v['p'].terminate()
        # Workers obey complete optimizer boundaries; preserve PID evidence rather than killing mid-update.
        pids()
if __name__=='__main__':main()

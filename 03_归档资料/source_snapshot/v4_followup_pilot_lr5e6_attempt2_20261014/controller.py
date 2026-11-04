"""Durable bounded controller; never starts old experiment controllers."""
from lib import *
import traceback, signal
from timebox_policy import make_timebox

STATE=ROOT/'CONTROLLER_STATE.json'
def exact_alive(q):
    try:
        pid=q['pid'];p=Path('/proc',str(pid))
        return ticks(pid)==q['start_ticks'] and p.stat().st_uid==os.getuid() and p.joinpath('cwd').resolve()==ROOT and str(ROOT/'worker.py') in p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (KeyError,OSError):return False
def launch(mode,method,gpu,update=None):
    resource_check(gpu);storage_check(1_950_000_000)
    cmd=[sys.executable,str(ROOT/'worker.py'),'--mode',mode,'--method',method,'--gpu',str(gpu)]
    if update is not None:cmd+=['--update',str(update)]
    log=ROOT/'logs'/f'{mode}_{method}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
    with log.open('ab') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    q={'pid':p.pid,'start_ticks':ticks(p.pid),'mode':mode,'method':method,'gpu':gpu,'update':update,'log':str(log),'started_server_time':time.time()}
    append(ROOT/'launches.jsonl',q);return q
def stage_file(mode,method):
    if mode=='gate':return ROOT/'gates'/method/'PASS.json'
    if mode=='train':return ROOT/'runs'/method/'TRAIN_FINISHED.json'
    return ROOT/'evaluation'/f'U{read(STATE)["matched_update"]:05d}'/method/'COMPLETE.json'
def finish_report(state):
    import numpy as np,csv
    u=state.get('matched_update');report=['# 小数据 D0 与连续物理差分验证','',f'匹配训练节点：{u if u else "尚无共同里程碑"}。共同初始化；Train1000/G4；各目标U1000；主评价Val200 sources/400 inputs，FP32 online。','']
    summary={'status':'INCOMPLETE','matched_update':u,'methods':{},'training':{g:read(ROOT/'runs'/g/'TRAIN_FINISHED.json') if (ROOT/'runs'/g/'TRAIN_FINISHED.json').exists() else None for g in GROUPS}}
    if u and all((ROOT/'evaluation'/f'U{u:05d}'/g/'COMPLETE.json').exists() for g in GROUPS):
        rr={g:[x for p in sorted((ROOT/'evaluation'/f'U{u:05d}'/g/'parts').glob('*.json')) for x in read(p)] for g in GROUPS}
        assert [x['start_id'] for x in rr['D0']]==[x['start_id'] for x in rr['CONT_DELTA_R0']]
        summary['methods']={g:read(ROOT/'evaluation'/f'U{u:05d}'/g/'SUMMARY.json') for g in GROUPS};summary['status']='PASS_MATCHED_PILOT_EVALUATION'
        report+=['| 方法／输出 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---:|---:|---:|']
        for g in GROUPS:
            for j,m in enumerate(summary['methods'][g]['step_means']):report.append(f'| {g} C{j+1} | {m["psnr"]:.6f} | {m["ssim"]:.6f} | {m["lpips"]:.6f} |')
        bysource={}
        for d,r in zip(rr['D0'],rr['CONT_DELTA_R0']):
            bysource.setdefault(d['source_id'],[]).append([r['steps'][-1][k]-d['steps'][-1][k] for k in ['psnr','ssim','lpips']])
        delta=np.array([np.mean(x,axis=0) for x in bysource.values()]);assert len(delta)==200
        rng=np.random.default_rng(20261013);boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean(0) for _ in range(3000)])
        summary['paired_vs_D0']={k:{'mean':float(delta[:,i].mean()),'source_bootstrap95CI':[float(x) for x in np.quantile(boot[:,i],[.025,.975])],'source_win_fraction':float((delta[:,i]*(1 if k!='lpips' else -1)>0).mean())} for i,k in enumerate(['psnr','ssim','lpips'])}
        report+=['','配对差异（连续法最终输出 − D0）：',json.dumps(summary['paired_vs_D0'],ensure_ascii=False,indent=2),'','这是单个训练seed、有限预算的小数据验证。置信区间反映验证source人口，不代表训练seed稳定性。','相同样本曝光不代表相同计算量；连续法含额外网络和物理渲染。']
        atomic(ROOT/'COMPLETE.json',{'status':'PASS_MATCHED_PILOT_EVALUATION','matched_update':u,'both_U1000':all(x and x['update']==1000 for x in summary['training'].values()),'time':time.time()})
    else:report+=['尚未完成配对评价；缺项保持未完成，不用不匹配节点替代。']
    atomic(ROOT/'SUMMARY.json',summary);(ROOT/'PILOT_REPORT.md').write_text('\n'.join(report)+'\n')
    if any(exact_alive(q) for q in state.get('jobs',{}).values()):
        atomic(ROOT/'FINALIZATION_PENDING_WORKERS.json',{'stage':state['stage'],'time':time.time()})
    else:
        atomic(ROOT/'CPU_FINALIZED.json',{'status':summary['status'],'summary_sha256':sha(ROOT/'SUMMARY.json'),'report_sha256':sha(ROOT/'PILOT_REPORT.md'),'time':time.time()})

def main():
    lock=(ROOT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c,identity=verify();assert read(ROOT/'CPU_PREPARE_COMPLETE.json')['status']=='PASS'
    offset=read(ROOT/'CLOCK_OFFSET.json')['server_minus_local_seconds']
    # A new, explicitly authorized six-hour attempt; the original box is read-only.
    if not (ROOT/'TIMEBOX.json').exists():
        for gpu in c['allowed_gpus']:resource_check(gpu)
        atomic(ROOT/'TIMEBOX.json',make_timebox(c,offset,time.time()))
    box=read(ROOT/'TIMEBOX.json');assert time.time()<box['gpu_deadline_server']
    state=read(STATE) if STATE.exists() else {'stage':'gate','jobs':{},'retries':{},'matched_update':None}
    atomic(ROOT/'CONTROLLER_RECEIPT.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'cwd':str(ROOT),'identity':identity,'time':time.time()})
    if not (ROOT/'DEADLINE_GUARD_RECEIPT.json').exists():
        with (ROOT/'logs/deadline_guard.log').open('ab') as f:subprocess.Popen([sys.executable,str(ROOT/'deadline_guard.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    while time.time()<box['final_deadline_server']:
        stage=state['stage']
        if stage=='complete':break
        if stage in ['gate','train'] and time.time()>=box['training_deadline_server']:
            if stage=='gate':state['stage']='deadline';break
        if stage=='eval' and time.time()>=box['gpu_deadline_server']:state['stage']='deadline';break
        all_done=True
        for method,gpu in zip(GROUPS,c['allowed_gpus']):
            key=stage+'_'+method;target=stage_file(stage,method)
            if target.exists():continue
            all_done=False;q=state['jobs'].get(key)
            if q and exact_alive(q):continue
            if q:
                worker_status=ROOT/'gates'/method/'STATUS.json' if stage=='gate' else ROOT/'runs'/method/('EVAL_STATUS.json' if stage=='eval' else 'STATUS.json')
                if worker_status.exists() and read(worker_status).get('status')=='ERROR':
                    text=read(worker_status).get('error','').lower()
                    if any(word in text for word in ['numerical_gate','nonfinite','non-finite','non finite']):
                        state['stage']='numeric_failure';state['failed_method']=method;atomic(STATE,state);finish_report(state);return
                state['retries'][key]=state['retries'].get(key,0)+1
                if state['retries'][key]>2:state['stage']='failed';atomic(STATE,state);finish_report(state);return
                append(ROOT/'infra.jsonl',{'event':'worker_exit_without_commit','job':q,'retry':state['retries'][key],'time':time.time()})
            try:
                if stage=='train' and time.time()>=box['training_deadline_server']:continue
                state['jobs'][key]=launch(stage,method,gpu,state.get('matched_update'))
                state.pop('wait',None)
            except RuntimeError as e:
                if 'WAIT_RESOURCE' not in str(e):raise
                state['wait']={'method':method,'reason':str(e),'time':time.time()}
            atomic(STATE,state)
        if all_done:
            if stage=='gate':state['stage']='train';atomic(ROOT/'LAUNCH_GATE.json',{'status':'PASS','identity':identity,'methods':{g:read(ROOT/'gates'/g/'PASS.json') for g in GROUPS},'time':time.time()})
            elif stage=='train':
                common=set(c['milestones'])
                for g in GROUPS:common&={int(p.stem.split('_U')[1]) for p in (ROOT/'runs'/g/'milestones').glob('online_U*.pt')}
                if not common:state['stage']='insufficient_matched_budget';break
                state['matched_update']=max(common);state['stage']='eval'
            elif stage=='eval':state['stage']='complete'
        atomic(STATE,state);time.sleep(15)
    atomic(STATE,state);finish_report(state)
if __name__=='__main__':
    try:main()
    except Exception as e:atomic(ROOT/'CONTROLLER_ERROR.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise

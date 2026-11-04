"""Explicitly authorized co-tenancy; signals only proved own descendants."""
import argparse, fcntl, json, os, signal, statistics, subprocess, sys, time
from pathlib import Path
from guard_distributed import (parse_npu_smi,process_info,OwnedProcessTree,
    stop_owned_groups,atomic_json,utc_now)

ROOT=Path(__file__).resolve().parents[1]

def known_peer(process):
    """Accept only a live descendant of another registered followup worker."""
    roots={}
    for path in (ROOT/'resources').glob('active_*.json'):
        try:
            info=json.loads(path.read_text())['identity']
            actual=process_info(info['pid'])
            if actual['starttime_ticks']==info['starttime_ticks']:roots[actual['pid']]=actual
        except (OSError,ValueError,KeyError):continue
    try:
        current=process_info(process['pid']);seen=set()
        if current['starttime_ticks']!=process.get('starttime_ticks'):return False
        while current['pid']>1 and current['pid'] not in seen:
            if current['pid'] in roots:return True
            seen.add(current['pid']);current=process_info(current['ppid'])
    except (OSError,ValueError):return False
    return False

def training_peer_active():
    for p in (ROOT/'resources').glob('active_training_*.json'):
        try:
            saved=json.loads(p.read_text())['identity'];now=process_info(saved['pid'])
            if now['starttime_ticks']==saved['starttime_ticks'] and now['state']!='Z':return True
        except (OSError,ValueError,KeyError):continue
    return False

def policy_latency(pid, since=None):
    """Read only scalar timings already emitted by the existing WALA server."""
    try:
        argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if not any(x.endswith('/policy.py') for x in argv) or '--out' not in argv:return None
        ipc=Path(argv[argv.index('--out')+1])/'ipc'
        files=sorted(ipc.glob('*.response.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:128]
        values=[]
        for p in files:
            if since is not None and p.stat().st_mtime<=since:continue
            row=json.loads(p.read_text())
            if row.get('resize_audit') is None and isinstance(row.get('seconds'),(int,float)):
                values.append(row['seconds'])
        if len(values)<12:return {'n':len(values),'median':None}
        values.sort()
        return {'n':len(values),'median':statistics.median(values),'p90':values[int(.9*(len(values)-1))]}
    except (OSError,ValueError,IndexError):return None

def snapshot(cards):
    p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=20)
    s=parse_npu_smi(p.stdout,cards)
    for c in s['cards']:
        c['hbm_free_mb']=c['hbm_total_mb']-c['hbm_used_mb']
        for p in c['processes']:
            try:
                p.update(process_info(p['pid']))
                p['cwd']=os.readlink(f"/proc/{p['pid']}/cwd")
                p['uid']=Path(f"/proc/{p['pid']}").stat().st_uid
            except FileNotFoundError:p['exited_during_snapshot']=True
    return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cards',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=float,required=True)
    ap.add_argument('--own-cap-gib',type=float,required=True)
    ap.add_argument('--reserve-gib',type=float,default=4)
    ap.add_argument('--role',choices=('training','inference'),required=True)
    ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    cards=tuple(int(x) for x in a.cards.split(','))
    if not cards or any(x not in range(8) for x in cards):raise ValueError('Invalid cards')
    if os.uname().nodename!='liteserver-de36':raise RuntimeError('Only explicitly mapped de36 is allowed')
    out=Path(a.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must be in followup root')
    out.mkdir(parents=True,exist_ok=False)
    locks=[]
    for card in sorted(cards):
        f=(ROOT/f'.followup_{a.role}_card{card}.lock').open('a')
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    worker=owner=None;status='failed';reason=None
    try:
        baseline=snapshot(cards);atomic_json(out/'baseline.json',baseline)
        external={}
        for c in baseline['cards']:
            if c['health']!='OK':raise RuntimeError('Unhealthy NPU')
            if c['hbm_free_mb']<(a.own_cap_gib+a.reserve_gib)*1024:
                raise RuntimeError('Insufficient measured HBM for declared cap and reserve')
            for p in c['processes']:
                if 'starttime_ticks' not in p:raise RuntimeError('Unknown external identity')
                external[(p['pid'],p['starttime_ticks'])]=c['card']
        latency_before={str(pid):policy_latency(pid) for pid,_ in external}
        atomic_json(out/'external_latency_baseline.json',latency_before)
        env=os.environ.copy();env['ASCEND_RT_VISIBLE_DEVICES']=a.cards
        env['FOLLOWUP_OWN_CAP_GIB']=str(a.own_cap_gib)
        cmd=a.command[1:] if a.command[:1]==['--'] else a.command
        log=(out/'worker.log').open('w')
        worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,start_new_session=True)
        owner=OwnedProcessTree(process_info(worker.pid))
        registry=ROOT/'resources'/f'active_{a.role}_{a.cards.replace(",","_")}.json'
        atomic_json(registry,{'identity':process_info(worker.pid),'cards':cards,'role':a.role})
        atomic_json(out/'launch.json',{'pid':worker.pid,'identity':process_info(worker.pid),
            'cards':cards,'command':cmd,'utc':utc_now(),'external':list(map(list,external))})
        started=time.monotonic();started_wall=time.time();next_poll=0;slow_polls={}
        latency_since=started_wall;was_training=False;latency_grace_until=0
        def stop(sig,frame):raise KeyboardInterrupt('Guard interrupted')
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
        while worker.poll() is None:
            owner.refresh()
            if time.monotonic()-started>a.seconds:raise TimeoutError('Guard duration exhausted')
            if time.monotonic()<next_poll:time.sleep(.5);continue
            snap=snapshot(cards);owner.refresh();atomic_json(out/'telemetry_latest.json',snap)
            training_now=training_peer_active() if a.role=='inference' else False
            if a.role=='inference' and (training_now or was_training):
                latency_since=time.time();latency_grace_until=time.time()+60;slow_polls={}
            was_training=training_now
            latency_now={str(pid):policy_latency(pid,latency_since) for pid,_ in external}
            atomic_json(out/'external_latency_during.json',latency_now)
            for pid,after in latency_now.items():
                before=latency_before[pid]
                if before and after and before.get('median') and after.get('median'):
                    slow=(after['median']>1.15*before['median'] and after['p90']>1.25*before['p90'])
                    slow_polls[pid]=slow_polls.get(pid,0)+1 if slow else 0
                    if slow_polls[pid]>=2 and not training_now and time.time()>=latency_grace_until:
                        raise RuntimeError('Existing policy latency noticeably increased; stop only our work')
            with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
            for c in snap['cards']:
                if c['health']!='OK' or c['hbm_free_mb']<a.reserve_gib*1024:
                    raise RuntimeError('NPU health or HBM reserve gate failed')
                ours=[p for p in c['processes'] if owner.owns(p)]
                if sum(p['memory_mb'] for p in ours)>a.own_cap_gib*1024+1024:
                    raise RuntimeError('Own memory cap exceeded including runtime allowance')
                unknown=[p for p in c['processes'] if not owner.owns(p) and
                    (p['pid'],p.get('starttime_ticks')) not in external and not known_peer(p)]
                if unknown:raise RuntimeError('A new unqualified external process appeared')
            next_poll=time.monotonic()+5
        if worker.returncode:raise RuntimeError(f'Worker exit {worker.returncode}')
        status='completed'
    except BaseException as e:reason=repr(e)
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        cleanup=stop_owned_groups(owner,worker) if owner is not None else []
        result={'status':status,'reason':reason,'utc':utc_now(),'cleanup':cleanup,
                'external_processes_never_signalled':True}
        if worker is not None:worker.wait(timeout=10)
        atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    return int(status!='completed')

if __name__=='__main__':sys.exit(main())

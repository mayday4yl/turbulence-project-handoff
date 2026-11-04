"""CPU-only publication recovery after all native qualification gates completed.

Never launch an optimizer or silently reuse incomplete/changed native evidence.
The formal guard still takes a fresh resource baseline immediately before launch.
"""
import argparse, datetime, json, statistics, time
from pathlib import Path
from training_controller import ROOT, choose_candidate, digest, freeze_selection, save

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=('T1_DIRECT','T2_FEEDBACK_R'),required=True)
    ap.add_argument('--attempt',type=int,required=True)
    ap.add_argument('--resource-snapshot',type=Path,required=True)
    a=ap.parse_args();task=a.task;prefix=f'a{a.attempt}'
    q=ROOT/task/'qualification';evidence={}
    if (ROOT/task/'formal').exists() or (ROOT/'configs'/f'{task}_formal.json').exists():
        raise RuntimeError('Recovery only publishes an unstarted formal configuration')
    code={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'code/train_worker.py',*sorted((ROOT/'code/v031').glob('*.py'))]}
    def read(p):
        p=Path(p);evidence[str(p.relative_to(ROOT))]=digest(p)
        return json.loads(p.read_text())
    def completed(folder,mb,mode=None):
        if read(folder/'guard/result.json')['status']!='completed':raise RuntimeError('Guard did not pass')
        r=read(folder/'worker/results.json')
        if r['status']!='passed':raise RuntimeError('Worker did not pass')
        if mode and (r['mode']!=mode or r['microbatch']!=mb):raise RuntimeError('Mode/MB mismatch')
        ranks=r['rank_results']
        if sorted(x['rank'] for x in ranks)!=list(range(4)):raise RuntimeError('Four ranks required')
        for x in ranks:
            if x['status']!='passed' or x['identity']['actual_code_sha256']!=code:
                raise RuntimeError('Native proof trainer changed or rank failed')
            if x['effective_batch']!=32 or x['microbatch']!=mb:raise RuntimeError('Batch mismatch')
        return r
    native=read(ROOT/task/'native_semantics.json')
    if native['status']!='passed':raise RuntimeError('Native semantics incomplete')
    ledger=read(ROOT/task/'SMOKE_LEDGER.json')
    if ledger['consumed']>30 or any('consumed' not in x for x in ledger['reservations']):
        raise RuntimeError('Unreconciled smoke budget')
    candidates=[];caps=[]
    for mb in (8,4,2,1):
        perf=completed(q/f'{prefix}_speed_mb{mb}',mb,'performance-smoke')
        reuse=read(q/f'{prefix}_pressure_reuse_mb{mb}.json')
        pressure_path=Path(reuse['source']);memory=read(pressure_path)
        if read(pressure_path.parents[1]/'guard/result.json')['status']!='completed':
            raise RuntimeError('Pressure resource gate failed')
        if memory['status']!='passed' or memory['mode']!='memory-smoke' or memory['microbatch']<mb:
            raise RuntimeError('Pressure evidence insufficient')
        if memory['identity']['actual_code_sha256']!=code:raise RuntimeError('Pressure code changed')
        cap=reuse['actual_cap_gib'];caps.append(cap)
        if memory['max_peak_reserved_bytes']>=(cap-1)*2**30:raise RuntimeError('Pressure margin insufficient')
        d=perf['measured_seconds_per_update']
        if len(d)!=3 or max(d)>1.5*min(d):raise RuntimeError('Performance not stable')
        candidates.append({'microbatch':mb,'median_seconds':perf['median_seconds_per_update'],
            'peak_reserved_bytes':max(memory['max_peak_reserved_bytes'],perf['max_peak_reserved_bytes']),
            'relative_spread':(max(d)-min(d))/statistics.median(d)})
    selected=choose_candidate(candidates);mb=selected['microbatch']
    ddpf=q/f'{prefix}_ddp_mb{mb}';ddp=completed(ddpf,mb)
    if any(x['config_sha256']!=digest(ROOT/'configs'/f'{task}_mb{mb}.json') for x in ddp['rank_results']):
        raise RuntimeError('Selected numerical configuration changed')
    resume=completed(q/f'{prefix}_resume_mb{mb}',mb,'resume-check')
    for x in resume['rank_results']:
        b=x['resume_boundary_parity'];r=x['resume_parity']
        if not (b['exact'] and b['checked_before_any_restored_optimizer_attempt'] and r['passed'] and r['identical_suffix_start_ids']):
            raise RuntimeError('Exact restore / bounded FP32 continuation gate failed')
    baseline=read(q/f'{prefix}_speed_mb{mb}'/'guard/baseline.json')
    current=read(a.resource_snapshot.resolve())
    if time.time()-datetime.datetime.fromisoformat(current['utc']).timestamp()>600:raise RuntimeError('Resource snapshot stale')
    old={c['card']:c for c in baseline['cards']}
    ids=lambda c:{(p['pid'],p.get('starttime_ticks')) for p in c['processes']}
    if set(old)!={0,1,2,3} or {c['card'] for c in current['cards']}!=set(old):raise RuntimeError('Resource mapping mismatch')
    for c in current['cards']:
        if c['health']!='OK' or c['hbm_free_mb']<(min(caps)+8)*1024 or not ids(c)<=ids(old[c['card']]):
            raise RuntimeError('Current resources differ materially from qualified co-load')
    if statistics.mean(c['aicore_percent'] for c in current['cards'])>statistics.mean(c['aicore_percent'] for c in baseline['cards'])+20:
        raise RuntimeError('Current load increased materially')
    selected.update(own_cap_gib=min(caps),all_candidates=candidates,
        expected_1000_update_seconds=1000*selected['median_seconds'],
        qualification_unix=datetime.datetime.fromisoformat(read(q/f'{prefix}_resume_mb{mb}'/'guard/result.json')['utc']).timestamp(),
        actual_shared_snapshot=baseline,recovery_prelaunch_snapshot=current,
        publication_recovered_unix=time.time(),qualification_evidence_sha256=evidence,
        recovery_note='Completed native gates reused after JSON circular-reference publication fault; zero additional optimizer updates.')
    json.dumps(selected,allow_nan=False)
    freeze_selection(task,selected)
    save(ROOT/task/'QUALIFICATION_PUBLICATION_RECOVERY.json',{'status':'passed','attempt':a.attempt,
         'microbatch':mb,'accumulation':8//mb,'smoke_updates_unchanged':ledger['consumed'],
         'new_optimizer_updates':0,'recovered_unix':time.time(),'evidence_sha256':evidence})
    print(json.dumps({'status':'passed','microbatch':mb,'seconds_per_update':selected['median_seconds'],
        'smoke_updates':ledger['consumed'],'additional_optimizer_updates':0}))

if __name__=='__main__':main()

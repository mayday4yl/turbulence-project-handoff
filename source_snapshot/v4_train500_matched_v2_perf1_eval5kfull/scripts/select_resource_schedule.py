"""Select bounded, measured legal execution by four-arm makespan, not fastest arm."""
import itertools,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4.runtime import save_json,sha256
G=('D0','R0','R08','L08')
PAIR={5:['R08','D0'],6:['R0','L08']}


def simulate(solo,shared,units,mode,dynamic):
    remaining={g:float(units) for g in G};mapping={g:gpu for gpu,gs in PAIR.items() for g in gs};elapsed=0.;moves=[]
    while remaining:
        by={gpu:[g for g in remaining if mapping[g]==gpu] for gpu in (5,6)}
        if dynamic:
            for src,dst in [(5,6),(6,5)]:
                if not by[dst] and len(by[src])>1:
                    g=max(by[src],key=lambda x:remaining[x]);mapping[g]=dst;moves.append({'group':g,'at_seconds':elapsed,'to':dst});by[src].remove(g);by[dst].append(g)
        speed={}
        for gpu,groups in by.items():
            if len(groups)==1:speed[groups[0]]=1/solo[groups[0]]
            elif len(groups)==2:
                for g in groups:speed[g]=1/(sum(solo[h] for h in groups) if mode=='serialized_updates' else shared[g])
        duration=min(remaining[g]/speed[g] for g in remaining)
        elapsed+=duration
        for g in list(remaining):
            remaining[g]-=duration*speed[g]
            if remaining[g]<1e-7:del remaining[g]
    return elapsed,moves


def main():
    p=ROOT/'reports/PERFORMANCE_STATUS.json';report=json.loads(p.read_text())
    assert report['status']=='BENCHMARK_PASS_NEEDS_SELECTION_AND_EXECUTION_ACCEPTANCE'
    profiles={}
    for g in G:
        shared=report['reports'][g];solo=report['solo_reports'][g]
        assert shared['status']==solo['status']=='PASS'
        assert shared['optimizer_updates']==solo['optimizer_updates']==0
        def case(r,name):return next(x for x in r['cases'] if x.get('name')==name)
        def rate(c):return statistics.median(c['seconds'][1:])
        c=case(shared,'concurrent_m2');z=case(shared,'serial_m2');s=case(solo,'solo_m2')
        # Equal-count shared trials let the faster peer finish first. Later
        # samples then contain solo work and must not be called steady shared.
        # Use a conservative observed maximum for scheduling until formal
        # overlapping windows provide a calibrated shared rate.
        profiles[g]={'solo_sec_update':rate(s),'shared_sec_update':max(c['seconds']),'serialized_compute_sec_update':rate(z),
             'shared_estimate_basis':'conservative maximum; equal-count trial includes peer-finish transients, not calibrated steady shared',
             'solo_samples_seconds':s['seconds'],'shared_samples_seconds':c['seconds'],'serialized_samples_seconds':z['seconds'],
             'solo_peak_allocated_GiB':s['peak_allocated_GiB'],'shared_peak_allocated_GiB':c['peak_allocated_GiB'],
             'serialized_peak_allocated_GiB':z['peak_allocated_GiB']}
    solo={g:profiles[g]['solo_sec_update'] for g in G};shared={g:profiles[g]['shared_sec_update'] for g in G}
    serial={g:profiles[g]['serialized_compute_sec_update'] for g in G}
    candidates=[]
    for name,mode,dynamic,reference in [('shared_fixed','concurrent',False,solo),('shared_dynamic','concurrent',True,solo),('serialized_dynamic','serialized_updates',True,serial)]:
        value,moves=simulate(reference,shared,5000,mode,dynamic)
        candidates.append({'name':name,'compute_scheduling':mode,'dynamic_relocation':dynamic,'training_5k_seconds':value,'predicted_moves':moves})
    # True GPU-exclusive nonpreemptive job queues; enumerate two-card partitions.
    for size in range(1,4):
        for left in itertools.combinations(G,size):
            if 'D0' not in left:continue
            right=[g for g in G if g not in left]
            value=max(sum(solo[g]*5000 for g in left),sum(solo[g]*5000 for g in right))
            candidates.append({'name':'solo_queue_'+'_'.join(left),'compute_scheduling':'solo_queue','dynamic_relocation':False,
               'training_5k_seconds':value,'queues':{'5':sorted(left,key=lambda g:-solo[g]),'6':sorted(right,key=lambda g:-solo[g])}})
    selected=min(candidates,key=lambda x:x['training_5k_seconds'])
    # Existing warm metric timings: same FP32 network work without first-use Input baseline.
    parent=ROOT.parent/'v4_train500_matched_v2';eval_rates={}
    for g in G:
        a=json.loads((parent/f'reports/shared_gpu_acceptance/{g}.json').read_text())
        t={x['population']:x['seconds'] for x in a['full_evaluation'] if x['weights']=='EMA'}
        eval_rates[g]={'regular':sum(t.values()),'fulltrain_final':20*t['train_monitor400']+t['val800']}
    estimates={}
    for u in [5000,7500,10000]:
        training=selected['training_5k_seconds']*u/5000
        eval_total=sum((u//250-1)*v['regular']+(v['fulltrain_final'] if u==10000 else v['regular']) for v in eval_rates.values())
        # All task work is counted; equal-card lower bound and initial-mapping upper estimate.
        eval_upper=max(sum((u//250-1)*eval_rates[g]['regular']+(eval_rates[g]['fulltrain_final'] if u==10000 else eval_rates[g]['regular']) for g in gs) for gs in PAIR.values())
        estimates[str(u)]={'training_compute_hours_estimate':training/3600,
            'ordinary_RAW_evaluation_wall_hours_range':[eval_total/2/3600,eval_upper/3600],
            'save_and_report_hours_allowance':[.15*u/5000,.75*u/5000],
            'subtotal_excluding_refinement_hours_range':[(training+eval_total/2)/3600+.15*u/5000,(training+eval_upper)/3600+.75*u/5000],
            'refinement_auxiliary_hours':'UNKNOWN until acceptance timing; excluded from subtotal',
            'not_a_complete_delivery_ETA':True}
    value={'status':'SELECTED_PENDING_EXECUTION_GATES','selection_objective':'minimum estimated all-four common-milestone wall-clock makespan',
         'gpu_indices':[5,6],'microbatch':2,'accumulation':16,'effective_batch':32,'optimizer_steps_in_benchmark':0,
         'profiles':profiles,'candidates':candidates,'selected':selected,'ETA':estimates,
         'benchmark_report_sha256':sha256(p),'initial_gpu_mapping':PAIR,'stage_reselection':'Use actual compute/wait/eval timings at each common stage; no group may cross early',
         'assumptions':['Bounded4-gradient timings, median last3; no optimizer updates, therefore optimizer/EMA cost not calibrated',
             'Same-card conflicts and unequal completion modeled; dynamic migration uses measured solo rate',
             'Different GPU pairing performance not silently assumed measured','Full formal stable timings supersede these estimates'],
         'time':time.time()}
    save_json(ROOT/'protocol/RESOURCE_SCHEDULE.json',value);print(json.dumps({'selected':selected,'ETA':estimates},indent=2))

if __name__=='__main__':main()

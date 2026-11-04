"""Execution ETA from actual optimizer timings; RAW schedule and auxiliary costs."""
import json,statistics,time
from pathlib import Path
from .runtime import save_json
from .stages import GROUPS,NODES

ROOT=Path(__file__).resolve().parents[1]

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def compute_span(updates,rates,mapping):
    remaining={g:float(v) for g,v in updates.items() if v>0};mapping=dict(mapping);elapsed=0.
    while remaining:
        by={gpu:[g for g in remaining if mapping[g]==gpu] for gpu in (5,6)}
        for src,dst in [(5,6),(6,5)]:
            if not by[dst] and len(by[src])>1:
                g=max(by[src],key=lambda g:remaining[g]);by[src].remove(g);by[dst].append(g);mapping[g]=dst
        speeds={g:1/sum(rates[h] for h in gs) for gs in by.values() for g in gs}
        dt=min(remaining[g]/speeds[g] for g in remaining);elapsed+=dt
        remaining={g:v-dt*speeds[g] for g,v in remaining.items() if v-dt*speeds[g]>1e-7}
    return elapsed

def eta_report(run):
    run=Path(run);schedule=read(ROOT/'protocol/RESOURCE_SCHEDULE.json');rates={};updates={};groups={};eval_rates={};save_rates={}
    initial={'R08':5,'D0':5,'R0':6,'L08':6}
    mapping={**initial,**read(run/'PIDS.json',{}).get('gpu_mapping',{})}
    for g in GROUPS:
        base=run/g;status=read(base/'STATUS.json',{});updates[g]=status.get('update',0)
        lines=(base/'train_steps.jsonl').read_text().splitlines() if (base/'train_steps.jsonl').exists() else []
        rows=[]
        for line in lines:
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue
        stable=[r for r in rows if r['update']>25][-100:]
        rates[g]=statistics.median(r['update_seconds'] for r in stable) if len(stable)>=20 else schedule['profiles'][g]['serialized_compute_sec_update']
        pops={}
        for f in sorted((base/'evaluations').glob('U*/raw/*/aggregate.json')):
            a=read(f)
            if a['elapsed_seconds']>0:pops[f.parent.name]=a['elapsed_seconds']
        if 'train_monitor400' not in pops:pops['train_monitor400']=pops.get('train8000',1800)/20
        eval_rates[g]=pops
        timings=[]
        if (base/'timings.jsonl').exists():
            for line in (base/'timings.jsonl').read_text().splitlines():
                try:timings.append(json.loads(line))
                except json.JSONDecodeError:continue
        saves=[r['seconds'] for r in timings if r['kind']=='checkpoint_save'][-20:]
        save_rates[g]=statistics.median(saves) if saves else 2.
        groups[g]={'update':updates[g],'compute_sec_update':rates[g],'stable_samples':len(stable),
                   'rate_basis':'formal_optimizer_updates' if len(stable)>=20 else 'zero_step_benchmark_proxy',
                   'time_categories':read(base/'TIMEBOOK.json',{})}
    ref=[]
    for g in ['R0','R08']:
        r=read(ROOT/f'reports/REFINEMENT_RUNTIME_{g}.json')
        ref.append(r['estimated_val800_seconds'] if r else None)
    result={'time':time.time(),'groups':groups,'ETA_remaining_hours':{},
            'notes':['Two-card critical path; serialized updates and approved single-GPU boundary relocation',
                     'Each common stage resets initial pairing; no EMA inference; full-train at5k and10k only',
                     'Intervals are engineering allowances, not statistical confidence intervals'],
            'refinement_per_common_node_seconds_estimate':max(ref) if all(v is not None for v in ref) else None}
    for target in NODES:
        if (run/'stages'/f'U{target:05d}'/'RELEASE.json').exists():continue
        train=0.;remaining_evaluation=0.;saving=0.;cursor=dict(updates);map_now=mapping
        for node in [n for n in NODES if n<=target]:
            if all(cursor[g]>=node for g in GROUPS):continue
            train+=compute_span({g:max(0,node-cursor[g]) for g in GROUPS},rates,map_now)
            cursor={g:max(node,cursor[g]) for g in GROUPS};map_now=initial
        card_evaluation={5:0.,6:0.}
        for g in GROUPS:
            for u in range(((updates[g]+249)//250)*250,target+1,250):
                if (run/g/'evaluations'/f'U{u:05d}'/'NODE_COMMIT.json').exists():continue
                pops=['train8000','val800'] if u in (5000,10000) else ['train_monitor400','val800']
                value=sum(eval_rates[g].get(p,eval_rates[g].get('train_monitor400',60)*(20 if p=='train8000' else 2 if p=='val800' else 1)) for p in pops)
                card_evaluation[initial[g]]+=value
            saving+=max(0,target-updates[g])/25*save_rates[g]
        remaining_evaluation=max(card_evaluation.values())
        nodes=[n for n in NODES if n<=target and not (run/'stages'/f'U{n:05d}'/'RELEASE.json').exists()]
        refine=(max(ref)*len(nodes)) if all(v is not None for v in ref) else None
        overhead=saving/2+len(nodes)*600
        subtotal=train+remaining_evaluation+overhead
        result['ETA_remaining_hours'][str(target)]={'training':train/3600,'RAW_evaluation':remaining_evaluation/3600,
            'saving_and_reporting_allowance':overhead/3600,'refinement':None if refine is None else refine/3600,
            'complete_delivery_range':None if refine is None else [(subtotal+refine)/3600,(1.2*subtotal+1.5*refine)/3600]}
    save_json(run/'ETA.json',result)
    if all(v['stable_samples']>=20 for v in groups.values()) and not (run/'STABLE_PROFILE.json').exists():
        save_json(run/'STABLE_PROFILE.json',{'status':'FIRST_STABLE_TIMING_PROFILE','time':time.time(),'groups':groups,
            'scope':'Existing official timing records; no additional optimizer steps or candidate benchmarks',
            'next_action':'Only largest1-2 measured bottlenecks eligible for follow-up; no automatic scientific changes'})

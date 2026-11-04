"""Conservative two-card ETA; distinguish formal measured rates from preflight extrapolation."""
import json,statistics,time
from pathlib import Path
from .runtime import save_json
from .stages import GROUPS,NODES

def eta_report(run,frozen):
    run=Path(run);groups={};rates={};updates={};durations={}
    for g in GROUPS:
        base=run/g;rows=[]
        if (base/'train_steps.jsonl').exists():
            for line in (base/'train_steps.jsonl').read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
        stable=[r for r in rows if r['update']>25][-100:];updates[g]=rows[-1]['update'] if rows else 0
        rates[g]=statistics.median(r['update_seconds'] for r in stable) if len(stable)>=20 else None
        durations[g]={}
        for path in sorted((base/'evaluations').glob('U*/raw/*/aggregate.json')):
            value=json.loads(path.read_text())
            if value.get('elapsed_seconds',0)>0:durations[g][path.parent.name]=value['elapsed_seconds']
        groups[g]={'update':updates[g],'stable_samples':len(stable),'compute_sec_update':rates[g],'basis':'formal_optimizer_update_median_after_U25' if rates[g] else 'NOT_YET_STABLE','timebook':json.loads((base/'TIMEBOOK.json').read_text()) if (base/'TIMEBOOK.json').exists() else {}}
    result={'time':time.time(),'groups':groups,'complete_delivery_remaining_hours':{},'assumptions':['Two card critical path with fixed pairings and serialized complete updates; no sum of both GPU wall times','Evaluation estimates from existing committed evaluations; missing fulltrain8000 estimated from monitor400 x20','Future refinement extrapolated from preflight G4 timing until full val800 measured','20-40 percent overhead range includes saving, contention, CPU reports and stage waiting; not a statistical confidence interval','Best checkpoint backfill worst-case included in final endpoint; no automatic budget extension']}
    if all(rates[g] is not None for g in GROUPS):
        root=Path(__file__).resolve().parents[1];pre=json.loads((root/'reports/ACCEPTANCE.json').read_text())
        refine=[]
        for g in ('R0','R08'):
            measured=list((run/'refinement_ablation').glob(f'U*/{g}/SUMMARY.json'))
            refine.append(json.loads(measured[-1].read_text())['runtime']['total_this_invocation_seconds'] if measured else pre['groups'][g]['checks']['refinement']['seconds_per_stage_sample']*1200+durations[g].get('val800',0))
        for node in NODES:
            if (run/'stages'/f'U{node:05d}'/'RELEASE.json').exists():continue
            cards=[]
            for cohort in frozen['mapping'].values():
                seconds=0
                for g in cohort:
                    seconds+=max(0,node-updates[g])*rates[g]
                    for u in range(((updates[g]+249)//250)*250,node+1,250):
                        if (run/g/'evaluations'/f'U{u:05d}/NODE_COMMIT.json').exists():continue
                        pop='train8000' if u in NODES else 'train_monitor400'
                        seconds+=durations[g].get(pop,durations[g].get('train_monitor400',60)*20)+durations[g].get('val800',120)
                    if node==2000:seconds+=durations[g].get('train8000',durations[g].get('train_monitor400',60)*20)
                cards.append(seconds)
            remaining_nodes=[u for u in NODES if u<=node and not (run/'stages'/f'U{u:05d}/RELEASE.json').exists()]
            subtotal=max(cards)+len(remaining_nodes)*(max(refine)+600)
            result['complete_delivery_remaining_hours'][str(node)]=[subtotal*1.2/3600,subtotal*1.4/3600]
        if not (run/'STABLE_PROFILE.json').exists():save_json(run/'STABLE_PROFILE.json',{'status':'FORMAL_STABLE_TIMING','groups':groups,'time':time.time(),'extra_training_updates_for_profile':0})
    save_json(run/'ETA.json',result)

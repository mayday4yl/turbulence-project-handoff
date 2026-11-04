"""Read-only scientific inputs; deterministic evaluation partition and merge."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import run_auto_to_1000 as frozen
import run_parallel_to_1000 as prior

ROOT = frozen.ROOT
BASE = ROOT/'artifacts/evaluation_sharded_4card'
CANON = ROOT/'artifacts/evaluation/update_001000'
AUTH = ROOT/'artifacts/EVAL_SHARDED_AUTHORIZATION.json'
DELIVERY = ROOT/'artifacts/EVAL_SHARDED_SHA256.json'
CARDS = (0,1,2,3)


def verify():
    cfg = prior.verify()
    a = frozen.read(AUTH)
    if a['physical_cards'] != list(CARDS) or a['training_updates_allowed'] != 0:
        raise RuntimeError('Four-card evaluation authorization differs')
    if a['per_worker_batch_size'] != 1 or a['precision'] != 'fp32':
        raise RuntimeError('Frozen numerical configuration differs')
    if prior.last_committed() != 1000:
        raise RuntimeError('Evaluation-only handoff requires the complete 1000 checkpoint')
    for rel, digest in frozen.read(DELIVERY).items():
        if frozen.digest(ROOT/rel) != digest:
            raise RuntimeError('Sharded delivery hash mismatch: '+rel)
    return cfg


def validate_row(row, source, weights, mode):
    g = source['initial_G']
    if any(row.get(k)!=v for k,v in {'start_id':source['start_id'],
            'trajectory_id':source['trajectory_id'],'initial_G':g,'weights':weights,
            'mode':mode,'global_update':1000,'batch_size':1}.items()):
        raise RuntimeError('Evaluation row identity mismatch')
    steps = row['predictions']
    if len(steps)!=g:
        raise RuntimeError('Incomplete legal prefixes')
    for k,p in enumerate(steps,1):
        if p['K']!=k or p['current_g']!=g-k+1:
            raise RuntimeError('Wrong prefix order')
        for metric in ['clean_mse','network_redegradation_mse']:
            if not math.isfinite(p[metric]) or p[metric]<0:
                raise RuntimeError('Invalid evaluation metric')
        if p['clean_psnr_db'] is None:
            if not (p['zero_mse'] and p['psnr_infinite']):
                raise RuntimeError('Invalid nonfinite PSNR marker')
        elif not math.isfinite(p['clean_psnr_db']):
            raise RuntimeError('Nonfinite PSNR')
        refinement = p['refinement']
        if (mode=='refined' and k<g) != (refinement is not None):
            raise RuntimeError('Refinement transition coverage differs')
        if refinement:
            if refinement['iterations']!=40 or not refinement['finite'] or not refinement['gradient_connected']:
                raise RuntimeError('Refinement definition or numerics differ')
            if not (math.isfinite(refinement['best_objective']) and
                    refinement['best_objective'] <= refinement['selected_initial_objective']):
                raise RuntimeError('Invalid best iterate')


def canonical_identity(weights):
    identity = deepcopy(frozen.read(CANON/'raw/progress.json')['identity'])
    identity['weights'] = weights
    return identity


def rowfile(base, mode, start_id):
    return base/mode/'rows'/(hashlib.sha256(start_id.encode()).hexdigest()+'.json')


def partition(starts, existing, cards=CARDS):
    tasks = []
    for weights in ['raw','ema']:
        for mode in ['unrefined','refined']:
            for row in starts:
                key = (weights,mode,row['start_id'])
                if key in existing:
                    continue
                cost = ({4:9.8,3:6.7,2:3.5,1:.3} if mode=='refined'
                        else {4:.56,3:.46,2:.37,1:.29})[row['initial_G']]
                tasks.append((cost,key))
    loads = {card:0. for card in cards}
    assigned = {card:[] for card in cards}
    for cost,key in sorted(tasks,key=lambda p:(-p[0],p[1])):
        card = min(cards,key=lambda c:(loads[c],c))
        assigned[card].append(key);loads[card]+=cost
    flattened = [key for group in assigned.values() for key in group]
    expected = {(w,m,r['start_id']) for w in ['raw','ema'] for m in ['unrefined','refined'] for r in starts}
    if len(flattened)!=len(set(flattened)) or set(flattened)&set(existing) or set(flattened)|set(existing)!=expected:
        raise RuntimeError('Partition coverage or duplication failure')
    return assigned,loads


def make_plan():
    cfg = verify()
    starts = prior.read_lines(Path(cfg['dataset_root'])/'start_manifest.jsonl')
    sources = {r['start_id']:r for r in starts}
    assert len(sources)==1600
    existing = {}
    previous_progress = {}
    for weights in ['raw','ema']:
        folder = CANON/weights
        if (folder/'progress.json').exists():
            p = frozen.read(folder/'progress.json')
            if p['identity']!=canonical_identity(weights):
                raise RuntimeError('Existing evaluation provenance differs')
            previous_progress[weights] = p
        for mode in ['unrefined','refined']:
            for f in sorted((folder/mode/'rows').glob('*.json')):
                if weights not in previous_progress:
                    raise RuntimeError('Existing row lacks execution identity')
                row = frozen.read(f)
                validate_row(row,sources[row['start_id']],weights,mode)
                if f!=rowfile(folder,mode,row['start_id']):
                    raise RuntimeError('Row filename differs from start identity')
                key=(weights,mode,row['start_id'])
                if key in existing:
                    raise RuntimeError('Duplicate reused row')
                existing[key]={'file':str(f),'sha256':frozen.digest(f)}
    assignments,loads = partition(starts,existing)
    jobs=[]
    subset_root=Path(cfg['dataset_root'])/'subsets/v031_eval4_20260917'
    subset_root.mkdir(parents=True,exist_ok=True)
    for card,keys in assignments.items():
        for weights in ['raw','ema']:
            for mode in ['unrefined','refined']:
                selected={key[2] for key in keys if key[:2]==(weights,mode)}
                if not selected:
                    continue
                subset=[row for row in starts if row['start_id'] in selected]
                name=f'card{card}_{weights}_{mode}'
                manifest=subset_root/(name+'.jsonl')
                body=''.join(json.dumps(row,sort_keys=True)+'\n' for row in subset)
                if manifest.exists() and manifest.read_text()!=body:
                    raise RuntimeError('Existing shard manifest conflict')
                manifest.write_text(body)
                jobs.append({'name':name,'card':card,'weights':weights,'mode':mode,
                    'manifest':str(manifest.relative_to(cfg['dataset_root'])),
                    'manifest_sha256':frozen.digest(manifest),'count':len(subset),
                    'start_ids':[r['start_id'] for r in subset],
                    'output':str(BASE/'shards'/name)})
    result={'utc':frozen.utc(),'global_update':1000,'cards':list(CARDS),
        'checkpoint_sha256':frozen.digest(ROOT/'runs/formal/checkpoint_001000.pt'),
        'original_manifest_sha256':frozen.digest(Path(cfg['dataset_root'])/'start_manifest.jsonl'),
        'identities':{w:canonical_identity(w) for w in ['raw','ema']},
        'previous_progress':previous_progress,
        'reused':[dict(weights=k[0],mode=k[1],start_id=k[2],**v) for k,v in existing.items()],
        'jobs':jobs,'estimated_seconds_per_card':loads,'training_updates':0}
    frozen.write(BASE/'PLAN.json',result)
    return result


def verify_job(job, plan):
    folder=Path(job['output'])
    if not (folder/'summary.json').exists():
        return False
    summary=frozen.read(folder/'summary.json');progress=frozen.read(folder/'progress.json')
    identity=deepcopy(plan['identities'][job['weights']])
    identity.update(manifest_sha256=job['manifest_sha256'],modes=[job['mode']])
    if (summary['identity']!=identity or progress['identity']!=identity or
            summary['status']!='complete' or progress['status']!='complete' or
            summary['start_count_per_mode']!=job['count'] or
            not summary['network_unchanged'] or not summary['simulator_unchanged']):
        raise RuntimeError('Shard completion identity or state check failed')
    rows=prior.read_lines(folder/job['mode']/'per_start.jsonl')
    if len(rows)!=job['count'] or {r['start_id'] for r in rows}!=set(job['start_ids']):
        raise RuntimeError('Shard completion coverage differs')
    return True


def add_unique(groups, row, source, weights, mode):
    validate_row(row,source,weights,mode)
    key=(weights,mode,row['start_id'])
    if key in groups:
        raise RuntimeError('Duplicate merged start')
    groups[key]=row


def merge():
    cfg=verify();plan=frozen.read(BASE/'PLAN.json')
    from v031.evaluation import summarize_records
    starts=prior.read_lines(Path(cfg['dataset_root'])/'start_manifest.jsonl')
    sources={r['start_id']:r for r in starts}
    groups={};timings=defaultdict(list);events=defaultdict(list);provenance=[]
    for old in plan['reused']:
        if frozen.digest(old['file'])!=old['sha256']:
            raise RuntimeError('Reused committed result changed')
        row=frozen.read(old['file'])
        add_unique(groups,row,sources[row['start_id']],old['weights'],old['mode'])
    for weights,p in plan['previous_progress'].items():
        timings[weights].extend(p.get('completed_batches',[]))
        events[weights].extend(p.get('batch_size_events',[]))
    for job in plan['jobs']:
        if not verify_job(job,plan):
            raise RuntimeError('Cannot merge incomplete shard')
        folder=Path(job['output']);progress=frozen.read(folder/'progress.json')
        timings[job['weights']].extend(progress['completed_batches'])
        events[job['weights']].extend(progress.get('batch_size_events',[]))
        for row in prior.read_lines(folder/job['mode']/'per_start.jsonl'):
            add_unique(groups,row,sources[row['start_id']],job['weights'],job['mode'])
            source=rowfile(folder,job['mode'],row['start_id'])
            target=rowfile(CANON/job['weights'],job['mode'],row['start_id'])
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if frozen.digest(target)!=frozen.digest(source):
                    raise RuntimeError('Canonical row conflict')
            else:
                temp=target.with_suffix('.merge.tmp');shutil.copyfile(source,temp);temp.replace(target)
        for preview in (folder/job['mode']/'previews').glob('*.png'):
            target=CANON/job['weights']/job['mode']/'previews'/preview.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                shutil.copyfile(preview,target)
        provenance.append({'job':job['name'],'card':job['card'],'count':job['count'],
            'summary_sha256':frozen.digest(folder/'summary.json'),
            'manifest_sha256':job['manifest_sha256']})
    expected={(w,m,sid) for w in ['raw','ema'] for m in ['unrefined','refined'] for sid in sources}
    if set(groups)!=expected:
        raise RuntimeError('Merged evaluation coverage is not exactly 6400 paths')
    timing_reconciliations=[]
    for weights in ['raw','ema']:
        summaries={};folder=CANON/weights;folder.mkdir(parents=True,exist_ok=True)
        for mode in ['unrefined','refined']:
            rows=[groups[(weights,mode,row['start_id'])] for row in starts]
            for g in [1,2,3,4]:
                recorded=sum(b['batch_size'] for b in timings[weights] if b['mode']==mode and b['G']==g)
                missing=400-recorded
                # A TERM may arrive after one atomic row commit but before its
                # progress commit. Its scalar metrics are reusable; wall timing
                # is explicitly unavailable and represented only by a lower bound.
                if missing:
                    if missing!=1:
                        raise RuntimeError('Unexpected timing-count reconciliation')
                    candidates=[o for o in plan['reused'] if o['weights']==weights and o['mode']==mode
                                and sources[o['start_id']]['initial_G']==g]
                    last=max(candidates,key=lambda o:Path(o['file']).stat().st_mtime_ns)
                    seconds=groups[(weights,mode,last['start_id'])]['batch_seconds']
                    timings[weights].append({'mode':mode,'G':g,'batch_size':1,'seconds':seconds,
                        'batch_wall_seconds':seconds,'wall_time_is_lower_bound':True})
                    timing_reconciliations.append({'weights':weights,'mode':mode,'G':g,
                        'reason':'committed row precedes progress commit; rollout-only wall lower bound'})
            out=folder/mode/'per_start.jsonl';temp=out.with_suffix('.merge.tmp')
            temp.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows));temp.replace(out)
            summaries[mode]=summarize_records(rows,tie_psnr_db=cfg['evaluation']['tie_psnr_db'])
            frozen.write(folder/mode/'G_curves.json',summaries[mode]['G_curves'])
            for start_id in cfg['evaluation']['fixed_preview_start_ids']:
                name=hashlib.sha256(start_id.encode()).hexdigest()[:20]+'.png'
                if not (folder/mode/'previews'/name).exists():
                    raise RuntimeError('A prespecified preview is missing')
        identity=plan['identities'][weights]
        execution={'kind':'disjoint_sample_shards','plan':str(BASE/'PLAN.json'),
            'plan_sha256':frozen.digest(BASE/'PLAN.json'),'cross_card_acceptance':str(BASE/'CROSS_CARD_ACCEPTANCE.json'),
            'jobs':[p for p in provenance if p['job'].split('_')[1]==weights],
            'timing_reconciliations':timing_reconciliations}
        summary={'identity':identity,'status':'complete','start_count_per_mode':1600,
            'network_unchanged':True,'simulator_unchanged':True,'modes':summaries,
            'batch_size_events':events[weights],
            'measured_rollout_seconds':sum(b['seconds'] for b in timings[weights]),
            'measured_batch_wall_seconds':sum(b['batch_wall_seconds'] for b in timings[weights]),
            'completing_process_seconds':None,'peak_allocated_bytes':None,
            'timing_scope':'SUM of sample batch wall times across independent cards; not elapsed time; see explicit lower-bound reconciliation if any',
            'execution_provenance':execution}
        progress={'identity':identity,'status':'complete','completed_batches':timings[weights],
            'batch_size_events':events[weights],'execution_provenance':execution}
        frozen.write(folder/'summary.json',summary);frozen.write(folder/'progress.json',progress)
        if not prior.verify_evaluation(cfg,1000,weights):
            raise RuntimeError('Original canonical completion gate failed')
    result={'status':'passed','utc':frozen.utc(),'total_unique_weight_mode_starts':len(groups),
        'reused_rows':len(plan['reused']),'new_rows':len(groups)-len(plan['reused']),
        'all_authoritative_ids_and_legal_prefixes_present':True,'shards':provenance,
        'timing_reconciliations':timing_reconciliations}
    frozen.write(BASE/'MERGE_ACCEPTANCE.json',result)
    return result

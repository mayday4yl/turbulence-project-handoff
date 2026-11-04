"""Bounded static sample shards: diagnostics, 14 dev configurations, one confirmation."""
from pathlib import Path
from dataclasses import asdict
import argparse, json, math, os, statistics, time, traceback
from i1_worker import init, run_path, ROOT
from v031.followup_refinement import candidates
from v031.followup_diagnostics import diagnose
from v031.runtime import save_json,sha256,state_hash
from v031.training import memory_stats
import torch
import numpy as np

def provenance():
    names=['i1_batch.py','i1_worker.py','v031/followup.py','v031/followup_refinement.py','v031/followup_diagnostics.py']
    return {n:sha256(ROOT/'code'/n) for n in names}

OUT=ROOT/'I1_REFINE_REPAIR'

def rank_and_freeze():
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    splits=json.loads((ROOT/'shared/manifests/FROZEN_SPLITS.json').read_text())
    ids=splits['splits']['refine_dev8']['start_ids']
    def read(c,s):return json.loads((OUT/'dev'/c/s/'result.json').read_text())
    base={s:read('U0',s) for s in ids}
    if any(r.get('status')!='complete' for r in base.values()):raise RuntimeError('A U0 baseline failed; candidate ranking blocked')
    ranking=[]
    for c in candidates()[2:]:
        rows=[read(c.name,s) for s in ids];reasons=[];deltas={4:[],3:[],2:[]};by_source={};seconds=0;calls={'network':0,'F':0,'VJP':0}
        for s,r in zip(ids,rows):
            if r.get('status')!='complete':reasons.append(s);continue
            p=r['metrics'][-1]['PSNR'];b=base[s]['metrics'][-1]['PSNR']
            if p is None or b is None or not math.isfinite(p-b):reasons.append(s);continue
            g=r['metrics'][-1]['G'];d=p-b;deltas[g].append(d)
            by_source.setdefault(r['source_id'],[]).append(d);seconds+=r['seconds']
            for k in calls:calls[k]+=r['calls'][k]
        valid=not reasons and all(len(v)==8 for v in deltas.values()) and len(by_source)==8
        dg={str(k):statistics.mean(v) if valid else None for k,v in deltas.items()}
        score=statistics.mean(dg.values()) if valid else None
        med=statistics.median(statistics.mean(v) for v in by_source.values()) if valid else None
        eligible=valid and score>=.05 and min(dg.values())>=-.02 and med>0
        ranking.append({'candidate':asdict(c),'valid':valid,'failed_start_ids':reasons,'Delta_G':dg,
            'A':score,'source_median_delta':med,'eligible':eligible,'seconds':seconds,'calls':calls})
    eligible=[r for r in ranking if r['eligible']]
    selected=None
    if eligible:
        top=max(r['A'] for r in eligible);near=[r for r in eligible if r['A']>=top-.02]
        fastest=min(r['seconds'] for r in near);tied=[r for r in near if r['seconds']<=fastest*1.03]
        selected=min(tied,key=lambda r:({'R':0,'P':1,'L':2}[r['candidate']['family']],
            r['candidate']['iterations'],r['candidate']['learning_rate'],r['candidate']['radius'] or 0))['candidate']
    result={'status':'selected_unique_candidate' if selected else 'no_repair_candidate_supported_on_dev',
        'selected':selected,'ranking':sorted(ranking,key=lambda r:r['A'] if r['A'] is not None else -math.inf,reverse=True),
        'frozen_unix':time.time(),'selection_code_sha256':sha256(__file__),
        'solver_sha256':sha256(ROOT/'code/v031/followup_refinement.py'),
        'dev_manifest_sha256':sha256(ROOT/'shared/manifests/refine_dev8.jsonl'),
        'confirm_used_for_selection':False,'heldout_used_for_selection':False,
        'timing_design':'same start assigned to same card for every candidate; compare all24 work times and report actual call counts'}
    # Atomic, never overwrite a previously frozen choice.
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.link(tmp,path);tmp.unlink()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--shards',type=int,required=True)
    ap.add_argument('--preflight',required=True);a=ap.parse_args()
    proof=json.loads(Path(a.preflight).read_text())
    if proof.get('status')!='passed':raise RuntimeError('Native preflight has not passed')
    if not 0<=a.rank<a.shards<=8:raise ValueError('Invalid static sharding')
    if proof['source_hashes']['followup_refinement.py']!=sha256(ROOT/'code/v031/followup_refinement.py'):
        raise RuntimeError('Accepted solver identity changed')
    frozen=provenance()
    session=OUT/'BATCH_IDENTITY.json'
    identity={'shards':a.shards,'code_hashes':frozen,'preflight_sha256':sha256(a.preflight)}
    tmp=OUT/f'.identity_rank{a.rank}.tmp';tmp.write_text(json.dumps(identity,sort_keys=True))
    try:os.link(tmp,session)
    except FileExistsError:pass
    finally:tmp.unlink()
    if json.loads(session.read_text())!=identity:raise RuntimeError('Frozen batch identity differs')
    cfg,model,clean,sim,data=init();baseline=(state_hash(model),state_hash(sim))
    save_json(OUT/f'worker_rank{a.rank}_progress.json',{'stage':'initialized','time':time.time(),'code_hashes':frozen})
    def already_done(path):
        if not path.exists():return False
        old=json.loads(path.read_text())
        if old.get('code_hashes')!=frozen:raise RuntimeError('Existing result has a different implementation')
        return True
    splits=json.loads((ROOT/'shared/manifests/FROZEN_SPLITS.json').read_text())
    lookup={r['start_id']:i for i,r in enumerate(data.records)}
    def execute(stage,tasklist,diagnostic=False):
        for number,(sid,candidate) in enumerate(tasklist):
            # Stable start assignment across candidates avoids cross-card candidate timing bias.
            stable=tasklist_start_order.index(sid)
            if stable%a.shards!=a.rank:continue
            folder=OUT/stage/('diagnostic' if diagnostic else candidate.name)/sid
            result_path=folder/'result.json'
            if already_done(result_path):continue
            folder.mkdir(parents=True,exist_ok=True);i=lookup[sid];r=data.records[i]
            selected_sources=(splits['diagnostic_arrays_dev_sources'] if stage in ('diagnostics','dev')
                              else splits['diagnostic_arrays_confirm_sources'])
            arrays=folder/'arrays' if r['source_id'] in selected_sources else None
            try:
                if diagnostic:
                    result=diagnose(model=model,clean=clean,sim=sim,data=data,index=i,
                        include_direction=r['source_id'] in splits['splits']['refine_dev8']['source_ids'][:4],
                        arrays=arrays,save_full_xi=r['source_id'] in splits['full_xi_saved_sources'])
                else:
                    result,predictions=run_path(model,clean,sim,data[i],candidate,arrays)
                    result.update(status='complete',source_id=r['source_id'])
                    if arrays is not None:
                        import numpy as np
                        arrays.mkdir(parents=True,exist_ok=True)
                        for k,c in enumerate(predictions,1):np.save(arrays/f'clean_k{k}.npy',c.numpy())
                result.update(card=os.environ['ASCEND_RT_VISIBLE_DEVICES'],rank=a.rank,shards=a.shards,
                    stage=stage,completed_unix=time.time(),solver_sha256=sha256(ROOT/'code/v031/followup_refinement.py'))
            except Exception as e:
                result={'status':'failed','start_id':sid,'source_id':r['source_id'],'stage':stage,
                        'candidate':None if candidate is None else asdict(candidate),'error':repr(e),'traceback':traceback.format_exc()}
                if diagnostic:
                    save_json(result_path,result);raise
            result['code_hashes']=frozen
            save_json(result_path,result)
            save_json(OUT/f'worker_rank{a.rank}_progress.json',{'stage':stage,'last':str(result_path),
                'status':result['status'],'memory':memory_stats(torch.device('npu:0')),'time':time.time()})
        done=OUT/f'{stage}_rank{a.rank}.done.json';save_json(done,{'rank':a.rank,'shards':a.shards,'code_hashes':frozen,'time':time.time()})
        # Bounded stage coordination; no accelerator operations while waiting.
        deadline=time.time()+24*3600
        while not all((OUT/f'{stage}_rank{r}.done.json').exists() for r in range(a.shards)):
            if time.time()>deadline:raise TimeoutError('I1 stage coordination timed out; no science retry')
            time.sleep(10)
    dev=splits['splits']['refine_dev8']['start_ids'];tasklist_start_order=dev
    diag=[sid for sid in dev if data.records[lookup[sid]]['initial_G']==4]
    execute('diagnostics',[(s,None) for s in diag],True)
    # Global barriers preserve the declared P-before-L order.
    for phase,families in [('dev_R',('U0','R')),('dev_P',('P',)),('dev_L',('L',))]:
        tasklist=[(s,c) for c in candidates() if c.family in families for s in dev]
        # Outputs have one canonical dev location; barrier has a distinct phase name.
        for number,(sid,candidate) in enumerate(tasklist):
            if dev.index(sid)%a.shards!=a.rank:continue
            folder=OUT/'dev'/candidate.name/sid
            if already_done(folder/'result.json'):continue
            folder.mkdir(parents=True,exist_ok=True);i=lookup[sid];r=data.records[i]
            arrays=folder/'arrays' if r['source_id'] in splits['diagnostic_arrays_dev_sources'] else None
            try:
                result,predictions=run_path(model,clean,sim,data[i],candidate,arrays)
                result.update(status='complete',source_id=r['source_id'],card=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
                              completed_unix=time.time(),stage='dev',rank=a.rank,shards=a.shards)
                if arrays is not None:
                    arrays.mkdir(parents=True,exist_ok=True)
                    for k,c in enumerate(predictions,1):np.save(arrays/f'clean_k{k}.npy',c.numpy())
            except Exception as e:
                result={'status':'failed','start_id':sid,'source_id':r['source_id'],'candidate':asdict(candidate),
                        'error':repr(e),'traceback':traceback.format_exc()}
            result['code_hashes']=frozen
            save_json(folder/'result.json',result)
            save_json(OUT/f'worker_rank{a.rank}_progress.json',{'stage':phase,'last':str(folder),
                'status':result['status'],'memory':memory_stats(torch.device('npu:0')),'time':time.time()})
        save_json(OUT/f'{phase}_rank{a.rank}.done.json',{'rank':a.rank,'shards':a.shards,'code_hashes':frozen,'time':time.time()})
        deadline=time.time()+24*3600
        while not all((OUT/f'{phase}_rank{r}.done.json').exists() for r in range(a.shards)):
            if time.time()>deadline:raise TimeoutError('I1 development phase barrier timed out')
            time.sleep(10)
    if a.rank==0:rank_and_freeze()
    while not (OUT/'selection.json').exists():time.sleep(2)
    choice=json.loads((OUT/'selection.json').read_text())
    if choice['selected']:
        selected=next(c for c in candidates() if asdict(c)==choice['selected'])
        confirm=splits['splits']['refine_confirm16']['start_ids'];tasklist_start_order=confirm
        execute('confirm',[(s,c) for c in (candidates()[0],candidates()[1],selected) for s in confirm])
    if provenance()!=frozen:raise RuntimeError('Implementation changed while batch was running')
    after=(state_hash(model),state_hash(sim))
    if baseline!=after:raise RuntimeError('Frozen inference state changed')
    save_json(OUT/f'worker_rank{a.rank}_complete.json',{'status':'dev_confirm_complete','frozen_state_unchanged':True,
        'selection_status':choice['status'],'heldout_pending':bool(choice['selected']),'time':time.time()})

if __name__=='__main__':main()

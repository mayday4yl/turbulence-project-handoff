"""Independent FP32 evaluation processes; no optimizer, sampler or training state."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,os,time
import numpy as np
import torch
from physical_bridge.model import DualHeadUNet
from physical_bridge.simulator import PiRNP2SSimulator
from v031.followup import CleanOnly
from v031.followup_refinement import candidates
from v031.data import StartDataset
from v031.runtime import save_json,sha256,verify_source,configure_npu,state_hash
from v031.training import synchronize,memory_stats
from i1_worker import run_path,quality

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    job=json.loads(Path(a.job).read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'shared/original_train_config.json').read_text());verify_source(cfg)
    import torch_npu
    configure_npu(cfg);torch.npu.set_per_process_memory_fraction(float(os.environ['FOLLOWUP_OWN_CAP_GIB'])/64,0)
    cp=Path(job['checkpoint']);assert sha256(cp)==job['checkpoint_sha256']
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    assert ck['global_update']==job['update']
    full=DualHeadUNet(**cfg['network'])
    task=job['task']
    model=CleanOnly(full) if task=='T1_DIRECT' else full
    model.load_state_dict(ck['network'],strict=True);del ck,full
    model=model.to('npu:0').eval().requires_grad_(False)
    sim=None if task=='T1_DIRECT' else PiRNP2SSimulator(cfg['asset_dir'],**cfg['physics'],device='npu:0').eval().requires_grad_(False)
    clean=CleanOnly(model).eval().requires_grad_(False) if task in ('V031_UNREFINED','I1_HELDOUT') else None
    data=StartDataset(job['dataset_root'],job.get('manifest','start_manifest.jsonl'))
    chosen=None
    if task=='I1_HELDOUT':
        selection=json.loads((ROOT/'I1_REFINE_REPAIR/selection.json').read_text())
        assert selection['selected'] and job['selection_sha256']==sha256(ROOT/'I1_REFINE_REPAIR/selection.json')
        allowed=[candidates()[0],candidates()[1],next(c for c in candidates() if asdict(c)==selection['selected'])]
        chosen=next(c for c in allowed if c.name==job['candidate'])
    frozen={'worker':sha256(__file__),'checkpoint':job['checkpoint_sha256'],'manifest':sha256(data.root/job.get('manifest','start_manifest.jsonl')),
        'task':task,'update':job['update'],'candidate':job.get('candidate'),'solver':sha256(ROOT/'code/v031/followup_refinement.py')}
    snapshot_dir=ROOT/'shared/source_snapshots';snapshot_dir.mkdir(exist_ok=True)
    snapshot_file=snapshot_dir/f"evaluate_followup_{frozen['worker']}.py"
    if not snapshot_file.exists():snapshot_file.write_bytes(Path(__file__).read_bytes())
    assert sha256(snapshot_file)==frozen['worker']
    before=(state_hash(model),None if sim is None else state_hash(sim))
    splits=json.loads((ROOT/'shared/manifests/FROZEN_SPLITS.json').read_text())
    preview=set(splits['legacy_preview_start_ids'])
    extra=set(splits['additional_preview_source_ids'])
    if job['subset']=='heldout32':extra={r['source_id'] for r in data.records[:8]}
    # The fixed64 subset need not contain any of the five frozen preview sources.
    # Resolve their first trajectory in the authoritative full manifest; this
    # neither adds subset samples nor substitutes a different preview trajectory.
    first_trajectory={s:min(r['trajectory_id'] for r in data.trajectories.values() if r['source_id']==s) for s in extra}
    count=0;started=time.perf_counter()
    for i,row in enumerate(data.records):
        if i%job['shards']!=job['rank']:continue
        if 'start_ids' in job and row['start_id'] not in job['start_ids']:continue
        if task=='I1_HELDOUT' and row['initial_G']==1:continue
        folder=out/row['start_id'];result_path=folder/'result.json'
        if result_path.exists():
            assert json.loads(result_path.read_text())['identity']==frozen;continue
        item=data[i];G=int(item['G'])
        arrays=folder/'arrays' if item['start_id'] in preview or (row['source_id'] in extra and row['trajectory_id']==first_trajectory[row['source_id']]) else None
        if arrays is not None:arrays.mkdir(parents=True,exist_ok=True)
        if task in ('V031_UNREFINED','I1_HELDOUT'):
            result,preds=run_path(model,clean,sim,item,chosen or candidates()[0],arrays)
        else:
            y=item['observation'].unsqueeze(0).to('npu:0');gt=item['clean'].unsqueeze(0).to('npu:0');state=y
            metrics=[];preds=[];calls={'network':0,'F':0,'VJP':0};start=time.perf_counter()
            with torch.no_grad():
                for k,s in enumerate((G,) if task=='T1_DIRECT' else range(G,0,-1),1):
                    p=model(state,state.new_tensor([float(s)]));calls['network']+=1
                    c=p if task=='T1_DIRECT' else p.clean
                    lr=None
                    if sim is not None:
                        reconstructed=sim.render(c,p.realization,G).output;calls['F']+=1
                        lr=float((reconstructed-y).square().mean())
                    metrics.append({'k':k,'G':G,'current_g':s,**quality(c,gt),'LR':lr})
                    preds.append(c.cpu());state=c
                synchronize(y.device)
            result={'start_id':item['start_id'],'trajectory_id':item['trajectory_id'],'metrics':metrics,'input':quality(y,gt),
                'calls':calls,'seconds':time.perf_counter()-start,'weights':'raw','checkpoint_update':job['update']}
        if arrays is not None:
            for k,c in enumerate(preds,1):np.save(arrays/f'clean_k{k}.npy',c.numpy())
            np.save(arrays/'GT.npy',item['clean'].numpy());np.save(arrays/'Y.npy',item['observation'].numpy())
        result.update(source_id=row['source_id'],task=task,subset=job['subset'],status='complete',identity=frozen,
            card=os.environ['ASCEND_RT_VISIBLE_DEVICES'],completed_unix=time.time())
        save_json(result_path,result);count+=1
        save_json(out/f'progress_rank{job["rank"]}.json',{'last':row['start_id'],'new_starts':count,'time':time.time()})
    assert before==(state_hash(model),None if sim is None else state_hash(sim))
    save_json(out/f'complete_rank{job["rank"]}.json',{'status':'complete','identity':frozen,'new_starts':count,
        'seconds':time.perf_counter()-started,'frozen_network_simulator_unchanged':True,'memory':memory_stats(torch.device('npu:0'))})

if __name__=='__main__':main()

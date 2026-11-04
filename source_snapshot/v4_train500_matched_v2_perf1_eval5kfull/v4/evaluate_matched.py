"""Streaming full-float metrics, same-rollout progression, fixed visual tensors."""
from collections import defaultdict
from pathlib import Path
import hashlib,json,math,os,sys,time,shutil
import numpy as np
import torch
from .matched import ROOT,eta_batch,seed_for,SEED
from .runtime import save_json,sha256
sys.path.insert(0,str(ROOT/'scripts'))
from metrics import Metrics,SPEC

KEYS=['psnr','ssim','lpips','mse','path_clean_MSE','path_reconstruction_MSE',
      'path_weighted_reconstruction_MSE','path_total','psnr_first','delta_prog_PSNR']

def output_digest(tensor):
    array=np.ascontiguousarray(tensor.detach().cpu().numpy(),dtype='<f4')
    if array.shape!=(3,256,256) or not np.isfinite(array).all():raise ValueError('Invalid FP32 image')
    return hashlib.sha256(array.tobytes(order='C')).hexdigest(),array


def aggregates(rows):
    def mean(items):
        return {k:float(np.mean([r[k] for r in items],dtype=np.float64)) for k in KEYS if all(k in r for r in items)}
    sources=defaultdict(list);groups=defaultdict(list)
    for r in rows:sources[r['source_id']].append(r);groups[str(r['G'])].append(r)
    return {'overall':mean(rows),'per_source':{s:mean(v) for s,v in sorted(sources.items())},
            'per_G':{g:mean(v) for g,v in sorted(groups.items())},'start_count':len(rows),'source_count':len(sources)}


class Evaluator:
    def __init__(self,store,protocol_hash,environment_identity):
        self.store=store;self.protocol_hash=protocol_hash;self.environment_identity=environment_identity
        freeze=ROOT/'protocol/FROZEN.json'
        self.execution_hash=json.loads(freeze.read_text()).get('execution_hash') if freeze.exists() else 'UNFROZEN_ACCEPTANCE_ONLY'
        self.baseline={}

    def derive_subset(self,full_destination,rows,destination):
        """Reuse committed values only when every network batch is identical."""
        full_destination=Path(full_destination);destination=Path(destination)
        commit=json.loads((full_destination/'COMMIT.json').read_text())
        for name,digest in commit['file_sha256'].items():
            if sha256(full_destination/name)!=digest:raise ValueError('Full evaluation changed')
        all_rows=[json.loads(line) for line in (full_destination/'per_start.jsonl').read_text().splitlines()]
        position={r['start_id']:i for i,r in enumerate(all_rows)}
        selected=[r['start_id'] for r in rows]
        if len(selected)%2:raise ValueError('Subset batch grouping changed')
        for i in range(0,len(selected),2):
            a,b=position[selected[i]],position[selected[i+1]]
            if a%2 or b!=a+1:raise ValueError('Subset does not preserve frozen batch2 groups')
        if destination.exists():
            existing=json.loads((destination/'COMMIT.json').read_text())
            if existing.get('derived_from_commit_sha256')!=sha256(full_destination/'COMMIT.json'):
                raise ValueError('Existing subset provenance mismatch')
            return json.loads((destination/'aggregate.json').read_text())
        work=destination.with_name(destination.name+'.partial');work.mkdir(parents=True,exist_ok=False)
        subset=[all_rows[position[sid]] for sid in selected]
        baseline={r['start_id']:r for r in json.loads((full_destination/'baseline_per_start.json').read_text())}
        baseline=[baseline[sid] for sid in selected]
        (work/'per_start.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in subset))
        save_json(work/'baseline_per_start.json',baseline)
        original=json.loads((full_destination/'aggregate.json').read_text())
        aggregate={**aggregates(subset),'baseline':aggregates(baseline),
                   **{k:original[k] for k in ['update','weights','method','checkpoint_hash']},
                   'elapsed_seconds':0.,'inference_reused':True}
        save_json(work/'aggregate.json',aggregate)
        for sid in selected:
            for source in (full_destination/'fixed_fp32').glob(sid.replace('::','_')+'_*.npy'):
                target=work/'fixed_fp32';target.mkdir(exist_ok=True);shutil.copy2(source,target/source.name)
        save_json(work/'COMMIT.json',{'checkpoint_hash':commit['checkpoint_hash'],
                  'protocol_hash':commit['protocol_hash'],'metrics_spec':SPEC,
                  'derived_from_commit_sha256':sha256(full_destination/'COMMIT.json'),
                  'file_sha256':{str(p.relative_to(work)):sha256(p) for p in work.rglob('*') if p.is_file()}})
        work.rename(destination);return aggregate

    @torch.no_grad()
    def evaluate(self,objective,rows,destination,*,update,weights,checkpoint_hash,metrics=None):
        destination=Path(destination)
        if destination.exists():
            identity=json.loads((destination/'COMMIT.json').read_text())
            if identity['checkpoint_hash']!=checkpoint_hash or identity['protocol_hash']!=self.protocol_hash:
                raise ValueError('Existing evaluation belongs to another protocol/checkpoint')
            for name,digest in identity['file_sha256'].items():
                if sha256(destination/name)!=digest:raise ValueError('Existing evaluation hash mismatch')
            return json.loads((destination/'aggregate.json').read_text())
        work=destination.with_name(destination.name+'.partial')
        if work.exists():raise RuntimeError('Uncommitted evaluation evidence exists; review before replay')
        work.mkdir(parents=True)
        device=next(objective.network.parameters()).device
        metrics=Metrics(device) if metrics is None else metrics
        was_training=objective.training;objective.eval()
        all_rows=[];baseline_rows=[];begin=time.monotonic()
        try:
            with (work/'per_start.jsonl').open('w') as output_file:
                for offset in range(0,len(rows),2):
                    chunk=rows[offset:offset+2];C,Y,G=self.store.batch(chunk,device)
                    prediction=objective(C,Y,G,eta=eta_batch(chunk,device))
                    values=metrics(prediction['final_clean'],C)
                    first_mse=(prediction['first_clean']-C).square().flatten(1).mean(1).cpu().tolist()
                    for i,r in enumerate(chunk):
                        if r['start_id'] not in self.baseline:
                            self.baseline[r['start_id']]=metrics(Y[i:i+1],C[i:i+1])[0]
                        baseline_rows.append({**{k:r[k] for k in ('source_id','trajectory_id','start_id','split','G')},**self.baseline[r['start_id']]})
                        if first_mse[i]<=0:raise FloatingPointError('Invalid first prediction MSE')
                        first_psnr=-10*math.log10(first_mse[i])
                        final_hash,final_array=output_digest(prediction['final_clean'][i])
                        first_hash,_=output_digest(prediction['first_clean'][i])
                        lc=float(prediction['path_clean'][i]);lr=float(prediction['path_reconstruction'][i])
                        row={**{k:r[k] for k in ('source_id','trajectory_id','start_id','split','G')},**values[i],
                             'method':objective.group,'update':update,'epoch':update/250,'exposures':32*update,
                             'weights':weights,'checkpoint_hash':checkpoint_hash,'protocol_hash':self.protocol_hash,
                             'environment_identity':self.environment_identity,'execution_hash':self.execution_hash,'shape':[3,256,256],'dtype':'little_endian_float32',
                             'eta_seed':seed_for(f'eval|{SEED}|{r["start_id"]}'),
                             'path_clean_MSE':lc,'path_reconstruction_MSE':lr,
                             'path_weighted_reconstruction_MSE':.2*lr,'path_total':float(prediction['path_losses'][i]),
                             'psnr_first':first_psnr,'psnr_final':values[i]['psnr'],
                             'delta_prog_PSNR':values[i]['psnr']-first_psnr,
                             'output_hash_first':first_hash,'output_hash_final':final_hash}
                        if any(not math.isfinite(row[k]) for k in KEYS):raise FloatingPointError('Nonfinite evaluation record')
                        all_rows.append(row);output_file.write(json.dumps(row,allow_nan=False)+'\n')
                        if r['start_id'] in self.store.visual_ids:
                            target=work/'fixed_fp32';target.mkdir(exist_ok=True)
                            stem=r['start_id'].replace('::','_')
                            np.save(target/f'{stem}_output.npy',final_array,allow_pickle=False)
                            for label,value in [('clean',C[i]),('input',Y[i])]:
                                _,array=output_digest(value);np.save(target/f'{stem}_{label}.npy',array,allow_pickle=False)
                    del C,Y,G,prediction,values
            aggregate=aggregates(all_rows);aggregate['baseline']=aggregates(baseline_rows)
            aggregate.update(update=update,weights=weights,method=objective.group,
                             elapsed_seconds=time.monotonic()-begin,checkpoint_hash=checkpoint_hash)
            save_json(work/'aggregate.json',aggregate)
            save_json(work/'baseline_per_start.json',baseline_rows)
            hashes={str(p.relative_to(work)):sha256(p) for p in work.rglob('*') if p.is_file()}
            save_json(work/'COMMIT.json',{'checkpoint_hash':checkpoint_hash,'protocol_hash':self.protocol_hash,
                                       'file_sha256':hashes,'metrics_spec':SPEC,'execution_hash':self.execution_hash})
            work.rename(destination)
            return aggregate
        finally:
            objective.train(was_training)

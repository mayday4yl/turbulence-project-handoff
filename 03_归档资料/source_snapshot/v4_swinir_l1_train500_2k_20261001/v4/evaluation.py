"""One frozen FP32 evaluation path; public refinement is always per-sample."""
import json
import math
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from .core import mse
from .inference import infer
from .training import GROUPS, evaluation_seed, evaluation_groups, rng_state, set_rng_state
from .runtime import save_json


def predict(model,simulator,Y,G,records,group,refinement):
    if group=='D0':
        if refinement:
            raise ValueError('D0 has no refinement')
        with torch.no_grad(),torch.autocast(device_type=Y.device.type,enabled=False):
            pred = model(torch.cat((Y,Y),1),torch.stack((G,torch.zeros_like(G)),1))
        return {'clean':pred.clean,'first':pred.clean,'prediction_count':len(G),
                'transition_count':0,'refinement_sample_count':0,'trace':[]}
    first = []
    def capture(module,args,result):
        if not first:
            first.append(result.clean.detach())
    hook = model.register_forward_hook(capture)
    try:
        result = infer(model,simulator,Y,G,noise_seeds=[evaluation_seed(r['start_id']) for r in records],
                       T0=GROUPS[group],refinement=refinement,retain_trace=True)
    finally:
        hook.remove()
    result['first'] = first[0]
    return result


def psnr(mse_value):
    # The paired data/continuous model outputs must not be exact matches in this protocol.
    if not math.isfinite(mse_value) or mse_value <= 0:
        raise FloatingPointError(f'Cannot record finite PSNR for MSE={mse_value}')
    return -10*math.log10(mse_value)


def summarize(rows):
    by_source_g = defaultdict(list)
    for row in rows:
        by_source_g[(row['source_id'],row['G'])].append(row)
    names = ['mse','psnr','input_psnr','first_psnr','gain_over_input','progressive_gain']
    source_g = [{'source_id':sid,'G':g,**{k:sum(r[k] for r in rr)/len(rr) for k in names}}
                for (sid,g),rr in sorted(by_source_g.items())]
    by_g = {str(g):{k:sum(r[k] for r in source_g if r['G']==g)/sum(r['G']==g for r in source_g)
                    for k in names} for g in (1,2,3,4)}
    overall = {k:sum(by_g[str(g)][k] for g in (1,2,3,4))/4 for k in names}
    return {'starts':len(rows),'sources':len({r['source_id'] for r in rows}),
            'aggregation':'mean_realization_then_source_then_equal_G','overall':overall,
            'by_G':by_g,'source_G':source_g}


def save_panel(path,C,Y,first,final):
    # Clamping here is strictly for display; all metrics use untouched floating outputs.
    panels = [x.detach().cpu().clamp(0,1).permute(1,2,0).numpy() for x in (C,Y,first,final)]
    image = np.concatenate(panels,axis=1)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(image*255).astype(np.uint8)).save(path)


def evaluate(model,simulator,store,records,*,group,weights,refinement,protocol,E,run_dir,
             list_name,noise_generator,checkpoint_identity):
    rank=dist.get_rank();world=dist.get_world_size();device=next(model.parameters()).device
    identifier=f'E{E:05d}_{list_name}_{weights}_ref{int(refinement)}'
    out=Path(run_dir)/'evaluation'/identifier
    out.mkdir(parents=True,exist_ok=True)
    final_path=out/'results.json'
    # Resume skips only complete products bound to the same protocol and checkpoint.
    skip=False
    if final_path.exists():
        old=json.loads(final_path.read_text())
        skip=(old.get('protocol_hash')==protocol['protocol_hash'] and
              old.get('checkpoint_identity')==checkpoint_identity and old.get('status')=='COMPLETE')
    flag=torch.tensor(int(skip),device=device);dist.broadcast(flag,0)
    if bool(flag):
        return json.loads(final_path.read_text())['summary'] if rank==0 else None
    state=rng_state(noise_generator);was_training=model.training
    model.eval();simulator.eval()
    rows=[];execution=[];started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for batch_index,batch in enumerate(evaluation_groups(records,protocol['eval_network_batch'],world)):
            if batch['rank']!=rank:continue
            rr=batch['records'];C,Y,G=store.batch(rr,device)
            begin=time.monotonic();torch.cuda.synchronize()
            result=predict(model,simulator,Y,G,rr,group,refinement)
            torch.cuda.synchronize();elapsed=time.monotonic()-begin
            errors=mse(result['clean'],C).cpu().tolist()
            first_errors=mse(result['first'],C).cpu().tolist()
            input_errors=mse(Y,C).cpu().tolist()
            for j,r in enumerate(rr):
                p,fp,ip=psnr(errors[j]),psnr(first_errors[j]),psnr(input_errors[j])
                rows.append({k:r[k] for k in ('start_id','trajectory_id','source_id','split','G')} |
                            {'mse':errors[j],'psnr':p,'first_psnr':fp,'input_psnr':ip,
                             'gain_over_input':p-ip,'progressive_gain':p-fp,
                             'eta_seed':evaluation_seed(r['start_id'])})
                if list_name in ('train64','val64') and records.index(r)<8:
                    save_panel(out/'visuals'/f"{records.index(r):02d}_C_Y_first_final.png",
                               C[j],Y[j],result['first'][j],result['clean'][j])
            execution.append({'batch_index':batch_index,'rank':rank,'start_ids':[r['start_id'] for r in rr],
                              'actual_network_batch':len(rr),'actual_refine_batch':1 if refinement else None,
                              'seconds':elapsed,'trace':result['trace'],
                              'prediction_count':result['prediction_count'],
                              'transition_count':result['transition_count'],
                              'refinement_sample_count':result['refinement_sample_count']})
            if refinement:
                # Preserve scalar candidates/update and best-selection evidence, not enormous xi tensors.
                for record in result['refinements']:
                    execution[-1].setdefault('refinement',[]).append({
                        'initial_objective':record.initial_objective,'best_objective':record.best_objective,
                        'best_iterations':record.best_iterations,'candidates':record.candidates,
                        'optimizer':record.optimizer,'invariance':record.invariance})
            del result,C,Y,G
            if len(execution)%20==0:
                save_json(out/f'progress_rank{rank}.json',{'batches':len(execution),'starts':len(rows),
                          'elapsed_seconds':time.monotonic()-started,'last_start':rr[-1]['start_id']})
        save_json(out/f'rank{rank}.json',{'rows':rows,'execution':execution,
                  'seconds':time.monotonic()-started,'peak_GiB':torch.cuda.max_memory_allocated()/2**30})
        dist.barrier()
        if rank==0:
            pieces=[json.loads((out/f'rank{i}.json').read_text()) for i in range(world)]
            merged={r['start_id']:r for p in pieces for r in p['rows']}
            if len(merged)!=len(records) or set(merged)!={r['start_id'] for r in records}:
                raise RuntimeError('Evaluation coverage mismatch or duplicate starts')
            ordered=[merged[r['start_id']] for r in records]
            summary=summarize(ordered)
            save_json(final_path,{'status':'COMPLETE','group':group,'E':E,'weights':weights,
                'refinement':refinement,'refinement_execution':'per_sample_serial' if refinement else 'off',
                'protocol_hash':protocol['protocol_hash'],'checkpoint_identity':checkpoint_identity,
                'list_name':list_name,'data_manifest_hashes':store.dataset.manifest_hashes,
                'grouping_manifest_sha256':protocol['evaluation_grouping_sha256'],
                'nominal_network_batch':protocol['eval_network_batch'],'nominal_refine_batch':1,
                'precision':protocol['precision'],'rows':ordered,'summary':summary,
                'seconds':max(p['seconds'] for p in pieces),'peak_GiB':max(p['peak_GiB'] for p in pieces),
                'execution':[e for p in pieces for e in p['execution']]})
            return summary
    finally:
        model.train(was_training);simulator.eval();set_rng_state(state,noise_generator)

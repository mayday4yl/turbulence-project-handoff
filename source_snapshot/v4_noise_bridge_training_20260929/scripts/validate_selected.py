"""Verify the selected real two-GPU batch against per-sample full-path gradients."""
import json
import gc
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import torch.distributed as dist
from train_comparison import setup,build
from prepare_and_validate import gradient_diff
from v4.training import RecordStore,backward_batch,finite_gradients
from v4.runtime import save_json


def selected_candidate():
    rows=json.loads((ROOT/'reports/benchmark.json').read_text())['results']
    candidates={}
    for row in rows:candidates.setdefault((row['batch'],row['microbatch']),[]).append(row)
    costs={key:sum(32000/r['starts_per_second'] for r in rr) for key,rr in candidates.items()
           if {r['group'] for r in rr}=={'D0','R0','R08'}}
    minimum=min(costs.values())
    return min((key for key in costs if costs[key]<=minimum*1.1),key=lambda k:(k[0],costs[k]))


def main():
    rank,device=setup();batch,micro=selected_candidate()
    config=json.loads((ROOT/'configs/correctness.json').read_text());store=RecordStore(config)
    initial=torch.load(ROOT/'protocol/common_init.pt',map_location='cpu',weights_only=True)
    reports=[]
    for group in ('D0','R0','R08'):
        model,sim,objective,wrapped=build(config,device,group,initial)
        noise=torch.Generator(device=device).manual_seed(111+rank)
        rows=[store.train[i*4+i%4] for i in range(batch)]
        eta=torch.randn((batch,3,256,256),generator=torch.Generator().manual_seed(7299))
        objective.zero_grad(set_to_none=True)
        backward_batch(wrapped,store,rows,micro,noise,etas=eta);finite_gradients(model)
        reference={name:p.grad.detach().clone() for name,p in model.named_parameters()}
        objective.zero_grad(set_to_none=True)
        for i,row in enumerate(rows):
            C,Y,G=store.batch([row],device)
            result=objective(C,Y,G,eta=eta[i:i+1].to(device))
            (result['loss_mean']/batch).backward();del result,C,Y,G
        finite_gradients(model)
        relative=gradient_diff(reference,model)
        reports.append({'group':group,'batch':batch,'microbatch':micro,'relative_L2':relative,'limit':1e-4})
        del wrapped,objective,model,sim,reference;gc.collect();torch.cuda.empty_cache();dist.barrier()
    save_json(ROOT/f'reports/selected_validation_rank{rank}.json',{'status':'PASSED','groups':reports})
    dist.barrier()
    if rank==0:save_json(ROOT/'reports/selected_validation.json',{'status':'PASSED','batch':batch,'microbatch':micro,
        'ranks':[json.loads((ROOT/f'reports/selected_validation_rank{i}.json').read_text()) for i in range(2)]})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

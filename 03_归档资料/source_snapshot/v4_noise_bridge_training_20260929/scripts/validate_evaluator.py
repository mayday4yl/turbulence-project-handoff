"""Exercise the actual distributed evaluation writer and its RNG isolation before launch."""
from pathlib import Path
import json
import sys
import gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import torch.distributed as dist
from train_comparison import setup,build
from v4.training import RecordStore,rng_state,make_ema
from v4.evaluation import evaluate
from v4.runtime import save_json,sha256


def main():
    rank,device=setup();config=json.loads((ROOT/'configs/correctness.json').read_text());store=RecordStore(config)
    initial=torch.load(ROOT/'protocol/common_init.pt',map_location='cpu',weights_only=True)
    protocol={'protocol_hash':'TEMPORARY_EVALUATOR_VALIDATION','eval_network_batch':2,
              'precision':'FP32_autocast_and_TF32_off','evaluation_grouping_sha256':'temporary_fixture'}
    for group in ('D0','R0','R08'):
        model,sim,obj,wrapped=build(config,device,group,initial);ema=make_ema(model)
        noise=torch.Generator(device=device).manual_seed(584+rank);before=rng_state(noise)
        for weight,network in [('raw',model),('ema',ema)]:
            result=evaluate(network,sim,store,store.val[:8],group=group,weights=weight,refinement=False,
                protocol=protocol,E=0,run_dir=ROOT/'reports/evaluator_fixture'/group,list_name='fixture8',
                noise_generator=noise,checkpoint_identity=sha256(ROOT/'protocol/common_init.pt'))
            if rank==0:assert result['starts']==8 and set(result['by_G'])=={'1','2','3','4'}
        after=rng_state(noise)
        for key in ('cpu','cuda','eta'):assert torch.equal(before[key],after[key])
        del wrapped,obj,model,sim,ema;gc.collect();torch.cuda.empty_cache();dist.barrier()
    if rank==0:save_json(ROOT/'reports/evaluator_validation.json',{'status':'PASSED',
        'real_distributed_writer':True,'groups':['D0','R0','R08'],'weights':['raw','ema'],
        'fixture_starts':8,'evaluation_batch':2,'CPU_CUDA_eta_rng_unchanged':True})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

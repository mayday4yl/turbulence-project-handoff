#!/usr/bin/env python3
"""Zero-optimizer diagnostic: fresh/rebuilt DDP vs fixed constructor buckets."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from check_distributed_gradients import backward_pass,gradient_comparison
from train_worker import make_engine,resolved_identity
from v031.data import StartDataset
from v031.distributed_runtime import configure_local_rank
from v031.runtime import save_json,state_hash,verify_source
import v031.training as training
from v031.training import capture_rank_rng,restore_rank_rng,load_training_checkpoint

ROOT=Path('/data2/yanglin/v031_overfit100_g4321_n4')

def compare(a,b):
    result=gradient_comparison(a,b)
    result['exact']=all(torch.equal(a[k],b[k]) for k in a)
    result['max_abs']=max(float((a[k].double()-b[k].double()).abs().max()) for k in a)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--only-default',action='store_true');parser.add_argument('--deterministic-operators',action='store_true')
    args=parser.parse_args();args.microbatch=1;args.mode='performance-smoke'
    rank=int(os.environ['LOCAL_RANK']);config=json.loads((ROOT/'configs/stage2_candidate_mb1.json').read_text())
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    runtime=configure_local_rank(config,rank);verify_source(config)
    if args.deterministic_operators:torch.use_deterministic_algorithms(True)
    runtime['torch_deterministic_algorithms']=torch.are_deterministic_algorithms_enabled()
    dist.init_process_group('hccl',timeout=timedelta(minutes=30));dist.barrier()
    dataset=StartDataset(config['dataset_root']);identity=resolved_identity(config,dataset)
    checkpoint=ROOT/'artifacts/stage2_resume_mb1_hccl02/smoke_resume_000001.pt'
    result={'rank':rank,'status':'running','optimizer_updates':0,'checkpoint':str(checkpoint),'runtime':runtime,'hccl_deterministic':os.environ.get('HCCL_DETERMINISTIC'),'variants':{}}
    save_json(args.output/f'result_rank{rank}.json',result)
    try:
        for unused in ([False] if args.only_default else [False,True]):
            def constructor(*a,**kw):
                kw['find_unused_parameters']=unused
                return DistributedDataParallel(*a,**kw)
            with patch.object(training,'DistributedDataParallel',side_effect=constructor):
                engine=make_engine(config,dataset,identity,args,rank)
            load_training_checkpoint(engine,checkpoint,run_until_update=2)
            network_hash=state_hash(engine.network);sim_hash=state_hash(engine.rollout.simulator)
            rng=capture_rank_rng(engine.device);indices=engine.sampler.next_indices()
            local=indices[rank*8:(rank+1)*8]
            data={'find_unused_parameters':unused,'start_ids':[dataset.records[i]['start_id'] for i in indices],'passes':[]}
            result['variants'][str(unused)]=data
            reference=None;previous=None
            for number in range(3 if not unused else 2):
                restore_rank_rng(rng,engine.device)
                before=bool(engine.module._has_rebuilt_buckets)
                metrics,grads=backward_pass(engine,local,ddp=True)
                entry={'pass':number,'rebuild_before':before,'rebuild_after':bool(engine.module._has_rebuilt_buckets),'ddp':engine.module._get_ddp_logging_data()}
                if reference is not None:entry['versus_fresh']=compare(reference,grads)
                if previous is not None:entry['versus_previous']=compare(previous,grads)
                if number==0:reference=grads
                previous=grads
                data['passes'].append(entry)
                save_json(args.output/f'result_rank{rank}.json',result)
            assert network_hash==state_hash(engine.network) and sim_hash==state_hash(engine.rollout.simulator)
            assert engine.optimizer_step_attempts==engine.optimizer_step_calls==0
            data['network_simulator_unchanged']=True
            del engine,reference,previous,grads
            gc.collect();torch.npu.empty_cache();dist.barrier()
        result['status']='diagnostic_complete'
        save_json(args.output/f'result_rank{rank}.json',result);dist.barrier()
        if rank==0:save_json(args.output/'results.json',{'status':'diagnostic_complete','optimizer_updates':0,'ranks':[json.loads((args.output/f'result_rank{i}.json').read_text()) for i in range(4)]})
    finally:dist.destroy_process_group()

if __name__=='__main__':main()

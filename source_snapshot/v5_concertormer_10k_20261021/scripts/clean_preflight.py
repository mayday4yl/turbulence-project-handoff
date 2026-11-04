"""One disposable B48 update, official zero-update Val1000, full resume check."""
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from runtime.factories import training_components
from runtime.controller import ROOT, atomic_json, file_sha
from training.runner import RunConfig, load_initial_model
from training.core import unroll_training, network_context
from training.evaluation import evaluate_full_validation, synchronize, memory_stats
from training.state import UpdateBatchSampler, UpdateLRScheduler, gather_rank_state, atomic_torch_save
from scripts.benchmark_distributed import checkpoint_compatibility


HERE = ROOT/'preparation/clean_only_20261021'
INITIAL = ROOT/'weights/clean_only_GoPro_U0.pt'
RUNTIME = dict(world_size=6, microbatch=2, accumulation_steps=4, effective_batch=48,
    network_precision='bf16', activation_checkpointing=False, checkpoint_policy='all',
    talking_backend='conv2d_equivalent', psf_backend='native')


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    data=ROOT/'data/v5_places_c50_s2000_v250_t250_g4_r4_20261021'
    config=RunConfig(method='D0',objective='clean_only', output_dir=str(HERE/'temporary_update'),
        initial_checkpoint=str(INITIAL),readiness_report='preflight_only',**RUNTIME,
        experiment_metadata={'dataset_root':str(data),'data_ready_sha256':file_sha(data/'DATA_READY.json')})
    components=training_components(config)
    rank=dist.get_rank();device=components['device'];model=components['model']
    if rank==0:
        if INITIAL.exists():raise FileExistsError('Refusing to recreate clean U0')
        official=ROOT/'weights/Concertormer_GoPro_official.pth'
        if file_sha(official)!='e6a55a1f9ae53c818d2bef9280d9b428973d8776df008b151df226ae94137b3b':
            raise RuntimeError('Official GoPro checkpoint hash mismatch')
        provenance=model.load_official_checkpoint(official)
        atomic_torch_save({'model':model.state_dict(),'provenance':provenance,
                          'objective':'clean_only','formal_optimizer_updates':0},INITIAL)
        atomic_json(INITIAL.with_suffix('.json'),{'sha256':file_sha(INITIAL),'provenance':provenance,
            'inputs':['Y4'],'outputs':['clean'],'loss':'clean MAE weight 1.0','time_modules':False,
            'raw_modules':False,'simulator':False})
    dist.barrier()
    load_initial_model(model,INITIAL);model.to(device)
    if any(not k.startswith('backbone.') for k in model.state_dict()):raise RuntimeError('Extra D0 modules')
    batch=next(iter(DataLoader(components['train_dataset'],batch_size=2)))
    x=batch['observation'].to(device)
    model.eval()
    with torch.no_grad(), network_context(device,'bf16'):
        a=model(x);b=model.backbone(x)[-1]
    error=(a.float()-b.float()).abs().max().item()
    if not torch.allclose(a,b,rtol=1e-5,atol=1e-6):raise RuntimeError('Official RGB parity failed')
    del a,b,x,batch
    baseline=evaluate_full_validation(model,None,components['val_dataset'],device=device,method='D0',
        output_dir=HERE/'zero_update_baseline',update=0,network_precision='bf16',batch_size=4,
        workers=2,extended_metrics=True,objective='clean_only')
    if rank==0:print(json.dumps({'event':'zero_update_val_complete','result':baseline}),flush=True)
    model.train();model=DDP(model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
    groups=model.module.optimizer_parameter_groups()
    optimizer=torch.optim.AdamW(groups,lr=5e-5,weight_decay=1e-4)
    scheduler=UpdateLRScheduler(optimizer,max_updates=10000,warmup_updates=300)
    sampler=UpdateBatchSampler(8000,seed=20261021,microbatch=2,accumulation_steps=4,world_size=6,rank=rank,max_updates=1)
    loader=DataLoader(components['train_dataset'],batch_sampler=sampler,num_workers=0)
    scheduler.step_for_update(1);optimizer.zero_grad(set_to_none=True);started=time.monotonic();losses=[]
    before=model.module.backbone.ending.weight.detach().clone()
    for i,batch in enumerate(loader):
        with model.no_sync() if i<3 else nullcontext():
            result=unroll_training(model,None,batch['observation'].to(device),batch['clean'].to(device),
                method='D0',objective='clean_only',path_seed=0,network_precision='bf16')
            if not torch.isfinite(result.loss).item():raise FloatingPointError('Nonfinite clean loss')
            (result.loss/4).backward();losses.append(result.loss.item())
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:raise RuntimeError(f'Missing gradients: {missing}')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if norm.item()<=0:raise RuntimeError('Zero global gradient')
    optimizer.step();synchronize(device)
    if torch.equal(before,model.module.backbone.ending.weight):raise RuntimeError('RGB weights did not update')
    ranks=gather_rank_state({'rank':rank,'clean_mae':sum(losses)/4,'grad_norm':norm.item(),
        'official_rgb_max_abs_error':error,'seconds':time.monotonic()-started,**memory_stats(device)})
    args=SimpleNamespace(output=str(HERE/'D0_READY.json'),method='D0',initial=str(INITIAL),microbatch=2,
        effective_batch=48,checkpointing=False,checkpoint_policy='all',talking_backend='conv2d_equivalent',
        precision='bf16',seed=20261021,keep_checkpoint_probe=True,objective='clean_only')
    checkpoint=checkpoint_compatibility(model,None,optimizer,scheduler,args,device,rank,6,4,1,
        {'path':str(INITIAL),'sha256':file_sha(INITIAL)})
    if rank==0:
        atomic_json(HERE/'D0_READY.json',{'status':'READY_FOR_TRAINING','method':'D0','validated_methods':['D0'],
            'objective':'clean_only','initial_checkpoint_sha256':file_sha(INITIAL),'runtime':RUNTIME,
            'formal_optimizer_updates':0,'temporary_updates_discarded':1,'full_validation':baseline,
            'rank_checks':ranks,'checkpoint_compatibility':checkpoint,'test_used':False})
        print('CLEAN_ONLY_D0_PREFLIGHT_PASS',flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

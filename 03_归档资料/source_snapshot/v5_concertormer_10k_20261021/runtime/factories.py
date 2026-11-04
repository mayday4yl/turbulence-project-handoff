"""Construct only the selected V5 experiment and the accepted FP32 simulator."""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import os
import torch
import torch.distributed as dist
from models import build_model
from data_pipeline.dataset import V5Dataset
from .environment import configure, make_simulator, ROOT


def training_components(config, device_or_none=None):
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    device=device_or_none or f'npu:{local_rank}'
    environment=configure(device)
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=config.world_size:
        raise ValueError('Launcher and selected world_size disagree')
    if world == 6 and os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '2,3,4,5,6,7':
        raise ValueError('Six-card execution must explicitly map de36 physical devices 2 through 7')
    if world>1 and not dist.is_initialized():
        dist.init_process_group(backend='hccl' if str(device).startswith('npu') else 'gloo',
                                timeout=timedelta(minutes=10))
    metadata=config.experiment_metadata or {}
    data_root=Path(metadata.get('dataset_root',ROOT/'data/v5_places_c50_s2000_v250_t250_g4_r4_20261021'))
    ready_path=data_root/'DATA_READY.json'
    if not ready_path.is_file():
        raise RuntimeError('Full V5 dataset not READY')
    ready=json.loads(ready_path.read_text())
    if ready['status']!='DATA_READY' or ready['sample_counts']!={'train':8000,'val':1000,'test':1000}:
        raise RuntimeError('V5 dataset counts/readiness mismatch')
    expected=metadata.get('data_ready_sha256')
    if expected is None or hashlib.sha256(ready_path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Run must bind the exact DATA_READY identity')
    checkpointing=config.activation_checkpointing
    from models.clean_concertormer import build_clean_model
    builder = build_clean_model if config.objective == 'clean_only' else build_model
    model=builder(activation_checkpointing=checkpointing,checkpoint_policy=config.checkpoint_policy,
                      seed=config.seed,talking_backend=config.talking_backend)
    simulator = None
    if config.objective != 'clean_only':
        simulator=make_simulator(device)
        if config.psf_backend == 'adjoint':
            from optimizations.psf_adjoint_20261021.adjoint import install
            simulator=install(simulator)
        simulator.eval()
        for parameter in simulator.parameters():parameter.requires_grad_(False)
    train=V5Dataset(data_root,'train',require_ready=True,allow_test=False)
    val=V5Dataset(data_root,'val',require_ready=True,allow_test=False)
    return {'model':model,'simulator':simulator,'train_dataset':train,'val_dataset':val,
            'device':device}

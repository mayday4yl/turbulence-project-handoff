"""Read-only reuse of the frozen models, Simulator and canonical base data."""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import os

import torch.distributed as dist

from models import build_model
from data_pipeline.dataset import V5Dataset
from runtime.environment import configure, make_simulator, ROOT


def _bound_json(metadata, key):
    path = metadata.get(key + "_path")
    expected = metadata.get(key + "_sha256")
    if not path or not expected or not Path(path).is_file():
        raise RuntimeError(f"Missing mandatory {key} readiness identity")
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{key} readiness artifact changed")
    return json.loads(path.read_text())


def training_components(config, device_or_none=None):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = device_or_none or f"npu:{local_rank}"
    configure(device)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != config.world_size:
        raise ValueError("Launcher and configured world_size disagree")
    if world == 6 and os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3,4,5,6,7":
        raise ValueError("Six-card A/B must explicitly map de36 devices 2 through 7")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("hccl" if str(device).startswith("npu") else "gloo",
                                timeout=timedelta(minutes=10))
    metadata = config.experiment_metadata or {}
    data_root = Path(metadata.get("dataset_root", ROOT / "data/v5_places_c50_s2000_v250_t250_g4_r4_20261021"))
    ready_path = data_root / "DATA_READY.json"
    if not ready_path.is_file() or hashlib.sha256(ready_path.read_bytes()).hexdigest() != metadata.get("data_ready_sha256"):
        raise RuntimeError("A/B require the unchanged original DATA_READY identity")
    ready = json.loads(ready_path.read_text())
    if ready.get("status") != "DATA_READY" or ready.get("sample_counts") != {"train": 8000, "val": 1000, "test": 1000}:
        raise RuntimeError("Original V5 dataset readiness/count mismatch")
    # New-stage audits may repair only new assets. Their immutable reports must
    # accompany each run; this factory never modifies the frozen base dataset.
    audit = _bound_json(metadata, "alignment_data_audit")
    if audit.get("status") not in ("PASS", "PASSED", "READY"):
        raise RuntimeError("Train/Val/Test source and realization audit has not passed")
    model = build_model(activation_checkpointing=config.activation_checkpointing,
        checkpoint_policy=config.checkpoint_policy, seed=config.seed, talking_backend=config.talking_backend)
    simulator = make_simulator(device)
    if config.psf_backend == "adjoint":
        from optimizations.psf_adjoint_20261021.adjoint import install
        simulator = install(simulator)
    simulator.eval()
    for parameter in simulator.parameters():
        parameter.requires_grad_(False)
    val = V5Dataset(data_root, "val", require_ready=True, allow_test=False)
    components = {"model": model, "simulator": simulator, "device": device, "val_dataset": val}
    if config.method == "A":
        from alignment_data.dataset import AlignmentDataset
        cache_root = metadata.get("alignment_cache_root")
        if not cache_root:
            raise RuntimeError("A requires an explicit versioned precomputed physical cache")
        components["train_dataset"] = AlignmentDataset(data_root, cache_root, split="train")
        components["val_a_dataset"] = AlignmentDataset(data_root, cache_root, split="val")
    elif config.method == "B":
        components["train_dataset"] = V5Dataset(data_root, "train", require_ready=True, allow_test=False)
    else:
        raise ValueError("Only A/B may use this factory")
    return components

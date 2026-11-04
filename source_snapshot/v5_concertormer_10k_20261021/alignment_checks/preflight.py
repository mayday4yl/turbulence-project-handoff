"""Bounded six-card A/B correctness checks; all optimizer updates are discarded.

The parent controller owns resource leases and launches this only after D0 has
completed. Never run this concurrently with the live D0 job. This uses the new
A/B computation and full checkpoint writer, not a substitute test training path.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from alignment_training.core import unroll_training, infer
from alignment_training.runner import (RunConfig, load_initial_model,
                                        save_training_checkpoint, scientific_config)
from alignment_training.state import (UpdateBatchSampler, UpdateLRScheduler, atomic_json,
    capture_rng, restore_rng, gather_rank_state, file_sha256, canonical_hash)
from runtime.environment import synchronize, memory_stats
from .environment_audit import runtime_flags
from .probes import (check_result_gradients, check_psf_adjoint, record_render_strengths,
                     verify_render_strengths, ForwardHotspots, tree_equal)


RUNTIME_FIELDS = ("world_size", "microbatch", "accumulation_steps", "effective_batch",
                  "network_precision", "activation_checkpointing", "checkpoint_policy",
                  "talking_backend", "psf_backend")


def finite_all(flag, device):
    value = torch.tensor([int(bool(flag))], dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())


def make_sampler(dataset, config, rank):
    kwargs = dict(seed=config.seed, microbatch=config.microbatch,
                  accumulation_steps=config.accumulation_steps, world_size=config.world_size,
                  rank=rank, start_update=1, max_updates=config.max_updates)
    return dataset.make_batch_sampler(**kwargs) if config.method == "A" else UpdateBatchSampler(len(dataset), **kwargs)


def forward(model, simulator, batch, config, device, *, path_seed, retain_trace=False):
    extra = {"time": batch["t"].to(device), "strength": batch["strength"].to(device)} if config.method == "A" else {}
    return unroll_training(model, simulator, batch["observation"].to(device), batch["clean"].to(device),
        method=config.method, path_seed=path_seed, network_precision=config.network_precision,
        objective=config.objective, retain_trace=retain_trace, **extra)


def checkpoint_probe(model, optimizer, scheduler, sampler, config, device, directory, initial_identity):
    """Exactly restore saved values and every rank's RNG and sampler position."""
    rank = dist.get_rank()
    probe_config = replace(config, output_dir=str(directory), resume_checkpoint=None)
    if rank == 0 and directory.exists():
        raise FileExistsError(directory)
    dist.barrier()
    synchronize(device)
    started = time.perf_counter()
    save_training_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
        config=probe_config, completed_update=2,
        counters={"formal_optimizer_updates": 0, "temporary_optimizer_updates_discarded": 2},
        best_psnr=0., best_update=2, initial_identity=initial_identity, device=device,
        make_best=True, periodic=False, rolling=True)
    synchronize(device)
    write_seconds = time.perf_counter() - started
    path = directory / "checkpoints/latest.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "rank_states", "sampler", "config_hash", "initial_identity"}
    if not required.issubset(saved) or saved["world_size"] != 6 or len(saved["rank_states"]) != 6:
        raise RuntimeError("Incomplete six-card checkpoint")
    raw = model.module
    if not tree_equal(raw.state_dict(), saved["model"]):
        raise RuntimeError("Saved model differs from current model")
    if not tree_equal(optimizer.state_dict(), saved["optimizer"]):
        raise RuntimeError("Saved optimizer differs from current optimizer")
    if not tree_equal(scheduler.state_dict(), saved["scheduler"]):
        raise RuntimeError("Saved scheduler differs from current scheduler")
    expected_sampler = sampler.state_dict(2)
    if not tree_equal(expected_sampler, saved["sampler"]):
        raise RuntimeError("Saved sampler position differs from completed update")
    # Deliberately perturb a model value, LR and RNG, then restore through the
    # same public state APIs used by production. No extra optimizer update.
    with torch.no_grad():
        next(p for p in raw.parameters() if p.requires_grad).add_(.125)
    optimizer.param_groups[0]["lr"] = .123
    random.random(); np.random.random(); torch.rand(3); torch.rand(3, device=device)
    raw.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng(saved["rank_states"][rank]["rng"], device)
    if not tree_equal(capture_rng(device), saved["rank_states"][rank]["rng"]):
        raise RuntimeError("Per-rank Python/NumPy/CPU/NPU RNG did not restore exactly")
    if not tree_equal(raw.state_dict(), saved["model"]) or not tree_equal(optimizer.state_dict(), saved["optimizer"]):
        raise RuntimeError("Model/optimizer restore is not exact")
    if not tree_equal(scheduler.state_dict(), saved["scheduler"]):
        raise RuntimeError("Scheduler restore is not exact")
    regenerated = make_sampler_for_resume(sampler, config, rank, saved["sampler"])
    for micro in range(config.accumulation_steps):
        if not tree_equal(sampler.indices_for(3, micro), regenerated.indices_for(3, micro)):
            raise RuntimeError("Next-update data/cache sampling changed on resume")
    synchronize(device)
    records = gather_rank_state({"rank": rank, "write_seconds": write_seconds,
        "write_restore_seconds": time.perf_counter() - started, "rng_exact": True,
        "model_optimizer_scheduler_exact": True, "next_sampler_indices_exact": True})
    del saved
    return {"status": "PASS", "path": str(path), "bytes": path.stat().st_size,
            "rank_records": records, "temporary_only": True,
            "formal_optimizer_updates": 0, "next_update_sampling_exact": True}


def make_sampler_for_resume(sampler, config, rank, saved_state):
    """Rebuild through the actual sampler constructor while retaining cache identity."""
    # A may have cache-specific constructor state; its own from-state checker is
    # preferred. The immutable indexed stream also permits a pure index replay.
    completed_update = saved_state["completed_update"]
    if hasattr(sampler, "clone_for_resume"):
        result = sampler.clone_for_resume(completed_update + 1, config.max_updates)
        if not tree_equal(result.state_dict(completed_update), saved_state):
            raise RuntimeError("Sampler clone differs from saved state")
        return result
    import copy
    result = copy.deepcopy(sampler)
    if hasattr(result, "load_state_dict"):
        result.load_state_dict(saved_state)
    result.start_update = completed_update + 1
    result.max_updates = config.max_updates
    if not tree_equal(result.state_dict(completed_update), saved_state):
        raise RuntimeError("Sampler cloned resume state differs")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--initialization-report", required=True)
    p.add_argument("--d0-status", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--factory", default="alignment_training.factories:training_components")
    args = p.parse_args(argv)
    raw_config = json.loads(Path(args.config).read_text())
    config = RunConfig(**raw_config.get("config", raw_config))
    config.validate()
    if config.world_size != 6 or config.effective_batch != 48 or config.microbatch != 2 or config.accumulation_steps != 4:
        raise ValueError("This check is explicitly bound to six-card micro2 x accum4 B48")
    if config.network_precision != "bf16" or config.psf_backend != "adjoint":
        raise ValueError("A/B must use BF16 network and the accepted FP32 adjoint PSF path")
    d0 = json.loads(Path(args.d0_status).read_text())
    if d0.get("completed_update") != 10000 or not str(d0.get("status", "")).startswith("TRAINING_COMPLETE"):
        raise RuntimeError("D0 is not finished; refusing to allocate any NPU context")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3,4,5,6,7":
        raise RuntimeError("Explicit physical cards 2-7 mapping required")
    rank = int(os.environ.get("RANK", "0"))
    output = Path(args.output)
    rank_output = output.with_name(output.stem + f".rank{rank}.json")
    if output.exists() or rank_output.exists():
        raise FileExistsError("Preserve previous preflight evidence; use a new report path")
    report = {"schema": "v5_alignment_readiness_v1", "status": "RUNNING",
              "method": config.method, "objective": config.objective,
              "runtime": {k: getattr(config, k) for k in RUNTIME_FIELDS},
              "formal_optimizer_updates": 0, "test_used": False,
              "started_unix": time.time(), "measurements": [],
              "config_sha256": file_sha256(args.config),
              "scientific_config_hash": canonical_hash(scientific_config(config)),
              "initial_checkpoint_sha256": file_sha256(config.initial_checkpoint),
              "initialization_report_sha256": file_sha256(args.initialization_report),
              "experiment_metadata": config.experiment_metadata}
    atomic_json(report, rank_output)
    success = False
    try:
        initialization = json.loads(Path(args.initialization_report).read_text())
        if (initialization.get("status") != "INITIALIZATION_PASS"
                or initialization.get("stage") != config.method
                or initialization.get("initial_checkpoint_sha256") != report["initial_checkpoint_sha256"]):
            raise RuntimeError("Initialization evidence is not bound to these real parent weights")
        module_name, function_name = args.factory.split(":")
        factory = getattr(importlib.import_module(module_name), function_name)
        components = factory(config)
        device, model, simulator = components["device"], components["model"], components["simulator"]
        report["runtime_flags"] = runtime_flags()
        if dist.get_world_size() != 6:
            raise RuntimeError("Six HCCL ranks were not initialized")
        random.seed(config.seed + rank); np.random.seed((config.seed + rank) % 2**32)
        torch.manual_seed(config.seed + rank); torch.npu.manual_seed(config.seed + rank)
        initial = load_initial_model(model, config.initial_checkpoint)
        initial_identity = {"path": config.initial_checkpoint, "sha256": report["initial_checkpoint_sha256"],
                            "provenance": initial.get("provenance")}
        del initial
        model.to(device).train()
        if not getattr(simulator, "_psf_adjoint_candidate", False):
            raise RuntimeError("Accepted exact PSF adjoint optimization not installed")
        if any(p.requires_grad for p in simulator.parameters()):
            raise RuntimeError("Simulator parameters must remain frozen")
        sampler = make_sampler(components["train_dataset"], config, rank)
        loader = DataLoader(components["train_dataset"], batch_sampler=sampler, num_workers=0,
                            generator=torch.Generator().manual_seed(config.seed + 991 + rank))
        # One real input gradient probe, outside timed throughput. This does not
        # call optimizer.step and is discarded before the update test.
        diagnostic = next(iter(loader))
        with record_render_strengths(simulator) as strengths, ForwardHotspots(
                model, simulator, lambda: synchronize(device)) as hotspot:
            result = forward(model, simulator, diagnostic, config, device,
                             path_seed=config.seed + rank, retain_trace=True)
        report["forward_hotspots"] = {"inclusive_nested_times_do_not_sum": True,
            "cold_diagnostic_synchronous_not_throughput": True, "records": hotspot.records}
        report["loss_strength_semantics"] = verify_render_strengths(strengths, config.method,
            strength=diagnostic["strength"] if config.method == "A" else None, paths=result.paths)
        synchronize(device)
        started = time.perf_counter()
        report["effective_gradients"] = check_result_gradients(result, config.method)
        synchronize(device)
        report["diagnostic_backward_seconds"] = time.perf_counter() - started
        del result, diagnostic
        model.zero_grad(set_to_none=True)
        torch.npu.empty_cache()
        report["psf_adjoint_regression"] = check_psf_adjoint(device, torch.float32)
        # Validate the real deployment path too: A's single-step training must
        # not be mistaken for its four-step model-selection inference. These
        # are only 12 distinct Val examples across ranks, never Val1000 metrics.
        val_samples = [components["val_dataset"][rank * 2 + i] for i in range(2)]
        val_observed = torch.stack([r["observation"] for r in val_samples]).to(device)
        val_clean = torch.stack([r["clean"] for r in val_samples]).to(device)
        model.eval()
        val_output, val_trace = infer(model, simulator, val_observed, method=config.method,
            objective=config.objective, network_precision=config.network_precision,
            N=4, gamma=1., diagnostic_target=val_clean)
        if val_trace["nfe"] != 4 or len(val_trace["stage_metrics"]) != 4:
            raise RuntimeError("The actual A/B inference path must have four predicted stages")
        if not finite_all(torch.isfinite(val_output).all().item() and all(
                torch.isfinite(v).all().item() for row in val_trace["stage_metrics"] for v in row.values()), device):
            raise FloatingPointError("Nonfinite actual-initialization four-step inference")
        report["four_step_inference_finite"] = {"passed": True, "local_val_indices": [rank * 2, rank * 2 + 1],
            "nfe": 4, "gamma": 1., "refinement": False,
            "scope": "bounded finite-path check; not a full validation result or best-selection metric"}
        del val_samples, val_observed, val_clean, val_output, val_trace
        model.train()
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(config.backbone_lr, config.new_lr,
            config.weight_decay), betas=(.9, .999), eps=1e-8)
        scheduler = UpdateLRScheduler(optimizer, max_updates=config.max_updates, warmup_updates=config.warmup_updates)
        if optimizer.state or scheduler.last_update != 0:
            raise RuntimeError("Stage optimizer/scheduler must be newly initialized")
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False,
                    find_unused_parameters=False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        iterator = iter(loader)
        for update in (1, 2):
            dist.barrier(); synchronize(device)
            torch.npu.reset_peak_memory_stats(device)
            started = time.perf_counter()
            scheduler.step_for_update(update); optimizer.zero_grad(set_to_none=True)
            losses = torch.zeros(3, dtype=torch.float32, device=device)
            endpoint_count, data_seconds, compute_seconds = 0, 0., 0.
            for micro in range(config.accumulation_steps):
                input_started = time.perf_counter()
                batch = next(iterator)
                data_seconds += time.perf_counter() - input_started
                if config.method == "A":
                    endpoint_count += int((batch["t"] == 0).sum())
                compute_started = time.perf_counter()
                with model.no_sync() if micro + 1 < config.accumulation_steps else nullcontext():
                    result = forward(model, simulator, batch, config, device,
                        path_seed=config.seed + (update - 1) * 24 + micro * 6 + rank)
                    if not finite_all(torch.isfinite(result.loss).item(), device):
                        raise FloatingPointError("Nonfinite full-update loss")
                    (result.loss / config.accumulation_steps).backward()
                compute_seconds += time.perf_counter() - compute_started
                losses += torch.stack((result.loss.detach(), result.clean_loss.detach(),
                                       result.observation_loss.detach())) / config.accumulation_steps
                del result, batch
            missing = [n for n, p0 in model.module.named_parameters() if p0.requires_grad and p0.grad is None]
            if not finite_all(not missing, device):
                raise RuntimeError(f"Missing trainable gradients on rank {rank}: {missing[:20]}")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=False)
            if not finite_all(torch.isfinite(grad_norm).item() and grad_norm.item() > 0, device):
                raise FloatingPointError("Nonfinite or zero full gradient norm")
            group_gradients = {}
            for prefix in ("backbone.ending", "time.films", "raw_head"):
                values = [p0.grad.float().square().sum() for n, p0 in model.module.named_parameters()
                          if n.startswith(prefix) and p0.requires_grad and p0.grad is not None]
                norm = torch.stack(values).sum().sqrt().item()
                if not finite_all(math.isfinite(norm) and norm > 0, device):
                    raise RuntimeError(f"No effective parameter gradient in {prefix}")
                group_gradients[prefix] = norm
            optimizer.step(); synchronize(device)
            elapsed = time.perf_counter() - started
            rows = gather_rank_state({"rank": rank, "seconds": elapsed,
                "total_loss": losses[0].item(), "clean_mae": losses[1].item(), "observation_mae": losses[2].item(),
                "grad_norm_before_clip": grad_norm.item(), "parameter_gradient_norms_after_clip": group_gradients,
                "finite_gradients": True, "all_trainable_gradients_present": True,
                "endpoint_samples": endpoint_count, "data_loader_seconds": data_seconds,
                "forward_backward_host_seconds": compute_seconds,
                "ddp_sync_policy": "no_sync first 3 microbatches; synchronize 4th",
                **memory_stats(device)})
            if config.method == "A" and sum(r["endpoint_samples"] for r in rows) != 12:
                raise RuntimeError("A full B48 update did not have exactly 12 endpoints and 36 internal states")
            seconds = max(r["seconds"] for r in rows)
            report["measurements"].append({"temporary_update": update, "phase": "warmup" if update == 1 else "timed",
                "max_rank_seconds": seconds, "global_samples_per_second": 48 / seconds, "rank_records": rows})
            atomic_json(report, rank_output)
            if rank == 0:
                print(json.dumps({"event": "alignment_preflight_update", "stage": config.method,
                                  "temporary_update": update, "seconds": seconds}), flush=True)
        report["checkpoint_compatibility"] = checkpoint_probe(model, optimizer, scheduler, sampler, config,
            device, output.parent / (output.stem + "_checkpoint_probe"), initial_identity)
        report.update(status="READY_FOR_TRAINING", validated_methods=[config.method],
            temporary_optimizer_updates_discarded=2, initialization_evidence=initialization,
            stable_throughput_claim=False,
            scope="two disposable full updates and exact state restore; no convergence/physical-model certification")
        success = True
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["finished_unix"] = time.time()
        atomic_json(report, rank_output)
        if rank == 0:
            atomic_json(report, output)
            print(json.dumps({"status": report["status"], "method": config.method,
                              "output": str(output), "error": report.get("error")}), flush=True)
        if success and dist.is_initialized():
            dist.barrier(); dist.destroy_process_group()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

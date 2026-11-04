#!/usr/bin/env python3
"""Zero-update real four-rank FP32 gradient-normalization acceptance.

Through guard_distributed.py and run_distributed_env.sh, invoke:
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    code/check_distributed_gradients.py --config configs/candidate.json \
    --output artifacts/ddp_gradients_mb1_01 --microbatch 1

One fixed mixed-G global batch of 32 is differentiated twice at the same initial
weights and rank-local RNG state. The reference bypasses DDP, differentiates
local path SUM / 32 using canonical microbatch one, and explicitly SUM-reduces
parameter gradients. The actual DDP path differentiates local path SUM * 4/32
using the selected microbatch and its real accumulation pattern. This checks
both DDP normalization and native cross-microbatch numerical equivalence.
No clipping, optimizer step, training refinement, or GT xi loading is performed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist

from train_worker import make_engine, resolved_identity
from v031.data import StartDataset, collate_starts
from v031.distributed_runtime import configure_local_rank
from v031.runtime import save_json, sha256, state_hash, verify_source
from v031.training import (capture_rank_rng, memory_stats, restore_rank_rng,
                           synchronize)


GRADIENT_RELATIVE_L2_MAX = 1e-4
GRADIENT_COSINE_MIN = 0.99999
FORWARD_ATOL = 1e-6
FORWARD_RTOL = 1e-5
BRANCHES = ("clean_head", "zernike_head", "tilt_head")


def gradients_to_cpu(network):
    gradients = {}
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Disconnected trainable parameter: {name}")
        value = parameter.grad.detach().cpu().contiguous()
        if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Gradient is not finite FP32: {name}")
        gradients[name] = value.clone()
    return gradients


def gradient_comparison(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError("Compared trainable gradient keys differ")
    ref_square, candidate_square, difference_square, products = [], [], [], []
    for name in reference:
        left, right = reference[name].double(), candidate[name].double()
        if left.shape != right.shape:
            raise RuntimeError(f"Compared gradient shape differs: {name}")
        ref_square.append(float(left.square().sum()))
        candidate_square.append(float(right.square().sum()))
        difference_square.append(float((left - right).square().sum()))
        products.append(float((left * right).sum()))
    nr, nc = math.fsum(ref_square), math.fsum(candidate_square)
    nd, dot = math.fsum(difference_square), math.fsum(products)
    relative = math.sqrt(nd / nr) if nr > 0 else (0.0 if nd == 0 else None)
    cosine = dot / math.sqrt(nr * nc) if nr > 0 and nc > 0 else None
    return {"tensor_count": len(reference), "reference_norm": math.sqrt(nr),
            "candidate_norm": math.sqrt(nc), "difference_norm": math.sqrt(nd),
            "relative_l2_difference": relative, "cosine": cosine,
            "both_zero": nr == 0 and nc == 0}


def forward_comparison(reference, candidate):
    if reference.shape != candidate.shape:
        raise RuntimeError("Compared path-metric shape differs")
    difference = (reference.double() - candidate.double()).abs()
    return {"passed": bool(torch.allclose(reference, candidate, rtol=FORWARD_RTOL,
                                          atol=FORWARD_ATOL)),
            "max_absolute_difference": float(difference.max()),
            "reference": reference.tolist(), "candidate": candidate.tolist(),
            "columns": ["path_total", "path_LC", "path_LR"],
            "atol": FORWARD_ATOL, "rtol": FORWARD_RTOL}


def backward_pass(engine, local_indices, *, ddp, microbatch=None):
    microbatch = engine.microbatch if microbatch is None else microbatch
    if microbatch not in {1, 2, 4, 8} or len(local_indices) % microbatch:
        raise ValueError("Gradient check needs a supported exact microbatch divisor")
    engine.optimizer.zero_grad(set_to_none=True)
    metrics = []
    scale = 4.0 / 32.0 if ddp else 1.0 / 32.0
    for offset in range(0, len(local_indices), microbatch):
        batch = collate_starts([engine.dataset[i] for i in
                                local_indices[offset:offset + microbatch]])
        clean, observation, gs = [batch[key].to(engine.device) for key in
                                  ("clean", "observation", "G")]
        final = offset + microbatch == len(local_indices)
        context = engine.module.no_sync() if ddp and not final else nullcontext()
        with context:
            module = engine.module if ddp else engine.rollout
            output = module(clean, observation, gs, batch["severity_path"])
            rows = torch.stack([output["path_losses"], output["path_clean"],
                                output["path_reconstruction"]], dim=1)
            if not torch.isfinite(rows).all().item():
                raise FloatingPointError("Nonfinite path loss")
            metrics.append(rows.detach().cpu())
            (output["loss_sum"] * scale).backward()
        del rows, output, clean, observation, gs, batch
    if not ddp:
        # All ranks call collectives in the same parameter order. This is an
        # independent explicit SUM, with neither DDP averaging nor clipping.
        for name, parameter in engine.network.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    raise RuntimeError(f"Disconnected reference parameter: {name}")
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    synchronize(engine.device)
    return torch.cat(metrics), gradients_to_cpu(engine.network)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=[1, 2, 4, 8], default=1)
    args = parser.parse_args()
    # make_engine uses this solely to keep formal=False; no worker update loop
    # or smoke-update allocation is called by this acceptance entry point.
    args.mode = "performance-smoke"
    config = json.loads(args.config.read_text())
    if config.get("seed") != 20260916 or config.get("lambda_R") != 0.2:
        raise ValueError("Frozen seed 20260916 and lambda_R 0.2 are required")
    training = config["training"]
    for key, expected in {"precision": "fp32", "world_size": 4, "effective_batch": 32,
                          "microbatch": args.microbatch,
                          "accumulation": 8 // args.microbatch}.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must equal {expected!r}")
    rank = int(os.environ["LOCAL_RANK"])
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    result = {"status": "initializing", "rank": rank, "world_size": 4,
              "microbatch": args.microbatch, "accumulation": 8 // args.microbatch,
              "reference_microbatch": 1, "reference_accumulation": 8,
              "actual_microbatch": args.microbatch, "actual_accumulation": 8 // args.microbatch,
              "comparison_scope": "canonical_mb1_manual_SUM_vs_selected_microbatch_DDP",
              "formal_training_started": False, "restoration_optimizer_updates": 0,
              "smoke_optimizer_updates": 0, "effective_batch": 32,
              "source_sha256": sha256(__file__), "config_sha256": sha256(args.config),
              "thresholds": {"full_gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
                             "full_gradient_cosine_min": GRADIENT_COSINE_MIN,
                             "forward_pathloss_atol": FORWARD_ATOL,
                             "forward_pathloss_rtol": FORWARD_RTOL}}
    started = time.perf_counter()
    try:
        result["execution"] = configure_local_rank(config, rank)
        verify_source(config)
        dist.init_process_group("hccl", timeout=timedelta(minutes=30))
        dist.barrier()
        dataset = StartDataset(config["dataset_root"])
        if len(dataset) != 1600:
            raise ValueError("The accepted dataset must have 1600 starts")
        identity = resolved_identity(config, dataset)
        engine = make_engine(config, dataset, identity, args, rank)
        result["identity"] = engine.identity
        indices = engine.sampler.next_indices()
        records = [dataset.records[i] for i in indices]
        if {int(row["initial_G"]) for row in records} != {1, 2, 3, 4}:
            raise RuntimeError("Frozen first global batch does not cover all four G values")
        result["global_start_ids"] = [row["start_id"] for row in records]
        result["global_initial_G"] = [int(row["initial_G"]) for row in records]
        identities = [None] * 4
        initial_network = state_hash(engine.network)
        initial_simulator = state_hash(engine.rollout.simulator)
        for field, actual in (("expected_initial_network_sha256", initial_network),
                              ("expected_simulator_sha256", initial_simulator)):
            if field in config and config[field] != actual:
                raise RuntimeError(f"Accepted initial state hash mismatch: {field}")
        dist.all_gather_object(identities, {"network": initial_network,
                                           "simulator": initial_simulator,
                                           "start_ids": result["global_start_ids"]})
        if any(value != identities[0] for value in identities):
            raise RuntimeError("Rank initial weights, simulator state or global IDs differ")
        result["initial_network_sha256"] = initial_network
        result["initial_simulator_sha256"] = initial_simulator
        local_indices = indices[rank * 8:(rank + 1) * 8]
        initial_rng = capture_rank_rng(engine.device)
        reference_metrics, reference = backward_pass(engine, local_indices, ddp=False, microbatch=1)
        restore_rank_rng(initial_rng, engine.device)
        actual_metrics, actual = backward_pass(engine, local_indices, ddp=True, microbatch=args.microbatch)
        comparison = gradient_comparison(reference, actual)
        result["full_gradient"] = comparison
        result["forward_path_losses"] = forward_comparison(reference_metrics, actual_metrics)
        result["branch_gradients"] = {}
        for branch in BRANCHES:
            names = [name for name in reference if name.split(".")[0] == branch]
            if not names:
                raise RuntimeError(f"Required network head not found: {branch}")
            result["branch_gradients"][branch] = {
                "all_parameters_connected_in_both_passes": True,
                "names": names,
                **gradient_comparison({name: reference[name] for name in names},
                                      {name: actual[name] for name in names})}
        result["zero_branch_note"] = (
            "A connected zero gradient is reported separately from a disconnected graph; "
            "zero initial native-xi heads can have zero-valued derivatives.")
        result["network_unchanged"] = initial_network == state_hash(engine.network)
        result["simulator_unchanged"] = initial_simulator == state_hash(engine.rollout.simulator)
        result["optimizer_step_calls"] = engine.optimizer_step_calls
        result["optimizer_step_attempts"] = engine.optimizer_step_attempts
        result["scheduler_completed_updates"] = engine.completed_updates
        result["optimizer_state_empty"] = not bool(engine.optimizer.state)
        result["all_trainable_parameters_connected"] = True
        result["gradient_tensor_count"] = len(reference)
        result["reference_local_sum_scale"] = 1.0 / 32.0
        result["ddp_local_sum_scale"] = 4.0 / 32.0
        result["backward_passes_of_same_global_batch"] = 2
        passed = (comparison["reference_norm"] > 0
                  and comparison["cosine"] is not None
                  and comparison["cosine"] >= GRADIENT_COSINE_MIN
                  and comparison["relative_l2_difference"] is not None
                  and comparison["relative_l2_difference"] <= GRADIENT_RELATIVE_L2_MAX
                  and result["forward_path_losses"]["passed"]
                  and result["network_unchanged"] and result["simulator_unchanged"]
                  and result["optimizer_step_calls"] == 0
                  and result["optimizer_step_attempts"] == 0
                  and result["scheduler_completed_updates"] == 0
                  and result["optimizer_state_empty"])
        result["status"] = "passed" if passed else "failed"
        result["wall_seconds"] = time.perf_counter() - started
        result["memory"] = memory_stats(engine.device)
        save_json(output / f"results_rank{rank}.json", result)
        decision = torch.tensor(int(passed), device=engine.device)
        dist.all_reduce(decision, op=dist.ReduceOp.MIN)
        dist.barrier()
        if rank == 0:
            rank_results = [json.loads((output / f"results_rank{i}.json").read_text())
                            for i in range(4)]
            save_json(output / "results.json", {
                "status": "passed" if decision.item() else "failed",
                "restoration_optimizer_updates": 0, "smoke_optimizer_updates": 0,
                "rank_results": rank_results})
        if not decision.item():
            raise RuntimeError("Real DDP normalization/unchanged-state acceptance failed")
        return 0
    except BaseException as error:
        result.update({"status": "failed", "error": repr(error),
                       "traceback": traceback.format_exc(),
                       "wall_seconds": time.perf_counter() - started})
        if output.is_dir():
            save_json(output / f"error_rank{rank}.json", result)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

"""Full, source-aware validation; no test-set entry point."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import math
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .core import infer
from .state import atomic_json, gather_rank_state, capture_rng, restore_rng


def distributed_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def synchronize(device):
    device = torch.device(device)
    if device.type in ("npu", "cuda"):
        getattr(torch, device.type).synchronize(device)


def memory_stats(device):
    device = torch.device(device)
    if device.type not in ("npu", "cuda"):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    backend = getattr(torch, device.type)
    return {"peak_allocated_bytes": backend.max_memory_allocated(device),
            "peak_reserved_bytes": backend.max_memory_reserved(device)}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _batch_strings(batch, name, count):
    values = batch.get(name)
    if values is None:
        raise ValueError(f"Validation data missing {name}")
    if isinstance(values, str):
        values = [values]
    if len(values) != count:
        raise ValueError(f"Validation {name} cardinality mismatch")
    return [str(value) for value in values]


def _checked_psnr(mse, label):
    # Finite RGB values can still overflow when squared in FP32. Reject an
    # invalid metric before it can be serialized as a successful validation.
    if not (torch.isfinite(mse) & (mse >= 0)).all().item():
        raise FloatingPointError(f"nonfinite or negative full validation {label} MSE")
    psnr = -10. * torch.log10(mse)
    # Exact zero MSE retains the mathematical PSNR=+inf definition. No epsilon
    # floor or clipping is introduced into checkpoint selection.
    valid = torch.isfinite(psnr) | ((psnr == math.inf) & (mse == 0))
    if not valid.all().item():
        raise FloatingPointError(f"invalid full validation {label} PSNR")
    return psnr.cpu().tolist()


def evaluate_full_validation(model, simulator, dataset, *, device, method,
                             output_dir, update, network_precision="fp32",
                             batch_size=1, workers=0, refinement=None,
                             expected_size=1000, extended_metrics=False, objective='v5_dual'):
    clean_only = objective == 'clean_only'
    if getattr(dataset, "split", "val") not in ("val", "validation"):
        raise ValueError("Only the validation split may be used during model selection")
    if len(dataset) != expected_size:
        raise ValueError(f"Full Val{expected_size} required, found {len(dataset)}")
    rank, world = distributed_info()
    rng_before = capture_rng(device)
    indices = list(range(rank, len(dataset), world))
    worker_options = {"multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=workers, drop_last=False,
                        generator=torch.Generator().manual_seed(20261021 + rank), **worker_options)
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()
    synchronize(device)
    started = time.monotonic()
    rows = []
    try:
        for batch in loader:
            clean = batch["clean"].to(device=device, dtype=torch.float32)
            observation = batch["observation"].to(device=device, dtype=torch.float32)
            strength = torch.as_tensor(batch.get("G", batch.get("strength", float("nan"))))
            if not bool((strength == 4).all()):
                raise ValueError("V5-v1 validation observations must be generated at G=4")
            output, diagnostics = infer(raw_model, simulator, observation, method=method,
                                       network_precision=network_precision, refinement=refinement,
                                       diagnostic_target=clean if extended_metrics else None, objective=objective)
            if not torch.isfinite(output).all().item():
                raise FloatingPointError("nonfinite full validation output")
            # One RGB MSE per complete image; no clamp, crop, Y-channel or 8-bit
            # quantization. Reduce in FP32, then convert to Python for averaging.
            mse = (output.float() - clean).square().flatten(1).mean(1)
            clipped_mse = (output.float().clamp(0, 1) - clean).square().flatten(1).mean(1)
            psnr = _checked_psnr(mse, "unclamped RGB")
            clipped_psnr = _checked_psnr(clipped_mse, "clipped auxiliary RGB")
            source_ids = _batch_strings(batch, "source_id", len(output))
            sample_key = "sample_id" if "sample_id" in batch else "realization_id"
            sample_ids = _batch_strings(batch, sample_key, len(output))
            mse_values = mse.cpu().tolist()
            if extended_metrics:
                from .image_metrics import image_metrics
                ssim, lpips_values = image_metrics(output, clean)
                input_psnr = _checked_psnr((observation - clean).square().flatten(1).mean(1), 'input RGB')
                if clean_only:
                    clean_mae = (output.float() - clean).abs().flatten(1).mean(1).cpu().tolist()
                stages = [{**{k: v.cpu().tolist() for k, v in values.items()},
                           'psnr': _checked_psnr(values['rgb_mse'], 'stage RGB')}
                          for values in diagnostics.get('stage_metrics', [])]
            for i in range(len(output)):
                row = {"sample_id": sample_ids[i], "source_id": source_ids[i],
                       "psnr": psnr[i], "clipped_psnr_auxiliary": clipped_psnr[i],
                       "rgb_mse": mse_values[i]}
                if extended_metrics:
                    row.update(ssim=ssim[i], lpips=lpips_values[i], input_psnr=input_psnr[i],
                        psnr_gain_vs_input=psnr[i] - input_psnr[i],
                        stages=[{k: v[i] for k, v in stage.items()} for stage in stages])
                    if clean_only:
                        row.pop('stages')
                        row.update(clean_mae=clean_mae[i], clean_loss=clean_mae[i], total_loss=clean_mae[i])
                    else:
                        for key in ('total_loss', 'clean_loss', 'reprojection_loss'):
                            row[key] = sum(stage[key][i] for stage in stages) / len(stages)
                if refinement is not None:
                    # Refinement diagnostics are batch objective means. They
                    # are not mislabeled as per-image when batch_size > 1.
                    row["refinement_batch_diagnostics"] = diagnostics["refinement"]
                rows.append(row)
        synchronize(device)
        local_elapsed = time.monotonic() - started
        rank_values = gather_rank_state({"rows": rows, "elapsed_seconds": local_elapsed,
                                         **memory_stats(device)})
        all_rows = sorted([row for item in rank_values for row in item["rows"]],
                          key=lambda row: row["sample_id"])
        if len(all_rows) != expected_size or len({r["sample_id"] for r in all_rows}) != expected_size:
            raise RuntimeError("Full validation has missing or duplicate realization IDs")
        sources = Counter(row["source_id"] for row in all_rows)
        if expected_size == 1000 and (len(sources) != 250 or set(sources.values()) != {4}):
            raise RuntimeError("Full Val1000 must contain exactly 250 sources with four realizations each")
        summary = {"event": "full_validation", "update": update, "method": method,
                   "view": "original_Val1000_real_recursive_rollout",
                   "used_for_best_selection": True,
                   "observation_target": "original_G4_and_Y4",
                   "count": len(all_rows), "source_count": len({r["source_id"] for r in all_rows}),
                   "psnr": sum(r["psnr"] for r in all_rows) / expected_size,
                   "clipped_psnr_auxiliary": sum(r["clipped_psnr_auxiliary"] for r in all_rows) / expected_size,
                   "selection_metric": "mean_per_image_full_rgb_psnr_unclamped",
                   "refinement": refinement is not None, "inference_nfe": 1 if method == "D0" else 4,
                   "gamma": 1., "elapsed_seconds": max(item["elapsed_seconds"] for item in rank_values),
                   "rank_resources": [{k: v for k, v in item.items() if k != "rows"} for item in rank_values]}
        if extended_metrics:
            from .image_metrics import SPEC
            loss_keys = ('total_loss', 'clean_loss', 'clean_mae') if clean_only else ('total_loss', 'clean_loss', 'reprojection_loss')
            summary.update({key: sum(row[key] for row in all_rows) / expected_size for key in
                ('ssim', 'lpips', 'input_psnr', 'psnr_gain_vs_input', *loss_keys)})
            if not clean_only:
                summary['stage_psnr'] = [sum(r['stages'][s]['psnr'] for r in all_rows) / expected_size
                                         for s in range(summary['inference_nfe'])]
                summary['stage_losses'] = [{k: sum(r['stages'][s][k] for r in all_rows) / expected_size
                    for k in ('total_loss', 'clean_loss', 'reprojection_loss')} for s in range(summary['inference_nfe'])]
            summary['metric_spec'] = {**SPEC, 'loss': 'mean absolute clean RGB error, weight 1.0'} if clean_only else SPEC
            summary['objective'] = objective
        for name in ("psnr", "clipped_psnr_auxiliary"):
            if math.isnan(summary[name]) or summary[name] == -math.inf:
                raise FloatingPointError(f"invalid full validation aggregate {name}")
        if rank == 0:
            suffix = "refinement_on" if refinement is not None else "refinement_off"
            prefix = Path(output_dir) / "validation" / f"update_{update:06d}_{suffix}"
            atomic_json(json_safe({"summary": summary, "images": all_rows}), prefix.with_suffix(".json"))
            append_jsonl(Path(output_dir) / "validation_curve.jsonl", summary)
        return summary
    finally:
        raw_model.train(was_training)
        restore_rng(rng_before, device)


def evaluate_val_a(model, simulator, dataset, *, device, output_dir, update,
                   network_precision="fp32", batch_size=1, workers=0,
                   expected_size=1000, extended_metrics=True):
    """Fixed teacher-state diagnostic view, never used to select a checkpoint.

    All image metric definitions are exactly those used by the original Val1000
    evaluator. Only input time, current-g physical target and NFE differ.
    """
    from .core import unroll_training
    if getattr(dataset, "split", None) not in ("val", "validation"):
        raise ValueError("Val-A must use only the original validation sources")
    if len(dataset) != expected_size:
        raise ValueError("Val-A must contain its complete fixed view")
    rank, world = distributed_info()
    rng_before = capture_rng(device)
    indices = list(range(rank, len(dataset), world))
    worker_options = {"multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
        shuffle=False, num_workers=workers, drop_last=False,
        generator=torch.Generator().manual_seed(20261022 + rank), **worker_options)
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()
    synchronize(device)
    started = time.monotonic()
    rows = []
    try:
        for batch in loader:
            target = batch["clean"].to(device=device, dtype=torch.float32)
            observation = batch["observation"].to(device=device, dtype=torch.float32)
            time_values = torch.as_tensor(batch["t"], dtype=torch.float32, device=device)
            strength = torch.as_tensor(batch["strength"], dtype=torch.float32, device=device)
            with torch.no_grad():
                result = unroll_training(raw_model, simulator, observation, target,
                    method="A", objective="precomputed_current_g", path_seed=0,
                    time=time_values, strength=strength, network_precision=network_precision,
                    retain_trace=True)
                output, matched = result.final_clean, result.trace[0]["matched"]
                if not (torch.isfinite(output).all() & torch.isfinite(matched).all()).item():
                    raise FloatingPointError("Nonfinite fixed Val-A clean or physical reconstruction")
                mse = (output - target).square().flatten(1).mean(1)
                psnr = _checked_psnr(mse, "Val-A current-state clean RGB")
                clipped_psnr = _checked_psnr((output.clamp(0, 1) - target).square().flatten(1).mean(1), "Val-A auxiliary clipped RGB")
                input_psnr = _checked_psnr((observation - target).square().flatten(1).mean(1), "Val-A current Yg RGB")
                lc = (output - target).abs().flatten(1).mean(1).cpu().tolist()
                lo = (matched - observation).abs().flatten(1).mean(1).cpu().tolist()
                observation_mse = (matched - observation).square().flatten(1).mean(1)
                if not torch.isfinite(observation_mse).all().item():
                    raise FloatingPointError("Nonfinite Val-A physical error")
                observation_mse = observation_mse.cpu().tolist()
                if extended_metrics:
                    from .image_metrics import image_metrics
                    ssim, lpips_values = image_metrics(output, target)
            sources = _batch_strings(batch, "source_id", len(output))
            sample_ids = _batch_strings(batch, "sample_id", len(output))
            t_values, g_values = time_values.cpu().tolist(), strength.cpu().tolist()
            for i in range(len(output)):
                row = {"sample_id": sample_ids[i], "source_id": sources[i],
                    "t": t_values[i], "g": g_values[i], "psnr": psnr[i],
                    "clipped_psnr_auxiliary": clipped_psnr[i],
                    "current_input_psnr": input_psnr[i],
                    "psnr_gain_vs_current_input": psnr[i] - input_psnr[i],
                    "clean_loss": lc[i], "reprojection_loss": lo[i],
                    "total_loss": .9 * lc[i] + .1 * lo[i],
                    "observation_mse": observation_mse[i]}
                if extended_metrics:
                    row.update(ssim=ssim[i], lpips=lpips_values[i])
                rows.append(row)
        synchronize(device)
        ranks = gather_rank_state({"rows": rows, "elapsed_seconds": time.monotonic() - started,
                                   **memory_stats(device)})
        all_rows = sorted([row for item in ranks for row in item["rows"]], key=lambda row: row["sample_id"])
        if len(all_rows) != expected_size or len({r["sample_id"] for r in all_rows}) != expected_size:
            raise RuntimeError("Val-A has missing or duplicate identities")
        sources = Counter(r["source_id"] for r in all_rows)
        if expected_size == 1000 and (len(sources) != 250 or set(sources.values()) != {4}):
            raise RuntimeError("Val-A1000 must use the same 250 validation sources and four realizations")
        endpoint_count = sum(row["t"] == 0 for row in all_rows)
        if expected_size == 1000 and endpoint_count != 250:
            raise RuntimeError("Val-A must have exactly 250 endpoints and 750 fixed internal states")
        from .image_metrics import SPEC
        summary = {"event": "val_a_single_step_validation", "method": "A", "update": update,
            "view": "fixed_ValA1000_precomputed_physical_states", "used_for_best_selection": False,
            "count": expected_size, "source_count": len(sources), "endpoint_count": endpoint_count,
            "internal_count": expected_size - endpoint_count, "refinement": False,
            "inference_nfe": 1, "observation_target": "current_g_and_Yg",
            "elapsed_seconds": max(item["elapsed_seconds"] for item in ranks),
            "rank_resources": [{k: v for k, v in item.items() if k != "rows"} for item in ranks],
            "metric_spec": {**SPEC, "loss": "single-state 0.9 clean MAE + 0.1 current-g Yg reprojection MAE"}}
        keys = ["psnr", "clipped_psnr_auxiliary", "current_input_psnr", "psnr_gain_vs_current_input",
                "clean_loss", "reprojection_loss", "total_loss", "observation_mse"]
        if extended_metrics:
            keys += ["ssim", "lpips"]
        for key in keys:
            summary[key] = sum(row[key] for row in all_rows) / expected_size
            if math.isnan(summary[key]) or summary[key] == -math.inf:
                raise FloatingPointError(f"Invalid fixed Val-A aggregate {key}")
        bins = [{"label": "t=0", "rows": [row for row in all_rows if row["t"] == 0]}]
        bins += [{"label": f"({index / 10:.1f},{(index+1) / 10:.1f})",
                  "rows": [row for row in all_rows if row["t"] > 0 and min(int(row["t"] * 10), 9) == index]}
                 for index in range(10)]
        summary["time_bins"] = [{"label": item["label"], "count": len(item["rows"]),
            **({key: sum(row[key] for row in item["rows"]) / len(item["rows"])
                for key in ("psnr", "clean_loss", "reprojection_loss")} if item["rows"] else {})}
            for item in bins]
        if rank == 0:
            path = Path(output_dir) / "validation_a" / f"update_{update:06d}.json"
            atomic_json(json_safe({"summary": summary, "images": all_rows}), path)
            append_jsonl(Path(output_dir) / "val_a_curve.jsonl", summary)
        return summary
    finally:
        raw_model.train(was_training)
        restore_rng(rng_before, device)

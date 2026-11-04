"""Freeze real selected-parent weights, with explicit A/B inheritance checks.

Run only after the previous stage has finished and the requested devices have
been independently checked free. A CPU invocation cannot issue NPU readiness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from models import build_model
from models.clean_concertormer import build_clean_model
from scripts.prepare_clean_transfer import transfer
from training.core import network_context
from training.state import atomic_json, atomic_torch_save, file_sha256


def validate_parent(parent, summary, stage):
    expected_parent, budget = ("D0", 10000) if stage == "A" else ("A", 6000)
    if parent.get("method") != expected_parent:
        raise ValueError(f"{stage} must inherit {expected_parent}, not {parent.get('method')}")
    if parent.get("completed_update") != parent.get("best_update"):
        raise ValueError("The parent must be the full-Val-selected best checkpoint")
    if (summary.get("method") != expected_parent
            or not str(summary.get("status", "")).startswith("TRAINING_COMPLETE")
            or summary.get("completed_update") != budget
            or summary.get("best_update") != parent["completed_update"]):
        raise ValueError("Parent completion summary does not identify the finished selected best")
    if summary.get("test_used") is not False:
        raise ValueError("Parent summary must explicitly preserve Test1000 lock")
    if stage == "A" and parent.get("config", {}).get("objective") != "clean_only":
        raise ValueError("A requires the distinct clean-only D0")
    if stage == "B" and parent.get("config", {}).get("objective") != "precomputed_current_g":
        raise ValueError("B requires the authorized current-g precomputed A objective")


def transfer_model(parent, stage, *, seed=20261021, config=None):
    """Return a fresh module; never mutate parent tensors or optimizer state."""
    if stage == "A":
        model = transfer(parent, seed=seed, config=config)
        inherited = {"backbone." + k: v for k, v in model.backbone.state_dict().items()}
        if set(inherited) != set(parent["model"]):
            raise RuntimeError("D0 backbone/RGB key set changed during A inheritance")
        mismatches = [k for k, v in inherited.items() if not torch.equal(v, parent["model"][k])]
        if mismatches:
            raise RuntimeError(f"D0 backbone/RGB values changed: {mismatches[:10]}")
        if any(torch.count_nonzero(p).item() for film in model.time.films for p in film.parameters()):
            raise RuntimeError("A FiLM output scale/shift must be exactly zero at initialization")
        return model, {"inherited_tensors": len(inherited), "backbone_rgb_exact_copy": True,
                       "neutral_time": True, "new_adapter_seed": seed,
                       "raw_head": "seeded PyTorch default Conv2d initialization"}
    if stage != "B":
        raise ValueError("Stage must be A or B")
    if config is None:
        model = build_model(seed=seed, talking_backend="conv2d_equivalent")
    else:
        from models.v5_concertormer import V5Concertormer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = V5Concertormer(config=config, talking_backend="conv2d_equivalent")
    model.load_state_dict(parent["model"], strict=True)
    copied = model.state_dict()
    if set(copied) != set(parent["model"]) or any(
            not torch.equal(v, parent["model"][k]) for k, v in copied.items()):
        raise RuntimeError("B must inherit all A tensors exactly, including time and raw head")
    if not any(k.startswith("time.") for k in copied) or not any(k.startswith("raw_head.") for k in copied):
        raise RuntimeError("Missing inherited time/raw-xi modules")
    return model, {"inherited_tensors": len(copied), "all_model_tensors_exact_copy": True,
                   "time_and_raw_inherited_without_reset": True,
                   "new_adapter_seed": None}


def clean_parity(model, parent, images, *, device, config=None):
    if config is None:
        clean = build_clean_model(talking_backend="conv2d_equivalent")
    else:
        from models.clean_concertormer import CleanConcertormer
        clean = CleanConcertormer(config=config, talking_backend="conv2d_equivalent")
    clean.load_state_dict(parent["model"], strict=True)
    clean.to(device).eval()
    model.to(device).eval()
    images = images.to(device=device, dtype=torch.float32)
    errors = {}
    for precision in ("fp32", "bf16"):
        with torch.no_grad(), network_context(device, precision):
            expected = clean(images)
            actual, _ = model(images, torch.zeros(len(images), device=device))
        delta = (actual.float() - expected.float()).abs()
        if not torch.isfinite(actual).all() or not torch.allclose(
                actual.float(), expected.float(), rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"Actual parent first-clean parity failed in {precision}: {delta.max().item()}")
        errors[precision] = {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
                             "rtol": 1e-5, "atol": 1e-6}
    del clean, actual, expected, images
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("A", "B"), required=True)
    for key in ("parent-checkpoint", "parent-summary", "output", "report", "dataset"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--device", default="npu:0")
    args = p.parse_args(argv)
    report = {"schema": "v5_alignment_initialization_v1", "stage": args.stage,
              "status": "RUNNING", "formal_optimizer_updates": 0,
              "started_unix": time.time(), "device": args.device,
              "test_used": False, "optimizer_and_scheduler_reset": True}
    output = Path(args.output)
    if output.exists() or Path(args.report).exists():
        raise FileExistsError("Initialization output/report already exists; inspect it before retrying")
    try:
        parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
        summary = json.loads(Path(args.parent_summary).read_text())
        validate_parent(parent, summary, args.stage)
        model, inheritance = transfer_model(parent, args.stage, seed=args.seed)
        report.update(inheritance=inheritance,
                      parent_checkpoint_sha256=file_sha256(args.parent_checkpoint),
                      parent_summary_sha256=file_sha256(args.parent_summary),
                      parent_best_update=parent["completed_update"])
        if args.stage == "A":
            from runtime.environment import configure
            from data_pipeline.dataset import V5Dataset
            report["environment"] = configure(args.device)
            dataset = V5Dataset(args.dataset, "val", require_ready=True, allow_test=False)
            samples = [dataset[i] for i in (0, 4)]
            images = torch.stack([s["observation"] for s in samples])
            report["first_clean_parity"] = clean_parity(model, parent, images, device=args.device)
            report["parity_input_indices"] = [0, 4]
        initial = {"schema": "v5_alignment_initial_weights_v1", "target_stage": args.stage,
                   "method": parent["method"], "completed_update": parent["completed_update"],
                   "best_update": parent["best_update"], "model": {
                       k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "counters": parent.get("counters", {}), "optimizer_reset": True,
                   "provenance": {**inheritance, "parent_sha256": report["parent_checkpoint_sha256"],
                                  "parent_summary_sha256": report["parent_summary_sha256"]}}
        atomic_torch_save(initial, output)
        report.update(initial_checkpoint_sha256=file_sha256(output),
                      status="INITIALIZATION_PASS" if args.stage == "B" or args.device.startswith("npu") else "CPU_ONLY_PASS",
                      device_parity_executed=args.stage == "A",
                      ready_for_formal_training=False,
                      remaining_check="stage-specific full six-card update, gradients and exact resume")
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_unix"] = time.time()
        atomic_json(report, args.report)
    print(json.dumps({k: report[k] for k in ("status", "stage", "initial_checkpoint_sha256")}), flush=True)


if __name__ == "__main__":
    main()

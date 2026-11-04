#!/usr/bin/env python3
"""Compare original/equivalent talking convolutions on completed TRAIN Y4.

Forward-only diagnostic: no optimizer, backward, new data, validation, or test.
The same loaded U0 model is switched between explicit operator backends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_model
from runtime.environment import configure, memory_stats, synchronize, SIM_SHA256


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(tensor):
    x = tensor.detach().cpu().double()
    finite = bool(torch.isfinite(x).all())
    if not finite:
        raise FloatingPointError("Backend comparison encountered nonfinite values")
    return {"shape": list(x.shape), "dtype": str(tensor.dtype), "finite": finite,
            "min": x.min().item(), "max": x.max().item(), "mean": x.mean().item(),
            "std": x.std(unbiased=False).item(), "rms": x.square().mean().sqrt().item()}


def differences(reference, candidate):
    a, b = reference.double(), candidate.double()
    delta = a-b
    denominator = torch.linalg.vector_norm(a).item()
    return {"max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item(),
            "relative_l2": torch.linalg.vector_norm(delta).item()/max(denominator, 1e-30),
            "reference_l2": denominator, "byte_equal": torch.equal(reference, candidate)}


def completed_train_samples(dataset, count):
    """Select distinct completed train sources and verify exact source/Y4 bytes."""
    dataset = Path(dataset)
    freeze_path = dataset/"SOURCE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    for name in ("sample_plan", "source_manifest"):
        if sha(dataset/f"{name}.jsonl") != freeze[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
    identity_path = dataset/"GENERATION_IDENTITY.json"
    identity = json.loads(identity_path.read_text())
    if identity["simulator_sha256"] != SIM_SHA256 or identity["G"] != 4:
        raise ValueError("Y4 generation is not the accepted V5 G4 Simulator")
    if identity["source_freeze_sha256"] != sha(freeze_path):
        raise ValueError("Generation/source freeze identity mismatch")
    sources = {r["source_id"]: r for r in map(json.loads, (dataset/"source_manifest.jsonl").read_text().splitlines())}
    selected, seen_sources = [], set()
    for row in map(json.loads, (dataset/"sample_plan.jsonl").read_text().splitlines()):
        if row["split"] != "train" or row["source_id"] in seen_sources:
            continue
        record_path = dataset/row["record_ref"]
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        if row["G"] != 4 or record["sample_plan_row"] != row:
            raise ValueError("Completed train record does not match frozen G4 plan")
        if record["generation_identity_sha256"] != sha(identity_path):
            raise ValueError("Completed train record generation hash mismatch")
        source = sources[row["source_id"]]
        if source["split"] != "train" or source["clean"]["ref"] != row["clean_ref"]:
            raise ValueError("Selected clean is not the matching train source")
        for ref, expected in ((row["clean_ref"], source["clean"]["sha256"]),
                              (row["observation_ref"], record["observation_sha256"])):
            if sha(dataset/ref) != expected:
                raise ValueError(f"Selected array file hash mismatch: {ref}")
        clean = np.load(dataset/row["clean_ref"], allow_pickle=False)
        observed = np.load(dataset/row["observation_ref"], allow_pickle=False)
        for array in (clean, observed):
            if array.dtype != np.float32 or array.shape != (3,256,256) or not np.isfinite(array).all():
                raise ValueError("Selected array must be finite FP32 CHW256")
        selected.append((row, clean, observed, {"record_sha256":sha(record_path),
                        "clean_file_sha256":source["clean"]["sha256"],
                        "observation_file_sha256":record["observation_sha256"]}))
        seen_sources.add(row["source_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} distinct completed train sources, found {len(selected)}")
    return selected, {"source_freeze_sha256":sha(freeze_path),
                      "generation_identity_sha256":sha(identity_path),
                      "sample_plan_sha256":freeze["sample_plan_sha256"],
                      "all_selected_split":"train", "all_selected_G":4,
                      "distinct_source_count":len(seen_sources),
                      "full_dataset_ready_required":False}


def compare_model(model, observation, target, *, t, device):
    """CPU-readable outputs avoid retaining two large raw predictions on NPU."""
    if model.training:
        raise ValueError("Backend comparison must use model.eval()")
    before_versions = {name:p._version for name,p in model.named_parameters()}
    outputs, backend_records = {}, {}
    for backend in ("conv3d", "conv2d_equivalent"):
        model.set_talking_backend(backend)
        if str(device).startswith("npu"):
            torch.npu.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            clean, raw = model(observation, t)
        synchronize(device)
        duration = time.perf_counter()-start
        values = {"clean":clean.detach().cpu(),
                  "raw_zernike":raw.epsilon_zernike.detach().cpu(),
                  "raw_tilt":raw.epsilon_tilt.detach().cpu()}
        if any(v.dtype != torch.float32 or v.requires_grad for v in values.values()):
            raise RuntimeError("Comparison output must remain FP32 without a gradient graph")
        backend_records[backend] = {"seconds":duration,
            "outputs":{name:stats(v) for name,v in values.items()}, **memory_stats(device)}
        outputs[backend] = values
        del clean,raw
    after_versions = {name:p._version for name,p in model.named_parameters()}
    if before_versions != after_versions:
        raise RuntimeError("Model parameters were mutated during forward-only comparison")
    return {"observation_Y4":stats(observation), "target_clean":stats(target),
            "t":t.detach().cpu().tolist(), "backends":backend_records,
            "differences":{name:differences(outputs["conv3d"][name], outputs["conv2d_equivalent"][name])
                           for name in outputs["conv3d"]}, "parameter_versions_unchanged":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu")
    p.add_argument("--initial",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--samples",type=int,default=3,help="Distinct completed TRAIN sources, evaluated one at a time")
    p.add_argument("--time",type=float,default=0.)
    args=p.parse_args()
    if not 1 <= args.samples <= 8 or not 0 <= args.time <= 1:
        raise ValueError("Require 1..8 train samples and 0 <= t <= 1")
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing comparison report")
    report={"status":"RUNNING", "arguments":vars(args), "started_unix":time.time(),
            "scope":"forward-only backend comparison on completed train Y4",
            "formal_optimizer_updates":0,"backward_executed":False,"test_used":False,"samples":[]}
    def save():path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    try:
        selected,report["data_identity"]=completed_train_samples(args.dataset,args.samples)
        report["environment"]=configure(args.device)
        initial=torch.load(args.initial,map_location="cpu",weights_only=False)
        model=build_model(activation_checkpointing=False,talking_backend="conv3d")
        model.load_state_dict(initial["model"],strict=True)
        report["initial_sha256"]=sha(args.initial)
        report["initial_provenance"]=initial.get("provenance")
        report["initial_formal_optimizer_updates"]=initial.get("formal_optimizer_updates")
        if initial.get("formal_optimizer_updates") != 0:
            raise ValueError("This comparison requires shared untrained adapted U0")
        del initial
        model.to(device=args.device,dtype=torch.float32).eval()
        for row,clean,observed,identities in selected:
            y=torch.from_numpy(observed[None]).to(args.device)
            c=torch.from_numpy(clean[None]).to(args.device)
            t=torch.tensor([args.time],dtype=torch.float32,device=args.device)
            result=compare_model(model,y,c,t=t,device=args.device)
            result.update(sample_id=row["sample_id"],source_id=row["source_id"],**identities)
            report["samples"].append(result);save()
        # Completion is evidence, not a hidden tolerance-based model acceptance.
        report["status"]="COMPARISON_COMPLETE"
        report["tolerance_gate_applied"]=False
    except Exception as exc:
        report.update(status="FAILED",error=repr(exc),traceback=traceback.format_exc())
    report["finished_unix"]=time.time();save()
    print(json.dumps(report,allow_nan=False),flush=True)
    return 0 if report["status"]=="COMPARISON_COMPLETE" else 1


if __name__=="__main__":raise SystemExit(main())

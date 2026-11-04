"""Bounded, resumable physical intermediate-state production; no model training."""
import argparse
import collections
import fcntl
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from data_pipeline.common import atomic, digest, npy_bytes, rows, sha, verify_array, write_json, write_rows
from .identity import ROOT, SCHEMA, check_runtime, load_base, load_legacy_generator
from .plans import DEFAULT_SEED, cache_directory, prepare_plan


def require_audit(cache_root, base):
    for name in ("DATA_AUDIT.json", "RAW_REPLAY.json"):
        path = Path(cache_root) / "audit" / name
        if not path.is_file():
            raise RuntimeError(f"Required pre-generation audit not available: {path}")
        audit = json.loads(path.read_text())
        if audit["status"] != "PASS" or audit["base_identity"] != base:
            raise ValueError(f"Data audit identity/status mismatch: {name}")
    raw = json.loads((Path(cache_root) / "audit/RAW_REPLAY.json").read_text())
    if raw["counts"] != base["sample_counts"] or not raw["all_hashes_exact"]:
        raise ValueError("All original realization seeds must replay exactly before production")


def _check_record(directory, row, base_root, generation_hash, base_record):
    record = json.loads((directory / row["record_ref"]).read_text())
    if record["plan_row"] != row or record["generation_identity_sha256"] != generation_hash:
        raise ValueError("Cache record identity changed")
    if record["raw_sha256"] != base_record["raw_sha256"]:
        raise ValueError("Cache realization no longer matches original Y4")
    root = Path(base_root) if row["observation_root"] == "base" else directory
    output = verify_array(root / row["observation_ref"], record["observation_sha256"])
    if digest(output.tobytes()) != record["observation_pixel_sha256"]:
        raise ValueError("Cached pixel hash mismatch")
    return record


def generate(base_root, cache_root, simulator_root, *, split, cycle=0, seed=DEFAULT_SEED,
             device="cpu", worker_index=0, worker_count=1, max_states=0):
    """The caller owns device scheduling. Never launches other workers/devices."""
    import torch
    if not 0 <= worker_index < worker_count:
        raise ValueError("Invalid generator sharding")
    root, cache_root = Path(base_root), Path(cache_root)
    base, sources, base_samples, base_records = load_base(root)
    require_audit(cache_root, base)
    prepare_plan(root, cache_root, split, cycle, seed)
    directory = cache_directory(cache_root, split, cycle)
    plan = rows(directory / "PLAN.jsonl")
    helper = load_legacy_generator()
    frozen_generation = json.loads((root / "GENERATION_IDENTITY.json").read_text())
    if sha(ROOT / "data_pipeline/generate.py") != frozen_generation["generator_file_sha256"]:
        raise ValueError("Frozen raw RNG generator helper changed")
    sim, Realization, runtime = helper.load_runtime(simulator_root, device, cache_root / "compiler_cache" / device.replace(":", "_"))
    check_runtime(runtime, base)
    generation = {"schema": SCHEMA, "cache_identity_sha256": sha(cache_root / "CACHE_IDENTITY.json"),
                  "base_generation_sha256": base["files_sha256"]["GENERATION_IDENTITY.json"],
                  "runtime": runtime, "plan_sha256": sha(directory / "PLAN.jsonl"),
                  "generator_sha256": sha(Path(__file__)),
                  "time_plan_implementation_sha256": sha(Path(__file__).with_name("plans.py")),
                  "raw_policy": "Exact original CPU B=1 raw helper and per-field SHA; never draw a new realization",
                  "output": "unclamped float32 CHW npy", "Y4_replay_max_abs_threshold": 2e-5}
    write_json(directory / "GENERATION_IDENTITY.json", generation)
    generation_hash = sha(directory / "GENERATION_IDENTITY.json")
    # A replay receipt is reusable between cache cycles only with identical physical runtime.
    physics_identity = {"base": base, "runtime": runtime, "raw_generator_sha256": frozen_generation["generator_file_sha256"]}
    base_by_id = {r["sample_id"]: r for r in base_samples}
    source_by_id = {r["source_id"]: r for r in sources}
    by_source = collections.defaultdict(list)
    for row in plan:
        by_source[row["source_id"]].append(row)
    started = time.perf_counter()
    made = resumed = replayed = 0
    for source_position, (source_id, source_rows) in enumerate(by_source.items()):
        if source_position % worker_count != worker_index:
            continue
        lock_path = directory / "locks" / (source_id + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            pending = []
            for row in source_rows:
                if (directory / row["record_ref"]).is_file():
                    _check_record(directory, row, root, generation_hash, base_records[row["base_sample_id"]])
                    resumed += 1
                else:
                    pending.append(row)
            if not pending:
                continue
            source = source_by_id[source_id]
            clean = verify_array(root / source["clean"]["ref"], source["clean"]["sha256"])
            by_base = collections.defaultdict(list)
            for row in pending:
                by_base[row["base_sample_id"]].append(row)
            with torch.no_grad(), torch.autocast(device_type=str(device).split(":")[0], enabled=False):
                image = torch.from_numpy(clean[None]).to(device)
                prepared = sim.prepare_image(image)
                for sample_id, members in by_base.items():
                    raw, raw_hash = helper.raw_noise(base_by_id[sample_id])
                    original_record = base_records[sample_id]
                    if raw_hash != original_record["raw_sha256"]:
                        raise ValueError(f"Original raw realization did not replay: {sample_id}")
                    xi = Realization(*(x.to(device) for x in raw))
                    replay_path = cache_root / "audit/y4_replay" / f"{sample_id}.json"
                    if replay_path.is_file():
                        replay = json.loads(replay_path.read_text())
                        if replay["physics_identity"] != physics_identity or replay["raw_sha256"] != raw_hash:
                            raise ValueError("Prior Y4 replay receipt has different physics/realization")
                        if replay["base_observation_sha256"] != original_record["observation_sha256"] or replay["max_abs"] > 2e-5:
                            raise ValueError("Prior Y4 replay receipt is invalid")
                    else:
                        y4_result = sim.render_prepared(prepared, xi, 4.0)
                        helper.sync(device)
                        y4 = y4_result.output[0].cpu().numpy()
                        original = verify_array(root / base_by_id[sample_id]["observation_ref"], original_record["observation_sha256"])
                        error = float(np.abs(y4 - original).max())
                        if not np.isfinite(error) or error > 2e-5:
                            raise ValueError(f"Original Y4 replay failed for {sample_id}: max_abs={error}")
                        replay = {"schema": SCHEMA, "physics_identity": physics_identity, "sample_id": sample_id,
                                  "raw_sha256": raw_hash, "base_observation_sha256": original_record["observation_sha256"],
                                  "max_abs": error, "exact": bool(np.array_equal(y4, original)),
                                  "threshold": 2e-5, "test_model_evaluation": False}
                        write_json(replay_path, replay)
                        replayed += 1
                        del y4_result, y4, original
                    for row in members:
                        tick = time.perf_counter()
                        if row["is_endpoint"]:
                            output = verify_array(root / row["observation_ref"], original_record["observation_sha256"])
                            file_hash = original_record["observation_sha256"]
                        else:
                            result = sim.render_prepared(prepared, xi, row["strength"])
                            helper.sync(device)
                            output = result.output[0].cpu().numpy()
                            if output.dtype != np.float32 or output.shape != (3, 256, 256) or not np.isfinite(output).all():
                                raise ValueError("Invalid physical intermediate image")
                            encoded = npy_bytes(output)
                            atomic(directory / row["observation_ref"], encoded)
                            file_hash = digest(encoded)
                            del result, encoded
                        record = {"schema": SCHEMA, "plan_row": row, "generation_identity_sha256": generation_hash,
                                  "raw_sha256": raw_hash, "Y4_replay_receipt_sha256": sha(replay_path),
                                  "observation_sha256": file_hash, "observation_pixel_sha256": digest(output.tobytes()),
                                  "minimum": float(output.min()), "maximum": float(output.max()),
                                  "finite": True, "device": str(device), "host": socket.gethostname(), "pid": os.getpid(),
                                  "elapsed_seconds": time.perf_counter() - tick}
                        write_json(directory / row["record_ref"], record)
                        made += 1
                        del output
                        if max_states and made >= max_states:
                            break
                    del raw, xi
                    if max_states and made >= max_states:
                        break
                del image, prepared
            progress = {"status": "GENERATING", "split": split, "cache_cycle": cycle, "device": str(device),
                        "worker_index": worker_index, "worker_count": worker_count, "made": made, "resumed": resumed,
                        "Y4_replayed": replayed, "elapsed_seconds": time.perf_counter() - started,
                        "last_source_id": source_id, "pid": os.getpid()}
            write_json(directory / "workers" / f"worker_{worker_index:03d}.json", progress, frozen=False)
            if made % 1000 < len(source_rows):
                print(json.dumps(progress), flush=True)
            if max_states and made >= max_states:
                break
    summary = {"status": "WORKER_FINISHED", "split": split, "cache_cycle": cycle, "device": str(device),
               "worker_index": worker_index, "worker_count": worker_count, "made": made, "resumed": resumed,
               "Y4_replayed": replayed, "elapsed_seconds": time.perf_counter() - started,
               "peak_allocated_bytes": torch.npu.max_memory_allocated(device) if str(device).startswith("npu") else None}
    write_json(directory / "workers" / f"worker_{worker_index:03d}.json", summary, frozen=False)
    return summary


def finalize(base_root, cache_root, split, cycle=0):
    root, cache_root = Path(base_root), Path(cache_root)
    base, _, _, original_records = load_base(root)
    require_audit(cache_root, base)
    directory = cache_directory(cache_root, split, cycle)
    identity = json.loads((directory / "PLAN_IDENTITY.json").read_text())
    plan = rows(directory / "PLAN.jsonl")
    if sha(directory / "PLAN.jsonl") != identity["plan_sha256"]:
        raise ValueError("Plan changed after freeze")
    generation_hash = sha(directory / "GENERATION_IDENTITY.json")
    records, replay_errors = [], {}
    for row in plan:
        original = original_records[row["base_sample_id"]]
        record = _check_record(directory, row, root, generation_hash, original)
        replay_path = cache_root / "audit/y4_replay" / f"{row['base_sample_id']}.json"
        if sha(replay_path) != record["Y4_replay_receipt_sha256"]:
            raise ValueError("Y4 replay receipt changed")
        replay = json.loads(replay_path.read_text())
        if replay["raw_sha256"] != original["raw_sha256"] or replay["max_abs"] > 2e-5:
            raise ValueError("Invalid original realization/Y4 replay")
        replay_errors[row["base_sample_id"]] = replay["max_abs"]
        records.append(record)
    write_rows(directory / "MANIFEST.jsonl", records)
    result = {"schema": SCHEMA, "status": "READY", "split": split, "cache_cycle": cycle,
              "cache_identity_sha256": sha(cache_root / "CACHE_IDENTITY.json"),
              "plan_sha256": sha(directory / "PLAN.jsonl"), "manifest_sha256": sha(directory / "MANIFEST.jsonl"),
              "generation_identity_sha256": generation_hash, "records": len(records),
              "endpoints": sum(r["is_endpoint"] for r in plan), "internal": sum(not r["is_endpoint"] for r in plan),
              "time_bin_counts": dict(sorted(collections.Counter(r["time_slot"] for r in plan if not r["is_endpoint"]).items())),
              "original_realizations_verified": len(replay_errors), "Y4_replay_max_abs": max(replay_errors.values(), default=0),
              "source_count": len({r["source_id"] for r in plan}), "test_model_evaluation": False}
    write_json(directory / "READY.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--cycle", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--simulator-root", default="/data2/yanglin/v5_v1_simulator_20261020")
    p.add_argument("--device", default="cpu")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--worker-count", type=int, default=1)
    p.add_argument("--max-states", type=int, default=0)
    p.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    if a.finalize:
        result = finalize(a.base_root, a.cache_root, a.split, a.cycle)
    else:
        result = generate(a.base_root, a.cache_root, a.simulator_root, split=a.split, cycle=a.cycle,
                          seed=a.seed, device=a.device, worker_index=a.worker_index,
                          worker_count=a.worker_count, max_states=a.max_states)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

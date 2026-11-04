"""Read live metadata and process identities; never load a model or checkpoint.

Usage: python3 scripts/inspect_live_campaign.py
Writes a timestamped evidence/heartbeat snapshot and prints a compact summary.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc), "path": str(path)}


def same_process(actor):
    try:
        fields = (Path("/proc") / str(actor["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(actor["start_ticks"])
    except (OSError, KeyError, IndexError):
        return False


def last_update(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 24000))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if "update" in row and "cursor" in row:
                return {k: row.get(k) for k in [
                    "update", "extension_new_updates", "extension_new_presentations",
                    "started_unix", "completed_unix", "cursor", "lr", "loss", "grad_norm",
                ]}
        except (ValueError, TypeError):
            pass
    return {"read_error": "No complete update row"}


def inspect():
    now = time.time()
    result = {"unix": now, "root_status": read(ROOT / "STATUS.json"),
              "actors": {}, "branches": {}, "source_manifests": {}}
    for name in ["ACTIVE_CONTROLLER", "ACTIVE_CPU_FINALIZER"]:
        actor = read(ROOT / "evidence" / (name + ".json"))
        result["actors"][name] = {k: actor.get(k) for k in ["pid", "start_ticks"]}
        result["actors"][name]["alive_same_start"] = same_process(actor)
    for branch, actor in result["root_status"].get("actors", {}).items():
        result["actors"]["torchrun_" + branch] = {
            "pid": actor["pid"], "start_ticks": actor["start_ticks"],
            "alive_same_start": same_process(actor),
        }
    for branch in ["original", "improved"]:
        run = ROOT / "runs" / branch
        pointer = read(run / "LATEST.json")
        checkpoint = Path(pointer.get("path", "/absent"))
        row = {"status": read(run / "STATUS.json"), "latest_checkpoint": pointer,
               "checkpoint_exists": checkpoint.is_file(),
               "checkpoint_bytes_match": checkpoint.is_file() and checkpoint.stat().st_size == pointer.get("bytes"),
               "ranks": [], "last_updates_by_rank": {}, "summaries": [], "active_evaluations": []}
        result["branches"][branch] = row
        for path in sorted(run.glob("RANK_*.json")):
            actor = read(path)
            row["ranks"].append({k: actor.get(k) for k in ["rank", "pid", "start_ticks"]}
                               | {"alive_same_start": same_process(actor)})
        for path in sorted(run.glob("UPDATES_rank*.jsonl")):
            row["last_updates_by_rank"][path.name] = last_update(path)
        for directory in sorted((run / "validation").iterdir()):
            if not directory.is_dir():
                continue
            summary_path = directory / "SUMMARY.json"
            if summary_path.exists():
                summary = read(summary_path)
                row["summaries"].append({"path": str(summary_path.resolve()), **{
                    k: summary.get(k) for k in ["update", "weights", "full", "count", "by_g"]}})
            else:
                groups = list((directory / "groups").glob("G*/*.json"))
                progress = list(directory.glob("PROGRESS_rank*.json"))
                timestamps = [path.stat().st_mtime for path in groups + progress]
                timestamps.append(directory.stat().st_mtime)
                row["active_evaluations"].append({
                    "directory": str(directory), "saved_groups": len(groups),
                    "saved_images": 4 * len(groups), "last_artifact_age": now - max(timestamps),
                    "progress": [read(path) for path in progress],
                })
        row["summaries"].sort(key=lambda value: (value.get("update") or -1, value.get("weights") or ""))
        row["failures"] = [read(path) for path in run.glob("FAILURE*.json")]
    for name in ["RUN_SOURCE.json", "POSTPROCESS_SOURCE.json"]:
        manifest = read(ROOT / "protocol" / name)
        result["source_manifests"][name] = {
            rel: (ROOT / rel).is_file() and hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
            for rel, expected in manifest["files"].items()
        }
    result["cpu_finalizer_status"] = read(ROOT / "evidence/CPU_FINALIZER_STATUS.json")
    result["stop_request_present"] = (ROOT / "STOP_REQUEST.json").exists()
    result["model_validation_complete"] = (ROOT / "TRAINING_AND_MODEL_VALIDATION_COMPLETE.json").exists()
    result["disk_free_GiB"] = shutil.disk_usage(ROOT).free / 2**30
    result["LIVE_ACTORS_age"] = now - (ROOT / "evidence/LIVE_ACTORS.json").stat().st_mtime
    result["inspector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = ROOT / "evidence/heartbeat" / (str(int(now)) + ".json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    compact = {k: v for k, v in result.items() if k not in ["branches", "root_status"]}
    compact["root_status"] = result["root_status"].get("status")
    compact["branches"] = {}
    for branch, row in result["branches"].items():
        short = {k: v for k, v in row.items() if k not in ["summaries", "last_updates_by_rank"]}
        short["last_update"] = row["last_updates_by_rank"].get("UPDATES_rank0.jsonl")
        short["rank_update_numbers"] = {k: v.get("update") for k, v in row["last_updates_by_rank"].items()}
        short["latest_summaries"] = [{k: v for k, v in entry.items() if k != "by_g"}
                                    | {"g4": (entry.get("by_g") or {}).get("4")}
                                    for entry in row["summaries"][-2:]]
        compact["branches"][branch] = short
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    inspect()

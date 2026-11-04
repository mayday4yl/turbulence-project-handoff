#!/usr/bin/env python3
"""Bounded supervisor for the authorized de36 physical NPUs 0--3.

Launches argv without shell interpretation. Signals target only groups whose
members were proved to descend from our worker, including torchrun's setsid
ranks. PID plus kernel starttime protects ownership against PID reuse.
Importing this module is inert, so the telemetry parser can be tested on CPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


PROJECT = Path("/data2/yanglin/v031_followup_20260918")
CARDS = (0,)
INTERVAL_SECONDS = 5.0


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_info(pid):
    """Read identity/ancestry atomically from one Linux /proc stat record."""
    value = Path(f"/proc/{int(pid)}/stat").read_text()
    fields = value[value.rfind(")") + 2:].split()
    return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
            "pgid": int(fields[2]), "session": int(fields[3]),
            "starttime_ticks": int(fields[19])}


def process_table():
    result = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            info = process_info(int(entry.name))
            result[info["pid"]] = info
        except (OSError, ValueError, IndexError):
            continue  # Unknown/unreadable identities are never owned or signalled.
    return result


class OwnedProcessTree:
    """Prove descendants before accepting their independent process groups.

    A registered PID remains ours after reparenting only while its kernel
    starttime is unchanged. Unknown orphan groups are deliberately not adopted.
    """
    def __init__(self, root_info):
        self.root_pid = root_info["pid"]
        self.identities = {root_info["pid"]: root_info["starttime_ticks"]}
        self.discovery = {root_info["pid"]: {**root_info, "discovered_from_parent": None}}
        self.current = {root_info["pid"]: dict(root_info)}

    def refresh(self, table=None):
        table = process_table() if table is None else table
        verified = {pid for pid, start in self.identities.items()
                    if pid in table and table[pid]["starttime_ticks"] == start}
        changed = True
        while changed:
            changed = False
            for pid, info in table.items():
                parent = info["ppid"]
                if pid in verified or parent not in verified:
                    continue
                if info["starttime_ticks"] < table[parent]["starttime_ticks"]:
                    continue
                # A PID reuse is a new identity, accepted only after fresh
                # ancestry proof from a currently verified parent.
                self.identities[pid] = info["starttime_ticks"]
                self.discovery[pid] = {**info, "discovered_from_parent": parent}
                verified.add(pid)
                changed = True
        self.current = {pid: table[pid] for pid in verified}
        return self.current

    def owns(self, info):
        return (info.get("starttime_ticks") is not None and
                self.identities.get(info["pid"]) == info["starttime_ticks"])

    def groups(self):
        # Zombies have already stopped running and need no process-group signal.
        return sorted({info["pgid"] for info in self.current.values()
                       if info["state"] != "Z"})

    def evidence(self):
        return {"root_pid": self.root_pid, "known_identities": list(self.discovery.values()),
                "live_processes": list(self.current.values()), "live_groups": self.groups()}


def parse_npu_smi(raw, cards=CARDS):
    """Parse the real 25.5.1 table, rejecting missing/ambiguous card coverage."""
    health_rows = {}
    active_card = None
    for line in raw.splitlines():
        row = re.match(r"^\|\s*(\d+)\s+([^\s|]+)\s*\|\s*([A-Za-z_]+)\s*\|([^|]+)\|", line)
        if row:
            active_card = int(row[1])
            if active_card in health_rows:
                raise ValueError(f"Duplicate card health row: {active_card}")
            values = row[4].split()
            health_rows[active_card] = {
                "card": active_card, "name": row[2], "health": row[3],
                "power_w": float(values[0]) if values else None,
                "temperature_c": float(values[1]) if len(values) > 1 else None,
                "processes": [],
            }
            continue
        memory = re.match(r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|", line)
        if memory and active_card is not None:
            health_rows[active_card].update({
                "aicore_percent": int(memory[1]), "hbm_used_mb": int(memory[2]),
                "hbm_total_mb": int(memory[3]),
            })
    process_coverage = set()
    for match in re.finditer(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|", raw, re.M):
        card = int(match[1])
        process_coverage.add(card)
        if card in health_rows:
            health_rows[card]["processes"].append({
                "pid": int(match[3]), "chip": int(match[2]),
                "name": match[4].strip(), "memory_mb": int(match[5]),
            })
    empty = {int(match[1]) for match in re.finditer(r"No running processes found in NPU\s+(\d+)\s*\|", raw)}
    contradictory = process_coverage & empty
    if contradictory & set(cards):
        raise ValueError(f"Contradictory process table: {sorted(contradictory)}")
    for card in cards:
        if card not in health_rows or card not in process_coverage | empty:
            raise ValueError(f"Missing health/process table for selected card {card}")
        if "hbm_total_mb" not in health_rows[card]:
            raise ValueError(f"Missing memory table for selected card {card}")
    return {"utc": utc_now(), "cards": [health_rows[c] for c in cards]}


def snapshot(timeout=20):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                            timeout=timeout, check=True)
    parsed = parse_npu_smi(result.stdout)
    for card in parsed["cards"]:
        current = []
        for process in card["processes"]:
            try:
                process.update(process_info(process["pid"]))
            except (FileNotFoundError, ProcessLookupError):
                continue  # It exited between the npu-smi snapshot and /proc read.
            except OSError as exc:
                process["pgid"] = None  # Unknown ownership fails closed below.
                process["ownership_error"] = repr(exc)
            try:
                process["cwd"] = os.readlink(f"/proc/{process['pid']}/cwd")
            except OSError:
                process["cwd"] = None
            current.append(process)
        card["processes"] = current
    return result.stdout, parsed


def check_snapshot(snap, pgid=None, memory_limit_gib=58.0, ownership=None):
    for card in snap["cards"]:
        if card["health"] != "OK":
            raise RuntimeError(f"NPU {card['card']} is unhealthy: {card['health']}")
        if pgid is None:
            if card["processes"]:
                raise RuntimeError(f"NPU {card['card']} is occupied; no worker launched")
            continue
        external = [p for p in card["processes"]
                    if not (ownership.owns(p) if ownership is not None else p.get("pgid") == pgid)]
        if external:
            raise RuntimeError(f"External or unowned process appeared on NPU {card['card']}; stopping only our group: {external}")
        own_mb = sum(p["memory_mb"] for p in card["processes"])
        card["own_process_memory_mb"] = own_mb
        if own_mb > memory_limit_gib * 1024:
            raise RuntimeError(f"NPU {card['card']} own allocation {own_mb} MiB exceeds {memory_limit_gib} GiB guard cap")


def stop_owned_groups(ownership, worker=None):
    """Signal only groups with a currently verified, non-reused own member."""
    events = []
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        ownership.refresh()
        groups = ownership.groups()
        if not groups:
            break
        for pgid in groups:
            # Revalidate immediately before every signal, rather than trusting
            # a PGID remembered before the leader exited or was reparented.
            ownership.refresh()
            members = [p for p in ownership.current.values()
                       if p["pgid"] == pgid and p["state"] != "Z"]
            if not members:
                continue
            try:
                os.killpg(pgid, sig)
                events.append({"utc": utc_now(), "pgid": pgid,
                               "signal": sig.name, "verified_members": members})
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + grace
        while ownership.groups() and time.monotonic() < deadline:
            if worker is not None:
                worker.poll()  # Reap the leader so its zombie cannot prolong cleanup.
            time.sleep(0.2)
            ownership.refresh()
    return events


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--memory-limit-gib", type=float, default=58.0)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("--command needs an executable and optional arguments")
    if args.timeout_seconds <= 0 or not 0 < args.memory_limit_gib <= 61:
        parser.error("timeout must be positive and the memory cap must be in (0, 61] GiB")
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(PROJECT.resolve()) or args.run_dir == PROJECT.resolve():
        parser.error(f"run-dir must be an isolated child directory of {PROJECT}")
    return args


def main(argv=None):
    args = arguments(argv)
    if (PROJECT / "STOP_REQUESTED").exists():
        raise RuntimeError("Project is stopped; worker launch forbidden")
    identity_path = PROJECT / "artifacts/run_manifest.json"
    if not identity_path.is_file():
        raise RuntimeError("Stage1 preparation identity is required before NPU work")
    identity = json.loads(identity_path.read_text())
    if identity.get("dataset_id") != "places_p2s_bridge_v03_overfit100_g4321_n4" or identity.get("method_version") != "V0.31":
        raise RuntimeError("Dataset identity conflict")
    if shutil.disk_usage(PROJECT).free < (1 << 40):
        raise RuntimeError("Data2 free space is below 1 TiB")
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    # A project-level lock prevents two of our guards racing for the same cards.
    lock = (PROJECT / ".guard_npu_0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "launch.json").exists() or (root / "guard_result.json").exists():
        raise RuntimeError("Run directory was already attempted; create a new attempt identity")
    worker = None
    ownership = None
    log = None
    reason = None
    cleanup = []
    started = time.monotonic()
    worker_started = None
    launch = {"utc": utc_now(), "guard_pid": os.getpid(), "host": os.uname().nodename,
              "cards": list(CARDS), "argv": args.command, "run_dir": str(root),
              "timeout_seconds": args.timeout_seconds, "memory_limit_gib": args.memory_limit_gib,
              "poll_interval_seconds": INTERVAL_SECONDS}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        code = root / "guard_code_snapshot"
        code.mkdir(exist_ok=False)
        hashes = {}
        for source in (Path(__file__).resolve(), PROJECT / "code/run_env.sh"):
            shutil.copy2(source, code / source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        atomic_json(code / "sha256.json", hashes)
        for index in range(2):
            raw, snap = snapshot()
            (root / f"preflight-{index}.txt").write_text(raw)
            atomic_json(root / f"preflight-{index}.json", snap)
            check_snapshot(snap)
            if index == 0:
                time.sleep(2)
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        env["OVERFIT100_RUN_DIR"] = str(root)
        log = (root / "worker.log").open("w")
        worker_started = time.monotonic()
        # Assign worker identity before handling a termination signal, so there
        # is no Popen/assignment interval in which cleanup lacks the new PGID.
        blocked = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            worker = subprocess.Popen(args.command, cwd=PROJECT, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True,
                                      preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
            ownership = OwnedProcessTree(process_info(worker.pid))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        launch.update({"worker_pid": worker.pid, "worker_pgid": worker.pid,
                       "worker_started_utc": utc_now()})
        atomic_json(root / "launch.json", launch)
        print(json.dumps({"launched": launch}), flush=True)
        next_telemetry = time.monotonic()
        while True:
            # Ancestry sampling is faster than NPU telemetry so torchrun ranks
            # and compiler workers are registered before parent termination.
            ownership.refresh()
            if worker.poll() is not None:
                break
            remaining = args.timeout_seconds - (time.monotonic() - worker_started)
            if remaining <= 0:
                raise TimeoutError("Declared worker wall-clock budget exhausted (including compilation)")
            if time.monotonic() < next_telemetry:
                time.sleep(max(0, min(0.5, remaining, next_telemetry - time.monotonic())))
                continue
            raw, snap = snapshot(timeout=min(20, max(0.1, remaining)))
            ownership.refresh()
            atomic_json(root / "owned_processes.json", ownership.evidence())
            snap["worker_elapsed_seconds"] = time.monotonic() - worker_started
            (root / "telemetry_latest.txt").write_text(raw)
            atomic_json(root / "telemetry_latest.json", snap)
            with (root / "telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(snap) + "\n")
            check_snapshot(snap, worker.pid, args.memory_limit_gib, ownership)
            atomic_json(root / "telemetry_latest.json", snap)
            next_telemetry = time.monotonic() + INTERVAL_SECONDS
        if worker.returncode:
            reason = f"Worker exited with return code {worker.returncode}"
    except BaseException as exc:
        reason = repr(exc)
    finally:
        # Prevent repeated terminal signals interrupting cleanup. Kill only our
        # group even if its torchrun leader has already exited but children live.
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if worker is not None:
            try:
                if ownership is not None:
                    cleanup = stop_owned_groups(ownership, worker)
                    atomic_json(root / "owned_processes.json", ownership.evidence())
            except Exception as exc:
                cleanup.append({"utc": utc_now(), "pgid": worker.pid, "cleanup_error": repr(exc)})
                reason = (reason + "; " if reason else "") + "Own-group cleanup failed: " + repr(exc)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                reason = reason or "Worker was not reaped after own-group cleanup"
        if log is not None:
            log.close()
        post = None
        for index in range(3):
            try:
                raw, post = snapshot()
                (root / "postflight.txt").write_text(raw)
                atomic_json(root / "postflight.json", post)
                if ownership is None or not any(ownership.owns(p) for c in post["cards"] for p in c["processes"]):
                    break
            except Exception as exc:
                post = {"snapshot_error": repr(exc)}
                break
            time.sleep(2)
        released = None if post is None or "snapshot_error" in post else (
            ownership is None or not any(ownership.owns(p) for c in post["cards"] for p in c["processes"]))
        if released is False:
            reason = reason or "Own NPU allocation remains visible after cleanup"
        elif released is None:
            reason = reason or "Postflight could not verify release of our NPU allocation"
        result = {"utc": utc_now(), "status": "failed" if reason else "completed",
                  "returncode": None if worker is None else worker.returncode,
                  "reason": reason, "elapsed_seconds": time.monotonic() - started,
                  "worker_elapsed_seconds": None if worker_started is None else time.monotonic() - worker_started,
                  "cleanup": cleanup, "own_npu_processes_released": released,
                  "ownership": None if ownership is None else ownership.evidence(),
                  "postflight": post, "launch": launch}
        atomic_json(root / "guard_result.json", result)
        print(json.dumps(result), flush=True)
        lock.close()
    return 1 if reason else 0


if __name__ == "__main__":
    sys.exit(main())

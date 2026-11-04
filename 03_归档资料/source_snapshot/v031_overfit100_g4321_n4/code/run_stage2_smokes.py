#!/usr/bin/env python3
"""One bounded sequential implementation of STAGE2_RUNTIME_SELECTION_PROTOCOL.

Run only after the parent has authorized and passed the initial zero-update DDP
check. This driver delegates every optimizer smoke to smoke_controller.py. It
never freezes a configuration, runs inference calibration, or starts formal
training. An unrelated failure stops the driver; repairs and new attempt tags
must be explicit. Importing this file never initializes an accelerator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import traceback


ROOT = Path("/data2/yanglin/v031_overfit100_g4321_n4")
MICROBATCHES = (1, 2, 4, 8)
REPORT_NAME = "artifacts/STAGE2_RUNTIME_SELECTION.json"
LEDGER_NAME = "artifacts/STAGE2_SMOKE_LEDGER.json"
INITIAL_DDP_NAME = "artifacts/stage2_ddp_gradients_mb1_01/results.json"


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evidence(path):
    return {"path": str(path), "sha256": sha256(path)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def successful_gradient_evidence(path, microbatch):
    result = read(path)
    require(result.get("status") == "passed", f"DDP normalization did not pass: {path}")
    require(result.get("restoration_optimizer_updates") == 0
            and result.get("smoke_optimizer_updates") == 0,
            "Gradient acceptance must have zero optimizer updates")
    ranks = result.get("rank_results", [])
    require(len(ranks) == 4, "DDP evidence must cover all four ranks")
    for rank, row in enumerate(ranks):
        require(row.get("status") == "passed" and row.get("rank") == rank
                and row.get("microbatch") == microbatch, "DDP evidence rank/config mismatch")
        require(row.get("optimizer_step_calls") == row.get("optimizer_step_attempts") == 0,
                "DDP evidence includes optimizer work")
        require(row.get("network_unchanged") is True and row.get("simulator_unchanged") is True,
                "DDP acceptance changed frozen state")
    require(all(row["global_start_ids"] == ranks[0]["global_start_ids"] for row in ranks),
            "DDP evidence rank sample identities differ")
    return result


def memory_failure_reason(guard, rank_reports, worker_log=""):
    """Only explicit allocator OOM or this guard's own 61-GiB cap can exclude.

    A peer HCCL failure after an explicit rank OOM is a secondary failure. Any
    ownership/health/release/cleanup problem remains blocking even with an OOM.
    """
    if not guard or guard.get("own_npu_processes_released") is not True:
        return None
    reason = guard.get("reason") or ""
    if any(token in reason.lower() for token in
           ["unhealthy", "external", "unowned", "cleanup failed", "postflight", "remains visible"]):
        return None
    if any("cleanup_error" in event for event in guard.get("cleanup", [])):
        return None
    if re.search(r"own allocation [0-9]+ MiB exceeds 61(?:\.0)? GiB guard cap", reason):
        return "owned_process_memory_exceeded_61_gib_guard"
    explicit_errors = [row for row in rank_reports if row and
                       (row.get("error_type") == "OutOfMemoryError"
                        or re.search(r"\b(?:NPU|CUDA) out of memory\b", row.get("error", ""), re.I))]
    if any(row and row.get("error_type") in {"FloatingPointError", "AssertionError", "ValueError"}
           for row in rank_reports):
        return None
    if explicit_errors:
        return "explicit_device_allocator_out_of_memory"
    # A torchrun peer can terminate before its JSON failure record is written.
    # Require an exception-shaped device OOM line, not a generic memory keyword.
    if re.search(r"(?:torch\.)?OutOfMemoryError:|RuntimeError:\s*(?:NPU|CUDA) out of memory", worker_log, re.I):
        return "explicit_device_allocator_out_of_memory_in_worker_log"
    return None


def needs_confirmation(first, second):
    first_median, second_median = statistics.median(first), statistics.median(second)
    relative_gap = (max(first_median, second_median) - min(first_median, second_median)) / min(first_median, second_median)
    overlap = max(min(first), min(second)) <= min(max(first), max(second))
    return relative_gap <= 0.03 or overlap, {"relative_median_gap": relative_gap, "ranges_overlap": overlap}


def choose_candidate(candidates):
    """Use all measured times; medians within 3% of fastest tie on peak memory."""
    ranked = sorted(candidates, key=lambda c: (statistics.median(c["measured_seconds"]), c["microbatch"]))
    fastest = statistics.median(ranked[0]["measured_seconds"])
    tied = [row for row in ranked if statistics.median(row["measured_seconds"]) <= fastest * 1.03]
    selected = min(tied, key=lambda c: (c["peak_allocated_bytes"], c["microbatch"]))
    return selected, [{"microbatch": row["microbatch"],
                       "median_seconds_per_update": statistics.median(row["measured_seconds"]),
                       "measurements": list(row["measured_seconds"]),
                       "peak_allocated_bytes": row["peak_allocated_bytes"],
                       "within_3_percent_of_fastest": row in tied} for row in ranked]


class SelectionDriver:
    def __init__(self, root, tag, initial_ddp_result=None):
        self.root, self.tag = Path(root), tag
        self.report_path = self.root / REPORT_NAME
        self.report = {"status": "initializing", "started_utc": now(), "driver_pid": os.getpid(),
                       "attempt_tag": tag, "formal_training_started": False,
                       "training_config_frozen": False, "inference_policy_frozen": False,
                       "precision": "fp32", "effective_batch": 32, "world_size": 4,
                       "automatic_retries": 0, "maximum_new_smoke_reservation": 35,
                       "candidates": {str(mb): {"microbatch": mb, "status": "not_tested"} for mb in MICROBATCHES},
                       "attempts": []}
        self.configs = {}
        self.memory_stream = None
        self.performance_stream = None
        self.performance_weights = None
        self.common_identity = None
        self.initial_ddp_path = Path(initial_ddp_result) if initial_ddp_result else self.root / INITIAL_DDP_NAME

    def persist(self):
        self.report["updated_utc"] = now()
        ledger_path = self.root / LEDGER_NAME
        if ledger_path.exists():
            ledger = read(ledger_path)
            self.report["budget"] = {key: ledger[key] for key in
                                     ["budget_effective_batch_updates", "actual_completed_updates",
                                      "conservative_consumed_updates", "reserved_updates"]}
            self.report["ledger"] = evidence(ledger_path)
        write(self.report_path, self.report)

    def verify_prerequisites(self):
        path = self.initial_ddp_path
        initial = successful_gradient_evidence(path, 1)
        self.initial_ddp = initial
        self.verify_current_code()
        self.report["initial_ddp_normalization"] = evidence(path)
        self.report["protocol"] = evidence(self.root / "docs/STAGE2_RUNTIME_SELECTION_PROTOCOL.md")
        self.report["driver_source"] = evidence(Path(__file__).resolve())
        self.report["controller_source"] = evidence(self.root / "code/smoke_controller.py")
        ledger = read(self.root / LEDGER_NAME)
        require(ledger["budget_effective_batch_updates"] == 60, "Smoke ledger budget must remain 60")
        require(ledger["reserved_updates"] == 0, "A prior unresolved smoke reservation requires investigation")
        require(0 <= ledger["conservative_consumed_updates"] <= 60, "Invalid consumed smoke budget")
        first = None
        self.report["candidate_configs"] = {}
        for mb in MICROBATCHES:
            config_path = self.root / f"configs/stage2_candidate_mb{mb}.json"
            config = read(config_path)
            require(config["training"]["microbatch"] == mb and config["training"]["accumulation"] == 8 // mb,
                    "Candidate factorization mismatch")
            require(config["training"]["precision"] == "fp32" and config["training"]["effective_batch"] == 32
                    and config["training"]["world_size"] == 4, "Only the frozen FP32/32-path/four-rank candidates are allowed")
            normalized = json.loads(json.dumps(config))
            normalized["training"].pop("microbatch")
            normalized["training"].pop("accumulation")
            if first is None:
                first = normalized
            require(normalized == first, "Candidate configs differ beyond permitted microbatch/accumulation")
            self.configs[mb] = config
            self.report["candidate_configs"][str(mb)] = evidence(config_path)
        require(initial["rank_results"][0]["config_sha256"] == self.report["candidate_configs"]["1"]["sha256"],
                "Initial DDP check used a different candidate config")

    def verify_current_code(self):
        code = self.initial_ddp["rank_results"][0]["identity"]["actual_code_sha256"]
        for relative, expected in code.items():
            require(sha256(self.root / relative) == expected,
                    f"Source changed after the accepted initial DDP check: {relative}")

    def launch_smoke(self, mode, mb, label, measure):
        self.verify_current_code()
        attempt_id = f"stage2_{label}_mb{mb}_{self.tag}"
        ledger_before = read(self.root / LEDGER_NAME)
        require(not any(row["id"] == attempt_id for row in ledger_before["attempts"]),
                f"Attempt already exists: {attempt_id}")
        config_path = self.root / f"configs/stage2_candidate_mb{mb}.json"
        require(sha256(config_path) == self.report["candidate_configs"][str(mb)]["sha256"],
                "Candidate config changed during selection")
        command = [sys.executable, str(self.root / "code/smoke_controller.py"), "--id", attempt_id,
                   "--mode", mode, "--microbatch", str(mb), "--warmup-updates", "1",
                   "--measure-updates", str(measure)]
        summary = {"id": attempt_id, "mode": mode, "microbatch": mb, "command": command,
                   "started_utc": now(), "status": "running"}
        self.report["attempts"].append(summary)
        self.report["phase"] = label
        self.persist()
        exit_code = subprocess.call(command, cwd=self.root, stdin=subprocess.DEVNULL)
        ledger = read(self.root / LEDGER_NAME)
        matching = [row for row in ledger["attempts"] if row["id"] == attempt_id]
        require(len(matching) == 1, "Smoke controller did not produce one durable ledger row")
        row = matching[0]
        require(row["status"] in {"passed", "failed"} and ledger["reserved_updates"] == 0,
                "Smoke accounting is unresolved; no further launch is permitted")
        require(ledger["conservative_consumed_updates"] <= 60, "Smoke budget exceeded")
        output = self.root / "artifacts" / attempt_id
        run = self.root / "runs" / attempt_id
        guard_path = run / "guard_result.json"
        guard = read(guard_path) if guard_path.exists() else None
        rank_reports = []
        for rank in range(4):
            paths = [output / f"results_rank{rank}.json", output / f"error_rank{rank}.json"]
            rank_reports.append(next((read(path) for path in paths if path.exists()), None))
        summary.update({"status": row["status"], "exit_code": exit_code,
                        "completed_utc": now(), "ledger_row": row,
                        "guard_evidence": evidence(guard_path) if guard_path.exists() else None})
        self.persist()
        if exit_code != 0:
            require(row["status"] == "failed", "Controller exit and ledger status disagree")
            worker_log_path = run / "worker.log"
            worker_log = worker_log_path.read_text(errors="replace") if worker_log_path.exists() else ""
            reason = memory_failure_reason(guard, rank_reports, worker_log) if mode == "memory-smoke" else None
            if reason is not None:
                summary["candidate_exclusion_reason"] = reason
                self.persist()
                return {"passed": False, "reason": reason, "attempt": summary}
            raise RuntimeError(f"Non-excludable {mode} failure: {attempt_id}; inspect its guard/worker evidence")
        require(row["status"] == "passed", "Controller exit and ledger status disagree")
        require(guard and guard.get("status") == "completed" and guard.get("own_npu_processes_released") is True,
                "Smoke did not finish with verified release of owned NPU processes")
        result, logs = self.validate_success(output, mode, mb, measure)
        summary["result_evidence"] = evidence(output / "results.json")
        summary["rank_log_evidence"] = [evidence(output / f"updates_rank{rank}.jsonl") for rank in range(4)]
        summary["optimizer_step_attempts"] = result["optimizer_step_attempts_this_invocation"]
        self.persist()
        return {"passed": True, "result": result, "logs": logs, "attempt": summary}

    def validate_success(self, output, mode, mb, measure):
        result = read(output / "results.json")
        expected_steps = 1 + measure * (2 if mode == "resume-check" else 1)
        all_logs = []
        for rank in range(4):
            row = read(output / f"results_rank{rank}.json")
            require(row.get("status") == "passed" and row.get("mode") == mode and row.get("rank") == rank,
                    "Smoke successful rank result is missing or inconsistent")
            require(row.get("precision") == "fp32" and row.get("microbatch") == mb
                    and row.get("accumulation") == 8 // mb and row.get("world_size") == 4,
                    "Smoke actual precision/factorization mismatch")
            require(row.get("optimizer_step_attempts_this_invocation") == expected_steps
                    and row.get("physical_optimizer_steps_this_invocation") == expected_steps
                    and row.get("smoke_optimizer_updates") == expected_steps,
                    "Smoke physical/attempted update counts differ from reserved protocol")
            require(row.get("formal_training_started") is False and row.get("simulator_unchanged") is True
                    and row.get("optimizer_states_allocated") is True,
                    "Smoke must be disposable and include real optimizer state allocation")
            require(row["initial_network_sha256"] == self.configs[mb]["expected_initial_network_sha256"]
                    and row["initial_simulator_sha256"] == self.configs[mb]["expected_simulator_sha256"],
                    "Smoke initial state differs from accepted original initialization")
            identity = dict(row["identity"])
            identity.pop("numerical_config_sha256")
            if self.common_identity is None:
                self.common_identity = identity
            require(identity == self.common_identity, "Code/data/assets identity changed between smokes")
            initial_identity = dict(self.initial_ddp["rank_results"][rank]["identity"])
            initial_identity.pop("numerical_config_sha256")
            require(identity == initial_identity, "Smoke identity differs from initial DDP acceptance")
            logs = [json.loads(line) for line in (output / f"updates_rank{rank}.jsonl").read_text().splitlines() if line.strip()]
            require(len(logs) == expected_steps, "Smoke must account for every physical update")
            for number, entry in enumerate(logs, 1):
                require(entry["global_update"] == 0 and entry["formal"] is False
                        and entry["finite_loss"] is True and entry["finite_gradients"] is True,
                        "Smoke contains formal or nonfinite update state")
                require(entry["physical_optimizer_steps_this_invocation"] == number
                        and entry["optimizer_step_attempts_this_invocation"] == number,
                        "Smoke log physical update sequence is inconsistent")
                require(len(entry["global_start_ids"]) == len(set(entry["global_start_ids"])) == 32,
                        "Global batch does not contain 32 distinct starts")
                weights = entry["path_weights"]
                require(len(weights) == 32 and [item["start_id"] for item in weights] == entry["global_start_ids"],
                        "Path-weight identities differ from sampled starts")
                for item in weights:
                    require(item["G"] in [1, 2, 3, 4] and item["path_weight"] == 1 / 32
                            and item["step_weight"] == 1 / (32 * item["G"]) and item["lambda_R"] == .2,
                            "Path/prediction weighting changed")
                require(math.isfinite(entry["seconds"]) and entry["seconds"] > 0,
                        "Nonfinite or nonpositive update timing")
            all_logs.append(logs)
        logs = all_logs[0]
        ids = [row["global_start_ids"] for row in logs]
        weights = [row["path_weights"] for row in logs]
        require(all([r["global_start_ids"] for r in rows] == ids
                    and [r["path_weights"] for r in rows] == weights for rows in all_logs),
                "Rank global sample streams or weights differ")
        if mode == "memory-smoke":
            require(all(item["G"] == 4 for row in weights for item in row), "Memory pressure must use only full G4 paths")
            if self.memory_stream is None:
                self.memory_stream = ids
            require(ids == self.memory_stream, "Memory candidates did not use the same seeded G4 stream")
        elif mode == "performance-smoke":
            require(len(result["measured_seconds_per_update"]) == 3, "Exactly three steady-state timings required")
            require([entry["phase"] for entry in logs] == ["warmup", "measure", "measure", "measure"],
                    "Performance warmup/measurement split differs from protocol")
            require(result["measured_seconds_per_update"] == [entry["seconds"] for entry in logs[1:]],
                    "Rank-zero summary timings differ from actual update logs")
            require({item["G"] for row in weights for item in row} == {1, 2, 3, 4},
                    "Representative performance stream does not cover all G values")
            require(ids[0] == self.initial_ddp["rank_results"][0]["global_start_ids"],
                    "Performance first batch differs from independently verified DDP batch")
            if self.performance_stream is None:
                self.performance_stream, self.performance_weights = ids, weights
            require(ids == self.performance_stream and weights == self.performance_weights,
                    "Performance candidates/retests did not share identical IDs and weights")
        else:
            require(self.performance_stream is not None and ids == [self.performance_stream[0], self.performance_stream[1], self.performance_stream[1]],
                    "Resume check did not replay exactly the expected original suffix IDs")
            for rank in range(4):
                boundary = read(output / f"resume_boundary_parity_rank{rank}.json")
                require(boundary.get("exact") is True
                        and boundary.get("checked_before_any_restored_optimizer_attempt") is True
                        and boundary.get("completed_updates") == 1
                        and boundary.get("tolerance") == {"rtol": 0.0, "atol": 0.0},
                        "Selected immediate-boundary exact restore did not pass before suffix on all ranks")
            require(all(read(output / f"resume_parity_rank{rank}.json")["passed"] is True
                        and read(output / f"resume_parity_rank{rank}.json")["identical_suffix_start_ids"] is True
                        for rank in range(4)), "Selected exact-control-state and FP32 resume parity did not pass on all ranks")
        return result, logs

    def recheck_selected_gradients(self, mb):
        self.verify_current_code()
        if mb == 1:
            return {"reused_initial_mb1": True, **self.report["initial_ddp_normalization"]}
        attempt_id = f"stage2_ddp_gradients_mb{mb}_selected_{self.tag}"
        output, run = self.root / "artifacts" / attempt_id, self.root / "runs" / attempt_id
        require(not output.exists() and not run.exists(), "Selected DDP recheck paths must be fresh")
        command = [sys.executable, str(self.root / "code/guard_distributed.py"), "--run-dir", str(run),
                   "--timeout-seconds", "7200", "--memory-limit-gib", "61", "--command",
                   "bash", str(self.root / "code/run_distributed_env.sh"), "-m", "torch.distributed.run",
                   "--standalone", "--nnodes=1", "--nproc_per_node=4", "code/check_distributed_gradients.py",
                   "--config", str(self.root / f"configs/stage2_candidate_mb{mb}.json"),
                   "--output", str(output), "--microbatch", str(mb)]
        self.report["phase"] = "selected_microbatch_gradient_recheck"
        self.report["selected_gradient_command"] = command
        self.persist()
        code = subprocess.call(command, cwd=self.root, stdin=subprocess.DEVNULL)
        require(code == 0, "Selected-microbatch real DDP recheck failed")
        guard = read(run / "guard_result.json")
        require(guard.get("status") == "completed" and guard.get("own_npu_processes_released") is True,
                "Selected DDP recheck did not release owned NPU allocations")
        checked = successful_gradient_evidence(output / "results.json", mb)
        for row in checked["rank_results"]:
            identity = dict(row["identity"])
            identity.pop("numerical_config_sha256")
            require(identity == self.common_identity, "Selected DDP recheck identity changed")
        return {"reused_initial_mb1": False, **evidence(output / "results.json"),
                "guard": evidence(run / "guard_result.json")}

    def run(self):
        self.verify_prerequisites()
        self.report["status"] = "running"
        self.persist()
        feasible = []
        for mb in MICROBATCHES:
            attempt = self.launch_smoke("memory-smoke", mb, "memory", 1)
            candidate = self.report["candidates"][str(mb)]
            candidate["memory_attempt"] = attempt["attempt"]["id"]
            if not attempt["passed"]:
                candidate.update(status="excluded", exclusion_reason=attempt["reason"],
                                 performance_not_tested_reason="Longest-path memory qualification failed")
            else:
                candidate.update(status="memory_passed", memory_peak_allocated_bytes=attempt["result"]["max_peak_allocated_bytes"],
                                 memory_peak_reserved_bytes=attempt["result"]["max_peak_reserved_bytes"])
                feasible.append(mb)
            self.persist()
        require(feasible, "No memory-qualified candidate remains")
        performance = []
        for mb in feasible:
            attempt = self.launch_smoke("performance-smoke", mb, "performance", 3)
            result = attempt["result"]
            candidate = self.report["candidates"][str(mb)]
            candidate.update(status="performance_passed", measured_seconds=list(result["measured_seconds_per_update"]),
                             performance_attempts=[attempt["attempt"]["id"]],
                             peak_allocated_bytes=max(candidate["memory_peak_allocated_bytes"], result["max_peak_allocated_bytes"]),
                             cold_first_update_seconds=[result["cold_first_update_seconds"]],
                             setup_seconds=[result["setup_seconds"]],
                             timing_scope=result["timing_scope"])
            performance.append(candidate)
            self.persist()
        ranked = sorted(performance, key=lambda row: (statistics.median(row["measured_seconds"]), row["microbatch"]))
        confirm = False
        trigger = {"reason": "Only one memory-qualified candidate"}
        if len(ranked) >= 2:
            confirm, trigger = needs_confirmation(ranked[0]["measured_seconds"], ranked[1]["measured_seconds"])
        self.report["confirmation_rule"] = {"triggered": confirm, "evidence": trigger,
                                            "maximum_confirmation_runs": 2, "additional_retests_allowed": False}
        if confirm:
            for candidate in ranked[:2]:
                mb = candidate["microbatch"]
                attempt = self.launch_smoke("performance-smoke", mb, "performance_confirm", 3)
                result = attempt["result"]
                candidate["measured_seconds"].extend(result["measured_seconds_per_update"])
                candidate["performance_attempts"].append(attempt["attempt"]["id"])
                candidate["peak_allocated_bytes"] = max(candidate["peak_allocated_bytes"], result["max_peak_allocated_bytes"])
                candidate["cold_first_update_seconds"].append(result["cold_first_update_seconds"])
                candidate["setup_seconds"].append(result["setup_seconds"])
                self.persist()
        selected, ranking = choose_candidate(performance)
        mb = selected["microbatch"]
        self.report["ranking"] = ranking
        self.report["provisional_selected_microbatch"] = mb
        resume = self.launch_smoke("resume-check", mb, "resume", 1)
        self.report["selected_resume_check"] = resume["attempt"]["result_evidence"]
        self.report["selected_ddp_normalization"] = self.recheck_selected_gradients(mb)
        median = statistics.median(selected["measured_seconds"])
        self.report.update({"status": "passed_waiting_inference_smoke_and_freeze", "completed_utc": now(),
                            "selected_microbatch": mb, "selected_accumulation": 8 // mb,
                            "selected_training": self.configs[mb]["training"],
                            "schedule": self.configs[mb]["scheduler"],
                            "selected_median_seconds_per_update": median,
                            "selected_peak_allocated_bytes": selected["peak_allocated_bytes"],
                            "peak_memory_basis": "maximum allocated bytes across qualified longest-path memory and mixed-G performance attempts",
                            "estimated_training_1000_updates_seconds": median * 1000,
                            "estimate_excludes": ["calibration", "baseline and checkpoint evaluations",
                                                  "JSON/checkpoint persistence", "cold setup/compilation"],
                            "selection_uses_image_quality": False,
                            "inference_mechanism_smoke": "required_separate_parent_step",
                            "formal_training_started": False})
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-tag", default="01", help="Fresh explicit suffix; no retries are implicit")
    parser.add_argument("--initial-ddp-result", type=Path,
                        help="Explicit replacement for the default initial mb1 evidence after a reviewed code change")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.attempt_tag):
        parser.error("Attempt tag must be alphanumeric/underscore")
    lock = (ROOT / ".stage2_selection_driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    driver = SelectionDriver(ROOT, args.attempt_tag, args.initial_ddp_result)
    require(not driver.report_path.exists(), "Selection report already exists; inspect/archive it explicitly before a new attempt")
    try:
        driver.run()
    except BaseException as exc:
        driver.report.update({"status": "blocked", "error": repr(exc), "traceback": traceback.format_exc(),
                              "completed_utc": now(), "formal_training_started": False})
        for candidate in driver.report["candidates"].values():
            if candidate["status"] == "not_tested":
                candidate["not_tested_reason"] = "Selection halted on a blocking error before this candidate"
        driver.persist()
        raise
    finally:
        lock.close()
    print(json.dumps({"status": driver.report["status"], "selected_microbatch": driver.report["selected_microbatch"],
                      "report": str(driver.report_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

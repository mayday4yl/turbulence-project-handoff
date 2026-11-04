"""Bounded, measured B32 confirmation of the fastest single-card candidates."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.controller import parse_npu_processes, atomic_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint-policy", choices=("highest_third", "highest_half", "highest"),
                    default="highest_third")
parser.add_argument("--tag", choices=("optimized", "optimized_half", "optimized_highest"),
                    default="optimized")
args = parser.parse_args()
REPORT = ROOT / f"reports/four_card_{args.tag}_workflow.json"


def free_cards():
    snapshot = subprocess.run(["npu-smi", "info"], capture_output=True,
                              text=True, check=True, timeout=30).stdout
    occupied = parse_npu_processes(snapshot, [0, 1, 2, 3])
    if any(occupied.values()):
        raise RuntimeError(f"Selected cards are occupied: {occupied}")
    return snapshot


def main():
    if REPORT.exists():
        raise RuntimeError("Refusing to overwrite a previous workflow record")
    report = {"scope": "bounded optimized four-card B32 benchmark; zero formal updates",
              "status": "STARTING", "pid": os.getpid(), "started_unix": time.time(),
              "physical_cards": [0, 1, 2, 3], "formal_training_launched": False,
              "resource_snapshot_before_launch": free_cards(), "commands": []}
    atomic_json(REPORT, report)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
    # Confirm the slow recursive arm first. D0's separately measured winner
    # retains every activation; all scientific settings remain identical.
    for method, policy in [("R0", args.checkpoint_policy), ("D0", "none")]:
        free_cards()
        output = f"reports/benchmark_{method}_w4_b2_bf16_{policy}_{args.tag}.json"
        command = ["bash", "run_env.sh", "-m", "torch.distributed.run",
                   "--nproc_per_node", "4", "--master_port", "29623",
                   "scripts/benchmark_distributed.py", "--initial", "weights/adapted_U0.pt",
                   "--dataset", "data/v5_places_c50_s2000_v250_t250_g4_r4_20261021",
                   "--output", output, "--method", method, "--precision", "bf16",
                   "--microbatch", "2", "--talking-backend", "conv2d_equivalent",
                   "--warmup", "1", "--updates", "2"]
        if policy != "none":
            command.extend(["--checkpointing", "--checkpoint-policy", policy])
        with (ROOT / output).with_suffix(".log").open("xb", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record = {"method": method, "checkpoint_policy": policy, "command": command,
                      "pid": child.pid, "started_unix": time.time(), "output": output}
            report.update(status="RUNNING_BOUNDED_GATE", active_method=method,
                          active_child_pid=child.pid)
            report["commands"].append(record)
            atomic_json(REPORT, report)
            try:
                code = child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=15)
                raise RuntimeError(f"{method} bounded benchmark exceeded 20 minutes")
        record.update(return_code=code, finished_unix=time.time())
        atomic_json(REPORT, report)
        if code != 0:
            raise RuntimeError(f"{method} benchmark failed; preserve and inspect {output}")
    report.update(status="BOUNDED_GATES_FINISHED", finished_unix=time.time(),
                  resource_snapshot_after_completion=free_cards())
    atomic_json(REPORT, report)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if REPORT.exists():
            report = json.loads(REPORT.read_text())
            if report.get("pid") == os.getpid():
                report.update(status="FAILED_OR_INCOMPLETE", error=repr(exc),
                              finished_unix=time.time(), formal_training_launched=False)
                atomic_json(REPORT, report)
        raise

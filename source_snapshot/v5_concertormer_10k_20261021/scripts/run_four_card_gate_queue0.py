"""One bounded queue0 retry after the preserved queue1 Conv2DBackpropInput failure."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.controller import parse_npu_processes, atomic_json


def main():
    report = {"scope": "one-off four-card B32 preflight; zero formal updates",
              "status": "WAITING_RESOURCES", "pid": os.getpid(), "started_unix": time.time(),
              "physical_cards": [0, 1, 2, 3], "task_queue_enable": "0",
              "formal_training_launched": False, "commands": []}
    path = ROOT / "reports/four_card_gate_queue0_workflow.json"
    atomic_json(path, report)
    deadline = time.monotonic() + 900
    while True:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=30)
        processes = parse_npu_processes(result.stdout, report["physical_cards"])
        if not any(processes.values()):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Selected cards not free within the bounded wait: {processes}")
        time.sleep(10)
    report["resource_snapshot_before_launch"] = result.stdout
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
    for method in ["D0", "R0"]:
        phase_deadline = time.monotonic() + 300
        while True:
            check = subprocess.run(["npu-smi", "info"], capture_output=True,
                                   text=True, check=True, timeout=30)
            processes = parse_npu_processes(check.stdout, report["physical_cards"])
            if not any(processes.values()):
                break
            if time.monotonic() >= phase_deadline:
                raise RuntimeError(f"Selected cards became occupied before {method}: {processes}")
            time.sleep(10)
        output = f"reports/benchmark_{method}_w4_b2_bf16_ac_queue0.json"
        command = ["bash", "run_env.sh", "-m", "torch.distributed.run",
                   "--nproc_per_node", "4", "--master_port", "29622",
                   "scripts/benchmark_distributed.py", "--initial", "weights/adapted_U0.pt",
                   "--dataset", "data/v5_places_c50_s2000_v250_t250_g4_r4_20261021",
                   "--output", output, "--method", method, "--precision", "bf16",
                   "--microbatch", "2", "--checkpointing", "--talking-backend", "conv2d_equivalent",
                   "--warmup", "0" if method == "D0" else "1", "--updates", "1"]
        with (ROOT / output).with_suffix(".log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record = {"method": method, "command": command, "pid": child.pid,
                      "started_unix": time.time(), "output": output}
            report.update(status="RUNNING_BOUNDED_GATE", active_method=method,
                          active_child_pid=child.pid)
            report["commands"].append(record)
            atomic_json(path, report)
            try:
                code = child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
                raise RuntimeError(f"{method} bounded HCCL gate exceeded 20 minutes")
        record.update(return_code=code, finished_unix=time.time())
        atomic_json(path, report)
        if code != 0:
            raise RuntimeError(f"{method} HCCL gate failed; inspect {output}")
    report.update(status="BOUNDED_GATES_FINISHED", finished_unix=time.time(),
                  formal_training_launched=False)
    atomic_json(path, report)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        path = ROOT / "reports/four_card_gate_queue0_workflow.json"
        report = json.loads(path.read_text()) if path.exists() else {}
        report.update(status="FAILED_OR_INCOMPLETE", error=repr(exc),
                      finished_unix=time.time(), formal_training_launched=False)
        atomic_json(path, report)
        raise

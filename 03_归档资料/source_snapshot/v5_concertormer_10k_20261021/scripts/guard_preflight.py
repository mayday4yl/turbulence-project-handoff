"""Bound the three explicitly identified one-off preflight process groups."""
from pathlib import Path
import json
import os
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["probe_D0_equiv_fp32_b4_noac", "probe_R0_equiv_fp32_b4_ac",
         "probe_R0_equiv_bf16_b8_ac"]
LIMIT = 1800


def main():
    pending = set(NAMES)
    while pending:
        for name in list(pending):
            launch = json.loads((ROOT / "reports" / f"{name}.launch.json").read_text())
            report_path = ROOT / "reports" / f"{name}.json"
            report = json.loads(report_path.read_text())
            pid = launch["pid"]
            proc = Path(f"/proc/{pid}/cmdline")
            if report.get("status") in {"PASS", "FAIL"} or not proc.exists():
                pending.remove(name)
                continue
            if time.time() - launch["started_unix"] < LIMIT:
                continue
            arguments = proc.read_bytes().decode().strip("\0").split("\0")
            expected_output = f"reports/{name}.json"
            # Never select by Python process name alone or target reused PIDs.
            if ("scripts/probe_backend.py" not in arguments or
                    "--output" not in arguments or
                    arguments[arguments.index("--output") + 1] != expected_output or
                    os.getpgid(pid) != pid):
                raise RuntimeError(f"Process identity mismatch; refusing to stop {pid}")
            record = {"status": "INCOMPLETE_OPERATIONAL_TIME_LIMIT", "pid": pid,
                      "limit_seconds": LIMIT, "unix_time": time.time(),
                      "reason": "One-off candidate exceeded 30-minute preflight wall budget; not a numerical FAIL or PASS.",
                      "last_original_report": report, "arguments": arguments}
            # Preserve the original probe JSON, including incomplete measurements.
            destination = ROOT / "reports" / f"{name}.termination.json"
            destination.write_text(json.dumps(record, indent=2) + "\n")
            os.killpg(pid, signal.SIGTERM)
            # A process group is exclusively the verified detached probe and its
            # descendants. This does not target any other experiment's workers.
            time.sleep(10)
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                pass
            else:
                os.killpg(pid, signal.SIGKILL)
            record["stop_signal_sent"] = True
            destination.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"probe": name, "status": record["status"], "pid": pid}), flush=True)
            pending.remove(name)
        if pending:
            time.sleep(10)


if __name__ == "__main__":
    main()

"""Run bounded CPU semantic tests and save honest CPU-only evidence."""
import argparse
import io
import platform
import sys
import time
import unittest

import torch

from training.state import atomic_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromName("alignment_checks.test_checks")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema": "v5_alignment_cpu_checks_v1", "status": "CPU_ONLY_PASS" if result.wasSuccessful() else "FAIL",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "elapsed_seconds": time.time() - started, "torch": str(torch.__version__),
              "python": sys.version, "hostname": platform.node(), "output": stream.getvalue(),
              "formal_optimizer_updates": 0, "npu_process_started": False,
              "scope": "small CPU mock/structural semantics; actual D0/A selected weights and six-card NPU preflight still required"}
    atomic_json(report, args.report)
    print(stream.getvalue(), end="")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

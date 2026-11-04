"""Read-only Ascend environment inventory; never initializes a device context.

Version capture alone is not a compatibility PASS. The stage's actual six-card
forward/backward/checkpoint check is the compatibility evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args, timeout=20):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout[-40000:], "stderr": result.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "unavailable": str(exc)}


def audit():
    versions = {}
    for package in ("torch", "torch-npu", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = {}
    for pattern in (
        "/usr/local/Ascend/driver/version.info", "/usr/local/Ascend/firmware/version.info",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info",
        "/usr/local/Ascend/ascend-toolkit/latest/*/hccl/version.info",
        "/usr/local/Ascend/cann/*/version.cfg",
        "/etc/ascend_install.info",
    ):
        for name in glob.glob(pattern):
            path = Path(name)
            try:
                files[str(path)] = {"resolved": str(path.resolve()), "text": path.read_text()[:16000]}
            except OSError as exc:
                files[str(path)] = {"unavailable": str(exc)}
    libraries = []
    for pattern in ("/usr/local/Ascend/ascend-toolkit/latest/*/lib64/libhccl.so*",
                    "/usr/local/Ascend/ascend-toolkit/latest/lib64/libhccl.so*",
                    "/usr/local/Ascend/cann/*/lib64/libhccl.so*"):
        libraries.extend({"path": p, "resolved": str(Path(p).resolve())} for p in glob.glob(pattern))
    safe_names = ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_HOME_PATH",
                  "ASCEND_TOOLKIT_HOME", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING",
                  "OMP_NUM_THREADS", "HCCL_CONNECT_TIMEOUT", "HCCL_EXEC_TIMEOUT")
    info = command([shutil.which("npu-smi") or "npu-smi", "info"])
    return {"schema": "v5_alignment_environment_audit_v1", "status": "CAPTURED",
            "captured_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
            "platform": platform.platform(), "python": sys.version, "python_executable": sys.executable,
            "cpu_count": os.cpu_count(), "package_versions": versions, "version_files": files,
            "hccl_libraries": libraries, "environment": {n: os.environ.get(n) for n in safe_names},
            "npu_smi": info, "cpu_memory": command(["free", "-b"]),
            "processes": command(["ps", "-eo", "user,pid,ppid,etime,pcpu,pmem,comm", "--sort=-pcpu"]),
            "npu_context_created": False, "global_environment_modified": False,
            "authorized_physical_cards": [2, 3, 4, 5, 6, 7], "excluded_physical_cards": [0, 1],
            "compatibility_conclusion": "Versions captured; actual stage full-update readiness is required",
            "hf32": "Read from configured training process in preflight; no passive value asserted",
            "test_used": False}


def runtime_flags():
    """Call only inside an already authorized, configured NPU process."""
    import torch
    import torch_npu
    # torch_npu 2.7 exposes this option through a setter-only Python wrapper.
    # Read the backing option, as its HF32 getters do; never mutate it here.
    internal = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    if isinstance(internal, bytes):
        internal = internal.decode()
    result = {"hf32_conv": bool(torch.npu.conv.allow_hf32),
              "hf32_matmul": bool(torch.npu.matmul.allow_hf32),
              "internal_format": {"enable": True, "disable": False}.get(internal),
              "internal_format_raw": internal,
              "internal_format_query": "torch_npu._C._npu_getOption"}
    if hasattr(torch.npu, "aclnn"):
        result["hf32_aclnn"] = bool(torch.npu.aclnn.allow_hf32)
    if any(v for k, v in result.items() if k.startswith("hf32_")):
        raise RuntimeError("HF32 must stay OFF for the A/B FP32 Simulator path")
    try:
        import torch_npu
        version = getattr(torch_npu._C, "_hccl_get_version", None)
        result["hccl_runtime_version"] = str(version()) if callable(version) else None
    except (AttributeError, RuntimeError, TypeError) as exc:
        result["hccl_version_query_unavailable"] = str(exc)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit(), indent=2, ensure_ascii=False) + "\n")
    print(str(path))


if __name__ == "__main__":
    main()

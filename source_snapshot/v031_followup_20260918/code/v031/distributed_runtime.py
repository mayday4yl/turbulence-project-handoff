"""Rank-local initialization using the validated Stage 1 NPU execution options.

Importing this module does not import torch_npu or initialize an NPU. Call
configure_local_rank before model construction, RNG operations, or HCCL setup.
"""
from __future__ import annotations

import os


def configure_local_rank(config, local_rank):
    """Select this torchrun rank first, then apply and report backend options.

    Returns a JSON-serializable provenance dictionary; the caller owns process
    group initialization and teardown. This entry point is restricted to one
    four-rank node on the authorized physical devices 0,1,2,3.
    """
    if isinstance(local_rank, bool) or not isinstance(local_rank, int) or local_rank not in range(4):
        raise ValueError("local_rank must be an integer in 0..3")
    if os.environ.get("LOCAL_RANK") != str(local_rank):
        raise RuntimeError("LOCAL_RANK must match the explicit local_rank argument")
    if os.environ.get("WORLD_SIZE") != "4":
        raise RuntimeError("Distributed V0.31 requires WORLD_SIZE=4")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        raise RuntimeError("Only ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 is authorized")
    rank = os.environ.get("RANK")
    if rank is not None and rank != str(local_rank):
        raise RuntimeError("Single-node V0.31 requires RANK=LOCAL_RANK")

    runtime = config["npu_execution"]
    for name, key in (("TASK_QUEUE_ENABLE", "task_queue_enable"),
                      ("ASCEND_LAUNCH_BLOCKING", "launch_blocking")):
        if os.environ.get(name) != str(runtime[key]):
            raise RuntimeError(
                f"{name} must be set to {runtime[key]} before importing torch_npu; "
                "use run_distributed_env.sh")

    import torch
    import torch_npu

    # Do not call the Stage 1 configure_npu: it unconditionally selects npu:0.
    # The first device operation selects our own local rank before lazy init.
    device = f"npu:{local_rank}"
    torch.npu.set_device(device)
    torch.npu.init()
    if int(torch.npu.current_device()) != local_rank:
        raise RuntimeError("NPU current device does not match LOCAL_RANK")
    torch.npu.set_compile_mode(jit_compile=runtime["jit_compile"])
    torch.npu.set_option({
        "ACL_PRECISION_MODE": runtime["precision_mode"],
        "NPU_FUZZY_COMPILE_BLACKLIST": ",".join(runtime["force_jit_ops"]),
    })
    torch.npu.config.allow_internal_format = runtime["allow_internal_format"]
    torch.npu.conv.allow_hf32 = runtime["conv_hf32"]
    torch.npu.matmul.allow_hf32 = runtime["matmul_hf32"]
    if hasattr(torch.npu, "aclnn"):
        torch.npu.aclnn.allow_hf32 = runtime["aclnn_hf32"]

    # Read the same raw options used by Stage 1 acceptance, including the
    # per-operator JIT exception needed by the 33x33 convolution backward.
    raw = {
        name: torch_npu._C._npu_getOption(name)
        for name in ("jitCompile", "ALLOW_INTERNAL_FORMAT", "ALLOW_CONV_HF32",
                     "ALLOW_MATMUL_HF32", "ACL_PRECISION_MODE",
                     "NPU_FUZZY_COMPILE_BLACKLIST")
    }
    # This torch_npu stack exposes allow_internal_format as write-only;
    # diagnose_conv's first strict-JIT attempt verified that its getter raises.
    internal_format = raw["ALLOW_INTERNAL_FORMAT"]
    internal_format = (internal_format.decode() if isinstance(internal_format, bytes)
                       else str(internal_format))
    if internal_format not in ("enable", "disable"):
        raise RuntimeError("NPU ALLOW_INTERNAL_FORMAT returned an unknown value")
    effective = {
        "is_jit_compile_false": bool(torch.npu.is_jit_compile_false()),
        "allow_internal_format": internal_format == "enable",
        "conv_allow_hf32": bool(torch.npu.conv.allow_hf32),
        "matmul_allow_hf32": bool(torch.npu.matmul.allow_hf32),
        "aclnn_allow_hf32": (bool(torch.npu.aclnn.allow_hf32)
                             if hasattr(torch.npu, "aclnn") else None),
        "raw_backend_flags": {name: str(value) for name, value in raw.items()},
    }
    expected = {
        "is_jit_compile_false": not runtime["jit_compile"],
        "allow_internal_format": runtime["allow_internal_format"],
        "conv_allow_hf32": runtime["conv_hf32"],
        "matmul_allow_hf32": runtime["matmul_hf32"],
    }
    if hasattr(torch.npu, "aclnn"):
        expected["aclnn_allow_hf32"] = runtime["aclnn_hf32"]
    for name, value in expected.items():
        if effective[name] != value:
            raise RuntimeError(f"NPU backend option {name} did not take effect")
    for name, expected_value in (
        ("ACL_PRECISION_MODE", runtime["precision_mode"]),
        ("NPU_FUZZY_COMPILE_BLACKLIST", ",".join(runtime["force_jit_ops"])),
    ):
        value = raw[name].decode() if isinstance(raw[name], bytes) else str(raw[name])
        if value != expected_value:
            raise RuntimeError(f"NPU backend option {name} did not take effect")

    return {
        "local_rank": local_rank,
        "rank": int(rank) if rank is not None else local_rank,
        "world_size": 4,
        "visible_physical_devices": "0,1,2,3",
        "physical_device": local_rank,
        "device": device,
        "current_device": int(torch.npu.current_device()),
        "device_properties": str(torch.npu.get_device_properties(device)),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "npu_execution_declared": dict(runtime),
        "npu_execution_effective": effective,
    }

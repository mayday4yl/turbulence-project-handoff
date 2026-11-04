"""Use the accepted V5 NPU numerical settings without changing physics."""
from pathlib import Path
import hashlib
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(os.environ.get("V5_SIMULATOR_ROOT", str(ROOT.parent / "v5_v1_simulator_20261020")))
SIM_SHA256 = "0f2f6f54a578b3c36cb69861815c7eee5476f75ccd91cac863bd001345bd9636"


def bind_simulator_imports():
    actual = hashlib.sha256((SIM_ROOT / "simulator_v5.py").read_bytes()).hexdigest()
    if actual != SIM_SHA256:
        raise RuntimeError(f"Accepted Simulator identity changed: {actual}")
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    return SIM_ROOT


def configure(device="cpu", *, conv_backward_jit=True):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    result = {"torch": str(torch.__version__), "device": str(device),
              "simulator_dtype": "float32", "simulator_sha256": SIM_SHA256}
    if str(device).startswith("npu"):
        import torch_npu
        torch.npu.set_device(device)
        torch.npu.init()
        torch.npu.set_compile_mode(jit_compile=False)
        cache = Path(os.environ.get("V5_NPU_KERNEL_CACHE", str(ROOT / "cache" / "npu_kernels")))
        cache.mkdir(parents=True, exist_ok=True)
        options = {
            "ACL_PRECISION_MODE": "must_keep_origin_dtype",
            "ACL_OP_COMPILER_CACHE_MODE": "enable",
            "ACL_OP_COMPILER_CACHE_DIR": str(cache),
        }
        if conv_backward_jit:
            options["NPU_FUZZY_COMPILE_BLACKLIST"] = "Conv2DBackpropInput"
        torch.npu.set_option(options)
        result['conv_backward_jit_exception'] = bool(conv_backward_jit)
        torch.npu.config.allow_internal_format = True
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        if hasattr(torch.npu, "aclnn"):
            torch.npu.aclnn.allow_hf32 = False
        result.update(torch_npu=str(torch_npu.__version__), jit_compile=False,
                      hf32=False, internal_format=True, explicit_CPU_fallback=False,
                      task_queue_enable=os.environ.get("TASK_QUEUE_ENABLE"),
                      ascend_launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"))
    return result


def make_simulator(device="cpu"):
    bind_simulator_imports()
    from simulator_v5 import build_simulator
    return build_simulator(device=device)


def synchronize(device):
    if str(device).startswith("npu"):
        torch.npu.synchronize(device)


def memory_stats(device):
    if str(device).startswith("npu"):
        return {"peak_allocated_bytes": torch.npu.max_memory_allocated(device),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(device),
                "allocated_bytes": torch.npu.memory_allocated(device),
                "reserved_bytes": torch.npu.memory_reserved(device)}
    return {}

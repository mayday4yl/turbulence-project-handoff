"""Bind the A/B-only entry point to the new explicit execution authorization."""
from pathlib import Path
import json

from .state import file_sha256


def require_alignment_authorization(config):
    metadata = config.experiment_metadata or {}
    artifacts = {}
    for kind in ("authorization", "plan"):
        path, expected = metadata.get(kind + "_path"), metadata.get(kind + "_sha256")
        if not path or not expected or not Path(path).is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Missing or changed A/B {kind} identity")
        artifacts[kind] = json.loads(Path(path).read_text())
    authorization, plan = artifacts["authorization"], artifacts["plan"]
    if authorization.get("status") != "AUTHORIZED":
        raise RuntimeError("A/B authorization is not active")
    if authorization.get("approved_execution_plan_sha256") != metadata["plan_sha256"]:
        raise RuntimeError("A/B authorization does not bind this execution plan")
    if authorization.get("approved_execution_sequence") != ["A", "B"] or plan.get("execution_sequence") != ["A", "B"]:
        raise RuntimeError("Only the newly authorized A then B sequence may execute")
    budgets = {"A": 6000, "B": 4000}
    if authorization.get("stage_budgets") != budgets or config.method not in budgets:
        raise RuntimeError("Only A6000 and B4000 are authorized")
    stage = plan.get("stages", {}).get(config.method, {})
    if stage.get("max_updates") != config.max_updates or stage.get("warmup_updates") != config.warmup_updates:
        raise RuntimeError("A/B configuration and stage LR horizon differ")
    cards = [2, 3, 4, 5, 6, 7]
    if (authorization.get("physical_cards") != cards or metadata.get("physical_cards") != cards or
            config.world_size != 6 or config.effective_batch != 48 or authorization.get("effective_batch") != 48):
        raise RuntimeError("Formal A/B runs require de36 devices 2-7 and global B48")
    if config.network_precision != "bf16" or config.psf_backend != "adjoint" or config.talking_backend != "conv2d_equivalent":
        raise RuntimeError("Use the accepted BF16-network/FP32-adjoint Simulator runtime")
    if (config.train_log_interval, config.val_interval, config.checkpoint_interval) != (10, 250, 250):
        raise RuntimeError("Current A/B authorization requires log10/full-val250/checkpoint250")
    if not config.extended_validation_metrics:
        raise RuntimeError("PSNR, SSIM, LPIPS and physical errors are required")
    return artifacts

"""Strict frozen-base identities and safe access to the original RNG helper."""
import importlib.util
import json
import sys
from pathlib import Path

from data_pipeline.common import rows, sha

SCHEMA = "v5-alignment-physical-state-v1"
ROOT = Path(__file__).resolve().parents[1]


def load_legacy_generator():
    """Load the frozen implementation without leaking its absolute common import."""
    common_path = ROOT / "data_pipeline/common.py"
    generator_path = ROOT / "data_pipeline/generate.py"
    spec = importlib.util.spec_from_file_location("_v5_alignment_legacy_common", common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    sentinel = object()
    previous = sys.modules.get("common", sentinel)
    try:
        sys.modules["common"] = common
        spec = importlib.util.spec_from_file_location("_v5_alignment_legacy_generator", generator_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is sentinel:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    return module


def base_identity(base_root):
    root = Path(base_root)
    freeze = json.loads((root / "SOURCE_FREEZE.json").read_text())
    generation = json.loads((root / "GENERATION_IDENTITY.json").read_text())
    ready = json.loads((root / "DATA_READY.json").read_text())
    identity = {name: sha(root / name) for name in (
        "SOURCE_FREEZE.json", "source_manifest.jsonl", "sample_plan.jsonl",
        "GENERATION_IDENTITY.json", "sample_manifest.jsonl", "DATA_READY.json")}
    if identity["source_manifest.jsonl"] != freeze["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    if identity["sample_plan.jsonl"] != freeze["sample_plan_sha256"]:
        raise ValueError("Frozen sample plan changed")
    if identity["SOURCE_FREEZE.json"] != ready["source_freeze_sha256"]:
        raise ValueError("Frozen source identity differs from DATA_READY")
    if identity["GENERATION_IDENTITY.json"] != ready["generation_identity_sha256"]:
        raise ValueError("Frozen generator identity differs from DATA_READY")
    if identity["sample_manifest.jsonl"] != ready["sample_manifest_sha256"]:
        raise ValueError("Frozen sample manifest changed")
    if ready["status"] != "DATA_READY" or freeze["expected_samples"] != {"train": 8000, "val": 1000, "test": 1000}:
        raise ValueError("Unexpected frozen dataset readiness/counts")
    if generation["G"] != 4.0 or generation["clamp"] or generation["physics_dtype"] != "float32":
        raise ValueError("Frozen physical generation definition is unsupported")
    if freeze["old_observations_reused"] or not generation["smooth_psf"] or not generation["pixel_center_grid"] or generation["tilt_fft"] != "legacy_full":
        raise ValueError("Frozen accepted V5 simulator semantics differ")
    return {"schema": SCHEMA, "files_sha256": identity,
            "simulator_sha256": generation["simulator_sha256"],
            "vendor_simulator_sha256": generation["vendor_simulator_sha256"],
            "asset_sha256": generation["asset_sha256"],
            "physics": freeze["physics"], "raw_torch_version": generation["torch"],
            "source_counts": freeze["expected_sources"], "sample_counts": freeze["expected_samples"],
            "G": 4.0, "test_model_evaluation": False}


def load_base(base_root):
    root = Path(base_root)
    identity = base_identity(root)
    sources = rows(root / "source_manifest.jsonl")
    samples = rows(root / "sample_plan.jsonl")
    records = rows(root / "sample_manifest.jsonl")
    record_by_id = {row["sample_plan_row"]["sample_id"]: row for row in records}
    if len(record_by_id) != len(samples):
        raise ValueError("Duplicate/missing base sample records")
    for sample in samples:
        if record_by_id[sample["sample_id"]]["sample_plan_row"] != sample:
            raise ValueError("Base sample and realization record mismatch")
    return identity, sources, samples, record_by_id


def check_runtime(runtime, base):
    for field in ("simulator_sha256", "vendor_simulator_sha256", "asset_sha256"):
        if runtime[field] != base[field]:
            raise ValueError(f"Simulator identity changed: {field}")
    if runtime["torch"] != base["raw_torch_version"]:
        raise ValueError("Raw RNG replay requires the original torch version")
    for field, expected in (("G", 4.0), ("physics_dtype", "float32"), ("hf32", False),
                            ("clamp", False), ("smooth_psf", True), ("pixel_center_grid", True),
                            ("tilt_fft", "legacy_full")):
        if runtime[field] != expected:
            raise ValueError(f"Simulator semantic mismatch: {field}")

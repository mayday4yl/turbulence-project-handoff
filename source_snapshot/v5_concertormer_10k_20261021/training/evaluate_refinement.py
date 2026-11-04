"""Evaluate each selected best on full Val1000 with explicit refinement settings.

This does not select another checkpoint and cannot construct/use a test split.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import importlib
import json

import torch

from .core import RefinementConfig
from .evaluation import evaluate_full_validation, distributed_info, json_safe
from .runner import RunConfig, load_initial_model
from .state import atomic_json, file_sha256


def evaluate_best(config, refinement, *, model, simulator, val_dataset, device,
                  train_dataset=None):
    refinement.validate()
    best_path = Path(config.output_dir) / "checkpoints" / "best.pt"
    state = load_initial_model(model, best_path)
    if state.get("method") != config.method or state.get("best_update") != state.get("completed_update"):
        raise ValueError("Expected this method's full-validation selected best checkpoint")
    model.to(device)
    result = evaluate_full_validation(model, simulator, val_dataset, device=device,
        method=config.method, output_dir=config.output_dir, update=state["completed_update"],
        network_precision=config.network_precision, batch_size=config.val_batch_size,
        workers=config.workers, refinement=refinement, expected_size=1000)
    if distributed_info()[0] == 0:
        atomic_json(json_safe({"status": "BEST_FULL_VALIDATION_REFINEMENT_COMPLETE", "method": config.method,
                     "best_checkpoint": str(best_path), "best_checkpoint_sha256": file_sha256(best_path),
                     "selected_without_refinement": True, "selection_update": state["completed_update"],
                     "refinement_configuration": asdict(refinement), "result": result,
                     "test_used": False}), Path(config.output_dir) / "REFINEMENT_EVALUATION.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--refinement-config", required=True)
    parser.add_argument("--factory", default="runtime.factories:training_components")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = RunConfig(**json.loads(Path(args.config).read_text()))
    refinement = RefinementConfig(**json.loads(Path(args.refinement_config).read_text()))
    module, name = args.factory.split(":", 1)
    components = getattr(importlib.import_module(module), name)(config, args.device)
    evaluate_best(config, refinement, **components)


if __name__ == "__main__":
    main()

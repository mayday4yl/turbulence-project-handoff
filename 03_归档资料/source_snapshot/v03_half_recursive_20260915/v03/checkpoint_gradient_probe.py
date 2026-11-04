"""Read-only gradient diagnostics of a NEW V0.3-half pilot checkpoint.

No optimizer, weight update, held-out test, or training supervision is added.
Each of the four authorized ranks tests the same fixed train source/path cases
as the accepted initial gradient probe, now at the trained candidate weights.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from .common import (atomic_json, build_model, build_simulator, choose_sample,
                     seed_all, sha256_file, verify_base)
from .core import RolloutModule
from .train import state_digest


def tensor_gradient_record(tensor: torch.Tensor) -> dict:
    gradient = tensor.grad
    record = {"shape": list(tensor.shape), "requires_grad": tensor.requires_grad,
              "grad_fn": type(tensor.grad_fn).__name__ if tensor.grad_fn else None,
              "gradient_exists": gradient is not None}
    if gradient is None:
        return {**record, "finite": None, "nonzero": False,
                "max_abs": None, "l2": None}
    gradient = gradient.detach().float()
    values = torch.stack((torch.isfinite(gradient).all().float(),
                          gradient.abs().max(), gradient.norm())).cpu().tolist()
    finite = bool(values[0])
    return {**record, "finite": finite,
            "nonzero": finite and values[1] > 0,
            "max_abs": values[1] if math.isfinite(values[1]) else None,
            "l2": values[2] if math.isfinite(values[2]) else None}


def parameter_gradient_summary(network: torch.nn.Module, gradients) -> dict:
    parameters = list(network.parameters())
    by_id = {id(parameter): gradient for parameter, gradient
             in zip(parameters, gradients, strict=True)}
    groups = {"all": parameters, **network.branch_parameters()}
    result = {}
    for name, values in groups.items():
        selected = [by_id[id(parameter)] for parameter in values
                    if by_id[id(parameter)] is not None]
        if not selected:
            result[name] = {"l2": 0., "finite": True,
                            "tensors_with_gradient": 0, "parameter_tensors": len(values)}
            continue
        flags = torch.stack([torch.isfinite(value).all() for value in selected]).all()
        norm = torch.stack([value.detach().float().norm() for value in selected]).norm()
        finite, l2 = torch.stack((flags.float(), norm)).cpu().tolist()
        result[name] = {"l2": l2 if math.isfinite(l2) else None,
                        "finite": bool(finite) and math.isfinite(l2),
                        "tensors_with_gradient": len(selected),
                        "parameter_tensors": len(values)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"rank-{rank}-results.json"
    if result_path.exists():
        raise FileExistsError("Probe output already exists; retain it and use a new directory")
    result = {"status": "starting", "rank": rank,
              "config_file_sha256": sha256_file(args.config),
              "optimizer_created": False, "weights_updated": False,
              "checkpoint_loaded": True, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256_file(args.checkpoint), "distributed_gradient_sync": False,
              "phases": {}, "purpose": "training gradient diagnostics only"}
    device = None

    def progress(stage: str, **details) -> None:
        event = {"stage": stage, "rank": rank, "time": time.time(), **details}
        result["last_event"] = event
        atomic_json(args.output_dir / f"rank-{rank}-progress.json", event)
        atomic_json(result_path, result)
        print(json.dumps(event), flush=True)

    try:
        progress("imports_and_source_check")
        import torch_npu  # noqa: F401
        source_before = verify_base(config)
        from physical_bridge.data import BridgeArchive, evenly_spaced_schedule
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.set_num_threads(int(config["training"].get("cpu_threads", 4)))
        seed_all(int(config["seed"]))
        network = build_model(config, device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (checkpoint.get("mode") != "pilot" or checkpoint.get("config") != config
                or not args.checkpoint.resolve().is_relative_to(Path(config["dataset_root"]).parents[1] / "runs")):
            raise ValueError("Only this half-data pilot checkpoint may be diagnosed")
        network.load_state_dict(checkpoint["model"], strict=True)
        result["checkpoint_update"] = checkpoint["update"]
        del checkpoint
        initial_hash = state_digest(network.state_dict())
        simulator = build_simulator(config, device)
        module = RolloutModule(network, simulator,
                               float(config["training"]["lambda_R"]),
                               network_bfloat16=True)
        module.train()
        specifications = ((4.5, 5), (4.5, 3), (1.5, 3), (.75, 0))
        original_g, requested_k = specifications[rank % len(specifications)]
        schedule = tuple(evenly_spaced_schedule(original_g, requested_k))
        selected = choose_sample(config, 0, 0)
        archive = BridgeArchive(config["dataset_root"], "train")
        record = archive.records[selected["record_index"]]
        clean = archive.clean(record).unsqueeze(0).to(device)
        observation = archive.state(record, original_g).unsqueeze(0).to(device)
        archive.store.close()
        del archive
        strength = torch.tensor([original_g], device=device, dtype=torch.float32)
        result.update({"seed": config["seed"], "initial_model_hash": initial_hash,
                       "sample": {"sample_id": 0, "key": record["key"],
                                  "record_index": selected["record_index"],
                                  "split": "train", "G": original_g},
                       "requested_K": requested_k, "schedule": list(schedule),
                       "actual_transitions": len(schedule) - 1,
                       "network_precision": "BF16 autocast with FP32 parameters",
                       "simulator_and_losses": "FP32, full image MSE",
                       "source_sha256_before": source_before,
                       "core_sha256": sha256_file(Path(__file__).with_name("core.py"))})

        def cleanup() -> None:
            network.zero_grad(set_to_none=True)
            gc.collect()
            torch.npu.synchronize(device)
            torch.npu.empty_cache()

        def begin_phase(name: str) -> float:
            cleanup()
            torch.npu.reset_peak_memory_stats(device)
            progress(name + "_start")
            return time.monotonic()

        def end_phase(name: str, started: float, payload: dict) -> None:
            torch.npu.synchronize(device)
            result["phases"][name] = {
                **payload, "wall_seconds_including_cold_compile": time.monotonic() - started,
                "peak_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.npu.max_memory_reserved(device))}
            progress(name + "_complete")

        name = "final_clean_only_connected"
        started = begin_phase(name)
        output = module(clean, observation, strength, [schedule], retain_intermediates=True)
        trace = output["trace"]
        final_loss = trace[-1]["clean_losses"].mean()
        final_loss.backward()
        earlier = []
        for index, entry in enumerate(trace[:-1]):
            tensors = {key: tensor_gradient_record(entry[key]) for key in
                       ("clean", "epsilon_zernike", "epsilon_tilt", "next_state")}
            earlier.append({"step": index, "tensors": tensors,
                            "next_network_input_is_generated_state":
                                trace[index + 1]["input"] is entry["next_state"]})
        positive_pass = all(
            item["next_network_input_is_generated_state"]
            and all(value["gradient_exists"] and value["finite"] and value["nonzero"]
                    for value in item["tensors"].values()) for item in earlier)
        final_parameter_summary = parameter_gradient_summary(
            network, [parameter.grad for parameter in network.parameters()])
        positive_pass = positive_pass and final_parameter_summary["all"]["finite"] \
            and final_parameter_summary["all"]["l2"] > 0
        end_phase(name, started, {"passed": positive_pass,
                                  "scope": "cross-step" if earlier else "one-shot backward only",
                                  "loss": float(final_loss.detach().cpu()),
                                  "earlier_steps": earlier,
                                  "parameter_gradients": final_parameter_summary})
        del output, trace, final_loss
        entry = None  # Release the loop variable's last graph-bearing trace dict.

        if len(schedule) > 1:
            name = "transition_gradient_zero_hook_control"
            started = begin_phase(name)
            output = module(clean, observation, strength, [schedule], retain_intermediates=True)
            trace = output["trace"]
            handles = [entry["next_state"].register_hook(torch.zeros_like)
                       for entry in trace[:-1]]
            final_loss = trace[-1]["clean_losses"].mean()
            final_loss.backward()
            negative = []
            for index, entry in enumerate(trace[:-1]):
                tensors = {key: tensor_gradient_record(entry[key]) for key in
                           ("clean", "epsilon_zernike", "epsilon_tilt")}
                negative.append({"step": index, "tensors": tensors})
            negative_pass = all(
                not value["gradient_exists"] or (value["finite"] and value["max_abs"] == 0)
                for item in negative for value in item["tensors"].values())
            for handle in handles:
                handle.remove()
            end_phase(name, started, {
                "passed": negative_pass, "earlier_steps": negative,
                "control_definition": "diagnostic-only hooks zero transition gradients; this is not a detach test",
                "forward_core_changed": False})
            del output, trace, final_loss, handles, entry
        else:
            negative_pass = True
            result["phases"]["transition_gradient_zero_hook_control"] = {
                "status": "not_applicable", "reason": "schedule has no transitions"}

        component_records = {}
        for label, loss_key in (("clean", "clean_loss_sum"),
                                ("reconstruction", "reconstruction_loss_sum")):
            name = f"component_gradient_{label}"
            started = begin_phase(name)
            output = module(clean, observation, strength, [schedule])
            component_loss = output[loss_key]
            gradients = torch.autograd.grad(component_loss, tuple(network.parameters()),
                                            allow_unused=True, retain_graph=False)
            summary = parameter_gradient_summary(network, gradients)
            component_records[label] = {"loss": float(component_loss.detach().cpu()),
                                         "parameter_gradients": summary}
            end_phase(name, started, {**component_records[label],
                                      "passed": all(item["finite"] for item in summary.values()),
                                      "loss_weight": 1., "path_prediction_denominator": len(schedule)})
            del output, component_loss, gradients
        ratios = {}
        for branch in ("all", "shared", "clean", "realization"):
            clean_norm = component_records["clean"]["parameter_gradients"][branch]["l2"]
            reconstruction_norm = component_records["reconstruction"]["parameter_gradients"][branch]["l2"]
            ratio = (reconstruction_norm / clean_norm
                     if clean_norm is not None and clean_norm > 0
                     and reconstruction_norm is not None else None)
            ratios[branch] = {"unweighted_reconstruction_over_clean_l2": ratio,
                              "undefined_reason": "clean gradient norm is zero or nonfinite" if ratio is None else None,
                              "weighted_candidate_ratios": {
                                  str(weight): ratio * weight if ratio is not None else None
                                  for weight in (.05, .1, .2)}}
        result["gradient_scale_ratios"] = ratios
        cleanup()
        final_hash = state_digest(network.state_dict())
        result["final_model_hash"] = final_hash
        result["weights_unchanged"] = initial_hash == final_hash
        result["source_sha256_after"] = verify_base(config)
        all_components_finite = all(
            item["finite"] for component in component_records.values()
            for item in component["parameter_gradients"].values())
        result["passed"] = bool(positive_pass and negative_pass and all_components_finite
                                 and result["weights_unchanged"])
        result["status"] = "completed" if result["passed"] else "failed_checks"
        progress("probe_finished")
        if not result["passed"]:
            raise AssertionError("Gradient probe checks failed; inspect the per-rank results")
    except BaseException as error:
        result["status"] = "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error),
                             "traceback": traceback.format_exc()}
        atomic_json(result_path, result)
        raise


if __name__ == "__main__":
    main()

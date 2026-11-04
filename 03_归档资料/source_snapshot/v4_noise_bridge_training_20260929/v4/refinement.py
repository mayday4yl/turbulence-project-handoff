"""V4 fixed native-xi refinement. This module does not optimize network weights.

The public entry point executes samples separately, with one fresh optimizer per
sample and stage. The private native solver retains a summed sample objective
for diagnosis only; CUDA batched trajectory equivalence is not claimed.
Every call uses only the current prediction as both initialization and center.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import uuid

import torch
from physical_bridge.simulator import Realization

LAMBDA_REF = 0.01
LEARNING_RATE = 0.01
REFINEMENT_STEPS = 40
AUDIT_ATOL = 2e-7
AUDIT_RTOL = 1e-6


def clone_xi(value: Realization, *, requires_grad: bool = False) -> Realization:
    return Realization(*(x.detach().float().clone().requires_grad_(requires_grad)
                         for x in (value.epsilon_zernike, value.epsilon_tilt)))


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    """Hash persistent parameters/buffers; no alteration of device or tensors."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


@contextmanager
def _frozen_eval(*modules):
    modules = tuple(module for module in modules if module is not None)
    children = {id(child): child for module in modules for child in module.modules()}
    parameters = {id(p): p for module in modules for p in module.parameters()}
    training = {key: child.training for key, child in children.items()}
    requires_grad = {key: p.requires_grad for key, p in parameters.items()}
    try:
        for child in children.values():
            child.training = False
        for p in parameters.values():
            p.requires_grad_(False)
        yield
    finally:
        for key, child in children.items():
            child.training = training[key]
        for key, p in parameters.items():
            p.requires_grad_(requires_grad[key])


def combined_proximity(realization: Realization, center: Realization) -> torch.Tensor:
    """One mean per sample over BOTH raw blocks, with a single denominator."""
    pairs = ((realization.epsilon_zernike, center.epsilon_zernike),
             (realization.epsilon_tilt, center.epsilon_tilt))
    batch = pairs[0][0].shape[0]
    if batch < 1 or any(a.ndim < 2 or a.shape != b.shape or a.shape[0] != batch
                        for a, b in pairs):
        raise ValueError("Native blocks require equal center shapes and nonempty batch axes")
    d_xi = sum(a[0].numel() for a, _ in pairs)
    if d_xi == 0:
        raise ValueError("Empty native xi")
    return sum((a - b.detach()).square().flatten(1).sum(1) for a, b in pairs) / d_xi


@dataclass(frozen=True)
class ObjectiveTerms:
    total: torch.Tensor
    observation: torch.Tensor
    proximity: torch.Tensor


def refinement_objective(*, simulator, clean, realization, observation,
                         original_g, center) -> ObjectiveTerms:
    """Fixed clean/Y/G/center; only realization is differentiated."""
    with torch.autocast(device_type=clean.device.type, enabled=False):
        rendered = simulator.render(clean.detach().float(), realization, original_g).output
        if rendered.shape != observation.shape or rendered.ndim != 4 or rendered.shape[1] != 3:
            raise ValueError("Expected equal RGB [B,3,H,W] render and observation")
        residual = (rendered.float() - observation.detach().float()).square().flatten(1).mean(1)
        proximity = combined_proximity(realization, center)
        total = residual + LAMBDA_REF * proximity
        for name, value in (("observation objective", residual), ("proximity", proximity),
                            ("total objective", total)):
            _finite(value, name)
        return ObjectiveTerms(total, residual, proximity)


@dataclass
class RefinementResult:
    realization: Realization
    initial_objective: list[float]
    best_objective: list[float]
    best_iterations: list[int]
    candidates: list[dict]
    trace: list[dict]
    invariance: dict
    optimizer: dict


class RefinementFailure(RuntimeError):
    def __init__(self, message, *, failed_update, candidates, trace):
        super().__init__(message)
        self.failed_update = failed_update
        self.candidates = candidates
        self.trace = trace
        self.success = False


def _pair_where(mask, first, second):
    return Realization(*(torch.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
                         for a, b in zip((first.epsilon_zernike, first.epsilon_tilt),
                                         (second.epsilon_zernike, second.epsilon_tilt))))


def _solve_native(*, simulator, clean: torch.Tensor, xi_hat: Realization,
                  observation: torch.Tensor, original_g, network=None,
                  audit_updates: bool = False) -> RefinementResult:
    """Private solver; public production execution supplies exactly one sample.

Call this only at nonterminal inference stages. The rollout owns stage legality;
this solver deliberately accepts the original G, never the next stage's g.
No previous xi or optimizer state can be supplied. Caller no_grad is supported;
inference_mode is rejected because it suppresses the required input derivatives.
"""
    candidates, trace, step = [], [], 0
    if torch.is_inference_mode_enabled():
        raise ValueError("Refinement needs input derivatives; use no_grad, not inference_mode")
    if any(p.requires_grad for p in simulator.parameters()):
        raise ValueError("Simulator parameters must already be frozen before refinement")
    if clean.ndim != 4 or clean.shape[1] != 3 or clean.shape != observation.shape:
        raise ValueError("Expected matching clean/Y [B,3,H,W]")
    batch = clean.shape[0]
    g = torch.as_tensor(original_g, device=clean.device)
    if g.ndim == 0:
        g = g.repeat(batch)
    if g.shape != (batch,) or not bool(((g >= 1) & (g <= 4) & (g == g.round())).all()):
        raise ValueError("Original G must contain one legal 1/2/3/4 value per sample")
    g = g.detach().float().clone()
    fixed_clean = clean.detach().float().clone()
    fixed_y = observation.detach().float().clone()
    center = clone_xi(xi_hat)
    current = clone_xi(xi_hat, requires_grad=True)
    blocks = (current.epsilon_zernike, current.epsilon_tilt)
    if any(x.shape[0] != batch or x.device != clean.device for x in blocks):
        raise ValueError("Native xi must match the clean batch and device")
    combined_proximity(current, center)  # Also validate native block layouts.
    fixed_tensors = {"caller_clean": clean, "caller_Y": observation,
                     "caller_xi_zernike": xi_hat.epsilon_zernike,
                     "caller_xi_tilt": xi_hat.epsilon_tilt,
                     "fixed_clean": fixed_clean, "fixed_Y": fixed_y,
                     "fixed_G": g, "center_zernike": center.epsilon_zernike,
                     "center_tilt": center.epsilon_tilt}
    before = {name: tensor_hash(value) for name, value in fixed_tensors.items()}
    before["simulator"] = module_hash(simulator)
    if network is not None:
        before["network"] = module_hash(network)
    optimizer = torch.optim.Adam(blocks, lr=LEARNING_RATE, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
    optimizer_info = {"instance_id": str(uuid.uuid4()), "state_entries_at_start": len(optimizer.state),
                      "steps": REFINEMENT_STEPS, "lr": LEARNING_RATE, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0, "lambda_ref": LAMBDA_REF,
                      "d_xi_per_sample": sum(x[0].numel() for x in blocks),
                      "differentiated_reduction": "sum_of_per_sample_objectives",
                      "optimized_blocks": ["epsilon_zernike", "epsilon_tilt"],
                      "audit_updates": audit_updates}
    if audit_updates:
        # Independent mathematical moments, in double precision, consume the
        # same observed gradients. No second simulator execution is involved.
        reference_m = [torch.zeros_like(x, dtype=torch.float64) for x in blocks]
        reference_v = [torch.zeros_like(x, dtype=torch.float64) for x in blocks]
        optimizer_info["adam_audit"] = {"reference_dtype": "float64", "atol": AUDIT_ATOL,
            "rtol": AUDIT_RTOL, "gradient_source": "same_observed_gradient",
            "independent_moment_states": True}
    try:
        for name, value in fixed_tensors.items():
            _finite(value, name)
        with _frozen_eval(simulator, network), torch.enable_grad(), torch.autocast(
                device_type=clean.device.type, enabled=False):
            def evaluate(update):
                terms = refinement_objective(simulator=simulator, clean=fixed_clean,
                    realization=current, observation=fixed_y, original_g=g, center=center)
                candidates.append({"update": update, "total": terms.total.detach().cpu().tolist(),
                    "observation": terms.observation.detach().cpu().tolist(),
                    "proximity": terms.proximity.detach().cpu().tolist()})
                if audit_updates:
                    candidates[-1]["xi_sha256"] = [
                        {"epsilon_zernike": tensor_hash(current.epsilon_zernike[b:b+1]),
                         "epsilon_tilt": tensor_hash(current.epsilon_tilt[b:b+1])}
                        for b in range(batch)]
                return terms

            terms = evaluate(0)
            initial = terms.total.detach().clone()
            best_values = initial.clone()
            best_iterations = torch.zeros(batch, dtype=torch.long, device=clean.device)
            best = clone_xi(current)
            for step in range(1, REFINEMENT_STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                terms.total.sum().backward()
                gradient_norms = []
                for name, block in zip(("zernike", "tilt"), blocks):
                    if block.grad is None:
                        raise RuntimeError(f"Disconnected {name} gradient")
                    _finite(block.grad, f"{name} gradient")
                    gradient_norms.append(block.grad.detach().flatten(1).norm(dim=1).cpu().tolist())
                if audit_updates:
                    expected_parameters = []
                    for index, block in enumerate(blocks):
                        gradient = block.grad.detach().double()
                        reference_m[index] = .9 * reference_m[index] + .1 * gradient
                        reference_v[index] = .999 * reference_v[index] + .001 * gradient.square()
                        corrected_m = reference_m[index] / (1 - .9 ** step)
                        corrected_v = reference_v[index] / (1 - .999 ** step)
                        expected_parameters.append(block.detach().double() -
                            .01 * corrected_m / (corrected_v.sqrt() + 1e-8))
                optimizer.step()
                for name, block in zip(("zernike", "tilt"), blocks):
                    _finite(block, f"updated {name}")
                audit = {}
                if audit_updates:
                    for index, (name, block) in enumerate(zip(("zernike", "tilt"), blocks)):
                        state = optimizer.state[block]
                        comparisons = {"parameter": (block, expected_parameters[index]),
                            "first_moment": (state["exp_avg"], reference_m[index]),
                            "second_moment": (state["exp_avg_sq"], reference_v[index])}
                        audit[name] = {}
                        for label, (actual, expected) in comparisons.items():
                            actual = actual.detach().double()
                            torch.testing.assert_close(actual, expected, atol=AUDIT_ATOL,
                                                       rtol=AUDIT_RTOL)
                            audit[name][label + "_max_abs"] = float((actual - expected).abs().max())
                        audit[name]["passed"] = True
                    del expected_parameters
                del terms
                terms = evaluate(step)
                with torch.no_grad():
                    better = terms.total.detach() < best_values
                    best = clone_xi(_pair_where(better, current, best))
                    best_iterations = torch.where(better, step, best_iterations)
                    best_values = torch.minimum(best_values, terms.total.detach())
                trace.append({"update": step, "gradient_l2_zernike": gradient_norms[0],
                    "gradient_l2_tilt": gradient_norms[1], "finite": True,
                    "best_iterations": best_iterations.cpu().tolist(),
                    "best_objective": best_values.cpu().tolist()})
                if audit_updates:
                    trace[-1]["adam_update_audit"] = audit
        after = {name: tensor_hash(value) for name, value in fixed_tensors.items()}
        after["simulator"] = module_hash(simulator)
        if network is not None:
            after["network"] = module_hash(network)
        unchanged = {name: before[name] == after[name] for name in before}
        if not all(unchanged.values()):
            raise RuntimeError(f"Refinement mutated fixed objects: {unchanged}")
        optimizer_info["state_steps_at_end"] = [int(optimizer.state[x]["step"].item()) for x in blocks]
        invariance = {"unchanged": unchanged, "before_sha256": before, "after_sha256": after}
        if audit_updates:
            selected_hashes = [{"epsilon_zernike": tensor_hash(best.epsilon_zernike[b:b+1]),
                                "epsilon_tilt": tensor_hash(best.epsilon_tilt[b:b+1])}
                               for b in range(batch)]
            candidate_hashes = [candidates[iteration]["xi_sha256"][b]
                                for b, iteration in enumerate(best_iterations.cpu().tolist())]
            if selected_hashes != candidate_hashes:
                raise RuntimeError("Selected native pair does not match its total-J candidate")
            invariance["best_pair_matches_candidate"] = [True] * batch
            invariance["selected_pair_sha256"] = selected_hashes
            optimizer_info["adam_audit"]["updates_verified"] = REFINEMENT_STEPS
        return RefinementResult(best, initial.cpu().tolist(), best_values.cpu().tolist(),
            best_iterations.cpu().tolist(), candidates, trace,
            invariance, optimizer_info)
    except (RuntimeError, FloatingPointError, ValueError, AssertionError) as exc:
        raise RefinementFailure(str(exc), failed_update=step, candidates=candidates, trace=trace) from exc


def refine_xi(*, simulator, clean: torch.Tensor, xi_hat: Realization,
              observation: torch.Tensor, original_g, network=None,
              audit_updates: bool = False) -> RefinementResult:
    """Production refinement: 40 updates / 41 candidates independently per sample.

Serial execution avoids an unsupported claim of CUDA batched solver equivalence.
It does not guarantee repeated-run native xi reproducibility: the unchanged
simulator can contain nondeterministic CUDA backward operations. Optional audits
validate actual Adam arithmetic and paired selection within each execution.
"""
    if clean.ndim != 4 or clean.shape[1] != 3 or clean.shape != observation.shape:
        raise ValueError("Expected matching clean/Y [B,3,H,W]")
    batch = clean.shape[0]
    if batch < 1 or any(x.ndim < 2 or x.shape[0] != batch for x in
                       (xi_hat.epsilon_zernike, xi_hat.epsilon_tilt)):
        raise ValueError("Expected matching nonempty clean/Y/native-xi batch")
    g = torch.as_tensor(original_g, device=clean.device)
    if g.ndim == 0:
        g = g.repeat(batch)
    if g.shape != (batch,):
        raise ValueError("Original G must contain one value per sample")
    results = []
    for b in range(batch):
        try:
            results.append(_solve_native(simulator=simulator, clean=clean[b:b+1],
                xi_hat=Realization(xi_hat.epsilon_zernike[b:b+1], xi_hat.epsilon_tilt[b:b+1]),
                observation=observation[b:b+1], original_g=g[b:b+1], network=network,
                audit_updates=audit_updates))
        except RefinementFailure as exc:
            exc.sample_index = b
            exc.completed_samples = len(results)
            raise
    per_sample_optimizer = [dict(result.optimizer) for result in results]
    if batch == 1:
        result = results[0]
        result.optimizer.update(execution="per_sample_serial", per_sample=per_sample_optimizer,
            instance_id_kind="optimizer", optimizer_instances=1,
            candidate_count_per_sample=REFINEMENT_STEPS + 1,
            simulator_objective_calls=REFINEMENT_STEPS + 1)
        return result
    candidates = []
    for index in range(REFINEMENT_STEPS + 1):
        candidate = {"update": index}
        for key in ("total", "observation", "proximity"):
            candidate[key] = [result.candidates[index][key][0] for result in results]
        if audit_updates:
            candidate["xi_sha256"] = [result.candidates[index]["xi_sha256"][0] for result in results]
        candidates.append(candidate)
    trace = []
    for index in range(REFINEMENT_STEPS):
        row = {"update": index + 1, "finite": all(result.trace[index]["finite"] for result in results)}
        for key in ("gradient_l2_zernike", "gradient_l2_tilt", "best_iterations", "best_objective"):
            row[key] = [result.trace[index][key][0] for result in results]
        if audit_updates:
            row["adam_update_audit_per_sample"] = [result.trace[index]["adam_update_audit"] for result in results]
        trace.append(row)
    optimizer = dict(per_sample_optimizer[0])
    optimizer.update(instance_id=str(uuid.uuid4()), execution="per_sample_serial",
        instance_id_kind="serial_group", per_sample=per_sample_optimizer, optimizer_instances=batch,
        candidate_count_per_sample=REFINEMENT_STEPS + 1,
        simulator_objective_calls=(REFINEMENT_STEPS + 1) * batch,
        state_steps_at_end_per_sample=[r.optimizer["state_steps_at_end"] for r in results])
    invariance = {"unchanged": {key: all(r.invariance["unchanged"][key] for r in results)
                                for key in results[0].invariance["unchanged"]},
                  "per_sample": [r.invariance for r in results]}
    if audit_updates:
        invariance["best_pair_matches_candidate"] = [r.invariance["best_pair_matches_candidate"][0]
                                                     for r in results]
    return RefinementResult(
        Realization(torch.cat([r.realization.epsilon_zernike for r in results]),
                    torch.cat([r.realization.epsilon_tilt for r in results])),
        [r.initial_objective[0] for r in results], [r.best_objective[0] for r in results],
        [r.best_iterations[0] for r in results], candidates, trace, invariance, optimizer)

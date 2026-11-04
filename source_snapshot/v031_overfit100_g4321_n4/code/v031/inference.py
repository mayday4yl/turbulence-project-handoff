"""V0.31 inference selection/orchestration, without a frozen numeric solver.

Stage 1 implements the method's selection rule only. It supplies no optimizer,
learning rate, iteration count, clipping, projection, or stopping defaults.
``run_selection_logic_test`` is deliberately separate from production inference:
an injected stub exercises control flow and does not validate restoration quality.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol

import torch

from physical_bridge.simulator import Realization


Reduction = Literal["sum", "mean"]


class RefinementConfigPending(RuntimeError):
    """Multi-step inference needs an explicitly frozen stage-two protocol."""


@dataclass(frozen=True)
class RefinementNumericConfig:
    """Explicit pending interface, not a proposed numerical configuration.

    ``None`` always means undecided. An explicitly disabled policy must instead
    be represented by e.g. ``{"kind": "none"}``. Optimizer options must enumerate
    its relevant hyperparameters; the eventual refiner must not inherit library
    defaults accidentally. Policies also declare any behavior on nonfinite
    values and optimizer-state reuse across calls.
    """

    optimizer: str | None = None
    optimizer_options: Mapping[str, Any] | None = None
    learning_rate: float | None = None
    first_iterations: int | None = None
    later_iterations: int | None = None
    objective_reduction: Reduction | None = None
    stopping_policy: Mapping[str, Any] | None = None
    return_policy: Mapping[str, Any] | None = None
    optimizer_state_policy: Mapping[str, Any] | None = None
    gradient_clip_policy: Mapping[str, Any] | None = None
    projection_policy: Mapping[str, Any] | None = None
    gradient_normalization_policy: Mapping[str, Any] | None = None
    nonfinite_policy: Mapping[str, Any] | None = None

    @property
    def pending_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self) if getattr(self, field.name) is None)

    def require_complete(self) -> None:
        if self.pending_fields:
            raise RefinementConfigPending(
                "refinement_numeric_config_pending: " + ", ".join(self.pending_fields)
            )
        _validate_reduction(self.objective_reduction)
        if not self.optimizer or self.learning_rate <= 0:
            raise ValueError("The declared optimizer and positive learning rate are required")
        for name in ("first_iterations", "later_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _validate_reduction(reduction: str | None) -> None:
    if reduction not in ("sum", "mean"):
        raise ValueError("objective_reduction must explicitly be 'sum' or 'mean'")


def observation_objective(
    *,
    simulator: torch.nn.Module,
    current_clean: torch.Tensor,
    realization: Realization,
    observation: torch.Tensor,
    initial_G: int,
    reduction: Reduction,
) -> torch.Tensor:
    """Full-image observation residual with fixed current clean and original G/Y.

    This objective has no GT argument, crop, prox/prior, clamp, or auxiliary term.
    Detaching fixed inputs preserves derivatives with respect to the two xi
    tensors; simulator parameters are frozen by the inference orchestration.
    """
    _validate_reduction(reduction)
    _severity_path(initial_G, one_shot=False)
    with torch.autocast(device_type=observation.device.type, enabled=False):
        reconstruction = simulator.render(
            current_clean.detach().float(), realization, initial_G
        ).output
        if reconstruction.shape != observation.shape:
            raise ValueError("Simulator reconstruction and original observation must match")
        residual = (reconstruction.float() - observation.detach().float()).square()
        return residual.sum() if reduction == "sum" else residual.mean()


@dataclass(frozen=True)
class InitializationSelection:
    realization: Realization
    source: Literal["network", "previous_refined"]
    previous_objective: float | None
    network_objective: float | None


def select_initialization(
    *,
    network_realization: Realization,
    previous_refined: Realization | None,
    objective: Callable[[Realization], torch.Tensor],
) -> InitializationSelection:
    """Evaluate both *whole* candidates using one current-step objective.

    No previous score is accepted. At the first step there is no comparison.
    Ties inherit the complete previous refined realization.
    """
    if previous_refined is None:
        return InitializationSelection(network_realization, "network", None, None)
    with torch.no_grad():
        previous_score = _finite_score(objective(previous_refined))
        network_score = _finite_score(objective(network_realization))
    if previous_score <= network_score:
        return InitializationSelection(
            previous_refined, "previous_refined", previous_score, network_score
        )
    return InitializationSelection(network_realization, "network", previous_score, network_score)


def _finite_score(value: torch.Tensor) -> float:
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Candidate objective must be one finite scalar")
    return float(value.detach().cpu())


@dataclass(frozen=True)
class RefinementRequest:
    """The refiner optimizes xi using this exact already-bound objective."""

    objective: Callable[[Realization], torch.Tensor]
    initialization: Realization
    numeric_config: RefinementNumericConfig
    maximum_iterations: int | None
    prediction_index: int  # one-based


class Refiner(Protocol):
    def __call__(self, request: RefinementRequest) -> Realization: ...


@dataclass(frozen=True)
class InferenceStep:
    prediction_index: int
    current_g: int
    initial_G: int
    input_source: str
    next_g: int | None
    initialization_source: str | None
    previous_objective: float | None
    network_objective: float | None
    refinement_calls: int


@dataclass(frozen=True)
class InferenceResult:
    clean: torch.Tensor
    severity_path: tuple[int, ...]
    steps: tuple[InferenceStep, ...]
    network_predictions: int
    transition_renders: int
    candidate_objective_renders: int
    refinement_calls: int
    refinement_numeric_config: str
    validation_scope: str


def _severity_path(initial_G: int, *, one_shot: bool) -> tuple[int, ...]:
    if isinstance(initial_G, bool) or initial_G not in (4, 3, 2, 1):
        raise ValueError("initial_G must be one of 4, 3, 2, 1")
    value = int(initial_G)
    return (value,) if one_shot else tuple(range(value, 0, -1))


@contextmanager
def _frozen_eval(*modules: torch.nn.Module) -> Iterator[None]:
    """Freeze parameters and buffers during inference; restore caller flags."""
    submodules = {id(child): child for module in modules for child in module.modules()}
    parameters = {id(p): p for module in modules for p in module.parameters()}
    training = {key: child.training for key, child in submodules.items()}
    requires_grad = {key: p.requires_grad for key, p in parameters.items()}
    try:
        for child in submodules.values():
            child.training = False
        for p in parameters.values():
            p.requires_grad_(False)
        yield
    finally:
        for key, child in submodules.items():
            child.training = training[key]
        for key, p in parameters.items():
            p.requires_grad_(requires_grad[key])


def _clone_realization(value: Realization) -> Realization:
    return Realization(
        value.epsilon_zernike.detach().float().clone(),
        value.epsilon_tilt.detach().float().clone(),
    )


def run_inference(
    *,
    model: torch.nn.Module,
    simulator: torch.nn.Module,
    observation: torch.Tensor,
    initial_G: int,
    one_shot: bool = False,
    numeric_config: RefinementNumericConfig | None = None,
    refiner: Refiner | None = None,
) -> InferenceResult:
    """Production entry: multi-step inference fails closed while config is pending.

    One-shot always makes exactly one prediction, at any original G. Stage 1
    provides no production numerical refiner; stage 2 must supply and validate it.
    """
    path = _severity_path(initial_G, one_shot=one_shot)
    config = numeric_config or RefinementNumericConfig()
    if len(path) > 1:
        config.require_complete()
        if refiner is None:
            raise RefinementConfigPending("A stage-two numeric refiner must be supplied")
    return _run_rollout(
        model=model, simulator=simulator, observation=observation, path=path,
        config=config, refiner=refiner, validation_scope="inference_no_quality_claim",
    )


def run_selection_logic_test(
    *,
    model: torch.nn.Module,
    simulator: torch.nn.Module,
    observation: torch.Tensor,
    initial_G: int,
    objective_reduction: Reduction,
    injected_test_refiner: Refiner,
    one_shot: bool = False,
) -> InferenceResult:
    """Stage-one stub harness: NOT a numeric-solver or restoration-quality test.

    The reduction is explicit and is shared by comparison and the injected
    refiner's objective. No iteration count or numerical policy is supplied.
    """
    _validate_reduction(objective_reduction)
    return _run_rollout(
        model=model, simulator=simulator, observation=observation,
        path=_severity_path(initial_G, one_shot=one_shot),
        config=RefinementNumericConfig(objective_reduction=objective_reduction),
        refiner=injected_test_refiner, validation_scope="selection_logic_test_only",
    )


def _run_rollout(
    *,
    model: torch.nn.Module,
    simulator: torch.nn.Module,
    observation: torch.Tensor,
    path: tuple[int, ...],
    config: RefinementNumericConfig,
    refiner: Refiner | None,
    validation_scope: str,
) -> InferenceResult:
    if observation.ndim != 4 or observation.shape[:2] != (1, 3):
        raise ValueError("Inference expects one RGB observation with shape [1,3,H,W]")
    current = observation.detach().float()
    original_observation = current
    initial_G = path[0]
    previous_refined = None
    steps = []
    candidate_renders = 0
    refinement_calls = 0
    with _frozen_eval(model, simulator):
        for index, current_g in enumerate(path):
            with torch.no_grad(), torch.autocast(
                device_type=observation.device.type, enabled=False
            ):
                prediction = model(
                    current,
                    torch.tensor([current_g], device=current.device, dtype=torch.float32),
                )
                current_clean = prediction.clean.detach().float()
            next_g = path[index + 1] if index + 1 < len(path) else None
            selection = None
            if next_g is not None:
                # Each closure binds this prediction's clean and the original G/Y.
                def objective(candidate: Realization, fixed_clean=current_clean) -> torch.Tensor:
                    return observation_objective(
                        simulator=simulator, current_clean=fixed_clean,
                        realization=candidate, observation=original_observation,
                        initial_G=initial_G, reduction=config.objective_reduction,
                    )

                selection = select_initialization(
                    network_realization=prediction.realization,
                    previous_refined=previous_refined, objective=objective,
                )
                candidate_renders += 0 if previous_refined is None else 2
                request = RefinementRequest(
                    objective=objective,
                    initialization=_clone_realization(selection.realization),
                    numeric_config=config,
                    maximum_iterations=(
                        config.first_iterations if index == 0 else config.later_iterations
                    ),
                    prediction_index=index + 1,
                )
                # Only the selected whole pair reaches the refiner, exactly once.
                with torch.enable_grad():
                    previous_refined = _clone_realization(refiner(request))
                refinement_calls += 1
                with torch.no_grad(), torch.autocast(
                    device_type=observation.device.type, enabled=False
                ):
                    current = simulator.render(current_clean, previous_refined, next_g).output.float()
            steps.append(InferenceStep(
                prediction_index=index + 1, current_g=current_g, initial_G=initial_G,
                input_source="original_observation" if index == 0 else "predicted_transition",
                next_g=next_g,
                initialization_source=None if selection is None else selection.source,
                previous_objective=None if selection is None else selection.previous_objective,
                network_objective=None if selection is None else selection.network_objective,
                refinement_calls=int(next_g is not None),
            ))
    return InferenceResult(
        clean=current_clean, severity_path=path, steps=tuple(steps),
        network_predictions=len(path), transition_renders=len(path) - 1,
        candidate_objective_renders=candidate_renders, refinement_calls=refinement_calls,
        refinement_numeric_config=("pending_stage2" if config.pending_fields else "explicitly_supplied"),
        validation_scope=validation_scope,
    )

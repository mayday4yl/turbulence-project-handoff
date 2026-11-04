"""Toy control-flow checks only; these are not numeric refinement/quality tests."""

import inspect
import unittest
from types import SimpleNamespace

import torch

from physical_bridge.simulator import Realization
from v031.inference import (
    RefinementConfigPending,
    RefinementNumericConfig,
    observation_objective,
    run_inference,
    run_selection_logic_test,
    select_initialization,
)


def xi(zernike, tilt):
    return Realization(
        torch.full((1, 1, 1, 1), float(zernike)),
        torch.full((1, 1, 1, 2), float(tilt)),
    )


class ToySimulator(torch.nn.Module):
    """Scalar-branch renderer, intentionally not a physical simulator fixture."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def render(self, clean, realization, severity):
        self.calls.append((clean.detach().clone(), realization, severity))
        value = realization.epsilon_zernike.mean() + realization.epsilon_tilt.mean()
        return SimpleNamespace(output=clean + self.weight * severity * value)


class ToyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def forward(self, image, severity):
        self.calls.append((image.detach().clone(), float(severity.item())))
        current_g = float(severity.item())
        return SimpleNamespace(
            clean=image * self.weight + current_g * 0.1,
            realization=xi(current_g * 0.2, current_g * 0.3),
        )


class IdentityRefinerStub:
    """No optimization: inspect the common objective and return initialization."""

    def __init__(self):
        self.requests = []
        self.objective_values = []

    def __call__(self, request):
        self.requests.append(request)
        self.objective_values.append(float(request.objective(request.initialization)))
        return request.initialization


class InferenceSelectionTest(unittest.TestCase):
    def objective(self, simulator, clean, observation, initial_G=1, reduction="sum"):
        return lambda realization: observation_objective(
            simulator=simulator, current_clean=clean, realization=realization,
            observation=observation, initial_G=initial_G, reduction=reduction,
        )

    def test_whole_pair_is_selected_without_component_mixing(self):
        simulator = ToySimulator()
        previous, network = xi(1, 3), xi(2, 1)
        selected = select_initialization(
            previous_refined=previous, network_realization=network,
            objective=self.objective(simulator, torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)),
        )
        self.assertIs(selected.realization, network)
        self.assertIs(selected.realization.epsilon_zernike, network.epsilon_zernike)
        self.assertIs(selected.realization.epsilon_tilt, network.epsilon_tilt)
        self.assertEqual(selected.source, "network")
        self.assertEqual(len(simulator.calls), 2)

    def test_candidates_are_reevaluated_with_the_same_current_clean(self):
        simulator = ToySimulator()
        previous, network = xi(0, 1), xi(2, 1)
        observation = torch.zeros(1, 3, 2, 2)
        old_clean = torch.zeros_like(observation)
        first = select_initialization(
            previous_refined=previous, network_realization=network,
            objective=self.objective(simulator, old_clean, observation),
        )
        self.assertEqual(first.source, "previous_refined")
        current_clean = torch.full_like(observation, -3)
        second = select_initialization(
            previous_refined=previous, network_realization=network,
            objective=self.objective(simulator, current_clean, observation),
        )
        self.assertEqual(second.source, "network")
        self.assertEqual(second.previous_objective, 48)
        self.assertEqual(second.network_objective, 0)
        for clean, _, severity in simulator.calls[-2:]:
            self.assertTrue(torch.equal(clean, current_clean))
            self.assertEqual(severity, 1)

    def test_equal_scores_inherit_the_complete_previous_realization(self):
        simulator = ToySimulator()
        previous, network = xi(1, 3), xi(3, 1)
        selected = select_initialization(
            previous_refined=previous, network_realization=network,
            objective=self.objective(simulator, torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)),
        )
        self.assertIs(selected.realization, previous)
        self.assertEqual(selected.source, "previous_refined")
        self.assertEqual(selected.previous_objective, selected.network_objective)

    def test_first_step_uses_network_without_a_comparison(self):
        network = xi(1, 2)

        def forbidden(_):
            self.fail("First step has no previous candidate to compare")

        selected = select_initialization(
            previous_refined=None, network_realization=network, objective=forbidden,
        )
        self.assertIs(selected.realization, network)
        self.assertIsNone(selected.network_objective)

    def test_objective_reduction_is_explicit_full_image_and_has_no_regularizer(self):
        simulator = ToySimulator()
        clean = torch.ones(1, 3, 2, 2)
        observation = torch.zeros_like(clean)
        # Nonzero components sum to zero: adding a prior would change the answer.
        realization = xi(10, -10)
        summed = self.objective(simulator, clean, observation, reduction="sum")(realization)
        averaged = self.objective(simulator, clean, observation, reduction="mean")(realization)
        self.assertEqual(float(summed), 12)
        self.assertEqual(float(averaged), 1)
        with self.assertRaises(ValueError):
            self.objective(simulator, clean, observation, reduction=None)(realization)

    def test_objective_fixes_clean_and_observation_but_allows_xi_gradients(self):
        simulator = ToySimulator().requires_grad_(False)
        clean = torch.ones(1, 3, 2, 2, requires_grad=True)
        observation = torch.zeros_like(clean, requires_grad=True)
        realization = xi(1, 2)
        realization.epsilon_zernike.requires_grad_(True)
        realization.epsilon_tilt.requires_grad_(True)
        objective = self.objective(simulator, clean, observation)(realization)
        objective.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNone(observation.grad)
        self.assertIsNone(simulator.weight.grad)
        self.assertGreater(float(realization.epsilon_zernike.grad.abs().sum()), 0)
        self.assertGreater(float(realization.epsilon_tilt.grad.abs().sum()), 0)

    def test_nonfinite_scores_do_not_silently_choose_a_candidate(self):
        with self.assertRaises(FloatingPointError):
            select_initialization(
                network_realization=xi(1, 2), previous_refined=xi(3, 4),
                objective=lambda _: torch.tensor(float("nan")),
            )

    def test_multistep_pending_config_fails_before_network_or_simulator_calls(self):
        for config in (None, RefinementNumericConfig(objective_reduction="mean")):
            model, simulator = ToyNetwork(), ToySimulator()
            with self.assertRaisesRegex(RefinementConfigPending, "refinement_numeric_config_pending"):
                run_inference(
                    model=model, simulator=simulator, observation=torch.zeros(1, 3, 2, 2),
                    initial_G=4, numeric_config=config, refiner=IdentityRefinerStub(),
                )
            self.assertEqual(model.calls, [])
            self.assertEqual(simulator.calls, [])

    def test_k1_at_all_initial_severities_has_no_refinement_or_render(self):
        for initial_G in (4, 3, 2, 1):
            model, simulator, stub = ToyNetwork(), ToySimulator(), IdentityRefinerStub()
            result = run_inference(
                model=model, simulator=simulator, observation=torch.zeros(1, 3, 2, 2),
                initial_G=initial_G, one_shot=True, refiner=stub,
            )
            self.assertEqual(result.severity_path, (initial_G,))
            self.assertEqual(result.network_predictions, 1)
            self.assertEqual(result.refinement_calls, 0)
            self.assertEqual(result.transition_renders, 0)
            self.assertEqual(result.candidate_objective_renders, 0)
            self.assertEqual(len(stub.requests), 0)
            self.assertEqual(simulator.calls, [])

    def test_full_tail_counts_and_original_observation_binding(self):
        for initial_G in (4, 3, 2, 1):
            model, simulator, stub = ToyNetwork(), ToySimulator(), IdentityRefinerStub()
            observation = torch.full((1, 3, 2, 2), 0.1)
            result = run_selection_logic_test(
                model=model, simulator=simulator, observation=observation,
                initial_G=initial_G, objective_reduction="mean", injected_test_refiner=stub,
            )
            self.assertEqual(result.network_predictions, initial_G)
            self.assertEqual(result.transition_renders, initial_G - 1)
            self.assertEqual(result.refinement_calls, initial_G - 1)
            self.assertEqual(len(stub.requests), initial_G - 1)
            self.assertEqual(result.candidate_objective_renders, max(0, 2 * (initial_G - 2)))
            self.assertEqual(result.validation_scope, "selection_logic_test_only")
            self.assertEqual(result.refinement_numeric_config, "pending_stage2")
            self.assertEqual([g for _, g in model.calls], list(range(initial_G, 0, -1)))
            self.assertIsNone(result.steps[-1].next_g)
            self.assertEqual(result.steps[-1].refinement_calls, 0)
            self.assertIsNone(result.steps[-1].initialization_source)
            for request in stub.requests:
                self.assertIsNone(request.maximum_iterations)
                self.assertEqual(request.numeric_config.objective_reduction, "mean")
            # Each comparison evaluates both candidates; then the stub evaluates
            # only the selected candidate using the exact same bound objective.
            for step, request, stub_score in zip(result.steps[1:-1], stub.requests[1:], stub.objective_values[1:]):
                expected = (step.previous_objective if step.initialization_source == "previous_refined"
                            else step.network_objective)
                self.assertAlmostEqual(expected, stub_score, places=5)
                objective_score = float(request.objective(request.initialization))
                self.assertAlmostEqual(objective_score, stub_score, places=5)
            # All non-transition renders evaluate original G, including later
            # predictions with weaker current g; bound closures retain their clean.
            cursor = 0
            for step in result.steps[:-1]:
                comparisons = 0 if step.prediction_index == 1 else 2
                for _, _, g in simulator.calls[cursor:cursor + comparisons + 1]:
                    self.assertEqual(g, initial_G)
                fixed_clean, selected, _ = simulator.calls[cursor + comparisons]
                xi_value = selected.epsilon_zernike.mean() + selected.epsilon_tilt.mean()
                manual_score = (fixed_clean + initial_G * xi_value - observation).square().mean()
                self.assertAlmostEqual(float(manual_score), stub.objective_values[step.prediction_index - 1])
                self.assertEqual(simulator.calls[cursor + comparisons + 1][2], step.next_g)
                cursor += comparisons + 2
            self.assertTrue(torch.equal(observation, torch.full_like(observation, 0.1)))

    def test_network_and_simulator_weights_gradients_and_modes_are_unchanged(self):
        model, simulator = ToyNetwork(), ToySimulator()
        model.train()
        simulator.eval()
        model.weight.grad = torch.tensor(7.0)
        before_model = model.weight.detach().clone()
        before_simulator = simulator.weight.detach().clone()

        def gradient_probe_stub(request):
            # Probe only: no optimizer construction or update. This checks the
            # refiner can differentiate xi without accumulating network grads.
            selected = request.initialization
            selected.epsilon_zernike.requires_grad_(True)
            selected.epsilon_tilt.requires_grad_(True)
            request.objective(selected).backward()
            self.assertFalse(model.weight.requires_grad)
            self.assertFalse(simulator.weight.requires_grad)
            self.assertIsNotNone(selected.epsilon_zernike.grad)
            self.assertIsNotNone(selected.epsilon_tilt.grad)
            return selected

        run_selection_logic_test(
            model=model, simulator=simulator, observation=torch.ones(1, 3, 2, 2),
            initial_G=4, objective_reduction="sum", injected_test_refiner=gradient_probe_stub,
        )
        self.assertTrue(torch.equal(before_model, model.weight))
        self.assertTrue(torch.equal(before_simulator, simulator.weight))
        self.assertEqual(float(model.weight.grad), 7)
        self.assertIsNone(simulator.weight.grad)
        self.assertTrue(model.training)
        self.assertFalse(simulator.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertTrue(simulator.weight.requires_grad)

    def test_solver_entry_has_no_ground_truth_or_legacy_method_arguments(self):
        forbidden = {"true_clean", "true_realization", "true_states", "clean_gt", "xi_gt", "method"}
        self.assertTrue(forbidden.isdisjoint(inspect.signature(run_inference).parameters))
        self.assertTrue(forbidden.isdisjoint(inspect.signature(select_initialization).parameters))


if __name__ == "__main__":
    unittest.main()

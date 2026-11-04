"""CPU numerical/control tests; real-operator smoke is separately required."""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from physical_bridge.simulator import Realization

from v031.evaluation import (_checkpoint_state, clean_metrics, paired_gain,
                            rollout_predictions, summarize_records)
from v031.inference import run_inference
from v031.refinement import (NativeAdamRefiner, RefinementFailure, numeric_config,
                            per_sample_objective, rank_candidates, select_batched_initialization,
                            solve_adam)


def xi(z, t):
    return Realization(torch.tensor(z, dtype=torch.float32).reshape(-1, 1, 1, 1),
                       torch.tensor(t, dtype=torch.float32).reshape(-1, 1, 1, 1))


class Simulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def render(self, clean, realization, severity):
        offset = realization.epsilon_zernike + 2 * realization.epsilon_tilt
        return SimpleNamespace(output=clean + severity * self.weight * offset)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.register_buffer("fixed_buffer", torch.tensor(3.0))

    def forward(self, image, severity):
        batch = image.shape[0]
        return SimpleNamespace(clean=image * self.weight + severity[:, None, None, None] * 0.1,
                               realization=xi([0.15] * batch, [-0.05] * batch))


class NumericRefinementTests(unittest.TestCase):
    def test_batched_solver_matches_individual_and_gradient_scale(self):
        initial = xi([1, -2], [0.5, -0.25])
        target = torch.tensor([0.2, 0.8])
        def objective(value):
            return (value.epsilon_zernike.flatten() + 2 * value.epsilon_tilt.flatten() - target).square()
        batched = solve_adam(objective, initial, learning_rate=0.01, iterations=20)
        # First derivative is 2*(z+2t-target), unaffected by batch size.
        expected_gradient = (2 * (initial.epsilon_zernike.flatten()
                                 + 2 * initial.epsilon_tilt.flatten() - target)).abs()
        torch.testing.assert_close(torch.tensor(batched.trace[0]["gradient_l2_zernike"]), expected_gradient)
        for b in range(2):
            individual = solve_adam(lambda value: (value.epsilon_zernike.flatten()
                + 2 * value.epsilon_tilt.flatten() - target[b]).square(),
                xi([initial.epsilon_zernike.flatten()[b]], [initial.epsilon_tilt.flatten()[b]]),
                learning_rate=0.01, iterations=20)
            for name in ("epsilon_zernike", "epsilon_tilt"):
                torch.testing.assert_close(getattr(batched.realization, name)[b:b + 1],
                                           getattr(individual.realization, name), rtol=0, atol=0)

    def test_best_includes_initialization_and_both_native_blocks(self):
        initial = xi([0.001], [0.001])
        result = solve_adam(lambda v: (v.epsilon_zernike.flatten() + v.epsilon_tilt.flatten()).square(),
                            initial, learning_rate=0.1, iterations=10)
        self.assertLessEqual(result.best_objective[0], result.initial_objective[0])
        self.assertLess(result.initial_objective[0], result.trace[0]["objective_after"][0])
        self.assertTrue(any(row["update_l2_zernike"][0] > 0 for row in result.trace))
        self.assertTrue(any(row["update_l2_tilt"][0] > 0 for row in result.trace))
        self.assertEqual(len(result.trace), 10)

    def test_whole_pair_selection_is_independent_per_sample_and_tie_inherits(self):
        network, previous = xi([1, 3, 2], [1, 1, 1]), xi([3, 1, 1], [1, 1, 2])
        def objective(v):
            return (v.epsilon_zernike.flatten() + v.epsilon_tilt.flatten()).square()
        selected, sources, _, _ = select_batched_initialization(network, previous, objective)
        self.assertEqual(sources, ["network", "previous_refined", "previous_refined"])
        torch.testing.assert_close(selected.epsilon_zernike.flatten(), torch.tensor([1., 1., 1.]))
        torch.testing.assert_close(selected.epsilon_tilt.flatten(), torch.tensor([1., 1., 2.]))

    def test_disconnected_block_fails_but_connected_zero_is_recorded(self):
        with self.assertRaises(RefinementFailure):
            solve_adam(lambda v: v.epsilon_zernike.flatten().square(), xi([1], [1]),
                       learning_rate=0.01, iterations=10)
        result = solve_adam(lambda v: (0 * v.epsilon_zernike.flatten()
                                      + 0 * v.epsilon_tilt.flatten()).square(),
                            xi([1], [1]), learning_rate=0.01, iterations=10)
        self.assertEqual(result.trace[0]["gradient_l2_zernike"], [0.0])
        self.assertEqual(result.trace[0]["update_l2_tilt"], [0.0])

    def test_full_rollout_no_final_refinement_and_weights_flags_unchanged(self):
        model, simulator = Network(), Simulator()
        model.weight.grad = torch.tensor(7.0)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        observation = torch.zeros(2, 3, 2, 2)
        for g in (4, 3, 2, 1):
            result = rollout_predictions(model=model, simulator=simulator, observation=observation,
                initial_G=g, mode="refined", solver_config={"learning_rate": 0.01, "iterations": 10})
            self.assertEqual([step.current_g for step in result], list(range(g, 0, -1)))
            self.assertIsNone(result[-1].refinement)
            self.assertEqual(sum(step.refinement is not None for step in result), g - 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertEqual(model.weight.grad.item(), 7)
        self.assertIsNone(simulator.weight.grad)

    def test_existing_single_sample_selection_api_accepts_numeric_adapter(self):
        model, simulator = Network(), Simulator()
        result = run_inference(model=model, simulator=simulator, observation=torch.zeros(1, 3, 2, 2),
                               initial_G=3, numeric_config=numeric_config(0.01, 10),
                               refiner=NativeAdamRefiner())
        self.assertEqual(result.network_predictions, 3)
        self.assertEqual(result.refinement_calls, 2)

    def test_original_observation_objective_has_per_image_mean_and_fixed_clean(self):
        simulator = Simulator().requires_grad_(False)
        clean = torch.ones(2, 3, 2, 2, requires_grad=True)
        observation = torch.zeros_like(clean, requires_grad=True)
        value = xi([1, 2], [1, 2])
        value.epsilon_zernike.requires_grad_(True)
        value.epsilon_tilt.requires_grad_(True)
        scores = per_sample_objective(simulator=simulator, current_clean=clean, realization=value,
                                      observation=observation, initial_G=4)
        torch.testing.assert_close(scores, torch.tensor([169., 625.]))
        scores.sum().backward()
        self.assertIsNone(clean.grad)
        self.assertIsNone(observation.grad)
        self.assertIsNotNone(value.epsilon_tilt.grad)


class MetricsAndRankingTests(unittest.TestCase):
    def test_checkpoint_requires_resolved_identity_and_formal_state(self):
        resolved = {"declared": {"run": "test"}, "actual_code_sha256": {"file": "hash"}}
        state = {"weight": torch.tensor(1.0)}
        payload = {"format": "v031_distributed_update_boundary_v1", "identity": resolved,
                   "formal_training": True, "smoke_optimizer_updates": 0,
                   "global_update": 100, "completed_updates": 100,
                   "network": state, "ema": {"network": state}}
        with patch("v031.evaluation._resolved_identity", return_value=resolved), \
                patch("torch.load", return_value=payload):
            loaded, update = _checkpoint_state("unused.pt", {"identity": resolved["declared"]}, "raw")
            self.assertIs(loaded, state)
            self.assertEqual(update, 100)
            for key, invalid in (("identity", resolved["declared"]), ("formal_training", False),
                                 ("fixture_mode", True), ("smoke_optimizer_updates", 1),
                                 ("completed_updates", 99)):
                altered = {**payload, key: invalid}
                with patch("torch.load", return_value=altered), self.assertRaises(ValueError):
                    _checkpoint_state("unused.pt", {"identity": resolved["declared"]}, "raw")

    def test_float_psnr_zero_and_explicit_ties(self):
        metrics = clean_metrics(torch.tensor([[[[0.0]]], [[[2.0]]]]), torch.zeros(2, 1, 1, 1))
        self.assertTrue(metrics[0]["zero_mse"])
        self.assertIsNone(metrics[0]["clean_psnr_db"])
        self.assertAlmostEqual(metrics[1]["clean_psnr_db"], -6.020599913)
        self.assertEqual(paired_gain(metrics[0], metrics[0], tie_psnr_db=1e-6)["outcome"], "tie")
        self.assertEqual(paired_gain(metrics[1], metrics[0], tie_psnr_db=1e-6)["outcome"], "win")
        a = {"clean_psnr_db": 30., "clean_mse": .001, "zero_mse": False}
        b = {"clean_psnr_db": 30.0000001, "clean_mse": .001, "zero_mse": False}
        self.assertEqual(paired_gain(a, b, tie_psnr_db=1e-6)["outcome"], "tie")

    def test_rank_uses_all_probes_prefix_budget_and_time_tie_then_steps_lr(self):
        probes = []
        for index in range(24):
            probe = {"probe_id": str(index)}
            for lr in (.001, .01, .1):
                probe[str(lr)] = {"network_unchanged": True, "prefixes": {str(k): {
                    "valid": True, "initial_objective": [1.], "best_objective": [.5],
                    "cumulative_seconds": (1 if k == 10 else 1.01 if k == 20 else 2)} for k in (10, 20, 40)}}
            probes.append(probe)
        ranking = rank_candidates(probes)
        self.assertEqual(ranking["ranking"][0]["iterations"], 10)
        self.assertEqual(ranking["ranking"][0]["learning_rate"], .001)
        probes[0]["0.001"]["prefixes"].pop("40")
        ranking = rank_candidates(probes)
        invalid = [x for x in ranking["candidates"] if not x["valid"]]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["iterations"], 40)
        self.assertEqual(invalid[0]["failed_probe_ids"], ["0"])
        with self.assertRaises(ValueError):
            rank_candidates(probes[:-1])

    def test_g1_summary_has_no_multistep_gain(self):
        metric = {"clean_mse": .01, "clean_psnr_db": 20., "zero_mse": False,
                  "network_redegradation_mse": .1}
        result = summarize_records([{"initial_G": 1, "predictions": [metric]}], tie_psnr_db=1e-6)
        row = result["paired_full_minus_one_shot"][0]
        self.assertFalse(row["multistep_applicable"])
        self.assertEqual((row["win"], row["tie"], row["loss"]), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()

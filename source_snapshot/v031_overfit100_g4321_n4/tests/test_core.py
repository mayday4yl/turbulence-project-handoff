"""CPU semantics tests; real-network/NPU acceptance is a separate artifact."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from v031.core import RolloutModule, ddp_loss_scale, full_image_mse, severity_path
from v031.data import StartDataset


@dataclass
class Xi:
    epsilon_zernike: torch.Tensor
    epsilon_tilt: torch.Tensor


@dataclass
class Prepared:
    image_mean: torch.Tensor
    dictionary_images: torch.Tensor


class ToyNetwork(nn.Module):
    """Nondegenerate deterministic fixture, not the production architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.clean_head = nn.Conv2d(3, 3, 1)
        self.xi_head = nn.Conv2d(3, 2, 1)
        with torch.no_grad():
            self.clean_head.weight.fill_(0.17)
            self.clean_head.bias.fill_(0.11)
            self.xi_head.weight.fill_(0.09)
            self.xi_head.bias.fill_(0.02)
        self.calls = []

    def forward(self, image, g):
        self.calls.append((image, g))
        # The convolutions exercise CPU BF16 autocast when enabled.
        clean = self.clean_head(image)
        xi = self.xi_head(image)
        return SimpleNamespace(
            clean=clean,
            realization=Xi(xi[:, :1], xi.permute(0, 2, 3, 1)))


class ToySimulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtypes = []

    def prepare_image(self, clean):
        return Prepared(clean, clean * 0.2)

    def render_prepared(self, prepared, xi, g):
        self.seen_dtypes.append((prepared.image_mean.dtype,
                                xi.epsilon_zernike.dtype, xi.epsilon_tilt.dtype))
        z = xi.epsilon_zernike.mean(dim=(1, 2, 3))
        t = xi.epsilon_tilt.mean(dim=(1, 2, 3))
        output = (prepared.image_mean + prepared.dictionary_images
                  + (g * (0.07 * z + 0.03 * t))[:, None, None, None])
        return SimpleNamespace(output=output)


def fixture_inputs():
    clean = torch.linspace(0.1, 0.8, 4 * 3 * 2 * 2).reshape(4, 3, 2, 2)
    observed = clean * 0.5 + torch.tensor([0.05, 0.12, 0.20, 0.29])[:, None, None, None]
    return clean, observed, torch.tensor([4., 3., 2., 1.])


class CoreTests(unittest.TestCase):
    def module(self, *, hybrid=False):
        return RolloutModule(ToyNetwork(), ToySimulator(), 0.2,
                             network_bfloat16=hybrid)

    def test_step_metrics_preserve_mixed_path_loss_and_gradient(self):
        clean, observation, gs = fixture_inputs()
        reference, logged = self.module(), self.module()
        before = reference(clean, observation, gs)
        after = logged(clean, observation, gs, collect_step_metrics=True)
        torch.testing.assert_close(after['path_losses'], before['path_losses'], rtol=0, atol=0)
        before['loss_mean'].backward()
        after['loss_mean'].backward()
        for (name_a, a), (name_b, b) in zip(reference.named_parameters(), logged.named_parameters()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertEqual(before['step_metrics'], [])
        self.assertEqual(after['trace'], [])
        reconstructed = torch.zeros(4)
        for row in after['step_metrics']:
            self.assertEqual(row['active_path_indices'], tuple(range(5-row['step'])))
            for key in ('current_g', 'clean_losses', 'reconstruction_losses'):
                self.assertFalse(row[key].requires_grad)
            for local, path in enumerate(row['active_path_indices']):
                reconstructed[path] += (row['clean_losses'][local] + .2*row['reconstruction_losses'][local]) / gs[path]
        torch.testing.assert_close(reconstructed, after['path_losses'].detach())

    def test_complete_schedule_authority_and_rejections(self):
        self.assertEqual([severity_path(g) for g in [4, 3, 2, 1]],
                         [(4, 3, 2, 1), (3, 2, 1), (2, 1), (1,)])
        for invalid in [0, 1.5, 4.5, float("nan")]:
            with self.assertRaises(ValueError):
                severity_path(invalid)
        clean, observation, gs = fixture_inputs()
        invalid_schedules = [
            [(4, 3, 2), (3, 2, 1), (2, 1), (1,)],
            [(4, 2, 1), (3, 2, 1), (2, 1), (1,)],
            [(4, 3, 2, 1), (3,), (2, 1), (1,)],
            [(4, 3, 2, 1), (3, 2, 1), (2, 1), (1, 0)],
        ]
        model = self.module()
        for schedules in invalid_schedules:
            with self.assertRaises(ValueError):
                model(clean, observation, gs, schedules)
        self.assertEqual(len(model.network.calls), 0)
        for coefficient in [0, 0.1, 1.0, float("nan")]:
            with self.assertRaises(ValueError):
                RolloutModule(ToyNetwork(), ToySimulator(), coefficient)

    def test_four_separate_start_traces_and_last_losses(self):
        clean, observation, gs = fixture_inputs()
        for index, g in enumerate([4, 3, 2, 1]):
            module = self.module()
            result = module(clean[index:index + 1], observation[index:index + 1],
                            gs[index:index + 1], retain_intermediates=True)
            self.assertEqual(result["prediction_count"], g)
            self.assertEqual(result["logical_render_count"], 2 * g - 1)
            self.assertEqual(result["training_refinement_count"], 0)
            self.assertEqual(len(result["trace"]), g)
            for k, trace in enumerate(result["trace"]):
                self.assertEqual(trace["current_g"].tolist(), [float(g - k)])
                self.assertEqual(trace["original_g"].tolist(), [float(g)])
                torch.testing.assert_close(trace["original_observation"], observation[index:index + 1])
                self.assertEqual(trace["clean_losses"].numel(), 1)
                self.assertEqual(trace["reconstruction_losses"].numel(), 1)
                if k:
                    self.assertIs(trace["input"], result["trace"][k - 1]["next_state"])
            self.assertIsNone(result["trace"][-1]["next_state"])

    def test_mixed_batch_path_mean_and_original_observation(self):
        clean, observation, gs = fixture_inputs()
        result = self.module()(clean, observation, gs, retain_intermediates=True)
        per_path_clean = [[] for _ in range(4)]
        per_path_reconstruction = [[] for _ in range(4)]
        total_predictions = []
        for step, trace in enumerate(result["trace"]):
            self.assertEqual(trace["active_path_indices"], tuple(range(4 - step)))
            for local, original in enumerate(trace["active_path_indices"]):
                # Compute independently from actual predictions and the initial
                # tensors; current_g/input must never replace original G/Y.
                predicted = trace["clean"][local:local + 1]
                z = trace["epsilon_zernike"][local:local + 1].mean()
                t = trace["epsilon_tilt"][local:local + 1].mean()
                reobserved = 1.2 * predicted + gs[original] * (0.07 * z + 0.03 * t)
                lc = (predicted - clean[original:original + 1]).square().mean()
                lr = (reobserved - observation[original:original + 1]).square().mean()
                per_path_clean[original].append(lc)
                per_path_reconstruction[original].append(lr)
                total_predictions.append(lc + 0.2 * lr)
        expected_c = torch.stack([torch.stack(path).mean() for path in per_path_clean])
        expected_r = torch.stack([torch.stack(path).mean() for path in per_path_reconstruction])
        expected_losses = expected_c + 0.2 * expected_r
        torch.testing.assert_close(result["path_clean"], expected_c)
        torch.testing.assert_close(result["path_reconstruction"], expected_r)
        torch.testing.assert_close(result["path_losses"], expected_losses)
        torch.testing.assert_close(result["loss_mean"], expected_losses.mean())
        torch.testing.assert_close(result["loss_sum"], expected_losses.sum())
        # This fixture discriminates the forbidden pooled-prediction denominator.
        self.assertGreater(abs((result["loss_mean"] - torch.stack(total_predictions).mean()).item()), 1e-4)
        self.assertEqual(result["prediction_count"], 10)
        self.assertEqual(result["logical_render_count"], 16)
        self.assertEqual(result["batched_render_call_count"], 7)

    def test_microbatch_gradient_accumulation_matches_mixed_batch(self):
        clean, observation, gs = fixture_inputs()
        full = self.module()
        separate = copy.deepcopy(full)
        full(clean, observation, gs)["loss_mean"].backward()
        # Unequal microbatch sizes must scale path SUM by effective batch size.
        for start, stop in [(0, 1), (1, 3), (3, 4)]:
            part = separate(clean[start:stop], observation[start:stop], gs[start:stop])
            (part["loss_sum"] / 4).backward()
        for full_p, separate_p in zip(full.network.parameters(), separate.network.parameters()):
            self.assertIsNotNone(full_p.grad)
            self.assertGreater(full_p.grad.norm().item(), 0)
            torch.testing.assert_close(full_p.grad, separate_p.grad, rtol=1e-5, atol=1e-7)
        self.assertEqual(ddp_loss_scale(32, 4), 0.125)

    def test_terminal_probe_to_both_native_branches(self):
        clean, observation, gs = fixture_inputs()
        result = self.module()(clean[:1], observation[:1], gs[:1], retain_intermediates=True)
        probe = full_image_mse(result["trace"][-1]["clean"], clean[:1]).sum()
        queried = [trace[key] for trace in result["trace"][:-1]
                   for key in ["clean", "epsilon_zernike", "epsilon_tilt"]]
        grads = torch.autograd.grad(probe, queried, allow_unused=True)
        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertGreater(grad.norm().item(), 0)

    def test_bfloat16_network_casts_all_native_outputs_back_to_fp32(self):
        clean, observation, gs = fixture_inputs()
        module = self.module(hybrid=True)
        result = module(clean[:1], observation[:1], gs[:1], retain_intermediates=True)
        for trace in result["trace"]:
            for key in ["input", "clean", "epsilon_zernike", "epsilon_tilt",
                        "clean_losses", "reconstruction_losses", "reconstructed"]:
                self.assertEqual(trace[key].dtype, torch.float32, key)
            if trace["next_state"] is not None:
                self.assertEqual(trace["next_state"].dtype, torch.float32)
                self.assertTrue(trace["next_state"].requires_grad)
        for key in ["path_clean", "path_reconstruction", "path_losses", "loss_sum", "loss_mean"]:
            self.assertEqual(result[key].dtype, torch.float32, key)
        self.assertTrue(all(dtypes == (torch.float32,) * 3 for dtypes in module.simulator.seen_dtypes))
        result["loss_mean"].backward()
        self.assertTrue(all(parameter.dtype == torch.float32 and parameter.grad is not None
                            for parameter in module.network.parameters()))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "clean").mkdir()
        (self.root / "states").mkdir()
        (self.root / "subsets").mkdir()
        self.source = {"source_id": "s", "clean_ref": "clean/c.npy"}
        self.trajectory = {"trajectory_id": "t", "source_id": "s", "clean_ref": "clean/c.npy",
                           **{f"x{g}_ref": f"states/X_{g}.npy" for g in [4, 3, 2, 1]},
                           "epsilon_zernike_ref": "xi/z.npy", "epsilon_tilt_ref": "xi/t.npy"}
        self.records = [{"start_id": f"t_G{g}", "source_id": "s", "trajectory_id": "t",
                         "clean_ref": "clean/c.npy", "initial_state_ref": f"states/X_{g}.npy",
                         "initial_G": g, "severity_path": list(severity_path(g)),
                         "num_predictions": g} for g in [4, 3, 2, 1]]
        self.write_jsonl("source_manifest.jsonl", [self.source])
        self.write_jsonl("trajectory_manifest.jsonl", [self.trajectory])
        self.write_jsonl("start_manifest.jsonl", self.records)
        np.save(self.root / "clean/c.npy", np.full((3, 256, 256), 0.2, np.float32))
        np.save(self.root / "states/X_4.npy", np.full((3, 256, 256), 0.3, np.float32))

    def tearDown(self):
        self.temp.cleanup()

    def write_jsonl(self, name, records):
        (self.root / name).write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_normal_access_opens_only_clean_and_one_observation(self):
        dataset = StartDataset(self.root)
        real_load = np.load
        with mock.patch("v031.data.np.load", wraps=real_load) as loader:
            item = dataset[0]
        self.assertEqual([call.args[0].relative_to(self.root).as_posix()
                          for call in loader.call_args_list], ["clean/c.npy", "states/X_4.npy"])
        self.assertEqual(set(item), {"clean", "observation", "G", "severity_path", "start_id", "trajectory_id"})
        self.assertEqual(item["severity_path"], (4, 3, 2, 1))
        self.assertEqual(item["clean"].dtype, torch.float32)

    def test_subset_is_exact_authoritative_selection(self):
        self.write_jsonl("subsets/selected.jsonl", [self.records[0]])
        dataset = StartDataset(self.root, "subsets/selected.jsonl")
        self.assertEqual(len(dataset), 1)
        changed = dict(self.records[0], num_predictions=1)
        self.write_jsonl("subsets/selected.jsonl", [changed])
        with self.assertRaises(ValueError):
            StartDataset(self.root, "subsets/selected.jsonl")

    def test_root_escape_absolute_path_and_symlink_rejected(self):
        dataset = StartDataset(self.root)
        for reference in ["../outside.npy", "/tmp/outside.npy"]:
            with self.assertRaises(ValueError):
                dataset._safe_path(reference)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            dataset._safe_path("escape/outside.npy")

    def test_no_implicit_fp32_conversion_of_authoritative_arrays(self):
        np.save(self.root / "states/X_4.npy", np.zeros((3, 256, 256), np.float16))
        with self.assertRaises(ValueError):
            StartDataset(self.root)[0]


if __name__ == "__main__":
    unittest.main()

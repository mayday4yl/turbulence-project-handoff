"""Bounded CPU checks; these are not NPU throughput or convergence evidence."""
from dataclasses import replace
from pathlib import Path
from contextlib import nullcontext
import json
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from training.test_semantics import TinyModel, TinySimulator, TinyDataset
from alignment_training.core import unroll_training, infer
from alignment_training.evaluation import evaluate_full_validation, evaluate_val_a
from alignment_training.runner import RunConfig, train
from alignment_training.state import UpdateBatchSampler, file_sha256


class CountModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, image, time):
        self.calls += 1
        return super().forward(image, time)


class StrengthSimulator(TinySimulator):
    def __init__(self):
        super().__init__()
        self.strengths = []
    def render_prepared(self, prepared, raw, strength):
        self.strengths.append(torch.as_tensor(strength).clone())
        return super().render_prepared(prepared, raw, strength)


class ValA(TinyDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        time = 0. if index % 4 == 0 else .1 + index / 10
        item.update(t=torch.tensor(time), strength=torch.tensor(4. * (1. - time)))
        item.pop("G")
        return item


class MixedSampler(UpdateBatchSampler):
    def indices_for(self, update, micro_index):
        indices = super().indices_for(update, micro_index)
        start = micro_index * self.world_size * self.microbatch + self.rank * self.microbatch
        return [(index, (start + offset) % 4 == 0) for offset, index in enumerate(indices)]
    def state_dict(self, completed_update):
        state = super().state_dict(completed_update)
        state["kind"] = "test_fixed_endpoint_policy"
        return state


class TrainA(torch.utils.data.Dataset):
    split = "train"
    def __len__(self):
        return 8000
    def __getitem__(self, key):
        index, endpoint = key
        target = torch.full((3, 4, 4), .3 + (index % 50) / 500)
        t = 0. if endpoint else .15 + (index % 7) / 10
        return {"clean": target, "observation": target + .1, "t": torch.tensor(t),
                "strength": torch.tensor(4 * (1 - t)), "sample_id": str(index)}
    def make_batch_sampler(self, **kwargs):
        return MixedSampler(len(self), **kwargs)


class CanonicalVal(TinyDataset):
    def __len__(self):
        return 1000


class TrainB(TinyDataset):
    split = "train"
    def __len__(self):
        return 8000
    def __getitem__(self, index):
        return super().__getitem__(index % 8)


class StochasticTinyModel(TinyModel):
    def forward(self, image, time):
        return super().forward(image + .001 * torch.rand_like(image), time)


def fake_full_validation(model, simulator, dataset, *, output_dir, update, **kwargs):
    # Deliberately consume all RNG families, as real LPIPS initialization can.
    # The production evaluator's RNG preservation is tested independently.
    summary = {"event": "full_validation", "update": update, "psnr": 20. + update / 1000,
               "elapsed_seconds": .01, "stage_psnr": [20.] * 4,
               "psnr_gain_vs_input": 2., "view": "original_Val1000_real_recursive_rollout"}
    path = Path(output_dir) / "validation_curve.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(summary) + "\n")
    return summary


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.x = torch.rand(4, 3, 4, 4) * .3 + .4
        self.y = self.x * .7
        self.t = torch.tensor([0., .2, .5, .8])
        self.g = 4 * (1 - self.t)

    def test_a_one_prediction_current_g_and_both_heads_receive_gradients(self):
        model, simulator = CountModel(), StrengthSimulator()
        result = unroll_training(model, simulator, self.x, self.y, method="A",
            objective="precomputed_current_g", path_seed=1, time=self.t, strength=self.g, retain_trace=True)
        self.assertEqual(model.calls, 1)
        self.assertEqual(simulator.render_calls, 1)
        self.assertTrue(torch.equal(simulator.strengths[0], self.g))
        trace = result.trace[0]
        expected = .9 * (trace["clean"] - self.y).abs().mean() + .1 * (trace["matched"] - self.x).abs().mean()
        self.assertTrue(torch.equal(result.loss, expected))
        result.loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_a_rejects_wrong_strength_or_t_endpoint(self):
        for times, strength in [(self.t, torch.ones(4) * 4),
                                (torch.ones(4), torch.zeros(4)),
                                (torch.full((4,), -.1), torch.full((4,), 4.4))]:
            with self.assertRaises(ValueError):
                unroll_training(TinyModel(), TinySimulator(), self.x, self.y,
                    method="A", objective="precomputed_current_g", path_seed=1, time=times, strength=strength)

    def test_b_retains_four_step_original_G4_and_cross_step_raw_gradients(self):
        model, simulator = CountModel(), StrengthSimulator()
        result = unroll_training(model, simulator, self.x, self.y, method="B",
            objective="recursive_alignment", path_seed=123, retain_trace=True)
        self.assertEqual(model.calls, 4)
        self.assertEqual(simulator.render_calls, 7)
        self.assertTrue(torch.all(result.paths.t[:, 1:] > result.paths.t[:, :-1]))
        for index in (0, 2, 4, 6):
            self.assertTrue(torch.equal(simulator.strengths[index], torch.tensor(4.)))
        first = result.trace[0]
        gradients = torch.autograd.grad(result.trace[-1]["loss"], (first["clean"],
            first["raw"].epsilon_zernike, first["raw"].epsilon_tilt), retain_graph=True)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        result.loss.backward()

    def test_a_microbatch_accumulation_equals_global_batch(self):
        full, micro = TinyModel(), TinyModel()
        result = unroll_training(full, TinySimulator(), self.x, self.y, method="A",
            objective="precomputed_current_g", path_seed=1, time=self.t, strength=self.g)
        result.loss.backward()
        for index in range(4):
            result = unroll_training(micro, TinySimulator(), self.x[index:index+1], self.y[index:index+1],
                method="A", objective="precomputed_current_g", path_seed=9,
                time=self.t[index:index+1], strength=self.g[index:index+1])
            (result.loss / 4).backward()
        for a, b in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-8)

    def test_two_validation_views_use_correct_paths_and_preserve_rng(self):
        model = CountModel()
        with tempfile.TemporaryDirectory() as folder:
            before = torch.get_rng_state().clone()
            np_before = np.random.get_state()
            py_before = random.getstate()
            full = evaluate_full_validation(model, TinySimulator(), TinyDataset(), device="cpu",
                method="A", output_dir=folder, update=1, expected_size=8, batch_size=2,
                extended_metrics=False, objective="precomputed_current_g")
            self.assertEqual(model.calls, 16)
            current = evaluate_val_a(model, TinySimulator(), ValA(), device="cpu", output_dir=folder,
                update=1, expected_size=8, batch_size=2, extended_metrics=False)
            self.assertEqual(model.calls, 20)
            self.assertTrue(full["used_for_best_selection"])
            self.assertFalse(current["used_for_best_selection"])
            self.assertEqual(current["endpoint_count"], 2)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(py_before, random.getstate())
            self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))
            self.assertTrue(Path(folder, "validation_curve.jsonl").is_file())
            self.assertTrue(Path(folder, "val_a_curve.jsonl").is_file())

    def test_stage_specific_horizons_and_hyperparameters(self):
        common = dict(output_dir="x", initial_checkpoint="i", readiness_report="r", parent_training_summary="p")
        a = RunConfig(method="A", **common)
        a.validate()
        b = RunConfig(method="B", objective="recursive_alignment", max_updates=4000,
            warmup_updates=100, backbone_lr=2.5e-5, new_lr=5e-5, **common)
        b.validate()
        for config in (replace(a, max_updates=10000), replace(b, objective="precomputed_current_g"),
                       replace(b, warmup_updates=300), replace(a, method="R0")):
            with self.assertRaises(ValueError):
                config.validate()

    def test_complete_a_resume_matches_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            initial = folder / "initial.pt"
            torch.save({"model": StochasticTinyModel().state_dict(), "method": "D0",
                        "completed_update": 7250, "best_update": 7250}, initial)
            parent = folder / "parent.json"
            parent.write_text(json.dumps({"method": "D0", "status": "TRAINING_COMPLETE_TEST",
                "completed_update": 10000, "best_update": 7250,
                "counters": {"optimizer_updates": 10000, "run_elapsed_seconds": 100.}}))
            common = dict(method="A", initial_checkpoint=str(initial), readiness_report=str(folder / "ready.json"),
                parent_training_summary=str(parent), microbatch=1, accumulation_steps=4, world_size=1,
                effective_batch=4, network_precision="fp32", activation_checkpointing=False,
                checkpoint_policy="all", talking_backend="conv3d", psf_backend="native", workers=0,
                extended_validation_metrics=False)
            ready = {"status": "PASS", "method": "A", "objective": "precomputed_current_g",
                "initial_checkpoint_sha256": file_sha256(initial), "runtime": {key: common[key] for key in
                    ("microbatch", "world_size", "network_precision", "activation_checkpointing", "talking_backend", "checkpoint_policy")}}
            (folder / "ready.json").write_text(json.dumps(ready))
            components = lambda: dict(model=StochasticTinyModel(), simulator=TinySimulator(),
                train_dataset=TrainA(), val_dataset=CanonicalVal(), val_a_dataset=CanonicalVal(), device="cpu")
            with mock.patch("alignment_training.runner.evaluate_full_validation", side_effect=fake_full_validation), \
                 mock.patch("alignment_training.runner.evaluate_val_a", return_value={"event": "val_a", "elapsed_seconds": .01, "psnr": 999.}):
                whole = RunConfig(output_dir=str(folder / "whole"), stop_after_update=3, **common)
                train(whole, **components())
                partial = RunConfig(output_dir=str(folder / "resume"), stop_after_update=1, **common)
                train(partial, **components())
                resume = replace(partial, stop_after_update=3, resume_checkpoint=str(folder / "resume/checkpoints/latest.pt"))
                train(resume, **components())
            a = torch.load(folder / "whole/checkpoints/latest.pt", weights_only=False)
            b = torch.load(folder / "resume/checkpoints/latest.pt", weights_only=False)
            self.assertEqual(a["schema"], "v5_alignment_training_checkpoint_v1")
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["sampler"], b["sampler"])
            self.assertEqual(b["scheduler"]["max_updates"], 6000)
            self.assertEqual(b["completed_update"], 3)
            self.assertLess(b["best_psnr"], 100.)  # Val-A 999 never selects best.
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for key in a["optimizer"]["state"]:
                for name in a["optimizer"]["state"][key]:
                    self.assertTrue(torch.equal(a["optimizer"]["state"][key][name], b["optimizer"]["state"][key][name]))
            self.assertEqual(b["counters"]["sample_exposures"], 12)
            self.assertEqual(b["counters"]["network_forward_exposures"], 12)

    def test_real_a_sampler_update_and_cache_handoff_boundaries(self):
        from alignment_data.dataset import AlignmentBatchSampler
        first = AlignmentBatchSampler(8000, start_update=1, max_updates=2000, rank=0)
        resumed = AlignmentBatchSampler(8000, start_update=2001, max_updates=4000, rank=0)
        self.assertEqual(next(iter(first)), first.indices_for(1, 0))
        self.assertEqual(len(first), 8000)
        self.assertEqual(first.state_dict(2000), resumed.state_dict(2000))
        other_rank = AlignmentBatchSampler(8000, start_update=2001, max_updates=4000, rank=5)
        self.assertEqual(other_rank.state_dict(2000), first.state_dict(2000))
        self.assertEqual(next(iter(resumed)), resumed.indices_for(2001, 0))
        before = first.global_indices_for(2000)
        after = resumed.global_indices_for(2001)
        self.assertEqual(sum(row[0] == "endpoint" for row in after), 12)
        self.assertTrue(all(row[1] == 0 for row in before if row[0] == "internal"))
        self.assertTrue(all(row[1] == 1 for row in after if row[0] == "internal"))

    def test_b_full_state_inheritance_and_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            initial_model = TinyModel()
            with torch.no_grad():
                initial_model.raw_scale.fill_(.48)
                initial_model.time_scale.fill_(.19)
            initial = folder / "a_best.pt"
            torch.save({"model": initial_model.state_dict(), "method": "A", "completed_update": 5750,
                "best_update": 5750, "optimizer": {"invalid_parent_optimizer": True}}, initial)
            parent = folder / "parent.json"
            parent.write_text(json.dumps({"method": "A", "status": "TRAINING_COMPLETE_STAGE_HANDOFF_READY",
                "completed_update": 6000, "best_update": 5750,
                "counters": {"optimizer_updates": 6000, "run_elapsed_seconds": 200.,
                    "cumulative_optimizer_updates_including_parent": 16000,
                    "cumulative_run_elapsed_seconds_including_parent": 300.}}))
            config = RunConfig(method="B", objective="recursive_alignment", max_updates=4000,
                warmup_updates=100, backbone_lr=2.5e-5, new_lr=5e-5, output_dir=str(folder / "B"),
                initial_checkpoint=str(initial), readiness_report=str(folder / "ready.json"),
                parent_training_summary=str(parent), stop_after_update=1,
                microbatch=1, accumulation_steps=2, world_size=1, effective_batch=2,
                network_precision="fp32", activation_checkpointing=False, checkpoint_policy="all",
                talking_backend="conv3d", psf_backend="native", workers=0, extended_validation_metrics=False)
            (folder / "ready.json").write_text(json.dumps({"status": "PASS", "method": "B",
                "objective": "recursive_alignment", "initial_checkpoint_sha256": file_sha256(initial),
                "runtime": {key: getattr(config, key) for key in ("microbatch", "world_size", "network_precision",
                    "activation_checkpointing", "talking_backend", "checkpoint_policy")}}))
            with mock.patch("alignment_training.runner.evaluate_full_validation", side_effect=fake_full_validation):
                train(config, model=TinyModel(), simulator=TinySimulator(), train_dataset=TrainB(),
                      val_dataset=CanonicalVal(), device="cpu")
            start = torch.load(folder / "B/checkpoints/rolling_update_000000.pt", weights_only=False)
            after = torch.load(folder / "B/checkpoints/latest.pt", weights_only=False)
            self.assertEqual(start["optimizer"]["state"], {})
            self.assertEqual(start["scheduler"]["last_update"], 0)
            self.assertIsNone(start["best_update"])
            for key, value in initial_model.state_dict().items():
                self.assertTrue(torch.equal(start["model"][key], value))
            self.assertFalse(after["initial_identity"]["time_and_raw_reinitialized"])
            self.assertEqual(after["scheduler"]["max_updates"], 4000)
            self.assertEqual(after["scheduler"]["last_update"], 1)
            self.assertEqual(after["counters"]["cumulative_optimizer_updates_including_parent"], 16001)
            self.assertTrue(all(state["step"].item() == 1 for state in after["optimizer"]["state"].values()))


if __name__ == "__main__":
    unittest.main()

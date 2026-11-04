"""CPU semantic tests for the bounded engine; no NPU claim is made here."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from test_core import ToyNetwork, ToySimulator, fixture_inputs
from v031.core import RolloutModule, severity_path
from v031.data import collate_starts
from v031.scheduler import UpdateIndexedScheduler
from v031.training import (FP32EMA, GlobalStartSampler, TrainingEngine, capture_rank_rng,
                           compare_states, load_training_checkpoint, parameter_groups,
                           save_training_checkpoint)


class TinyStarts:
    def __init__(self, count=64):
        self.records = [{"start_id": f"start-{i}", "initial_G": 4 - i % 4}
                        for i in range(count)]
        self.clean, self.observation, self.gs = fixture_inputs()

    def __getitem__(self, index):
        g = self.records[index]["initial_G"]
        return {"clean": self.clean[index % 4], "observation": self.observation[index % 4],
                "G": torch.tensor(float(g)), "severity_path": severity_path(g),
                "start_id": f"start-{index}", "trajectory_id": f"trajectory-{index // 4}"}


def engine(*, microbatch=4, identity=None):
    return TrainingEngine(ToyNetwork(), ToySimulator(), TinyStarts(), device="cpu",
                          microbatch=microbatch, identity=identity or {"fixture": "training-test"})


def snapshot(value):
    return deepcopy({"network": value.network.state_dict(), "optimizer": value.optimizer.state_dict(),
                     "ema": value.ema.state_dict(), "scheduler": value.scheduler.state_dict(),
                     "sampler": value.sampler.state_dict(), "counts": value.counts,
                     "rng": capture_rank_rng(value.device)})


class SamplerTests(unittest.TestCase):
    def test_full_epoch_no_drops_duplicates_and_dedicated_rng(self):
        before = torch.get_rng_state().clone()
        sampler = GlobalStartSampler([str(i) for i in range(1600)])
        epoch = [index for _ in range(50) for index in sampler.next_indices()]
        self.assertEqual(sorted(epoch), list(range(1600)))
        self.assertEqual(sampler.consumed, 1600)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        restored = GlobalStartSampler([str(i) for i in range(1600)])
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(sampler.next_indices(), restored.next_indices())
        self.assertEqual(sampler.epoch, 1)

    def test_mid_epoch_restore_and_invalid_state(self):
        original = GlobalStartSampler([str(i) for i in range(1600)])
        for _ in range(17):
            original.next_indices()
        state = original.state_dict()
        other = GlobalStartSampler([str(i) for i in range(1600)])
        other.load_state_dict(state)
        for _ in range(85):
            self.assertEqual(original.next_indices(), other.next_indices())
        bad = deepcopy(state)
        bad["order"][0] = bad["order"][1]
        with self.assertRaises(ValueError):
            other.load_state_dict(bad)
        with self.assertRaises(ValueError):
            GlobalStartSampler(["different"]).load_state_dict(state)


class GroupAndEMATests(unittest.TestCase):
    def test_all_heads_bias_and_norm_classification(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.GroupNorm(2, 4), nn.Conv2d(4, 3, 1, bias=False))
        groups, report = parameter_groups(model)
        self.assertEqual(report["decay"]["names"], ["0.weight", "2.weight"])
        self.assertEqual(report["no_decay"]["names"], ["0.bias", "1.bias", "1.weight"])
        self.assertEqual(report["total_trainable_parameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual({id(p) for group in groups for p in group["params"]}, {id(p) for p in model.parameters()})

    def test_ema_formula_fp32_and_copy_all_buffers(self):
        model = nn.Linear(2, 1)
        model.register_buffer("floating_buffer", torch.tensor([1.0]))
        model.register_buffer("integer_buffer", torch.tensor([1], dtype=torch.int64))
        ema = FP32EMA(model)
        old = deepcopy(ema.state_dict())
        with torch.no_grad():
            model.weight.add_(3)
            model.bias.add_(2)
            model.floating_buffer.add_(7)
            model.integer_buffer.add_(4)
        ema.update(model)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(ema.network[name], old["network"][name] * .996 + parameter * .004)
            self.assertEqual(ema.network[name].dtype, torch.float32)
        self.assertTrue(torch.equal(ema.network["floating_buffer"], model.floating_buffer))
        self.assertTrue(torch.equal(ema.network["integer_buffer"], model.integer_buffer))
        restored = FP32EMA(model)
        restored.load_state_dict(ema.state_dict())
        self.assertTrue(compare_states(ema.state_dict(), restored.state_dict())["exact"])


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_accumulated_update_matches_independent_full_global_mean(self):
        trained = engine(microbatch=2)
        reference = ToyNetwork()
        simulator = ToySimulator()
        sampler = GlobalStartSampler([r["start_id"] for r in trained.dataset.records])
        indices = sampler.next_indices()
        batch = collate_starts([trained.dataset[i] for i in indices])
        groups, _ = parameter_groups(reference)
        optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(.9, .999), eps=1e-8,
                                      weight_decay=1e-4, foreach=False)
        scheduler = UpdateIndexedScheduler(optimizer)
        result = RolloutModule(reference, simulator, .2)(batch["clean"], batch["observation"], batch["G"])
        result["loss_mean"].backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        scheduler.mark_update_completed()
        logged = trained.update()
        self.assertAlmostEqual(logged["loss"], result["loss_mean"].item(), places=6)
        self.assertAlmostEqual(logged["preclip_global_norm"], norm.item(), places=5)
        for first, second in zip(trained.network.parameters(), reference.parameters()):
            torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
        self.assertEqual(logged["global_update"], 0)
        self.assertEqual(logged["smoke_optimizer_updates"], 1)
        self.assertEqual(logged["cumulative_counts"]["paths"], 32)
        self.assertEqual(logged["cumulative_counts"]["true_inputs"], 32)
        self.assertEqual(logged["cumulative_counts"]["by_current_g"]["1"]["predictions"], 32)
        self.assertEqual([r["start_id"] for r in logged["path_weights"]], logged["global_start_ids"])
        for row in logged["per_step"]:
            self.assertEqual(row["per_sample_LC_weight"], 1.0 / (32 * row["initial_G"]))
            self.assertEqual(row["current_g"], row["initial_G"] - row["K"] + 1)

    def test_checkpoint_exact_optimizer_ema_sampler_rng_and_next_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = engine()
            original.update()
            checkpoint = Path(temporary) / "boundary.pt"
            rng_before_save = capture_rank_rng(original.device)
            boundary_reference = snapshot(original)
            saved = save_training_checkpoint(original, checkpoint, run_until_update=2)
            self.assertEqual(saved["sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertTrue(compare_states(rng_before_save, capture_rank_rng(original.device))["exact"])
            random_reference = (random.random(), torch.rand(3))
            next_reference = original.update()
            reference_state = snapshot(original)
            restored = engine()
            loaded = load_training_checkpoint(restored, checkpoint, run_until_update=2)
            self.assertEqual(loaded["completed_updates"], 1)
            self.assertEqual(restored.optimizer_step_attempts, 0)
            self.assertTrue(compare_states(boundary_reference, snapshot(restored))["exact"])
            random_restored = (random.random(), torch.rand(3))
            self.assertEqual(random_reference[0], random_restored[0])
            self.assertTrue(torch.equal(random_reference[1], random_restored[1]))
            next_restored = restored.update()
            self.assertEqual(next_reference["global_start_ids"], next_restored["global_start_ids"])
            self.assertEqual(next_reference["loss"], next_restored["loss"])
            self.assertTrue(compare_states(reference_state, snapshot(restored))["exact"])
            with self.assertRaises(FileExistsError):
                save_training_checkpoint(restored, checkpoint, run_until_update=2)

    def test_four_rank_sum_scaling_equals_one_global_mean(self):
        dataset = TinyStarts()
        indices = GlobalStartSampler([r["start_id"] for r in dataset.records]).next_indices()
        reference = RolloutModule(ToyNetwork(), ToySimulator(), .2)
        global_batch = collate_starts([dataset[i] for i in indices])
        reference(global_batch["clean"], global_batch["observation"], global_batch["G"])["loss_mean"].backward()
        rank_gradients = []
        for rank in range(4):
            replica = RolloutModule(ToyNetwork(), ToySimulator(), .2)
            local = indices[rank * 8:(rank + 1) * 8]
            for offset in range(0, 8, 2):
                batch = collate_starts([dataset[i] for i in local[offset:offset + 2]])
                (replica(batch["clean"], batch["observation"], batch["G"])["loss_sum"] * (4 / 32)).backward()
            rank_gradients.append([p.grad.clone() for p in replica.network.parameters()])
        for index, parameter in enumerate(reference.network.parameters()):
            averaged = torch.stack([grads[index] for grads in rank_gradients]).mean(0)
            torch.testing.assert_close(averaged, parameter.grad, rtol=1e-5, atol=1e-7)

    def test_resume_identity_boundary_and_smoke_formal_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = engine()
            original.update()
            checkpoint = Path(temporary) / "boundary.pt"
            save_training_checkpoint(original, checkpoint, run_until_update=1)
            with self.assertRaises(ValueError):
                load_training_checkpoint(engine(identity={"fixture": "wrong"}), checkpoint, run_until_update=1)
            with self.assertRaises(ValueError):
                load_training_checkpoint(engine(), checkpoint, run_until_update=2)
            resumed = engine()
            load_training_checkpoint(resumed, checkpoint, run_until_update=2, authorize_boundary_extension=True)
            self.assertEqual(resumed.completed_updates, 1)
            payload = torch.load(checkpoint, weights_only=True)
            self.assertFalse(payload["formal_training"])
            self.assertEqual(payload["global_update"], 0)
            self.assertEqual(payload["smoke_optimizer_updates"], 1)
            self.assertEqual(len(payload["rng_by_rank"]), 1)
            self.assertIsNone(payload["rng_by_rank"][0]["torch_npu"])

    def test_initialized_boundary_zero_and_explicit_future_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            initial = engine()
            checkpoint = Path(temporary) / "initial.pt"
            save_training_checkpoint(initial, checkpoint, run_until_update=0)
            self.assertEqual(initial.optimizer_step_attempts, 0)
            self.assertFalse(initial.optimizer.state)
            with self.assertRaises(ValueError):
                load_training_checkpoint(engine(), checkpoint, run_until_update=100)
            resumed = engine()
            load_training_checkpoint(resumed, checkpoint, run_until_update=100,
                                     authorize_boundary_extension=True)
            first = initial.update()
            replay = resumed.update()
            self.assertEqual(first["global_start_ids"], replay["global_start_ids"])
            self.assertEqual(first["loss"], replay["loss"])
            self.assertAlmostEqual(first["lr_used"], 1e-4 / 50, places=18)
            self.assertTrue(compare_states(initial.network.state_dict(), resumed.network.state_dict())["exact"])
            with self.assertRaises(ValueError):
                load_training_checkpoint(engine(), checkpoint, run_until_update=1500,
                                         authorize_boundary_extension=True)
            future = {"explicit_user_authorization": True, "authorized_run_until_update": 1500,
                      "authorization_id": "TEST_FIXTURE_EXPLICIT_FUTURE_AUTHORIZATION"}
            with self.assertRaises(ValueError):
                load_training_checkpoint(engine(), checkpoint, run_until_update=1500, authorization=future)
            for invalid in [dict(future, explicit_user_authorization=False),
                            dict(future, authorization_id=""),
                            dict(future, authorized_run_until_update=1200)]:
                with self.assertRaises(ValueError):
                    load_training_checkpoint(engine(), checkpoint, run_until_update=1500,
                                             authorize_boundary_extension=True, authorization=invalid)
            future_engine = engine()
            load_training_checkpoint(future_engine, checkpoint, run_until_update=1500,
                                     authorize_boundary_extension=True, authorization=future)
            self.assertEqual(future_engine.completed_updates, 0)
            self.assertAlmostEqual(future_engine.scheduler.next_lr, 1e-4 / 50, places=18)

    def test_partially_failed_optimizer_attempt_counts_against_smoke_budget(self):
        failed = engine()
        def partial_failure():
            with torch.no_grad():
                next(failed.network.parameters()).add_(1)
            raise RuntimeError("simulated OOM after a partial optimizer mutation")
        with mock.patch.object(failed.optimizer, "step", side_effect=partial_failure):
            with self.assertRaisesRegex(RuntimeError, "partial optimizer"):
                failed.update()
        self.assertEqual(failed.optimizer_step_attempts, 1)
        self.assertEqual(failed.optimizer_step_calls, 0)
        self.assertEqual(failed.completed_updates, 0)
        self.assertTrue(failed.poisoned)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                save_training_checkpoint(failed, Path(temporary) / "invalid.pt", run_until_update=1)

    def test_nonfinite_is_fatal_no_step_or_checkpoint(self):
        failed = engine()
        failed.dataset.observation[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            failed.update()
        self.assertTrue(failed.poisoned)
        self.assertEqual(failed.optimizer_step_calls, 0)
        with self.assertRaises(RuntimeError):
            failed.update()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                save_training_checkpoint(failed, Path(temporary) / "invalid.pt", run_until_update=1)

    def test_batchnorm_semantics_change_rejected(self):
        model = ToyNetwork()
        model.extra_batchnorm = nn.BatchNorm2d(3)
        with self.assertRaises(ValueError):
            TrainingEngine(model, ToySimulator(), TinyStarts(), device="cpu", microbatch=2,
                           identity={"fixture": "batchnorm"})


if __name__ == "__main__":
    unittest.main()

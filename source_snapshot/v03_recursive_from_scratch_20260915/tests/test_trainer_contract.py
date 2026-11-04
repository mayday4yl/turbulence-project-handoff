"""Small CPU mathematics checks, not evidence of NPU simulator compatibility.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_trainer_contract.py
The separate full-resolution NPU acceptance must use the original simulator.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from v03.core import RolloutModule, ddp_loss_scale, rank_path_counts
from v03.train import (EMA, UpdateBatchSampler, UpdateScheduler, capture_rng,
                       cpu_tree, repair_metrics_prefix, restore_rng, state_digest)


@dataclass(frozen=True)
class Realization:
    epsilon_zernike: torch.Tensor
    epsilon_tilt: torch.Tensor


@dataclass(frozen=True)
class Prepared:
    image_mean: torch.Tensor
    dictionary_images: torch.Tensor


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.clean_scale = nn.Parameter(torch.tensor(.6))
        self.zernike_scale = nn.Parameter(torch.tensor(.12))
        self.tilt_scale = nn.Parameter(torch.tensor(.08))

    def forward(self, image, severity):
        clean = torch.sigmoid(self.clean_scale * image + severity[:, None, None, None] * .05)
        realization = Realization(image * self.zernike_scale, image * self.tilt_scale)
        return SimpleNamespace(clean=clean, realization=realization)


class ToySimulator(nn.Module):
    def prepare_image(self, image):
        return Prepared(image, image.square())

    def render_prepared(self, prepared, realization, severity):
        strength = severity[:, None, None, None]
        output = (.8 * prepared.image_mean + .01 * prepared.dictionary_images
                  + strength * (.04 * realization.epsilon_zernike
                                + .02 * realization.epsilon_tilt))
        return SimpleNamespace(output=output)


class TrainerContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.clean = torch.rand(5, 3, 4, 4)
        self.observation = torch.rand_like(self.clean)
        self.paths = [(4.5, 3., 1.5, .75), (.75,), (1.5, .75), (3., .75), (2.25,)]
        self.severities = torch.tensor([path[0] for path in self.paths])

    def module(self):
        return RolloutModule(ToyNetwork(), ToySimulator(), .1, network_bfloat16=False)

    def test_mixed_path_loss_matches_independent_path_means(self):
        module = self.module()
        batch = module(self.clean, self.observation, self.severities, self.paths)
        losses = []
        for index, path in enumerate(self.paths):
            item = module(self.clean[index:index+1], self.observation[index:index+1],
                          self.severities[index:index+1], [path])
            losses.append(item["loss_sum"])
        torch.testing.assert_close(batch["loss_sum"], torch.stack(losses).sum())
        self.assertEqual(int(batch["prediction_count"]), sum(map(len, self.paths)))

    def test_final_loss_reaches_early_clean_xi_and_direct_next_input(self):
        module = self.module()
        result = module(self.clean[:1], self.observation[:1], self.severities[:1],
                        [self.paths[0]], retain_intermediates=True)
        trace = result["trace"]
        trace[-1]["clean_losses"].sum().backward()
        for step in range(len(trace)-1):
            self.assertIs(trace[step]["next_state"], trace[step+1]["input"])
            for name in ("clean", "epsilon_zernike", "epsilon_tilt", "next_state"):
                gradient = trace[step][name].grad
                self.assertIsNotNone(gradient, (step, name))
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().max()), 0.)

    def test_transition_zero_gradient_hook_control_blocks_early_predictions(self):
        module = self.module()
        result = module(self.clean[:1], self.observation[:1], self.severities[:1],
                        [self.paths[0]], retain_intermediates=True)
        trace = result["trace"]
        handles = [entry["next_state"].register_hook(torch.zeros_like)
                   for entry in trace[:-1]]
        trace[-1]["clean_losses"].sum().backward()
        for entry in trace[:-1]:
            for name in ("clean", "epsilon_zernike", "epsilon_tilt"):
                gradient = entry[name].grad
                self.assertTrue(gradient is None or bool((gradient == 0).all()))
        for handle in handles:
            handle.remove()

    def test_normalization_survives_uneven_ranks_accumulation_and_dummy_tail(self):
        reference = self.module()
        result = reference(self.clean, self.observation, self.severities, self.paths)
        (result["loss_sum"] / 5).backward()
        expected = [p.grad.clone() for p in reference.network.parameters()]
        rank_gradients = []
        seen_ids = []
        for rank in range(4):
            module = self.module()
            sampler = UpdateBatchSampler(0, 1, 5, 1, rank, 4)
            chunks = sampler.chunks(0)
            self.assertEqual(len(chunks), 2)
            for chunk in chunks:
                ids = [sample_id for sample_id, _, _ in chunk]
                weight = torch.tensor([float(valid) for _, _, valid in chunk])
                seen_ids += [sample_id for sample_id, _, valid in chunk if valid]
                output = module(self.clean[ids], self.observation[ids], self.severities[ids],
                                [self.paths[index] for index in ids], weight)
                (output["loss_sum"] * ddp_loss_scale(5, 4)).backward()
            rank_gradients.append([p.grad.clone() for p in module.network.parameters()])
        self.assertEqual(sorted(seen_ids), list(range(5)))
        for index, target in enumerate(expected):
            simulated_ddp = torch.stack([gradients[index] for gradients in rank_gradients]).mean(0)
            torch.testing.assert_close(simulated_ddp, target, atol=1e-7, rtol=1e-6)

    def test_balanced_and_budgeted_assignment_preserves_loss_gradient(self):
        reference = self.module()
        result = reference(self.clean, self.observation, self.severities, self.paths)
        (result["loss_sum"] / 5).backward()
        expected = [p.grad.clone() for p in reference.network.parameters()]
        config = {"training": {"balance_paths_by_length": True,
                               "microbatch_prediction_budget": 4}}
        selector = lambda sample_id, update: {"schedule": self.paths[sample_id]}
        rank_gradients = []
        for rank in range(4):
            module = self.module()
            sampler = UpdateBatchSampler(0, 1, 5, 3, rank, 4,
                                         config=config, path_selector=selector)
            for chunk in sampler:
                ids = [sample_id for sample_id, _, _ in chunk]
                weights = torch.tensor([float(valid) for _, _, valid in chunk])
                result = module(self.clean[ids], self.observation[ids], self.severities[ids],
                                [self.paths[index] for index in ids], weights)
                (result["loss_sum"] * ddp_loss_scale(5, 4)).backward()
            rank_gradients.append([p.grad.clone() for p in module.network.parameters()])
        for index, target in enumerate(expected):
            simulated_ddp = torch.stack([gradients[index] for gradients in rank_gradients]).mean(0)
            torch.testing.assert_close(simulated_ddp, target, atol=1e-7, rtol=1e-6)

    def test_balanced_layout_has_same_global_ids_legal_budgets_and_balanced_load(self):
        lengths = (6, 1, 2, 1, 4, 2, 1, 3, 1, 6, 2, 1, 5, 1, 4, 1,
                   6, 1, 2, 1, 4, 2, 1, 3, 1, 6, 2, 1, 5, 1, 4, 1)
        selector = lambda sample_id, update: {"schedule": tuple(range(lengths[sample_id % 32]))}
        config = {"training": {"balance_paths_by_length": True,
                               "microbatch_prediction_budget": 8}}
        sampler = UpdateBatchSampler(0, 2, 32, 4, 0, 4,
                                     config=config, path_selector=selector)
        plans = sampler.all_rank_chunks(0)
        actual = [sample_id for rank in plans for chunk in rank
                  for sample_id, _, valid in chunk if valid]
        self.assertEqual(sorted(actual), list(range(32)))
        self.assertEqual(len(set(map(len, plans))), 1)
        loads = []
        for rank in plans:
            load = 0
            for chunk in rank:
                self.assertLessEqual(len(chunk), 4)
                prediction_count = sum(lengths[sample_id] for sample_id, _, valid in chunk if valid)
                self.assertLessEqual(prediction_count, 8)
                load += prediction_count
            loads.append(load)
        self.assertLessEqual(max(loads) - min(loads), max(lengths))
        resumed = UpdateBatchSampler(1, 2, 32, 4, 0, 4,
                                     config=config, path_selector=selector)
        self.assertEqual(resumed.all_rank_chunks(1), sampler.all_rank_chunks(1))

    def test_budget_only_preserves_original_rank_assignment_and_tail_ids(self):
        selector = lambda sample_id, update: {"schedule": (3., 2.25, 1.5, .75)}
        config = {"training": {"microbatch_prediction_budget": 4}}
        sampler = UpdateBatchSampler(0, 2, 9, 4, 0, 4, maximum_paths=14,
                                     config=config, path_selector=selector)
        for update in range(2):
            planned = sampler.all_rank_chunks(update)
            baseline = UpdateBatchSampler(0, 2, 9, 4, 0, 4, maximum_paths=14)
            for rank in range(4):
                actual_ids = [sample_id for chunk in planned[rank]
                              for sample_id, _, valid in chunk if valid]
                original_ids = [sample_id for chunk in baseline.all_rank_chunks(update)[rank]
                                for sample_id, _, valid in chunk if valid]
                self.assertEqual(actual_ids, original_ids)
                self.assertTrue(all(sum(valid for _, _, valid in chunk) <= 1
                                    for chunk in planned[rank]))

    def test_prediction_budget_never_splits_or_silently_truncates_a_path(self):
        config = {"training": {"balance_paths_by_length": True,
                               "microbatch_prediction_budget": 3}}
        sampler = UpdateBatchSampler(0, 1, 4, 4, 0, 4, config=config,
                                     path_selector=lambda *_: {"schedule": (4.5, 3., 1.5, .75)})
        with self.assertRaisesRegex(ValueError, "intact path exceeds"):
            sampler.chunks(0)

    def test_tail_sample_ids_and_weights_are_resume_stable(self):
        for rank in range(4):
            whole = UpdateBatchSampler(0, 3, 5, 2, rank, 4, maximum_paths=12)
            resumed = UpdateBatchSampler(2, 3, 5, 2, rank, 4, maximum_paths=12)
            self.assertEqual(list(resumed), whole.chunks(2))
        self.assertEqual(rank_path_counts(2, 4), [1, 1, 0, 0])
        real = [sample_id for rank in range(4)
                for chunk in UpdateBatchSampler(2, 3, 5, 2, rank, 4, maximum_paths=12)
                for sample_id, _, valid in chunk if valid]
        self.assertEqual(sorted(real), [10, 11])

    def test_scheduler_ema_optimizer_rng_resume_matches_next_cpu_update(self):
        cfg = {"maximum_updates": 10, "warmup_updates": 2,
               "learning_rate": .01, "final_learning_rate": .001}
        original = ToyNetwork()
        optimizer = torch.optim.AdamW(original.parameters(), lr=.01)
        scheduler = UpdateScheduler(optimizer, cfg)
        ema = EMA(original, .9)

        def update(model, opt, sch, average):
            sch.set_for_update(sch.completed_updates)
            opt.zero_grad(set_to_none=True)
            value = sum((parameter - torch.rand_like(parameter)).square()
                        for parameter in model.parameters())
            value.backward()
            opt.step()
            average.update(model)
            sch.completed_updates += 1

        update(original, optimizer, scheduler, ema)
        checkpoint = copy.deepcopy({"model": cpu_tree(original.state_dict()),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "ema": ema.shadow, "rng": capture_rng(torch.device("cpu"))})
        update(original, optimizer, scheduler, ema)
        resumed = ToyNetwork()
        resumed.load_state_dict(checkpoint["model"])
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        resumed_scheduler = UpdateScheduler(resumed_optimizer, cfg)
        resumed_scheduler.load_state_dict(checkpoint["scheduler"])
        resumed_ema = EMA(resumed, .9)
        resumed_ema.shadow = checkpoint["ema"]
        restore_rng(checkpoint["rng"], torch.device("cpu"))
        update(resumed, resumed_optimizer, resumed_scheduler, resumed_ema)
        self.assertEqual(state_digest(original.state_dict()), state_digest(resumed.state_dict()))
        self.assertEqual(state_digest(ema.shadow), state_digest(resumed_ema.shadow))
        self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_resume_metrics_keeps_backup_and_only_valid_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_metrics.jsonl"
            original = b'{"update": 1, "loss": 1}\n{"update": 2, "loss": 2}\n{"update": 3'
            path.write_bytes(original)
            repaired = repair_metrics_prefix(path, 1)
            self.assertEqual(Path(repaired["backup"]).read_bytes(), original)
            self.assertEqual([json.loads(line)["update"] for line in path.read_text().splitlines()], [1])


if __name__ == "__main__":
    unittest.main()

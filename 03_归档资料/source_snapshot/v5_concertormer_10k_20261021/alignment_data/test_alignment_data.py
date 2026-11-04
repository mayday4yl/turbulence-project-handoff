"""CPU tests for actual time-distribution and distributed resume invariants."""
import copy
import random
import unittest

import numpy as np
import torch

from .dataset import AlignmentBatchSampler
from .plans import make_plan, sampled_time, stable_seed


def samples(split, count):
    return [{"split": split, "sample_id": f"{split}_s{i // 4:04d}_r{i % 4:02d}",
             "realization_id": f"{split}_s{i // 4:04d}_r{i % 4:02d}",
             "realization_index": i % 4, "source_id": f"source_{split}_{i // 4}",
             "seed_zernike": i * 2 + 1, "seed_tilt": i * 2 + 2,
             "clean_ref": f"clean/{i // 4}.npy", "clean_sha256": "test",
             "observation_ref": f"observations/{split}_{i}.npy"} for i in range(count)]


class TimePlanTests(unittest.TestCase):
    def test_fp32_strict_bounds_and_same_mapping_as_training(self):
        for realization in range(500):
            for slot in range(10):
                seed = stable_seed("time-boundary-test", realization, slot)
                t, g = sampled_time(seed, slot)
                self.assertGreater(t, slot / 10)
                self.assertLess(t, (slot + 1) / 10)
                self.assertGreater(g, 0)
                self.assertLess(g, 4)
                self.assertEqual(g, (4 * (1 - torch.tensor(t, dtype=torch.float32))).item())

    def test_train_strata_seed_independence_and_no_training_rng_consumption(self):
        base = samples("train", 80)
        random.seed(971)
        np.random.seed(417)
        torch.manual_seed(83)
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        first = make_plan(base, "train", cycle=0)
        second = make_plan(base, "train", cycle=1)
        self.assertEqual(first, make_plan(base, "train", cycle=0))
        self.assertEqual(len(first), 800)
        self.assertEqual(len({r["time_seed"] for r in first + second}), 1600)
        for offset in range(0, len(first), 10):
            rowset = first[offset:offset + 10]
            self.assertEqual([r["time_slot"] for r in rowset], list(range(10)))
            self.assertEqual(len({r["base_sample_id"] for r in rowset}), 1)
            self.assertEqual(len({r["seed_zernike"] for r in rowset}), 1)
            self.assertEqual(len({r["seed_tilt"] for r in rowset}), 1)
        self.assertEqual(random.getstate(), state[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], state[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state[2]))

    def test_val_a_exact_distribution_source_coverage_and_fixed_identity(self):
        base = samples("val", 1000)
        plan = make_plan(base, "val")
        self.assertEqual(plan, make_plan(base, "val"))
        self.assertEqual(len(plan), 1000)
        self.assertEqual(len({r["sample_id"] for r in plan}), 1000)
        self.assertEqual(sum(r["is_endpoint"] for r in plan), 250)
        for slot in range(10):
            self.assertEqual(sum(r["time_slot"] == slot for r in plan), 75)
        for source in {r["source_id"] for r in plan}:
            members = [r for r in plan if r["source_id"] == source]
            self.assertEqual(len(members), 4)
            self.assertEqual(sum(r["is_endpoint"] for r in members), 1)
        with self.assertRaises(ValueError):
            make_plan(base, "test")


class SamplerTests(unittest.TestCase):
    def test_six_ranks_reconstruct_exact_global_batch(self):
        samplers = [AlignmentBatchSampler(8000, rank=rank) for rank in range(6)]
        for update in (1, 2, 1999, 2000, 2001, 3999, 4000, 4001, 5999, 6000):
            reconstructed = []
            for micro in range(4):
                for sampler in samplers:
                    reconstructed.extend(sampler.indices_for(update, micro))
            self.assertEqual(reconstructed, samplers[0].global_indices_for(update))
            self.assertEqual(sum(r[0] == "endpoint" for r in reconstructed), 12)
            self.assertEqual(sum(r[0] == "internal" for r in reconstructed), 36)
            self.assertEqual({r[1] for r in reconstructed if r[0] == "internal"}, {(update - 1) // 2000})

    def test_all_6000_updates_budget_and_without_replacement_per_cache(self):
        sampler = AlignmentBatchSampler(8000)
        consumed = [set(), set(), set()]
        endpoint_count = 0
        for update in range(1, 6001):
            for index in sampler.global_indices_for(update):
                if index[0] == "endpoint":
                    endpoint_count += 1
                else:
                    _, cycle, record = index
                    self.assertNotIn(record, consumed[cycle])
                    consumed[cycle].add(record)
        self.assertEqual([len(c) for c in consumed], [72000, 72000, 72000])
        self.assertEqual(endpoint_count, 72000)
        self.assertEqual(len(sampler), 6000 * 4)

    def test_resume_survives_prefetch_rank_and_segment_end_changes(self):
        old = AlignmentBatchSampler(8000, rank=0, start_update=1, max_updates=2000,
                                    data_identity={"frozen": "same"})
        saved = copy.deepcopy(old.state_dict(2000))
        # Prefetching accesses a future immutable index; it cannot consume a mutable stream.
        for update in (2001, 2002, 2003):
            old.global_indices_for(update)
        new = AlignmentBatchSampler(8000, rank=5, start_update=2001, max_updates=4000,
                                    data_identity={"frozen": "same"})
        self.assertEqual(saved, new.state_dict(2000))
        new.load_state_dict(saved)
        self.assertEqual(new.global_indices_for(2001), old.global_indices_for(2001))
        self.assertEqual(new.start_update, 2001)
        self.assertEqual(len(new), 2000 * 4)
        self.assertEqual(new.required_cycles(), [1])
        other = AlignmentBatchSampler(8000, seed=22, data_identity={"frozen": "same"})
        with self.assertRaises(ValueError):
            other.load_state_dict(saved)

    def test_first_and_last_updates_not_dropped(self):
        sampler = AlignmentBatchSampler(8000, start_update=1, max_updates=1)
        actual = list(sampler)
        self.assertEqual(len(actual), 4)
        self.assertEqual(actual[0], sampler.indices_for(1, 0))
        last = AlignmentBatchSampler(8000, start_update=6000, max_updates=6000)
        self.assertEqual(len(list(last)), 4)
        done = AlignmentBatchSampler(8000, start_update=6001, max_updates=6000)
        self.assertEqual(list(done), [])
        with self.assertRaises(IndexError):
            sampler.indices_for(0, 0)
        with self.assertRaises(IndexError):
            sampler.indices_for(6001, 0)


if __name__ == "__main__":
    unittest.main()

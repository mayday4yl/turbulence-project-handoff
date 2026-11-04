import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from alignment_training.core import unroll_training
from models.clean_concertormer import CleanConcertormer
from models.v5_concertormer import RawXi
from training.state import capture_rng, restore_rng, UpdateBatchSampler
from .prepare_initial import transfer_model, validate_parent, clean_parity
from .probes import (check_result_gradients, check_psf_adjoint, record_render_strengths,
                     verify_render_strengths, tree_equal)


SMALL = dict(img_channel=3, width=8, ffn_expansion_factor=2, enc_blk_nums=[1, 1, 1],
             middle_blk_num=1, dec_blk_nums=[1, 1, 1], enc_heads=[1, 2, 4],
             middle_heads=8, dec_heads=[4, 2, 1])


class TinyDual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.8))
        self.xi = torch.nn.Parameter(torch.tensor(.13))

    def forward(self, z, t):
        clean = z * self.scale + .02
        raw_z = z[:, :1] * self.xi + .03
        raw_t = z[:, :2] * self.xi + .07
        return clean, RawXi(raw_z, raw_t)


class TinySimulator:
    def prepare_image(self, clean):
        return clean

    def render_prepared(self, prepared, raw, g):
        g = torch.as_tensor(g, dtype=prepared.dtype, device=prepared.device)
        if g.ndim == 0:
            g = g.expand(len(prepared))
        return prepared * (1 + .01 * g[:, None, None, None]) + .025 * g[:, None, None, None] * (
            raw.epsilon_zernike + raw.epsilon_tilt.mean(1, keepdim=True))

    def render(self, clean, raw, g):
        return self.render_prepared(self.prepare_image(clean), raw, g)


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_runtime_flags_support_setter_only_internal_format(self):
        from .environment_audit import runtime_flags
        npu = SimpleNamespace(config=object(), conv=SimpleNamespace(allow_hf32=False),
                              matmul=SimpleNamespace(allow_hf32=False))
        backend = SimpleNamespace(_C=SimpleNamespace(_npu_getOption=lambda name: b"enable"))
        with patch.object(torch, "npu", npu, create=True), patch.dict("sys.modules", {"torch_npu": backend}):
            result = runtime_flags()
            self.assertIs(result["internal_format"], True)
            self.assertFalse(result["hf32_conv"])
            npu.conv.allow_hf32 = True
            with self.assertRaisesRegex(RuntimeError, "HF32 must stay OFF"):
                runtime_flags()

    def test_d0_parent_completion_and_objective_are_mandatory(self):
        parent = {"method": "D0", "completed_update": 9750, "best_update": 9750,
                  "config": {"objective": "clean_only"}}
        summary = {"method": "D0", "status": "TRAINING_COMPLETE_REFINEMENT_EVALUATION_PENDING",
                   "completed_update": 10000, "best_update": 9750, "test_used": False}
        validate_parent(parent, summary, "A")
        with self.assertRaises(ValueError):
            validate_parent(parent, dict(summary, completed_update=9750), "A")
        with self.assertRaises(ValueError):
            validate_parent(dict(parent, config={"objective": "v5_dual"}), summary, "A")
        with self.assertRaises(ValueError):
            validate_parent(parent, dict(summary, test_used=True), "A")

    def test_a_reproducible_neutral_time_and_actual_rgb_parity(self):
        d0 = CleanConcertormer(config=SMALL)
        parent = {"model": d0.state_dict()}
        a, evidence = transfer_model(parent, "A", config=SMALL)
        repeated, _ = transfer_model(parent, "A", config=SMALL)
        self.assertTrue(tree_equal(a.state_dict(), repeated.state_dict()))
        self.assertTrue(evidence["backbone_rgb_exact_copy"])
        self.assertTrue(evidence["neutral_time"])
        parity = clean_parity(a, parent, torch.rand(1, 3, 256, 256), device="cpu", config=SMALL)
        self.assertEqual(parity["fp32"]["max_abs"], 0.)
        self.assertEqual(parity["bf16"]["max_abs"], 0.)

    def test_b_inherits_trained_time_raw_without_zero_reset(self):
        d0 = CleanConcertormer(config=SMALL)
        a, _ = transfer_model({"model": d0.state_dict()}, "A", config=SMALL)
        with torch.no_grad():
            for film in a.time.films:
                film.weight.fill_(.0123)
                film.bias.fill_(.0045)
            a.raw_head[0].weight.add_(.125)
        parent = {"model": a.state_dict()}
        b, evidence = transfer_model(parent, "B", seed=47, config=SMALL)
        self.assertTrue(tree_equal(parent["model"], b.state_dict()))
        self.assertTrue(evidence["time_and_raw_inherited_without_reset"])
        self.assertGreater(torch.count_nonzero(b.time.films[0].weight).item(), 0)

    def test_a_current_g_reconstruction_and_effective_raw_gradients(self):
        model, sim = TinyDual(), TinySimulator()
        x, target = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
        t = torch.tensor([0., .375]); g = 4 * (1 - t)
        with record_render_strengths(sim) as records:
            result = unroll_training(model, sim, x, target, method="A", objective="precomputed_current_g",
                time=t, strength=g, path_seed=0, retain_trace=True)
        evidence = verify_render_strengths(records, "A", strength=g)
        self.assertEqual(evidence["strength"], [4., 2.5])
        self.assertTrue(torch.equal(result.loss, .9 * result.clean_loss + .1 * result.observation_loss))
        self.assertEqual(check_result_gradients(result, "A")["source"], "current_g_observation_loss")
        with self.assertRaises(RuntimeError):
            verify_render_strengths(records, "A", strength=torch.tensor([4., 4.]))

    def test_b_strict_paths_g4_matching_and_last_to_first_gradients(self):
        model, sim = TinyDual(), TinySimulator()
        x, target = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
        with record_render_strengths(sim) as records:
            result = unroll_training(model, sim, x, target, method="B", objective="recursive_alignment",
                                     path_seed=73, retain_trace=True)
        self.assertEqual(verify_render_strengths(records, "B", paths=result.paths)["physical_calls"], 7)
        self.assertEqual(check_result_gradients(result, "B")["source"], "last_stage_loss_to_first_prediction")

    def test_b_negative_control_detach_is_detected(self):
        model, sim = TinyDual(), TinySimulator()
        x, target = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
        result = unroll_training(model, sim, x, target, method="B", objective="recursive_alignment",
                                 path_seed=73, retain_trace=True)
        # Replace the checked first prediction with an independently recomputed
        # prediction. A legitimate check must detect the broken path.
        clean, raw = model(x.detach(), torch.zeros(2))
        result.trace[0] = dict(result.trace[0], clean=clean, raw=raw)
        with self.assertRaisesRegex(RuntimeError, "Missing/nonfinite gradient"):
            check_result_gradients(result, "B")

    def test_exact_psf_adjoint_reflection_vjp_cpu64(self):
        result = check_psf_adjoint("cpu", torch.float64)
        self.assertLessEqual(result["gradient_relative_l2"], 1e-10)

    def test_rng_and_indexed_sampler_restore(self):
        random.seed(13); np.random.seed(13); torch.manual_seed(13)
        state = capture_rng("cpu")
        expected = [random.random(), np.random.random(), torch.rand(5)]
        restore_rng(state, "cpu")
        actual = [random.random(), np.random.random(), torch.rand(5)]
        self.assertTrue(tree_equal(actual, expected))
        sampler = UpdateBatchSampler(8000, seed=73, microbatch=2, accumulation_steps=4,
                                    world_size=6, rank=2, start_update=1, max_updates=2)
        restore = UpdateBatchSampler(8000, seed=73, microbatch=2, accumulation_steps=4,
                                    world_size=6, rank=2, start_update=3, max_updates=6000)
        self.assertEqual(sampler.state_dict(2), restore.state_dict(2))
        for micro in range(4):
            self.assertEqual(sampler.indices_for(3, micro), restore.indices_for(3, micro))

    def test_a_sampler_exact_25_75_and_checkpoint_boundary(self):
        from alignment_data.dataset import AlignmentBatchSampler
        sampler = AlignmentBatchSampler(8000, seed=73, rank=0, data_identity={"manifest": "fixed"})
        all_ranks = [AlignmentBatchSampler(8000, seed=73, rank=r, data_identity={"manifest": "fixed"}) for r in range(6)]
        for update in (1, 2, 2000, 2001, 4000, 4001, 6000):
            items = [v for micro in range(4) for rank in range(6) for v in all_ranks[rank].indices_for(update, micro)]
            self.assertEqual(len(items), 48)
            self.assertEqual(sum(i[0] == "endpoint" for i in items), 12)
            self.assertEqual(sum(i[0] == "internal" for i in items), 36)
        saved = sampler.state_dict(2000)
        resumed = copy.deepcopy(sampler)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.start_update, 2001)
        self.assertEqual(resumed.indices_for(2001, 0), sampler.indices_for(2001, 0))
        self.assertTrue(any(i[0] == "internal" and i[1] == 1 for i in resumed.global_indices_for(2001)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

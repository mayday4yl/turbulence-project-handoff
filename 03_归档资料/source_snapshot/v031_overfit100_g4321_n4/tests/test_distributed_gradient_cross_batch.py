"""CPU-only four-process Gloo check of the actual gradient acceptance helper.

Run from project root:
  PYTHONPATH=code:tests python -m unittest discover -s tests \
    -p test_distributed_gradient_cross_batch.py -v
No NPU backend is imported or initialized by this test.
"""
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from check_distributed_gradients import (backward_pass, forward_comparison,
    gradient_comparison, GRADIENT_COSINE_MIN, GRADIENT_RELATIVE_L2_MAX)
from test_core import ToyNetwork, ToySimulator
from test_training import TinyStarts
from v031.runtime import state_hash
from v031.training import TrainingEngine, capture_rank_rng, restore_rank_rng


def gloo_rank(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        for actual_microbatch in (2, 4, 8):
            torch.manual_seed(20260916)
            engine = TrainingEngine(ToyNetwork(), ToySimulator(), TinyStarts(),
                device="cpu", microbatch=actual_microbatch, rank=rank, world_size=4,
                identity={"fixture": "cpu_cross_microbatch_gradient"})
            global_indices = engine.sampler.next_indices()
            local_indices = global_indices[rank * 8:(rank + 1) * 8]
            initial = state_hash(engine.network)
            rng = capture_rank_rng(engine.device)
            expected_metrics, expected = backward_pass(engine, local_indices,
                                                       ddp=False, microbatch=1)
            restore_rank_rng(rng, engine.device)
            actual_metrics, actual = backward_pass(engine, local_indices,
                ddp=True, microbatch=actual_microbatch)
            gradient = gradient_comparison(expected, actual)
            assert gradient["reference_norm"] > 0
            assert gradient["cosine"] >= GRADIENT_COSINE_MIN, gradient
            assert gradient["relative_l2_difference"] <= GRADIENT_RELATIVE_L2_MAX, gradient
            assert forward_comparison(expected_metrics, actual_metrics)["passed"]
            assert state_hash(engine.network) == initial
            assert engine.microbatch == actual_microbatch
            assert engine.accumulation == 8 // actual_microbatch
            assert engine.optimizer_step_calls == engine.optimizer_step_attempts == 0
            assert engine.completed_updates == 0
            assert not engine.optimizer.state
            dist.barrier()
            del engine
    finally:
        dist.destroy_process_group()


class DistributedCrossMicrobatchTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo backend required")
    def test_canonical_mb1_reference_against_mb2_mb4_mb8(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(gloo_rank, args=(rendezvous,), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()

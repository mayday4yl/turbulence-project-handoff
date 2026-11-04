"""Two-rank CPU check of A/B no_sync accumulation against serial gradients."""
from contextlib import nullcontext
from pathlib import Path
import json
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from training.test_semantics import TinyModel, TinySimulator
from .core import unroll_training
from .state import gather_rank_state, atomic_json


def sample(rank, micro, method):
    image = torch.full((1, 3, 4, 4), .45 + .02 * rank + .01 * micro)
    extra = {}
    if method == "A":
        t = torch.tensor([0. if rank == 0 and micro == 0 else .1 + .2 * micro + .1 * rank])
        extra = {"time": t, "strength": 4 * (1 - t)}
    return image, image * .7, extra


def worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    checks = []
    try:
        for method, objective in (("A", "precomputed_current_g"), ("B", "recursive_alignment")):
            model = TinyModel()
            ddp = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=False)
            for micro in range(2):
                image, target, extra = sample(rank, micro, method)
                with ddp.no_sync() if micro == 0 else nullcontext():
                    result = unroll_training(ddp, TinySimulator(), image, target,
                        method=method, objective=objective, path_seed=123 + 2 * micro + rank, **extra)
                    (result.loss / 2).backward()
            got = torch.stack([parameter.grad for parameter in model.parameters()])
            reference = TinyModel()
            for reference_rank in range(world):
                for micro in range(2):
                    image, target, extra = sample(reference_rank, micro, method)
                    result = unroll_training(reference, TinySimulator(), image, target,
                        method=method, objective=objective, path_seed=123 + 2 * micro + reference_rank, **extra)
                    (result.loss / (2 * world)).backward()
            expected = torch.stack([parameter.grad for parameter in reference.parameters()])
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-8)
            checks.append({"method": method, "gradient_max_abs_error": (got - expected).abs().max().item()})
            del ddp, model
        ranks = gather_rank_state({"rank": rank, "checks": checks})
        if rank == 0:
            atomic_json({"status": "PASS", "backend": "gloo_cpu", "world_size": world,
                         "ranks": ranks, "not_npu_compatibility_evidence": True}, output)
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "result.json"
        mp.spawn(worker, args=(2, str(Path(folder) / "rendezvous"), str(output)), nprocs=2, join=True)
        print(output.read_text())


if __name__ == "__main__":
    main()

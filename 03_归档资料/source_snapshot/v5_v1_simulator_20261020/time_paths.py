"""Four-call V5-v1 time paths; this module does not implement training.

Training draws three independent U[0,1) values on CPU in float64, converts to the
requested output dtype, and sorts them. A whole triplet is resampled if the
actual representable t or g=4*(1-t) values have an endpoint or duplicate. This
excludes degenerate discrete floating-point samples only: there is no clamp,
jitter, minimum-spacing threshold, or redistribution of accepted triplets.

Validation is performed in the final dtype on CPU, before device transfer.
Same seed, batch size, dtype and retry limit reproduce the same CPU samples.
The fifth path point is the physical endpoint, not a fifth network call.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

_DTYPES = (torch.float16, torch.float32, torch.float64)


@dataclass(frozen=True)
class TimePaths:
    t: torch.Tensor
    g: torch.Tensor
    retry_counts: torch.Tensor
    seed: int | None

    @property
    def nfe(self) -> int:
        return 4


class TimePathSamplingError(RuntimeError):
    """Rejection sampling exhausted the explicitly bounded additional draws."""

    def __init__(
        self, *, invalid_indices: list[int], retry_counts: list[int],
        dtype: torch.dtype, max_retries: int,
    ) -> None:
        self.invalid_indices = invalid_indices
        self.retry_counts = retry_counts
        self.dtype = dtype
        self.max_retries = max_retries
        super().__init__(
            f"V5 time-path sampling failed for {len(invalid_indices)} path(s) "
            f"after the initial draw and {max_retries} retries per path "
            f"in {dtype}; invalid batch indices={invalid_indices[:20]}. "
            "No clamp, jitter, or spacing adjustment was applied."
        )


def _validate_arguments(batch_size: int, dtype: torch.dtype) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if dtype not in _DTYPES:
        raise ValueError("dtype must be torch.float16, torch.float32, or torch.float64")


def _draw_triplets(count: int, generator: torch.Generator) -> torch.Tensor:
    return torch.rand((count, 3), generator=generator, dtype=torch.float64, device="cpu")


def _valid_triplets(t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(t).all(dim=1)
        & torch.isfinite(g).all(dim=1)
        & (t[:, 0] > 0)
        & (t[:, -1] < 1)
        & (t[:, 1:] > t[:, :-1]).all(dim=1)
        & (g[:, 0] < 4)
        & (g[:, -1] > 0)
        & (g[:, 1:] < g[:, :-1]).all(dim=1)
    )


def sample_training_paths(
    batch_size: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_retries: int = 64,
) -> TimePaths:
    """Return t,g [B,5] and per-path counts of additional triplet draws.

    max_retries=0 allows only the initial draw. Valid rows are retained while
    every invalid row independently receives a complete new triplet. If any
    row remains invalid after its retry budget, raise TimePathSamplingError;
    partial results are not returned.
    """
    _validate_arguments(batch_size, dtype)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    internal_t = torch.empty((batch_size, 3), dtype=dtype)
    internal_g = torch.empty_like(internal_t)
    retry_counts = torch.zeros(batch_size, dtype=torch.int64)
    pending = torch.arange(batch_size)
    for draw_number in range(max_retries + 1):
        sampled = _draw_triplets(len(pending), generator).to(dtype=dtype)
        sampled = sampled.sort(dim=1).values
        sampled_g = 4 * (1 - sampled)
        valid = _valid_triplets(sampled, sampled_g)
        accepted = pending[valid]
        internal_t[accepted] = sampled[valid]
        internal_g[accepted] = sampled_g[valid]
        pending = pending[~valid]
        if not len(pending):
            break
        if draw_number == max_retries:
            raise TimePathSamplingError(
                invalid_indices=pending.tolist(),
                retry_counts=retry_counts.tolist(),
                dtype=dtype,
                max_retries=max_retries,
            )
        retry_counts[pending] += 1
    t = torch.cat((torch.zeros((batch_size, 1), dtype=dtype), internal_t,
                   torch.ones((batch_size, 1), dtype=dtype)), dim=1)
    g = torch.cat((torch.full((batch_size, 1), 4, dtype=dtype), internal_g,
                   torch.zeros((batch_size, 1), dtype=dtype)), dim=1)
    return TimePaths(t.to(device=device), g.to(device=device),
                     retry_counts.to(device=device), seed)


def fixed_inference_paths(
    batch_size: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> TimePaths:
    """Return t=[0,.25,.5,.75,1], g=[4,3,2,1,0], with NFE=4."""
    _validate_arguments(batch_size, dtype)
    t = torch.tensor([0., .25, .5, .75, 1.], dtype=dtype).repeat(batch_size, 1)
    g = 4 * (1 - t)
    return TimePaths(t.to(device=device), g.to(device=device),
                     torch.zeros(batch_size, dtype=torch.int64, device=device), None)

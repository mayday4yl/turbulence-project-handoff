"""CPU-validated V5-v1 strength/time interfaces; no model or simulator runs.

Global time is t=1-g/4, including when G<4. Training always has four calls.
Inference NFE is the number of intervals in its validated path. All random
draws use a local seeded CPU float64 generator; valid rows are never redrawn.
No endpoint clamp, jitter, hidden minimum spacing or threshold relaxation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math
import torch

G_MAX = 4.0
TRAIN_NFE = 4
_DTYPES = (torch.float16, torch.float32, torch.float64)


@dataclass(frozen=True)
class StartStrengthConfig:
    mode: str = "fixed"
    fixed_G: float = 4.0
    bounds: tuple[float, float] | None = None
    seed: int | None = None
    allow_uniform: bool = False


@dataclass(frozen=True)
class StartStrengths:
    G: torch.Tensor
    t0: torch.Tensor
    retry_counts: torch.Tensor
    seed: int | None
    mode: str


@dataclass(frozen=True)
class TimePaths:
    t: torch.Tensor
    g: torch.Tensor
    retry_counts: torch.Tensor
    seed: int | None

    @property
    def nfe(self) -> int:
        return self.t.shape[1] - 1

    @property
    def G(self) -> torch.Tensor:
        return self.g[:, 0]


class ScheduleSamplingError(RuntimeError):
    def __init__(self, *, kind: str, invalid_indices: list[int],
                 retry_counts: list[int], dtype: torch.dtype, max_retries: int):
        self.kind = kind
        self.invalid_indices = invalid_indices
        self.retry_counts = retry_counts
        self.dtype = dtype
        self.max_retries = max_retries
        super().__init__(
            f"{kind} sampling exhausted initial draw + {max_retries} retries "
            f"in {dtype}; invalid batch indices={invalid_indices[:20]}. "
            "No clamp, jitter or spacing adjustment was applied."
        )


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _arguments(batch_size, dtype, max_retries=None, seed=None):
    _positive_int(batch_size, "batch_size")
    if dtype not in _DTYPES:
        raise ValueError("dtype must be float16, float32 or float64")
    if max_retries is not None and (isinstance(max_retries, bool) or
            not isinstance(max_retries, int) or max_retries < 0):
        raise ValueError("max_retries must be a nonnegative integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer")


def _require_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("an explicit integer seed is required")


def _cpu_vector(value, batch_size, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor or Python values")
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"{name} must contain real, nonboolean values")
        value = value.detach().to(dtype=torch.float64)
    else:
        try:
            value = torch.as_tensor(value, dtype=torch.float64, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must be a real scalar or batch vector") from exc
    if value.ndim == 0:
        value = value.repeat(batch_size)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must be scalar or shape [{batch_size}]")
    return value


def _strength_valid(G64, G, t0):
    return (torch.isfinite(G64) & (G64 > 0) & (G64 <= G_MAX)
            & torch.isfinite(G) & (G > 0) & (G <= G_MAX)
            & torch.isfinite(t0) & (t0 >= 0) & (t0 < 1))


def _room_for_three(G, t0):
    # Three representable predecessors of one establish whether any legal
    # triplet exists. This is a feasibility check, never the sampling rule.
    one = torch.tensor(1., dtype=t0.dtype)
    zero = torch.tensor(0., dtype=t0.dtype)
    last = torch.nextafter(one, zero)
    middle = torch.nextafter(last, zero)
    first = torch.nextafter(middle, zero)
    return (t0 < first) & (G > 4 * (1 - first))


def _starts(G, batch_size, dtype, *, require_three):
    G64 = _cpu_vector(G, batch_size, "G")
    result = G64.to(dtype=dtype)
    # Time is formed in float64, then represented in the requested dtype.
    t064 = 1 - result.to(torch.float64) / G_MAX
    t0 = t064.to(dtype=dtype)
    valid = _strength_valid(G64, result, t0)
    if require_three:
        valid &= _room_for_three(result, t0)
    if not bool(valid.all()):
        raise ValueError(
            f"G must satisfy 0<G<=4 and admit a representable "
            f"{'four-step' if require_three else 'nonempty'} global-time path "
            f"in {dtype}; invalid indices={(~valid).nonzero().flatten().tolist()}"
        )
    return result, t0, t064


def _draw_uniform(count, generator):
    return torch.rand(count, generator=generator, dtype=torch.float64, device="cpu")


def _draw_triplets(count, generator):
    return torch.rand((count, 3), generator=generator, dtype=torch.float64, device="cpu")


def sample_start_strengths(batch_size: int, *, config: StartStrengthConfig | None = None,
                           dtype=torch.float32, device="cpu", max_retries=64) -> StartStrengths:
    """Training start interface. V5-v1's active default is fixed G=4.

    Future uniform use requires allow_uniform=True, explicit bounds and seed.
    Accepted G is strictly inside its bounds after dtype conversion. Both
    endpoints, nonfinite draws and strengths lacking three internal times are
    rejected with bounded whole-value retries. It does NOT generate Y_G.
    """
    config = config or StartStrengthConfig()
    _arguments(batch_size, dtype, max_retries)
    if config.mode == "fixed":
        if config.bounds is not None or config.allow_uniform:
            raise ValueError("fixed mode must not carry uniform bounds/opt-in")
        G, t0, _ = _starts(config.fixed_G, batch_size, dtype, require_three=True)
        return StartStrengths(G.to(device), t0.to(device),
                              torch.zeros(batch_size, dtype=torch.int64).to(device),
                              None, "fixed")
    if config.mode != "uniform":
        raise ValueError("mode must be fixed or uniform")
    if not config.allow_uniform:
        raise ValueError("uniform is disabled in V5-v1; explicit opt-in is required")
    _require_seed(config.seed)
    if config.bounds is None or len(config.bounds) != 2:
        raise ValueError("uniform requires explicit (lower, upper) bounds")
    low, high = map(float, config.bounds)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= G_MAX):
        raise ValueError("uniform bounds must satisfy 0<=lower<upper<=4")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    G = torch.empty(batch_size, dtype=dtype)
    t0 = torch.empty_like(G)
    retries = torch.zeros(batch_size, dtype=torch.int64)
    pending = torch.arange(batch_size)
    for number in range(max_retries + 1):
        draws = _draw_uniform(len(pending), generator)
        raw = low + (high - low) * draws
        values = raw.to(dtype=dtype)
        times = (1 - values.to(torch.float64) / G_MAX).to(dtype=dtype)
        valid = (torch.isfinite(draws) & (draws > 0) & (draws < 1)
                 & _strength_valid(raw, values, times)
                 & (values.to(torch.float64) > low) & (values.to(torch.float64) < high)
                 & _room_for_three(values, times))
        accepted = pending[valid]
        G[accepted], t0[accepted] = values[valid], times[valid]
        pending = pending[~valid]
        if not len(pending):
            break
        if number == max_retries:
            raise ScheduleSamplingError(kind="start strength", invalid_indices=pending.tolist(),
                                        retry_counts=retries.tolist(), dtype=dtype,
                                        max_retries=max_retries)
        retries[pending] += 1
    return StartStrengths(G.to(device), t0.to(device), retries.to(device), config.seed, "uniform")


def sample_training_paths(batch_size: int, *, seed: int, G=4.0,
                          dtype=torch.float32, device="cpu", max_retries=64) -> TimePaths:
    """Return [B,5] paths: t0, sorted iid U(t0,1) triplet, 1; NFE=4.

    G may be scalar or a CPU vector of length B. Draw a complete new triplet
    for each invalid row only. Same default seed/dtype reproduces frozen V5
    G=4 paths exactly, including rejection counts.
    """
    _arguments(batch_size, dtype, max_retries, seed)
    _require_seed(seed)
    G, t0, t064 = _starts(G, batch_size, dtype, require_three=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    internal_t = torch.empty((batch_size, 3), dtype=dtype)
    internal_g = torch.empty_like(internal_t)
    retries = torch.zeros(batch_size, dtype=torch.int64)
    pending = torch.arange(batch_size)
    for number in range(max_retries + 1):
        draws = _draw_triplets(len(pending), generator)
        start = t064[pending, None]
        sampled = (start + (1 - start) * draws).to(dtype=dtype).sort(dim=1).values
        sampled_g = 4 * (1 - sampled)
        valid = (torch.isfinite(draws).all(dim=1)
                 & (draws > 0).all(dim=1) & (draws < 1).all(dim=1)
                 & torch.isfinite(sampled).all(dim=1)
                 & torch.isfinite(sampled_g).all(dim=1)
                 & (sampled[:, 0] > t0[pending]) & (sampled[:, -1] < 1)
                 & (sampled[:, 1:] > sampled[:, :-1]).all(dim=1)
                 & (sampled_g[:, 0] < G[pending]) & (sampled_g[:, -1] > 0)
                 & (sampled_g[:, 1:] < sampled_g[:, :-1]).all(dim=1))
        accepted = pending[valid]
        internal_t[accepted], internal_g[accepted] = sampled[valid], sampled_g[valid]
        pending = pending[~valid]
        if not len(pending):
            break
        if number == max_retries:
            raise ScheduleSamplingError(kind="internal time", invalid_indices=pending.tolist(),
                                        retry_counts=retries.tolist(), dtype=dtype,
                                        max_retries=max_retries)
        retries[pending] += 1
    t = torch.cat((t0[:, None], internal_t, torch.ones((batch_size, 1), dtype=dtype)), dim=1)
    g = torch.cat((G[:, None], internal_g, torch.zeros((batch_size, 1), dtype=dtype)), dim=1)
    return TimePaths(t.to(device), g.to(device), retries.to(device), seed)


def inference_paths(G=4.0, *, batch_size: int | None = None, N: int | None = None,
                    gamma: float = 1.0, time_grid=None, dtype=torch.float32,
                    device="cpu") -> TimePaths:
    """General inference g_i=G*(1-(i/N)**gamma), global t_i=1-g_i/4.

    Without a grid N defaults to 4. Explicit grid is global time, shape [N+1]
    or [B,N+1], and determines N unless N is explicitly supplied. Endpoints
    must be exactly represented t0 and 1; no endpoint insertion/correction.
    A grid and gamma!=1 are ambiguous and rejected. All collisions/nonfinite
    results after output-dtype conversion raise, never silently shorten NFE.
    """
    if batch_size is None:
        shape = G.shape if isinstance(G, torch.Tensor) else torch.as_tensor(G).shape
        batch_size = shape[0] if len(shape) == 1 else 1
    _arguments(batch_size, dtype)
    G, t0, _ = _starts(G, batch_size, dtype, require_three=False)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if N is not None:
        _positive_int(N, "N")
    if time_grid is None:
        N = 4 if N is None else N
        fraction = torch.arange(N + 1, dtype=torch.float64) / N
        g64 = G.to(torch.float64)[:, None] * (1 - fraction.pow(gamma))[None, :]
        t = (1 - g64 / G_MAX).to(dtype=dtype)
        g = g64.to(dtype=dtype)
    else:
        if gamma != 1:
            raise ValueError("time_grid and nondefault gamma cannot be combined")
        if isinstance(time_grid, torch.Tensor) and time_grid.device.type != "cpu":
            raise ValueError("time_grid must be provided on CPU")
        raw = torch.as_tensor(time_grid, device="cpu")
        if raw.dtype == torch.bool or raw.is_complex():
            raise ValueError("time_grid must contain real, nonboolean values")
        # Python float lists must not undergo implicit FP32 conversion first.
        raw = torch.as_tensor(time_grid, dtype=torch.float64, device="cpu")
        if raw.ndim == 1:
            raw = raw[None, :].repeat(batch_size, 1)
        if raw.ndim != 2 or raw.shape[0] != batch_size or raw.shape[1] < 2:
            raise ValueError("time_grid must have shape [N+1] or [B,N+1], N>=1")
        inferred = raw.shape[1] - 1
        if N is not None and N != inferred:
            raise ValueError("N does not match time_grid intervals")
        N = inferred
        if not bool((torch.isfinite(raw).all(dim=1)
                     & (raw >= 0).all(dim=1) & (raw <= 1).all(dim=1)
                     & (raw[:, 1:] > raw[:, :-1]).all(dim=1)
                     & (raw[:, -1] == 1)).all()):
            raise ValueError("time_grid must be finite, strictly increasing in [0,1], ending exactly at 1")
        t = raw.to(dtype=dtype)
        g = 4 * (1 - t)
        # Endpoint G is physical input metadata, subject only to dtype cast.
        g[:, 0] = G
    valid = (torch.isfinite(t).all(dim=1) & torch.isfinite(g).all(dim=1)
             & (t[:, 0] == t0) & (t[:, -1] == 1)
             & (g[:, 0] == G) & (g[:, -1] == 0)
             & (t[:, 1:] > t[:, :-1]).all(dim=1)
             & (g[:, 1:] < g[:, :-1]).all(dim=1)
             & (t >= 0).all(dim=1) & (t <= 1).all(dim=1)
             & (g >= 0).all(dim=1) & (g <= G_MAX).all(dim=1))
    if not bool(valid.all()):
        raise ValueError(f"inference path violates endpoint/finite/strict-order conditions in {dtype}; "
                         f"invalid indices={(~valid).nonzero().flatten().tolist()}")
    return TimePaths(t.to(device), g.to(device),
                     torch.zeros(batch_size, dtype=torch.int64).to(device), None)


def validate_observation_strength(requested_G, observation_metadata: Mapping, *,
                                  batch_size: int = 1) -> None:
    """Require explicit per-observation G; exact metadata equality, no relabel.

    This guards provenance, not pixel content. The caller must generate Y_G
    using that strength and bind its metadata when constructing a future
    dataset. A filename such as Y4 is not sufficient provenance.
    """
    _positive_int(batch_size, "batch_size")
    if not isinstance(observation_metadata, Mapping) or "G" not in observation_metadata:
        raise ValueError("observation metadata must contain explicit G")
    requested = _cpu_vector(requested_G, batch_size, "requested_G")
    actual = _cpu_vector(observation_metadata["G"], batch_size, "observation G")
    for values in (requested, actual):
        if not bool((torch.isfinite(values) & (values > 0) & (values <= G_MAX)).all()):
            raise ValueError("observation/requested G must satisfy 0<G<=4")
    mismatched = requested != actual
    if bool(mismatched.any()):
        raise ValueError("observation strength mismatch: generate the matching Y_G; "
                         "Y4 cannot substitute for another strength. "
                         f"indices={mismatched.nonzero().flatten().tolist()}")

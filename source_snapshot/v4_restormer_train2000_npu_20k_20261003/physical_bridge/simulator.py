"""Device-compatible wrapper for the frozen PiRN/P2S simulator.

This module preserves the mathematics and renderer order in the upstream
``assets/simulator_new.py`` while removing hard-coded CUDA calls and exposing
the severity-independent Gaussian sources explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FROZEN_G_GRID = (0.75, 1.50, 2.25, 3.00, 3.75, 4.50)


@dataclass(frozen=True)
class Realization:
    """Severity-independent Gaussian sources used by PiRN/P2S."""

    epsilon_zernike: torch.Tensor
    epsilon_tilt: torch.Tensor


@dataclass(frozen=True)
class EffectiveState:
    zernike: torch.Tensor
    displacement: torch.Tensor


@dataclass(frozen=True)
class PreparedImage:
    image_mean: torch.Tensor
    dictionary_images: torch.Tensor


@dataclass(frozen=True)
class RenderResult:
    output: torch.Tensor
    blur_only: torch.Tensor
    flow: torch.Tensor
    weights: torch.Tensor
    effective_state: EffectiveState


class _P2S(nn.Module):
    """Frozen upstream P2S mapping, including its literal fc2 reuse."""

    def __init__(self, input_dim: int = 36, output_dim: int = 100) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc2(y))
        return self.out(y)


def _noll_to_zernike_index(j: int) -> tuple[int, int]:
    if j == 0:
        raise ValueError("Noll indices start at 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def _noll_covariance(z_count: int = 36) -> torch.Tensor:
    """Literal device-neutral form of upstream ``_nollCovMat(36, 1, 1)``."""

    x = torch.linspace(0, z_count - 1, z_count, dtype=torch.float32)
    xv, yv = torch.meshgrid(x, x, indexing="ij")
    indices = torch.tensor([_noll_to_zernike_index(i + 1) for i in range(z_count)])
    n = indices[:, 0].float()
    m = indices[:, 1].float()
    ni, nj = torch.meshgrid(n, n, indexing="ij")
    mi, mj = torch.meshgrid(m, m, indexing="ij")
    mask = (torch.abs(mi) == torch.abs(mj)) * (torch.remainder(xv - yv, 2) == 0)
    numerator = torch.lgamma(torch.tensor(14.0 / 3.0)) + torch.lgamma(
        (ni + nj - 5.0 / 3.0) / 2.0
    )
    denominator = (
        torch.lgamma((-ni + nj + 17.0 / 3.0) / 2.0)
        + torch.lgamma((ni - nj + 17.0 / 3.0) / 2.0)
        + torch.lgamma((ni + nj + 23.0 / 3.0) / 2.0)
    )
    coefficient = (
        0.0072
        * (np.pi ** (8.0 / 3.0))
        * torch.sqrt((ni + 1) * (nj + 1))
        * ((-1) ** (((ni + nj - 2 * torch.abs(mi)) / 2.0) * mask))
    )
    covariance = coefficient * torch.exp(numerator) / torch.exp(denominator) * mask
    covariance[0, 0] = 1
    return covariance.float()


def _asset_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


class PiRNP2SSimulator(nn.Module):
    """Frozen PiRN/P2S renderer with explicit raw realization and batched severity."""

    def __init__(
        self,
        asset_dir: str | Path,
        *,
        image_size: int = 256,
        rf_range: int = 512,
        aperture_diameter: float = 0.071,
        propagation_distance: float = 100.0,
        wavelength: float = 5e-7,
        correlation: float = -0.05,
        tilt_threshold: float = 0.02,
        adjustment: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.asset_dir = Path(asset_dir)
        self.image_size = int(image_size)
        self.rf_range = max(int(rf_range), self.image_size)
        self.aperture_diameter = float(aperture_diameter)
        self.propagation_distance = float(propagation_distance)
        self.wavelength = float(wavelength)
        self.correlation = float(correlation)
        self.tilt_threshold = float(tilt_threshold)
        self.adjustment = float(adjustment)
        self.device_spec = torch.device(device)

        mapping = _P2S()
        mapping.load_state_dict(_asset_load(self.asset_dir / "P2S_model.pt"))
        dictionary: Mapping[str, torch.Tensor] = _asset_load(
            self.asset_dir / "dictionary.pt"
        )
        i0_i2 = _asset_load(self.asset_dir / "I0_I2.pt")

        self.mapping = mapping
        self.register_buffer("mu", dictionary["mu"].reshape(1, 1, 33, 33).float())
        self.register_buffer(
            "basis_psf", dictionary["dictionary"].unsqueeze(1).float()
        )
        self.register_buffer("r_z", self._build_zernike_mixing())
        self.register_buffer("sqrt_psd", self._build_spatial_sqrt_psd())
        tilt_constant, tilt_s_half_unit = self._build_unit_tilt(i0_i2)
        self.tilt_constant = float(tilt_constant)
        self.register_buffer("tilt_s_half_unit", tilt_s_half_unit)
        self.register_buffer("pixel_position", self._build_pixel_position())

        self.requires_grad_(False)
        self.eval()
        self.to(self.device_spec)

    @property
    def realization_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "epsilon_zernike": (36, self.rf_range, self.rf_range),
            "epsilon_tilt": (self.rf_range, self.rf_range, 2),
        }

    def protocol(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "rf_range": self.rf_range,
            "D": self.aperture_diameter,
            "L": self.propagation_distance,
            "wavelength": self.wavelength,
            "corr": self.correlation,
            "thre": self.tilt_threshold,
            "adj": self.adjustment,
            "renderer_order": "blur_then_tilt",
            "g_grid": list(FROZEN_G_GRID),
            "realization_shapes": self.realization_shapes,
        }

    def _build_zernike_mixing(self) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eig(_noll_covariance(36))
        return torch.real(eigenvectors * torch.sqrt(eigenvalues)).float()

    def _build_spatial_sqrt_psd(self) -> torch.Tensor:
        coords = torch.linspace(
            0, self.rf_range - 1, self.rf_range, dtype=torch.float32
        ) - self.rf_range / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        distance = torch.exp(self.correlation * torch.sqrt(xx.square() + yy.square()))
        return torch.sqrt(torch.fft.fft2(distance))

    def _build_unit_tilt(self, i0_i2: Mapping[str, Any]) -> tuple[float, torch.Tensor]:
        size = self.rf_range
        diameter = self.aperture_diameter
        propagation = self.propagation_distance
        wavelength = self.wavelength
        delta0 = propagation * wavelength / (2.0 * diameter) * self.adjustment
        c1 = 2.0 * ((24.0 / 5.0) * math.gamma(6.0 / 5.0)) ** (5.0 / 6.0)
        c2 = 4.0 * c1 / math.pi * math.gamma(11.0 / 6.0) ** 2
        smax = delta0 / diameter * size
        spacing = delta0 / diameter

        source_s = torch.as_tensor(i0_i2["s"]).detach().cpu()
        s_arr = torch.arange(0, smax, spacing, dtype=torch.float32)
        sample_indices = torch.argmin(
            torch.abs(s_arr.unsqueeze(1) - source_s.unsqueeze(0)), dim=1
        )
        i0 = torch.as_tensor(i0_i2["i0_integral"])[sample_indices].float()
        i2 = torch.as_tensor(i0_i2["i2_integral"])[sample_indices].float()
        i0_zero = torch.as_tensor(i0_i2["I0_0"], dtype=torch.float32)

        yy, xx = torch.meshgrid(
            torch.arange(1, size + 1), torch.arange(1, size + 1), indexing="ij"
        )
        radius = torch.sqrt((yy - size / 2.0) ** 2 + (xx - size / 2.0) ** 2)
        c0 = (i0[radius.long()] + i2[radius.long()]) / i0_zero
        c0[size // 2, size // 2] = 1.0
        c0_scaled = (
            c0
            * i0_zero
            * c2
            * 2.0
            * math.pi
            * (0.5 ** (5.0 / 3.0))
            * (2.0 * wavelength / (math.pi * diameter)) ** 2
        )
        s_half = torch.sqrt(torch.fft.fft2(c0_scaled))
        maximum = s_half.abs().max()
        if maximum > 0:
            s_half = torch.where(
                s_half.abs() < self.tilt_threshold * maximum,
                torch.zeros_like(s_half),
                s_half,
            )
        constant = math.sqrt(2.0) * size * (propagation / delta0)
        return constant, s_half

    def _build_pixel_position(self) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(0, self.image_size),
            torch.arange(0, self.image_size),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).float()

    @staticmethod
    def _seeded_noise(shape: Sequence[int], seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)

    def sample_realization(
        self,
        batch_size: int,
        *,
        seed_zernike: int,
        seed_tilt: int,
    ) -> Realization:
        z_shape = (batch_size, *self.realization_shapes["epsilon_zernike"])
        t_shape = (batch_size, *self.realization_shapes["epsilon_tilt"])
        return Realization(
            self._seeded_noise(z_shape, seed_zernike).to(self.device_spec),
            self._seeded_noise(t_shape, seed_tilt).to(self.device_spec),
        )

    def _strength_tensor(
        self, g: float | Sequence[float] | torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        value = torch.as_tensor(g, dtype=torch.float32, device=self.device_spec)
        if value.ndim == 0:
            value = value.repeat(batch_size)
        value = value.reshape(-1)
        if value.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} severities, got {value.numel()}")
        if not bool(torch.all(value > 0).detach().cpu()):
            raise ValueError("PiRN/P2S severity must be positive")
        return value

    def effective_state(
        self,
        realization: Realization,
        g: float | Sequence[float] | torch.Tensor,
    ) -> EffectiveState:
        epsilon_z = realization.epsilon_zernike.float()
        epsilon_t = realization.epsilon_tilt.float()
        batch_size = epsilon_z.shape[0]
        expected_z = (batch_size, 36, self.rf_range, self.rf_range)
        expected_t = (batch_size, self.rf_range, self.rf_range, 2)
        if tuple(epsilon_z.shape) != expected_z or tuple(epsilon_t.shape) != expected_t:
            raise ValueError(
                f"Realization shape mismatch: z={tuple(epsilon_z.shape)}, "
                f"tilt={tuple(epsilon_t.shape)}"
            )
        amplitude = self._strength_tensor(g, batch_size).pow(5.0 / 6.0)
        scaled_z = amplitude[:, None, None, None] * epsilon_z
        mixed = torch.einsum("ij,kj...->ki...", self.r_z, scaled_z)
        zernike = torch.fft.ifft2(
            self.sqrt_psd[None, None] * torch.fft.fft2(mixed, dim=(2, 3)),
            dim=(-2, -1),
        ).real[:, :, : self.image_size, : self.image_size]

        s_half_pair = torch.stack(
            (self.tilt_s_half_unit.real, self.tilt_s_half_unit.imag), dim=-1
        )
        displacement = torch.fft.irfft2(
            s_half_pair[None]
            * amplitude[:, None, None, None]
            * epsilon_t,
            s=(self.rf_range, self.rf_range),
            dim=(1, 2),
        ) * self.tilt_constant
        displacement = displacement[:, : self.image_size, : self.image_size, :]
        return EffectiveState(zernike=zernike, displacement=displacement)

    def prepare_image(self, image: torch.Tensor) -> PreparedImage:
        if image.ndim != 4 or image.shape[1:] != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(f"Expected [B,3,{self.image_size},{self.image_size}]")
        image = image.to(self.device_spec, dtype=torch.float32)
        batch_size, channels = image.shape[:2]
        padded = F.pad(
            image.reshape(-1, 1, self.image_size, self.image_size),
            (16, 16, 16, 16),
            mode="reflect",
        )
        image_mean = F.conv2d(padded, self.mu).reshape(
            batch_size, channels, self.image_size, self.image_size
        )
        dictionary_images = F.conv2d(padded, self.basis_psf).reshape(
            batch_size, channels, 100, self.image_size, self.image_size
        )
        return PreparedImage(image_mean, dictionary_images)

    def render_prepared(
        self,
        prepared: PreparedImage,
        realization: Realization,
        g: float | Sequence[float] | torch.Tensor,
    ) -> RenderResult:
        state = self.effective_state(realization, g)
        batch_size = state.zernike.shape[0]
        weights = self.mapping(
            state.zernike.permute(0, 2, 3, 1).reshape(
                batch_size, self.image_size * self.image_size, 36
            )
        ).reshape(batch_size, self.image_size, self.image_size, 100)
        weights = weights.permute(0, 3, 1, 2)
        blur_only = (
            torch.sum(weights[:, None] * prepared.dictionary_images, dim=2)
            + prepared.image_mean
        )
        denominator = torch.tensor(
            (self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.device_spec,
        ) - 1.0
        flow = 2.0 * (self.pixel_position + state.displacement) / denominator - 1.0
        output = F.grid_sample(
            blur_only,
            flow,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return RenderResult(output, blur_only, flow, weights, state)

    def render(
        self,
        image: torch.Tensor,
        realization: Realization,
        g: float | Sequence[float] | torch.Tensor,
    ) -> RenderResult:
        return self.render_prepared(self.prepare_image(image), realization, g)

    def simulator_parameter_gradients(self) -> dict[str, torch.Tensor | None]:
        return {name: parameter.grad for name, parameter in self.named_parameters()}


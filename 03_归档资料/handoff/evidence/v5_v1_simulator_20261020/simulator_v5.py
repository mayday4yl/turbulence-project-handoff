"""V5-v1 simulator candidate. Final acceptance is deliberately not frozen.

Inherited physics/assets are the pinned V4 PiRN/P2S implementation. Changes:
S1 impulse/physical-PSF interpolation BEFORE warp; S2 pixel-center sampling;
S3 full-complex inverse transform translated from legacy P2S.
No network, optimizer, dataset writer, or inference refinement lives here.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vendor.v4_simulator import (
    PiRNP2SSimulator, Realization, EffectiveState, RenderResult,
)

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PreparedImageV5:
    image_mean: torch.Tensor
    dictionary_images: torch.Tensor
    original: torch.Tensor


def bind_canonical_buffers(simulator, asset_dir=ROOT / "assets"):
    """Use archived generating-side buffers, avoiding eig/FFT rebuild drift."""
    asset_dir = Path(asset_dir)
    meta = json.loads((asset_dir / "CANONICAL_CPU_REFERENCE.json").read_text())
    path = asset_dir / "CANONICAL_CPU_REFERENCE.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["npz_sha256"]:
        raise ValueError("Canonical buffer file hash mismatch")
    # This archive is meaningful only for its original physical configuration.
    for name, expected in meta["physics"].items():
        if getattr(simulator, name) != expected:
            raise ValueError(f"Canonical configuration mismatch: {name}")
    with np.load(path, allow_pickle=False) as archive:
        for name, record in meta["buffers"].items():
            array = archive[name].copy()
            if hashlib.sha256(array.tobytes()).hexdigest() != record["raw_bytes_sha256"]:
                raise ValueError(f"Canonical buffer hash mismatch: {name}")
            setattr(simulator, name, torch.from_numpy(array).to(simulator.device_spec))
    simulator.tilt_constant = meta["tilt_constant"]
    return simulator


class V5Simulator(PiRNP2SSimulator):
    """Differentiable V5 candidate with a shape-compatible raw-xi interface.

    The flags exist only for attributable S1/S2/S3 local ablations. They must be
    recorded in every report. The all-enabled candidate is the V5 default.
    """

    def __init__(self, asset_dir=ROOT / "assets", *, smooth_psf=True,
                 pixel_center_grid=True, tilt_fft="legacy_full", **kwargs):
        if tilt_fft not in ("legacy_full", "v4_irfft"):
            raise ValueError("Unknown tilt FFT semantics")
        super().__init__(asset_dir, **kwargs)
        self.smooth_psf = bool(smooth_psf)
        self.pixel_center_grid = bool(pixel_center_grid)
        self.tilt_fft = tilt_fft

    def protocol(self):
        return {
            **super().protocol(),
            "version": "V5-v1-simulator-candidate",
            "status": "LOCAL_CHECKS_ONLY_FINAL_GATE_PENDING",
            "g_grid": None, "g_interval": [0.0, 4.0],
            "psf_smooth": self.smooth_psf,
            "psf_formula": "(1-g/4)*delta+(g/4)*h_g",
            "pixel_center_grid": self.pixel_center_grid,
            "tilt_fft": self.tilt_fft,
            "tilt_fft_normalization": "backward",
            "raw_xi_sampling": "unchanged independent real standard Gaussian blocks",
            "blur_padding": "reflect", "warp_padding": "border",
            "warp_align_corners": False,
            "psf_normalization": "inherited basis sums; measured, no extra renormalization",
            "physical_covariance_validation": "not established by legacy API equivalence",
        }

    def _strength_tensor(self, g, batch_size):
        value = torch.as_tensor(g, dtype=self.mu.dtype, device=self.mu.device)
        if value.ndim == 0:
            value = value.repeat(batch_size)
        value = value.reshape(-1)
        if value.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} severities, got {value.numel()}")
        if not bool((torch.isfinite(value) & (value >= 0) & (value <= 4)).all()):
            raise ValueError("V5 severity must be finite and in [0,4]")
        return value

    def effective_state(self, realization, g):
        # Follow module placement, including after .to(device). No CPU fallback.
        device, dtype = self.mu.device, self.mu.dtype
        epsilon_z = realization.epsilon_zernike.to(device=device, dtype=dtype)
        epsilon_t = realization.epsilon_tilt.to(device=device, dtype=dtype)
        b = epsilon_z.shape[0]
        if epsilon_z.shape != (b, 36, self.rf_range, self.rf_range):
            raise ValueError("Zernike raw-xi shape mismatch")
        if epsilon_t.shape != (b, self.rf_range, self.rf_range, 2):
            raise ValueError("Tilt raw-xi shape mismatch")
        amplitude = self._strength_tensor(g, b).pow(5 / 6)
        mixed = torch.einsum("ij,kj...->ki...", self.r_z,
                             epsilon_z * amplitude[:, None, None, None])
        zernike = torch.fft.ifft2(
            self.sqrt_psd[None, None] * torch.fft.fft2(mixed, dim=(2, 3)),
            dim=(-2, -1), norm="backward",
        ).real[:, :, :self.image_size, :self.image_size]
        if self.tilt_fft == "legacy_full":
            # Two independent real raw fields, each multiplied by the SAME full
            # complex filter. The last axis is dx/dy, NOT a real/imag pair.
            frequency = (self.tilt_s_half_unit[None, :, :, None]
                         * amplitude[:, None, None, None] * epsilon_t)
            displacement = torch.fft.ifft2(
                frequency, dim=(1, 2), norm="backward"
            ).real * self.tilt_constant
        else:
            # Exact V4 operator retained only as a diagnostic control.
            pair = torch.stack((self.tilt_s_half_unit.real,
                                self.tilt_s_half_unit.imag), -1)
            displacement = torch.fft.irfft2(
                pair[None] * amplitude[:, None, None, None] * epsilon_t,
                s=(self.rf_range, self.rf_range), dim=(1, 2),
            ) * self.tilt_constant
        return EffectiveState(
            zernike, displacement[:, :self.image_size, :self.image_size, :])

    def prepare_image(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected [B,3,{self.image_size},{self.image_size}]")
        image = image.to(device=self.mu.device, dtype=self.mu.dtype)
        b, c = image.shape[:2]
        padded = F.pad(image.reshape(-1, 1, self.image_size, self.image_size),
                       (16, 16, 16, 16), mode="reflect")
        mean = F.conv2d(padded, self.mu).reshape(b, c, self.image_size, self.image_size)
        basis = F.conv2d(padded, self.basis_psf).reshape(
            b, c, 100, self.image_size, self.image_size)
        return PreparedImageV5(mean, basis, image)

    def sampling_grid(self, displacement):
        position = self.pixel_position.to(dtype=displacement.dtype)
        if self.pixel_center_grid:
            return 2 * (position + displacement + 0.5) / self.image_size - 1
        return 2 * (position + displacement) / (self.image_size - 1) - 1

    def warp(self, image, displacement):
        return F.grid_sample(image, self.sampling_grid(displacement),
                             mode="bilinear", padding_mode="border",
                             align_corners=False)

    def render_prepared(self, prepared, realization, g):
        if not isinstance(prepared, PreparedImageV5):
            raise TypeError("V5 requires PreparedImageV5 with original image")
        state = self.effective_state(realization, g)
        b = state.zernike.shape[0]
        strength = self._strength_tensor(g, b)
        weights = self.mapping(
            state.zernike.permute(0, 2, 3, 1).reshape(
                b, self.image_size * self.image_size, 36)
        ).reshape(b, self.image_size, self.image_size, 100).permute(0, 3, 1, 2)
        physical_blur = (weights[:, None] * prepared.dictionary_images).sum(2) + prepared.image_mean
        if self.smooth_psf:
            a = strength[:, None, None, None] / 4
            # Exact kernel-mixture algebra for spatially varying PSFs, before
            # warp, with the original reflected-convolution boundary semantics.
            blur = (1 - a) * prepared.original + a * physical_blur
        else:
            blur = physical_blur
        flow = self.sampling_grid(state.displacement)
        output = F.grid_sample(blur, flow, mode="bilinear",
                               padding_mode="border", align_corners=False)
        return RenderResult(output, blur, flow, weights, state)


def build_simulator(*, device="cpu", **kwargs):
    # Build and bind immutable physics on CPU, then transfer all buffers.
    sim = V5Simulator(device="cpu", **kwargs)
    bind_canonical_buffers(sim)
    sim.to(device)
    sim.device_spec = torch.device(device)
    return sim

"""Opt-in execution candidate: unchanged PSF forward, exact linear adjoint.

For valid unit-stride cross correlation A_w x, A_w^T v equals
conv2d(v, flip(w.transpose(0, 1)), padding=k-1). Reflection padding
stays outside this Function, so PyTorch retains its original boundary adjoint.
No approximate kernels, reduced precision, trainable-kernel gradients or detach.
The accepted Simulator source and the production default are not modified.
"""
from types import MethodType
import torch
import torch.nn.functional as F


class FrozenPSFConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, weight, adjoint_weight):
        if weight.requires_grad or adjoint_weight.requires_grad:
            raise ValueError("Only frozen kernels are supported")
        if weight.ndim != 4 or weight.shape[1:] != (1, 33, 33):
            raise ValueError("Only the canonical unit-stride 33x33 kernels are supported")
        if image.dtype != weight.dtype or adjoint_weight.dtype != weight.dtype:
            raise ValueError("Mixed precision is forbidden in the PSF candidate")
        if image.dtype not in (torch.float32, torch.float64):
            raise ValueError("FP32 execution or CPU FP64 reference only")
        ctx.save_for_backward(adjoint_weight)
        return F.conv2d(image, weight)

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            result = F.conv2d(gradient, weight, padding=32)
        return result, None, None


def adjoint_kernel(weight):
    return weight.transpose(0, 1).flip(-1, -2).contiguous()


def install(simulator):
    """Install only on an explicitly supplied candidate instance after placement."""
    if getattr(simulator, "_psf_adjoint_candidate", False):
        raise ValueError("Candidate already installed")
    from simulator_v5 import PreparedImageV5
    if simulator.mu.requires_grad or simulator.basis_psf.requires_grad:
        raise ValueError("Simulator kernels must be frozen")
    if simulator.mu.dtype != torch.float32 or simulator.basis_psf.dtype != torch.float32:
        raise ValueError("The V5 candidate must retain the FP32 Simulator")
    simulator.register_buffer("_psf_mean_adjoint", adjoint_kernel(simulator.mu), persistent=False)
    simulator.register_buffer("_psf_basis_adjoint", adjoint_kernel(simulator.basis_psf), persistent=False)
    simulator._original_prepare_image = simulator.prepare_image

    def prepare_image(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected canonical 256x256 RGB image")
        image = image.to(device=self.mu.device, dtype=self.mu.dtype)
        b, c = image.shape[:2]
        padded = F.pad(image.reshape(-1, 1, self.image_size, self.image_size),
                       (16, 16, 16, 16), mode="reflect")
        mean = FrozenPSFConv.apply(padded, self.mu, self._psf_mean_adjoint).reshape(
            b, c, self.image_size, self.image_size)
        basis = FrozenPSFConv.apply(padded, self.basis_psf, self._psf_basis_adjoint).reshape(
            b, c, 100, self.image_size, self.image_size)
        return PreparedImageV5(mean, basis, image)

    simulator.prepare_image = MethodType(prepare_image, simulator)
    simulator._psf_adjoint_candidate = True
    return simulator

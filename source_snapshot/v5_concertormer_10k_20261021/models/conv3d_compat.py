"""Function-preserving implementation of (3,3,1) talking convolutions.

The last spatial axis is independent: it can be folded into batch while
performing the identical 2D convolution. Original 5D weight shapes and state
dictionary keys remain unchanged. No tensor is detached in either backend.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AxisIndependentConv3d(nn.Module):
    BACKENDS = ("conv3d", "conv2d_equivalent")

    def __init__(self, original: nn.Conv3d, *, backend="conv3d"):
        super().__init__()
        if (original.kernel_size[-1] != 1 or original.stride[-1] != 1 or
                original.dilation[-1] != 1 or original.padding_mode != "zeros"):
            raise ValueError("Only last-axis independent zero-padding Conv3d is supported")
        if not isinstance(original.padding, str) and original.padding[-1] != 0:
            raise ValueError("The independent final axis must have zero padding")
        self.in_channels, self.out_channels = original.in_channels, original.out_channels
        self.kernel_size, self.stride = original.kernel_size, original.stride
        self.padding, self.dilation, self.groups = original.padding, original.dilation, original.groups
        # Reuse the exact Parameter objects: no new RNG draws or initialization.
        self.weight, self.bias = original.weight, original.bias
        self.set_backend(backend)

    def set_backend(self, backend):
        if backend not in self.BACKENDS:
            raise ValueError(f"Talking convolution backend must be one of {self.BACKENDS}")
        self.backend = backend

    def forward(self, x):
        if self.backend == "conv3d":
            return F.conv3d(x, self.weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        if x.ndim != 5:
            raise ValueError("Talking convolution requires B,C,D,H,W input")
        b, c, d, h, w = x.shape
        images = x.permute(0, 4, 1, 2, 3).reshape(b*w, c, d, h)
        padding = self.padding if isinstance(self.padding, str) else self.padding[:2]
        output = F.conv2d(images, self.weight.squeeze(-1), self.bias,
                          self.stride[:2], padding, self.dilation[:2], self.groups)
        _, channels, out_d, out_h = output.shape
        return output.reshape(b, w, channels, out_d, out_h).permute(0, 2, 3, 4, 1)


def install_talking_backend(backbone, backend):
    """Replace only the official attention b/l talking operators."""
    for module in backbone.modules():
        for name in ("b_talking", "l_talking"):
            original = getattr(module, name, None)
            if isinstance(original, nn.Conv3d):
                setattr(module, name, AxisIndependentConv3d(original, backend=backend))
            elif isinstance(original, AxisIndependentConv3d):
                original.set_backend(backend)

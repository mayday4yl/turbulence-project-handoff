"""Minimal V4 adapter; original V3 backbone and both decoders stay intact."""
import torch
from torch import nn
from physical_bridge.model import DualHeadUNet


class JointCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, condition):
        if condition.ndim != 2 or condition.shape[1] != 2:
            raise ValueError("Expected original-valued [g,t] with shape [B,2]")
        if not torch.isfinite(condition).all():
            raise ValueError("Non-finite condition")
        return self.mlp(condition.float())


class NoiseBridgeUNet(DualHeadUNet):
    def __init__(self, stats, **config):
        if config.get('severity_embedding_dim', 256) != 256:
            raise ValueError('V4 joint condition dimension is fixed at 256')
        super().__init__(**config)
        self.input = nn.Conv2d(6, self.input.out_channels, 3, padding=1)
        self.severity = JointCondition()
        self.stats = stats

    def forward(self, image, condition):
        if image.ndim != 4 or image.shape[1] != 6 or condition.shape != (image.shape[0], 2):
            raise ValueError('Expected concat(Z,Y) [B,6,H,W] and [g,t] [B,2]')
        if image.dtype != torch.float32:
            raise ValueError('FP32 correctness implementation requires float32 input')
        # Deliberately no image clamp, normalization, or condition rescaling.
        return super().forward(image, condition)

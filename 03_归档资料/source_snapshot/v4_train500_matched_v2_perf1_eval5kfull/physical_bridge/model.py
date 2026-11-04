"""Severity-conditioned dual-head residual U-Net for the frozen bridge task."""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .simulator import Realization


def _groups(channels: int, requested: int) -> int:
    groups = min(channels, requested)
    while channels % groups:
        groups -= 1
    return groups


class SeverityEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("Severity embedding dimension must be even")
        self.dimension = dimension
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, severity: torch.Tensor) -> torch.Tensor:
        severity = severity.float().reshape(-1)
        normalized = (severity - 0.75) / (4.50 - 0.75)
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=severity.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = normalized[:, None] * frequencies[None] * 1000.0
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return self.mlp(embedding.to(next(self.mlp.parameters()).dtype))


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.film = nn.Linear(embedding_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.film(embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(value)


class EncoderLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        groups: int,
        blocks: int,
    ) -> None:
        super().__init__()
        layers = [FiLMResBlock(in_channels, out_channels, embedding_dim, groups)]
        layers.extend(
            FiLMResBlock(out_channels, out_channels, embedding_dim, groups)
            for _ in range(blocks - 1)
        )
        self.blocks = nn.ModuleList(layers)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value, embedding)
        return value


class DecoderLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        embedding_dim: int,
        groups: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.up = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        layers = [
            FiLMResBlock(out_channels + skip_channels, out_channels, embedding_dim, groups)
        ]
        layers.extend(
            FiLMResBlock(out_channels, out_channels, embedding_dim, groups)
            for _ in range(blocks - 1)
        )
        self.blocks = nn.ModuleList(layers)

    def forward(
        self, value: torch.Tensor, skip: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="nearest")
        value = self.up(value)
        value = torch.cat((value, skip), dim=1)
        for block in self.blocks:
            value = block(value, embedding)
        return value


class DecoderBranch(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        embedding_dim: int,
        groups: int,
        blocks: int,
    ) -> None:
        super().__init__()
        levels = []
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            levels.append(
                DecoderLevel(
                    current,
                    skip_channels,
                    skip_channels,
                    embedding_dim,
                    groups,
                    blocks,
                )
            )
            current = skip_channels
        self.levels = nn.ModuleList(levels)

    def forward(
        self, bottleneck: torch.Tensor, skips: Sequence[torch.Tensor], embedding: torch.Tensor
    ) -> torch.Tensor:
        value = bottleneck
        for level, skip in zip(self.levels, reversed(skips[:-1]), strict=True):
            value = level(value, skip, embedding)
        return value


class BridgePrediction(NamedTuple):
    clean: torch.Tensor
    epsilon_zernike: torch.Tensor
    epsilon_tilt: torch.Tensor

    @property
    def realization(self) -> Realization:
        return Realization(self.epsilon_zernike, self.epsilon_tilt)


class DualHeadUNet(nn.Module):
    """Frozen v0.1 network family; output realization stays simulator-native."""

    def __init__(
        self,
        *,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4, 4),
        residual_blocks_per_level: int = 2,
        group_norm_groups: int = 32,
        severity_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        channels = [base_channels * int(multiplier) for multiplier in channel_multipliers]
        self.config = {
            "base_channels": base_channels,
            "channel_multipliers": list(channel_multipliers),
            "residual_blocks_per_level": residual_blocks_per_level,
            "group_norm_groups": group_norm_groups,
            "severity_embedding_dim": severity_embedding_dim,
        }
        self.severity = SeverityEmbedding(severity_embedding_dim)
        self.input = nn.Conv2d(3, channels[0], 3, padding=1)
        encoder = []
        downsample = []
        current = channels[0]
        for index, output_channels in enumerate(channels):
            encoder.append(
                EncoderLevel(
                    current,
                    output_channels,
                    severity_embedding_dim,
                    group_norm_groups,
                    residual_blocks_per_level,
                )
            )
            current = output_channels
            if index != len(channels) - 1:
                downsample.append(nn.Conv2d(current, channels[index + 1], 3, stride=2, padding=1))
                current = channels[index + 1]
        self.encoder = nn.ModuleList(encoder)
        self.downsample = nn.ModuleList(downsample)
        self.middle = nn.ModuleList(
            [
                FiLMResBlock(channels[-1], channels[-1], severity_embedding_dim, group_norm_groups),
                FiLMResBlock(channels[-1], channels[-1], severity_embedding_dim, group_norm_groups),
            ]
        )
        self.clean_decoder = DecoderBranch(
            channels,
            severity_embedding_dim,
            group_norm_groups,
            residual_blocks_per_level,
        )
        self.realization_decoder = DecoderBranch(
            channels,
            severity_embedding_dim,
            group_norm_groups,
            residual_blocks_per_level,
        )
        self.clean_head = nn.Conv2d(channels[0], 3, 3, padding=1)
        self.realization_up = nn.Conv2d(channels[0], channels[0], 3, padding=1)
        self.realization_block = FiLMResBlock(
            channels[0], channels[0], severity_embedding_dim, group_norm_groups
        )
        self.zernike_head = nn.Conv2d(channels[0], 36, 3, padding=1)
        self.tilt_head = nn.Conv2d(channels[0], 2, 3, padding=1)
        self._initialize_realization_heads()

    def _initialize_realization_heads(self) -> None:
        # A zero raw-Gaussian estimate is finite and neutral for the frozen simulator.
        nn.init.zeros_(self.zernike_head.weight)
        nn.init.zeros_(self.zernike_head.bias)
        nn.init.zeros_(self.tilt_head.weight)
        nn.init.zeros_(self.tilt_head.bias)

    def forward(self, image: torch.Tensor, severity: torch.Tensor) -> BridgePrediction:
        embedding = self.severity(severity)
        value = self.input(image)
        skips = []
        for index, level in enumerate(self.encoder):
            value = level(value, embedding)
            skips.append(value)
            if index < len(self.downsample):
                value = self.downsample[index](value)
        for block in self.middle:
            value = block(value, embedding)

        clean_features = self.clean_decoder(value, skips, embedding)
        clean = torch.sigmoid(self.clean_head(clean_features)).float()

        realization_features = self.realization_decoder(value, skips, embedding)
        realization_features = F.interpolate(
            realization_features, scale_factor=2.0, mode="nearest"
        )
        realization_features = self.realization_up(realization_features)
        realization_features = self.realization_block(realization_features, embedding)
        epsilon_zernike = self.zernike_head(realization_features).float()
        epsilon_tilt = self.tilt_head(realization_features).permute(0, 2, 3, 1).float()
        return BridgePrediction(clean, epsilon_zernike, epsilon_tilt)

    def branch_parameters(self) -> dict[str, list[nn.Parameter]]:
        return {
            "shared": [
                *self.severity.parameters(),
                *self.input.parameters(),
                *self.encoder.parameters(),
                *self.downsample.parameters(),
                *self.middle.parameters(),
            ],
            "clean": [*self.clean_decoder.parameters(), *self.clean_head.parameters()],
            "realization": [
                *self.realization_decoder.parameters(),
                *self.realization_up.parameters(),
                *self.realization_block.parameters(),
                *self.zernike_head.parameters(),
                *self.tilt_head.parameters(),
            ],
        }

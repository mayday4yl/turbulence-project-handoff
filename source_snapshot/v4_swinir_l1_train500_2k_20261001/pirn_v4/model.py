"""PiRN Swin/RSTB backbone with the existing V4 public tensor interface.

This is an optional architecture candidate, not a pretrained PiRN restoration
pipeline. The vendored SwinIR implementation supplies the shallow embedding,
RSTBs and nearest-plus-convolution RGB reconstruction modules. Its RGB mean
subtraction and coupled input/output channel handling are deliberately bypassed.
V4 still supplies raw [g, t], a six-channel concat(Z, Y), a sigmoid clean output,
and two simulator-native raw Gaussian realization blocks.

The new realization decoder has no U-Net skip connections: PiRN's transformer
features do not expose the old U-Net's multiscale feature hierarchy. Defaults
map the shared 32x32 grid to the native 512x512 realization grid in four steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from physical_bridge.model import BridgePrediction, FiLMResBlock
from v4.model import JointCondition
from vendor.pirn.swinir import SwinIR


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _RealizationUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.block = FiLMResBlock(channels, channels, 256, groups=32)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        return self.block(F.silu(self.conv(value)), embedding)


class PiRNNoiseBridge(nn.Module):
    """Freshly initialized PiRN backbone adapted to V4's dual-output contract.

    Production defaults retain the public PiRN RSTB width/depth/window layout.
    Smaller widths/depths and image sizes are accepted for interface tests; they
    do not constitute validation of the default network or physical simulator.

    The default deterministic backbone has no stochastic depth. Activation
    checkpointing is intentionally rejected pending separate BPTT validation.
    No pretrained or three-channel checkpoint is implicitly loaded or migrated.
    """

    def __init__(
        self,
        stats: nn.Module,
        *,
        image_size: int = 256,
        embed_dim: int = 180,
        depths: Sequence[int] = (6,) * 8,
        num_heads: Sequence[int] = (6,) * 8,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        unshuffle_scale: int = 8,
        decoder_channels: int = 64,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        image_size = _positive_int("image_size", image_size)
        embed_dim = _positive_int("embed_dim", embed_dim)
        window_size = _positive_int("window_size", window_size)
        decoder_channels = _positive_int("decoder_channels", decoder_channels)
        unshuffle_scale = _positive_int("unshuffle_scale", unshuffle_scale)
        if unshuffle_scale not in (2, 4, 8):
            raise ValueError("PiRN nearest+conv supports unshuffle_scale 2, 4 or 8")
        if image_size % (unshuffle_scale * window_size):
            raise ValueError("image_size must be divisible by unshuffle_scale * window_size")
        depths, num_heads = tuple(depths), tuple(num_heads)
        if not depths or len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have equal nonzero length")
        for index, (depth, heads) in enumerate(zip(depths, num_heads, strict=True)):
            _positive_int(f"depths[{index}]", depth)
            _positive_int(f"num_heads[{index}]", heads)
            if embed_dim % heads:
                raise ValueError("embed_dim must be divisible by every num_heads entry")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(embed_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must define a finite positive hidden dimension")
        if not math.isfinite(drop_path_rate) or not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be finite and in [0, 1)")
        if use_checkpoint is not False:
            raise ValueError("Activation checkpointing is not validated for this candidate")
        if not isinstance(stats, nn.Module):
            raise TypeError("stats must be the frozen RGB statistics nn.Module")
        if any(parameter.requires_grad for parameter in stats.parameters()):
            raise ValueError("RGB statistics cannot contain trainable parameters")

        self.image_size = image_size
        self.realization_size = 2 * image_size
        self.stats = stats
        self.severity = JointCondition()
        grid_size = image_size // unshuffle_scale
        base = SwinIR(
            img_size=grid_size,
            patch_size=1,
            in_chans=6,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=float(mlp_ratio),
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=float(drop_path_rate),
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            sf=unshuffle_scale,
            img_range=1.0,
            upsampler="nearest+conv",
            resi_connection="1conv",
            unshuffle=True,
            unshuffle_scale=unshuffle_scale,
        )
        # Transfer only used modules. Do not register the original six-channel
        # output convolution or call base.forward (which also transforms RGB).
        self.conv_first = base.conv_first
        self.patch_embed = base.patch_embed
        self.patch_unembed = base.patch_unembed
        self.pos_drop = base.pos_drop
        self.layers = base.layers
        self.norm = base.norm
        self.conv_after_body = base.conv_after_body
        self.condition_film = nn.ModuleList(nn.Linear(256, 2 * embed_dim) for _ in depths)
        for film in self.condition_film:
            # This candidate starts from scratch. Nonzero projection weights
            # let raw [g,t] affect clean at initialization; zero projections
            # plus zero realization heads would block the condition MLP's
            # first-step gradient entirely.
            nn.init.trunc_normal_(film.weight, std=0.02)
            nn.init.zeros_(film.bias)

        self.conv_before_upsample = base.conv_before_upsample
        self.clean_upsample = nn.ModuleList(
            getattr(base, f"conv_up{index + 1}")
            for index in range(int(math.log2(unshuffle_scale)))
        )
        self.conv_hr = base.conv_hr
        self.lrelu = base.lrelu
        self.clean_head = nn.Conv2d(64, 3, 3, padding=1)

        self.realization_project = nn.Conv2d(embed_dim, decoder_channels, 3, padding=1)
        self.realization_decoder = nn.ModuleList(
            _RealizationUpsample(decoder_channels)
            for _ in range(int(math.log2(unshuffle_scale)) + 1)
        )
        self.zernike_head = nn.Conv2d(decoder_channels, 36, 3, padding=1)
        self.tilt_head = nn.Conv2d(decoder_channels, 2, 3, padding=1)
        for head in (self.zernike_head, self.tilt_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.config = {
            "architecture": "pirn_swin_rgb_noise_bridge_v1",
            "source_module": "vendor.pirn.swinir.SwinIR",
            "image_size": image_size,
            "in_channels": 6,
            "embed_dim": embed_dim,
            "depths": list(depths),
            "num_heads": list(num_heads),
            "window_size": window_size,
            "mlp_ratio": float(mlp_ratio),
            "unshuffle_scale": unshuffle_scale,
            "decoder_channels": decoder_channels,
            "drop_path_rate": float(drop_path_rate),
            "use_checkpoint": False,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "patch_size": 1,
            "patch_norm": True,
            "ape": False,
            "resi_connection": "1conv",
            "condition": "raw_g_t_mlp_2_256_256_silu",
            "condition_injection": "pre_rstb_token_film_plus_xi_decoder_film",
            "condition_projection_init": "trunc_normal_std_0.02_bias_0",
            "input_transform": "none",
            "clean_decoder": "pirn_nearest_conv",
            "clean_output": "sigmoid_rgb",
            "realization_decoder": "project_nearest_conv_silu_film_no_skips",
            "realization_size": self.realization_size,
            "epsilon_zernike_layout": "B_36_H2_W2",
            "epsilon_tilt_layout": "B_H2_W2_2",
            "realization_head_initialization": "zeros",
            "initialization": "fresh_no_pretrained_migration",
            "input_dtype": "float32",
        }

    def _validate_inputs(self, image: torch.Tensor, condition: torch.Tensor) -> None:
        expected = (6, self.image_size, self.image_size)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected or image.shape[0] < 1:
            raise ValueError(f"Expected concat(Z,Y) [B,{','.join(map(str, expected))}], B >= 1")
        if condition.shape != (image.shape[0], 2):
            raise ValueError("Expected raw [g,t] with shape [B,2]")
        if image.dtype != torch.float32:
            raise ValueError("The candidate FP32 interface requires float32 image input")
        if condition.device != image.device:
            raise ValueError("image and condition must be on the same device")
        if condition.is_complex() or condition.dtype == torch.bool:
            raise ValueError("condition must contain real-valued g and t")
        if not torch.isfinite(image).all() or not torch.isfinite(condition).all():
            raise ValueError("Non-finite image or condition")
        g, t = condition[:, 0], condition[:, 1]
        if not ((g >= 1) & (g <= 4) & (g == g.round())).all():
            raise ValueError("g must be an original-valued integer in {1,2,3,4}")
        if not ((t >= 0) & (t <= 1)).all():
            raise ValueError("t must be in [0,1]")

    def _shared_features(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(image, condition)
        embedding = self.severity(condition)
        # Deliberately preserve values outside the RGB display range. There is
        # no additional normalization, padding, image clamp or condition scale.
        shallow = self.conv_first(image)
        spatial_size = tuple(shallow.shape[-2:])
        tokens = self.pos_drop(self.patch_embed(shallow))
        for layer, film in zip(self.layers, self.condition_film, strict=True):
            gamma, beta = film(embedding).chunk(2, dim=1)
            tokens = tokens * (1.0 + gamma[:, None, :]) + beta[:, None, :]
            tokens = layer(tokens, spatial_size)
        body = self.patch_unembed(self.norm(tokens), spatial_size)
        return shallow + self.conv_after_body(body), embedding

    def _clean(self, shared: torch.Tensor) -> torch.Tensor:
        value = self.conv_before_upsample(shared)
        for conv in self.clean_upsample:
            value = self.lrelu(conv(F.interpolate(value, scale_factor=2, mode="nearest")))
        return torch.sigmoid(self.clean_head(self.lrelu(self.conv_hr(value)))).float()

    def forward_clean(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return clean only, bypassing realization decoding without detaching."""
        shared, _ = self._shared_features(image, condition)
        return self._clean(shared)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> BridgePrediction:
        shared, embedding = self._shared_features(image, condition)
        clean = self._clean(shared)
        realization = self.realization_project(shared)
        for level in self.realization_decoder:
            realization = level(realization, embedding)
        epsilon_zernike = self.zernike_head(realization).float()
        epsilon_tilt = self.tilt_head(realization).permute(0, 2, 3, 1).float()
        return BridgePrediction(clean, epsilon_zernike, epsilon_tilt)

    def branch_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Expose a disjoint, exhaustive partition like the existing V4 model."""
        shared_modules = (
            self.severity, self.conv_first, self.patch_embed, self.patch_unembed,
            self.pos_drop, self.layers, self.norm, self.conv_after_body, self.condition_film,
        )
        clean_modules = (
            self.conv_before_upsample, self.clean_upsample, self.conv_hr, self.lrelu,
            self.clean_head,
        )
        realization_modules = (
            self.realization_project, self.realization_decoder,
            self.zernike_head, self.tilt_head,
        )
        return {
            name: [parameter for module in modules for parameter in module.parameters()]
            for name, modules in (
                ("shared", shared_modules), ("clean", clean_modules),
                ("realization", realization_modules),
            )
        }

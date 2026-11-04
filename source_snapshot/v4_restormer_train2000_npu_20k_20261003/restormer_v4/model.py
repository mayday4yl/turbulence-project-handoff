"""D2Turb's released Restormer hierarchy adapted to V4's dual-output contract.

The vendored Restormer supplies the patch embedding, all four encoder scales,
three decoder scales, skip concatenations and full-resolution refinement. Its
RGB residual output is not used: V4 supplies six-channel concat(Z, Y), raw [g,t],
a three-channel sigmoid clean prediction and two unbounded raw-xi fields.
Inputs retain the existing RGB uint8/255 units; simulated observations and
noisy Z may be outside [0,1]. Frozen train-population statistics are registered
as buffers for the external V4 bridge equations, not applied to this input.

FiLM acts immediately before each encoder/latent/decoder/refinement stage. Set
the eight ``condition_film`` projections to zero on an acceptance copy to match
the official Restormer's refinement features, with identical shared weights.
``forward_features`` exposes that tensor without detaching it; the transferred
modules retain their official names for feature hooks and gradient comparison.

All backbone blocks come from the pinned D2Turb inference release, including
torch.nn.LayerNorm with eps=1e-5 and attention temperature [1, heads, 1, 1].
This fresh V4 candidate does not reproduce D2Turb's complete method or training
pipeline and does not load pretrained weights.
At the default size, refined 256x256 features feed the RGB head directly and a
separate conditioned upsampling block produces simulator-native 512x512 xi.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from npu_runtime.backend import autocast

from physical_bridge.model import BridgePrediction, FiLMResBlock
from v4.model import JointCondition
from vendor.restormer.d2turb.code.models.restormer import Restormer


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _initialize_film(projection: nn.Linear) -> None:
    # Nonzero weights permit clean-loss gradients into the raw [g,t] MLP on
    # the initial pass, even while both realization heads are zero initialized.
    nn.init.trunc_normal_(projection.weight, std=0.02)
    nn.init.zeros_(projection.bias)


class _RealizationUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.block = FiLMResBlock(channels, channels, 256, groups=32)
        _initialize_film(self.block.film)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        return self.block(F.silu(self.conv(value)), embedding)


class RestormerNoiseBridge(nn.Module):
    """Released D2Turb Restormer hierarchy with V4 conditioning and output heads.

    Defaults preserve the public dim48/[4,6,6,8]/[1,2,4,8] layout. Smaller sizes
    and widths are supported for interface acceptance, not as evidence that the
    default model has completed full-resolution acceptance or training.
    """

    _OFFICIAL_SHARED_MODULES = (
        "embed_conv", "encoders", "downs", "ups", "reduces", "decoders", "refinement",
    )

    def __init__(
        self,
        stats: nn.Module,
        *,
        image_size: int = 256,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        decoder_channels: int = 64,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        image_size = _positive_int("image_size", image_size)
        dim = _positive_int("dim", dim)
        decoder_channels = _positive_int("decoder_channels", decoder_channels)
        num_refinement_blocks = _positive_int(
            "num_refinement_blocks", num_refinement_blocks
        )
        if image_size % 8:
            raise ValueError("Restormer's three downsampling levels require image_size divisible by 8")
        if dim % 2:
            raise ValueError("Restormer's first downsampling convolution requires even dim")
        num_blocks, heads = tuple(num_blocks), tuple(heads)
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("Restormer requires exactly four num_blocks and heads entries")
        for level, (blocks, head_count) in enumerate(zip(num_blocks, heads, strict=True)):
            _positive_int(f"num_blocks[{level}]", blocks)
            _positive_int(f"heads[{level}]", head_count)
            if dim * 2**level % head_count:
                raise ValueError(f"Restormer level {level + 1} channels must divide into its heads")
        if (isinstance(ffn_expansion_factor, bool)
                or not math.isfinite(ffn_expansion_factor)
                or ffn_expansion_factor <= 0
                or int(dim * ffn_expansion_factor) < 1):
            raise ValueError("ffn_expansion_factor must define a finite positive hidden dimension")
        if bias is not False:
            raise ValueError("D2Turb's released Restormer uses bias=False convolutions")
        if LayerNorm_type != "WithBias":
            raise ValueError("D2Turb's released Restormer uses torch.nn.LayerNorm with bias")
        if not isinstance(use_checkpoint, bool):
            raise ValueError("use_checkpoint must be boolean")
        self.use_checkpoint = use_checkpoint
        self.precision = "fp32"
        if not isinstance(stats, nn.Module):
            raise TypeError("stats must be the frozen RGB statistics nn.Module")
        if any(True for _ in stats.parameters()):
            raise ValueError("RGB statistics must contain buffers only, not model parameters")

        self.image_size = image_size
        self.realization_size = 2 * image_size
        self.stats = stats
        self.severity = JointCondition()

        channels = [dim * 2**level for level in range(4)]
        base = Restormer(
            num_blocks=list(num_blocks),
            num_heads=list(heads),
            channels=channels,
            num_refinement=num_refinement_blocks,
            expansion_factor=float(ffn_expansion_factor),
        )
        # The only shared-path convolution interface change is RGB -> concat(Z,Y).
        # Preserve the author's kernel, padding and lack of bias; initialize fresh.
        base.embed_conv = nn.Conv2d(6, dim, kernel_size=3, padding=1, bias=False)
        for name in self._OFFICIAL_SHARED_MODULES:
            setattr(self, name, getattr(base, name))
        # Do not register base.output or call base.forward: its RGB residual
        # addition does not implement the six-channel V4 input/output contract.

        stage_channels = {
            "encoder_level1": dim,
            "encoder_level2": 2 * dim,
            "encoder_level3": 4 * dim,
            "latent": 8 * dim,
            "decoder_level3": 4 * dim,
            "decoder_level2": 2 * dim,
            "decoder_level1": 2 * dim,
            "refinement": 2 * dim,
        }
        self.condition_film = nn.ModuleDict({
            name: nn.Linear(256, 2 * channels)
            for name, channels in stage_channels.items()
        })
        for projection in self.condition_film.values():
            _initialize_film(projection)

        self.clean_head = nn.Conv2d(2 * dim, 3, 3, padding=1)
        self.realization_project = nn.Conv2d(2 * dim, decoder_channels, 3, padding=1)
        self.realization_decoder = _RealizationUpsample(decoder_channels)
        self.zernike_head = nn.Conv2d(decoder_channels, 36, 3, padding=1)
        self.tilt_head = nn.Conv2d(decoder_channels, 2, 3, padding=1)
        for head in (self.zernike_head, self.tilt_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.config = {
            "architecture": "restormer_rgb_noise_bridge_v1",
            "source_module": "vendor.restormer.d2turb.code.models.restormer.Restormer",
            "source_commit": "bbc28441b7212cc55319124a081ed6c0001879ef",
            "source_arch_sha256": "70ba46b5f28b3d1e4780725bd742e7826642be0e488aac6e906891cdcb13331a",
            "source_scope": "D2Turb_released_inference_backbone_not_complete_method_or_training_pipeline",
            "image_size": image_size,
            "in_channels": 6,
            "dim": dim,
            "channels": channels,
            "num_blocks": list(num_blocks),
            "num_refinement_blocks": num_refinement_blocks,
            "heads": list(heads),
            "ffn_expansion_factor": float(ffn_expansion_factor),
            "bias": bias,
            "LayerNorm_type": LayerNorm_type,
            "actual_layer_norm": "torch.nn.LayerNorm_channels_elementwise_affine_with_bias",
            "layer_norm_eps": 1e-5,
            "attention_temperature_layout": "1_heads_1_1",
            "decoder_channels": decoder_channels,
            "use_checkpoint": use_checkpoint,
            "encoder_scales": [1, 2, 4, 8],
            "decoder_skip_scales": [4, 2, 1],
            "shared_feature_channels": 2 * dim,
            "shared_feature_size": image_size,
            "condition": "raw_g_t_mlp_2_256_256_silu",
            "condition_injection": "pre_encoder_latent_decoder_refinement_channel_film_plus_xi_block_film",
            "condition_film_channels": stage_channels,
            "condition_projection_init": "trunc_normal_std_0.02_bias_0",
            "input_transform": "none",
            "input_units": "RGB_uint8_div_255_with_unclamped_physical_and_noisy_states",
            "statistics_usage": "frozen_buffers_for_external_V4_bridge_not_input_normalization",
            "clean_output": "sigmoid_rgb",
            "official_output_residual": "omitted_for_V4_six_channel_to_RGB_contract",
            "realization_decoder": "refined_full_resolution_project_nearest_x2_conv_silu_film",
            "realization_size": self.realization_size,
            "epsilon_zernike_layout": "B_36_H2_W2",
            "epsilon_tilt_layout": "B_H2_W2_2",
            "realization_output": "unbounded_raw_gaussian_coordinates_before_g_scaling",
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

    def _film(self, name: str, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition_film[name](embedding).chunk(2, dim=1)
        return value * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def _stage(self, module, value):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, value, use_reentrant=False, preserve_rng_state=True)
        return module(value)

    def _backbone_features(self, image: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """Official feature path, with explicit pre-stage FiLM additions only."""
        encoded1 = self._stage(self.encoders[0], self._film("encoder_level1", self.embed_conv(image), embedding))
        encoded2 = self._stage(self.encoders[1], self._film("encoder_level2", self.downs[0](encoded1), embedding))
        encoded3 = self._stage(self.encoders[2], self._film("encoder_level3", self.downs[1](encoded2), embedding))
        latent = self._stage(self.encoders[3], self._film("latent", self.downs[2](encoded3), embedding))

        decoded3 = self.ups[0](latent)
        decoded3 = self.reduces[0](torch.cat([decoded3, encoded3], dim=1))
        decoded3 = self._stage(self.decoders[0], self._film("decoder_level3", decoded3, embedding))
        decoded2 = self.ups[1](decoded3)
        decoded2 = self.reduces[1](torch.cat([decoded2, encoded2], dim=1))
        decoded2 = self._stage(self.decoders[1], self._film("decoder_level2", decoded2, embedding))
        decoded1 = self.ups[2](decoded2)
        decoded1 = torch.cat([decoded1, encoded1], dim=1)
        decoded1 = self._stage(self.decoders[2], self._film("decoder_level1", decoded1, embedding))
        return self._stage(self.refinement, self._film("refinement", decoded1, embedding))

    def _shared_features(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(image, condition)
        embedding = self.severity(condition)
        # Preserve out-of-display-range physical pixels and raw g/t values.
        return self._backbone_features(image, embedding), embedding

    def forward_features(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Expose differentiable refined features for official-path parity checks."""
        with autocast(image.device, self.precision):
            shared, _ = self._shared_features(image, condition)
        return shared.float()

    def _clean(self, shared: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.clean_head(shared)).float()

    def forward_clean(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return RGB without realization decoding or gradient detachment."""
        with autocast(image.device, self.precision):
            shared, _ = self._shared_features(image, condition)
            return self._clean(shared)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> BridgePrediction:
        with autocast(image.device, self.precision):
            shared, embedding = self._shared_features(image, condition)
            clean = self._clean(shared)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                realization = checkpoint(self.realization_decoder, self.realization_project(shared),
                                         embedding, use_reentrant=False, preserve_rng_state=True)
            else:
                realization = self.realization_decoder(self.realization_project(shared), embedding)
            epsilon_zernike = self.zernike_head(realization).float()
            epsilon_tilt = self.tilt_head(realization).permute(0, 2, 3, 1).float()
        return BridgePrediction(clean, epsilon_zernike, epsilon_tilt)

    def branch_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Disjoint exhaustive registered model parameters; statistics are buffers."""
        shared_modules = tuple(getattr(self, name) for name in self._OFFICIAL_SHARED_MODULES)
        shared_modules += (self.severity, self.condition_film)
        return {
            name: [parameter for module in modules for parameter in module.parameters()]
            for name, modules in (
                ("shared", shared_modules),
                ("clean", (self.clean_head,)),
                ("realization", (
                    self.realization_project, self.realization_decoder,
                    self.zernike_head, self.tilt_head,
                )),
            )
        }

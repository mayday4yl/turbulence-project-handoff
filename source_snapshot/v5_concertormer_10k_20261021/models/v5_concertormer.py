"""Official Concertormer RGB residual restoration plus V5 time/raw-xi outputs.

The original full GoPro state dictionary is loaded strictly, including the
three auxiliary RGB heads. Those unused heads are frozen and not executed.
The final official RGB output is neither sigmoid-transformed nor clamped.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .official_concertormer import Concertormer
from .conv3d_compat import install_talking_backend

OFFICIAL_CONFIG = dict(img_channel=3, width=48, ffn_expansion_factor=2,
                       enc_blk_nums=[6, 8, 24], middle_blk_num=2,
                       dec_blk_nums=[24, 8, 8], enc_heads=[1, 2, 4],
                       middle_heads=8, dec_heads=[4, 2, 1])


@dataclass(frozen=True)
class RawXi:
    """Structurally identical to Simulator Realization, without import coupling."""
    epsilon_zernike: torch.Tensor
    epsilon_tilt: torch.Tensor


class TimeConditioning(nn.Module):
    def __init__(self, channels: list[int], hidden=128):
        super().__init__()
        # Fixed continuous frequencies, independent of NFE/G/gamma.
        self.register_buffer("frequencies", 2.0 ** torch.arange(8).float(), persistent=True)
        self.embedding = nn.Sequential(nn.Linear(17, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.films = nn.ModuleList([nn.Linear(hidden, 2 * c) for c in channels])
        for film in self.films:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def forward(self, t):
        phase = t[:, None] * self.frequencies[None, :] * torch.pi
        return self.embedding(torch.cat([t[:, None], phase.sin(), phase.cos()], dim=-1))

    def condition(self, x, embedding, index):
        scale, shift = self.films[index](embedding).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class V5Concertormer(nn.Module):
    def __init__(self, *, activation_checkpointing=False, checkpoint_policy="all",
                 realization_factory: Callable = RawXi, config=None,
                 talking_backend="conv3d"):
        super().__init__()
        self.config = dict(OFFICIAL_CONFIG if config is None else config)
        self.backbone = Concertormer(**self.config)
        self.set_talking_backend(talking_backend)
        width = self.config["width"]
        # Each encoder after its blocks, the bottleneck, and each decoder.
        self.time = TimeConditioning([width, width*2, width*4, width*8,
                                      width*4, width*2, width])
        self.raw_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                      nn.SiLU(), nn.Conv2d(width, 38*4, 3, padding=1),
                                      nn.PixelShuffle(2))
        self.activation_checkpointing = bool(activation_checkpointing)
        self.set_checkpoint_policy(checkpoint_policy)
        self.realization_factory = realization_factory
        # These three outputs have no path to final RGB in the official model.
        for parameter in self.backbone.endings.parameters():
            parameter.requires_grad_(False)
        self.pretrained_report = None

    def set_activation_checkpointing(self, enabled=True):
        self.activation_checkpointing = bool(enabled)

    def set_checkpoint_policy(self, policy):
        if policy not in ("all", "highest", "highest_half", "highest_third", "highres", "alternating", "highres_half_low"):
            raise ValueError(f"Unknown activation checkpoint policy: {policy}")
        # Execution-only setting: no parameters/buffers or state-dict changes.
        self.checkpoint_policy = policy

    def _checkpoint_block(self, index, x):
        highres = min(x.shape[-2:]) >= 128
        return (self.checkpoint_policy == "all" or
                (self.checkpoint_policy == "highest" and min(x.shape[-2:]) >= 256) or
                (self.checkpoint_policy == "highest_half" and min(x.shape[-2:]) >= 256 and index % 2 == 0) or
                (self.checkpoint_policy == "highest_third" and min(x.shape[-2:]) >= 256 and index % 3 == 0) or
                (self.checkpoint_policy == "highres" and highres) or
                (self.checkpoint_policy == "alternating" and index % 2 == 0) or
                (self.checkpoint_policy == "highres_half_low" and (highres or index % 2 == 0)))

    def set_talking_backend(self, backend):
        install_talking_backend(self.backbone, backend)
        self.talking_backend = backend

    def _blocks(self, blocks, x):
        for index, block in enumerate(blocks):
            if (self.activation_checkpointing and self.training and torch.is_grad_enabled()
                    and self._checkpoint_block(index, x)):
                # Non-reentrant: preserves Full-BPTT and works for D0's initial
                # input even when it does not itself require gradients.
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
        return x

    def forward(self, z, t):
        if z.ndim != 4 or z.shape[1] != 3:
            raise ValueError("V5 input must be RGB BCHW")
        if z.shape[-2:] != (256, 256):
            raise ValueError("Current V5 experiment requires 256x256 RGB and 512x512 raw xi")
        if t.ndim != 1 or t.shape[0] != z.shape[0]:
            raise ValueError("Global time t must have shape [B]")
        # Range/finite time validation belongs to the CPU schedule, avoiding
        # a device synchronization for every repeated model evaluation.
        inp = self.backbone.check_image_size(z)
        emb = self.time(t.to(device=z.device, dtype=z.dtype))
        x = self.backbone.intro(inp)
        inps = [F.interpolate(inp, scale_factor=0.5**(i+1), mode="bilinear")
                for i in range(self.backbone.n_layer)]
        encs = []
        for i, (encoder, down, intro, small) in enumerate(zip(
                self.backbone.encoders, self.backbone.downs, self.backbone.intros, inps)):
            x = self._blocks(encoder, x)
            x = self.time.condition(x, emb, i)
            encs.append(x)
            x = torch.cat([down(x), intro(small)], dim=1)
        x = self._blocks(self.backbone.middle_blks, x)
        x = self.time.condition(x, emb, 3)
        for i, (decoder, up, skip) in enumerate(zip(
                self.backbone.decoders, self.backbone.ups, reversed(encs))):
            x = self._blocks(decoder, torch.cat([up(x), skip], dim=1))
            x = self.time.condition(x, emb, 4+i)
        # Exact official residual restoration form and original RGB head.
        clean = (self.backbone.ending(x) + inp)[:, :, :256, :256]
        raw = self.raw_head(x)
        xi = self.realization_factory(
            epsilon_zernike=raw[:, :36],
            epsilon_tilt=raw[:, 36:].permute(0, 2, 3, 1))
        return clean, xi

    def optimizer_parameter_groups(self, backbone_lr=5e-5, new_lr=1e-4, weight_decay=1e-4):
        old, new = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (old if name.startswith("backbone.") else new).append(p)
        return [dict(params=old, lr=backbone_lr, initial_lr=backbone_lr,
                     weight_decay=weight_decay, role="official_backbone_and_rgb"),
                dict(params=new, lr=new_lr, initial_lr=new_lr,
                     weight_decay=weight_decay, role="new_time_and_raw_xi")]

    def load_official_checkpoint(self, path):
        path = Path(path)
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
        payload_key = "direct_state_dict"
        state = checkpoint_data
        if isinstance(checkpoint_data, dict):
            for candidate in ("params", "params_ema", "state_dict"):
                if candidate in checkpoint_data:
                    state = checkpoint_data[candidate]
                    payload_key = candidate
                    break
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Official checkpoint does not contain a tensor state dict")
        prefixes_removed = []
        for prefix in ("module.", "net_g."):
            if all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]: v for k, v in state.items()}
                prefixes_removed.append(prefix)
        incompatible = self.backbone.load_state_dict(state, strict=True)
        # Check exact values after loading, including original ending weights.
        loaded = self.backbone.state_dict()
        mismatched = [k for k in state if not torch.equal(state[k], loaded[k])]
        if mismatched:
            raise RuntimeError(f"Checkpoint values changed during loading: {mismatched}")
        self.pretrained_report = dict(
            path=str(path.resolve()), sha256=_sha256(path), payload_key=payload_key,
            prefixes_removed=prefixes_removed, loaded_tensors=len(state),
            loaded_elements=sum(v.numel() for v in state.values()),
            missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys),
            value_mismatches=mismatched, original_final_rgb_keys=[k for k in state if k.startswith("ending.")],
            frozen_auxiliary_rgb_keys=[k for k in state if k.startswith("endings.")],
            strict_load=True)
        return self.pretrained_report


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_model(*, official_checkpoint=None, activation_checkpointing=False, checkpoint_policy="all",
                realization_factory=RawXi, seed=20261021,
                talking_backend="conv3d"):
    """Build a reproducible U0 on CPU; save once and fork D0/R0 from that U0."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = V5Concertormer(activation_checkpointing=activation_checkpointing, checkpoint_policy=checkpoint_policy,
                              realization_factory=realization_factory,
                              talking_backend=talking_backend)
    if official_checkpoint is not None:
        model.load_official_checkpoint(official_checkpoint)
    return model

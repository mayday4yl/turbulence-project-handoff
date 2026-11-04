"""Clean-only D0: official RGB residual network, with no time or raw-xi modules."""
import torch
from torch import nn
from torch.nn import functional as F
from .official_concertormer import Concertormer
from .v5_concertormer import V5Concertormer, OFFICIAL_CONFIG


class CleanConcertormer(nn.Module):
    set_talking_backend = V5Concertormer.set_talking_backend
    set_checkpoint_policy = V5Concertormer.set_checkpoint_policy
    _checkpoint_block = V5Concertormer._checkpoint_block
    _blocks = V5Concertormer._blocks
    load_official_checkpoint = V5Concertormer.load_official_checkpoint

    def __init__(self, *, activation_checkpointing=False, checkpoint_policy='all',
                 talking_backend='conv2d_equivalent', config=None):
        super().__init__()
        self.config = dict(OFFICIAL_CONFIG if config is None else config)
        self.backbone = Concertormer(**self.config)
        self.set_talking_backend(talking_backend)
        self.activation_checkpointing = activation_checkpointing
        self.set_checkpoint_policy(checkpoint_policy)
        for p in self.backbone.endings.parameters():
            p.requires_grad_(False)
        self.pretrained_report = None

    def forward(self, observation):
        if observation.ndim != 4 or observation.shape[1:] != (3, 256, 256):
            raise ValueError('Clean-only D0 requires RGB [B,3,256,256]')
        inp = self.backbone.check_image_size(observation)
        x = self.backbone.intro(inp)
        inps = [F.interpolate(inp, scale_factor=0.5**(i+1), mode='bilinear')
                for i in range(self.backbone.n_layer)]
        encs = []
        for encoder, down, intro, small in zip(self.backbone.encoders, self.backbone.downs,
                                               self.backbone.intros, inps):
            x = self._blocks(encoder, x)
            encs.append(x)
            x = torch.cat([down(x), intro(small)], dim=1)
        x = self._blocks(self.backbone.middle_blks, x)
        for decoder, up, skip in zip(self.backbone.decoders, self.backbone.ups, reversed(encs)):
            x = self._blocks(decoder, torch.cat([up(x), skip], dim=1))
        return (self.backbone.ending(x) + inp)[:, :, :256, :256]

    def optimizer_parameter_groups(self, backbone_lr=5e-5, new_lr=None, weight_decay=1e-4):
        return [dict(params=[p for p in self.parameters() if p.requires_grad],
                     lr=backbone_lr, initial_lr=backbone_lr, weight_decay=weight_decay,
                     role='official_backbone_and_rgb')]


def build_clean_model(*, official_checkpoint=None, seed=20261021, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CleanConcertormer(**kwargs)
    if official_checkpoint is not None:
        model.load_official_checkpoint(official_checkpoint)
    return model

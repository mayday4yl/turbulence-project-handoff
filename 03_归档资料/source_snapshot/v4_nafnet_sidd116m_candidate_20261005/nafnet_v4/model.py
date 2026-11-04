"""Official NAFNet-SIDD width64 feature hierarchy with V4 conditioning/heads.

Keep all NAFBlocks, down/up operations and additive skips unchanged. V4 uses
Concat(Z,Y), raw [g,t] FiLM, sigmoid clean RGB and unbounded native raw xi.
Original RGB ending/input residual is replaced by the existing V4 head contract.
No SIDD denoising weights, PSNR loss, training recipe, or test-time TLC are used.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from physical_bridge.model import BridgePrediction, FiLMResBlock
from v4.model import JointCondition
from npu_runtime.backend import autocast
from .upstream import NAFNet, COMMIT


def initialize_film(projection):
    nn.init.trunc_normal_(projection.weight, std=.02)
    nn.init.zeros_(projection.bias)


class RealizationUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.block = FiLMResBlock(channels, channels, 256, groups=32)
        initialize_film(self.block.film)

    def forward(self, value, embedding):
        value = F.interpolate(value, scale_factor=2, mode='nearest')
        return self.block(F.silu(self.conv(value)), embedding)


class NAFNetNoiseBridge(nn.Module):
    _OFFICIAL_SHARED_MODULES = ('intro', 'encoders', 'downs', 'middle_blks', 'ups', 'decoders')

    def __init__(self, stats, *, image_size=256, width=64, enc_blk_nums=(2,2,4,8),
                 middle_blk_num=12, dec_blk_nums=(2,2,2,2), decoder_channels=64,
                 use_checkpoint=True):
        super().__init__()
        if (width, tuple(enc_blk_nums), middle_blk_num, tuple(dec_blk_nums)) != (64,(2,2,4,8),12,(2,2,2,2)):
            raise ValueError('This candidate is exactly official SIDD width64, not a reduced backbone')
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 16 or image_size % 16:
            raise ValueError('image_size must be a positive multiple of 16')
        if decoder_channels != 64 or not isinstance(use_checkpoint, bool):
            raise ValueError('Preserve V4 decoder width64 and boolean checkpoint setting')
        if not isinstance(stats, nn.Module) or any(True for _ in stats.parameters()):
            raise ValueError('Statistics must be frozen buffers only')
        self.image_size, self.realization_size = image_size, 2*image_size
        self.use_checkpoint, self.precision = use_checkpoint, 'fp32'
        self.stats, self.severity = stats, JointCondition()
        base = NAFNet(img_channel=6, width=width, enc_blk_nums=list(enc_blk_nums),
                      middle_blk_num=middle_blk_num, dec_blk_nums=list(dec_blk_nums))
        for name in self._OFFICIAL_SHARED_MODULES:
            setattr(self, name, getattr(base, name))
        stage_channels = {**{f'encoder{i}':width*2**i for i in range(4)}, 'middle':width*16,
                          **{f'decoder{i}':width*2**(3-i) for i in range(4)}}
        self.condition_film = nn.ModuleDict({name:nn.Linear(256,2*c) for name,c in stage_channels.items()})
        for projection in self.condition_film.values(): initialize_film(projection)
        self.clean_head = nn.Conv2d(width,3,3,padding=1)
        self.realization_project = nn.Conv2d(width,decoder_channels,3,padding=1)
        self.realization_decoder = RealizationUpsample(decoder_channels)
        self.zernike_head = nn.Conv2d(decoder_channels,36,3,padding=1)
        self.tilt_head = nn.Conv2d(decoder_channels,2,3,padding=1)
        for head in (self.zernike_head,self.tilt_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.config = {
            'architecture':'nafnet_sidd_width64_v4_v1', 'source_commit':COMMIT,
            'image_size':image_size, 'in_channels':6, 'width':width,
            'enc_blk_nums':list(enc_blk_nums),'middle_blk_num':middle_blk_num,'dec_blk_nums':list(dec_blk_nums),
            'decoder_channels':decoder_channels,'use_checkpoint':use_checkpoint,
            'official_nafblock':'unchanged_DW2_FFN2_SimpleGate_SCA_LayerNorm2d_eps1e-6',
            'official_beta_gamma_initialization':'zeros', 'test_time_TLC':False,
            'condition':'raw_g_t_mlp_2_256_256_silu', 'condition_film_channels':stage_channels,
            'condition_projection_init':'trunc_normal_std_0.02_bias_0',
            'condition_injection':'pre_encoder_middle_decoder_and_xi_block',
            'input_transform':'none', 'input_dtype':'float32',
            'input_units':'RGB_uint8_div_255_unclamped_states',
            'statistics_usage':'frozen_for_external_V4_bridge_not_input_normalization',
            'clean_output':'sigmoid_RGB', 'official_output_residual':'replaced_by_V4_heads',
            'realization_size':2*image_size, 'realization_output':'unbounded_raw_coordinates_before_g_scaling',
            'epsilon_zernike_layout':'B_36_H2_W2', 'epsilon_tilt_layout':'B_H2_W2_2',
            'realization_head_initialization':'zeros','initialization':'fresh_no_pretrained_weights',
        }

    def _validate_inputs(self, image, condition):
        if image.ndim != 4 or tuple(image.shape[1:]) != (6,self.image_size,self.image_size) or len(image)<1:
            raise ValueError('Expected nonempty Bx6ximage_sizeximage_size Concat(Z,Y)')
        if condition.shape != (len(image),2) or image.dtype != torch.float32 or condition.device != image.device:
            raise ValueError('Expected float32 image and raw [g,t] on the same device')
        if condition.is_complex() or condition.dtype == torch.bool:
            raise ValueError('Real raw g/t required')
        if not torch.isfinite(image).all() or not torch.isfinite(condition).all():
            raise ValueError('Non-finite model inputs')
        g,t = condition[:,0],condition[:,1]
        if not ((g>=1)&(g<=4)&(g==g.round())).all() or not ((t>=0)&(t<=1)).all():
            raise ValueError('g must be an integer in 1..4; t must be in 0..1')

    def _film(self, name, value, embedding):
        scale,shift = self.condition_film[name](embedding).chunk(2,dim=1)
        return value*(1+scale[:,:,None,None])+shift[:,:,None,None]

    def _stage(self, module, value):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module,value,use_reentrant=False,preserve_rng_state=True)
        return module(value)

    def _shared_features(self, image, condition):
        self._validate_inputs(image,condition)
        embedding = self.severity(condition)
        x,skips = self.intro(image),[]
        for i,(encoder,down) in enumerate(zip(self.encoders,self.downs)):
            x = self._stage(encoder,self._film(f'encoder{i}',x,embedding))
            skips.append(x); x = down(x)
        x = self._stage(self.middle_blks,self._film('middle',x,embedding))
        for i,(decoder,up,skip) in enumerate(zip(self.decoders,self.ups,reversed(skips))):
            x = self._stage(decoder,self._film(f'decoder{i}',up(x)+skip,embedding))
        return x,embedding

    def forward_features(self, image, condition):
        with autocast(image.device,self.precision):
            shared,_ = self._shared_features(image,condition)
        return shared.float()

    def forward_clean(self, image, condition):
        with autocast(image.device,self.precision):
            shared,_ = self._shared_features(image,condition)
            return torch.sigmoid(self.clean_head(shared)).float()

    def forward(self, image, condition):
        with autocast(image.device,self.precision):
            shared,embedding = self._shared_features(image,condition)
            clean = torch.sigmoid(self.clean_head(shared)).float()
            value = self.realization_project(shared)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                realization = checkpoint(self.realization_decoder,value,embedding,use_reentrant=False,preserve_rng_state=True)
            else:
                realization = self.realization_decoder(value,embedding)
            z = self.zernike_head(realization).float()
            tilt = self.tilt_head(realization).permute(0,2,3,1).float()
        return BridgePrediction(clean,z,tilt)

    def branch_parameters(self):
        modules = tuple(getattr(self,name) for name in self._OFFICIAL_SHARED_MODULES)
        modules += (self.severity,self.condition_film)
        return {'shared':[p for m in modules for p in m.parameters()],
                'clean':list(self.clean_head.parameters()),
                'realization':[p for m in (self.realization_project,self.realization_decoder,
                                           self.zernike_head,self.tilt_head) for p in m.parameters()]}

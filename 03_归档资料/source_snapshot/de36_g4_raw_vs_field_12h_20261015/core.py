from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from v4.model import NoiseBridgeUNet
from physical_bridge.simulator import EffectiveState, Realization, RenderResult
from common import CONFIG

class Prediction(NamedTuple):
    clean: torch.Tensor
    alpha_or_z: torch.Tensor
    u_or_t: torch.Tensor

class PilotUNet(NoiseBridgeUNet):
    """Identical learned layers in both arms; native Raw512 vs Field256 readout."""
    def __init__(self, arm, scales=None):
        super().__init__(None, **CONFIG['network'])
        if arm not in ('Raw', 'Field'): raise ValueError(arm)
        self.arm = arm
        if scales is not None:
            self.register_buffer('alpha_scale', torch.tensor(scales['alpha_rms']).reshape(1,36,1,1))
            self.register_buffer('u_scale', torch.tensor(scales['u_rms']).reshape(1,1,1,2))

    def features(self, image, condition):
        if image.dtype != torch.float32 or image.shape[1] != 6: raise ValueError('FP32 six-channel input required')
        embedding = self.severity(condition)
        value = self.input(image); skips = []
        for i, level in enumerate(self.encoder):
            value = level(value, embedding); skips.append(value)
            if i < len(self.downsample): value = self.downsample[i](value)
        for block in self.middle: value = block(value, embedding)
        return value, skips, embedding

    def clean_only(self, image, condition):
        value, skips, embedding = self.features(image, condition)
        hidden = self.clean_decoder(value, skips, embedding)
        return torch.sigmoid(self.clean_head(hidden)).float()

    def forward(self, image, condition):
        value, skips, embedding = self.features(image, condition)
        clean = torch.sigmoid(self.clean_head(self.clean_decoder(value, skips, embedding))).float()
        state = self.realization_decoder(value, skips, embedding)
        if self.arm == 'Raw': state = F.interpolate(state, scale_factor=2.0, mode='nearest')
        state = self.realization_block(self.realization_up(state), embedding)
        z = self.zernike_head(state).float()
        u = self.tilt_head(state).permute(0,2,3,1).float()
        if self.arm == 'Field':
            z = z * self.alpha_scale; u = u * self.u_scale
        return Prediction(clean, z, u)

def render_field(simulator, prepared, alpha_ref, u_ref, g):
    batch = alpha_ref.shape[0]
    amplitude = simulator._strength_tensor(g, batch).pow(5.0/6.0)
    alpha = alpha_ref * amplitude[:,None,None,None]
    u = u_ref * amplitude[:,None,None,None]
    state = EffectiveState(alpha, u)
    weights = simulator.mapping(alpha.permute(0,2,3,1).reshape(batch,65536,36)).reshape(batch,256,256,100).permute(0,3,1,2)
    blur = torch.sum(weights[:,None] * prepared.dictionary_images, dim=2) + prepared.image_mean
    denominator = torch.tensor((simulator.image_size,simulator.image_size),dtype=torch.float32,device=simulator.device_spec)-1.0
    flow = 2.0*(simulator.pixel_position + u)/denominator - 1.0
    output = F.grid_sample(blur, flow, mode='bilinear', padding_mode='border', align_corners=False)
    return RenderResult(output, blur, flow, weights, state)

def render_prediction(simulator, prepared, prediction, arm, g):
    if arm == 'Raw':
        return simulator.render_prepared(prepared, Realization(prediction.alpha_or_z,prediction.u_or_t),g)
    return render_field(simulator,prepared,prediction.alpha_or_z,prediction.u_or_t,g)

def l1(a,b): return (a.float()-b.float()).abs().flatten(1).mean(1)

def rollout(model, simulator, clean, y, retain=False):
    state=y; predictions=[]; transitions=[]; lc=[]; lr=[]
    for g in (4,3,2,1):
        p=model(torch.cat((state,y),1),y.new_tensor([[float(g),0.]]).expand(len(y),-1))
        if retain:
            for tensor in p: tensor.retain_grad()
        prepared=simulator.prepare_image(p.clean)
        reconstruction=render_prediction(simulator,prepared,p,model.arm,4.)
        lc.append(l1(p.clean,clean));lr.append(l1(reconstruction.output,y));predictions.append(p)
        if g>1:
            state=render_prediction(simulator,prepared,p,model.arm,float(g-1)).output
            transitions.append(state)
    lc=torch.stack(lc);lr=torch.stack(lr)
    return {'loss':(lc+.2*lr).mean(),'lc':lc,'lr':lr,'predictions':predictions,'transitions':transitions}

def optimizer(model):
    # Same bias/GroupNorm exclusion convention as existing V4 U-Net optimizer.
    excluded={id(p) for m in model.modules() if isinstance(m,(nn.LayerNorm,nn.GroupNorm)) for p in m.parameters()}
    decay=[]; no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad: (no_decay if name.endswith('bias') or id(p) in excluded else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':1e-4},{'params':no_decay,'weight_decay':0.}],
                            lr=1e-4,betas=(.9,.999),eps=1e-8,foreach=False)

def parameter_counts(model):
    branches=model.branch_parameters()
    return {'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            **{k:sum(p.numel() for p in v) for k,v in branches.items()},
            'predicted_state_shape':{'alpha_or_z':[1,36,512 if model.arm=='Raw' else 256,512 if model.arm=='Raw' else 256],
                                     'u_or_t':[1,512 if model.arm=='Raw' else 256,512 if model.arm=='Raw' else 256,2]}}

"""V6 Spec 1.0 plus explicitly approved Q1; official DiffUIR/physics unchanged."""
from contextlib import contextmanager
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from src.model import SinusoidalPosEmb
from method_impl.d0 import D0Restorer, derive_seed, isolated_rng
from scripts.runtime_utils import ROOT, sha

SIM_ROOT=ROOT.parent/'v5_v1_simulator_20261020'
if str(SIM_ROOT) not in sys.path:sys.path.insert(0,str(SIM_ROOT))
from simulator_v5 import build_simulator, Realization


class PhysicalTimeEmbedding(nn.Module):
    def __init__(self, original):
        super().__init__();self.diffusion=original
        self.physical=nn.Sequential(SinusoidalPosEmb(64),nn.Linear(64,256),nn.SiLU(),nn.Linear(256,256))
        nn.init.zeros_(self.physical[-1].weight);nn.init.zeros_(self.physical[-1].bias)
        self.current_t=None

    def forward(self, diffusion_time):
        t=self.current_t
        if t is None:raise RuntimeError('Physical time must be supplied explicitly')
        return self.diffusion(diffusion_time)+t[:,None]*self.physical(1000*t)


class XiResidual(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv1=nn.Conv2d(incoming,outgoing,3,padding=1)
        self.norm1=nn.GroupNorm(8,outgoing)
        self.conv2=nn.Conv2d(outgoing,outgoing,3,padding=1)
        self.norm2=nn.GroupNorm(8,outgoing)
        self.skip=nn.Identity() if incoming==outgoing else nn.Conv2d(incoming,outgoing,1)
        self.modulation=nn.Linear(128,2*outgoing)
        nn.init.zeros_(self.modulation.weight);nn.init.zeros_(self.modulation.bias)

    def forward(self,x,embedding):
        scale,shift=self.modulation(embedding).chunk(2,dim=1)
        h=self.norm1(self.conv1(x))
        h=F.silu(h*(1+scale[:,:,None,None])+shift[:,:,None,None])
        h=F.silu(self.norm2(self.conv2(h)))
        return h+self.skip(x)


class RawXiHead(nn.Module):
    """Two two-convolution residual blocks/scale; concat encoder skips.

    Engineering layout choices are explicit: 3x3 stem, stride2 3x3 down,
    bilinear align_corners=False then 3x3 width reduction, concat skip and
    first residual 1x1 shortcut when widths differ. No extra output gate.
    """
    def __init__(self):
        super().__init__();widths=(32,64,128,256)
        self.time=nn.Sequential(SinusoidalPosEmb(64),nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128))
        self.stem=nn.Conv2d(6,32,3,padding=1)
        self.encoder=nn.ModuleList([nn.ModuleList([XiResidual(c,c),XiResidual(c,c)]) for c in widths])
        self.down=nn.ModuleList([nn.Conv2d(a,b,3,stride=2,padding=1) for a,b in zip(widths[:-1],widths[1:])])
        self.up=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.decoder=nn.ModuleList([nn.ModuleList([XiResidual(2*c,c),XiResidual(c,c)]) for c in widths[-2::-1]])
        self.expand=nn.Sequential(nn.Conv2d(32,128,3,padding=1),nn.PixelShuffle(2))
        self.high_resolution=nn.ModuleList([XiResidual(32,32),XiResidual(32,32)])
        self.output=nn.Conv2d(32,38,1)
        nn.init.normal_(self.output.weight,0,.02);nn.init.zeros_(self.output.bias)

    def forward(self,observation,clean,physical_t):
        embedding=self.time(1000*physical_t)
        x=self.stem(torch.cat((2*observation-1,2*clean-1),dim=1));skips=[]
        for i,blocks in enumerate(self.encoder):
            for block in blocks:x=block(x,embedding)
            if i<3:skips.append(x);x=self.down[i](x)
        for conv,blocks in zip(self.up,self.decoder):
            skip=skips.pop();x=conv(F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False))
            x=torch.cat((x,skip),dim=1)
            for block in blocks:x=block(x,embedding)
        x=self.expand(x)
        for block in self.high_resolution:x=block(x,embedding)
        return self.output(x)


class ABRestorer(nn.Module):
    def __init__(self,*,d0_checkpoint=None,ab_state=None):
        super().__init__()
        assert (ROOT/'authority/V6_METHOD_Q1_XI_TIME_APPROVED_20261027.md').is_file()
        assert sha(SIM_ROOT/'simulator_v5.py')=='0f2f6f54a578b3c36cb69861815c7eee5476f75ccd91cac863bd001345bd9636'
        self.restorer=D0Restorer(ROOT.parent/'v5_diffuir_l_stage1_20261026/weights/model-300.pt')
        if d0_checkpoint is None:
            import json
            pointer=json.loads((ROOT/'distributed/D0_formal10000_01/BEST.json').read_text())
            d0_checkpoint=pointer['path'];assert sha(d0_checkpoint)==pointer['sha256']
        payload=torch.load(d0_checkpoint,map_location='cpu',weights_only=False)
        assert payload['update']==10000 and payload['schema']=='V6_D0_FULL_v1'
        self.restorer.network.load_state_dict(payload['ema']['model'],strict=True)
        self.initialization={'source':'D0_best_EMA','path':str(d0_checkpoint),'sha256':sha(d0_checkpoint)}
        original=self.restorer.network.unet0.time_mlp
        self.restorer.network.unet0.time_mlp=PhysicalTimeEmbedding(original)
        self.core=nn.ModuleDict({'network':self.restorer.network,'xi_head':RawXiHead()})
        if ab_state is not None:self.core.load_state_dict(ab_state,strict=True)
        self.simulator=build_simulator(device='cpu').eval().requires_grad_(False)
        assert self.simulator.smooth_psf and self.simulator.pixel_center_grid and self.simulator.tilt_fft=='legacy_full'

    @property
    def adapter(self):return self.core['network'].unet0.time_mlp.physical

    @contextmanager
    def physical_time(self,t):
        module=self.core['network'].unet0.time_mlp
        old=module.current_t;module.current_t=t
        try:yield
        finally:module.current_t=old

    def sample(self,x,t,*,seed,differentiable=True):
        with self.physical_time(t):
            return self.restorer.sample_clean(x,seed=seed,differentiable=differentiable)

    def residual(self,x,c,t,*,seed):
        with self.physical_time(t):return self.restorer.residual_loss(x,c,seed=seed)

    def predict(self,x,t,*,seed,differentiable=True):
        c=self.sample(x,t,seed=seed,differentiable=differentiable)
        raw=self.core['xi_head'](x,c,t)
        return c,raw

    def render(self,c,raw,g):
        assert raw.shape==(len(c),38,512,512)
        xi=Realization(raw[:,:36],raw[:,36:].permute(0,2,3,1))
        return self.simulator.render(c,xi,g).output

    def render_gt(self,c,raw_z,raw_t,g):
        with torch.no_grad():return self.simulator.render(c,Realization(raw_z,raw_t),g).output

    def parameter_groups(self,stage):
        adapter_ids={id(p) for p in self.adapter.parameters()}
        main=[p for p in self.core['network'].parameters() if id(p) not in adapter_ids]
        return [{'params':main,'name':'backbone','lr':1e-5 if stage=='A' else 5e-6},
                {'params':list(self.adapter.parameters()),'name':'physical_adapter','lr':5e-5 if stage=='A' else 1e-5},
                {'params':list(self.core['xi_head'].parameters()),'name':'xi_head','lr':5e-5 if stage=='A' else 1e-5}]

    def forward(self,y4,c,raw_z,raw_t,*,stage,seed,diagnostic=False):
        random=protocol_randomness(len(c),stage,seed)
        device=c.device
        if stage=='A':
            g=random['g'].to(device);t=1-g/4
            # Ground-truth synthesis only; no GT enters restoration or xi head.
            x=self.render_gt(c,raw_z,raw_t,g)
            residual,tau=self.residual(x,c,t,seed=derive_seed(seed,'q'))
            predicted,raw=self.predict(x,t,seed=derive_seed(seed,'s',0))
            reconstruction=self.render(predicted,raw,g)
            lc=(predicted-c).abs().mean();lp=(reconstruction-x).abs().mean()
            return {'loss':residual+.9*lc+.1*lp,'residual_loss':residual,'clean_loss':lc,'physical_loss':lp,
                    'clean':[predicted],'raw':[raw],'states':[x],'physical_time':t[:,None],
                    'g':g,'tau':tau,'seed':seed,'sampling_seeds':[derive_seed(seed,'s',0)],'reconstruction':[reconstruction]}
        assert stage=='B'
        times=random['times'].to(device);severity=4*(1-times)
        states=[y4];cleans=[];raws=[];physics=[];seeds=[]
        for i in range(4):
            s=derive_seed(seed,'s',i);seeds.append(s)
            prediction,raw=self.predict(states[-1],times[:,i],seed=s)
            cleans.append(prediction);raws.append(raw)
            physics.append(self.render(prediction,raw,4.))
            if i<3:states.append(self.render(prediction,raw,severity[:,i+1]))
        j=random['tf_index'].to(device);tf_t=times.gather(1,j[:,None]).flatten();tf_g=4*(1-tf_t)
        tf_x=self.render_gt(c,raw_z,raw_t,tf_g)
        residual,tau=self.residual(tf_x,c,tf_t,seed=derive_seed(seed,'tf_q'))
        lc=torch.stack([(x-c).abs().mean() for x in cleans]).mean()
        lp=torch.stack([(x-y4).abs().mean() for x in physics]).mean()
        return {'loss':.9*lc+.1*lp+.1*residual,'residual_loss':residual,'clean_loss':lc,'physical_loss':lp,
                'clean':cleans,'raw':raws,'states':states,'physical_time':times,'tf_index':j,'tf_time':tf_t,
                'tau':tau,'seed':seed,'sampling_seeds':seeds,'reconstruction':physics,'tf_observation':tf_x}


def protocol_randomness(batch,stage,seed):
    generator=torch.Generator(device='cpu').manual_seed(derive_seed(seed,'physical_protocol'))
    if stage=='A':
        high=torch.rand(batch,generator=generator)<.25
        g=torch.rand(batch,generator=generator)*4
        return {'g':torch.where(high,4.,g)}
    assert stage=='B'
    middle=torch.rand(batch,3,generator=generator).sort(dim=1).values
    valid=lambda a:(a[:,0]>0)&(a[:,1]>a[:,0])&(a[:,2]>a[:,1])&(a[:,2]<1)
    # FP32 can produce exact endpoints/ties; resample these measure-zero cases.
    while not bool(valid(middle).all()):
        bad=~valid(middle);middle[bad]=torch.rand(int(bad.sum()),3,generator=generator).sort(dim=1).values
    return {'times':torch.cat((torch.zeros(batch,1),middle,torch.ones(batch,1)),dim=1),
            'tf_index':torch.randint(0,4,(batch,),generator=generator)}

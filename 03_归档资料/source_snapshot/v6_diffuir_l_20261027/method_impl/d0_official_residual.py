"""U14927 continuation: official normalized residual L1, one training NFE.

Only the two training methods differ from the frozen joint-loss D0 wrapper.
Initialization, schedules, native p_losses/q_sample, and S3 inference are inherited
unchanged. Inputs stay in dataset RGB coordinates: p_losses normalizes ONCE.
"""
import torch
from method_impl.d0 import D0Restorer,derive_seed,isolated_rng


class D0OfficialResidualRestorer(D0Restorer):
    def residual_loss(self, observation, clean, *, seed):
        with isolated_rng(seed,observation.device):
            tau=torch.randint(0,1000,(len(observation),),device=observation.device).long()
            eps=torch.randn_like(clean)
            values=self.native.p_losses([clean,observation,['blur']*len(clean)],tau,noise=eps)
        if len(values)!=1:raise RuntimeError('Official pred_res must have exactly one loss')
        return values[0],tau

    def forward(self, observation, clean, *, seed):
        if observation.shape!=clean.shape or observation.shape[1:]!=(3,256,256):
            raise ValueError('RGB256 pairs required')
        if observation.dtype!=torch.float32 or clean.dtype!=torch.float32:
            raise ValueError('FP32 only')
        residual,tau=self.residual_loss(observation,clean,seed=derive_seed(seed,'q'))
        return {'loss':residual,'residual_loss':residual,'tau':tau}

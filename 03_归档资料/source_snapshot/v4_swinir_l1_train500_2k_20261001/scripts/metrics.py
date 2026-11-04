"""Frozen full-float RGB metrics; visualization never enters this module."""
import math
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity

SPEC = {
    'PSNR': {'implementation':'original v4.core.mse then -10*log10', 'data_range':1.0,
             'reduction':'per-image mean over RGB/H/W', 'arithmetic':'torch FP32 on CUDA'},
    'SSIM': {'implementation':'skimage.metrics.structural_similarity', 'package':'scikit-image==0.25.2',
             'data_range':1.0, 'channel_axis':2, 'gaussian_weights':True, 'win_size':11,
             'sigma':1.5, 'use_sample_covariance':False, 'K1':0.01, 'K2':0.03,
             'arithmetic':'float64 CPU; mean of RGB channel scores',
             'border':'library crops 5-pixel window border from SSIM map; no input crop'},
    'LPIPS': {'implementation':'official lpips.LPIPS', 'package':'lpips==0.1.4',
              'net':'alex', 'version':'0.1', 'pretrained':True, 'pnet_rand':False,
              'lpips':True, 'spatial':False, 'eval_mode':True, 'arithmetic':'FP32 CUDA',
              'normalization':'normalize=True, exactly 2*x-1 then official v0.1 ScalingLayer',
              'metric_batch':1},
    'preprocessing': {'resolution':[256,256], 'channels':'RGB', 'units':'original floating RGB',
                      'clamp':False, 'resize':False, 'quantization':False,
                      'per_image_standardization':False, 'PNG_used_for_metrics':False},
    'directions': {'psnr':'higher','ssim':'higher','lpips':'lower'},
}

class Metrics:
    def __init__(self, device):
        self.device=device
        self.lpips=lpips.LPIPS(net='alex', version='0.1', pretrained=True, lpips=True,
                               pnet_rand=False, pnet_tune=False, spatial=False,
                               eval_mode=True, verbose=False).to(device).eval()
        self.lpips.requires_grad_(False)

    def __call__(self, output, clean, *, permit_identical=False):
        if output.shape!=clean.shape or output.dtype!=torch.float32 or clean.dtype!=torch.float32:
            raise ValueError('Expected equal float32 RGB batches')
        if output.ndim!=4 or output.shape[1]!=3 or not torch.isfinite(output).all() or not torch.isfinite(clean).all():
            raise ValueError('Invalid RGB metrics input')
        # This is exactly the original per-image GPU MSE operation/order.
        errors=(output-clean).square().flatten(1).mean(1).cpu().tolist()
        out=output.detach().cpu().numpy(); ref=clean.detach().cpu().numpy()
        rows=[]
        with torch.no_grad(), torch.autocast(device_type='cuda',enabled=False):
            for b,error in enumerate(errors):
                if error<=0 and not permit_identical: raise FloatingPointError('Nonpositive MSE in formal result')
                perceptual=float(self.lpips(output[b:b+1],clean[b:b+1],normalize=True).item())
                structural=float(structural_similarity(
                    np.moveaxis(ref[b].astype(np.float64),0,2),np.moveaxis(out[b].astype(np.float64),0,2),
                    data_range=1.0,channel_axis=2,gaussian_weights=True,win_size=11,sigma=1.5,
                    use_sample_covariance=False,K1=0.01,K2=0.03))
                if not math.isfinite(structural) or not math.isfinite(perceptual):
                    raise FloatingPointError('Nonfinite perceptual metric')
                rows.append({'mse':error,'psnr':None if error==0 else -10*math.log10(error),
                             'ssim':structural,'lpips':perceptual,
                             'output_min':float(out[b].min()),'output_max':float(out[b].max())})
        return rows

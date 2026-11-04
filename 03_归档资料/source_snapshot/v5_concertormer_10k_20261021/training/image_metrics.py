"""Official LPIPS-Alex v0.1 and RGB Gaussian SSIM, with no image clipping."""
from functools import lru_cache
import math
import numpy as np
import torch


@lru_cache(maxsize=2)
def perceptual_model(device):
    import lpips
    model = lpips.LPIPS(net='alex', version='0.1', pretrained=True, lpips=True,
        pnet_rand=False, pnet_tune=False, spatial=False, eval_mode=True, verbose=False)
    return model.to(device).eval().requires_grad_(False)


def image_metrics(output, target):
    from skimage.metrics import structural_similarity
    with torch.no_grad(), torch.autocast(device_type=output.device.type, enabled=False):
        perceptual = perceptual_model(str(output.device))(output.float(), target.float(), normalize=True).flatten().cpu().tolist()
    predictions = output.detach().float().cpu().numpy()
    references = target.detach().float().cpu().numpy()
    structural = [float(structural_similarity(np.moveaxis(y.astype(np.float64), 0, 2),
        np.moveaxis(x.astype(np.float64), 0, 2), data_range=1., channel_axis=2,
        gaussian_weights=True, win_size=11, sigma=1.5, use_sample_covariance=False,
        K1=.01, K2=.03)) for x, y in zip(predictions, references)]
    if not all(math.isfinite(x) for x in perceptual + structural):
        raise FloatingPointError('Nonfinite SSIM or LPIPS')
    return structural, perceptual


SPEC = {'psnr': 'mean per-image RGB PSNR, range 1, no clamp/crop/resize',
        'ssim': 'skimage RGB; Gaussian 11x11 sigma1.5; population covariance; range1; float64 CPU',
        'lpips': 'official LPIPS AlexNet v0.1 pretrained; normalize=True; FP32; no clamp/resize',
        'loss': 'mean across stages of 0.9 clean MAE + 0.1 original-G4 reprojection MAE'}

"""Frozen full-image metrics. CPU FP32 LPIPS avoids accelerator-specific kernels."""
import math
import numpy as np
import torch

SPEC = {
    'PSNR': {'formula':'-10*log10(mean((output-clean)^2))','data_range':1.,'arithmetic':'FP32 CPU','reduction':'per-image RGB/H/W'},
    'SSIM': {'implementation':'skimage.metrics.structural_similarity','data_range':1.,'channel_axis':2,'gaussian_weights':True,'win_size':11,'sigma':1.5,'use_sample_covariance':False,'K1':.01,'K2':.03,'arithmetic':'float64 CPU'},
    'LPIPS': {'implementation':'lpips.LPIPS','net':'alex','version':'0.1','pretrained':True,'pnet_rand':False,'normalize':True,'spatial':False,'eval_mode':True,'arithmetic':'FP32 CPU','batch':1},
    'preprocessing': {'RGB':True,'range':'unclamped original FP32 RGB','crop':False,'resize':False,'quantization':False,'display_images_used':False}}

class Metrics:
    def __init__(self):
        import lpips
        from skimage.metrics import structural_similarity
        self.structural_similarity=structural_similarity
        self.lpips=lpips.LPIPS(net='alex',version='0.1',pretrained=True,lpips=True,pnet_rand=False,pnet_tune=False,spatial=False,eval_mode=True,verbose=False).cpu().eval()
        self.lpips.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, output, clean):
        output=output.detach().float().cpu();clean=clean.detach().float().cpu()
        if output.shape!=clean.shape or output.ndim!=4 or output.shape[1]!=3 or not torch.isfinite(output).all() or not torch.isfinite(clean).all():
            raise ValueError('Invalid full-float RGB metric inputs')
        answer=[]
        for i in range(len(output)):
            error=float((output[i]-clean[i]).square().mean())
            if error<=0:raise FloatingPointError('Nonpositive MSE; explicit identical-image diagnostic required')
            out=np.moveaxis(output[i].numpy().astype(np.float64),0,2)
            ref=np.moveaxis(clean[i].numpy().astype(np.float64),0,2)
            row={'mse':error,'psnr':-10*math.log10(error),
                 'ssim':float(self.structural_similarity(ref,out,data_range=1.,channel_axis=2,gaussian_weights=True,win_size=11,sigma=1.5,use_sample_covariance=False,K1=.01,K2=.03)),
                 'lpips':float(self.lpips(output[i:i+1],clean[i:i+1],normalize=True).item())}
            if not all(math.isfinite(x) for x in row.values()):raise FloatingPointError('Nonfinite image metric')
            answer.append(row)
        return answer

"""Only checkpoint admission/loading changes; mathematical ABRestorer stays immutable."""
import torch
from method_impl.ab import ABRestorer
from scripts.runtime_utils import ROOT,sha

def build_model(stage,checkpoint):
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if stage=='A':
        assert p['schema']=='V6_D0_FULL_v1' and p['update']>=10000
        model=ABRestorer()
        # Expose the original diffusion embedding while strictly loading D0 EMA.
        # Adapter parameters remain at their approved independent initialization.
        network=model.core['network'];time=network.unet0.time_mlp
        network.unet0.time_mlp=time.diffusion
        try:network.load_state_dict(p['ema']['model'],strict=True)
        finally:network.unet0.time_mlp=time
        source='continued_D0_best_EMA'
    else:
        assert stage=='B' and p['schema']=='V6_AB_FULL_v1' and p['config']['stage']=='A'
        model=ABRestorer(ab_state=p['ema']['model']);source='A_best_EMA'
    model.initialization={'source':source,'path':str(checkpoint),'sha256':sha(checkpoint),'update':p['update'],'factory_sha256':sha(__file__)}
    return model

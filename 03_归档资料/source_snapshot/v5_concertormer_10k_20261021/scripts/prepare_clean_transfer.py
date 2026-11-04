"""Fresh, reproducible V5 adapters on the finally selected clean D0 best."""
import argparse
from pathlib import Path
import torch
from models import build_model
from models.clean_concertormer import build_clean_model
from runtime.environment import configure
from runtime.controller import atomic_json, file_sha
from data_pipeline.dataset import V5Dataset
from training.core import network_context
from training.state import atomic_torch_save


def transfer(best, *, seed=20261021, config=None):
    from models.v5_concertormer import V5Concertormer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = V5Concertormer(config=config, talking_backend='conv2d_equivalent')
    state = best['model']
    if any(not key.startswith('backbone.') for key in state):
        raise ValueError('Parent is not a clean-only D0 state')
    model.backbone.load_state_dict({k[len('backbone.'):]: v for k,v in state.items()}, strict=True)
    if any(torch.count_nonzero(p).item() for film in model.time.films for p in film.parameters()):
        raise RuntimeError('Time modulation must start neutral')
    return model


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','output','report','dataset'):p.add_argument('--'+name, required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if (parent['method']!='D0' or parent['completed_update']!=parent['best_update']
            or parent['config'].get('objective')!='clean_only'):
        raise ValueError('Must inherit the selected clean-only D0 best')
    model=transfer(parent)
    clean=build_clean_model();clean.load_state_dict(parent['model'],strict=True)
    configure('npu:0');model.to('npu:0').eval();clean.to('npu:0').eval()
    dataset=V5Dataset(a.dataset,'val',require_ready=True,allow_test=False)
    x=torch.stack([dataset[i]['observation'] for i in (0,4)]).to('npu:0')
    errors={}
    for precision in ('fp32','bf16'):
        with torch.no_grad(),network_context('npu:0',precision):
            expected=clean(x);actual,_=model(x,torch.zeros(2,device='npu:0'))
        error=(expected.float()-actual.float()).abs().max().item()
        if not torch.allclose(expected.float(),actual.float(),rtol=1e-5,atol=1e-6):
            raise RuntimeError(f'Neutral mapping parity failed: {precision}: {error}')
        errors[precision]=error
    output={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'method':'D0','completed_update':parent['completed_update'],'best_update':parent['best_update'],
        'counters':parent['counters'],'provenance':{'parent_sha256':file_sha(a.checkpoint),
        'new_adapter_seed':20261021,'backbone_rgb_exact_copy':True,'neutral_time':True,
        'raw_head':'seeded PyTorch default Conv2d initialization'},'optimizer_reset':True}
    atomic_torch_save(output,a.output)
    atomic_json(a.report,{'status':'PASS','parent_sha256':file_sha(a.checkpoint),
        'initial_sha256':file_sha(a.output),'first_clean_max_abs_errors':errors,
        'rtol':1e-5,'atol':1e-6,'seed':20261021,'optimizer_and_scheduler':'new 10k plan'})


if __name__=='__main__':main()

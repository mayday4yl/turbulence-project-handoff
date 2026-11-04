"""Full A/B snapshots, keeping original D0 schema and files immutable."""
import torch
from method_impl.training_io import cpu_tree

def snapshot(model,opt,ema,*,update,cursor,rng,config,identity):
    return {'schema':'V6_AB_FULL_v1','core':cpu_tree(model.core.state_dict()),'ema':cpu_tree(ema.state_dict()),
        'optimizer':cpu_tree(opt.state_dict()),'scheduler':{'last_update':update,'total_updates':config['updates'],
        'peaks':config['lr_peaks'],'warmup_fraction':.02,'end_ratio':.1},'update':update,'cursor':cursor,
        'images_seen':cursor,'rng_by_rank':rng,'config':config,'identity':identity,'scaler':None,
        'native_buffers':{n:t.detach().cpu().clone() for n,t in model.restorer.native.named_buffers()},
        'sampling_buffers':{n:t.detach().cpu().clone() for n,t in model.restorer.sampling.named_buffers()}}

def restore(payload,model,opt,ema):
    assert payload['schema']=='V6_AB_FULL_v1' and payload['scaler'] is None
    model.core.load_state_dict(payload['core'],strict=True);opt.load_state_dict(payload['optimizer'])
    ema.model.load_state_dict(payload['ema']['model'],strict=True);ema.updates=payload['ema']['updates']
    assert payload['ema']['beta']==.995 and ema.updates==payload['update']
    for branch in ['native','sampling']:
        actual=dict(getattr(model.restorer,branch).named_buffers());saved=payload[branch+'_buffers']
        assert set(actual)==set(saved) and all(torch.equal(actual[n].cpu(),v) for n,v in saved.items())

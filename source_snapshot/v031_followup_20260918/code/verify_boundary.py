"""CPU-only complete-state and exact sampler endpoint verification."""
from pathlib import Path
from collections import Counter
import argparse,json
import torch
from v031.data import StartDataset
from v031.training import GlobalStartSampler,empty_counts,advance_counts,compare_states
from v031.checkpoint import _validate_fp32_training_state
from v031.scheduler import UpdateIndexedScheduler
from v031.runtime import save_json,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();p=Path(a.checkpoint);c=torch.load(p,map_location='cpu',weights_only=True)
    assert c['format']=='v031_followup_raw_boundary_v1' and c['formal_training'] is True
    assert c['global_update']==c['completed_updates']==1000 and c['smoke_optimizer_updates']==0
    assert c['world_size']==4 and c['effective_batch']==32 and len(c['rng_by_rank'])==4
    assert c['ema_enabled'] is False and 'ema' not in c and c['run_until_update']==1000
    _validate_fp32_training_state(c['network'],c['optimizer'],None)
    assert all(float(s['step'])==1000 for s in c['optimizer']['state'].values())
    assert c['optimizer']['state'] and all(g['lr']==1e-5 for g in c['optimizer']['param_groups'])
    class Holder:pass
    h=Holder();h.param_groups=c['optimizer']['param_groups']
    scheduler=UpdateIndexedScheduler(h,completed_updates=1000)
    assert scheduler.validate_state_dict(c['scheduler'])==1000
    data=StartDataset(c['identity']['dataset_root'])
    sampler=GlobalStartSampler([r['start_id'] for r in data.records],seed=20260916)
    exposure=Counter();counts=empty_counts(c['task'])
    for _ in range(1000):
        indices=sampler.next_indices();exposure.update(indices)
        advance_counts(counts,[data.records[i]['initial_G'] for i in indices])
    assert len(exposure)==1600 and set(exposure.values())=={20}
    assert compare_states(sampler.state_dict(),c['sampler'])['exact'] and counts==c['counts']
    for rng in c['rng_by_rank']:
        assert 'torch_npu_local' in rng and torch.is_tensor(rng['torch_npu_local'])
    save_json(a.output,{'status':'training_paused_at1000','checkpoint':str(p),'sha256':sha256(p),
        'counts':counts,'full_state_saved':True,'full_state_cpu_validated':True,'exact_exposures_per_start':20,
        'evaluation_pending':True,'task':c['task']})

if __name__=='__main__':main()

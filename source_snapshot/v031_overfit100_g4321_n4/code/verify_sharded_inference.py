"""CPU validation of actual native xi and clean tensors from every selected card."""
import json
from pathlib import Path
import torch
from sharded_eval_support import BASE, CARDS, verify
import run_auto_to_1000 as frozen

verify()
torch.set_num_threads(4)
reference=torch.load(BASE/'probes/card0/tensors.pt',map_location='cpu',weights_only=True)
reports=[]
identities=[]
for card in CARDS:
    folder=BASE/'probes'/f'card{card}'
    result=frozen.read(folder/'result.json')
    assert result['status']=='passed' and result['optimizer_updates']==0
    assert result['physical_card']==card and result['network_unchanged'] and result['simulator_unchanged']
    identities.append({k:result[k] for k in ['checkpoint_sha256','start_id','mode','reference_sha256']})
    candidate=torch.load(folder/'tensors.pt',map_location='cpu',weights_only=True)
    assert candidate.keys()==reference.keys()
    quantities=[]
    for key,a in candidate.items():
        b=reference[key]
        assert a.shape==b.shape and a.dtype==b.dtype==torch.float32 and bool(torch.isfinite(a).all())
        delta=a-b
        record={'name':key,'max_absolute_difference':float(delta.abs().max()),
            'relative_l2':float(delta.norm()/b.norm().clamp_min(1e-12)),
            'exact':torch.equal(a,b),'allclose':torch.allclose(a,b,rtol=1e-4,atol=1e-5)}
        quantities.append(record)
        if not record['allclose']:
            frozen.write(BASE/'CROSS_CARD_ACCEPTANCE.json',{'status':'failed','card':card,'quantity':record})
            raise RuntimeError('Native cross-card parity failed; thresholds remain unchanged')
    reports.append({'card':card,'quantities':quantities,'reference_scalar_check':result})
    del candidate
assert all(identity==identities[0] for identity in identities)
frozen.write(BASE/'CROSS_CARD_ACCEPTANCE.json',{'status':'passed','utc':frozen.utc(),
    'cards':list(CARDS),'native_atol':1e-5,'native_rtol':1e-4,
    'scope':'one fixed real G4 refined path; all four clean predictions and both native xi blocks before/after all three transitions',
    'identity':identities[0],'restoration_optimizer_updates':0,'reports':reports,
    'all_native_tensors_exact':all(q['exact'] for r in reports for q in r['quantities'])})
print(json.dumps({'status':'passed','cards':list(CARDS),'quantities_per_card':len(reference)}))

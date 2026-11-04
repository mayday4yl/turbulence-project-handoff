"""Stage2 logging-only core extension versus frozen Stage1, zero updates."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import torch
from v031.runtime import build, configure_npu, save_json, sha256, state_hash
from v031.core import RolloutModule
from v031.data import StartDataset, collate_starts

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
args = p.parse_args()
outdir = ROOT / args.output
outdir.mkdir(parents=True, exist_ok=False)
cfg = json.loads((ROOT/'configs/stage1_acceptance.json').read_text())
oldfile = ROOT/'artifacts/stage1_final_snapshot/code/v031/core.py'
spec = importlib.util.spec_from_file_location('frozen_stage1_core', oldfile)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
result = {'phase':'stage2_logging_equivalence', 'formal_training_started':False,
          'restoration_optimizer_updates':0, 'smoke_optimizer_updates':0,
          'old_core_sha256':sha256(oldfile), 'new_core_sha256':sha256(ROOT/'code/v031/core.py'),
          'config_sha256':sha256(ROOT/'configs/stage1_acceptance.json')}
for source in [Path(__file__), oldfile, ROOT/'code/v031/core.py']:
    name = 'old_core.py' if source == oldfile else source.name
    shutil.copy2(source, outdir/name)
try:
    import torch_npu
    result['runtime'] = configure_npu(cfg)
    model, sim = build(cfg, torch.device('npu:0'))
    initial = state_hash(model)
    dataset = StartDataset(cfg['dataset_root'], cfg['diagnostic_subset'])
    first = next(r['trajectory_id'] for r in dataset.records if r['initial_G']==4)
    indices = [next(i for i,r in enumerate(dataset.records) if r['trajectory_id']==first and r['initial_G']==g) for g in (4,1)]
    batch = collate_starts([dataset[i] for i in indices])
    cc, yy, gg = [batch[k].to('npu:0') for k in ('clean','observation','G')]
    result['start_ids'] = batch['start_id']
    values, gradients = {}, {}
    for name, cls, opts in [('frozen',old.RolloutModule,{}), ('logged',RolloutModule,{'collect_step_metrics':True})]:
        model.zero_grad(set_to_none=True)
        output = cls(model,sim,.2).train()(cc,yy,gg,**opts)
        values[name] = {k:output[k].detach().cpu() for k in ('path_losses','path_clean','path_reconstruction')}
        if name=='logged':
            terms = torch.zeros(2)
            for row in output['step_metrics']:
                assert all(not row[k].requires_grad for k in ('current_g','clean_losses','reconstruction_losses'))
                for local,j in enumerate(row['active_path_indices']):
                    terms[j] += (row['clean_losses'][local]+.2*row['reconstruction_losses'][local]).cpu()/batch['G'][j]
            result['metrics_path_loss_max_abs'] = float((terms-values[name]['path_losses']).abs().max())
            result['no_retained_native_trace'] = output['trace']==[]
        output['loss_mean'].backward()
        gradients[name] = {n:p.grad.detach().cpu().double() if p.grad is not None else None for n,p in model.named_parameters()}
        del output
        model.zero_grad(set_to_none=True)
        gc.collect();torch.npu.synchronize();torch.npu.empty_cache()
    dot = na = nb = diff = 0.
    connected = True
    for n,a in gradients['frozen'].items():
        b = gradients['logged'][n]
        connected &= (a is None)==(b is None)
        if a is None or b is None:continue
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        dot += float((a*b).sum());na += float(a.square().sum());nb += float(b.square().sum());diff += float((a-b).square().sum())
    result.update({'gradient_cosine':dot/(na*nb)**.5, 'gradient_relative_l2':(diff/na)**.5,
                   'same_connected_parameters':connected,
                   'forward_max_abs':max(float((values['frozen'][k]-values['logged'][k]).abs().max()) for k in values['frozen']),
                   'model_state_unchanged':initial==state_hash(model)})
    result['passed'] = (connected and result['model_state_unchanged'] and result['no_retained_native_trace']
                        and result['forward_max_abs']==0 and result['metrics_path_loss_max_abs']<1e-7
                        and result['gradient_cosine']>.99999 and result['gradient_relative_l2']<1e-4)
    result['status'] = 'passed' if result['passed'] else 'failed'
except BaseException as exc:
    result.update(status='error',passed=False,error=repr(exc))
    raise
finally:
    save_json(outdir/'results.json',result)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)

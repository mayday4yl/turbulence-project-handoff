"""Bounded runtime estimate using frozen serial refinement, no training updates."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'deps'), str(ROOT / 'scripts')]
os.environ['TORCH_HOME'] = str(ROOT / 'assets/torch')
import torch
from metrics import Metrics
from v4.matched import Store, build, seed_for, SEED
from v4.inference import infer
from v4.refinement import module_hash
from v4.runtime import configure_cuda, save_json


def main(group):
    configure_cuda(SEED)
    cfg = json.loads((ROOT / 'configs/runtime.json').read_text())
    store = Store(cfg)
    model, sim, _ = build(cfg, group)
    model.eval()
    metrics = Metrics('cuda')
    before = (module_hash(model), module_hash(sim))
    fixed = store.val[:8]
    assert sorted(r['G'] for r in fixed) == [1, 1, 2, 2, 3, 3, 4, 4]
    rows = []
    for offset in range(0, 8, 2):
        chunk = fixed[offset:offset + 2]
        C, Y, G = store.batch(chunk, 'cuda')
        seeds = [seed_for(f'eval|{SEED}|{r["start_id"]}') for r in chunk]
        torch.cuda.synchronize()
        start = time.monotonic()
        result = infer(model, sim, Y, G, noise_seeds=seeds,
                       T0=0. if group == 'R0' else .8, refinement=True)
        values = metrics(result['clean'], C)
        torch.cuda.synchronize()
        assert torch.isfinite(result['clean']).all()
        rows.append({'start_ids': [r['start_id'] for r in chunk],
                     'G': G.tolist(), 'total_seconds': time.monotonic()-start,
                     'refinement_seconds': result['refinement_seconds'],
                     'refinement_stages': result['refinement_sample_count'],
                     'metrics': values})
    assert before == (module_hash(model), module_hash(sim))
    assert sum(r['refinement_stages'] for r in rows) == 12
    # Full val800 has exactly 100 times this diagnostic's 8-start G mixture.
    estimate = 100 * sum(r['total_seconds'] for r in rows)
    save_json(ROOT / f'reports/REFINEMENT_RUNTIME_{group}.json', {
        'status': 'PASS_RUNTIME_PROFILE_ONLY', 'group': group,
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'optimizer_updates': 0,
        'initialization': 'frozen common init with train500 statistics, NOT trained milestone weights',
        'execution': 'FP32 batch2 with per_sample_serial 40-update refinement',
        'weights_unchanged': True, 'rows': rows,
        'estimated_val800_seconds': estimate,
        'uncertainty': '8 starts only; excludes full-data loading, result serialization, paired CI and stage report; replace with milestone measured runtime',
        'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['R0', 'R08'], required=True)
    main(parser.parse_args().group)

"""Full file/hash/shape audit plus bounded deterministic raw/renderer replay."""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import COUNTS, digest, rows, sha, verify_array, write_json, write_rows
from generate import load_runtime, raw_noise, sync


def finalize(a):
    root = Path(a.root)
    freeze = json.loads((root / 'SOURCE_FREEZE.json').read_text())
    sources, plan = rows(root / 'source_manifest.jsonl'), rows(root / 'sample_plan.jsonl')
    assert sha(root / 'source_manifest.jsonl') == freeze['source_manifest_sha256']
    assert sha(root / 'sample_plan.jsonl') == freeze['sample_plan_sha256']
    assert collections.Counter(x['split'] for x in sources) == COUNTS
    assert collections.Counter(x['split'] for x in plan) == {s: c * 4 for s, c in COUNTS.items()}
    for field in ['source_id', 'archive_member', 'original_sha256', 'clean_pixel_sha256']:
        assert len({x[field] for x in sources}) == 2500
    for x in sources:
        clean = verify_array(root / x['clean']['ref'], x['clean']['sha256'])
        assert digest(clean.tobytes()) == x['clean_pixel_sha256']
        assert sha(root / x['original_relative_path']) == x['original_sha256']
    generation = json.loads((root / 'GENERATION_IDENTITY.json').read_text())
    generation_hash = sha(root / 'GENERATION_IDENTITY.json')
    assert generation['source_freeze_sha256'] == sha(root / 'SOURCE_FREEZE.json')
    assert generation['torch'] == str(torch.__version__), 'RNG replay requires the frozen torch version'
    records, raw_hashes = [], set()
    for row in plan:
        record = json.loads((root / row['record_ref']).read_text())
        assert record['sample_plan_row'] == row
        assert record['generation_identity_sha256'] == generation_hash
        output = verify_array(root / row['observation_ref'], record['observation_sha256'])
        assert digest(output.tobytes()) == record['observation_pixel_sha256']
        for raw_hash in record['raw_sha256'].values():
            assert raw_hash not in raw_hashes, 'Duplicate raw field'
            raw_hashes.add(raw_hash)
        records.append(record)
    assert len(records) == 10000 and len(raw_hashes) == 20000
    # Deliberately include all split identities and one expanded test source, but
    # compute no restoration metrics, no model prediction, and no tuning on test.
    replay_rows = []
    for split in ['train', 'val', 'test']:
        candidates = [r for r in plan if r['split'] == split]
        replay_rows.extend([candidates[0], candidates[-1]])
    sim, Realization, runtime = load_runtime(a.simulator_root, a.device, root.parent.parent / 'cache/data_replay')
    for name, value in runtime.items():
        assert generation.get(name) == value, ('Generation runtime mismatch', name, generation.get(name), value)
    replay_results = []
    records_by_id = {r['sample_plan_row']['sample_id']: r for r in records}
    with torch.no_grad():
        for row in replay_rows:
            raw, raw_hash = raw_noise(row)
            record = records_by_id[row['sample_id']]
            assert raw_hash == record['raw_sha256']
            image = torch.from_numpy(verify_array(root / row['clean_ref'])[None]).to(a.device)
            xi = Realization(*(x.to(a.device) for x in raw))
            result = sim.render(image, xi, 4.0)
            sync(a.device)
            output = result.output[0].cpu().numpy()
            expected = verify_array(root / row['observation_ref'])
            max_abs = float(np.abs(output - expected).max())
            # Use the original frozen Gate's absolute CPU/NPU output bound.
            # Do not relax it with a relative tolerance after seeing results.
            if max_abs > 2e-5:
                raise ValueError(f'Render replay mismatch {row["sample_id"]}: {max_abs}')
            replay_results.append({'sample_id': row['sample_id'], 'raw_sha_exact': True,
                                   'observation_exact': bool(np.array_equal(output, expected)), 'max_abs': max_abs})
            del raw, xi, image, result
    cpu_reference = None
    if a.device.startswith('npu'):
        cpu_sim, CpuRealization, cpu_runtime = load_runtime(a.simulator_root, 'cpu', root.parent.parent / 'cache/data_replay_cpu')
        for key in ['simulator_sha256', 'vendor_simulator_sha256', 'asset_sha256', 'torch', 'raw_rng', 'raw_shapes', 'G', 'clamp', 'smooth_psf', 'pixel_center_grid', 'tilt_fft']:
            assert generation[key] == cpu_runtime[key], key
        row = next(row for row in plan if row['split'] == 'train')
        with torch.no_grad():
            raw, raw_hash = raw_noise(row)
            assert raw_hash == records_by_id[row['sample_id']]['raw_sha256']
            clean = torch.from_numpy(verify_array(root / row['clean_ref'])[None])
            output = cpu_sim.render(clean, CpuRealization(*raw), 4.0).output[0].numpy()
            expected = verify_array(root / row['observation_ref'])
            error = float(np.abs(output - expected).max())
            if error > 2e-5:
                raise ValueError(f'New train sample CPU/NPU reference mismatch: {error}')
            cpu_reference = {'sample_id': row['sample_id'], 'max_abs': error, 'threshold_max_abs': 2e-5,
                             'status': 'PASS', 'dtype': 'float32', 'test_used': False}
    write_rows(root / 'sample_manifest.jsonl', records)
    ready = {'status': 'DATA_READY', 'source_freeze_sha256': sha(root / 'SOURCE_FREEZE.json'),
             'generation_identity_sha256': generation_hash, 'sample_manifest_sha256': sha(root / 'sample_manifest.jsonl'),
             'source_counts': COUNTS, 'sample_counts': {s: c * 4 for s, c in COUNTS.items()},
             'verified_rgb_arrays': 12500, 'raw_hashes_unique': 20000, 'raw_and_render_replays': replay_results,
             'replay_tolerance': {'max_abs': 2e-5}, 'cpu_npu_train_reference': cpu_reference,
             'source_disjoint_exact': True, 'old_Y_reused': False, 'test_recovery_model_evaluation': False,
             'raw_storage': 'Seeds plus full raw SHA; no raw files. Replays require frozen CPU torch RNG/version.',
             'observation_value_policy': 'Unclamped FP32 simulator outputs', 'generator_elapsed_seconds_sum': sum(r['elapsed_seconds'] for r in records)}
    write_json(root / 'DATA_READY.json', ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--simulator-root', default='/data2/yanglin/v5_v1_simulator_20261020')
    parser.add_argument('--device', required=True, help='Same backend as generation; only six integrity replays')
    finalize(parser.parse_args())

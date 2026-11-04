"""Download the selected official SR asset and audit reused paired caches."""
import concurrent.futures
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data2/yanglin/v6_diffuir_l_20261027/execution_20261029_d0_multistrength160k')
FILE_ID = '1kIrvZ1X46jFRAYIScverDD3dT1F40Kip'
EXPECTED_BYTES = 548649183


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def download():
    folder = ROOT / 'weights'
    folder.mkdir(exist_ok=True)
    dest = folder / 'sr_gamma_1e7.pth'
    partial = folder / 'sr_gamma_1e7.pth.partial'
    if dest.exists():
        assert dest.stat().st_size == EXPECTED_BYTES
    else:
        url = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
        session = requests.Session()
        session.proxies.update(http='http://127.0.0.1:7890', https='http://127.0.0.1:7890')
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        started = time.time()
        with session.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            assert 'text/html' not in response.headers.get('Content-Type', '')
            if offset and response.status_code != 206:
                offset = 0
            done = offset
            with partial.open('ab' if offset else 'wb') as f:
                last = 0
                for block in response.iter_content(4 << 20):
                    if not block:
                        continue
                    f.write(block)
                    done += len(block)
                    if time.time() - last > 5:
                        write(ROOT/'evidence/DOWNLOAD_PROGRESS.json', {
                            'status': 'DOWNLOADING', 'bytes': done, 'expected': EXPECTED_BYTES,
                            'elapsed_seconds': time.time()-started, 'new_MB_per_second':
                            (done-offset)/1e6/max(time.time()-started, .001)})
                        last = time.time()
                f.flush()
                os.fsync(f.fileno())
        assert partial.stat().st_size == EXPECTED_BYTES
        os.replace(partial, dest)
    # Tensor strict loading is a separate engineering gate.
    write(ROOT/'evidence/WEIGHT_DOWNLOAD.json', {
        'status': 'PASS_DOWNLOAD_SIZE_AND_SHA_ONLY', 'task': 'official DIV2K SR',
        'file_id': FILE_ID, 'original_filename': 'gamma_1e7.pth',
        'path': str(dest), 'bytes': dest.stat().st_size, 'sha256': sha(dest),
        'strict_load': 'NOT_YET_EXECUTED'})
    print('WEIGHT_DOWNLOAD_COMPLETE', flush=True)


def check_array(item):
    path, expected = item
    assert sha(path) == expected, ('SHA mismatch', path)
    a = np.load(path, allow_pickle=False)
    assert a.shape == (3, 256, 256) and a.dtype == np.float32, (path, a.shape, a.dtype)
    assert np.isfinite(a).all(), ('Nonfinite', path)
    return float(a.min()), float(a.max())


def data_audit():
    gate = json.loads((OLD/'DATA_GATE.json').read_text())
    assert gate['status'] == 'PASS'
    train_path, val_path = OLD/'TRAIN_MANIFEST.jsonl', OLD/'VAL_MANIFEST.jsonl'
    assert sha(train_path) == gate['train_manifest_sha256']
    assert sha(val_path) == gate['val_manifest_sha256']
    train = [json.loads(s) for s in train_path.read_text().splitlines()]
    val = [json.loads(s) for s in val_path.read_text().splitlines()]
    assert len(train) == 160000 and len(val) == 1000
    assert Counter(r['g'] for r in train) == {1:40000,2:40000,3:40000,4:40000}
    groups = defaultdict(set)
    for row in train:
        groups[row['source_id']].add((row['realization_index'], row['g']))
    expected = {(r,g) for r in range(4) for g in [1,2,3,4]}
    assert len(groups) == 10000 and all(s == expected for s in groups.values())
    assert len({r['pair_id'] for r in train}) == 160000
    train_sources, val_sources = set(groups), {r['source_id'] for r in val}
    assert len(val_sources) == 250 and not train_sources & val_sources
    assert not {r['clean_sha256'] for r in train} & {r['clean_sha256'] for r in val}
    files = {}
    for row in train:
        files[row['clean_path']] = row['clean_sha256']
        files[row['observation_path']] = row['observation_sha256']
    for row in val:
        files[row['clean_path']] = row['clean_sha256']
        assert set(row['states']) == {'0.5','1','2','3','4'}
        for entry in row['states'].values():
            files[entry['path']] = entry['sha256']
    started = time.time()
    lo, hi = 1., 0.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for count, (a,b) in enumerate(pool.map(check_array, files.items()), 1):
            lo, hi = min(lo,a), max(hi,b)
            if count % 2000 == 0:
                write(ROOT/'evidence/DATA_PROGRESS.json', {
                    'checked': count, 'total': len(files), 'seconds': time.time()-started})
    selected = sorted(val_sources)[:25]
    write(ROOT/'protocol/MONITOR_SELECTION.json', {
        'sources': selected, 'sources_per_g': 25, 'realizations_per_source': 4,
        'observations_per_g': 100, 'g_values': [.5,1,2,3,4],
        'selection': 'first 25 lexicographic frozen validation source IDs; chosen before model execution'})
    write(ROOT/'evidence/DATA_GATE.json', {
        'status': 'PASS', 'train_pairs': 160000, 'train_clean_sources': 10000,
        'val_clean_sources': 250, 'val_per_strength': 1000,
        'counts_by_g': dict(Counter(r['g'] for r in train)),
        'train_manifest': str(train_path), 'train_manifest_sha256': sha(train_path),
        'val_manifest': str(val_path), 'val_manifest_sha256': sha(val_path),
        'all_files_freshly_SHA_shape_dtype_finite_checked': len(files),
        'input_range': [lo,hi], 'clipping_applied': False, 'seconds': time.time()-started,
        'test_used': False, 'source_disjoint': True, 'same_source_xi_all_four_g': True})
    print('DATA_GATE_COMPLETE', len(files), flush=True)


if __name__ == '__main__':
    (ROOT/'evidence').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(download), executor.submit(data_audit)]
        for job in jobs:
            job.result()
    write(ROOT/'evidence/ASSETS_READY.json', {'status':'PASS','training_updates':0,'model_calls':0})

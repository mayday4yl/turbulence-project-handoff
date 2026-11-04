#!/usr/bin/env python3
"""Build only the authorized overfit dataset; never imports or runs a model."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time
import traceback
import numpy as np
import PIL
from PIL import Image, ImageOps
import torch
from dataset import OverfitDataset, rows

PROJECT = Path('/data2/yanglin/v03_overfit100_g4321_n4')
DATASET_ID = 'places_p2s_bridge_v03_overfit100_g4321_n4'
DATA = PROJECT / 'data' / DATASET_ID
BASE = Path('/data0/yanglin/physical_bridge')
OLD = Path('/data2/yanglin/physical_bridge/data/places_p2s_bridge_v01')
ARCHIVE = Path('/data2/yanglin/physical_bridge/data/source_archives/val_256.tar')
ASSETS = Path('/data0/yanglin/vendor/PiRN/assets/utils')
G_VALUES = (4, 3, 2, 1)
SEED = 20260916
EXPECTED = {
    str(BASE / 'physical_bridge/simulator.py'): '937322fbd213d1674f315176daa47887c9d313d5dec517a1a2dbfbb91a4a042c',
    str(BASE / 'configs/simulator_v01.yaml'): '5444c2a426d69c559c9197ca8afdf8f20e3230767abb8e2a6d277d49a65e4b7f',
    str(ASSETS / 'P2S_model.pt'): '578c3cd09c5c44e66c1086ffc0aadcab99542b74d27b932e9ca601a3faf6a4d5',
    str(ASSETS / 'dictionary.pt'): '2204ace3e4b99f31d62d21d0ba19bae95c606393ba7df5111e2526172e6b24c6',
    str(ASSETS / 'I0_I2.pt'): 'aa99db54950df99465e5c8b7b7b16b7ed9236603b5c1a63d82dd8831b5712f2e',
}
PHYSICS = dict(image_size=256, rf_range=512, aperture_diameter=.071,
               propagation_distance=100., wavelength=5e-7, correlation=-.05,
               tilt_threshold=.02, adjustment=1.)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def save_json(path, value):
    atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def save_rows(path, values):
    atomic(path, ''.join(json.dumps(v, sort_keys=True, allow_nan=False) + '\n' for v in values).encode())


def save_array(path, value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite array: {path}')
    b = io.BytesIO()
    np.save(b, value, allow_pickle=False)
    atomic(path, b.getvalue())


def config():
    return json.loads((PROJECT / 'configs/build_config.json').read_text())


def source_order(s):
    return digest(('source-select|20260916|' + s).encode('utf-8')), s


def subset_order(s):
    return digest(('debug-subset|20260916|' + s).encode('utf-8')), s


def xi_seed(source, realization, block):
    payload = json.dumps([SEED, source, realization, block], ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)


def progress(stage, **values):
    value = dict(stage=stage, utc=now(), **values)
    save_json(PROJECT / 'artifacts/progress.json', value)
    print(json.dumps(value), flush=True)


def storage_check():
    free = shutil.disk_usage(PROJECT).free
    if free < 1 << 40:
        raise RuntimeError('Data2 free space is below 1 TiB')
    return free


def identities():
    paths = [*map(Path, EXPECTED), ARCHIVE,
             *(OLD / 'manifests' / f'source_{s}.jsonl' for s in ('train', 'val', 'test')),
             OLD / 'manifests/train.jsonl', BASE / 'physical_bridge/model.py',
             BASE / 'physical_bridge/inference.py', BASE / 'physical_bridge/data.py',
             BASE / 'scripts/train.py', BASE / 'configs/train_v01.yaml',
             BASE / 'configs/inference_v01.yaml']
    result = {str(p): sha(p) for p in paths}
    for p, wanted in EXPECTED.items():
        if result[p] != wanted:
            raise RuntimeError(f'Baseline identity mismatch: {p}')
    return result


def code_identity():
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted((PROJECT / 'code').glob('*'))
            if p.is_file()}


def preprocess(raw):
    with Image.open(io.BytesIO(raw)) as source:
        original_size = list(source.size)
        orientation = source.getexif().get(274)
        image = ImageOps.exif_transpose(source).convert('RGB')
        oriented_size = list(image.size)
        w, h = image.size
        if (w, h) == (256, 256):
            resized = (w, h)
            box = [0, 0, w, h]
            resized_applied = crop_applied = False
        else:
            short = min(w, h)
            resized = (w * 256 // short, h * 256 // short)
            image = image.resize(resized, Image.Resampling.BICUBIC)
            left, top = (resized[0] - 256) // 2, (resized[1] - 256) // 2
            box = [left, top, left + 256, top + 256]
            image = image.crop(tuple(box))
            resized_applied = crop_applied = True
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) / np.float32(255.)
    return array, dict(original_size_wh=original_size, exif_orientation=orientation,
                      exif_transpose=True, oriented_size_wh=oriented_size, convert_mode='RGB',
                      resize_size_wh=list(resized), resize_applied=resized_applied,
                      interpolation='Pillow BICUBIC', long_dimension_rounding='floor',
                      crop_box_xyxy=box, crop_applied=crop_applied, random_augmentation=False,
                      pillow_version=PIL.__version__, dtype='float32', layout='CHW', scale='uint8 / 255.0')


def prepare():
    if (PROJECT / 'artifacts/run_manifest.json').exists():
        raise RuntimeError('Prepare already exists; do not overwrite an existing identity')
    storage_check()
    input_hashes = identities()
    by_split = {s: rows(OLD / 'manifests' / f'source_{s}.jsonl') for s in ('train', 'val', 'test')}
    pool = {}
    for r in by_split['train']:
        sid = r['source_image_id']
        if sid != Path(sid).name or not sid.isascii() or sid.strip() != sid:
            raise ValueError('Unexpected source ID encoding')
        if r['source_split'] != 'train':
            raise ValueError('Non-train source in source train manifest')
        if sid in pool and pool[sid] != r:
            raise ValueError('Conflicting source identity')
        pool[sid] = r
    train_trajectory_ids = {r['source_image_id'] for r in rows(OLD / 'manifests/train.jsonl')}
    if set(pool) != train_trajectory_ids or len(pool) < 100:
        raise ValueError('Train source pool could not be verified')
    selected = sorted(pool, key=source_order)[:100]
    overlap = {s: sorted(set(selected) & {r['source_image_id'] for r in by_split[s]}) for s in ('val', 'test')}
    if any(overlap.values()):
        raise ValueError('Source leakage')
    sources = []
    with tarfile.open(ARCHIVE, 'r:') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        for sid in selected:
            original = pool[sid]
            raw = archive.extractfile(members[original['source_member']]).read()
            clean, prep = preprocess(raw)
            ref = f'clean/{sid}.npy'
            save_array(DATA / ref, clean)
            sources.append(dict(source_id=sid, original_split='train',
                                original_source_ref=dict(archive=str(ARCHIVE), member=original['source_member']),
                                original_content_sha256=digest(raw), preprocessing=prep,
                                clean_ref=ref, clean_sha256=sha(DATA / ref),
                                source_category_id=original['source_category_id'],
                                selection_digest=source_order(sid)[0]))
    save_rows(DATA / 'source_manifest.jsonl', sources)
    atomic(DATA / 'source_ids.txt', ('\n'.join(selected) + '\n').encode())
    planned = []
    for s in sources:
        for r in range(1, 5):
            tid = f"{Path(s['source_id']).stem}_r{r:02d}"
            planned.append(dict(trajectory_id=tid, source_id=s['source_id'], realization_index=r,
                                clean_ref=s['clean_ref'], clean_sha256=s['clean_sha256'],
                                seed_zernike=xi_seed(s['source_id'], r, 'epsilon_zernike'),
                                seed_tilt=xi_seed(s['source_id'], r, 'epsilon_tilt'),
                                epsilon_zernike_ref=f'xi/{tid}/epsilon_zernike.npy',
                                epsilon_tilt_ref=f'xi/{tid}/epsilon_tilt.npy',
                                severity_values=list(G_VALUES),
                                **{f'x{g}_ref': f'states/{tid}/X_{g}.npy' for g in G_VALUES}))
    seeds = [r[k] for r in planned for k in ('seed_zernike', 'seed_tilt')]
    if len(set(seeds)) != 800:
        raise ValueError('Seed collision')
    save_rows(PROJECT / 'artifacts/planned_trajectories.jsonl', planned)
    cfg = dict(dataset_id=DATASET_ID, dataset_build_seed=SEED, source_selection_seed=SEED,
               subset_selection_seed=SEED, source_count=100, realizations_per_source=4,
               severity_values=list(G_VALUES), renderer_order='blur_then_tilt', physics=PHYSICS,
               source_selection='SHA256(UTF8("source-select|20260916|" + source_id)), ascending digest then source_id',
               id_encoding='UTF-8; source basename exactly as original source manifest; no normalization; ASCII validated',
               xi_seed_algorithm='JSON compact UTF-8 [20260916,source_id,realization_index,block_name]; SHA256; first 8 bytes big endian & ((1<<63)-1)',
               realization_indices=[1, 2, 3, 4], native_sampling='PiRNP2SSimulator.sample_realization(batch_size=1), independent CPU torch.Generator FP32 blocks',
               authority_dtype='float32', clamp=False, render_batch_size=1,
               physical_devices=[0], logical_device='npu:0', jit_compile=False,
               replay_tolerance=None, engineering_configuration_frozen=False,
               preprocessing='EXIF transpose; RGB; identity for 256x256; otherwise short-side 256 bicubic with integer floor; center crop floor; FP32/255',
               path_semantics='num_predictions includes final prediction at g=1; no g=0; no network run',
               selected_source_manifest_sha256=sha(DATA / 'source_manifest.jsonl'))
    save_json(PROJECT / 'configs/build_config.json', cfg)
    run = dict(dataset_id=DATASET_ID, status='prepared', started_utc=now(), host=platform.node(),
               input_sha256_before=input_hashes, code_sha256=code_identity(),
               source_candidate_count=len(pool), source_overlap=overlap,
               selected_source_ids=selected, planned_trajectory_manifest_sha256=sha(PROJECT / 'artifacts/planned_trajectories.jsonl'),
               versions=dict(python=sys.version, pytorch=torch.__version__, numpy=np.__version__, pillow=PIL.__version__),
               remote_project=str(PROJECT), dataset_root=str(DATA), command=sys.argv,
               authoritative_array_expected_bytes=100*3*256*256*4 + 400*(36*512*512+512*512*2+4*3*256*256)*4,
               source_archive_untouched=True, root_disk_writes=False)
    save_json(PROJECT / 'artifacts/run_manifest.json', run)
    progress('prepared', sources=100, trajectories=400, expected_array_bytes=run['authoritative_array_expected_bytes'])


def verify_inputs(check_raw_archive=False):
    run = json.loads((PROJECT / 'artifacts/run_manifest.json').read_text())
    for p, wanted in run['input_sha256_before'].items():
        if not check_raw_archive and p == str(ARCHIVE):
            continue
        if sha(p) != wanted:
            raise RuntimeError('Protected input changed: ' + p)
    if code_identity() != run['code_sha256']:
        raise RuntimeError('Build code changed after prepare')
    return run


def simulator():
    verify_inputs()
    import torch_npu
    sys.path.insert(0, str(BASE))
    from physical_bridge.simulator import PiRNP2SSimulator
    torch.npu.set_device('npu:0')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(4)
    sim = PiRNP2SSimulator(ASSETS, **PHYSICS, device='npu:0')
    if sim.realization_shapes != {'epsilon_zernike': (36, 512, 512), 'epsilon_tilt': (512, 512, 2)}:
        raise RuntimeError('Native realization shape mismatch')
    if any(p.requires_grad for p in sim.parameters()):
        raise RuntimeError('Simulator is not frozen')
    run = json.loads((PROJECT / 'artifacts/run_manifest.json').read_text())
    run['versions'].update(torch_npu=torch_npu.__version__, cann_resolved_path=str(Path('/usr/local/Ascend/ascend-toolkit/latest').resolve()))
    for p in (Path('/usr/local/Ascend/ascend-toolkit/latest/version.cfg'), Path('/usr/local/Ascend/ascend-toolkit/latest/version.info')):
        if p.is_file():
            run['versions'][p.name] = p.read_text(errors='replace')
    run['actual_simulator_protocol'] = sim.protocol()
    run['note'] = 'Simulator protocol g_grid is inherited informational old six-level constant; this independent builder calls actual g=4,3,2,1 explicitly.'
    save_json(PROJECT / 'artifacts/run_manifest.json', run)
    return sim


def sample(sim, record):
    return sim.sample_realization(1, seed_zernike=record['seed_zernike'], seed_tilt=record['seed_tilt'])


def reload_xi(sim, record):
    from physical_bridge.simulator import Realization
    return Realization(*(torch.from_numpy(np.load(DATA / record[k], allow_pickle=False)).unsqueeze(0).to(sim.device_spec)
                         for k in ('epsilon_zernike_ref', 'epsilon_tilt_ref')))


def clean_tensor(sim, record):
    return torch.from_numpy(np.load(DATA / record['clean_ref'], allow_pickle=False)).unsqueeze(0).to(sim.device_spec)


def cpu_output(tensor):
    return tensor.detach().float().cpu().numpy()[0].copy()


def errors(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(max_abs=float(np.abs(d).max()), MSE=float(np.mean(d*d)), exact=bool(np.array_equal(a, b)))


def preflight():
    cfg = config()
    if cfg['engineering_configuration_frozen']:
        raise RuntimeError('Preflight already frozen')
    sim = simulator()
    records = rows(PROJECT / 'artifacts/planned_trajectories.jsonl')
    metrics = []
    timings = []
    start = time.monotonic()
    with torch.no_grad():
        for rec in (records[0], records[4]):
            clean = clean_tensor(sim, rec)
            xi = sample(sim, rec)
            xi2 = sample(sim, rec)
            for name in ('epsilon_zernike', 'epsilon_tilt'):
                a, b = cpu_output(getattr(xi, name)), cpu_output(getattr(xi2, name))
                if not np.array_equal(a, b):
                    raise RuntimeError('Native seed replay failed')
                path = PROJECT / 'artifacts/preflight_arrays' / rec['trajectory_id'] / f'{name}.npy'
                save_array(path, a)
                if not np.array_equal(a, np.load(path, allow_pickle=False)):
                    raise RuntimeError('FP32 xi roundtrip failed')
            from physical_bridge.simulator import Realization
            disk_xi = Realization(*(torch.from_numpy(np.load(PROJECT / 'artifacts/preflight_arrays' / rec['trajectory_id'] / f'{name}.npy', allow_pickle=False)).unsqueeze(0).to(sim.device_spec)
                                   for name in ('epsilon_zernike', 'epsilon_tilt')))
            prepared = sim.prepare_image(clean)
            for g in G_VALUES:
                t = time.monotonic()
                direct = cpu_output(sim.render(clean, xi, g).output)
                timings.append(time.monotonic() - t)
                cached = cpu_output(sim.render_prepared(prepared, xi, g).output)
                ref = PROJECT / 'artifacts/preflight_arrays' / rec['trajectory_id'] / f'X_{g}.npy'
                save_array(ref, cached)
                loaded = np.load(ref, allow_pickle=False)
                replay = cpu_output(sim.render_prepared(prepared, disk_xi, g).output)
                metrics.append(dict(trajectory_id=rec['trajectory_id'], g=g,
                                    repeated=errors(loaded, replay), cached_vs_direct=errors(direct, cached),
                                    fp32_roundtrip_exact=bool(np.array_equal(cached, loaded))))
    # Fixed conservative FP32 thresholds declared before full generation, never relaxed on failure.
    tolerance = dict(max_abs=1e-6, MSE=1e-12,
                     basis='Two trajectories x all four actual strengths; direct/cached and disk-reloaded xi repeated render tested before full build. Fixed FP32 thresholds, exact equality also reported.')
    for r in metrics:
        for name in ('repeated', 'cached_vs_direct'):
            if r[name]['max_abs'] > tolerance['max_abs'] or r[name]['MSE'] > tolerance['MSE']:
                raise RuntimeError('Preflight differences exceed fixed FP32 tolerance')
        if not r['fp32_roundtrip_exact']:
            raise RuntimeError('State roundtrip differs')
    cfg['replay_tolerance'] = tolerance
    cfg['engineering_configuration_frozen'] = True
    cfg['engineering_frozen_utc'] = now()
    cfg['image_preparation'] = 'Original simulator.prepare_image once per trajectory; render_prepared for each actual g; verified against independent original render calls'
    save_json(PROJECT / 'configs/build_config.json', cfg)
    result = dict(status='passed', metrics=metrics, render_seconds=timings,
                  steady_median_render_seconds=float(np.median(timings[1:])),
                  wall_seconds=time.monotonic()-start, tolerance=tolerance,
                  build_config_sha256=sha(PROJECT / 'configs/build_config.json'))
    save_json(PROJECT / 'artifacts/preflight.json', result)
    save_json(PROJECT / 'artifacts/config_freeze.json', {'sha256':result['build_config_sha256'], 'frozen_utc':now()})
    progress('preflight_passed', **result)


def build():
    cfg = config()
    freeze = json.loads((PROJECT / 'artifacts/config_freeze.json').read_text())
    if not cfg['engineering_configuration_frozen'] or sha(PROJECT / 'configs/build_config.json') != freeze['sha256']:
        raise RuntimeError('Engineering configuration is not frozen or changed')
    sim = simulator()
    records = rows(PROJECT / 'artifacts/planned_trajectories.jsonl')
    start_time = time.monotonic()
    completed = []
    identity = {p: v for p, v in EXPECTED.items()}
    with torch.no_grad():
        for n, planned in enumerate(records):
            storage_check()
            done_path = PROJECT / 'artifacts/completed_trajectories' / (planned['trajectory_id'] + '.json')
            if done_path.exists():
                done = json.loads(done_path.read_text())
                if done['build_config_sha256'] != freeze['sha256'] or any(sha(DATA / p) != h for p, h in done['file_sha256'].items()):
                    raise RuntimeError('Existing partial trajectory failed identity verification')
                completed.append(done)
                continue
            t = time.monotonic()
            rec = dict(planned)
            clean = clean_tensor(sim, rec)
            xi = sample(sim, rec)
            for name in ('epsilon_zernike', 'epsilon_tilt'):
                save_array(DATA / rec[name + '_ref'], cpu_output(getattr(xi, name)))
            prepared = sim.prepare_image(clean)
            stats = {}
            for g in G_VALUES:
                x = cpu_output(sim.render_prepared(prepared, xi, g).output)
                save_array(DATA / rec[f'x{g}_ref'], x)
                stats[str(g)] = dict(min=float(x.min()), max=float(x.max()),
                                     below_zero_fraction=float(np.mean(x < 0)), above_one_fraction=float(np.mean(x > 1)))
            file_hashes = {rec[k]: sha(DATA / rec[k]) for k in ('epsilon_zernike_ref', 'epsilon_tilt_ref', *(f'x{g}_ref' for g in G_VALUES))}
            rec.update(xi_sha256={name:file_hashes[rec[name+'_ref']] for name in ('epsilon_zernike', 'epsilon_tilt')},
                       simulator_identity=identity)
            done = dict(record=rec, file_sha256=file_hashes, state_statistics=stats,
                        build_config_sha256=freeze['sha256'], elapsed_seconds=time.monotonic()-t)
            save_json(done_path, done)
            completed.append(done)
            if (n+1) % 10 == 0 or n == 0:
                elapsed = time.monotonic()-start_time
                progress('generating', completed_trajectories=n+1, total_trajectories=400,
                         elapsed_seconds=elapsed, mean_seconds_per_trajectory=elapsed/(n+1),
                         remaining_generation_seconds=elapsed/(n+1)*(400-n-1))
    trajectories = [r['record'] for r in completed]
    save_rows(DATA / 'trajectory_manifest.jsonl', trajectories)
    starts = []
    for rec in trajectories:
        for g in G_VALUES:
            starts.append(dict(start_id=rec['trajectory_id']+f'_G{g}', trajectory_id=rec['trajectory_id'],
                               source_id=rec['source_id'], initial_G=g, initial_state_ref=rec[f'x{g}_ref'],
                               clean_ref=rec['clean_ref'], severity_path=list(range(g,0,-1)), num_predictions=g))
    save_rows(DATA / 'start_manifest.jsonl', starts)
    ordered = sorted(trajectories, key=lambda r:subset_order(r['trajectory_id']))
    subsets = {}
    for count in (1, 16, 400):
        chosen = ordered[:count]
        ids = {r['trajectory_id'] for r in chosen}
        selected_starts = [s for s in starts if s['trajectory_id'] in ids]
        prefix = f'subsets/overfit_{count}traj'
        save_rows(DATA / (prefix+'_trajectories.jsonl'), chosen)
        save_rows(DATA / (prefix+'_starts.jsonl'), selected_starts)
        counts = Counter(r['source_id'] for r in chosen)
        subsets[str(count)] = dict(trajectory_ids=[r['trajectory_id'] for r in chosen],
                                  start_ids=[r['start_id'] for r in selected_starts],
                                  source_count=len(counts), per_source_trajectory_counts=dict(counts),
                                  manifest_sha256={prefix+suffix:sha(DATA/(prefix+suffix)) for suffix in ('_trajectories.jsonl','_starts.jsonl')})
    save_json(DATA / 'subsets/subset_summary.json', dict(seed=SEED, algorithm='SHA256(UTF8("debug-subset|20260916|"+trajectory_id)), digest then ID ascending; prefixes 1,16,400', subsets=subsets))
    progress('generated', sources=100, trajectories=400, states=1600, starts=1600, elapsed_seconds=time.monotonic()-start_time)


def audit():
    cfg = config()
    freeze = json.loads((PROJECT / 'artifacts/config_freeze.json').read_text())
    if sha(PROJECT / 'configs/build_config.json') != freeze['sha256']:
        raise RuntimeError('Frozen configuration changed')
    sim = simulator()
    dataset = OverfitDataset(DATA)
    sources = rows(DATA / 'source_manifest.jsonl')
    trajectories = rows(DATA / 'trajectory_manifest.jsonl')
    starts = dataset.starts
    assert len(sources) == len(dataset.sources) == 100
    assert len(trajectories) == len(dataset.trajectories) == 400
    assert len(starts) == len({s['start_id'] for s in starts}) == 1600
    assert set(Counter(t['source_id'] for t in trajectories).values()) == {4}
    assert set(Counter(s['trajectory_id'] for s in starts).values()) == {4}
    original_pool = {r['source_image_id'] for r in rows(OLD/'manifests/source_train.jsonl')}
    expected_selected = sorted(original_pool, key=source_order)[:100]
    assert [s['source_id'] for s in sources] == expected_selected
    overlap = {split: sorted(set(dataset.sources) & {r['source_image_id'] for r in rows(OLD/'manifests'/f'source_{split}.jsonl')}) for split in ('val','test')}
    assert not any(overlap.values())
    with tarfile.open(ARCHIVE, 'r:') as raw_archive:
        for source in sources:
            raw = raw_archive.extractfile(source['original_source_ref']['member']).read()
            assert digest(raw) == source['original_content_sha256']
            clean, prep = preprocess(raw)
            assert prep == source['preprocessing']
            assert np.array_equal(clean, dataset.array(source['clean_ref'], (3,256,256)))
            assert sha(DATA/source['clean_ref']) == source['clean_sha256']
    expected_arrays = {s['clean_ref'] for s in sources}
    for rec in trajectories:
        assert rec['severity_values'] == [4,3,2,1]
        assert rec['clean_ref'] == dataset.sources[rec['source_id']]['clean_ref']
        assert rec['clean_sha256'] == dataset.sources[rec['source_id']]['clean_sha256']
        assert rec['simulator_identity'] == EXPECTED
        for block in ('epsilon_zernike','epsilon_tilt'):
            assert rec['seed_'+('zernike' if block=='epsilon_zernike' else 'tilt')] == xi_seed(rec['source_id'],rec['realization_index'],block)
            assert sha(DATA/rec[block+'_ref']) == rec['xi_sha256'][block]
            expected_arrays.add(rec[block+'_ref'])
        expected_arrays.update(rec[f'x{g}_ref'] for g in G_VALUES)
    actual_arrays = {str(p.relative_to(DATA)) for p in DATA.rglob('*.npy')}
    assert expected_arrays == actual_arrays and len(actual_arrays) == 2500
    assert len({r['seed_zernike'] for r in trajectories} | {r['seed_tilt'] for r in trajectories}) == 800
    assert len({r['xi_sha256']['epsilon_zernike'] for r in trajectories}) == 400
    assert len({r['xi_sha256']['epsilon_tilt'] for r in trajectories}) == 400
    for n, rec in enumerate(starts):
        loaded = dataset[n]
        t = dataset.trajectories[rec['trajectory_id']]
        g = rec['initial_G']
        assert g in G_VALUES and rec['severity_path'] == list(range(g,0,-1)) and rec['num_predictions'] == g
        assert rec['initial_state_ref'] == t[f'x{g}_ref'] and rec['clean_ref'] == t['clean_ref'] and rec['source_id'] == t['source_id']
        assert loaded['C'].shape == loaded['Y'].shape == (3,256,256)
    subset_summary = json.loads((DATA/'subsets/subset_summary.json').read_text())
    ordered_ids = sorted(dataset.trajectories, key=subset_order)
    subset_counts = {}
    for count in (1,16,400):
        prefix = DATA/f'subsets/overfit_{count}traj'
        tr = rows(str(prefix)+'_trajectories.jsonl')
        sr = rows(str(prefix)+'_starts.jsonl')
        assert [t['trajectory_id'] for t in tr] == ordered_ids[:count]
        assert len(sr) == 4*count and len({s['start_id'] for s in sr}) == 4*count
        expected_sr = [s for s in starts if s['trajectory_id'] in set(ordered_ids[:count])]
        assert sr == expected_sr and all(t == dataset.trajectories[t['trajectory_id']] for t in tr)
        subset_counts[str(count)] = dict(trajectories=count, starts=len(sr), sources=len({t['source_id'] for t in tr}))
        assert subset_counts[str(count)]['sources'] == subset_summary['subsets'][str(count)]['source_count']
        for ref, wanted in subset_summary['subsets'][str(count)]['manifest_sha256'].items():
            assert sha(DATA/ref) == wanted
        reader = OverfitDataset(DATA, f'subsets/overfit_{count}traj_starts.jsonl')
        assert len(reader) == 4*count
        reader[0]
    replay = []
    seed_rows = []
    start_time = time.monotonic()
    with torch.no_grad():
        for n, rec in enumerate(trajectories):
            xi_arrays = dataset.diagnostic_xi(rec['trajectory_id'])
            regenerated = sample(sim, rec)
            seed_check = {block: bool(np.array_equal(xi_arrays[block],cpu_output(getattr(regenerated,block)))) for block in xi_arrays}
            seed_rows.append(dict(trajectory_id=rec['trajectory_id'], **seed_check, passed=all(seed_check.values())))
            if not all(seed_check.values()):
                raise RuntimeError('Seed replay failed: '+rec['trajectory_id'])
            xi = reload_xi(sim, rec)
            clean = clean_tensor(sim, rec)
            prepared = sim.prepare_image(clean)
            for g in G_VALUES:
                stored = dataset.array(rec[f'x{g}_ref'], (3,256,256))
                rendered = cpu_output(sim.render_prepared(prepared,xi,g).output)
                metric = errors(stored,rendered)
                passed = metric['max_abs'] <= cfg['replay_tolerance']['max_abs'] and metric['MSE'] <= cfg['replay_tolerance']['MSE']
                replay.append(dict(trajectory_id=rec['trajectory_id'],g=g,**metric,**{'pass':passed}))
                if not passed:
                    save_rows(DATA/'replay_metrics.jsonl',replay)
                    raise RuntimeError('Full replay failed; thresholds will not be relaxed')
            if (n+1)%10==0 or n==0:
                save_rows(DATA/'replay_metrics.jsonl',replay)
                save_rows(DATA/'seed_replay_metrics.jsonl',seed_rows)
                progress('replaying', completed_trajectories=n+1, total_trajectories=400,
                         checked_states=len(replay),elapsed_seconds=time.monotonic()-start_time)
    verify_inputs(check_raw_archive=True)
    # Compare every generated file against its independently published per-trajectory hash.
    for rec in trajectories:
        completed=json.loads((PROJECT/'artifacts/completed_trajectories'/(rec['trajectory_id']+'.json')).read_text())
        assert completed['record']==rec
        for ref,wanted in completed['file_sha256'].items():
            assert sha(DATA/ref)==wanted
    stats = {}
    for g in G_VALUES:
        rs=[r for r in replay if r['g']==g]
        stats[str(g)]=dict(count=len(rs), max_abs=max(r['max_abs'] for r in rs),
                           max_MSE=max(r['MSE'] for r in rs), mean_MSE=float(np.mean([r['MSE'] for r in rs])),
                           exact_count=sum(r['exact'] for r in rs))
    arrays_bytes=sum((DATA/p).stat().st_size for p in expected_arrays)
    range_stats={}
    for g in G_VALUES:
        values=[json.loads((PROJECT/'artifacts/completed_trajectories'/(r['trajectory_id']+'.json')).read_text())['state_statistics'][str(g)] for r in trajectories]
        range_stats[str(g)]=dict(min=min(v['min'] for v in values),max=max(v['max'] for v in values),
                                 below_zero_fraction=float(np.mean([v['below_zero_fraction'] for v in values])),
                                 above_one_fraction=float(np.mean([v['above_one_fraction'] for v in values])))
    summary=dict(dataset_id=DATASET_ID,sources=100,clean=100,realizations=400,trajectories=400,
                 states=1600,start_paths=1600,authoritative_array_files=2500,authoritative_array_bytes=arrays_bytes,
                 simulator_output_range_statistics=range_stats,
                 severity_values=[4,3,2,1],num_predictions_semantics='total network predictions including final clean prediction; one-shot=1',
                 prediction_count_if_all_starts_run_once=4000,source_overlap=overlap,subsets=subset_counts,
                 main_manifest_sha256={name:sha(DATA/name) for name in ('source_manifest.jsonl','trajectory_manifest.jsonl','start_manifest.jsonl')},
                 theoretical_uniform_start_exposure={str(g):dict(real_start=1,predicted_state=4-g,total_predictions=5-g) for g in G_VALUES},
                 theoretical_path_mean_loss_share=[.0625,7/48,13/48,25/48],
                 theoretical_counts_are_not_a_training_policy=True,model_training_or_inference_executed=False)
    save_json(DATA/'dataset_summary.json',summary)
    report=dict(status='ready_for_overfit_training',meaning='Dataset construction and audit passed; future trainer adaptation/acceptance still required; no model run',
                audited_utc=now(),counts=summary,source_preprocessing_replay_passed=True,
                same_source_same_clean=True,same_trajectory_same_xi=True,seeds_unique=800,
                seed_replay=dict(checked_realizations=400,checked_blocks=800,exact_blocks=800),
                replay=dict(count=1600,max_abs=max(r['max_abs'] for r in replay),max_MSE=max(r['MSE'] for r in replay),
                            mean_MSE=float(np.mean([r['MSE'] for r in replay])),exact_count=sum(r['exact'] for r in replay),by_g=stats),
                frozen_tolerance=cfg['replay_tolerance'],build_config_sha256=freeze['sha256'],
                source_overlap=overlap,all_references_checked=True,subsets_nested=True,
                original_input_hashes_unchanged=True,generated_hashes_verified=True,
                replay_wall_seconds=time.monotonic()-start_time,
                simulator_parameter_gradients_none=all(p.grad is None for p in sim.parameters()))
    save_json(DATA/'integrity_report.json',report)
    text=f'''# Overfit100 数据交付审查\n\n状态：ready_for_overfit_training（仅数据就绪）\n\n100 sources / 100 clean / 400 xi / 400 trajectories / 1600 FP32 states / 1600 starts。\n\n原 train source pool 固定抽样；与原 val/test source overlap 为 0。每个 source 四条 trajectory 共用一个确定性 clean；同 trajectory 四档共用原生 xi。\n\n路径：[4,3,2,1] / [3,2,1] / [2,1] / [1]；num_predictions 为 4/3/2/1，one-shot=1。不生成 g=0。\n\n1600 状态全量 replay：max_abs={report['replay']['max_abs']:.12g}，max MSE={report['replay']['max_MSE']:.12g}，mean MSE={report['replay']['mean_MSE']:.12g}，完全相等 {report['replay']['exact_count']}/1600。预冻结容差 max_abs≤1e-6、MSE≤1e-12。\n\n400 个 xi 的 800 块 seed replay 全部逐元素完全一致。原始输入、代码和 assets 前后 hash 一致。\n\n子集：{json.dumps(subset_counts,ensure_ascii=False)}；集合嵌套。\n\n权威数组 {arrays_bytes} bytes；FP32、无 clamp。没有模型训练或 inference。\n\n服务器：{platform.node()}；物理 NPU 0；数据：{DATA}。\n\n顶层 hash_manifest.json 覆盖数据、manifests 和本报告，排除自身。它的 SHA256 在项目 artifacts/delivery.json。\n'''
    atomic(DATA/'integrity_report.md',text.encode())
    files={str(p.relative_to(DATA)):sha(p) for p in sorted(DATA.rglob('*')) if p.is_file() and p.name!='hash_manifest.json' and not p.name.endswith('.tmp')}
    save_json(DATA/'hash_manifest.json',dict(algorithm='SHA256',root='dataset root',excludes=['hash_manifest.json'],files=files))
    for ref,wanted in files.items():
        if sha(DATA/ref)!=wanted:
            raise RuntimeError('Final hash verification failed: '+ref)
    delivery=dict(status='ready_for_overfit_training',dataset_id=DATASET_ID,dataset_root=str(DATA),
                  completed_utc=now(),physical_device=0,host=platform.node(),
                  hash_manifest_sha256=sha(DATA/'hash_manifest.json'),hash_manifest_files=len(files),
                  total_dataset_bytes=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()),
                  main_manifest_sha256=summary['main_manifest_sha256'],replay=report['replay'],subsets=subset_counts,
                  run_manifest_sha256=sha(PROJECT/'artifacts/run_manifest.json'))
    save_json(PROJECT/'artifacts/delivery.json',delivery)
    progress('ready_for_overfit_training',**{k:v for k,v in delivery.items() if k!='status'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','preflight','build','audit','build-and-audit'])
    args=parser.parse_args()
    PROJECT.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage=='build-and-audit':
            build()
            audit()
        else:
            globals()[args.stage]()
    except BaseException as exc:
        save_json(PROJECT/'artifacts'/f'failure_{args.stage}_{int(time.time())}.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc(),command=sys.argv))
        raise


if __name__=='__main__':
    main()

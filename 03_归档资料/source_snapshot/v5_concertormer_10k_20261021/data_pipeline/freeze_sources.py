"""Freeze V5 clean sources; preserve historical splits and expand test safely.

CPU only. No old observation, raw xi, or recovery model is read.
"""
import argparse
import collections
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import COUNTS, DATASET_ID, SIMULATOR_SHA256, atomic, digest, npy_bytes, rows, sha, verify_array, write_json, write_rows
from source_preprocess import preprocess


def fingerprint(raw):
    # Same exact fingerprint definition as the historical source audit.
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert('L')
        d = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
        dh = sum(int(v) << i for i, v in enumerate((d[:, 1:] > d[:, :-1]).flat))
        array = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    basis = np.cos(np.pi * (np.arange(32) + .5) * np.arange(8)[:, None] / 32)
    dc = basis @ array @ basis.T
    ph = sum(int(v) << i for i, v in enumerate((dc > np.median(dc.flat[1:])).flat))
    return ph, dh


def key(value):
    return digest(f'{DATASET_ID}|20261021|{value}'.encode())


def freeze(a):
    root, old, expanded = Path(a.root), Path(a.old_root), Path(a.expanded_root)
    root.mkdir(parents=True, exist_ok=True)
    old_freeze = json.loads((old / 'SOURCE_FREEZE.json').read_text())
    assert sha(old / 'source_manifest.jsonl') == old_freeze['source_manifest_sha256']
    historical = rows(old / 'source_manifest.jsonl')
    assert collections.Counter(x['split'] for x in historical) == {'train': 2000, 'val': 250, 'test': 200}
    expanded_freeze = json.loads((expanded / 'SOURCE_FREEZE.json').read_text())
    assert sha(expanded / 'source_manifest.jsonl') == expanded_freeze['source_manifest_sha256']
    universe = {x['source_id']: x for x in rows(expanded / 'source_manifest.jsonl')}
    for x in historical:
        previous = universe.get(x['source_id'])
        assert previous is not None and previous['split'] == x['split']
        assert previous['original_sha256'] == x['original_sha256']
        assert previous['clean_pixel_sha256'] == x['clean_pixel_sha256']
    # All prior sources (including LSDIR) are exclusions for *new* test sources.
    original_hashes = {x['original_sha256'] for x in universe.values()}
    pixel_hashes = {x['clean_pixel_sha256'] for x in universe.values()}
    fingerprints = [(int(x['phash64'], 16), int(x['dhash64'], 16), x['source_id']) for x in universe.values()]
    selected = []
    for x in historical:
        raw = (old / x['original_relative_path']).read_bytes()
        assert digest(raw) == x['original_sha256']
        clean = verify_array(old / x['clean']['ref'], x['clean']['sha256'])
        assert digest(clean.tobytes()) == x['clean_pixel_sha256']
        replay, _ = preprocess(raw)
        assert np.array_equal(clean, replay), ('preprocessing drift', x['source_id'])
        atomic(root / x['clean']['ref'], (old / x['clean']['ref']).read_bytes())
        atomic(root / x['original_relative_path'], raw)
        selected.append({**x, 'original_path': str(root / x['original_relative_path']),
                         'dataset_source': 'Places365', 'prior_split': x['split'],
                         'clean_reused_from': str(old), 'observations_reused': False})
    categories = json.loads((old / 'categories.json').read_text())
    labels = {s.split()[0]: int(s.split()[1]) for s in (old / 'metadata/places365_val.txt').read_text().splitlines()}
    archive = Path(a.archive)
    assert sha(archive) == old_freeze['source_archive_sha256']
    rejections = []
    with tarfile.open(archive) as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for category in categories:
            candidates = sorted((sid for sid, c in labels.items() if c == category['category_id'] and sid not in universe), key=lambda sid: key('new_test|' + sid))
            accepted = False
            for sid in candidates:
                raw = tar.extractfile(members[sid]).read()
                clean, prep = preprocess(raw)
                original_hash, pixel_hash = digest(raw), digest(clean.tobytes())
                ph, dh = fingerprint(raw)
                near = next((other for pp, dd, other in fingerprints if (ph ^ pp).bit_count() <= 4 and (dh ^ dd).bit_count() <= 4), None)
                if original_hash in original_hashes or pixel_hash in pixel_hashes or near:
                    rejections.append({'source_id': sid, 'reason': 'exact_or_joint_phash4_dhash4_duplicate', 'near_duplicate_of': near})
                    continue
                original_ref, clean_ref = 'original/' + sid, 'clean/' + sid + '.npy'
                encoded = npy_bytes(clean)
                atomic(root / original_ref, raw)
                atomic(root / clean_ref, encoded)
                x = {'source_id': sid, 'source_index': len(selected), 'split': 'test', 'prior_split': None,
                     'dataset_source': 'Places365', 'archive_member': members[sid].name,
                     'original_path': str(root / original_ref), 'original_relative_path': original_ref,
                     'original_sha256': original_hash, 'original_bytes': len(raw),
                     'clean_pixel_sha256': pixel_hash, 'phash64': f'{ph:016x}', 'dhash64': f'{dh:016x}',
                     'preprocessing': prep, 'clean_reused_from': None, 'observations_reused': False,
                     'clean': {'ref': clean_ref, 'sha256': digest(encoded), 'shape': [3, 256, 256], 'dtype': 'float32', 'bytes': len(encoded)}, **category}
                selected.append(x)
                original_hashes.add(original_hash)
                pixel_hashes.add(pixel_hash)
                fingerprints.append((ph, dh, sid))
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f'No independent unused test source remains in category {category}')
    assert collections.Counter(x['split'] for x in selected) == COUNTS
    for field in ['source_id', 'archive_member', 'original_sha256', 'clean_pixel_sha256']:
        assert len({x[field] for x in selected}) == 2500, field
    for category in categories:
        assert collections.Counter(x['split'] for x in selected if x['category_id'] == category['category_id']) == {'train': 40, 'val': 5, 'test': 5}
    # Globally audit exact and heuristic pixel duplication independently of split.
    near_pairs = []
    for i, x in enumerate(selected):
        ph, dh = int(x['phash64'], 16), int(x['dhash64'], 16)
        for y in selected[:i]:
            if (ph ^ int(y['phash64'], 16)).bit_count() <= 4 and (dh ^ int(y['dhash64'], 16)).bit_count() <= 4:
                near_pairs.append([x['source_id'], y['source_id']])
    assert not near_pairs, near_pairs
    plan, seeds = [], set()
    for source in selected:
        for realization_index in range(4):
            seed_z, seed_t = [int(key(f'raw_seed|{source["source_id"]}|{realization_index}|{kind}')[:16], 16) % (2 ** 63 - 1) for kind in ['zernike', 'tilt']]
            assert seed_z != seed_t and not {seed_z, seed_t} & seeds
            seeds.update([seed_z, seed_t])
            sample_id = f'{source["split"]}_s{source["source_index"]:04d}_r{realization_index:02d}'
            plan.append({'sample_id': sample_id, 'realization_id': sample_id, 'source_id': source['source_id'],
                         'source_index': source['source_index'], 'split': source['split'], 'realization_index': realization_index,
                         'seed_zernike': seed_z, 'seed_tilt': seed_t, 'G': 4.0,
                         'clean_ref': source['clean']['ref'], 'clean_sha256': source['clean']['sha256'],
                         'observation_ref': f'observations/{source["split"]}/{sample_id}.npy',
                         'record_ref': f'records/{source["split"]}/{sample_id}.json'})
    write_rows(root / 'source_manifest.jsonl', selected)
    write_rows(root / 'sample_plan.jsonl', plan)
    atomic(root / 'categories.json', (old / 'categories.json').read_bytes())
    config = {'dataset_id': DATASET_ID, 'source_manifest_sha256': sha(root / 'source_manifest.jsonl'),
              'sample_plan_sha256': sha(root / 'sample_plan.jsonl'), 'expected_sources': COUNTS,
              'expected_samples': {k: v * 4 for k, v in COUNTS.items()}, 'G': 4.0,
              'old_clean_root': str(old), 'old_source_manifest_sha256': sha(old / 'source_manifest.jsonl'),
              'new_test_exclusion_root': str(expanded), 'new_test_exclusion_manifest_sha256': sha(expanded / 'source_manifest.jsonl'),
              'original_archive': str(archive), 'original_archive_sha256': sha(archive),
              'preprocessing_file_sha256': sha(Path(__file__).with_name('source_preprocess.py')),
              'selection_seed': 20261021, 'selection_rule': 'Preserve historical 2450 source/split identities; one SHA-ranked unused exact/near-distinct source per original category added to test; exclude all expanded historical sources.',
              'simulator_sha256': SIMULATOR_SHA256, 'physics': old_freeze['physics'],
              'raw_rng': 'Per sample and per kind torch.Generator(device=cpu).manual_seed; torch.randn FP32; B=1 shape fixed; torch version pinned by GENERATION_IDENTITY.json; raw SHA recorded; raw arrays not stored.',
              'raw_shapes': {'epsilon_zernike': [1, 36, 512, 512], 'epsilon_tilt': [1, 512, 512, 2]},
              'old_observations_reused': False, 'test_model_evaluation': False, 'random_augmentation': False,
              'scientific_target': 'Places fixed 50 classes, same-category unseen-image generalization',
              'clamp_observations': False, 'observation_dtype': 'float32', 'observation_layout': 'CHW'}
    write_json(root / 'SOURCE_FREEZE.json', config)
    write_json(root / 'SOURCE_AUDIT.json', {'status': 'PASS', 'source_count': 2500, 'exact_duplicate_count': 0,
               'joint_phash4_dhash4_duplicate_count': 0, 'old_clean_preprocessing_replay_exact': 2450,
               'old_splits_preserved': 2450, 'new_test_sources_never_in_expanded_historical_selection': 50,
               'new_test_exclusions': rejections, 'heuristic_limit': 'Exact pixels/IDs/bytes audited; pHash/dHash cannot rule out all recrops or semantic similarity.',
               'test_recovery_model_evaluation': False})
    print(json.dumps({'status': 'SOURCE_READY', 'root': str(root), 'sources': 2500, 'samples_planned': 10000}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--old-root', default='/data2/yanglin/datasets/places_p2s_canonical_20261003_c50_s2000_v250_t200_g4321_v1')
    parser.add_argument('--expanded-root', default='/data2/yanglin/datasets/places_lsdir_p2s_canonical_20261006_p4000_l4000_v1000_t200_g4321_v1')
    parser.add_argument('--archive', default='/data2/yanglin/physical_bridge/data/source_archives/val_256.tar')
    freeze(parser.parse_args())

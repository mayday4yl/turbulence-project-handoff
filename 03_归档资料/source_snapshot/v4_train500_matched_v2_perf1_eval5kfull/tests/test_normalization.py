"""CPU numerical and data-contract checks; these do not validate the simulator."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from v4.data import PhysicalDataset, read_array, sha256_file
from v4.normalization import (PROTOCOL, ChannelMoments, FrozenRGBStats,
                              StratifiedRGBStatistics, build_artifact,
                              load_artifact, save_artifact)


class StatisticsTests(unittest.TestCase):
    @staticmethod
    def arrays():
        rng = np.random.default_rng(7)
        return {(tid, g): rng.normal(g * 0.7, g * 0.2, (3, 3, 7))
                for tid in ['a', 'b'] for g in range(1, 5)}

    def run_statistics(self, block_pixels):
        accumulator = StratifiedRGBStatistics(['a', 'b'], block_pixels=block_pixels)
        for (tid, g), array in self.arrays().items():
            accumulator.add(tid, g, array, split='train')
        return accumulator.finalize()

    def test_direct_population_and_block_partition(self):
        results = [self.run_statistics(block) for block in [1, 5, 10000]]
        arrays = self.arrays()
        direct = np.concatenate([a.reshape(3, -1) for a in arrays.values()], axis=1)
        for result in results:
            np.testing.assert_allclose(result['mu'], direct.mean(axis=1), atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(result['variance'], direct.var(axis=1, ddof=0), atol=1e-14, rtol=1e-14)
        for result in results[1:]:
            np.testing.assert_allclose(result['mu'], results[0]['mu'], atol=1e-14)
            np.testing.assert_allclose(result['std'], results[0]['std'], atol=1e-14)

    def test_declared_weighted_normalized_population_has_zero_mean_unit_variance(self):
        result = self.run_statistics(5)
        mu = np.asarray(result['mu'])
        std = np.asarray(result['std'])
        group_means, group_second_moments = [], []
        arrays = self.arrays()
        for g in range(1, 5):
            pixels = np.concatenate([arrays[(tid, g)].reshape(3, -1) for tid in ['a', 'b']], axis=1)
            normalized = (pixels - mu[:, None]) / std[:, None]
            group_means.append(normalized.mean(1))
            group_second_moments.append((normalized ** 2).mean(1))
        weighted_mean = np.mean(group_means, axis=0)
        weighted_variance = np.mean(group_second_moments, axis=0) - weighted_mean ** 2
        np.testing.assert_allclose(weighted_mean, np.zeros(3), atol=1e-14)
        np.testing.assert_allclose(weighted_variance, np.ones(3), atol=1e-14)

    def test_between_group_mean_difference_is_included(self):
        accumulator = StratifiedRGBStatistics(['a'])
        for g in range(1, 5):
            accumulator.add('a', g, np.full((3, 2, 2), float(g)), split='train')
        result = accumulator.finalize()
        np.testing.assert_array_equal(result['mu'], [2.5] * 3)
        np.testing.assert_array_equal(result['variance'], [1.25] * 3)
        self.assertTrue(all(group['variance'] == [0.0] * 3 for group in result['groups'].values()))

    def test_equal_g_weight_despite_unequal_pixel_counts(self):
        accumulator = StratifiedRGBStatistics(['a'])
        for g in range(1, 5):
            accumulator.add('a', g, np.full((3, g, 1), float(g)), split='train')
        result = accumulator.finalize()
        np.testing.assert_allclose(result['mu'], [2.5] * 3)
        np.testing.assert_allclose(result['variance'], [1.25] * 3)

    def test_pixel_weight_inside_group(self):
        accumulator = StratifiedRGBStatistics(['a', 'b'])
        for g in range(1, 5):
            accumulator.add('a', g, np.zeros((3, 1, 1)), split='train')
            accumulator.add('b', g, np.ones((3, 1, 3)) * 4, split='train')
        result = accumulator.finalize()
        np.testing.assert_array_equal(result['mu'], [3] * 3)
        np.testing.assert_array_equal(result['variance'], [3] * 3)

    def test_duplicate_conflict_nontrain_unknown_missing_and_nonfinite_rejected(self):
        array = np.zeros((3, 2, 2))
        accumulator = StratifiedRGBStatistics(['a'])
        accumulator.add('a', 1, array, split='train')
        for value in [array, array + 1]:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                accumulator.add('a', 1, value, split='train')
        for split in ['val', 'test']:
            with self.assertRaisesRegex(ValueError, 'Non-train'):
                accumulator.add('a', 2, array, split=split)
        with self.assertRaisesRegex(ValueError, 'unauthorized'):
            accumulator.add('unknown', 2, array, split='train')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            accumulator.finalize()
        for bad in [np.nan, np.inf, -np.inf]:
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                accumulator.add('a', 2, array + bad, split='train')

    def test_population_variance_and_std_floor(self):
        moment = ChannelMoments()
        moment.update(np.array([[0, 2]] * 3))
        np.testing.assert_array_equal(moment.variance, [1.0] * 3)
        accumulator = StratifiedRGBStatistics(['a'])
        for g in range(1, 5):
            accumulator.add('a', g, np.ones((3, 1, 1)), split='train')
        np.testing.assert_array_equal(accumulator.finalize()['std'], [1e-6] * 3)

    def test_frozen_buffers_checkpoint_hash_and_tamper_rejection(self):
        payload = {'protocol': PROTOCOL, **self.run_statistics(5)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'statistics.json'
            artifact = save_artifact(payload, path)
            stats = FrozenRGBStats.from_artifact(path)
            self.assertEqual(len(list(stats.parameters())), 0)
            self.assertEqual(stats.artifact_hash, artifact['artifact_hash'])
            memory = io.BytesIO()
            torch.save(stats.state_dict(), memory)
            memory.seek(0)
            restored = FrozenRGBStats([0, 0, 0], [1, 1, 1])
            restored.load_state_dict(torch.load(memory, map_location='cpu', weights_only=True))
            torch.testing.assert_close(restored.mu, stats.mu, rtol=0, atol=0)
            torch.testing.assert_close(restored.s, stats.s, rtol=0, atol=0)
            self.assertEqual(restored.artifact_hash, stats.artifact_hash)
            artifact['mu'][0] += 0.01
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_artifact(path)

    def test_zero_noise_formula_inverse_formula_and_no_clamp(self):
        stats = FrozenRGBStats([0.2, 0.4, 0.6], [0.1, 0.2, 0.3]).double()
        X = torch.tensor([-.2, 0.8, 1.3], dtype=torch.float64).reshape(1, 3, 1, 1).expand(2, 3, 2, 2)
        eta = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2) - 12
        torch.testing.assert_close(stats.mix(X, eta, 0), X, rtol=0, atol=0)
        t = torch.tensor([0.8, 0.4], dtype=torch.float64)
        Z = stats.mix(X, eta, t)
        t_image = t[:, None, None, None]
        inverse = stats.mu + stats.s * ((1 - t_image) * ((X - stats.mu) / stats.s) + t_image * eta)
        torch.testing.assert_close(Z, inverse, rtol=1e-14, atol=1e-14)
        self.assertLess(Z.min().item(), 0)
        self.assertGreater(Z.max().item(), 1)
        with self.assertRaises(ValueError):
            stats.mix(X, eta * float('nan'), t)


class DatasetTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / 'subsets').mkdir()
        records = {}
        for name in ['clean', 'X_1', 'X_2', 'X_3', 'X_4']:
            array = np.empty((3, 256, 256), dtype=np.float32)
            array[:] = np.array([-0.1, 0.5, 1.1], dtype=np.float32)[:, None, None]
            if name != 'clean':
                array += int(name[-1]) * .1
            path = root / f'{name}.npy'
            np.save(path, array, allow_pickle=False)
            records[name] = {'ref': path.name, 'sha256': sha256_file(path), 'dtype': 'float32',
                             'shape': list(array.shape), 'bytes': path.stat().st_size}
        source = {'source_id': 'source_a', 'split': 'train', 'clean': records['clean'],
                  'preprocessing': {'layout': 'CHW', 'convert_mode': 'RGB', 'dtype': 'float32',
                                    'scale': 'uint8 / 255.0'}}
        row = {'trajectory_id': 'train_a_r00', 'source_id': 'source_a', 'split': 'train',
               'source': source, 'arrays': records}
        (root / 'source_manifest.jsonl').write_text(json.dumps(source) + '\n')
        (root / 'trajectory_manifest.jsonl').write_text(json.dumps(row) + '\n')
        (root / 'subsets/overfit100.json').write_text(json.dumps({'source_ids': ['source_a'],
                                                              'trajectory_ids': ['train_a_r00']}))
        (root / 'BUILD_IDENTITY.json').write_text(json.dumps({
            'config': {'dataset_id': 'synthetic_cpu_fixture', 'clamp': False,
                       'additional_bridge_noise_baked_into_dataset': False},
            'source_manifest_sha256': 'pre_generation_plan_has_a_distinct_hash',
        }))
        names = ['BUILD_IDENTITY.json', 'source_manifest.jsonl', 'trajectory_manifest.jsonl',
                 'subsets/overfit100.json']
        lock = root / 'asset_lock.json'
        lock.write_text(json.dumps({'dataset_identities': {name: {'sha256': sha256_file(root/name)}
                                                         for name in names}}))
        return records, source, row, lock

    def test_real_reader_path_artifact_and_no_input_clamp(self):
        with tempfile.TemporaryDirectory() as directory:
            records, source, row, lock = self.fixture(directory)
            dataset = PhysicalDataset(directory, asset_lock=lock)
            sample = dataset.sample('train_a_r00', 4)
            self.assertGreater(sample['Y'].max().item(), 1)
            self.assertLess(sample['C'].min().item(), 0)
            self.assertEqual(sample['Y'].dtype, torch.float32)
            artifact_path = Path(directory) / 'stats.json'
            artifact = build_artifact(directory, artifact_path, asset_lock=lock)
            self.assertEqual(artifact['state_count'], 4)
            self.assertEqual(artifact['population'], 'overfit100')
            self.assertTrue(artifact['asset_lock_verified'])
            self.assertEqual(len(artifact['statistics_state_manifest']), 4)
            self.assertEqual(FrozenRGBStats.from_artifact(artifact_path).artifact_hash,
                             artifact['artifact_hash'])
            with self.assertRaises(ValueError):
                dataset.sample('val_a_r00', 1)
            with self.assertRaises(ValueError):
                dataset.sample('train_a_r00', 0)
            bad = copy.deepcopy(records['X_1']); bad['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                read_array(directory, bad)
            bad['ref'] = '../outside.npy'
            with self.assertRaisesRegex(ValueError, 'escapes'):
                read_array(directory, bad)
            # A plan hash and a generated-manifest hash must not be conflated.
            source['split'] = 'val'
            (Path(directory) / 'source_manifest.jsonl').write_text(json.dumps(source)+'\n')
            with self.assertRaises(ValueError):
                PhysicalDataset(directory, asset_lock=lock)

    def test_invalid_population_duplicate_missing_and_nontrain(self):
        with tempfile.TemporaryDirectory() as directory:
            records, source, row, lock = self.fixture(directory)
            with self.assertRaises(ValueError):
                PhysicalDataset(directory, 'test')
            manifest = Path(directory) / 'trajectory_manifest.jsonl'
            manifest.write_text((json.dumps(row)+'\n') * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                PhysicalDataset(directory)
            broken = copy.deepcopy(row); del broken['arrays']['X_4']
            manifest.write_text(json.dumps(broken)+'\n')
            with self.assertRaisesRegex(ValueError, 'exactly g'):
                PhysicalDataset(directory)
            broken = copy.deepcopy(row); broken['split'] = 'val'
            manifest.write_text(json.dumps(broken)+'\n')
            with self.assertRaisesRegex(ValueError, 'Non-train'):
                PhysicalDataset(directory)


if __name__ == '__main__':
    unittest.main(verbosity=2)

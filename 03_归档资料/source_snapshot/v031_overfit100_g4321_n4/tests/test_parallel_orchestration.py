"""CPU-only scheduling, stop boundary, failed-child and ownership checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_parallel_to_1000 as parallel
from guard_distributed import OwnedProcessTree, check_snapshot


class ParallelTests(unittest.TestCase):
    cfg = {'training': {'microbatch': 1}, 'evaluation': {'batch_size': 1},
           'orchestration': {'full_evaluation_timeout_seconds': 43200,
                             'subset_evaluation_timeout_seconds': 7200,
                             'train_segment_timeout_seconds': 16037}}

    def simulate(self, *, fail=False):
        launched, evaluated, updates, reports = [], set(), [], []
        current = {'target': 500}

        def spawn(label, argv, attempt, timeout):
            launched.append((label, set(evaluated)))
            return {'launch': {'identity': {'pid': len(launched)}}, 'label': label, 'ticks': 0}

        def poll(job):
            job['ticks'] += 1
            if fail and job['label'].startswith('eval_'):
                raise RuntimeError('synthetic child failure')
            if job['ticks'] < (5 if job['label'].startswith('eval_') else 2):
                return False
            if job['label'].startswith('train_'):
                current['target'] = int(job['label'].split('_')[1])
                job['result'] = {'returncode': 0}
            else:
                evaluated.add(job['key'])
            return True

        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(parallel, ROOT=Path(directory), verify=lambda: self.cfg,
                                run_preflight=lambda attempt: None, last_committed=lambda: 500,
                                spawn=spawn, poll=poll, verify_segment=lambda target, result: 'sha',
                                verify_evaluation=lambda cfg, target, weights: (target, weights) in evaluated,
                                update=lambda **fields: updates.append(fields)), \
                 patch.object(parallel.frozen, 'read', side_effect=lambda path: {'global_update': current['target']}), \
                 patch.object(parallel.frozen, 'write'), patch.object(parallel.frozen, 'digest', return_value='sha'), \
                 patch.object(parallel.subprocess, 'run', side_effect=lambda argv, **kw: reports.append(int(argv[-1]))), \
                 patch.object(parallel.time, 'sleep'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic child failure'):
                        parallel.worker(1)
                else:
                    parallel.worker(1)
        return launched, evaluated, updates, reports

    def test_training_overlaps_older_evaluation_and_stops_at1000(self):
        launched, evaluated, updates, reports = self.simulate()
        training = [(name, done) for name, done in launched if name.startswith('train_')]
        self.assertEqual([name for name, _ in training], [f'train_{n:06d}' for n in range(600, 1001, 100)])
        self.assertNotIn((500, 'ema'), training[0][1])
        self.assertEqual(evaluated, {(n, w) for n in range(500, 1001, 100) for w in ('raw', 'ema')})
        self.assertEqual(reports, [500, 1000])
        self.assertTrue(any(x.get('phase') == 'parallel_training_and_evaluation' for x in updates))

    def test_failed_child_stops_new_work(self):
        launched, _, _, reports = self.simulate(fail=True)
        self.assertEqual([name for name, _ in launched], ['eval_000500_raw', 'train_000600'])
        self.assertEqual(reports, [])

    def test_world_and_boundary_are_preserved(self):
        for n in range(600, 1001, 100):
            args = parallel.train_command(self.cfg, n)
            self.assertIn('--nproc_per_node=4', args)
            self.assertEqual(args[args.index('--microbatch') + 1], '1')
            self.assertTrue(args[args.index('--resume') + 1].endswith(f'checkpoint_{n-100:06d}.pt'))
        for n in (0, 500, 601, 1100):
            with self.assertRaises(ValueError):
                parallel.train_command(self.cfg, n)

    def test_same_guard_accepts_siblings_but_rejects_foreign_and_reused_pid(self):
        def info(pid, parent, start):
            return {'pid': pid, 'ppid': parent, 'starttime_ticks': start,
                    'pgid': pid, 'session': pid, 'state': 'S'}
        table = {1: info(1, 0, 10), 2: info(2, 1, 11), 3: info(3, 1, 12)}
        own = OwnedProcessTree(table[1])
        own.refresh(table)
        processes = [{**table[p], 'memory_mb': 1500} for p in (2, 3)]
        snap = {'cards': [{'card': 0, 'health': 'OK', 'processes': processes}]}
        check_snapshot(snap, pgid=1, memory_limit_gib=61, ownership=own)
        processes.append({**info(4, 99, 13), 'memory_mb': 1})
        with self.assertRaisesRegex(RuntimeError, 'External or unowned'):
            check_snapshot(snap, pgid=1, ownership=own)
        processes.pop()
        processes[0]['starttime_ticks'] += 1
        with self.assertRaisesRegex(RuntimeError, 'External or unowned'):
            check_snapshot(snap, pgid=1, ownership=own)

    def test_partial_tail_is_never_counted_as_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'updates.jsonl'
            path.write_bytes(b'{"global_update":500}\n{"global_update":50')
            self.assertEqual(parallel.read_lines(path, allow_incomplete_tail=True), [{'global_update': 500}])
            with self.assertRaises(json.JSONDecodeError):
                parallel.read_lines(path)


if __name__ == '__main__':
    unittest.main()

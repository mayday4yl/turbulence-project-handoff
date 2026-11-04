"""CPU checks of coverage, numerical boundaries, frozen entry points and failures."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import sharded_eval_support as support
import run_sharded_evaluation as runner
from guard_distributed import check_snapshot, OwnedProcessTree


def starts():
    return [dict(start_id=f's{i}g{g}',trajectory_id=f's{i}',initial_G=g) for i in range(100) for g in range(1,5)]


def valid_row():
    return dict(start_id='s0g1',trajectory_id='s0',initial_G=1,weights='raw',mode='unrefined',
        global_update=1000,batch_size=1,predictions=[dict(K=1,current_g=1,clean_mse=.01,
        clean_psnr_db=20.,network_redegradation_mse=.002,refinement=None)])


class ShardTests(unittest.TestCase):
    def test_partition_preserves_all_work_and_completed_rows(self):
        rows=starts();existing={('raw','unrefined',r['start_id']) for r in rows}
        existing |= {('raw','refined',r['start_id']) for r in rows[:31]}
        assignment,loads=support.partition(rows,existing)
        actual=[k for values in assignment.values() for k in values]
        self.assertEqual(len(actual),4*len(rows)-len(existing))
        self.assertEqual(len(actual),len(set(actual)))
        self.assertFalse(set(actual)&existing)
        self.assertLessEqual(max(loads.values())-min(loads.values()),9.8)
        self.assertEqual(support.partition(rows,existing),(assignment,loads))

    def test_empty_remaining_work_does_not_create_jobs(self):
        rows=starts();existing={(w,m,r['start_id']) for w in ['raw','ema'] for m in ['unrefined','refined'] for r in rows}
        assignment,loads=support.partition(rows,existing)
        self.assertFalse(any(assignment.values()));self.assertEqual(sum(loads.values()),0)

    def test_unknown_reused_identity_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'coverage'):
            support.partition(starts(),{('raw','refined','unknown')})

    def test_rows_reject_incomplete_nonfinite_and_wrong_weight(self):
        source=starts()[0];row=valid_row();support.validate_row(row,source,'raw','unrefined')
        for key,value in [('batch_size',2),('global_update',1100),('weights','ema'),('predictions',[])]:
            wrong=deepcopy(row);wrong[key]=value
            with self.assertRaises(RuntimeError):support.validate_row(wrong,source,'raw','unrefined')
        wrong=deepcopy(row);wrong['predictions'][0]['clean_mse']=float('nan')
        with self.assertRaises(RuntimeError):support.validate_row(wrong,source,'raw','unrefined')

    def test_merge_rejects_duplicate_before_summaries(self):
        row=valid_row();source=starts()[0];groups={}
        support.add_unique(groups,row,source,'raw','unrefined')
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):
            support.add_unique(groups,row,source,'raw','unrefined')

    def test_evaluation_entry_is_fixed_checkpoint_batch_and_solver(self):
        job=dict(weights='ema',manifest='subsets/example.jsonl',output='/tmp/unused',mode='refined')
        argv=runner.job_args(job)
        self.assertEqual(argv[argv.index('--batch-size')+1],'1')
        self.assertTrue(argv[argv.index('--checkpoint')+1].endswith('checkpoint_001000.pt'))
        self.assertEqual(argv[argv.index('--numeric-config')+1],'artifacts/calibration_update100/calibration.json')
        self.assertNotIn('train_worker.py',argv)

    def test_failed_child_is_not_marked_complete(self):
        class Process:
            def poll(self):return 2
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);(folder/'worker.log').write_text('synthetic failure')
            stream=(folder/'dummy').open('w')
            with patch.object(runner.frozen,'write'):
                with self.assertRaisesRegex(RuntimeError,'failed'):
                    runner.finished(dict(process=Process(),stream=stream,folder=folder,
                        record={'label':'probe'},started=0))


if __name__=='__main__':unittest.main()

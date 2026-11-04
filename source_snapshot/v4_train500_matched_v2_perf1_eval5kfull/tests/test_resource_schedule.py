"""CPU-only scheduling and auxiliary metric tests; no CUDA correctness claims."""
import unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from v4.scheduling import relocation_candidate
from select_resource_schedule import simulate
from refinement_ablation import paired_rows

class ResourceTests(unittest.TestCase):
    def test_move_only_to_empty_authorized_card(self):
        self.assertIsNone(relocation_candidate({'D0':5,'R08':5,'R0':6,'L08':6},{}))
        self.assertEqual(relocation_candidate({'R0':6,'L08':6},{'R0':3000,'L08':2800}),('L08',5))
        self.assertIsNone(relocation_candidate({'R0':6},{}))
        with self.assertRaises(ValueError):relocation_candidate({'D0':4},{})
    def test_makespan_uses_parallel_cards_and_dynamic_finish(self):
        solo={'D0':1.,'R08':3.,'R0':3.,'L08':3.};shared={g:2*s for g,s in solo.items()}
        fixed,_=simulate(solo,shared,100,'concurrent',False)
        dynamic,moves=simulate(solo,shared,100,'concurrent',True)
        self.assertLess(dynamic,fixed);self.assertGreaterEqual(dynamic,500.)
        self.assertTrue(moves)
    def test_auxiliary_metric_direction_and_cluster(self):
        off=[];on=[]
        for s in range(100):
            for xi in range(2):
                for g in range(1,5):
                    row={'start_id':f'{s}/{xi}/{g}','source_id':str(s),'trajectory_id':f'{s}/{xi}',
                         'split':'val','G':g,'psnr':20.,'ssim':.6,'lpips':.3}
                    off.append(row);on.append({**row,'psnr':21.,'ssim':.61,'lpips':.28})
        result,matrix=paired_rows(on,off)
        self.assertEqual(matrix.shape,(10000,100))
        self.assertTrue(all(r['source_win_rate']==1. for r in result))
        self.assertAlmostEqual(result[-1]['mean_paired_delta'],-.02)
        with self.assertRaises(AssertionError):paired_rows(on[:-1],off)

if __name__=='__main__':unittest.main()

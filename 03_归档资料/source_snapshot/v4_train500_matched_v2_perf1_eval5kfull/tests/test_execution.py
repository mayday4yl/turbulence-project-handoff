"""CPU tests of stage/budget/report decisions; no model/simulator claims."""
from pathlib import Path
import importlib.util,json,sys,tempfile,threading,time,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from stage_report import paired_table,progression_table,rules
from v4.stages import verify_budget_scope,StageBarrier,late_stop,GROUPS
from v4.runtime import save_json,sha256

class ExecutionTests(unittest.TestCase):
    def test_existing_budget_is_never_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'protocol').mkdir()
            expected={g:10000 for g in GROUPS}
            save_json(root/'protocol/TRAINING_PLAN_SPEC.json',{'budget':{'stage1_updates_per_group':expected}})
            save_json(root/'protocol/AUTHORIZATION.json',{'updates_per_method':10000,'methods':list(GROUPS)})
            self.assertEqual(verify_budget_scope(root),expected)
            save_json(root/'protocol/FROZEN.json',{'updates':1000,'updates_per_group':{g:1000 for g in GROUPS}})
            before=(root/'protocol/FROZEN.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'BUDGET_CONFLICT'):verify_budget_scope(root)
            self.assertEqual(before,(root/'protocol/FROZEN.json').read_bytes())
            save_json(root/'protocol/FROZEN.json',{'updates':10000,'updates_per_group':expected})
            self.assertEqual(verify_budget_scope(root),expected)
    def test_stage_releases_lease_before_wait_and_obeys_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/'CONTROL.json'
            save_json(control,{'stop_at_common_update':5000,'stop_at_next_boundary':False,'authorized_default_endpoint':10000})
            ckpt=root/'checkpoint';ckpt.write_text('fixture')
            commit=root/'eval_commit';commit.write_text('fixture')
            manifest=root/'delivery';manifest.write_text('fixture')
            events=[]
            class Mutex:
                def park(self,u):events.append(('park',u))
            def release():
                while not (root/'stages/U05000/D0.READY.json').exists():time.sleep(.01)
                self.assertEqual(events,[('park',5000)])
                save_json(root/'stages/U05000/RELEASE.json',{'update':5000,'all_four_complete':True,'action':'stop',
                          'delivery_manifest':str(manifest),'delivery_manifest_sha256':sha256(manifest)})
            t=threading.Thread(target=release);t.start()
            stage=StageBarrier(root/'stages','D0',control,root/'PAUSED.json')
            self.assertEqual(stage.wait(5000,ckpt,commit,Mutex()),'stop');t.join()
            self.assertTrue(late_stop(json.loads(control.read_text()),root/'stages'))
    def test_paired_source_sign_CI_and_progression(self):
        values={g:[] for g in GROUPS}
        for s in range(100):
            for xi in range(2):
                for G in range(1,5):
                    for group,offset in [('D0',0.),('R0',1.),('R08',2.),('L08',3.)]:
                        first=20.+s*.01;delta=0. if G==1 else .1*G
                        values[group].append({'start_id':f'{s}/{xi}/{G}','source_id':f'{s:03d}','trajectory_id':f'{s}/{xi}',
                           'G':G,'split':'val','psnr':first+delta+offset,'psnr_first':first+offset,'delta_prog_PSNR':delta})
        paired,matrix=paired_table(values)
        self.assertEqual(matrix.shape,(10000,100));self.assertEqual(len(paired),5)
        r=next(r for r in paired if r['contrast']=='R08-L08')
        self.assertAlmostEqual(r['mean_paired_delta_PSNR'],-1.);self.assertAlmostEqual(r['CI95_low'],-1.)
        self.assertEqual(r['source_win_rate'],0.)
        prog=progression_table(values)
        self.assertEqual(len(prog),12)
        for r in prog:
            self.assertAlmostEqual(r['delta_PSNR'],.3 if r['G']=='Overall_G2to4' else .1*r['G'])
    def test_best_val_common_interval_and_global_tie(self):
        nodes=list(range(0,5001,250));scores={u:20. for u in nodes};scores[250]=20.0000008;scores[500]=20.0000016
        self.assertEqual(rules.best_update(scores,nodes,1e-6),250)
        with self.assertRaises(ValueError):rules.best_update({**scores,5250:30.},nodes,1e-6)

if __name__=='__main__':unittest.main()

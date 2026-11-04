"""CPU coordination negative controls; not a GPU memory/evaluation acceptance."""
from pathlib import Path
import json,multiprocessing as mp,tempfile,time,unittest
from v4.mutex import EvaluationMutex

def worker(directory,group,scenario):
    m=EvaluationMutex(directory,group,5,['R08','D0'],lambda:None,timeout=10)
    end=1 if group=='R08' else 2
    for u in range(end+1):
        if scenario=='failure' and group=='R08' and u==0:
            m.boundary(u,'eval0');m.fail('INTENTIONAL_TEST_EVALUATOR_EXCEPTION');return
        try:
            assert m.boundary(u,f'eval{u}')
            time.sleep(.03);m.complete(f'eval{u}')
            if scenario=='duplicate' and group=='R08' and u==0:
                try:m.boundary(u,'eval0')
                except RuntimeError as e:
                    assert 'Duplicate committed' in str(e)
                else:raise AssertionError('Duplicate evaluation accepted')
            if scenario=='parked_peer' and group=='R08' and u==0:
                m.park(u)
                deadline=time.monotonic()+10
                while not (Path(directory)/'peer_finished').exists():
                    if time.monotonic()>deadline:raise TimeoutError('Parked peer blocked evaluation')
                    time.sleep(.01)
                m.terminal(u);return
            if u==end:
                m.terminal(u)
                if scenario=='parked_peer':(Path(directory)/'peer_finished').write_text('done')
                return
            assert not m.boundary(u)
            time.sleep(.02)
        except RuntimeError as e:
            if scenario=='failure' and 'INTENTIONAL_TEST_EVALUATOR_EXCEPTION' in str(e):return
            raise

class MutexTests(unittest.TestCase):
    def run_case(self,scenario):
        with tempfile.TemporaryDirectory() as directory:
            ctx=mp.get_context('spawn')
            jobs=[ctx.Process(target=worker,args=(directory,g,scenario)) for g in ('R08','D0')]
            for p in jobs:p.start()
            for p in jobs:
                p.join(20)
                if p.is_alive():p.terminate();p.join();self.fail('Mutex deadlock')
                self.assertEqual(p.exitcode,0)
            events=[json.loads(s) for s in (Path(directory)/'gpu5.events.jsonl').read_text().splitlines()]
            owner=None
            for event in events:
                if event['event']=='grant':
                    self.assertIsNone(owner);owner=event['owner']
                    self.assertTrue(all(w['status'] in ('PAUSED','TERMINAL') for w in event['workers'].values()))
                elif event['event']=='update_permitted':self.assertIsNone(owner)
                elif event['event']=='evaluation_committed_release':
                    self.assertEqual(event['group'],owner);owner=None
            state=json.loads((Path(directory)/'gpu5.json').read_text())
            if scenario=='failure':self.assertIsNotNone(state['failed'])
            else:
                self.assertIsNone(owner);self.assertEqual(len(state['committed']),4 if scenario=='parked_peer' else 5)
                self.assertEqual(len(set(state['committed'])),len(state['committed']))
                self.assertTrue(all(w['status']=='TERMINAL' for w in state['workers'].values()))
    def test_simultaneous_and_one_finished(self):self.run_case('normal')
    def test_evaluator_failure_blocks_peer(self):self.run_case('failure')
    def test_duplicate_commit_rejected(self):self.run_case('duplicate')
    def test_stage_parked_peer_does_not_hold_eval_lease(self):self.run_case('parked_peer')

if __name__=='__main__':unittest.main()

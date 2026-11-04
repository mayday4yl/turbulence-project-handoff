"""CPU-only completion watcher; verifies endpoints before producing final package."""
from pathlib import Path
import fcntl,json,subprocess,time

ROOT=Path(__file__).resolve().parents[1]

def main():
    lock=(ROOT/'.finalize_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    required=[ROOT/'reports/TRAINING_COMPLETE.json',ROOT/'reports/EVALUATION_COMPLETE.json']
    while not all(p.exists() for p in required):time.sleep(60)
    for task in ('T1_DIRECT','T2_FEEDBACK_R'):
        subprocess.run(['bash','code/run_single_env.sh','code/verify_boundary.py','--checkpoint',
            str(ROOT/task/'formal/checkpoint_001000.pt'),'--output',str(ROOT/task/'COMPLETE_1000.json')],cwd=ROOT,check=True)
    subprocess.run(['bash','code/run_single_env.sh','code/report_followup.py','--mode','all'],cwd=ROOT,check=True)
    manifest=json.loads((ROOT/'reports/V031_FOLLOWUP_RESULTS_MANIFEST.json').read_text())
    (ROOT/'reports/COMPLETION.json').write_text(json.dumps({'status':'complete','T1_updates':1000,'T2_updates':1000,
        'I1_status':json.loads((ROOT/'I1_REFINE_REPAIR/selection.json').read_text())['status'],
        'package':manifest,'local_sync_and_visual_QA_pending':True,'time':time.time()},indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        (ROOT/'reports/FINALIZE_FAILURE.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()},indent=2)+'\n')
        raise

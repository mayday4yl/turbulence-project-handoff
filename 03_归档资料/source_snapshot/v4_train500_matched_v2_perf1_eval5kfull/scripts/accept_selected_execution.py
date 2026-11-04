"""Acceptance for selected serialized updates and boundary migration; no training steps."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from v4.runtime import save_json,sha256


def main():
    assert not (ROOT/'reports/RUN_PAUSED.json').exists()
    status=json.loads((ROOT/'reports/PERFORMANCE_STATUS.json').read_text())
    assert status['status']=='BENCHMARK_PASS_NEEDS_SELECTION_AND_EXECUTION_ACCEPTANCE'
    selected=json.loads((ROOT/'protocol/RESOURCE_SCHEDULE.json').read_text())['selected']
    assert selected['compute_scheduling']=='serialized_updates' and selected['dynamic_relocation']
    # CPU coordination checks cover boundary ACK, evaluator exceptions,
    # completed peers, budget guards and source-level paired summaries.
    log=(ROOT/'reports/selected_cpu_acceptance.log').open('x')
    completed=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    if completed.returncode:raise RuntimeError('Selected CPU coordination acceptance failed')
    save_json(ROOT/'reports/RESOURCE_CPU_ACCEPTANCE.json',{'status':'PASS','scope':'CPU coordination only; actual CUDA memory and gradients covered by benchmark and revision gates','log_sha256':sha256(ROOT/'reports/selected_cpu_acceptance.log')})
    # Restoring every actual U0 arm on both authorized cards tests migration
    # state and same frozen next-batch gradient/output, without optimizer.step.
    for pairs in [[('D0',5),('L08',6)],[('R08',5),('R0',6)],
                  [('D0',6),('L08',5)],[('R08',6),('R0',5)]]:
        children=[]
        for g,gpu in pairs:
            log=(ROOT/f'reports/accept_{g}_gpu{gpu}.log').open('x')
            command=[sys.executable,'-u',str(ROOT/'scripts/validate_execution_revision.py'),'--group',g,'--gpu',str(gpu)]
            if (g,gpu) in [('R08',5),('R0',6)]:command.append('--refine')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((g,gpu,child,log))
        failures=[]
        for g,gpu,child,log in children:
            code=child.wait();log.close()
            if code:failures.append((g,gpu,code))
        if failures:raise RuntimeError('Revision gate failed '+str(failures))
    reports={}
    for group in ('D0','R0','R08','L08'):
        root=ROOT/'reports/execution_revision'/group
        rows=[json.loads((root/f'gpu{gpu}.json').read_text()) for gpu in (5,6)]
        assert all(r['status']=='PASS' and r['optimizer_updates']==0 for r in rows)
        a=np.load(root/'gradient_gpu5.npy').astype(np.float64);b=np.load(root/'gradient_gpu6.npy').astype(np.float64)
        relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-300));assert relative<=1e-4
        x=np.load(root/'output_gpu5.npy');y=np.load(root/'output_gpu6.npy')
        np.testing.assert_allclose(x,y,rtol=2e-4,atol=2e-5)
        reports[group]={'cross_GPU_gradient_relative_L2':relative,'cross_GPU_output_max_abs':float(np.max(np.abs(x-y))),
                        'state_restore':[r['formal_U0_state_exact'] for r in rows],
                        'populated_optimizer_RNG_roundtrip':[r['populated_optimizer_RNG_roundtrip'] for r in rows],
                        'refinement':[r['refinement'] for r in rows if 'refinement' in r]}
    save_json(ROOT/'reports/EXECUTION_REVISION_ACCEPTANCE.json',{'status':'PASS','selected':selected,'optimizer_updates':0,'groups':reports,'time':time.time()})
    print('Selected execution GPU and CPU gates PASS',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save_json(ROOT/'reports/SELECTED_EXECUTION_GATE_FAILURE.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise

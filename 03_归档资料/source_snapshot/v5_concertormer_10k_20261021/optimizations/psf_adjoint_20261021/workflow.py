"""Finite four-card A/B benchmarks; zero formal training updates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from training.state import atomic_json

out=HERE/'FOUR_CARD_WORKFLOW.json'
if out.exists():raise ValueError('Preserve previous workflow evidence')
required=['primitive_cpu64.json','primitive_npu_fp32.json','full_bptt_parity.json','simulator_parity.json']
for name in required:
    record=json.loads((HERE/name).read_text())
    if not record['status'].startswith('PASS'):raise RuntimeError('Missing passing prerequisite: '+name)
status={'status':'RUNNING','formal_optimizer_updates':0,'started_unix':time.time(),'jobs':[],
        'prerequisite_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in required}}
atomic_json(status,out)
for method,backend in [('R0','native'),('R0','adjoint'),('D0','native'),('D0','adjoint')]:
    output=HERE/f'benchmark_{method}_{backend}_w4_b2.json'
    command=['bash','run_env.sh','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(HERE/'benchmark.py'),'--backend',backend,'--kind','distributed',
             '--initial','weights/adapted_U0.pt','--dataset','data/v5_places_c50_s2000_v250_t250_g4_r4_20261021',
             '--output',str(output),'--method',method,'--microbatch','2','--precision','bf16',
             '--talking-backend','conv2d_equivalent','--warmup','1','--updates','2']
    if method=='R0':command+=['--checkpointing','--checkpoint-policy','highest_half']
    log=output.with_suffix('.log')
    if output.exists() or log.exists():raise ValueError('Existing benchmark evidence')
    job={'method':method,'backend':backend,'command':command,'started_unix':time.time(),'status':'RUNNING'}
    status['jobs'].append(job);atomic_json(status,out)
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3')
    # GNU timeout owns the entire distributed child process group; no open-ended run.
    with log.open('x') as stream:
        result=subprocess.run(['timeout','--kill-after=20','600']+command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    report=json.loads(output.read_text()) if output.exists() else {}
    job.update(returncode=result.returncode,finished_unix=time.time(),status=report.get('status','NO_REPORT'))
    atomic_json(status,out)
    if result.returncode!=0 or job['status']!='PASS':
        status.update(status='STOPPED_ON_FAILURE',finished_unix=time.time());atomic_json(status,out)
        raise SystemExit(1)
status.update(status='BOUNDED_BENCHMARKS_COMPLETE',finished_unix=time.time());atomic_json(status,out)
print(json.dumps(status),flush=True)

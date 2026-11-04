#!/usr/bin/env python3
"""Final-byte DDP verification then the already bounded disposable smokes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data2/yanglin/v031_overfit100_g4321_n4')


def main():
    inference = json.loads((ROOT / 'artifacts/stage2_inference_batch1_01/smoke.json').read_text())
    if inference.get('status') != 'passed' or inference.get('network_simulator_unchanged') is not True:
        raise RuntimeError('The fixed single-image inference mechanism has not passed')
    name = 'stage2_ddp_gradients_mb1_02'
    command = [sys.executable, 'code/guard_distributed.py', '--run-dir', str(ROOT / 'runs' / name),
               '--timeout-seconds', '3600', '--memory-limit-gib', '61', '--command',
               'bash', 'code/run_distributed_env.sh', '-m', 'torch.distributed.run',
               '--standalone', '--nnodes=1', '--nproc_per_node=4', 'code/check_distributed_gradients.py',
               '--config', 'configs/stage2_candidate_mb1.json', '--output', f'artifacts/{name}', '--microbatch', '1']
    subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, check=True)
    subprocess.run([sys.executable, 'code/run_stage2_smokes.py', '--attempt-tag', '01',
                    '--initial-ddp-result', f'artifacts/{name}/results.json'],
                   cwd=ROOT, stdin=subprocess.DEVNULL, check=True)


if __name__ == '__main__':
    main()

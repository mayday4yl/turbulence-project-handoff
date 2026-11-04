#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673,GPU-8079b40c-338b-f5a5-2be6-db5255033862
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
V4_PY=/home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python
exec 8>protocol/gpu5.lock
exec 9>protocol/gpu6.lock
flock -n 8
flock -n 9
"$V4_PY" - <<'PY'
import os, subprocess
uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True)
conflicts=[row for row in output.splitlines() if any(uuid in row for uuid in uuids)]
if conflicts:raise SystemExit('Selected GPUs are occupied: '+repr(conflicts))
print('GPU guard passed: '+','.join(uuids),flush=True)
PY
if [[ "$1" == "evaluator" ]]; then
  exec "$V4_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/validate_evaluator.py
elif [[ "$1" == "selected" ]]; then
  exec "$V4_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/validate_selected.py
elif [[ "$1" == "train" ]]; then
  exec "$V4_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/train_comparison.py --protocol protocol/FROZEN.json
else
  exec "$V4_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/prepare_and_validate.py --phase "$1"
fi

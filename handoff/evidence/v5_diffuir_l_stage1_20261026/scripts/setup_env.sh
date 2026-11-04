#!/usr/bin/env bash
set -euo pipefail
DIFFUIR_ROOT=/data2/yanglin/v5_diffuir_l_stage1_20261026
mkdir -p "$DIFFUIR_ROOT"/{evidence,reports,scripts,weights,outputs,protocol,cache/ascend,cache/pip}
export PYTHONDONTWRITEBYTECODE=1
python3 -m venv "$DIFFUIR_ROOT/env"
printf '%s\n' /data2/yanglin/v5_resshift_stage1_20261026/env/lib/python3.10/site-packages /data0/yanglin/envs/physical_bridge/lib/python3.10/site-packages > "$DIFFUIR_ROOT/env/lib/python3.10/site-packages/verified_npu_runtime.pth"
export PIP_CACHE_DIR="$DIFFUIR_ROOT/cache/pip"
"$DIFFUIR_ROOT/env/bin/python" -m pip install 'Augmentor==0.2.12' 'accelerate==0.25.0' 'ema-pytorch==0.3.2' 'thop==0.1.1.post2209072238'
"$DIFFUIR_ROOT/env/bin/python" -m pip freeze > "$DIFFUIR_ROOT/evidence/pip_freeze.txt"

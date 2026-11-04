"""Sequential, read-only checkpoint evaluation using the frozen worker."""
from pathlib import Path
import subprocess, sys, json

ROOT = Path(__file__).resolve().parents[2]
jobs = json.loads(Path(sys.argv[1]).read_text())
for job in jobs:
    subprocess.run(['bash', 'code/run_single_env.sh', 'code/evaluate_followup.py', '--job', job], cwd=ROOT, check=True)

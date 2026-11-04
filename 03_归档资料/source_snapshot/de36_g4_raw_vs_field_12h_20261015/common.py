import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'protocol/CONFIG.json').read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()

def seed_for(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63-1)

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'); f.flush()

def experiment_deadline():
    return json.loads((ROOT / 'reports/EXPERIMENT_CLOCK.json').read_text())['deadline_epoch']

def check_deadline(reserve=0):
    if time.time() + reserve >= experiment_deadline(): raise RuntimeError('Experiment wall budget exhausted')

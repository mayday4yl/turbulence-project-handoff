import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GROUP_DEVICES = {'D0': 0, 'R0': 1, 'R08': 2, 'L08': 3}
MILESTONES = (5000, 10000, 20000)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
        d = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        if os.path.exists(name): os.unlink(name)


def append_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush()


def lr_at(update):
    if not 1 <= update <= 20000: raise ValueError('LR outside authorized 20k')
    if update <= 500: return 1e-4 * update / 500
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * (update-500)/19500))


def load_config(path=None):
    path = Path(path or ROOT/'configs/runtime.json')
    value = json.loads(path.read_text())
    for key in ('asset_dir', 'dataset_root', 'results_root'):
        if not Path(value[key]).is_absolute(): value[key] = str(ROOT/value[key])
    return value


def seed_for(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') % (2**63-1)

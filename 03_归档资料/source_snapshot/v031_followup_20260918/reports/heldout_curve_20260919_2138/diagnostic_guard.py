"""Reuse the accepted own-process guard; recognize the current policy filename."""
from pathlib import Path
import sys, json, statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import guard_shared

def policy_latency(pid, since=None):
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if not any(Path(x).name in ('policy.py', 'policy_language.py') for x in argv) or '--out' not in argv:
            return None
        ipc = Path(argv[argv.index('--out') + 1]) / 'ipc'
        files = sorted(ipc.glob('*.response.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:128]
        values = []
        for p in files:
            if since is not None and p.stat().st_mtime <= since:
                continue
            row = json.loads(p.read_text())
            if row.get('resize_audit') is None and isinstance(row.get('seconds'), (int, float)):
                values.append(row['seconds'])
        if len(values) < 12:
            return {'n': len(values), 'median': None}
        values.sort()
        return {'n': len(values), 'median': statistics.median(values), 'p90': values[int(.9 * (len(values)-1))]}
    except (OSError, ValueError, IndexError):
        return None

guard_shared.policy_latency = policy_latency
if __name__ == '__main__':
    sys.exit(guard_shared.main())

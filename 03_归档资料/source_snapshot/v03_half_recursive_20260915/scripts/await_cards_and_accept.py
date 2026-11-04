#!/usr/bin/env python3
"""CPU-only resource waiting; dispatch approved queues after health/idle checks."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from guard import PROJECT, snapshot, check_snapshot, atomic_json

OUT = PROJECT / 'artifacts/runtime/half_resource_dispatch_01'
DEADLINE = dt.datetime(2026,9,16,3,25,tzinfo=dt.timezone.utc).timestamp()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    state={'status':'waiting_for_authorized_cards','pid':os.getpid(),'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'queues':[]}
    atomic_json(OUT/'status.json',state)
    for queue_name in ('acceptance_queue.json','benchmark_queue.json'):
        while time.time() < DEADLINE:
            if (PROJECT/'STOP_REQUESTED').exists():
                state['status']='stopped_by_user';atomic_json(OUT/'status.json',state);return
            _,snap=snapshot()
            idle=all(c['health']=='OK' and not c['processes'] for c in snap['cards'])
            state.update(status='waiting_for_authorized_cards',next_queue=queue_name,last_snapshot=snap)
            atomic_json(OUT/'status.json',state)
            if idle:
                time.sleep(5)
                _,snap=snapshot()
                try:check_snapshot(snap)
                except RuntimeError:continue
                break
            time.sleep(45)
        else:
            state['status']='deadline_before_launch';atomic_json(OUT/'status.json',state);return
        state['status']='queue_running';atomic_json(OUT/'status.json',state)
        with (OUT/(queue_name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,str(PROJECT/'scripts/stage_queue.py'),'--queue',str(PROJECT/'configs'/queue_name)],cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['queues'].append({'config':queue_name,'returncode':result.returncode})
        if result.returncode:
            state['status']='queue_failed_needs_engineering_review';atomic_json(OUT/'status.json',state);return
    state['status']='engineering_queues_completed';atomic_json(OUT/'status.json',state)


if __name__=='__main__':main()

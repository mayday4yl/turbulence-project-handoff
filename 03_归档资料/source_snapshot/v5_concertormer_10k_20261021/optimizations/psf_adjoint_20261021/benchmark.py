"""Use the unchanged bounded benchmark with an explicit candidate factory.

There is no hook in the formal trainer. Every invocation starts from the same U0.
The sidecar binds source hashes and candidate choice to the benchmark output.
"""
import argparse
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from adjoint import install
from training.state import atomic_json


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--backend',choices=['native','adjoint'],required=True)
    p.add_argument('--kind',choices=['distributed','microbatch'],default='distributed')
    args,rest=p.parse_known_args()
    if args.kind=='distributed':
        from scripts import benchmark_distributed as target
    else:
        from scripts import benchmark_runtime_candidate as target
    output=Path(rest[rest.index('--output')+1])
    import os
    rank=int(os.environ.get('RANK','0'))
    if rank==0:
        sidecar=output.with_name(output.stem+'_implementation.json')
        if sidecar.exists():raise ValueError('Preserve previous evidence')
        atomic_json({'backend':args.backend,'formal_optimizer_updates':0,
                     'original_simulator_source_unchanged':True,'source_sha256':{
                        str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__),HERE/'adjoint.py',Path(target.__file__)]}},sidecar)
    if args.backend=='adjoint':
        original=target.make_simulator
        def factory(device):return install(original(device))
        target.make_simulator=factory
    sys.argv=[sys.argv[0]]+rest
    return target.main()


if __name__=='__main__':raise SystemExit(main())

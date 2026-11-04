"""Attributable device trace of exactly equal FP32 VJPs; no training."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import torch
import torch.nn.functional as F
from adjoint import FrozenPSFConv,adjoint_kernel
from runtime.environment import configure,make_simulator,synchronize
from training.state import atomic_json

p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
out=Path(args.output_dir)
if out.exists():raise ValueError('Preserve previous evidence')
out.mkdir(parents=True)
env=configure('npu:0');sim=make_simulator('npu:0')
for p in sim.parameters():p.requires_grad_(False)
generator=torch.Generator().manual_seed(77123)
x=torch.randn(6,1,288,288,generator=generator).to('npu:0').requires_grad_()
w=sim.basis_psf;adj=adjoint_kernel(w)
v=torch.randn(6,100,256,256,generator=generator).to('npu:0')*.001
import torch_npu
reports={}
for backend in ['native','adjoint']:
    def run():
        y=F.conv2d(x,w) if backend=='native' else FrozenPSFConv.apply(x,w,adj)
        return torch.autograd.grad(y,x,v)[0]
    for _ in range(2):run()
    synchronize('npu:0')
    directory=out/backend
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,profile_memory=False,with_stack=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory))) as prof:
        for _ in range(2):run()
        synchronize('npu:0')
    path=next(directory.glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
    totals=collections.Counter();counts=collections.Counter();properties={}
    with path.open() as stream:
        reader=csv.DictReader(stream)
        columns=reader.fieldnames
        for row in reader:
            totals[row['Name']]+=float(row['Duration(us)'])/1e6;counts[row['Name']]+=1
            properties[row['Name']]={k:v for k,v in row.items() if any(s in k.lower() for s in ['type','block','format','shape','dtype'])}
    reports[backend]={'csv_columns':columns,'sum_device_kernel_seconds':sum(totals.values()),
        'top_kernels':[{'name':n,'seconds':t,'calls':counts[n],'properties':properties[n]} for n,t in totals.most_common(10)],
        'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
report={'status':'CAPTURE_COMPLETE','scope':'two instrumented basis forward+VJP each, excludes reflection; not throughput reference',
        'formal_optimizer_updates':0,'environment':env,'backends':reports}
atomic_json(report,out/'SUMMARY.json');print(json.dumps(report),flush=True)

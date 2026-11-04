"""Primitive cause isolation and numerical checks, without optimizer updates."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import torch
import torch.nn.functional as F
from adjoint import FrozenPSFConv, adjoint_kernel
from runtime.environment import configure, make_simulator, synchronize, memory_stats
from training.state import atomic_json


def error(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    d = a-b
    return {"max_absolute": d.abs().max().item(), "relative_l2": (d.norm()/b.norm().clamp_min(1e-30)).item(),
            "reference_l2": b.norm().item(), "finite": bool(torch.isfinite(a).all())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("Preserve previous evidence")
    report = {"status": "RUNNING", "formal_optimizer_updates": 0, "rows": [], "started_unix": time.time()}
    atomic_json(report, out)
    try:
        report["environment"] = configure(args.device)
        sim = make_simulator(args.device)
        for p in sim.parameters():
            p.requires_grad_(False)
        gen = torch.Generator().manual_seed(20261022)
        if args.device == "cpu":
            for name, w in [("mean", sim.mu), ("basis", sim.basis_psf)]:
                w = w.double()
                x = torch.randn(2, 1, 21, 23, generator=gen, dtype=torch.float64).requires_grad_()
                pad = F.pad(x, (16,16,16,16), mode="reflect")
                y = F.conv2d(pad, w)
                v = torch.randn(y.shape, generator=gen, dtype=torch.float64) * .001
                ref = torch.autograd.grad(y, x, v)[0]
                z = FrozenPSFConv.apply(F.pad(x, (16,16,16,16), mode="reflect"), w, adjoint_kernel(w))
                actual = torch.autograd.grad(z, x, v)[0]
                d = torch.randn(x.shape, generator=gen, dtype=torch.float64)
                d /= d.norm()
                eps = 1e-4
                def fn(value):
                    return (F.conv2d(F.pad(value, (16,16,16,16), mode="reflect"), w)*v).sum()
                fd = ((fn(x+eps*d)-fn(x-eps*d))/(2*eps)).item()
                analytic = (actual*d).sum().item()
                row = {"kernel": name, "reflected_boundary": True, "forward_exact": torch.equal(y,z),
                       "gradient": error(actual,ref), "directional_analytic": analytic, "directional_finite_difference": fd,
                       "directional_relative_error": abs(fd-analytic)/max(abs(fd),abs(analytic),1e-20)}
                row["pass"] = row["forward_exact"] and row["gradient"]["relative_l2"] <= 1e-10 and row["directional_relative_error"] <= 1e-7
                report["rows"].append(row); atomic_json(report,out)
        else:
            for name,w in [("mean",sim.mu),("basis",sim.basis_psf)]:
                x = torch.randn(6,1,256,256,generator=gen).to(args.device).requires_grad_()
                v = (torch.randn(6,w.shape[0],256,256,generator=gen)*.001).to(args.device)
                adj = adjoint_kernel(w)
                values={}; times={}
                for backend in ("native", "adjoint"):
                    trials=[]
                    for repeat in range(4):
                        synchronize(args.device); start=time.perf_counter()
                        padded=F.pad(x,(16,16,16,16),mode="reflect")
                        y=F.conv2d(padded,w) if backend=="native" else FrozenPSFConv.apply(padded,w,adj)
                        synchronize(args.device); fw=time.perf_counter()-start
                        start=time.perf_counter()
                        grad=torch.autograd.grad(y,x,v)[0]
                        synchronize(args.device); bw=time.perf_counter()-start
                        trials.append({"repeat":repeat,"forward_seconds":fw,"backward_seconds":bw})
                        if repeat==3:
                            values[backend]=(y.detach().cpu(),grad.detach().cpu())
                        del y,grad,padded
                    times[backend]=trials
                actual,ref=values["adjoint"][1],values["native"][1]
                row={"kernel":name,"input_shape":list(x.shape),"weight_shape":list(w.shape),
                     "forward_exact":torch.equal(values["native"][0],values["adjoint"][0]),
                     "gradient":error(actual,ref),"allclose":torch.allclose(actual,ref,atol=2e-6,rtol=2e-5),
                     "timings":times,"native_backward_median_seconds":statistics.median(v["backward_seconds"] for v in times["native"][1:]),
                     "adjoint_backward_median_seconds":statistics.median(v["backward_seconds"] for v in times["adjoint"][1:]),
                     **memory_stats(args.device)}
                row["pass"]=row["forward_exact"] and row["gradient"]["finite"] and row["gradient"]["relative_l2"]<=2e-5 and row["allclose"]
                report["rows"].append(row);atomic_json(report,out)
                print(json.dumps(row),flush=True)
                del x,v,adj,values
        report["status"]="PASS_PRIMITIVE_ONLY" if all(r["pass"] for r in report["rows"]) else "FAIL"
    except BaseException as e:
        report.update(status="FAIL",error=repr(e),traceback=traceback.format_exc())
    report["finished_unix"]=time.time()
    atomic_json(report,out)
    print(json.dumps(report),flush=True)
    return 0 if report["status"]=="PASS_PRIMITIVE_ONLY" else 1


if __name__=="__main__":
    raise SystemExit(main())

"""Strict official loading and bounded CPU/NPU native-function comparison."""
import json
from pathlib import Path
import numpy as np
import torch
from core import ROOT,configure_npu,initialize,process,write


def main():
    torch.set_num_threads(4);cpu,report=initialize('cpu');cpu.eval()
    device=configure_npu(0);npu,_=initialize(device);npu.eval()
    generator=torch.Generator().manual_seed(843201)
    state=torch.rand((2,3,64,64),generator=generator)
    condition=torch.rand((2,3,64,64),generator=generator)
    clean=torch.rand((2,3,64,64),generator=generator)
    time=torch.tensor([3,77],dtype=torch.long)
    with torch.no_grad():
        a=cpu(state,condition,time);b=npu(state.to(device),condition.to(device),time.to(device)).cpu()
    output_error=float((a-b).abs().max());assert torch.allclose(a,b,atol=2e-4,rtol=2e-4),output_error
    c=process(torch.device('cpu'));n=process(device);c.set_mu(condition);n.set_mu(condition.to(device))
    t=time.reshape(-1,1,1,1)
    with torch.no_grad():
        score_cpu=c.get_score_from_noise(a,t)
        expected=c.reverse_sde_step_mean(state,score_cpu,t)
        optimum=c.reverse_optimum_step(state,clean,t)
        from vendor.unidb_gou.loss import MatchingLoss
        native=MatchingLoss('l1',False)(expected,optimum)
        manual=torch.mean(torch.abs(expected-optimum))
        manual_error=float(abs(native-manual))
        assert torch.allclose(native,manual,atol=1e-7,rtol=1e-6)
        score_npu=n.get_score_from_noise(b.to(device),t.to(device))
        expected_npu=n.reverse_sde_step_mean(state.to(device),score_npu,t.to(device))
        optimum_npu=n.reverse_optimum_step(state.to(device),clean.to(device),t.to(device))
        loss_npu=MatchingLoss('l1',False)(expected_npu,optimum_npu).cpu()
    loss_error=float(abs(native-loss_npu));assert torch.allclose(native,loss_npu,atol=2e-5,rtol=2e-4),loss_error
    # Native random-state sampler excludes t=0 and t=100, as the official code does.
    times,states=n.generate_random_states(clean.to(device),condition.to(device))
    assert times.shape==(2,1,1,1) and bool((times>=1).all()) and bool((times<100).all())
    assert bool(torch.isfinite(states).all())
    write(ROOT/'evidence/PREFLIGHT_NATIVE.json',{'status':'PASS','strict_load':report,
        'CPU_NPU_RGB64_forward_maxabs':output_error,'native_loss_vs_manual_L1_maxabs':manual_error,
        'CPU_NPU_native_objective_abs_error':loss_error,'random_tau_range':[1,99],
        'new_model_modules':False,'task_prompt':False,'native_loss_unchanged':True,
        'full_size_update_gate':'SEPARATE_DISTRIBUTED_BENCHMARK','formal_updates':0})
    print('PREFLIGHT_NATIVE_PASS',report,output_error,loss_error,flush=True)


if __name__=='__main__':main()

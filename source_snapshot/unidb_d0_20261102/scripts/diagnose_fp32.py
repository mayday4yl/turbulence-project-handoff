"""Read-only bounded layer comparison after initial CPU/NPU tolerance failure."""
import argparse,json
import torch
from core import ROOT,configure_npu,initialize,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--strict-option',action='store_true');parser.add_argument('--internal-off',action='store_true');a=parser.parse_args()
    torch.set_num_threads(4);cpu,identity=initialize('cpu');cpu.eval();device=configure_npu(0)
    if a.strict_option:torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
    if a.internal_off:torch.npu.config.allow_internal_format=False
    npu,_=initialize(device);npu.eval();generator=torch.Generator().manual_seed(843201)
    state=torch.rand((2,3,64,64),generator=generator);condition=torch.rand((2,3,64,64),generator=generator);t=torch.tensor([3,77])
    observed={};records=[];hooks=[]
    for name,module in cpu.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
            hooks.append(module.register_forward_hook(lambda m,x,y,n=name:observed.__setitem__(n,y.detach().clone())))
    with torch.no_grad():expected=cpu(state,condition,t)
    for hook in hooks:hook.remove()
    hooks=[]
    def capture(name,y):
        x=observed[name];y=y.detach().cpu();d=(x-y).double()
        rms=float(x.double().square().mean().sqrt());error=float(d.square().mean().sqrt())
        records.append({'name':name,'shape':list(x.shape),'maxabs':float(d.abs().max()),'RMSE':error,'reference_RMS':rms,'NRMSE':error/max(rms,1e-12)})
    for name,module in npu.named_modules():
        if name in observed:hooks.append(module.register_forward_hook(lambda m,x,y,n=name:capture(n,y)))
    with torch.no_grad():actual=npu(state.to(device),condition.to(device),t.to(device)).cpu()
    for hook in hooks:hook.remove()
    delta=(expected-actual).double()
    result={'strict_precision_option':a.strict_option,'internal_format_off':a.internal_off,'strict_load':identity,
        'maxabs':float(delta.abs().max()),'RMSE':float(delta.square().mean().sqrt()),
        'reference_RMS':float(expected.double().square().mean().sqrt()),'reference_absmax':float(expected.abs().max()),
        'reference_abs_quantiles':torch.quantile(expected.abs().flatten(),torch.tensor([.5,.95,.99])).tolist(),
        'layers':records,'training_updates':0}
    name=f'diagnose_fp32_strict{int(a.strict_option)}_off{int(a.internal_off)}.json'
    write(ROOT/'evidence'/name,result);print(json.dumps({k:v for k,v in result.items() if k!='layers'}),flush=True)


if __name__=='__main__':main()

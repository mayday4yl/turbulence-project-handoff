import json,math
import torch
from core import ROOT,configure_npu,write

device=configure_npu(0)
x=torch.linspace(-8,8,100001,dtype=torch.float32)
cpu=torch.nn.functional.gelu(x,approximate='none')
tanh=torch.nn.functional.gelu(x,approximate='tanh')
native=torch.nn.functional.gelu(x.to(device),approximate='none').cpu()
exact=(.5*x.to(device)*(1+torch.erf(x.to(device)/math.sqrt(2)))).cpu()
result={'native_gelu_vs_CPU_exact_maxabs':float((native-cpu).abs().max()),
 'native_gelu_vs_CPU_tanh_maxabs':float((native-tanh).abs().max()),
 'composed_erf_gelu_vs_CPU_exact_maxabs':float((exact-cpu).abs().max()),
 'native_silu_vs_CPU_maxabs':float((torch.nn.functional.silu(x.to(device)).cpu()-torch.nn.functional.silu(x)).abs().max())}
for mode in ['native','composed']:
 c=x.clone().requires_grad_();reference=torch.nn.functional.gelu(c,approximate='none');reference.sum().backward()
 n=x.to(device).requires_grad_()
 y=torch.nn.functional.gelu(n,approximate='none') if mode=='native' else .5*n*(1+torch.erf(n/math.sqrt(2)))
 y.sum().backward();result[mode+'_gelu_gradient_vs_CPU_maxabs']=float((n.grad.cpu()-c.grad).abs().max())
write(ROOT/'evidence/ACTIVATION_DIAGNOSIS.json',result);print(json.dumps(result),flush=True)

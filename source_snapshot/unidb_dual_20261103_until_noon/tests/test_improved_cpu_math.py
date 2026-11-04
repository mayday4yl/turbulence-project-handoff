"""Independent NumPy formulas + tiny CPU torch tensors; no real model/optimizer.

Run in an existing CPU-capable environment with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
All outputs stay in this new experiment root. The original core/vendor are read only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'unidb_d0_20261102'))
sys.path.insert(0,str(ROOT))
from core import process
from improved_objective import ImprovedObjective,lr_for_update,SCHEDULE_SPEC


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def assert_close(a,b,rtol=2e-4,atol=2e-4):
 a,b=np.asarray(a),np.asarray(b);assert np.allclose(a,b,rtol=rtol,atol=atol),(float(np.max(np.abs(a-b))),rtol,atol)
 return float(np.max(np.abs(a-b)))
def must_raise(fn):
 try:fn()
 except ValueError:return True
 raise AssertionError('Expected ValueError was not raised')


def independent_coefficients():
 # Float64 implementation derived directly from the mathematical cosine bridge.
 grid=np.linspace(0.,102.,103,dtype=np.float64)
 alpha=np.cos((grid/102.+.008)/1.008*math.pi/2.)**2;alpha/=alpha[0]
 theta=1.-alpha[1:-1];H=np.cumsum(theta)-theta[0];dt=-math.log(.005)/H[-1]
 lam=30./255.;r=1e-7;sigma2=2.*lam*lam*theta
 B=lam*lam*(-np.expm1(-2.*H*dt));BT=lam*lam*(-np.expm1(-2.*(H[-1]-H)*dt))
 V=B*BT/B[-1];m=np.exp(-H*dt)*(r+BT)/(r+B[-1]);kappa=theta+sigma2*np.exp(2.*(H-H[-1])*dt)/(r+BT)
 c=np.zeros(101);c[1:100]=sigma2[1:100]*dt/np.sqrt(V[1:100]);return {'theta':theta,'H':H,'dt':dt,'sigma2':sigma2,'V':V,'m':m,'kappa':kappa,'c':c}


class TinyPredictionNet(torch.nn.Module):
 def __init__(self,values):
  super().__init__();self.values=torch.nn.Parameter(values.clone());self.calls=0;self.last_tau=None
 def forward(self,state,condition,times):
  assert state.device.type==condition.device.type=='cpu';self.calls+=1;self.last_tau=times.detach().clone();return self.values


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=ROOT/'evidence/improved_cpu_math');args=ap.parse_args();out=args.output_dir
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261103)
 assert not torch.cuda.is_initialized();assert 'torch_npu' not in sys.modules,'CPU test must not autoload NPU backend'
 start=time.time();sde=process('cpu');ind=independent_coefficients();tt=torch.arange(1,100,dtype=torch.long).reshape(99,1,1,1)
 rng=np.random.default_rng(731103);clean=torch.tensor(rng.uniform(.1,.9,(99,3,4,4)),dtype=torch.float32)
 degraded=torch.clamp(clean+torch.tensor(rng.normal(0,.15,clean.shape),dtype=torch.float32),0.,1.)
 eps=torch.tensor(rng.normal(size=clean.shape),dtype=torch.float32);sde.set_mu(degraded)
 state=sde.f_mean(clean,tt)+sde.f_sigma(tt)*eps
 with torch.no_grad():
  mu0=sde.reverse_sde_step_mean(state,torch.zeros_like(state),tt);mu_star=sde.reverse_optimum_step(state,clean,tt)
  c=sde.sigmas[tt].square()*sde.dt/sde.f_sigma(tt);target=(mu0-mu_star)/c
 assert bool(torch.isfinite(c).all()) and bool((c>0).all()) and bool(torch.isfinite(target).all())
 coeff_error=assert_close(c.numpy().reshape(-1),ind['c'][1:100],rtol=2e-4,atol=2e-7)
 shape=(-1,1,1,1);t=np.arange(1,100);x=state.numpy().astype(np.float64);y=degraded.numpy().astype(np.float64);x0=clean.numpy().astype(np.float64)
 m=ind['m'];V=ind['V'];F=m[t]/m[t-1];K=F*V[t-1]/V[t]
 Mprev=y+m[t-1].reshape(shape)*(x0-y);Mt=y+m[t].reshape(shape)*(x0-y)
 star64=Mprev+K.reshape(shape)*(x-Mt);zero64=x+ind['kappa'][t].reshape(shape)*ind['dt']*(x-y)
 target64=(zero64-star64)/ind['c'][t].reshape(shape)
 mu0_error=assert_close(mu0.numpy(),zero64);mustar_error=assert_close(mu_star.numpy(),star64)
 target_error=assert_close(target.numpy(),target64,rtol=2e-4,atol=2e-4)
 # Test the production forward API using one tiny trainable output tensor.
 net=TinyPredictionNet(torch.tensor(rng.normal(size=state.shape),dtype=torch.float32));objective=ImprovedObjective(net,'cpu')
 def fixed_states(x0,mu):objective.sde.set_mu(mu);return tt.clone(),state.clone()
 objective.sde.generate_random_states=fixed_states
 parameter_before=net.values.detach().clone();loss,tau=objective(degraded,clean)
 assert net.calls==1 and tau.shape==(99,1,1,1) and not tau.requires_grad and not tau.is_floating_point()
 assert not any(v.requires_grad for v in objective.last_diagnostics.values())
 expected=(net.values-target.detach()).square().mean()
 assert_close(loss.detach().numpy(),expected.detach().numpy(),rtol=1e-6,atol=1e-7)
 mean_pred=sde.reverse_sde_step_mean(state,sde.get_score_from_noise(net.values,tt),tt)
 mu_loss=((mean_pred-mu_star.detach())/c).square().mean()
 mu_equiv=assert_close(loss.detach().numpy(),mu_loss.detach().numpy(),rtol=2e-4,atol=1e-7)
 pred_mu_equiv=assert_close(mean_pred.detach().numpy(),(mu0-c*net.values).detach().numpy(),rtol=2e-5,atol=2e-6)
 g_noise=torch.autograd.grad(loss,net.values,retain_graph=True)[0];g_mu=torch.autograd.grad(mu_loss,net.values,retain_graph=True)[0]
 gradient_equiv=assert_close(g_noise.numpy(),g_mu.numpy(),rtol=2e-4,atol=1e-7)
 analytic=2.*(net.values.detach()-target)/net.values.numel();assert_close(g_noise.numpy(),analytic.numpy(),rtol=1e-6,atol=1e-9)
 loss.backward();assert torch.equal(net.values.detach(),parameter_before),'Backward must not update parameters'
 assert bool(torch.isfinite(net.values.grad).all())
 # Loss/gradient sign and central finite difference, no optimizer/step.
 base=target.double();offset=torch.tensor(.25,dtype=torch.float64,requires_grad=True)
 sign_loss=(base+offset-base).square().mean();sign_grad=float(torch.autograd.grad(sign_loss,offset)[0]);assert_close(sign_grad,.5,rtol=1e-10,atol=1e-12)
 h=1e-5;fd=(((base+.25+h-base).square().mean()-(base+.25-h-base).square().mean())/(2*h)).item();assert_close(fd,sign_grad,rtol=1e-8,atol=1e-10)
 negative=torch.tensor(-.25,dtype=torch.float64,requires_grad=True);negative_loss=(base+negative-base).square().mean();assert float(torch.autograd.grad(negative_loss,negative)[0])<0
 true_epsilon_difference=float((target-eps).abs().max());assert true_epsilon_difference>1e-3,'epsilon_opt must not silently equal sampled epsilon'
 # Production rejection of forbidden boundary times before any network call.
 forbidden=[]
 for bad in [0,100]:
  objective.sde.generate_random_states=lambda x0,mu,bad=bad:(torch.full((99,1,1,1),bad,dtype=torch.long),state)
  calls=net.calls;must_raise(lambda:objective(degraded,clean));assert net.calls==calls;forbidden.append(bad)
 assert float(sde.f_sigma(0))==0. and float(sde.f_sigma(100))==0.
 assert bool(torch.isfinite(sde.r_sigma_1(tt)).all()) and float(sde.r_sigma_1(1))==0.
 assert_close(mu_star[0].numpy(),clean[0].numpy(),rtol=2e-5,atol=2e-6)
 # Native random-time support and batch1 flatten handling, independent CPU stream.
 all_tau,native_state=sde.generate_random_states(torch.full((4096,3,1,1),.5),torch.full((4096,3,1,1),.4))
 assert all_tau.shape==(4096,1,1,1) and set(all_tau.flatten().tolist())==set(range(1,100)) and bool(torch.isfinite(native_state).all())
 tiny=TinyPredictionNet(torch.zeros((1,3,2,2)));one=ImprovedObjective(tiny,'cpu');one_loss,one_tau=one(torch.ones(1,3,2,2)*.4,torch.ones(1,3,2,2)*.5)
 assert tiny.last_tau.shape==(1,) and one_tau.shape==(1,1,1,1) and one_loss.ndim==0 and tiny.calls==1
 # Deterministic schedule boundary and checkpoint-resume checks.
 begin,deadline,end=1000.,2000.,1200.;sched=[]
 for update,clock,anchor in [(1,1000.,None),(199,1198.,None),(200,1199.,None),(201,1200.,end),(750,1600.,end),(750,2000.,end),(900,2200.,end)]:
  sched.append({'next_update':update,'now_unix':clock,'warmup_end_unix':anchor,'lr':lr_for_update(update,begin,deadline,anchor,now_unix=clock)})
 assert_close(sched[0]['lr'],2e-7,rtol=1e-12,atol=0.);assert_close(sched[2]['lr'],2e-6,rtol=1e-12,atol=0.)
 assert_close(sched[3]['lr'],2e-6,rtol=1e-12,atol=0.);assert_close(sched[4]['lr'],1.1e-6,rtol=1e-12,atol=0.)
 assert sched[5]['lr']==sched[6]['lr']==2e-7
 for bad in [0,-1,1.5,True]:must_raise(lambda bad=bad:lr_for_update(bad,begin,deadline,now_unix=1100.))
 must_raise(lambda:lr_for_update(201,begin,deadline,None,now_unix=1300.));must_raise(lambda:lr_for_update(100,begin,deadline,end,now_unix=1300.))
 checkpoint=json.loads(json.dumps({'completed_updates':749,'start_unix':begin,'deadline':deadline,'warmup_end_unix':end,'spec':SCHEDULE_SPEC}))
 restored=lr_for_update(checkpoint['completed_updates']+1,checkpoint['start_unix'],checkpoint['deadline'],checkpoint['warmup_end_unix'],now_unix=1600.)
 assert restored==sched[4]['lr'];assert restored==lr_for_update(50000,begin,deadline,end,now_unix=1600.);assert restored<2e-6
 records=[]
 for t in range(101):
  active=1<=t<100
  records.append({'t':t,'training_allowed':active,'theta_native':float(sde.thetas[t]),'sigma_native':float(sde.sigmas[t]),'dt_native':float(sde.dt),'f_sigma_native':float(sde.f_sigma(t)),
   'm_native':float(sde.m(t)),'c_t_native':float(sde.sigmas[t]**2*sde.dt/sde.f_sigma(t)) if active else None,'c_t_float64_independent':float(ind['c'][t]) if active else None,
   'reverse_posterior_std_native':float(sde.r_sigma_1(t)) if active else None})
 out.mkdir(parents=True,exist_ok=True)
 with (out/'COEFFICIENTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 put(out/'SCHEDULER_BOUNDARIES.json',{'status':'PASS','spec':SCHEDULE_SPEC,'boundaries':sched,'restored_checkpoint':checkpoint,'resume_lr':restored})
 result={'status':'PASS','device':'cpu','real_restoration_model_calls':0,'tiny_mock_network_calls':2,'optimizer_objects':0,'optimizer_updates':0,
  'all_active_times':99,'native_random_support':list(range(1,100)),'forbidden_times_rejected_before_forward':forbidden,'batch1_api':'PASS',
  'c_t_independent_maxabs':coeff_error,'mu0_independent_maxabs':mu0_error,'mu_star_independent_maxabs':mustar_error,'epsilon_opt_independent_maxabs':target_error,
  'noise_MSE_vs_mu_residual_normalized_MSE_abs':mu_equiv,'native_mu_vs_mu0_minus_c_noise_maxabs':pred_mu_equiv,'gradient_equivalence_maxabs':gradient_equiv,
  'gradient_sign':sign_grad,'central_finite_difference':fd,'epsilon_opt_vs_sampled_epsilon_maxabs':true_epsilon_difference,'parameters_unchanged_after_backward':True,
  'no_NPU_backend_imported':True,'t100_scope':'f_sigma(100)=0 by construction; no posterior division or training target at t100 evaluated',
  'source_sha256':{str(p):sha(p) for p in [ROOT/'improved_objective.py',Path(__file__),ROOT.parent/'unidb_d0_20261102/core.py',ROOT.parent/'unidb_d0_20261102/vendor/unidb_gou/sde_utils.py']},
  'torch_version':torch.__version__,'elapsed_seconds':time.time()-start}
 put(out/'CPU_MATH_ACCEPTANCE.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

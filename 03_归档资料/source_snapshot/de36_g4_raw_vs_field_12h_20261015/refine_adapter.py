"""Exact accepted V4 solver; Field only adapts normalized state and renderer."""
import time
import torch
from torch import nn
from physical_bridge.simulator import Realization
from v4.refinement import refine_xi
from core import render_field,Prediction,render_prediction

class NormalizedFieldRenderer(nn.Module):
 def __init__(self,sim,model):
  super().__init__();self.sim=sim
  self.register_buffer('alpha_scale',model.alpha_scale.detach().clone())
  self.register_buffer('u_scale',model.u_scale.detach().clone())
  self.requires_grad_(False);self.eval()
 def render(self,clean,realization,g):
  return render_field(self.sim,self.sim.prepare_image(clean),realization.epsilon_zernike*self.alpha_scale,realization.epsilon_tilt*self.u_scale,g)

def refine_stage(model,sim,p,y):
 if model.arm=='Raw':
  adapter=sim;initial=Realization(p.alpha_or_z,p.u_or_t)
 else:
  adapter=NormalizedFieldRenderer(sim,model)
  initial=Realization(p.alpha_or_z/model.alpha_scale,p.u_or_t/model.u_scale)
 torch.npu.synchronize();start=time.monotonic()
 result=refine_xi(simulator=adapter,clean=p.clean,xi_hat=initial,observation=y,original_g=4.,network=model)
 torch.npu.synchronize();seconds=time.monotonic()-start
 z,u=result.realization.epsilon_zernike,result.realization.epsilon_tilt
 dz=z-initial.epsilon_zernike;du=u-initial.epsilon_tilt;den=z.numel()+u.numel()
 record={'seconds':seconds,'best_iteration':result.best_iterations[0],'J_before':result.initial_objective[0],'J_after':result.best_objective[0],
         'candidates':result.candidates,'trace':result.trace,'invariance':result.invariance,'optimizer':result.optimizer,
         'normalized_change_rms_alpha_or_z':float(dz.square().mean().sqrt()),'normalized_change_rms_u_or_t':float(du.square().mean().sqrt()),
         'stay_alpha_or_z_contribution':float(dz.square().sum()/den),'stay_u_or_t_contribution':float(du.square().sum()/den),
         'coordinate_system':'native_raw' if model.arm=='Raw' else 'frozen_RMS_normalized_field',
         'physical_regularization_equivalence_claimed':False}
 assert record['J_after']<=record['J_before']+1e-12
 if model.arm=='Field':z=z*model.alpha_scale;u=u*model.u_scale
 return Prediction(p.clean,z,u),record

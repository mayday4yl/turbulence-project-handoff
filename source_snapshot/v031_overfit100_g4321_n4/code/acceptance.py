"""Bounded real-network Stage 1 checks, with ZERO optimizer updates.

Negative controls exist only here as test hooks. The production rollout has no
detach option. This script never constructs a restoration optimizer.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from v031.runtime import build, save_json, seed_all, sha256, state_hash, verify_source, configure_npu
from v031.core import RolloutModule, full_image_mse
from v031.data import StartDataset

P=argparse.ArgumentParser()
P.add_argument('--phase',choices=['correctness','parity','derivatives','performance'],required=True)
P.add_argument('--output',required=True)
P.add_argument('--parity-result',help='Performance only: explicit parity results.json path, absolute or relative to V031_ROOT')
P.add_argument('--fp32-only',action='store_true',help='Performance only: explicitly measure FP32 alone; required when the parity gate did not pass')
args=P.parse_args()
if args.phase=='performance' and args.parity_result is None:
    P.error('--phase performance requires --parity-result')
if args.phase!='performance' and (args.parity_result is not None or args.fp32_only):
    P.error('--parity-result and --fp32-only apply only to --phase performance')
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/stage1_acceptance.json').read_text())
OUT=ROOT/args.output
if OUT.exists():
    raise FileExistsError('Retain old attempts; select a new output path')
OUT.mkdir(parents=True)
(OUT/'code_snapshot').mkdir()
for source in [Path(__file__),ROOT/'configs/stage1_acceptance.json',*(ROOT/'code/v031').glob('*.py')]:
    shutil.copy2(source,OUT/'code_snapshot'/source.name)
result={'phase':args.phase,'status':'running','formal_training_started':False,
        'restoration_optimizer_updates':0,'optimizer_created':False,
        'execution_environment':{name:os.environ.get(name) for name in ['ASCEND_LAUNCH_BLOCKING','TASK_QUEUE_ENABLE','ASCEND_RT_VISIBLE_DEVICES']},
        'config_sha256':sha256(ROOT/'configs/stage1_acceptance.json'),
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'code').rglob('*.py')}}
def progress(event,**kw):
    result['last_event']={'event':event,'time':time.time(),**kw}
    save_json(OUT/'results.json',result)
    print(json.dumps(result['last_event']),flush=True)
def sync():
    torch.npu.synchronize()
def cleanup(model):
    model.zero_grad(set_to_none=True)
    gc.collect(); sync(); torch.npu.empty_cache()
def grad_record(t,g):
    return {'shape':list(t.shape),'dtype':str(t.dtype),'requires_grad':t.requires_grad,
            'connected':g is not None,'finite':None if g is None else bool(torch.isfinite(g).all().cpu()),
            'norm':None if g is None else float(g.float().norm().cpu()),
            'max_abs':None if g is None else float(g.float().abs().max().cpu())}
def finite(t):
    return bool(torch.isfinite(t).all().cpu())
def summarize_trace(out,observation):
    entries=[]
    for k,t in enumerate(out['trace']):
        entries.append({'step':k+1,'active_path_indices':list(t['active_path_indices']),
          'input_source':'saved_Y' if k==0 else 'previous_predicted_transition',
          'input_is_previous_transition':True if k==0 else t['input'] is out['trace'][k-1]['next_state'],
          'current_g':t['current_g'].cpu().tolist(),'reconstruction_G':t['original_g'].cpu().tolist(),
          'reconstruction_Y_equals_saved_Y':torch.equal(t['original_observation'],observation[list(t['active_path_indices'])]),
          'next_g':None if t['next_g'] is None else t['next_g'].cpu().tolist(),
          'LC':t['clean_losses'].detach().cpu().tolist(),'LR':t['reconstruction_losses'].detach().cpu().tolist(),
          'shapes':{n:list(t[n].shape) for n in ['input','clean','epsilon_zernike','epsilon_tilt','reconstructed']},
          'dtypes':{n:str(t[n].dtype) for n in ['input','clean','epsilon_zernike','epsilon_tilt','reconstructed','clean_losses','reconstruction_losses']},
          'all_finite':all(finite(v) for n,v in t.items() if torch.is_tensor(v)),
        })
    return {'steps':entries,'path_clean':out['path_clean'].detach().cpu().tolist(),
            'path_reconstruction':out['path_reconstruction'].detach().cpu().tolist(),
            'path_losses':out['path_losses'].detach().cpu().tolist(),'loss_mean':float(out['loss_mean'].detach().cpu()),
            **{n:out[n] for n in ['prediction_count','logical_render_count','batched_render_call_count','training_refinement_count']}}
def probe(out,model,retain_graph=False):
    trace=out['trace']; tensors=[t[n] for t in trace[:-1] for n in ['clean','epsilon_zernike','epsilon_tilt']]
    loss=trace[-1]['clean_losses'].mean()
    # Retained native-output gradients from the terminal loss alone. A full
    # backward uses the same NPU kernel path as ordinary training backward.
    model.zero_grad(set_to_none=True)
    for t in tensors:t.grad=None
    loss.backward(retain_graph=retain_graph)
    gradients=[t.grad for t in tensors]
    records=[]
    for k in range(len(trace)-1):
        records.append({'step':k+1,**{name:grad_record(trace[k][name],gradients[3*k+j]) for j,name in enumerate(['clean','epsilon_zernike','epsilon_tilt'])}})
    model.zero_grad(set_to_none=True)
    for t in tensors:t.grad=None
    return {'terminal_clean_loss':float(loss.detach().cpu()),'steps':records,
            'measurement':'terminal-only backward with retained native-output gradients'}
def nonzero_probe(p):
    return all(r['connected'] and r['finite'] and r['max_abs']>0 for step in p['steps'] for n,r in step.items() if n!='step')
def snapshot_forward(out):
    return [{n:t[n].detach().cpu().clone() for n in ['input','clean']} for t in out['trace']]
def compare_forward(out,ref):
    diffs=[]; passed=True
    for t,r in zip(out['trace'],ref):
        for n in r:
            value=t[n].detach().cpu()
            diffs.append(float((value-r[n]).abs().max()))
            passed &= torch.allclose(value,r[n],atol=cfg['acceptance']['negative_forward_atol'],rtol=cfg['acceptance']['negative_forward_rtol'])
    return {'passed':bool(passed),'max_abs':max(diffs)}
def cpu_grads(model):
    return {n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.named_parameters()}
def grad_stats(grads):
    return {'norm':math.sqrt(sum(float(g.double().square().sum()) for g in grads.values() if g is not None)),
            'finite':all(bool(torch.isfinite(g).all()) for g in grads.values() if g is not None),
            'none_count':sum(g is None for g in grads.values())}
def gradient_comparison(a,b,names=None):
    dot=na=nb=diff2=0.; connected=True
    for n in a if names is None else names:
        x,y=a[n],b[n]
        if (x is None)!=(y is None):connected=False
        if x is None or y is None:continue
        x=x.double();y=y.double()
        dot+=float((x*y).sum());na+=float(x.square().sum());nb+=float(y.square().sum());diff2+=float((x-y).square().sum())
    return {'cosine':dot/math.sqrt(na*nb) if na*nb>0 else None,'reference_norm':math.sqrt(na),
            'candidate_norm':math.sqrt(nb),'same_connected_parameters':connected,
            'relative_l2_difference':math.sqrt(diff2/max(na,1e-30))}
def tensor_comparison(a,b):
    d=b.double()-a.double()
    return {'shape':list(a.shape),'max_abs':float(d.abs().max()),'rms_difference':float(d.square().mean().sqrt()),
            'reference_rms':float(a.double().square().mean().sqrt()),'candidate_rms':float(b.double().square().mean().sqrt())}

try:
    import torch_npu
    device=torch.device('npu:0');result['npu_execution']=configure_npu(cfg)
    torch.set_num_threads(4)
    model,sim=build(cfg,device);model.train()
    result['npu_execution_effective']={
      'is_jit_compile_false':bool(torch.npu.is_jit_compile_false()),
      'raw_jitCompile':str(torch_npu._C._npu_getOption('jitCompile')),
      'raw_allow_internal_format':str(torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')),
      'raw_precision_mode':str(torch_npu._C._npu_getOption('ACL_PRECISION_MODE')),
      'raw_force_jit_ops':str(torch_npu._C._npu_getOption('NPU_FUZZY_COMPILE_BLACKLIST')),
      'conv_allow_hf32':torch.npu.conv.allow_hf32,
      'matmul_allow_hf32':torch.npu.matmul.allow_hf32}
    assert not result['npu_execution_effective']['conv_allow_hf32']
    assert not result['npu_execution_effective']['matmul_allow_hf32']
    model_initial=state_hash(model);sim_initial=state_hash(sim)
    result.update({'initial_model_sha256':model_initial,'initial_simulator_sha256':sim_initial,
      'base_hashes_before':verify_source(cfg),'seed':cfg['seed'],'checkpoint_loaded':False,
      'initialization':'Unmodified original constructor, including zero native-xi heads; no checkpoint or training',
      'torch':torch.__version__,'torch_npu':torch_npu.__version__,
      'physical_device':int(os.environ['ASCEND_RT_VISIBLE_DEVICES']),
      'device':str(torch.npu.get_device_properties(device)),
      'batchnorm_modules':[n for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)],
      'parameter_count':sum(p.numel() for p in model.parameters()),
      'all_master_parameters_fp32':all(p.dtype==torch.float32 for p in model.parameters())})
    assert not result['batchnorm_modules']
    dataset=StartDataset(cfg['dataset_root'],cfg['diagnostic_subset'])
    idx=[i for i,r in enumerate(dataset.records) if r['initial_G']==4][:2]
    chosen=[dataset[i] for i in idx]
    first_trajectory=dataset.records[idx[0]]['trajectory_id']
    trace_indices={int(r['initial_G']):i for i,r in enumerate(dataset.records) if r['trajectory_id']==first_trajectory}
    assert set(trace_indices)=={1,2,3,4}
    def batch(indices):
        samples=[dataset[i] for i in indices]
        return (torch.stack([x['clean'] for x in samples]).to(device),
                torch.stack([x['observation'] for x in samples]).to(device),
                torch.stack([x['G'] for x in samples]).to(device))
    C,Y,G=batch([idx[0]])
    result['sample_start_ids']=[x['start_id'] for x in chosen]
    def module(bf16=False):return RolloutModule(model,sim,.2,network_bfloat16=bf16).train()
    if args.phase=='correctness':
        result['four_start_traces']=[]
        for initial_g in (4,3,2,1):
            progress('four_start_trace',G=initial_g)
            cc,yy,gg=batch([trace_indices[initial_g]])
            with torch.no_grad():out=module()(cc,yy,gg,retain_intermediates=True)
            trace=summarize_trace(out,yy);result['four_start_traces'].append(trace)
            assert trace['prediction_count']==initial_g and trace['logical_render_count']==2*initial_g-1
            assert trace['training_refinement_count']==0
            assert len(trace['steps'])==initial_g
            for k,t in enumerate(trace['steps']):
                assert t['reconstruction_Y_equals_saved_Y'] and t['all_finite'] and t['input_is_previous_transition']
                assert t['current_g']==[initial_g-k] and t['reconstruction_G']==[initial_g]
                assert t['next_g']==([initial_g-k-1] if k+1<initial_g else None)
            trace['semantic_checks_passed']=True
            del out;cleanup(model)
        progress('terminal_only_positive')
        seed_all(cfg['seed']);out=module()(C,Y,G,retain_intermediates=True)
        ref=snapshot_forward(out);positive=probe(out,model)
        result['terminal_only_probe']=positive
        result['positive_nonzero']=nonzero_probe(positive)
        del out;cleanup(model)
        result['negative_controls']=[]
        for boundary in (1,2,3):
            progress('negative_control',boundary=boundary)
            count=[0]
            def break_input(net,inputs):
                count[0]+=1
                if count[0]==boundary+1:
                    return (inputs[0].detach(),inputs[1])
                return inputs
            hook=model.register_forward_pre_hook(break_input)
            seed_all(cfg['seed'])
            try:out=module()(C,Y,G,retain_intermediates=True)
            finally:hook.remove()
            equivalence=compare_forward(out,ref);negative=probe(out,model)
            expected=all(not r['connected'] for step in negative['steps'][:boundary] for n,r in step.items() if n!='step')
            later=all(r['connected'] and r['finite'] and r['max_abs']>0 for step in negative['steps'][boundary:] for n,r in step.items() if n!='step')
            value_equal=abs(negative['terminal_clean_loss']-positive['terminal_clean_loss'])<=cfg['acceptance']['negative_forward_atol']
            result['negative_controls'].append({'boundary_after_prediction':boundary,'forward_equivalence':equivalence,
              'probe':negative,'expected_pre_boundary_disconnection':expected,'later_branches_connected':later,
              'probe_value_equal':value_equal,'passed':expected and later and equivalence['passed'] and value_equal})
            del out;cleanup(model)
        progress('mixed_length_aggregation')
        cc,yy,gg=batch([trace_indices[4],trace_indices[1]])
        out=module()(cc,yy,gg,retain_intermediates=True)
        manual=[]
        for j in range(2):
            terms=[(t['clean_losses'][t['active_path_indices'].index(j)]+.2*t['reconstruction_losses'][t['active_path_indices'].index(j)])
                   for t in out['trace'] if j in t['active_path_indices']]
            manual.append(torch.stack(terms).mean())
        manual_mean=torch.stack(manual).mean()
        maxdiff=float((manual_mean-out['loss_mean']).abs().detach().cpu())
        out['loss_mean'].backward();mixed_grads=cpu_grads(model)
        result['mixed_length']={'G':[4,1],'prediction_count':out['prediction_count'],
          'logical_render_count':out['logical_render_count'],'path_losses':out['path_losses'].detach().cpu().tolist(),
          'manual_mean':float(manual_mean.detach().cpu()),'implemented_mean':float(out['loss_mean'].detach().cpu()),'max_abs':maxdiff}
        del out,manual,manual_mean,terms;cleanup(model)
        for j in [trace_indices[4],trace_indices[1]]:
            c,y,g=batch([j]);single=module()(c,y,g)
            (single['loss_sum']/2).backward();del single
        accum=cpu_grads(model);cmp=gradient_comparison(mixed_grads,accum)
        result['mixed_length']['gradient_accumulation']=cmp
        result['mixed_length']['passed']=(maxdiff<1e-7 and cmp['cosine']>0.99999 and cmp['same_connected_parameters']
            and cmp['relative_l2_difference']<=cfg['acceptance']['gradient_accumulation_relative_l2_max'])
        del mixed_grads,accum;cleanup(model)
        result['passed']=result['positive_nonzero'] and all(x['passed'] for x in result['negative_controls']) and result['mixed_length']['passed']
    elif args.phase=='parity':
        payload={};gradients={};samples={}
        conv_dtypes=[]
        h=model.input.register_forward_hook(lambda m,a,o:conv_dtypes.append(str(o.dtype)))
        for name,bf16 in [('fp32',False),('hybrid_bf16',True)]:
            progress('parity_forward',precision=name);cleanup(model);seed_all(cfg['seed'])
            out=module(bf16)(C,Y,G,retain_intermediates=True)
            payload[name]=summarize_trace(out,Y)
            samples[name]={'final_clean':out['trace'][-1]['clean'].detach().cpu(),
                           **{n:out['trace'][0][n].detach().cpu() for n in ['clean','epsilon_zernike','epsilon_tilt']}}
            payload[name]['terminal_only_probe']=probe(out,model,retain_graph=True)
            progress('parity_backward',precision=name)
            out['loss_mean'].backward()
            gradients[name]=cpu_grads(model)
            payload[name]['parameter_gradient']=grad_stats(gradients[name])
            payload[name]['first_conv_dtypes']=conv_dtypes[:];conv_dtypes.clear()
            del out;cleanup(model)
        h.remove()
        ref=payload['fp32'];cand=payload['hybrid_bf16']
        rel=abs(cand['loss_mean']-ref['loss_mean'])/max(abs(ref['loss_mean']),1e-12)
        comparisons={'global':gradient_comparison(gradients['fp32'],gradients['hybrid_bf16']),
            'early_input_weight':gradient_comparison(gradients['fp32'],gradients['hybrid_bf16'],['input.weight']),
            'late_clean_head_weight':gradient_comparison(gradients['fp32'],gradients['hybrid_bf16'],['clean_head.weight']),
            'late_zernike_head_weight':gradient_comparison(gradients['fp32'],gradients['hybrid_bf16'],['zernike_head.weight'])}
        step_rel=[]
        for a,b in zip(ref['steps'],cand['steps']):
            step_rel.append({n:abs(b[n][0]-a[n][0])/max(abs(a[n][0]),1e-12) for n in ['LC','LR']})
        connections=nonzero_probe(ref['terminal_only_probe']) and nonzero_probe(cand['terminal_only_probe'])
        bf16_observed=len(cand['first_conv_dtypes'])==4 and all(d=='torch.bfloat16' for d in cand['first_conv_dtypes'])
        passed=(bf16_observed and comparisons['global']['same_connected_parameters'] and rel<=cfg['acceptance']['parity_total_relative_max'] and
                comparisons['global']['cosine']>=cfg['acceptance']['parity_global_gradient_cosine_min'] and
                max(v for x in step_rel for v in x.values())<=cfg['acceptance']['per_step_loss_relative_drift_max'] and connections and
                all(x['parameter_gradient']['finite'] and all(t['all_finite'] for t in x['steps']) for x in payload.values()))
        result.update({'precision_results':payload,'relative_total_loss_difference':rel,'per_step_relative_differences':step_rel,
            'gradient_comparison':comparisons,'tensor_comparison':{n:tensor_comparison(samples['fp32'][n],samples['hybrid_bf16'][n]) for n in samples['fp32']},
            'bf16_native_autocast_observed':bf16_observed,
            'passed':passed,'recommended_precision':'hybrid_bf16' if passed else 'fp32',
            'support_scope':'Observed full real network+simulator forward and backward, not a long-run stability claim'})
    elif args.phase=='derivatives':
        from physical_bridge.simulator import Realization
        gt=dataset.load_realization(idx[0])
        base={'clean':C.detach().clone(),'epsilon_zernike':gt.epsilon_zernike.unsqueeze(0).to(device),
              'epsilon_tilt':gt.epsilon_tilt.unsqueeze(0).to(device)}
        gen=torch.Generator().manual_seed(cfg['seed']+71)
        projection=torch.randn(C.shape,generator=gen)/math.sqrt(C.numel())
        def scalar(values):
            output=sim.render(values['clean'],Realization(values['epsilon_zernike'],values['epsilon_tilt']),4.).output
            return (output.to('cpu',dtype=torch.float64)*projection.double()).sum()
        result['directional_derivatives']=[]
        for branch in base:
            for dindex in range(cfg['acceptance']['finite_difference_directions_per_branch']):
                progress('directional_derivative',branch=branch,direction=dindex)
                values={n:t.detach().clone().requires_grad_(n==branch) for n,t in base.items()}
                direction=torch.randn(values[branch].shape,generator=gen)
                direction=(direction/direction.square().mean().sqrt()).to(device)
                f=scalar(values);grad=torch.autograd.grad(f,values[branch])[0]
                automatic=float((grad.cpu().double()*direction.cpu().double()).sum())
                checks=[]
                for eps in cfg['acceptance']['finite_difference_eps']:
                    with torch.no_grad():
                        plus={**values,branch:values[branch]+eps*direction}
                        minus={**values,branch:values[branch]-eps*direction}
                        fd=float((scalar(plus)-scalar(minus))/(2*eps))
                    err=abs(automatic-fd);tol=cfg['acceptance']['finite_difference_atol']+cfg['acceptance']['finite_difference_rtol']*max(abs(automatic),abs(fd))
                    checks.append({'epsilon':eps,'autograd':automatic,'central_difference':fd,'absolute_error':err,
                      'relative_error':err/max(abs(automatic),abs(fd),1e-12),'tolerance':tol,'passed':err<=tol})
                result['directional_derivatives'].append({'branch':branch,'direction_index':dindex,'gradient_finite':finite(grad),
                    'gradient_norm':float(grad.norm().cpu()),'checks':checks,'passed':all(x['passed'] for x in checks)})
                del values,grad,f,direction,plus,minus;cleanup(model)
        result['passed']=all(x['passed'] for x in result['directional_derivatives'])
    elif args.phase=='performance':
        parity_path=Path(args.parity_result)
        if not parity_path.is_absolute():parity_path=ROOT/parity_path
        parity_path=parity_path.resolve(strict=True)
        parity=json.loads(parity_path.read_text())
        if parity.get('phase')!='parity' or parity.get('status') not in ('passed','failed','error'):
            raise RuntimeError('Performance requires a completed parity result, not an unfinished or different phase')
        for key in ['config_sha256','initial_model_sha256','initial_simulator_sha256','base_hashes_before','sample_start_ids']:
            if parity[key]!=result[key]:raise RuntimeError(f'Parity/performance identity mismatch: {key}')
        for key in ['code/acceptance.py','code/v031/core.py','code/v031/runtime.py','code/v031/data.py']:
            if parity['code_sha256'][key]!=result['code_sha256'][key]:raise RuntimeError(f'Parity source mismatch: {key}')
        parity_passed=parity.get('passed') is True and parity['status']=='passed'
        if not parity_passed and not args.fp32_only:
            raise RuntimeError('Parity did not pass; only an explicit --fp32-only performance preflight is allowed')
        fallback_reason=None
        if not parity_passed:
            fallback_reason={
              'reason':'Parity gate did not pass; explicitly requested FP32-only performance preflight. BF16 remains unaccepted.',
              'parity_status':parity['status'],'parity_exception':parity.get('exception'),
              'parity_last_event':parity.get('last_event')}
        precisions=[('fp32',False)] if args.fp32_only else [('fp32',False),('hybrid_bf16',True)]
        result.update({'parity_result_path':str(parity_path),'parity_result_sha256':sha256(parity_path),
          'parity_source_identity_verified':True,'bf16_parity_passed':parity_passed,
          'fp32_only_requested':args.fp32_only,'fallback_reason':fallback_reason,
          'performance_scope':'fp32_only_fallback' if not parity_passed else ('fp32_only_requested' if args.fp32_only else 'fp32_and_accepted_hybrid_bf16'),
          'measured_precisions':[name for name,_ in precisions],
          'cross_precision_speedup_available':not args.fp32_only,
          'performance_pass_does_not_override_parity_gate':True})
        result['measurements']=[]
        result['optimizer_state_allocated']=False
        result['ema_allocated']=False
        result['optimizer_step_time_seconds']=None
        result['update_timing_scope']='32-path forward/backward estimate; excludes optimizer.step, clipping, EMA, communication, loading and checkpoint IO; no actual optimizer update executed'
        result['device_total_bytes']=int(torch.npu.get_device_properties(device).total_memory)
        for name,bf16 in precisions:
            for mb in cfg['acceptance']['microbatches']:
                cc,yy,gg=batch(idx[:mb]);rollout=module(bf16)
                progress('performance_warmup',precision=name,microbatch=mb)
                cleanup(model)
                warm=rollout(cc,yy,gg);warm['loss_mean'].backward();del warm;cleanup(model)
                samples=[]
                for repeat in range(cfg['acceptance']['timing_repeats']):
                    progress('performance_measurement',precision=name,microbatch=mb,repeat=repeat)
                    cleanup(model);torch.npu.reset_peak_memory_stats();sync();start=time.perf_counter()
                    out=rollout(cc,yy,gg);sync();middle=time.perf_counter()
                    out['loss_mean'].backward();sync();end=time.perf_counter()
                    samples.append({'forward_seconds':middle-start,'backward_seconds':end-middle,'total_seconds':end-start,
                      'seconds_per_path':(end-start)/mb,'estimated_32path_forward_backward_seconds':(end-start)*32/mb,
                      'peak_allocated_bytes':int(torch.npu.max_memory_allocated()),'peak_reserved_bytes':int(torch.npu.max_memory_reserved()),
                      'all_gradients_finite':all(p.grad is None or finite(p.grad) for p in model.parameters())})
                    del out;cleanup(model)
                result['measurements'].append({'precision':name,'microbatch':mb,'samples':samples})
        result['passed']=all(s['all_gradients_finite'] for m in result['measurements'] for s in m['samples'])
    cleanup(model)
    result['final_model_sha256']=state_hash(model)
    result['final_simulator_sha256']=state_hash(sim)
    result['model_and_simulator_unchanged']=model_initial==result['final_model_sha256'] and sim_initial==result['final_simulator_sha256']
    result['base_hashes_after']=verify_source(cfg)
    result['passed']=bool(result['passed'] and result['model_and_simulator_unchanged'])
    result['status']='passed' if result['passed'] else 'failed'
    progress('complete',passed=result['passed'])
except BaseException as exc:
    result.update({'status':'error','passed':False,'exception':repr(exc),'traceback':traceback.format_exc()})
    progress('error',error=repr(exc))
    raise
raise SystemExit(0 if result.get('passed') else 1)

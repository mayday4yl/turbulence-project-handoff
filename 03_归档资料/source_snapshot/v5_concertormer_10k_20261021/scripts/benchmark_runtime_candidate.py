"""Short throughput search on verified train pairs; all updates discarded."""
from pathlib import Path
import argparse
import json
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models import build_model
from runtime.environment import configure, make_simulator, synchronize, memory_stats
from scripts.benchmark_distributed import read_completed_train_pairs
from training.runner import load_initial_model
from training.state import atomic_json
import training.core as core


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', required=True)
    p.add_argument('--initial', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--microbatch', type=int, required=True)
    p.add_argument('--method', choices=['D0', 'R0'], default='R0')
    p.add_argument('--policy', choices=['none', 'all', 'highest', 'highest_half', 'highest_third', 'highres', 'alternating', 'highres_half_low'], required=True)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--talking-backend', default='conv2d_equivalent')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--cross-step', action='store_true')
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise RuntimeError('Preserve previous evidence; choose a new output path')
    report = {'status': 'RUNNING', 'arguments': vars(args), 'measurements': [],
              'scope': 'runtime candidate, verified train only; zero formal updates',
              'started_unix': time.time(), 'formal_optimizer_updates': 0}
    def save():
        atomic_json(report, out)
    save()
    try:
        report['environment'] = configure(args.device)
        model = build_model(activation_checkpointing=args.policy != 'none',
                            checkpoint_policy='all' if args.policy == 'none' else args.policy,
                            talking_backend=args.talking_backend)
        load_initial_model(model, args.initial)
        model.to(args.device).train()
        sim = make_simulator(args.device).eval()
        for param in sim.parameters():
            param.requires_grad_(False)
        pairs, report['data_identity'] = read_completed_train_pairs(args.dataset)
        selected = pairs[:args.microbatch]
        clean = torch.stack([r[1] for r in selected]).to(args.device)
        observed = torch.stack([r[2] for r in selected]).to(args.device)
        del pairs
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(), betas=(.9, .999), eps=1e-8)
        parameters = [p for p in model.parameters() if p.requires_grad]
        # Wall-clock decomposition synchronizes at phase boundaries. The same
        # instrumentation applies to every candidate; confirm winner with the
        # uninstrumented B32 benchmark before selecting a formal runtime.
        current = {}
        def timed(name, fn):
            def wrapped(*a, **kw):
                synchronize(args.device)
                started = time.perf_counter()
                result = fn(*a, **kw)
                synchronize(args.device)
                current[name] = current.get(name, 0.) + time.perf_counter() - started
                return result
            return wrapped
        model.forward = timed('network_forward_seconds', model.forward)
        core.prepare_fp32 = timed('simulator_prepare_seconds', core.prepare_fp32)
        core.render_fp32 = timed('simulator_render_seconds', core.render_fp32)
        sim_calls = 1 if args.method == 'D0' else 7
        report['forward_calls_per_microbatch'] = {'network': 1 if args.method == 'D0' else 4, 'simulator_render': sim_calls}
        for index in range(args.warmup + args.updates):
            current.clear()
            optimizer.zero_grad(set_to_none=True)
            torch.npu.reset_peak_memory_stats(args.device)
            synchronize(args.device)
            started = time.perf_counter()
            report['active_repeat'] = index
            report['active_phase'] = 'forward'
            save()
            profiler = None
            if args.profile and index == args.warmup:
                import torch_npu
                trace_dir = str(out.with_suffix('')) + '_profile'
                report['profiler_directory'] = trace_dir
                profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True, profile_memory=False, with_stack=False,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir))
                profiler.start()
            result = core.unroll_training(model, sim, observed, clean, method=args.method,
                path_seed=20261021, network_precision=args.precision,
                retain_trace=args.cross_step and index == 0)
            if not torch.isfinite(result.loss).item():
                raise FloatingPointError('Nonfinite loss')
            if args.cross_step and index == 0 and args.method == 'R0':
                first = result.trace[0]
                gradients = torch.autograd.grad((result.final_clean-clean).square().mean(),
                    [first['clean'], first['raw'].epsilon_zernike, first['raw'].epsilon_tilt], retain_graph=True)
                report['terminal_cross_step_gradients'] = {}
                for name, grad in zip(['clean', 'zernike', 'tilt'], gradients):
                    record = {'finite': bool(torch.isfinite(grad).all().item()),
                              'norm': grad.float().norm().item(), 'nonzero': torch.count_nonzero(grad).item()}
                    if not record['finite'] or not record['nonzero']:
                        raise RuntimeError(f'Cross-step gradient failed: {name}')
                    report['terminal_cross_step_gradients'][name] = record
                del gradients
            report['active_phase'] = 'backward'
            save()
            timed('backward_seconds', result.loss.backward)()
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise RuntimeError(f'Missing gradients: {missing}')
            norm = timed('clip_seconds', torch.nn.utils.clip_grad_norm_)(parameters, 1., error_if_nonfinite=True)
            timed('optimizer_seconds', optimizer.step)()
            synchronize(args.device)
            record = {'phase': 'warmup' if index < args.warmup else 'timed', 'repeat': index,
                      'seconds': time.perf_counter()-started, **current,
                      'loss': result.loss.item(), 'gradient_norm': norm.item(),
                      'all_trainable_gradients_present': True, 'finite': True, **memory_stats(args.device)}
            report['measurements'].append(record)
            print(json.dumps({'event': 'candidate_update', **record}), flush=True)
            del result, norm
            save()
            if profiler is not None:
                profiler.stop()
                report['profiler_enabled_not_throughput_reference'] = True
                save()
        records = [r for r in report['measurements'] if r['phase'] == 'timed']
        median = statistics.median(r['seconds'] for r in records)
        report['summary'] = {'timed_updates': len(records), 'median_microbatch_seconds': median,
            'median_seconds_per_sample': median/args.microbatch,
            'ideal_four_card_B32_seconds_estimate': median*8/args.microbatch,
            'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in records),
            'not_actual_B32_measurement': True}
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
    report['finished_unix'] = time.time()
    save()
    print(json.dumps({'status': report['status'], 'summary': report.get('summary'), 'error': report.get('error')}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

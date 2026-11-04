#!/usr/bin/env python3
"""Bounded receiver examples. No optimizer, controller, training or process signals.

Use the existing de36 environment; see docs/meeting_20261104/NPU_RUNBOOK.md.
Every calculation writes into a NEW output directory. CPU disables NPU autoload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['inspect', 'simulator', 'unidb'])
    p.add_argument('--project-root', default='/data2/yanglin')
    p.add_argument('--device', choices=['cpu', 'npu'], default='cpu')
    p.add_argument('--physical-card', type=int)
    p.add_argument('--confirm-card-is-allocated', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--weights', choices=['EMA', 'RAW'], default='EMA')
    p.add_argument('--action', choices=['forward', 'restore'], default='forward')
    p.add_argument('--g', type=float, choices=[.5, 1, 2, 3, 4], default=4)
    p.add_argument('--seed', type=int, default=843001)
    p.add_argument('--backward', action='store_true')
    a = p.parse_args()
    root = Path(a.project_root).resolve()
    simroot = root / 'v5_v1_simulator_20261020'
    uniroot = root / 'unidb_d0_20261102'
    report = {'task': a.task, 'unix': time.time(), 'python': sys.executable,
              'device': a.device, 'optimizer_updates': 0,
              'scope': 'receiver demonstration; not a formal benchmark'}
    if a.task == 'inspect':
        # Read small metadata and stat the paths actually used by the examples.
        # Do not import torch, initialize a device or hash large model/data files.
        paths, errors = {}, []

        def check_file(value, label):
            if not isinstance(value, (str, Path)) or not str(value).strip():
                errors.append(f'{label}: expected a non-empty file path')
                return None
            path = Path(value)
            try:
                present = path.is_file()
            except OSError as exc:
                errors.append(f'{label}: cannot inspect {path}: {exc}')
                present = False
            paths[str(path)] = present
            if not present:
                errors.append(f'{label}: file missing or not a regular file: {path}')
                return None
            return path

        def expect_keys(obj, names, label):
            if not isinstance(obj, dict):
                errors.append(f'{label}: expected a JSON object')
                return False
            missing = [name for name in names if name not in obj]
            if missing:
                errors.append(f'{label}: missing keys: {", ".join(missing)}')
            return not missing

        def read_object(path, label, keys=()):
            path = check_file(path, label)
            if path is None:
                return None
            try:
                value = json.loads(path.read_text())
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f'{label}: cannot read JSON: {exc}')
                return None
            return value if expect_keys(value, keys, label) else None

        fixed = [simroot / 'simulator_v5.py',
                 simroot / 'vendor/v4_simulator.py',
                 simroot / 'assets/P2S_model.pt',
                 simroot / 'assets/dictionary.pt', simroot / 'assets/I0_I2.pt',
                 simroot / 'assets/CANONICAL_CPU_REFERENCE.npz',
                 root / 'v6_diffuir_l_20261027/env/bin/python',
                 uniroot / 'core.py',
                 uniroot / 'vendor/unidb_gou/DenoisingUNet_arch.py',
                 uniroot / 'vendor/unidb_gou/module_util.py',
                 uniroot / 'vendor/unidb_gou/loss.py',
                 uniroot / 'vendor/unidb_gou/sde_utils.py']
        for path in fixed:
            check_file(path, 'required source/asset')
        canonical = read_object(simroot / 'assets/CANONICAL_CPU_REFERENCE.json',
                                'canonical buffers', ('npz_sha256', 'physics', 'buffers', 'tilt_constant'))
        if canonical is not None:
            for key in ('physics', 'buffers'):
                if not isinstance(canonical[key], dict) or not canonical[key]:
                    errors.append(f'canonical buffers: {key} must be a non-empty object')
            if isinstance(canonical['buffers'], dict):
                for name, record in canonical['buffers'].items():
                    expect_keys(record, ('raw_bytes_sha256',), f'canonical buffer {name}')
        upstream = read_object(uniroot / 'protocol/UPSTREAM_SOURCE.json',
                               'UniDB upstream manifest', ('files',))
        if upstream is not None:
            records = upstream['files']
            if not isinstance(records, dict) or not records:
                errors.append('UniDB upstream manifest: files must be a non-empty object')
            else:
                for name, record in records.items():
                    if not isinstance(name, str) or not name:
                        errors.append('UniDB upstream manifest: invalid filename')
                        continue
                    expect_keys(record, ('sha256',), f'upstream {name}')
                    check_file(uniroot / 'vendor/unidb_gou' / name, f'upstream {name}')
        # core.sources_ok checks this manifest only when it exists.
        run_source = uniroot / 'protocol/RUN_SOURCE.json'
        if run_source.exists():
            frozen = read_object(run_source, 'UniDB run source manifest', ('files',))
            if frozen is not None:
                records = frozen['files']
                if not isinstance(records, dict) or not records:
                    errors.append('UniDB run source manifest: files must be a non-empty object')
                else:
                    for name, expected in records.items():
                        if not isinstance(name, str) or not name or not isinstance(expected, str):
                            errors.append('UniDB run source manifest: invalid filename/hash entry')
                            continue
                        check_file(uniroot / name, f'run source {name}')
        weight = read_object(uniroot / 'evidence/WEIGHT_DOWNLOAD.json',
                             'official UniDB weight record', ('path', 'sha256', 'file_id'))
        if weight is not None:
            check_file(weight['path'], 'official UniDB initialization weight')
            if weight['file_id'] != '1kIrvZ1X46jFRAYIScverDD3dT1F40Kip':
                errors.append('official UniDB weight record: unexpected file_id')
        gate = read_object(uniroot / 'evidence/DATA_GATE.json',
                           'validation data record', ('val_manifest', 'val_manifest_sha256'))
        if gate is not None:
            manifest = check_file(gate['val_manifest'], 'validation manifest')
            if manifest is not None:
                try:
                    with manifest.open() as stream:
                        row = json.loads(next(stream))
                except (OSError, UnicodeError, ValueError, StopIteration) as exc:
                    errors.append(f'validation manifest: cannot read first JSON row: {exc}')
                else:
                    if expect_keys(row, ('sample_id', 'clean_path', 'clean_sha256', 'states'),
                                   'validation first sample'):
                        report['first_validation_sample'] = row['sample_id']
                        check_file(row['clean_path'], 'first validation clean image')
                        states = row['states']
                        if not isinstance(states, dict) or not states:
                            errors.append('validation first sample: states must be a non-empty object')
                        else:
                            if f'{a.g:g}' not in states:
                                errors.append(f'validation first sample: requested g={a.g:g} is missing')
                            for strength, entry in states.items():
                                label = f'first validation state g={strength}'
                                if expect_keys(entry, ('path', 'sha256'), label):
                                    check_file(entry['path'], label)
        if a.checkpoint:
            check_file(a.checkpoint, 'requested project checkpoint')
            if not a.checkpoint_sha256:
                errors.append('requested project checkpoint: --checkpoint-sha256 is required for execution')
        report.update(paths=paths, errors=errors,
                      scope='preflight file presence and small metadata structure only; not full environment certification',
                      checks_not_performed=['model/data SHA or payload validation',
                                            'Python package import and runtime compatibility',
                                            'NPU initialization or resource allocation',
                                            'all validation rows or training data'],
                      path_resolution='metadata paths used as recorded; --project-root does not rewrite them',
                      status='FAIL' if errors else 'PASS')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if errors else 0
    if a.out is None:
        p.error('--out is required for calculations; it must not exist')
    if a.device == 'npu':
        if a.physical_card is None or a.physical_card < 0 or not a.confirm_card_is_allocated:
            p.error('NPU needs --physical-card N and --confirm-card-is-allocated after live resource check')
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(a.physical_card)
    else:
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['TASK_QUEUE_ENABLE'] = '0'
    sys.dont_write_bytecode = True
    out = a.out.resolve()
    # Fail closed rather than mixing this example with an old run.
    out.mkdir(parents=True, exist_ok=False)
    cache = out / 'ascend_cache'
    cache.mkdir()
    os.environ['ASCEND_CACHE_PATH'] = str(cache)
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    device = torch.device('cpu')
    if a.device == 'npu':
        import torch_npu
        torch.npu.set_device(0)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        if hasattr(torch.npu, 'aclnn'):
            torch.npu.aclnn.allow_hf32 = False
        device = torch.device('npu:0')
    report['torch'] = torch.__version__
    report['torch_path'] = torch.__file__
    report['physical_card'] = a.physical_card if a.device == 'npu' else None
    torch.manual_seed(a.seed)
    if a.device == 'npu':
        torch.npu.manual_seed(a.seed)

    def save(name, tensor):
        array = tensor.detach().float().cpu().numpy()
        np.save(out / (name + '.npy'), array, allow_pickle=False)
        rgb = array[0] if array.ndim == 4 else array
        Image.fromarray((np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).round().astype('uint8')).save(out / (name + '.png'))

    begin = time.monotonic()
    if a.task == 'simulator':
        sys.path.insert(0, str(simroot))
        from simulator_v5 import build_simulator, Realization
        sim = build_simulator(device=device)
        # Synthetic teaching fixture, not a training or validation image.
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        clean = torch.stack([x / 255., y / 255., ((x // 16 + y // 16) % 2).float()])[None].to(device)
        raw = sim.sample_realization(1, seed_zernike=a.seed, seed_tilt=a.seed + 1)
        save('clean', clean)
        report['simulator_source_sha256'] = digest(simroot / 'simulator_v5.py')
        report['fixture'] = 'synthetic RGB ramps and checkerboard; fresh fixed Gaussian raw xi'
        report['xi_seed'] = [a.seed, a.seed + 1]
        report['strengths'] = {}
        with torch.no_grad():
            prepared = sim.prepare_image(clean)
            for g in [0, .5, 1, 2, 3, 4]:
                result = sim.render_prepared(prepared, raw, g)
                assert bool(torch.isfinite(result.output).all())
                save('g' + str(g), result.output)
                if g == 4:
                    save('g4_blur_only', result.blur_only)
                    np.save(out / 'g4_displacement_pixels.npy', result.effective_state.displacement.cpu().numpy(), allow_pickle=False)
                report['strengths'][str(g)] = {
                    'max_abs_from_clean': float((result.output-clean).abs().max()),
                    'flow_rms_pixels': float(result.effective_state.displacement.square().mean().sqrt()),
                    'min': float(result.output.min()), 'max': float(result.output.max())}
        assert report['strengths']['0']['max_abs_from_clean'] <= 2e-5
        if a.backward:
            c = clean.detach().requires_grad_(True)
            z = raw.epsilon_zernike.detach().requires_grad_(True)
            t = raw.epsilon_tilt.detach().requires_grad_(True)
            result = sim.render(c, Realization(z, t), 2.)
            loss = (result.output-clean).square().mean()
            grad = torch.autograd.grad(loss, (c, z, t))
            report['first_order_gradient_at_g2'] = [
                {'finite': bool(torch.isfinite(v).all()), 'norm': float(v.norm())} for v in grad]
            assert all(v['finite'] and v['norm'] > 0 for v in report['first_order_gradient_at_g2'])
            report['backward_scope'] = 'one synthetic point, C/raw-xi; not finite-difference acceptance, g derivative or full BPTT'
    else:
        sys.path.insert(0, str(uniroot))
        from core import initialize, process
        gate = json.loads((uniroot / 'evidence/DATA_GATE.json').read_text())
        manifest = Path(gate['val_manifest'])
        assert digest(manifest) == gate['val_manifest_sha256']
        with manifest.open() as f:
            row = json.loads(next(f))
        inp = row['states'][f'{a.g:g}']
        assert digest(inp['path']) == inp['sha256']
        assert digest(row['clean_path']) == row['clean_sha256']
        observation = torch.from_numpy(np.load(inp['path'], allow_pickle=False))[None].to(device)
        clean = torch.from_numpy(np.load(row['clean_path'], allow_pickle=False))[None].to(device)
        net, identity = initialize(device)
        report['initialization'] = identity
        if a.checkpoint:
            if not a.checkpoint_sha256:
                p.error('--checkpoint-sha256 is required; use the verified weight register')
            actual = digest(a.checkpoint)
            if actual != a.checkpoint_sha256:
                raise ValueError('checkpoint SHA mismatch')
            # Only project-authored, SHA-verified full checkpoints: they contain NumPy RNG.
            checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
            state = checkpoint['ema']['network'] if a.weights == 'EMA' else checkpoint['network']
            net.load_state_dict(state, strict=True)
            report['checkpoint'] = {'path': str(a.checkpoint), 'sha256': actual, 'weights': a.weights,
                                    'update': checkpoint['update']}
            del checkpoint, state
        net.eval()
        sde = process(device)
        sde.set_model(net)
        sde.set_mu(observation)
        calls = [0]
        hook = net.register_forward_hook(lambda m, args, value: calls.__setitem__(0, calls[0]+1))
        try:
            with torch.no_grad():
                if a.action == 'restore':
                    pred = sde.reverse_sde(observation, save_states=False)
                    save('prediction', pred)
                    assert calls[0] == 100
                else:
                    pred = sde.noise_fn(observation, torch.tensor([50.], device=device))
                    assert calls[0] == 1
                assert pred.shape == observation.shape and bool(torch.isfinite(pred).all())
        finally:
            hook.remove()
        save('input', observation)
        save('clean', clean)
        report.update(action=a.action, nfe=calls[0], sample_id=row['sample_id'], g=a.g,
                      input_sha256=inp['sha256'], clean_sha256=row['clean_sha256'],
                      note='B1 demonstration RNG differs from formal B4 per-source validation; forward is NOT a restored image')
    report.update(status='PASS', seconds=time.monotonic()-begin)
    (out / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

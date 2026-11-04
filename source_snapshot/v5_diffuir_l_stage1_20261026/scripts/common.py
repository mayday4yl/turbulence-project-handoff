"""Explicit adapters around the unchanged, pinned official DiffUIR implementation."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data2/yanglin/v5_concertormer_10k_20261021/data/v5_places_c50_s2000_v250_t250_g4_r4_20261021')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if hasattr(torch, 'npu'):
        torch.npu.manual_seed_all(value)


def setup_npu():
    import torch_npu
    torch.set_num_threads(4)
    torch.npu.set_device('npu:0')
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    # Official classes use the supplied tensor device. CUDA CLI/Trainer are not invoked.
    return torch.device('npu:0')


def build_model():
    from src.model import ResidualDiffusion, UnetRes
    net = UnetRes(dim=64, dim_mults=(1, 2, 4, 8), num_unet=1,
                  condition=True, objective='pred_res', test_res_or_noise='res')
    return ResidualDiffusion(net, image_size=256, timesteps=1000,
                             delta_end=1.8e-3, sampling_timesteps=3,
                             ddim_sampling_eta=0., objective='pred_res',
                             loss_type='l1', condition=True, sum_scale=.01,
                             test_res_or_noise='res')


def state_check(model, state):
    expected = model.state_dict()
    result = dict(missing=sorted(set(expected)-set(state)), unexpected=sorted(set(state)-set(expected)),
                  shape_mismatch=[], dtype_mismatch=[], value_mismatch=[], nonfinite=[])
    for k in set(expected) & set(state):
        if expected[k].shape != state[k].shape:
            result['shape_mismatch'].append(k)
        if expected[k].dtype != state[k].dtype:
            result['dtype_mismatch'].append(k)
        if state[k].is_floating_point() and not torch.isfinite(state[k]).all():
            result['nonfinite'].append(k)
    if any(result.values()):
        return result
    model.load_state_dict(state, strict=True)
    for k, v in model.state_dict().items():
        if not torch.equal(v.cpu(), state[k].cpu()):
            result['value_mismatch'].append(k)
    return result


def load_official(branch='ema', audit_path=None):
    if branch not in {'ema','model'}:
        raise ValueError('Choose the explicit official model or EMA branch')
    path = ROOT/'weights/model-300.pt'
    if path.stat().st_size != 580824471:
        raise RuntimeError('Official file size mismatch')
    digest = sha(path)
    if digest != 'b9d07ab6b32d37d895be65c50e99bbe6b2ae6e32823e7304ec2167bc5f18843f':
        raise RuntimeError('Downloaded/remote checkpoint SHA256 mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model()
    raw = checkpoint['model']
    ema_all = checkpoint['ema']
    ema = {k.removeprefix('ema_model.'): v for k, v in ema_all.items() if k.startswith('ema_model.')}
    audits = {}
    for tag, state in [('model', raw), ('ema_model', ema)]:
        audits[tag] = state_check(model, state)
        if any(audits[tag].values()):
            write_json(audit_path or ROOT/'evidence/CHECKPOINT_AUDIT_FAILURE.json', audits)
            raise RuntimeError(f'Strict official {tag} load failed')
    target = ema if branch == 'ema' else raw
    model.load_state_dict(target, strict=True)
    count = sum(p.numel() for p in model.parameters())
    if not 36_000_000 <= count <= 36_600_000:
        raise RuntimeError(f'Unexpected Large parameter count {count}')
    info = {'status': 'PASS', 'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
            'top_level_keys': list(checkpoint), 'step': int(checkpoint['step']),
            'architecture': {'dim':64,'dim_mults':[1,2,4,8], 'condition':True, 'num_unet':1,
                             'objective':'pred_res','parameters':count},
            'state_tensors':len(target), 'ema_keys':len(ema_all), 'audits':audits,
            'selected_branch':branch,'checkpoint_training_task_metadata':checkpoint.get('task', None),
            'no_key_rewrites':True, 'ema_prefix_selection':'ema_model. as used by official Trainer.test'}
    if branch == 'ema':
        before = {k: v.clone() for k,v in model.named_buffers()}
        # Match official Trainer.test: reset inference schedule, before device transfer.
        model.init()
        info['official_test_init_changed_buffers'] = [k for k,v in model.named_buffers() if not torch.equal(v,before[k])]
    if audit_path:
        write_json(audit_path,info)
    return model, info


def load_array(path, expected_sha):
    if sha(path) != expected_sha:
        raise RuntimeError(f'Data hash mismatch: {path}')
    a = np.load(path, allow_pickle=False)
    if a.dtype != np.float32 or a.shape != (3,256,256) or not np.isfinite(a).all():
        raise RuntimeError(f'Data format mismatch: {path}')
    return torch.from_numpy(a.copy()).unsqueeze(0)


def val4_pairs():
    selection = json.loads((ROOT/'protocol/V5_VAL4_SELECTION.json').read_text())
    return [{'id':s['sample_id'], 'clean':load_array(DATA/s['clean_ref'],s['clean_sha256']),
             'observation':load_array(DATA/s['observation_ref'],s['observation_sha256'])}
            for s in selection['samples']]


def image_metrics(pred, target):
    from skimage.metrics import structural_similarity
    p = pred.detach().float().cpu().numpy().squeeze(0)
    c = target.detach().float().cpu().numpy().squeeze(0)
    d = p.astype(np.float64)-c.astype(np.float64)
    mse = float(np.mean(d*d))
    return {'psnr':float(-10*np.log10(mse)), 'ssim':float(structural_similarity(
        c.transpose(1,2,0), p.transpose(1,2,0), data_range=1.,channel_axis=2,
        gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)),
        'mae':float(np.mean(np.abs(d))), 'mse':mse}


def evaluate(model, pairs, device, output, prefix='official_ema', random_seed=12345):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model.eval()
    rows=[]
    for pair in pairs:
        x=pair['observation'].to(device); clean=pair['clean']
        calls=[]
        def hook(module, args, result):
            calls.append({'shape':list(args[0].shape), 'time_condition':args[1].detach().cpu().tolist(),
                          'output_finite':bool(torch.isfinite(result).all()),
                          'output_min':float(result.min()),'output_max':float(result.max())})
        handle=model.model.unet0.register_forward_hook(hook)
        seed(random_seed)
        torch.npu.synchronize(); torch.npu.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.no_grad():
            outputs=model.sample(x,last=True,task='blur')
        torch.npu.synchronize(); elapsed=time.perf_counter()-start
        handle.remove()
        pred=outputs[-1]
        finite=bool(torch.isfinite(pred).all())
        if not finite or len(calls)!=3:
            raise RuntimeError('Inference finite/NFE gate failed')
        row={'id':pair['id'],'input_metrics':image_metrics(pair['observation'],clean),
             'metrics':image_metrics(pred,clean),'nfe':len(calls),'calls':calls,
             'returned_images':len(outputs),'min':float(pred.min()),'max':float(pred.max()),
             'finite':finite,'seconds':elapsed,'peak_allocated_bytes':torch.npu.max_memory_allocated(),
             'peak_reserved_bytes':torch.npu.max_memory_reserved(),
             'input_min':float(x.min()),'input_max':float(x.max())}
        for i,value in enumerate(outputs):
            np.save(output/f'{prefix}_{pair["id"]}_output{i}.npy',value.detach().cpu().numpy()[0])
        rows.append(row)
    result={'status':'PASS','branch':prefix,'seed_each_image':random_seed,'precision':'FP32',
            'samples':rows,'sample_count':len(rows),'test_images':0,
            'mean_metrics':{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in rows[0]['metrics']},
            'mean_input_metrics':{k:sum(r['input_metrics'][k] for r in rows)/len(rows) for k in rows[0]['input_metrics']},
            'timing_scope':'Synchronized sampling including audit hooks; excludes loading, metrics and file I/O',
            'metric_policy':'Original float RGB range1; no extra clamp/crop/resize; official internal clamps retained'}
    write_json(output/f'{prefix}_RESULT.json',result)
    return result

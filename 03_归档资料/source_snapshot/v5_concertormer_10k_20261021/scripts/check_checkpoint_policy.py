"""Small full-shaped CPU regression: execution policies preserve outputs/VJPs."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.v5_concertormer import V5Concertormer

torch.set_num_threads(2)
config = dict(img_channel=3, width=4, ffn_expansion_factor=2,
              enc_blk_nums=[2, 2, 2], middle_blk_num=2, dec_blk_nums=[2, 2, 2],
              enc_heads=[1, 1, 1], middle_heads=1, dec_heads=[1, 1, 1])
torch.manual_seed(87)
x = torch.randn(1, 3, 256, 256)
t = torch.tensor([.375])
reference = None
records = []
for policy in (sys.argv[1:] or ['none', 'all', 'highest', 'highest_half', 'highest_third', 'highres', 'alternating', 'highres_half_low']):
    torch.manual_seed(20261021)
    model = V5Concertormer(config=config, activation_checkpointing=policy != 'none',
                        checkpoint_policy='all' if policy == 'none' else policy,
                        talking_backend='conv2d_equivalent').train()
    image = x.clone().requires_grad_()
    clean, raw = model(image, t)
    loss = clean.square().mean()+.001*(raw.epsilon_zernike.square().mean()+raw.epsilon_tilt.square().mean())
    loss.backward()
    values = {'clean': clean.detach().clone(), 'input_gradient': image.grad.clone()}
    values.update({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    if reference is None:
        reference = values
    max_difference = max((values[k]-reference[k]).abs().max().item() for k in values)
    if max_difference > 1e-6 or values.keys() != reference.keys():
        raise AssertionError((policy, max_difference))
    records.append({'policy': policy, 'max_abs_output_or_gradient_difference': max_difference})
    del model, image, clean, raw, loss, values
print(json.dumps({'status': 'PASS', 'scope': 'small model CPU FP32, full 256/512 shapes', 'records': records}))

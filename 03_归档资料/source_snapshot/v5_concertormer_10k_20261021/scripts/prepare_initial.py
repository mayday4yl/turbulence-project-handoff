"""Save the one shared adapted U0 after an exact official checkpoint load."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import torch
from models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    target = Path(a.output)
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing shared U0: {target}')
    torch.set_num_threads(2)
    model = build_model(official_checkpoint=a.official, seed=20261021)
    assert model.pretrained_report['strict_load']
    record = {'model': model.state_dict(), 'provenance': model.pretrained_report,
              'seed': 20261021, 'formal_optimizer_updates': 0,
              'clean_parameterization': 'official_final_rgb_residual',
              'network_inputs': ['current_RGB_Z', 'global_time_t'],
              'simulator_sha256': '0f2f6f54a578b3c36cb69861815c7eee5476f75ccd91cac863bd001345bd9636'}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    result = {k:v for k,v in record.items() if k != 'model'}
    result.update(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()))
    target.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

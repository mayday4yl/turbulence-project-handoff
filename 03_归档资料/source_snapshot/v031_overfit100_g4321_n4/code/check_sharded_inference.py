"""Zero-update cross-card check using the unchanged full refined rollout."""
import argparse
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path('/data2/yanglin/v031_overfit100_g4321_n4')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu
    from v031.runtime import build, configure_npu, state_hash, save_json, sha256
    from v031.data import StartDataset
    import v031.evaluation as evaluation
    from check_distributed_gradients import FORWARD_ATOL, FORWARD_RTOL
    config = json.loads((root/'configs/train_v031.json').read_text())
    configure_npu(config)
    network, simulator = build(config, torch.device('npu:0'))
    checkpoint = root/'runs/formal/checkpoint_001000.pt'
    state, update = evaluation._checkpoint_state(checkpoint, config, 'raw')
    assert update == 1000
    network.load_state_dict(state, strict=True)
    del state
    dataset = StartDataset(config['dataset_root'])
    index = next(i for i,r in enumerate(dataset.records) if r['initial_G']==4)
    sample = dataset[index]
    refpath = root/'artifacts/evaluation/update_001000/raw/refined/rows'/(
        __import__('hashlib').sha256(sample['start_id'].encode()).hexdigest()+'.json')
    reference = json.loads(refpath.read_text())
    calibration = json.loads((root/'artifacts/calibration_update100/calibration.json').read_text())
    before = [state_hash(network), state_hash(simulator)]
    tensors = {}
    original_solver = evaluation.solve_adam
    calls = []
    def observed_solver(*a, **kw):
        result = original_solver(*a, **kw)
        i = len(calls)
        for name in ['epsilon_zernike','epsilon_tilt']:
            tensors[f'refined_{i}_{name}'] = getattr(result.realization,name).detach().cpu().clone()
        calls.append(i)
        return result
    evaluation.solve_adam = observed_solver
    def capture(k, g, clean, xi, y, G):
        for name in ['epsilon_zernike','epsilon_tilt']:
            tensors[f'network_{k}_{name}'] = getattr(xi,name).detach().cpu().clone()
    started = time.time()
    try:
        predictions = evaluation.rollout_predictions(model=network, simulator=simulator,
            observation=sample['observation'].unsqueeze(0).to('npu:0'), initial_G=4,
            mode='refined', solver_config=calibration['selected'], probe_callback=capture)
    finally:
        evaluation.solve_adam = original_solver
    target = sample['clean'].unsqueeze(0).to('npu:0')
    differences = []
    exact = True
    for k,(prediction, old) in enumerate(zip(predictions,reference['predictions'])):
        tensors[f'clean_{k}'] = prediction.clean.detach().cpu().clone()
        metrics = evaluation.clean_metrics(prediction.clean, target)[0]
        pairs = [(metrics['clean_mse'],old['clean_mse']),
                 (prediction.network_redegradation_mse[0],old['network_redegradation_mse'])]
        if prediction.refinement:
            now, prior = prediction.refinement[0], old['refinement']
            for key in ['initialization_source','iterations','gradient_connected','finite',
                        'nonzero_zernike_update','nonzero_tilt_update']:
                assert now[key] == prior[key], key
            for key in ['previous_candidate_objective','network_candidate_objective',
                        'selected_initial_objective','best_objective']:
                if now[key] is None or prior[key] is None:
                    assert now[key] == prior[key]
                else:
                    pairs.append((now[key],prior[key]))
        for a,b in pairs:
            assert math.isfinite(a) and abs(a-b) <= FORWARD_ATOL + FORWARD_RTOL*abs(b)
            differences.append(abs(a-b)); exact = exact and a==b
    assert before == [state_hash(network),state_hash(simulator)]
    assert len(calls)==3 and all(torch.isfinite(t).all() for t in tensors.values())
    torch.save(tensors,output/'tensors.pt')
    save_json(output/'result.json', {'status':'passed','physical_card':int(os.environ['V031_EVAL_CARD']),
        'checkpoint_sha256':sha256(checkpoint),'start_id':sample['start_id'],'mode':'refined',
        'reference_file':str(refpath),'reference_sha256':sha256(refpath),
        'scalar_atol':FORWARD_ATOL,'scalar_rtol':FORWARD_RTOL,
        'native_tensor_atol':1e-5,'native_tensor_rtol':1e-4,
        'max_absolute_metric_difference':max(differences),'scalar_metrics_exact':exact,
        'network_unchanged':True,'simulator_unchanged':True,'optimizer_updates':0,
        'work_start':started,'work_end':time.time(),'tensor_count':len(tensors)})


if __name__=='__main__':
    main()

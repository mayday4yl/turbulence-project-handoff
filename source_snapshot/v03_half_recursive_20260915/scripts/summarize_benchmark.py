#!/usr/bin/env python3
"""Summarize measured half-data benchmark costs without selecting science knobs."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def load_rows(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row['update'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate update rows: {path}')
    return rows


def summarize_run(root, step):
    run = root/'runs'/step['id']
    metrics = {r['update']:r for r in load_rows(run/'train_metrics.jsonl')}
    timing = {r['update']:r for r in load_rows(run/'timing.jsonl')}
    guard = run/'guard/guard_result.json'
    result = json.loads(guard.read_text()) if guard.exists() else None
    if result is not None:
        result = {k:result.get(k) for k in ('status','returncode','reason','elapsed_seconds','worker_elapsed_seconds','own_npu_processes_released')}
    config_arg = step['worker_args'][step['worker_args'].index('--config')+1]
    config = json.loads((root/config_arg).read_text())
    report = {'id':step['id'],'workload':config['workload'], 'configured_microbatch':config['training']['microbatch'],
              'effective_batch':config['training']['effective_batch'],'guard':result,
              'complete':6 in metrics and 6 in timing and result is not None and result['status']=='completed' and result['returncode']==0,
              'steady_update_ids':[u for u in (3,4,5) if u in metrics and u in timing],
              'all_logged_update_count':len(metrics)}
    if not report['steady_update_ids']:
        return report
    selected=[metrics[u] for u in report['steady_update_ids']]
    times=[timing[u]['full_update_wall_seconds'] for u in report['steady_update_ids']]
    seconds=sum(times);paths=sum(r['actual_global_paths'] for r in selected)
    predictions=sum(r['predictions'] for r in selected)
    signatures=[];lengths=Counter();physical_predictions=0
    for row in selected:
        ids=sorted(sample for rank in row['actual_rank_microbatches'] for chunk in rank['chunks'] for sample in chunk['real_sample_ids'])
        if len(ids)!=row['actual_global_paths'] or len(set(ids))!=len(ids):
            raise ValueError('Duplicate/missing global sample IDs')
        signatures.append({'update':row['update'],'global_sample_ids':ids,'path_counts':row['actual_path_counts'], 'seed':config['seed'],'sampling_seed':config['sampling_seed'],'dataset_root':config['dataset_root'],'curriculum':config['curriculum']})
        for name,count in row['actual_path_counts'].items():lengths[len(ast.literal_eval(name))]+=count
        physical_predictions+=sum(chunk['physical_predictions_including_dummy'] for rank in row['actual_rank_microbatches'] for chunk in rank['chunks'])
    report.update(physical_npu_count=selected[0]['world_size'],steady_total_wall_seconds=seconds,
                  steady_actual_global_paths=paths,steady_real_predictions=predictions,
                  trajectories_per_second=paths/seconds,network_predictions_per_second=physical_predictions/seconds,
                  useful_predictions_per_second=predictions/seconds,optimizer_updates_per_second=len(times)/seconds,
                  full_update_seconds_mean=statistics.fmean(times),full_update_seconds_min=min(times),full_update_seconds_max=max(times),
                  peak_allocated_bytes=max(r['peak_allocated_bytes_max_rank'] for r in metrics.values()),
                  peak_reserved_bytes=max(r['peak_reserved_bytes_max_rank'] for r in metrics.values()),
                  actual_gradient_accumulation_steps=sorted({r['microsteps'] for r in selected}),
                  workload_signature=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest(),
                  first_update_wall_seconds=timing.get(1,{}).get('full_update_wall_seconds'),
                  checkpoint_update_wall_seconds=timing.get(6,{}).get('full_update_wall_seconds'),
                  cold_start_note='First update may include shape compilation; startup and compilation are excluded from steady updates3-5. First-update time is not isolated compiler time.',
                  actual_prediction_count_exposure=dict(lengths),
                  length_exposure_trajectories_per_second={str(k):v/seconds for k,v in sorted(lengths.items())},
                  length_rate_scope='Exposure rates within the same mixed workload, not isolated per-length compute timings.',
                  steady_npu_hours=seconds*selected[0]['world_size']/3600,
                  total_attempt_npu_hours=(result['elapsed_seconds']*selected[0]['world_size']/3600 if result else None))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--queue',type=Path,default=Path('configs/benchmark_queue.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/benchmark_summary.json'))
    args=parser.parse_args();queue=json.loads((args.project/args.queue).read_text())
    rows=[summarize_run(args.project,step) for step in queue['steps']]
    groups={}
    for workload in sorted({row['workload']['kind'] for row in rows}):
        measured=[row for row in rows if row['workload']['kind']==workload and row['complete'] and len(row['steady_update_ids'])==3]
        signatures={row['workload_signature'] for row in measured}
        if len(signatures)>1:raise ValueError('Different sampled workload used in hardware comparisons: '+workload)
        groups[workload]={'completed_comparable_configs':len(measured),'same_workload_verified':bool(measured),
                          'fastest_measured':max(measured,key=lambda r:r['trajectories_per_second'])['id'] if measured else None}
    report={'status':'complete' if all(row['complete'] or row['guard'] is not None for row in rows) else 'partial',
            'source_queue':str(args.queue),'steady_update_ids':[3,4,5],'rows':rows,'comparison':groups,
            'formal_hyperparameters_selected':False,'noon_eta_requires_pilot_and_frozen_formal_batch':True}
    output=args.project/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['comparison']))


if __name__=='__main__':main()

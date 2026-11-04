"""Protect D0, build versioned physical states, and execute bounded A then B.

All NPU work belongs to explicitly spawned process groups. No external process
is terminated, and no method/precision/batch/budget is changed on failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
import traceback

from runtime.controller import atomic_json, file_sha, read_json, parse_npu_processes

ROOT = Path(__file__).resolve().parents[1]
CARDS = [2, 3, 4, 5, 6, 7]


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_rows(path):
    if not Path(path).exists():
        return []
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The append stream can be observed during a concurrent final write.
            continue
    return result


def validate_plan(plan):
    if (plan.get('schema') != 'v5_alignment_workflow_v1'
            or plan.get('execution_sequence') != ['A', 'B']
            or plan.get('physical_cards') != CARDS):
        raise ValueError('Only the authorized six-card A -> B workflow is supported')
    expected = {'A': (6000, 300, 5e-5, 1e-4, 'precomputed_current_g'),
                'B': (4000, 100, 2.5e-5, 5e-5, 'recursive_alignment')}
    for stage, values in expected.items():
        spec = plan['stages'][stage]
        if tuple(spec[k] for k in ('max_updates', 'warmup_updates', 'backbone_lr', 'new_lr', 'objective')) != values:
            raise ValueError(f'{stage} scientific specification changed')
    runtime = plan['runtime']
    fixed = {'world_size': 6, 'microbatch': 2, 'accumulation_steps': 4,
             'effective_batch': 48, 'network_precision': 'bf16', 'simulator_precision': 'fp32',
             'hf32': False, 'psf_backend': 'adjoint', 'talking_backend': 'conv2d_equivalent'}
    if any(runtime.get(k) != v for k, v in fixed.items()):
        raise ValueError('The approved six-card precision/batching/gradient path changed')
    if [plan.get(k) for k in ('train_log_interval', 'val_interval', 'checkpoint_interval')] != [10, 250, 250]:
        raise ValueError('Use the authorized log/validation/checkpoint intervals')
    if plan.get('test_used') is not False or plan.get('refinement') is not False:
        raise ValueError('Test is locked; routine refinement is OFF')
    if plan['stages']['A']['cache_boundaries'] != [2000, 4000, 6000]:
        raise ValueError('Expected three versioned 2000-update cache segments')
    return plan


def transient_failure(text):
    lower = text.lower()
    scientific = ('nonfinite', 'nan', 'out of memory', 'oom', 'gradient', 'identity mismatch',
                  'hash mismatch', 'configuration mismatch', 'leakage', 'corrupt', 'size mismatch')
    if any(token in lower for token in scientific):
        return False
    transient = ('input/output error', 'no space left on device', 'connection reset by peer',
                 'connection closed by peer', 'network is unreachable', 'temporary failure',
                 'socket timeout', 'hccl timeout', 'device lost', 'device task abort',
                 'resource temporarily unavailable', 'connection timed out')
    return any(token in lower for token in transient)


def inspect_checkpoint(path, method, *, best=False):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    schema = 'v5_full_training_checkpoint_v1' if method == 'D0' else 'v5_alignment_training_checkpoint_v1'
    required = {'model', 'optimizer', 'scheduler', 'sampler', 'rank_states', 'counters',
                'config', 'config_hash', 'initial_identity', 'completed_update', 'world_size'}
    if state.get('schema') != schema or required - state.keys():
        raise RuntimeError(f'{method} checkpoint is incomplete')
    if state.get('method') != method or state['world_size'] != 6 or len(state['rank_states']) != 6:
        raise RuntimeError('Checkpoint method or six-rank RNG state mismatch')
    update = state['completed_update']
    cap, warmup = {'D0': (10000, 300), 'A': (6000, 300), 'B': (4000, 100)}[method]
    if not isinstance(update, int) or not 0 <= update <= cap:
        raise RuntimeError('Checkpoint update exceeds the authorized stage budget')
    if (state['scheduler'].get('max_updates') != cap or state['scheduler'].get('warmup_updates') != warmup
            or state['config'].get('max_updates') != cap or state['config'].get('warmup_updates') != warmup):
        raise RuntimeError('Checkpoint scheduler horizon or warmup changed')
    if state['sampler']['completed_update'] != update or state['scheduler']['last_update'] != update:
        raise RuntimeError('Checkpoint optimizer/scheduler/sampler progress mismatch')
    if best and (state['best_update'] != update or update <= 0):
        raise RuntimeError('Selected checkpoint is not the recorded full-Val best')
    if state['config'].get('objective') != {'D0': 'clean_only', 'A': 'precomputed_current_g', 'B': 'recursive_alignment'}[method]:
        raise RuntimeError('Checkpoint objective mismatch')
    return json_safe({'path': str(path), 'sha256': file_sha(path), 'completed_update': update,
            'best_update': state['best_update'], 'best_psnr': state['best_psnr'],
            'config_hash': state['config_hash'], 'initial_identity': state['initial_identity'],
            'counters': state['counters']})


def preserve_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha(source) != file_sha(destination):
            raise RuntimeError(f'Frozen parent artifact changed: {source}')
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if file_sha(source) != file_sha(destination):
        raise RuntimeError('Parent archive copy identity mismatch')


class AlignmentWorkflow:
    def __init__(self, plan_path, authorization_path):
        self.plan_path, self.authorization_path = Path(plan_path).resolve(), Path(authorization_path).resolve()
        self.plan = validate_plan(read_json(self.plan_path))
        self.plan_sha, self.auth_sha = file_sha(self.plan_path), file_sha(self.authorization_path)
        self.authorization = read_json(self.authorization_path)
        if (self.authorization.get('status') != 'AUTHORIZED'
                or self.authorization.get('approved_execution_plan_sha256') != self.plan_sha):
            raise RuntimeError('The new user authorization does not bind this plan')
        self.root = self.plan_path.parent.parent
        self.reports = Path(self.plan['workflow_reports_dir'])
        self.preparation = Path(self.plan['preparation_root'])
        self.run_root, self.cache_root = Path(self.plan['run_root']), Path(self.plan['cache_root'])
        for path in (self.reports, self.preparation, self.run_root):
            path.mkdir(parents=True, exist_ok=True)
        self.children = []
        self.stop_requested = False
        clock = self.reports / 'WORKFLOW_CLOCK.json'
        self.started = read_json(clock)['started_at_unix'] if clock.exists() else time.time()
        self.code_sha = None

    def event(self, event, **values):
        with (self.reports / 'workflow_EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps(json_safe({'event': event, 'unix_time': time.time(), **values}), allow_nan=False) + '\n')

    def update(self, status, **values):
        atomic_json(self.reports / 'workflow_STATUS.json', {
            'status': status, 'unix_time': time.time(), 'controller_pid': os.getpid(),
            'started_at_unix': self.started, 'plan_path': str(self.plan_path), 'plan_sha256': self.plan_sha,
            'authorization_sha256': self.auth_sha, 'child_pids': [c.pid for c in self.children if c.poll() is None],
            'test_used': False, **values})

    def verify_frozen(self):
        if file_sha(self.plan_path) != self.plan_sha or file_sha(self.authorization_path) != self.auth_sha:
            raise RuntimeError('Authorized A/B plan changed while controller was active')
        manifest_path = Path(self.plan['code_manifest'])
        manifest = read_json(manifest_path)
        if manifest.get('schema') != 'v5_code_manifest_v1' or not manifest.get('files'):
            raise RuntimeError('Missing frozen A/B code manifest')
        actual_sha = file_sha(manifest_path)
        if self.code_sha is not None and actual_sha != self.code_sha:
            raise RuntimeError('Frozen A/B code manifest changed')
        for name, expected in manifest['files'].items():
            path = Path(name) if Path(name).is_absolute() else self.root / name
            if file_sha(path) != expected:
                raise RuntimeError(f'Frozen code identity mismatch: {name}')
        self.code_sha = actual_sha

    def environment(self):
        result = os.environ.copy()
        result.update({k: str(v) for k, v in self.plan.get('environment', {}).items()})
        result['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, CARDS))
        return result

    def python(self, module, *args):
        return ['bash', str(self.root / 'run_env.sh'), '-m', module, *map(str, args)]

    def distributed(self, module, *args):
        return self.python('torch.distributed.run', '--nproc_per_node', '6', '--master_port',
                           self.plan['master_port'], '-m', module, *args)

    def resources(self, stage):
        while True:
            if self.stop_requested:
                raise InterruptedError('Controller stop requested')
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=True)
            processes = parse_npu_processes(result.stdout, CARDS)
            atomic_json(self.reports / 'workflow_RESOURCES.json', {'unix_time': time.time(), 'stage': stage,
                        'selected_cards': CARDS, 'processes': processes, 'npu_smi_stdout': result.stdout})
            occupied = {str(card): pids for card, pids in processes.items() if pids}
            if not occupied:
                if shutil.disk_usage(self.root).free < 100 * 2**30:
                    raise RuntimeError('Insufficient disk for lossless cache and full checkpoints')
                return
            self.update('WAITING_RESOURCES', stage=stage, occupied_cards=occupied)
            time.sleep(60)

    def stop_handler(self, signum, frame):
        self.stop_requested = True

    def stop_owned_children(self):
        # Each child was started in its own new process session; no regex/PID guesses.
        for child in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        while any(c.poll() is None for c in self.children) and time.monotonic() < deadline:
            time.sleep(1)
        for child in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=30)

    def run_commands(self, stage, commands, *, npu=True, monitor_method=None):
        self.verify_frozen()
        if npu:
            self.resources(stage)
        logs = []
        self.children = []
        started = time.time()
        try:
            for index, command in enumerate(commands):
                log_path = self.reports / f'{stage}_{index}.log'
                stream = log_path.open('ab')
                logs.append((stream, log_path))
                self.event('child_start', stage=stage, index=index, command=command, log_path=str(log_path))
                child = subprocess.Popen(command, cwd=self.root, env=self.environment(),
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(child)
            last_monitor = 0
            while any(child.poll() is None for child in self.children):
                failed = [child for child in self.children if child.poll() not in (None, 0)]
                if self.stop_requested or failed:
                    self.stop_owned_children()
                    break
                if time.monotonic() - last_monitor >= 60:
                    self.update('RUNNING_CHILD', stage=stage, logs=[str(p) for _, p in logs])
                    if monitor_method:
                        self.progress_snapshot(monitor_method)
                    last_monitor = time.monotonic()
                time.sleep(1)
            codes = [child.wait() for child in self.children]
            self.event('children_finished', stage=stage, exit_codes=codes,
                       logs=[str(p) for _, p in logs], elapsed_seconds=time.time() - started)
            if self.stop_requested:
                raise InterruptedError('Only this controller owned child groups were stopped')
            if any(code != 0 for code in codes):
                raise RuntimeError(f'{stage} child failure: exits={codes}; logs={[str(p) for _, p in logs]}')
        finally:
            if any(child.poll() is None for child in self.children):
                self.stop_owned_children()
            for stream, _ in logs:
                stream.close()
            self.children = []

    def wait_d0(self):
        receipt = read_json(self.plan['old_successor_disable_receipt'])
        if receipt.get('status') != 'OLD_POST_D0_LAUNCH_DISABLED':
            raise RuntimeError('Obsolete post-D0 launch has not been safely disabled')
        run = Path(self.plan['D0_run'])
        while True:
            status = read_json(run / 'STATUS.json')
            if str(status.get('status', '')).startswith('TRAINING_COMPLETE'):
                if status.get('method') != 'D0' or status.get('completed_update') != 10000:
                    raise RuntimeError('D0 completion does not match its original budget')
                observed = self.reports / 'D0_COMPLETION_OBSERVED.json'
                if not observed.exists():
                    atomic_json(observed, {'unix_time': time.time(), 'D0_status_mtime': (run / 'STATUS.json').stat().st_mtime,
                                          'completed_update': 10000, 'D0_counters': status['counters']}, frozen=True)
                return self.freeze_parent('D0', run)
            if status.get('status') == 'FAILED':
                raise RuntimeError('Protected D0 failed; do not change or restart it implicitly')
            self.update('WAITING_PROTECTED_D0', D0_update=status.get('update', status.get('completed_update')),
                        D0_status=status.get('status'), old_successor='DISABLED')
            if self.stop_requested:
                raise InterruptedError('Stopped while waiting; D0 was never owned or signalled')
            time.sleep(60)

    def freeze_parent(self, method, run):
        run = Path(run)
        summary = read_json(run / 'STATUS.json')
        cap = {'D0': 10000, 'A': 6000, 'B': 4000}[method]
        if (not str(summary.get('status', '')).startswith('TRAINING_COMPLETE')
                or summary.get('completed_update') != cap or summary.get('test_used') is not False):
            raise RuntimeError(f'{method} has not finished its authorized full-budget selection')
        best = inspect_checkpoint(run / 'checkpoints/best.pt', method, best=True)
        latest = inspect_checkpoint(run / 'checkpoints/latest.pt', method)
        if latest['completed_update'] != cap or best['completed_update'] != summary['best_update']:
            raise RuntimeError('Best/latest selection does not match completion evidence')
        if (best['config_hash'] != latest['config_hash']
                or not math.isclose(best['best_psnr'], summary['best_psnr'], abs_tol=1e-8)
                or not math.isclose(latest['best_psnr'], summary['best_psnr'], abs_tol=1e-8)):
            raise RuntimeError('Parent best/latest scientific identity or selection score mismatch')
        if method == 'D0' and best['config_hash'] != self.plan['D0_scientific_config_hash']:
            raise RuntimeError('Protected D0 scientific configuration changed')
        curve = json_rows(run / 'validation_curve.jsonl')
        original = [row for row in curve if row.get('refinement') is False and row.get('count') == 1000]
        expected_nfe = 1 if method == 'D0' else 4
        if any(row.get('method') != method or row.get('source_count') != 250
               or row.get('inference_nfe') != expected_nfe for row in original):
            raise RuntimeError('Parent validation view, method, source count or NFE mismatch')
        observed = [row['update'] for row in original if row['update'] > 0]
        if sorted(observed) != list(range(250, cap + 1, 250)):
            raise RuntimeError('Parent full validation curve has missing or duplicate 250-update nodes')
        at_best = [row for row in original if row.get('update') == summary['best_update']]
        if len(at_best) != 1 or not math.isclose(at_best[0]['psnr'], summary['best_psnr'], abs_tol=1e-8):
            raise RuntimeError('Full original-Val best selection differs from the checkpoint')
        if max(row['psnr'] for row in original) > summary['best_psnr'] + 1e-8:
            raise RuntimeError('Selected parent is not maximum full original-Val PSNR')
        folder = self.reports / 'frozen' / method
        for name in ('STATUS.json', 'RUN_CONFIG.json', 'validation_curve.jsonl', 'metrics.jsonl',
                     'checkpoint_events.jsonl', 'INTERVAL_HISTORY.jsonl'):
            preserve_file(run / name, folder / name)
        preserve_file(run / 'checkpoints/best.pt', folder / 'best.pt')
        preserve_file(run / 'checkpoints/latest.pt', folder / 'latest.pt')
        record = {'status': 'FROZEN', 'method': method, 'cap': cap, 'best': best, 'latest': latest,
                  'summary_sha256': file_sha(folder / 'STATUS.json'), 'best_validation': at_best[0],
                  'test_used': False, 'all_original_results_preserved': True}
        atomic_json(folder / 'IDENTITY.json', record, frozen=True)
        self.event('parent_frozen', method=method, update=summary['best_update'], psnr=summary['best_psnr'])
        return folder

    def ensure_data_audits(self):
        audit = self.cache_root / 'audit'
        for mode, name in (('data', 'DATA_AUDIT.json'), ('raw', 'RAW_REPLAY.json')):
            path = audit / name
            if not path.exists():
                self.run_commands('audit_' + mode, [self.python('alignment_data.audit', '--base-root',
                    self.plan['dataset_root'], '--output-root', audit, '--mode', mode)], npu=False)
            result = read_json(path)
            if result.get('status') != 'PASS':
                raise RuntimeError(f'{name} did not pass')
        raw = read_json(audit / 'RAW_REPLAY.json')
        if raw.get('counts') != {'train': 8000, 'val': 1000, 'test': 1000} or raw.get('all_hashes_exact') is not True:
            raise RuntimeError('All original raw realizations must reproduce exactly')

    def ensure_cache(self, split, cycle=0):
        from alignment_data.plans import prepare_plan, cache_directory
        from alignment_data.dataset import AlignmentDataset
        prepare_plan(self.plan['dataset_root'], self.cache_root, split, cycle, self.plan['seed'])
        folder = cache_directory(self.cache_root, split, cycle)
        ready = folder / 'READY.json'
        if not ready.exists():
            base = ['--base-root', self.plan['dataset_root'], '--cache-root', self.cache_root,
                    '--split', split, '--cycle', cycle, '--seed', self.plan['seed']]
            commands = [self.python('alignment_data.generate', *base, '--device', f'npu:{rank}',
                        '--worker-index', rank, '--worker-count', 6) for rank in range(6)]
            started = time.time()
            for attempt in range(self.plan['max_transient_retries'] + 1):
                label = f'cache_{split}_{cycle:06d}_attempt{attempt}'
                try:
                    self.run_commands(label, commands)
                    break
                except RuntimeError:
                    tails = '\n'.join(p.read_text(errors='replace')[-30000:]
                                      for p in self.reports.glob(label + '_*.log'))
                    if attempt == self.plan['max_transient_retries'] or not transient_failure(tails):
                        raise
                    self.event('cache_transient_retry', split=split, cycle=cycle, attempt=attempt + 1,
                               policy='Reuse verified immutable existing records; same seeds and physics')
                    time.sleep(60)
            self.run_commands(f'cache_finalize_{split}_{cycle:06d}',
                              [self.python('alignment_data.generate', *base, '--finalize')], npu=False)
            self.event('cache_ready', split=split, cycle=cycle, elapsed_seconds=time.time() - started,
                       ready_sha256=file_sha(ready))
        # Validates frozen base/plan/manifest identities and complete item count.
        dataset = AlignmentDataset(self.plan['dataset_root'], self.cache_root, split=split, cycle=cycle)
        if len(dataset) != {'train': 8000, 'val': 1000}[split]:
            raise RuntimeError('Cache trajectory count changed')
        return ready

    def initial(self, stage, parent_folder):
        folder = self.preparation / stage
        folder.mkdir(parents=True, exist_ok=True)
        initial, report = folder / 'initial.pt', folder / 'INITIALIZATION.json'
        if report.exists():
            evidence = read_json(report)
            if evidence.get('status') != 'INITIALIZATION_PASS':
                raise RuntimeError('Previous initialization failed; preserve evidence and repair explicitly')
            if (evidence['parent_checkpoint_sha256'] != file_sha(parent_folder / 'best.pt')
                    or evidence['initial_checkpoint_sha256'] != file_sha(initial)):
                raise RuntimeError('Initialization parent identity mismatch')
        else:
            self.run_commands(f'initialize_{stage}', [self.python('alignment_checks.prepare_initial',
                '--stage', stage, '--parent-checkpoint', parent_folder / 'best.pt',
                '--parent-summary', parent_folder / 'STATUS.json', '--output', initial,
                '--report', report, '--dataset', self.plan['dataset_root'],
                '--seed', self.plan['seed'], '--device', 'npu:0')])
        return initial, report

    def config(self, stage, parent_folder, initial, *, endpoint, resume=None):
        from alignment_training.runner import RunConfig
        from dataclasses import asdict
        from alignment_training.authorization import require_alignment_authorization
        spec = self.plan['stages'][stage]
        runtime = {k: v for k, v in self.plan['runtime'].items()
                   if k not in ('simulator_precision', 'hf32')}
        audit = self.cache_root / 'audit/DATA_AUDIT.json'
        metadata = {'authorization_path': str(self.authorization_path), 'authorization_sha256': self.auth_sha,
                    'plan_path': str(self.plan_path), 'plan_sha256': self.plan_sha,
                    'execution_plan_sha256': self.plan_sha, 'code_manifest_sha256': self.code_sha,
                    'dataset_root': self.plan['dataset_root'],
                    'data_ready_sha256': file_sha(Path(self.plan['dataset_root']) / 'DATA_READY.json'),
                    'alignment_data_audit_path': str(audit), 'alignment_data_audit_sha256': file_sha(audit),
                    'raw_replay_path': str(self.cache_root / 'audit/RAW_REPLAY.json'),
                    'raw_replay_sha256': file_sha(self.cache_root / 'audit/RAW_REPLAY.json'),
                    'alignment_cache_root': str(self.cache_root), 'physical_cards': CARDS,
                    'test_used': False, 'refinement': False}
        selected = RunConfig(method=stage, objective=spec['objective'],
            output_dir=str(self.run_root / stage), initial_checkpoint=str(initial),
            readiness_report=str(self.preparation / stage / 'READY.json'),
            max_updates=spec['max_updates'], warmup_updates=spec['warmup_updates'],
            backbone_lr=spec['backbone_lr'], new_lr=spec['new_lr'],
            weight_decay=self.plan['weight_decay'], gradient_clip=self.plan['gradient_clip'],
            seed=self.plan['seed'], train_log_interval=self.plan['train_log_interval'],
            val_interval=self.plan['val_interval'], checkpoint_interval=self.plan['checkpoint_interval'],
            val_batch_size=self.plan['val_batch_size'], workers=self.plan['workers'],
            extended_validation_metrics=True, parent_training_summary=str(parent_folder / 'STATUS.json'),
            stop_after_update=endpoint, resume_checkpoint=str(resume) if resume else None,
            experiment_metadata=metadata, **runtime)
        selected.validate()
        require_alignment_authorization(selected)
        path = self.reports / 'configs' / f'{stage}_to_{endpoint:06d}.json'
        atomic_json(path, asdict(selected))
        return path, selected

    def readiness(self, stage, config_path, config, initialization_report):
        from alignment_training.runner import _validate_readiness
        path = Path(config.readiness_report)
        if not path.exists():
            self.run_commands(f'preflight_{stage}', [self.distributed('alignment_checks.preflight',
                '--config', config_path, '--initialization-report', initialization_report,
                '--d0-status', Path(self.plan['D0_run']) / 'STATUS.json', '--output', path)])
        _validate_readiness(config)

    def archive_uncommitted_tail(self, stage, checkpoint, reason):
        run = self.run_root / stage
        snap = self.reports / 'recovery' / stage / str(time.time_ns())
        snap.mkdir(parents=True)
        committed = checkpoint['completed_update']
        for filename in ('STATUS.json', 'metrics.jsonl', 'validation_curve.jsonl', 'val_a_curve.jsonl',
                         'checkpoint_events.jsonl', 'INTERVAL_HISTORY.jsonl'):
            source = run / filename
            if not source.exists():
                continue
            shutil.copy2(source, snap / filename)
            if filename.endswith('.jsonl') and filename != 'INTERVAL_HISTORY.jsonl':
                kept = [row for row in json_rows(source)
                        if row.get('update', row.get('completed_update', 0)) <= committed]
                temporary = source.with_suffix('.recovery.tmp')
                temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in kept))
                os.replace(temporary, source)
        # Preserve per-image diagnostics from an interrupted validation instead
        # of letting replay silently overwrite their original evidence.
        for directory_name in ('validation', 'validation_a'):
            directory = run / directory_name
            if not directory.exists():
                continue
            for artifact in directory.glob('update_*'):
                stem = artifact.name.split('.')[0]
                try:
                    artifact_update = int(stem.removeprefix('update_').split('_')[0])
                except ValueError:
                    continue
                if artifact_update > committed:
                    target = snap / directory_name / artifact.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(artifact), str(target))
        atomic_json(snap / 'RECOVERY.json', json_safe({'reason': reason, 'checkpoint': checkpoint,
                    'committed_update': committed, 'discarded_attempt_logs_preserved': True,
                    'method_or_seed_change': False}))
        self.event('recover_from_checkpoint', stage=stage, checkpoint=checkpoint, archive=str(snap), reason=reason)

    def train_segment(self, stage, parent_folder, initial, initialization_report, endpoint):
        run = self.run_root / stage
        status_path, latest = run / 'STATUS.json', run / 'checkpoints/latest.pt'
        if status_path.exists():
            status = read_json(status_path)
            if status.get('completed_update', -1) >= endpoint and 'COMPLETE' in status.get('status', ''):
                inspect_checkpoint(latest, stage)
                return
        resume = latest if latest.exists() else None
        if (run / 'RUN_CONFIG.json').exists() and resume is None:
            raise RuntimeError('Existing formal run has no resumable checkpoint; never silently restart')
        if resume is not None:
            committed = inspect_checkpoint(latest, stage)
            has_tail = any(row.get('update', row.get('completed_update', 0)) > committed['completed_update']
                           for filename in ('metrics.jsonl', 'validation_curve.jsonl', 'validation_a_curve.jsonl',
                                            'val_a_curve.jsonl', 'checkpoint_events.jsonl')
                           for row in json_rows(run / filename))
            if status_path.exists():
                prior_status = read_json(status_path)
                has_tail = (has_tail or prior_status.get('status') == 'FAILED'
                            or prior_status.get('update', prior_status.get('completed_update', 0)) > committed['completed_update'])
            if has_tail:
                self.archive_uncommitted_tail(stage, committed, 'Controller resumed with uncommitted log tail')
        config_path, config = self.config(stage, parent_folder, initial, endpoint=endpoint, resume=resume)
        self.readiness(stage, config_path, config, initialization_report)
        for attempt in range(self.plan['max_transient_retries'] + 1):
            try:
                self.run_commands(f'train_{stage}_to_{endpoint:06d}_attempt{attempt}',
                    [self.distributed('alignment_training.runner', '--config', config_path)], monitor_method=stage)
                result = read_json(status_path)
                if result.get('completed_update') != endpoint or 'COMPLETE' not in result.get('status', ''):
                    raise RuntimeError('Successful process exit lacks complete stage checkpoint evidence')
                inspect_checkpoint(latest, stage)
                self.progress_snapshot(stage)
                return
            except RuntimeError as exc:
                log = self.reports / f'train_{stage}_to_{endpoint:06d}_attempt{attempt}_0.log'
                tail = log.read_text(errors='replace')[-50000:] if log.exists() else ''
                if attempt == self.plan['max_transient_retries'] or not transient_failure(tail) or not latest.exists():
                    raise
                checkpoint = inspect_checkpoint(latest, stage)
                self.archive_uncommitted_tail(stage, checkpoint, str(exc))
                config_path, config = self.config(stage, parent_folder, initial, endpoint=endpoint, resume=latest)
                self.update('RECOVERING_TRANSIENT_FAILURE', stage=stage, attempt=attempt + 1,
                            resume_update=checkpoint['completed_update'])
                time.sleep(60)

    def progress_snapshot(self, stage):
        run = self.run_root / stage
        if not (run / 'STATUS.json').exists():
            return
        status = read_json(run / 'STATUS.json')
        updates = [r for r in json_rows(run / 'metrics.jsonl') if r.get('event') == 'train_log'][-20:]
        seconds = statistics.mean(r['seconds_per_update'] for r in updates) if updates else None
        curve = json_rows(run / 'validation_curve.jsonl')
        val_a = json_rows(run / 'val_a_curve.jsonl')
        saves = json_rows(run / 'checkpoint_events.jsonl')
        update = status.get('update', status.get('completed_update', 0))
        cap = self.plan['stages'][stage]['max_updates']
        val_cost = statistics.mean(r['elapsed_seconds'] for r in curve[-5:]) if curve else None
        if stage == 'A' and val_a:
            val_cost = (val_cost or 0) + statistics.mean(r['elapsed_seconds'] for r in val_a[-5:])
        save_cost = statistics.mean(r['checkpoint_elapsed_seconds'] for r in saves[-5:]) if saves else None
        remaining = (cap - update) * seconds if seconds is not None else None
        if remaining is not None and val_cost is not None:
            remaining += math.ceil((cap - update) / 250) * (val_cost + (save_cost or 0))
        snapshot = {'unix_time': time.time(), 'stage': stage, 'status': status,
                    'seconds_per_update_recent': seconds, 'samples_per_second_recent': 48 / seconds if seconds else None,
                    'mean_full_validation_seconds': val_cost, 'mean_checkpoint_seconds': save_cost,
                    'remaining_training_validation_seconds_estimate': remaining,
                    'ETA_excludes_future_cache_generation_and_later_stage': True,
                    'latest_validation': curve[-1] if curve else None, 'latest_val_A': val_a[-1] if val_a else None,
                    'test_used': False, 'free_disk_bytes': shutil.disk_usage(self.root).free}
        atomic_json(self.reports / f'{stage}_PROGRESS.json', snapshot)

    def final_report(self):
        records = {m: read_json(self.reports / 'frozen' / m / 'IDENTITY.json') for m in ('D0', 'A', 'B')}
        rows = {m: r['best_validation'] for m, r in records.items()}
        result = {'status': 'A6K_B4K_COMPLETE', 'results': rows,
                  'delta_B_minus_A_psnr': rows['B']['psnr'] - rows['A']['psnr'],
                  'delta_B_minus_D0_psnr': rows['B']['psnr'] - rows['D0']['psnr'],
                  'selected_identities': records, 'refinement': False, 'test_used': False,
                  'workflow_elapsed_seconds': time.time() - self.started}
        observed = read_json(self.reports / 'D0_COMPLETION_OBSERVED.json')
        result['cumulative_wall_seconds_including_full_D0_and_all_preparation'] = (
            observed['D0_counters']['run_elapsed_seconds'] + time.time() - observed['D0_status_mtime'])
        result['preparation_and_generation_events'] = [row for row in json_rows(self.reports / 'workflow_EVENTS.jsonl')
            if row.get('event') == 'children_finished' and not row.get('stage', '').startswith('train_')]
        atomic_json(self.reports / 'FINAL_RESULTS.json', result)
        lines = ['# V5 D0 → A6k → B4k 实际结果', '',
                 '三者使用相同原Val1000；D0为单步，A/B为真实均匀四步，refinement OFF。Val-A仅作单步诊断，Test1000未参与。', '',
                 '| 模型 | best update | PSNR | SSIM | LPIPS | 相对输入增益 |',
                 '|---|---:|---:|---:|---:|---:|']
        for m, row in rows.items():
            lines.append(f"| {m} | {row['update']} | {row['psnr']:.5f} | {row.get('ssim', float('nan')):.5f} | {row.get('lpips', float('nan')):.5f} | {row.get('psnr_gain_vs_input', float('nan')):.5f} |")
        lines += ['', f"B相对A：{result['delta_B_minus_A_psnr']:+.5f} dB；B相对D0：{result['delta_B_minus_D0_psnr']:+.5f} dB。",
                  '', '逐stage指标、完整配置/曲线/checkpoint身份和累计成本见FINAL_RESULTS.json及各run；未因验证波动改方法或扩大预算。',
                  '', 'Refinement ON和额外消融尚未执行，不根据本结果自动增加实验。']
        (self.reports / 'FINAL_REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
        return result

    def run(self):
        with (self.reports / 'workflow_LOCK').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, self.stop_handler)
            signal.signal(signal.SIGINT, self.stop_handler)
            try:
                self.verify_frozen()
                atomic_json(self.reports / 'WORKFLOW_CLOCK.json', {'started_at_unix': self.started}, frozen=True)
                atomic_json(self.reports / 'workflow_EXECUTION_PLAN.json', self.plan, frozen=True)
                parent = self.wait_d0()
                self.ensure_data_audits()
                environment = self.reports / 'ENVIRONMENT_AFTER_D0.json'
                if not environment.exists():
                    self.run_commands('environment_after_D0', [self.python('alignment_checks.environment_audit',
                                      '--output', environment)], npu=False)
                self.ensure_cache('val')
                self.ensure_cache('train', 0)
                initial, initialization = self.initial('A', parent)
                for cycle, endpoint in enumerate((2000, 4000, 6000)):
                    self.ensure_cache('train', cycle)
                    self.train_segment('A', parent, initial, initialization, endpoint)
                parent = self.freeze_parent('A', self.run_root / 'A')
                initial, initialization = self.initial('B', parent)
                self.train_segment('B', parent, initial, initialization, 4000)
                self.freeze_parent('B', self.run_root / 'B')
                self.final_report()
                self.update('ALL_AUTHORIZED_ALIGNMENT_COMPLETE', A_updates=6000, B_updates=4000,
                            final_report=str(self.reports / 'FINAL_REPORT_ZH.md'))
                return 0
            except BaseException as exc:
                if self.children:
                    self.stop_owned_children()
                self.update('STOPPED_REQUIRES_ATTENTION', error=repr(exc), traceback=traceback.format_exc(),
                            method_changed=False, budget_extended=False)
                self.event('workflow_failed', error=repr(exc), traceback=traceback.format_exc())
                return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    raise SystemExit(AlignmentWorkflow(args.plan, args.authorization).run())


if __name__ == '__main__':
    main()

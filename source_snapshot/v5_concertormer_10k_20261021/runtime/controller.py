"""Sequential, auditable V5 workflow orchestration; no implicit method changes.

The controller never trains directly, kills unrelated processes, retries failed
jobs, or constructs a test dataset. All accelerator jobs are explicit children.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('D0', 'R0', 'R0-D0Init')
RUNTIME_FIELDS = ('microbatch', 'world_size', 'network_precision', 'activation_checkpointing', 'talking_backend', 'checkpoint_policy')
SUCCESS = {'PASS', 'PASSED', 'READY', 'READY_FOR_TRAINING'}


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def scientific_hash(config):
    value = dict(config)
    value.pop('resume_checkpoint', None)
    value.pop('readiness_report', None)
    value.pop('stop_after_update', None)
    for key in ('train_log_interval', 'val_interval', 'checkpoint_interval', 'keep_recent_checkpoints'):
        value.pop(key, None)
    if value.get('extended_validation_metrics') is False:
        value.pop('extended_validation_metrics')
    if value.get('objective') == 'v5_dual':
        value.pop('objective')
    if value.get('psf_backend') == 'native':
        value.pop('psf_backend')
    if value.get('experiment_metadata'):
        value['experiment_metadata'] = dict(value['experiment_metadata'])
        value['experiment_metadata'].pop('execution_plan_sha256', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value, frozen=False):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f'Frozen workflow artifact changed: {path}')
        return
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_plan(plan):
    if plan.get('schema') != 'v5_workflow_v1':
        raise ValueError('Execution plan schema must be v5_workflow_v1')
    for key in ('dataset_root', 'initial_checkpoint', 'run_root', 'runtime', 'physical_cards', 'readiness_reports'):
        if key not in plan:
            raise ValueError(f'Missing execution plan field: {key}')
    runtime = plan['runtime']
    runtime.setdefault('checkpoint_policy', 'all')
    for key in (*RUNTIME_FIELDS, 'accumulation_steps', 'effective_batch'):
        if key not in runtime:
            raise ValueError(f'Missing selected runtime field: {key}')
    cards = plan['physical_cards']
    if not isinstance(cards, list) or len(cards) not in (1, 2, 4, 6) or len(set(cards)) != len(cards):
        raise ValueError('Select one, two, four or six distinct physical cards')
    if len(cards) == 6 and cards != [2, 3, 4, 5, 6, 7]:
        raise ValueError('Six-card de36 execution is limited to physical cards 2 through 7')
    if any(isinstance(card, bool) or not isinstance(card, int) or card < 0 for card in cards):
        raise ValueError('Physical cards must be nonnegative integer IDs')
    if runtime['world_size'] != len(cards):
        raise ValueError('world_size and selected physical cards differ')
    for key in ('microbatch', 'accumulation_steps', 'world_size'):
        if isinstance(runtime[key], bool) or not isinstance(runtime[key], int) or runtime[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if runtime['effective_batch'] not in (32, 36, 48) or runtime['microbatch'] * runtime['accumulation_steps'] * runtime['world_size'] != runtime['effective_batch']:
        raise ValueError('Selected batching must implement effective B32, B36 or B48')
    if runtime.get('psf_backend', 'native') not in ('native', 'adjoint'):
        raise ValueError('Unsupported PSF implementation')
    if runtime['network_precision'] not in ('fp32', 'bf16') or not isinstance(runtime['activation_checkpointing'], bool):
        raise ValueError('Invalid selected precision or activation checkpointing')
    if runtime['talking_backend'] not in ('conv3d', 'conv2d_equivalent'):
        raise ValueError('Unsupported talking convolution implementation')
    if runtime['checkpoint_policy'] not in ('all', 'highest', 'highest_half', 'highest_third', 'highres', 'alternating', 'highres_half_low'):
        raise ValueError('Unsupported checkpoint policy')
    overrides = plan.get('method_runtime_overrides', {})
    if set(overrides) - {'D0', 'R0'}:
        raise ValueError('R0-D0Init must inherit R0 execution settings')
    allowed = {'microbatch', 'accumulation_steps', 'activation_checkpointing', 'checkpoint_policy', 'psf_backend'}
    for method, override in overrides.items():
        if set(override) - allowed:
            raise ValueError('Per-method overrides may only change batching and recomputation')
        chosen = {**runtime, **override}
        if any(isinstance(chosen[k], bool) or not isinstance(chosen[k], int) or chosen[k] < 1
               for k in ('microbatch', 'accumulation_steps')):
            raise ValueError('Per-method batch/accumulation must be positive integers')
        if chosen['microbatch'] * chosen['accumulation_steps'] * chosen['world_size'] != runtime['effective_batch']:
            raise ValueError('Each method must retain the common effective batch')
        if (not isinstance(chosen['activation_checkpointing'], bool) or chosen['checkpoint_policy'] not in
                ('all', 'highest', 'highest_half', 'highest_third', 'highres', 'alternating', 'highres_half_low')):
            raise ValueError('Invalid per-method recomputation setting')
        if not plan.get('hccl_readiness_reports', {}).get(method):
            raise ValueError('Per-method execution requires its own matching HCCL evidence')
    selected_methods = ['D0', 'R0-D0Init'] if plan.get('d0_objective') == 'clean_only' else list(METHODS)
    if plan.get('methods', list(METHODS)) != selected_methods:
        raise ValueError('This controller implements sequential D0, R0, R0-D0Init only')
    updates = plan.get('max_updates', 10000)
    if isinstance(updates, bool) or not isinstance(updates, int) or not 300 < updates <= 10000 or updates % 500:
        raise ValueError('Budget must be a full-validation boundary, >300 and <=10000 updates')
    if not 30 <= plan.get('resource_retry_seconds', 60) <= 300:
        raise ValueError('Resource retry interval must be 30–300 seconds')
    if runtime['world_size'] > 1 and not plan.get('hccl_readiness_report'):
        raise ValueError('Multi-card execution requires same-runtime HCCL readiness evidence')
    if set(plan['readiness_reports']) != {'D0', 'R0'}:
        raise ValueError('Initial independent D0 and R0 readiness reports are required')
    if not 1024 <= plan.get('master_port', 29621) <= 65535:
        raise ValueError('Invalid torchrun master port')
    if plan.get('parallel_groups', False):
        raise ValueError('Parallel group execution is not implemented or silently enabled')
    if plan.get('code_manifest') is not None and not isinstance(plan['code_manifest'], str):
        raise ValueError('code_manifest must be a path string or null')
    refinement = plan.get('refinement_config')
    if refinement is not None:
        required = {'steps', 'learning_rate', 'proximity_weight', 'optimizer', 'proximity_reduction'}
        if not isinstance(refinement, dict) or set(refinement) != required:
            raise ValueError('refinement_config must be null or an explicit complete RefinementConfig object')
        if isinstance(refinement['steps'], bool) or not isinstance(refinement['steps'], int) or refinement['steps'] < 1:
            raise ValueError('Refinement steps must be a positive integer')
        if not math.isfinite(refinement['learning_rate']) or refinement['learning_rate'] <= 0 or not math.isfinite(refinement['proximity_weight']) or refinement['proximity_weight'] < 0:
            raise ValueError('Invalid explicit refinement learning rate/proximity weight')
        if refinement['optimizer'] not in ('adam', 'sgd') or refinement['proximity_reduction'] != 'all_elements_mean':
            raise ValueError('Unsupported explicit refinement optimizer/reduction')
    return plan


def parse_npu_processes(output, cards):
    """Fail closed on unrecognized npu-smi output; memory alone is insufficient."""
    if 'Process id' not in output:
        raise ValueError('npu-smi process table missing')
    table = output[output.index('Process id'):]
    found = {card: [] for card in cards}
    observed = set()
    for line in table.splitlines():
        empty = re.search(r'No running processes found in NPU\s+(\d+)', line)
        if empty:
            card = int(empty.group(1))
            if card in found:
                observed.add(card)
        row = re.match(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|', line)
        if row:
            card, _, pid = map(int, row.groups())
            if card in found:
                observed.add(card)
                found[card].append(pid)
    if observed != set(cards):
        raise ValueError(f'npu-smi has no unambiguous process entry for cards {sorted(set(cards) - observed)}')
    return found


def descendant_of(pid, ancestor):
    """Linux /proc ancestry; do not infer ownership from process names or user."""
    visited = set()
    while pid > 1 and pid not in visited:
        if pid == ancestor:
            return True
        visited.add(pid)
        try:
            stat = Path(f'/proc/{pid}/stat').read_text()
            pid = int(stat[stat.rfind(')') + 2:].split()[1])
        except (FileNotFoundError, ProcessLookupError):
            return False
    return False


def validate_runtime_report(report, runtime):
    if report.get('status') not in SUCCESS:
        raise RuntimeError(f'Readiness evidence did not pass: {report.get("status")}')
    actual = report.get('runtime', {})
    for field in RUNTIME_FIELDS:
        if actual.get(field, 'all' if field == 'checkpoint_policy' else None) != runtime[field]:
            raise RuntimeError(f'Readiness runtime mismatch: {field}: {actual.get(field)} != {runtime[field]}')
    if runtime['world_size'] == 6 or runtime.get('psf_backend', 'native') != 'native':
        for field in ('effective_batch', 'accumulation_steps', 'psf_backend'):
            if actual.get(field) != runtime.get(field):
                raise RuntimeError(f'Six-card/optimized readiness mismatch: {field}')


def validate_method_readiness(path, method, initial_checkpoint, runtime):
    report = read_json(path)
    validate_runtime_report(report, runtime)
    if method not in report.get('validated_methods', [report.get('method')]):
        raise RuntimeError(f'Readiness is not bound to {method}')
    if report.get('initial_checkpoint_sha256') != file_sha(initial_checkpoint):
        raise RuntimeError('Readiness does not bind the selected initialization')
    return report


def inspect_full_checkpoint(path, method, world_size, *, require_best=False):
    # These are this experiment's local full-state checkpoints, never untrusted
    # public weights; numpy RNG state is intentionally part of resume payloads.
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    required = {'model', 'optimizer', 'scheduler', 'sampler', 'rank_states', 'counters',
                'config', 'config_hash', 'initial_identity', 'completed_update', 'world_size'}
    if state.get('schema') != 'v5_full_training_checkpoint_v1' or required - state.keys():
        raise RuntimeError('Existing checkpoint is not complete/resumable')
    if state.get('method') != method or state['world_size'] != world_size:
        raise RuntimeError('Checkpoint method/world_size mismatch')
    if len(state['rank_states']) != world_size:
        raise RuntimeError('Checkpoint lacks rank RNG state')
    if require_best and state.get('best_update') != state['completed_update']:
        raise RuntimeError('Selected checkpoint is not a recorded full-validation best')
    if state['config_hash'] != scientific_hash(state['config']):
        raise RuntimeError('Full checkpoint stored configuration hash is inconsistent')
    result = {key: state.get(key) for key in ('schema', 'method', 'world_size', 'completed_update', 'best_update', 'best_psnr', 'config_hash', 'initial_identity')}
    del state
    return result


class Workflow:
    def __init__(self, plan_path, *, resume_existing=False):
        self.plan_path = Path(plan_path).resolve()
        self.plan = validate_plan(read_json(self.plan_path))
        self.plan_sha = file_sha(self.plan_path)
        self.resume_existing = resume_existing
        self.root = ROOT
        self.reports = Path(self.plan.get('workflow_reports_dir', self.root / 'reports'))
        self.reports.mkdir(parents=True, exist_ok=True)
        self.configs = self.reports / 'workflow_configs'
        self.configs.mkdir(exist_ok=True)
        self.run_root = Path(self.plan['run_root'])
        self.retry = self.plan.get('resource_retry_seconds', 60)
        self.child = None
        self.stop_requested = False
        self.started = time.time()
        self.status = {}
        self.code_manifest_sha = None

    def event(self, event, **kwargs):
        value = {'event': event, 'unix_time': time.time(), 'controller_pid': os.getpid(),
                 'plan_sha256': self.plan_sha, **kwargs}
        with (self.reports / 'workflow_EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def update(self, status, **kwargs):
        value = {'status': status, 'unix_time': time.time(), 'started_at_unix': self.started,
                 'elapsed_seconds': time.time() - self.started, 'controller_pid': os.getpid(),
                 'plan_path': str(self.plan_path), 'plan_sha256': self.plan_sha,
                 'test_used': False, **kwargs}
        if (status, kwargs) != (self.status.get('status'), self.status.get('detail')):
            self.event('status', status=status, **kwargs)
        self.status = {'status': status, 'detail': kwargs}
        atomic_json(self.reports / 'workflow_STATUS.json', value)

    def stop_handler(self, signum, _frame):
        self.stop_requested = True
        self.event('stop_requested', signal=signum)

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.stop_requested:
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        if self.stop_requested:
            raise InterruptedError('Controller stop requested')

    def wait_file(self, path, reason):
        path = Path(path)
        while not path.is_file():
            self.update(reason, waiting_for=str(path))
            self.sleep(self.retry)
        return path

    def validate_data(self):
        root = Path(self.plan['dataset_root'])
        ready_path = self.wait_file(root / 'DATA_READY.json', 'WAITING_DATA_READY')
        ready = read_json(ready_path)
        if ready.get('status') != 'DATA_READY' or ready.get('sample_counts') != {'train': 8000, 'val': 1000, 'test': 1000}:
            raise RuntimeError('Full V5 dataset is not READY with required counts')
        if ready.get('source_counts') != {'train': 2000, 'val': 250, 'test': 250} or ready.get('old_Y_reused') is not False or ready.get('test_recovery_model_evaluation') is not False:
            raise RuntimeError('Dataset provenance/counts/test isolation mismatch')
        for field, name in [('source_freeze_sha256', 'SOURCE_FREEZE.json'),
                            ('generation_identity_sha256', 'GENERATION_IDENTITY.json'),
                            ('sample_manifest_sha256', 'sample_manifest.jsonl')]:
            if ready.get(field) != file_sha(root / name):
                raise RuntimeError(f'DATA_READY identity mismatch: {name}')
        freeze = read_json(root / 'SOURCE_FREEZE.json')
        for field, name in [('source_manifest_sha256', 'source_manifest.jsonl'), ('sample_plan_sha256', 'sample_plan.jsonl')]:
            if freeze[field] != file_sha(root / name):
                raise RuntimeError(f'Data source/plan mismatch: {name}')
        return file_sha(ready_path)

    def child_environment(self):
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in self.plan.get('environment', {}).items()})
        environment['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(str(c) for c in self.plan['physical_cards'])
        return environment

    def verify_code_manifest(self):
        selected = self.plan.get('code_manifest')
        if selected is None:
            return None
        manifest_path = Path(selected)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        self.wait_file(manifest_path, 'WAITING_CODE_MANIFEST')
        actual_sha = file_sha(manifest_path)
        if self.code_manifest_sha is not None and actual_sha != self.code_manifest_sha:
            raise RuntimeError('Formal code manifest changed after controller start')
        manifest = read_json(manifest_path)
        if manifest.get('schema') != 'v5_code_manifest_v1' or not isinstance(manifest.get('files'), dict) or not manifest['files']:
            raise RuntimeError('Expected a nonempty v5_code_manifest_v1 files path-to-SHA dictionary')
        for name, expected in manifest['files'].items():
            path = Path(name)
            if not path.is_absolute():
                path = self.root / path
            if path.resolve() in (manifest_path.resolve(), self.plan_path):
                raise RuntimeError('Code manifest must exclude itself and the execution plan')
            if not re.fullmatch(r'[0-9a-f]{64}', str(expected)) or file_sha(path) != expected:
                raise RuntimeError(f'Frozen formal source identity mismatch: {name}')
        self.code_manifest_sha = actual_sha
        atomic_json(self.reports / 'workflow_CODE_MANIFEST.json',
                    {'source_path': str(manifest_path), 'sha256': actual_sha, 'manifest': manifest}, frozen=True)
        return actual_sha

    def wait_resources(self, stage):
        while True:
            try:
                result = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, timeout=30, check=True)
                processes = parse_npu_processes(result.stdout, self.plan['physical_cards'])
                foreign = {str(card): [pid for pid in pids if self.child is None or not descendant_of(pid, self.child.pid)]
                           for card, pids in processes.items()}
                foreign = {card: pids for card, pids in foreign.items() if pids}
                atomic_json(self.reports / 'workflow_RESOURCES.json', {'unix_time': time.time(), 'stage': stage,
                            'selected_cards': self.plan['physical_cards'], 'processes': processes, 'external_processes': foreign,
                            'npu_smi_stdout': result.stdout})
                if not foreign:
                    return
                reason = {'stage': stage, 'external_processes': foreign}
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                reason = {'stage': stage, 'resource_check_error': repr(exc)}
            self.update('WAITING_RESOURCES', **reason)
            self.sleep(self.retry)

    def run_command(self, stage, command):
        self.wait_resources(stage)
        if file_sha(self.plan_path) != self.plan_sha:
            raise RuntimeError('Execution plan changed while workflow was active')
        self.verify_code_manifest()
        if self.stop_requested:
            raise InterruptedError('Controller stop requested before child launch')
        log_path = self.reports / f'workflow_{stage}.log'
        self.event('command_start', stage=stage, command=command, log_path=str(log_path),
                   selected_cards=self.plan['physical_cards'], visible_devices=self.child_environment()['ASCEND_RT_VISIBLE_DEVICES'])
        started = time.time()
        with log_path.open('ab') as log:
            self.child = subprocess.Popen(command, cwd=self.root, env=self.child_environment(),
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.update('RUNNING_CHILD', stage=stage, child_pid=self.child.pid, command=command, log_path=str(log_path))
            self.event('child_spawned', stage=stage, pid=self.child.pid)
            while self.child.poll() is None:
                if self.stop_requested:
                    # The new process group contains only this launched child
                    # and its descendants; never match/kill processes by name.
                    self.event('child_group_sigterm', stage=stage, child_pid=self.child.pid)
                    try:
                        os.killpg(self.child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.event('child_group_sigkill_after_timeout', stage=stage, child_pid=self.child.pid)
                        os.killpg(self.child.pid, signal.SIGKILL)
                        self.child.wait(timeout=30)
                    break
                time.sleep(1)
            return_code = self.child.returncode
            pid = self.child.pid
            self.child = None
        self.event('command_finish', stage=stage, child_pid=pid, return_code=return_code,
                   elapsed_seconds=time.time() - started, log_path=str(log_path))
        if self.stop_requested:
            raise InterruptedError('Controller stopped its own child group; latest completed checkpoint retained')
        if return_code != 0:
            raise RuntimeError(f'{stage} failed (exit {return_code}); no automatic restart. See {log_path}')

    def launcher(self, module, *arguments):
        return ['bash', str(self.root / 'run_env.sh'), '-m', 'torch.distributed.run',
                '--nproc_per_node', str(self.plan['runtime']['world_size']),
                '--master_port', str(self.plan.get('master_port', 29621)),
                '-m', module, *map(str, arguments)]

    def runtime_for(self, method):
        key = 'D0' if method == 'D0' else 'R0'
        return {**self.plan['runtime'], **self.plan.get('method_runtime_overrides', {}).get(key, {})}

    def validate_hccl(self, method='R0'):
        runtime = self.runtime_for(method)
        if runtime['world_size'] == 1:
            return None
        key = 'D0' if method == 'D0' else 'R0'
        source = self.plan.get('hccl_readiness_reports', {}).get(key, self.plan['hccl_readiness_report'])
        path = self.wait_file(source, 'WAITING_HCCL_READINESS')
        evidence = read_json(path)
        validate_runtime_report(evidence, runtime)
        return {'path': str(path), 'sha256': file_sha(path)}

    def completed_summary(self, method, *, required=False, expected_config=None):
        path = self.run_root / method / 'STATUS.json'
        if not path.exists():
            if required:
                raise RuntimeError(f'Missing completed summary: {method}')
            return None
        summary = read_json(path)
        if not str(summary.get('status', '')).startswith('TRAINING_COMPLETE'):
            if required:
                raise RuntimeError(f'{method} has not finished selection')
            return None
        if summary.get('method') != method or summary.get('completed_update') != self.plan.get('max_updates', 10000) or summary.get('test_used') is not False:
            raise RuntimeError(f'{method} completion evidence does not match requested budget/test isolation')
        best = self.run_root / method / 'checkpoints/best.pt'
        record = inspect_full_checkpoint(best, method, self.plan['runtime']['world_size'], require_best=True)
        if record['completed_update'] != summary.get('best_update'):
            raise RuntimeError(f'{method} selected best differs from complete summary')
        if expected_config is not None:
            if record['config_hash'] != scientific_hash(expected_config):
                raise RuntimeError(f'{method} completed checkpoint configuration differs from execution plan')
            if record['initial_identity']['sha256'] != file_sha(expected_config['initial_checkpoint']):
                raise RuntimeError(f'{method} completed initialization differs from execution plan')
        if summary.get('counters', {}).get('optimizer_updates') != summary['completed_update']:
            raise RuntimeError(f'{method} optimizer-update accounting mismatch')
        curve = self.run_root / method / 'validation_curve.jsonl'
        values = [json.loads(line) for line in curve.read_text().splitlines() if line.strip()]
        observed = set()
        for value in values:
            if value.get('refinement') is False:
                if value.get('count') != 1000 or value.get('source_count') != 250 or value.get('method') != method:
                    raise RuntimeError(f'{method} validation curve contains a non-full validation')
                observed.add(value['update'])
        history_path = self.run_root / method / 'INTERVAL_HISTORY.jsonl'
        expected = set()
        history = [json.loads(x) for x in history_path.read_text().splitlines() if x.strip()]
        for i, h in enumerate(history):
            end = history[i+1]['effective_after_update'] if i+1 < len(history) else summary['completed_update']
            expected.update(u for u in range(h['effective_after_update']+1, end+1) if u % h['val_interval'] == 0)
            if h['stage_endpoint'] <= end:
                expected.add(h['stage_endpoint'])
        if observed != expected:
            raise RuntimeError(f'{method} full validation curve has missing/extra update boundaries')
        return summary

    def d0init_readiness(self, initial):
        runtime = self.runtime_for('R0-D0Init')
        hccl = self.validate_hccl()
        report_path = self.reports / 'workflow_R0-D0Init_READINESS.json'
        if report_path.exists():
            validate_method_readiness(report_path, 'R0-D0Init', initial, runtime)
            evidence = read_json(report_path)
            if evidence['hccl_evidence'] != hccl:
                raise RuntimeError('Inherited HCCL evidence changed')
            for source in evidence['source_reports']:
                if file_sha(source['path']) != source['sha256']:
                    raise RuntimeError('R0-D0Init chain evidence changed')
            return report_path
        probe_path = self.reports / 'workflow_R0-D0Init_CHAIN.json'
        command = ['bash', str(self.root / 'run_env.sh'), 'scripts/probe_backend.py', '--device', 'npu:0',
                   '--mode', 'chain', '--method', 'R0', '--initial', str(initial),
                   '--dataset', self.plan['dataset_root'], '--output', str(probe_path),
                   '--precision', runtime['network_precision'], '--microbatch', str(runtime['microbatch']),
                   '--talking-backend', runtime['talking_backend'], '--checkpoint-policy', runtime['checkpoint_policy'],
                   '--cross-step', '--repeats', '2', '--psf-backend', runtime.get('psf_backend', 'native')]
        if runtime['activation_checkpointing']:
            command.append('--checkpointing')
        self.run_command('R0-D0Init_chain_gate', command)
        probe = read_json(probe_path)
        if probe.get('status') != 'PASS' or probe.get('initial_sha256') != file_sha(initial):
            raise RuntimeError('D0-best R0 chain gate did not pass for actual selected checkpoint')
        args = probe.get('arguments', {})
        expected = {'mode': 'chain', 'method': 'R0', 'cross_step': True,
                    'precision': runtime['network_precision'], 'microbatch': runtime['microbatch'],
                    'checkpointing': runtime['activation_checkpointing'], 'talking_backend': runtime['talking_backend'],
                    'checkpoint_policy': runtime['checkpoint_policy'], 'psf_backend': runtime.get('psf_backend', 'native')}
        for field, value in expected.items():
            if args.get(field) != value:
                raise RuntimeError(f'R0-D0Init chain gate argument mismatch: {field}')
        cross = probe.get('terminal_cross_step_gradients', {})
        for key in ('first_clean', 'first_raw_zernike', 'first_raw_tilt'):
            value = cross.get(key, {})
            if value.get('finite') is not True or value.get('nonzero', 0) <= 0:
                raise RuntimeError(f'R0-D0Init lacks valid terminal cross-step gradient: {key}')
        evidence = {'status': 'READY_FOR_TRAINING', 'method': 'R0-D0Init',
                    'validated_methods': ['R0-D0Init'], 'initial_checkpoint_sha256': file_sha(initial),
                    'runtime': {**{key: runtime[key] for key in RUNTIME_FIELDS},
                                'effective_batch': runtime['effective_batch'],
                                'accumulation_steps': runtime['accumulation_steps'],
                                'psf_backend': runtime.get('psf_backend', 'native')},
                    'source_reports': [{'path': str(probe_path), 'sha256': file_sha(probe_path)}],
                    'hccl_evidence': hccl, 'optimizer_reset': True, 'formal_optimizer_updates': 0,
                    'parent_training_summary_sha256': file_sha(self.run_root / 'D0/STATUS.json')}
        atomic_json(report_path, evidence, frozen=True)
        return report_path

    def make_config(self, method, data_sha):
        runtime = self.runtime_for(method)
        self.validate_hccl(method)
        initial = Path(self.plan['initial_checkpoint'])
        parent_summary = None
        if method == 'R0-D0Init':
            self.completed_summary('D0', required=True)
            initial = self.run_root / 'D0/checkpoints/best.pt'
            if self.plan.get('d0_objective') == 'clean_only':
                initial = self.prepare_clean_d0_initialization(initial)
            parent_summary = str(self.run_root / 'D0/STATUS.json')
            readiness_path = self.d0init_readiness(initial)
        else:
            readiness_path = self.wait_file(self.plan['readiness_reports'][method], 'WAITING_METHOD_READINESS')
        validate_method_readiness(readiness_path, method, initial, runtime)
        config = {'method': method, 'output_dir': str(self.run_root / method), 'initial_checkpoint': str(initial),
                  'readiness_report': str(readiness_path), 'max_updates': self.plan.get('max_updates', 10000),
                  'warmup_updates': 300, 'microbatch': runtime['microbatch'],
                  'accumulation_steps': runtime['accumulation_steps'], 'world_size': runtime['world_size'],
                  'effective_batch': runtime['effective_batch'], 'backbone_lr': 5e-5, 'new_lr': 1e-4, 'weight_decay': 1e-4,
                  'gradient_clip': 1.0, 'network_precision': runtime['network_precision'],
                  'activation_checkpointing': runtime['activation_checkpointing'],
                  'checkpoint_policy': runtime['checkpoint_policy'],
                  'talking_backend': runtime['talking_backend'], 'seed': self.plan.get('seed', 20261021),
                  'val_interval': self.plan.get('val_interval', 250), 'save_interval': 1000,
                  'train_log_interval': self.plan.get('train_log_interval', 10),
                  'checkpoint_interval': self.plan.get('checkpoint_interval', 250),
                  'keep_recent_checkpoints': 2,
                  'extended_validation_metrics': self.plan.get('extended_validation_metrics', False),
                  'val_batch_size': self.plan.get('val_batch_size', 1),
                  'workers': self.plan.get('workers', 2), 'resume_checkpoint': None,
                  'parent_training_summary': parent_summary, 'ema': False,
                  'experiment_metadata': {'dataset_root': self.plan['dataset_root'], 'data_ready_sha256': data_sha,
                     'execution_plan_sha256': self.plan_sha, 'physical_cards': self.plan['physical_cards'],
                     'common_u0_sha256': file_sha(self.plan['initial_checkpoint']),
                     'code_manifest_sha256': self.code_manifest_sha, 'test_used': False}}
        if runtime.get('psf_backend', 'native') != 'native':
            config['psf_backend'] = runtime['psf_backend']
        if self.plan.get('d0_objective') == 'clean_only' and method == 'D0':
            config.update(objective='clean_only', new_lr=5e-5)
        return config

    def save_config(self, config):
        path = self.configs / f'{config["method"]}.json'
        # A resume path is operational; only this explicitly requested field may
        # change. All immutable/scientific fields remain exact.
        if path.exists():
            previous = read_json(path)
            a, b = dict(previous), dict(config)
            a.pop('resume_checkpoint', None)
            b.pop('resume_checkpoint', None)
            a.pop('stop_after_update', None)
            b.pop('stop_after_update', None)
            for key in ('train_log_interval', 'val_interval', 'checkpoint_interval'):
                a.pop(key, None); b.pop(key, None)
            if scientific_hash(a) != scientific_hash(b):
                raise RuntimeError(f'Existing method configuration changed: {path}')
        atomic_json(path, config)
        return path

    def validate_refinement_result(self, path, method, refinement):
        result = read_json(path)
        best = self.run_root / method / 'checkpoints/best.pt'
        summary = result.get('result', {})
        if (result.get('status') != 'BEST_FULL_VALIDATION_REFINEMENT_COMPLETE' or
                result.get('method') != method or result.get('best_checkpoint_sha256') != file_sha(best) or
                result.get('test_used') is not False or result.get('selected_without_refinement') is not True or
                result.get('refinement_configuration') != refinement or
                summary.get('count') != 1000 or summary.get('source_count') != 250 or
                summary.get('refinement') is not True or summary.get('gamma') != 1.):
            raise RuntimeError(f'{method} refinement evaluation is incomplete or identity/protocol differs')
        return result

    def run(self):
        if self.plan.get('stage_boundaries'):
            raise RuntimeError('Staged plans must use scripts/run_staged_workflow.py; automatic 10k execution is disabled')
        lock = (self.reports / 'workflow_LOCK').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another controller already holds this experiment workflow lock')
        signal.signal(signal.SIGTERM, self.stop_handler)
        signal.signal(signal.SIGINT, self.stop_handler)
        try:
            self.event('controller_start', resume_existing=self.resume_existing)
            atomic_json(self.reports / 'workflow_EXECUTION_PLAN.json', self.plan, frozen=True)
            self.verify_code_manifest()
            data_sha = self.validate_data()
            self.wait_file(self.plan['initial_checkpoint'], 'WAITING_INITIAL_WEIGHTS')
            self.validate_hccl()
            configs = {}
            for method in METHODS:
                config = self.make_config(method, data_sha)
                if self.completed_summary(method, expected_config=config):
                    configs[method] = self.save_config(config)
                    self.event('training_already_complete', method=method)
                    continue
                existing = Path(config['output_dir']) / 'RUN_CONFIG.json'
                if existing.exists():
                    if not self.resume_existing:
                        raise RuntimeError(f'{method} exists but is incomplete; rerun explicitly with --resume-existing after reviewing failure')
                    latest = Path(config['output_dir']) / 'checkpoints/latest.pt'
                    info = inspect_full_checkpoint(latest, method, self.plan['runtime']['world_size'])
                    if info['initial_identity']['sha256'] != file_sha(config['initial_checkpoint']):
                        raise RuntimeError('Resume initialization differs from the planned initial checkpoint')
                    if info['config_hash'] != scientific_hash(config):
                        raise RuntimeError('Resume scientific configuration differs from the frozen run')
                    config['resume_checkpoint'] = str(latest)
                    self.event('explicit_resume_selected', method=method, checkpoint=str(latest),
                               checkpoint_sha256=file_sha(latest), completed_update=info['completed_update'])
                configs[method] = self.save_config(config)
                self.run_command(f'{method}_training', self.launcher('training.runner', '--config', configs[method], '--factory', 'runtime.factories:training_components'))
                self.completed_summary(method, required=True, expected_config=config)
            refinement = self.plan.get('refinement_config')
            if refinement is None:
                self.update('TRAINING_COMPLETED_REFINEMENT_PENDING_CONFIG', methods=list(METHODS),
                            full_validation_size=1000, refinement='PENDING_EXPLICIT_CONFIGURATION',
                            independent_evaluation_entry='python -m training.evaluate_refinement --config <method.json> --refinement-config <explicit.json> --factory runtime.factories:training_components',
                            test_evaluation='NOT_RUN')
                return 0
            ref_path = self.configs / 'REFINEMENT.json'
            atomic_json(ref_path, refinement, frozen=True)
            for method in METHODS:
                result_path = self.run_root / method / 'REFINEMENT_EVALUATION.json'
                best_path = self.run_root / method / 'checkpoints/best.pt'
                if result_path.exists():
                    self.validate_refinement_result(result_path, method, refinement)
                    self.event('refinement_already_complete', method=method)
                    continue
                self.run_command(f'{method}_refinement_validation', self.launcher('training.evaluate_refinement',
                                 '--config', configs[method], '--refinement-config', ref_path,
                                 '--factory', 'runtime.factories:training_components'))
                self.validate_refinement_result(result_path, method, refinement)
            self.update('WORKFLOW_COMPLETE', methods=list(METHODS), full_validation_size=1000,
                        refinement='ON final selected best, no further checkpoint selection', test_evaluation='NOT_RUN')
            return 0
        except InterruptedError as exc:
            self.update('STOPPED', error=str(exc), resume_requires_explicit_flag=True)
            return 130
        except BaseException as exc:
            self.update('FAILED_REQUIRES_ATTENTION', error=repr(exc), traceback=traceback.format_exc(),
                        automatic_retry=False, resume_requires_explicit_flag=True)
            return 1
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

"""Execute one explicitly confirmed phase, then return for result review."""
from pathlib import Path
import fcntl
import json
import signal
import traceback

from .controller import (Workflow, atomic_json, read_json, inspect_full_checkpoint,
                         scientific_hash, file_sha)


class StagedWorkflow(Workflow):
    def __init__(self, plan_path, *, stage_target, resume_existing=False):
        super().__init__(plan_path, resume_existing=resume_existing)
        if self.plan.get('stage_boundaries') != [2000, 5000, 10000]:
            raise ValueError('Staged workflow requires explicit 2k, 5k, 10k boundaries')
        if self.plan.get('max_updates', 10000) != 10000:
            raise ValueError('Keep the same 10k scheduler horizon across all phases')
        if stage_target not in self.plan['stage_boundaries']:
            raise ValueError('Choose one absolute stage endpoint: 2000, 5000 or 10000')
        self.stage_target = stage_target
        self.methods = ('D0', 'R0')

    def make_config(self, method, data_sha):
        config = super().make_config(method, data_sha)
        config['stop_after_update'] = self.stage_target
        return config

    def stage_summary(self, method, config, *, required=False):
        folder = self.run_root / method
        path = folder / 'STATUS.json'
        if not path.exists():
            if required:
                raise RuntimeError(f'{method} stage status is missing')
            return None
        summary = read_json(path)
        if summary.get('completed_update', 0) > self.stage_target:
            raise RuntimeError('Cannot execute or report an earlier stage as the current run')
        if (summary.get('completed_update') != self.stage_target or summary.get('status') not in
                ('STAGE_COMPLETE_AWAITING_REVIEW', 'STAGE_COMPLETE_HANDOFF_READY', 'TRAINING_COMPLETE_REFINEMENT_EVALUATION_PENDING')):
            if required:
                raise RuntimeError(f'{method} has not completed this stage')
            return None
        checkpoint = inspect_full_checkpoint(folder / 'checkpoints/latest.pt', method,
                                             self.plan['runtime']['world_size'])
        if checkpoint['completed_update'] != self.stage_target or checkpoint['config_hash'] != scientific_hash(config):
            raise RuntimeError('Stage latest checkpoint or immutable configuration mismatch')
        curve = [json.loads(line) for line in (folder / 'validation_curve.jsonl').read_text().splitlines() if line.strip()]
        history = [json.loads(x) for x in (folder / 'INTERVAL_HISTORY.jsonl').read_text().splitlines() if x.strip()]
        expected = set()
        for i, h in enumerate(history):
            start = h['effective_after_update']
            end = history[i + 1]['effective_after_update'] if i + 1 < len(history) else self.stage_target
            expected.update(u for u in range(start + 1, end + 1) if u % h['val_interval'] == 0)
            if h['stage_endpoint'] <= end:
                expected.add(h['stage_endpoint'])
        if [r.get('update') for r in curve] != sorted(expected):
            raise RuntimeError('Stage must retain every full validation, without duplicates or gaps')
        if any(r.get('count') != 1000 or r.get('source_count') != 250 or r.get('refinement') is not False for r in curve):
            raise RuntimeError('Stage validation is incomplete or changed the protocol')
        best = inspect_full_checkpoint(folder / 'checkpoints/best.pt', method,
                                       self.plan['runtime']['world_size'], require_best=True)
        if best['completed_update'] != summary['best_update']:
            raise RuntimeError('Stage best checkpoint disagrees with full validation selection')
        return {**summary, 'validation_curve': curve, 'latest_checkpoint_sha256': file_sha(folder / 'checkpoints/latest.pt')}

    def run(self):
        lock = (self.reports / 'workflow_LOCK').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('Another controller already owns this workflow')
        signal.signal(signal.SIGTERM, self.stop_handler)
        signal.signal(signal.SIGINT, self.stop_handler)
        try:
            atomic_json(self.reports / f'workflow_PLAN_{self.plan_sha}.json', self.plan, frozen=True)
            atomic_json(self.reports / 'workflow_EXECUTION_PLAN.json', self.plan)
            self.verify_code_manifest()
            data_sha = self.validate_data()
            self.wait_file(self.plan['initial_checkpoint'], 'WAITING_INITIAL_WEIGHTS')
            self.validate_hccl()
            results = {}
            for method in self.methods:
                config = self.make_config(method, data_sha)
                done = self.stage_summary(method, config)
                if done:
                    results[method] = done
                    continue
                folder = Path(config['output_dir'])
                if (folder / 'RUN_CONFIG.json').exists():
                    if not self.resume_existing:
                        raise RuntimeError('Stage continuation requires --resume-existing; never restart optimizer/scheduler')
                    latest = folder / 'checkpoints/latest.pt'
                    checkpoint = inspect_full_checkpoint(latest, method, self.plan['runtime']['world_size'])
                    if checkpoint['config_hash'] != scientific_hash(config):
                        raise RuntimeError('Resume scientific configuration changed')
                    if checkpoint['initial_identity']['sha256'] != file_sha(config['initial_checkpoint']):
                        raise RuntimeError('Resume initialization changed')
                    if checkpoint['completed_update'] >= self.stage_target:
                        raise RuntimeError('Stage checkpoint exists without a matching completed report; inspect before resuming')
                    config['resume_checkpoint'] = str(latest)
                elif self.stage_target != 2000:
                    raise RuntimeError('A later phase cannot start a fresh run')
                path = self.save_config(config)
                self.run_command(f'{method}_to_{self.stage_target}', self.launcher('training.runner',
                    '--config', path, '--factory', 'runtime.factories:training_components'))
                results[method] = self.stage_summary(method, config, required=True)
            report = {'status': 'PAIRED_STAGE_COMPLETE_REVIEW_REQUIRED', 'stage_target': self.stage_target,
                'method_order': list(self.methods), 'results': results, 'test_used': False,
                'next_stage_automatic': False, 'r0_d0init_started': False,
                'scheduler_horizon': 10000, 'resume_from': 'latest full checkpoint, never best'}
            atomic_json(self.reports / f'PAIRED_STAGE_{self.stage_target:06d}.json', report, frozen=True)
            self.update('PAIRED_STAGE_COMPLETE_REVIEW_REQUIRED', stage_target=self.stage_target,
                        methods=list(self.methods), next_stage_automatic=False, r0_d0init_started=False)
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

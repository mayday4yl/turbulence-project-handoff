"""User-approved finite sequence: D0 2k, R0 2k, D0 10k, R0 10k, Init 10k."""
from pathlib import Path
import fcntl
import json
import signal
import traceback

from .controller import atomic_json, read_json, file_sha, inspect_full_checkpoint, scientific_hash
from .staged import StagedWorkflow

SEQUENCE = [{'method': m, 'target': u} for m, u in
            [('D0', 2000), ('R0', 2000), ('D0', 10000), ('R0', 10000), ('R0-D0Init', 10000)]]


class ContinuousWorkflow(StagedWorkflow):
    def __init__(self, plan_path, *, resume_existing=False):
        super().__init__(plan_path, stage_target=2000, resume_existing=resume_existing)
        if self.plan.get('execution_sequence') != SEQUENCE:
            raise ValueError('Execution order differs from the approved finite sequence')

    def run(self):
        lock = (self.reports / 'workflow_LOCK').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('Another controller already owns this workflow')
        signal.signal(signal.SIGTERM, self.stop_handler)
        signal.signal(signal.SIGINT, self.stop_handler)
        ledger_path = self.reports / 'CONTINUOUS_PHASES.json'
        ledger = read_json(ledger_path) if ledger_path.exists() else {'completed': []}
        try:
            atomic_json(self.reports / f'workflow_PLAN_{self.plan_sha}.json', self.plan, frozen=True)
            atomic_json(self.reports / 'workflow_EXECUTION_PLAN.json', self.plan)
            self.verify_code_manifest()
            data_sha = self.validate_data()
            self.wait_file(self.plan['initial_checkpoint'], 'WAITING_INITIAL_WEIGHTS')
            self.validate_hccl()
            for phase_index, phase in enumerate(SEQUENCE):
                method, self.stage_target = phase['method'], phase['target']
                key = f'{method}_to_{self.stage_target}'
                if phase_index < len(ledger['completed']):
                    recorded = ledger['completed'][phase_index]
                    if recorded['phase'] != phase or not Path(recorded['summary_path']).is_file():
                        raise RuntimeError('Completed phase ledger is inconsistent')
                    continue
                config = self.make_config(method, data_sha)
                done = self.stage_summary(method, config)
                if done is None:
                    folder = Path(config['output_dir'])
                    if (folder / 'RUN_CONFIG.json').exists():
                        planned_continuation = phase_index in (2, 3)
                        if not self.resume_existing and not planned_continuation:
                            raise RuntimeError('An interrupted existing phase needs explicit --resume-existing')
                        latest = folder / 'checkpoints/latest.pt'
                        checkpoint = inspect_full_checkpoint(latest, method, self.plan['runtime']['world_size'])
                        if checkpoint['config_hash'] != scientific_hash(config):
                            raise RuntimeError('Continuation changed immutable scientific configuration')
                        if checkpoint['initial_identity']['sha256'] != file_sha(config['initial_checkpoint']):
                            raise RuntimeError('Continuation initialization changed')
                        config['resume_checkpoint'] = str(latest)
                    elif phase_index in (2, 3):
                        raise RuntimeError('Continuation cannot restart D0/R0 from U0')
                    path = self.save_config(config)
                    self.run_command(key, self.launcher('training.runner', '--config', path,
                                                       '--factory', 'runtime.factories:training_components'))
                    done = self.stage_summary(method, config, required=True)
                summary_path = self.reports / f'PHASE_{key}.json'
                atomic_json(summary_path, done, frozen=True)
                ledger['completed'].append({'phase': phase, 'summary_path': str(summary_path),
                                            'summary_sha256': file_sha(summary_path)})
                atomic_json(ledger_path, ledger)
                self.event('phase_complete_continuing_automatically', phase=phase,
                           best_psnr=done['best_psnr'], best_update=done['best_update'])
                if phase_index == 1:
                    paired = {m: read_json(self.reports / f'PHASE_{m}_to_2000.json') for m in ('D0', 'R0')}
                    atomic_json(self.reports / 'PAIRED_STAGE_002000.json', {
                        'status': 'PAIRED_2K_COMPLETE_CONTINUING_AUTOMATICALLY', 'results': paired,
                        'test_used': False, 'next_phase': SEQUENCE[2]})
                    lines = ['# D0 / R0 2k comparison', '', '| Method | Best update | Best PSNR | Latest PSNR | Input gain | Active hours |',
                             '|---|---:|---:|---:|---:|---:|']
                    for m, result in paired.items():
                        latest = result['validation_curve'][-1]
                        lines.append(f"| {m} | {result['best_update']} | {result['best_psnr']:.4f} | {latest['psnr']:.4f} | {latest.get('psnr_gain_vs_input', float('nan')):.4f} | {result['counters']['run_elapsed_seconds']/3600:.2f} |")
                    lines += ['', 'Full Val1000; refinement OFF; test locked. Continue D0 to 10k without an approval pause.', '']
                    (self.reports / 'PAIRED_STAGE_002000.md').write_text('\n'.join(lines), encoding='utf8')
            self.update('ALL_THREE_GROUPS_COMPLETE', per_group_update_cap=10000,
                        final_refinement='PENDING_SEPARATE_CONFIGURATION', test_evaluation='NOT_RUN',
                        budget_extension=False)
            return 0
        except InterruptedError as exc:
            self.update('STOPPED', error=str(exc), resume_requires_explicit_flag=True)
            return 130
        except BaseException as exc:
            self.update('FAILED_REQUIRES_ATTENTION', error=repr(exc), traceback=traceback.format_exc(),
                        automatic_retry=False, method_change=False)
            return 1
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

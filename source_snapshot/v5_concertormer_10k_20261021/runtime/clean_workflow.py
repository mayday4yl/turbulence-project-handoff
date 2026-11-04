"""Finite clean-only D0 -> V5 R0-D0Init execution, each capped at 10k."""
import fcntl
import signal
import traceback
from pathlib import Path
from .controller import Workflow, atomic_json, file_sha, read_json, inspect_full_checkpoint, scientific_hash

SEQUENCE = [{'method': 'D0', 'target': 10000}, {'method': 'R0-D0Init', 'target': 10000}]


class CleanWorkflow(Workflow):
    def __init__(self, plan_path, **kwargs):
        super().__init__(plan_path, **kwargs)
        if self.plan.get('d0_objective') != 'clean_only' or self.plan.get('execution_sequence') != SEQUENCE:
            raise ValueError('Only the approved clean D0 followed by R0-D0Init is permitted')

    def prepare_clean_d0_initialization(self, best):
        output = self.reports / 'R0-D0Init_initial.pt'
        report = self.reports / 'R0-D0Init_mapping.json'
        if not report.exists():
            self.run_command('D0_best_to_V5_neutral_initialization', ['bash', str(self.root/'run_env.sh'),
                'scripts/prepare_clean_transfer.py', '--checkpoint', str(best), '--output', str(output),
                '--report', str(report), '--dataset', self.plan['dataset_root']])
        evidence = read_json(report)
        if (evidence.get('status') != 'PASS' or evidence['parent_sha256'] != file_sha(best)
                or evidence['initial_sha256'] != file_sha(output)):
            raise RuntimeError('D0 to R0 initialization parity/identity check failed')
        return output

    def make_config(self, method, data_sha):
        config = super().make_config(method, data_sha)
        config['stop_after_update'] = 10000
        return config

    def run(self):
        lock = (self.reports/'workflow_LOCK').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, self.stop_handler)
        signal.signal(signal.SIGINT, self.stop_handler)
        try:
            atomic_json(self.reports/'workflow_EXECUTION_PLAN.json', self.plan)
            self.verify_code_manifest()
            data_sha = self.validate_data()
            self.wait_file(self.plan['initial_checkpoint'], 'WAITING_INITIAL_WEIGHTS')
            for phase in SEQUENCE:
                method = phase['method']
                config = self.make_config(method, data_sha)
                done = self.completed_summary(method, expected_config=config)
                if done is None:
                    folder = Path(config['output_dir'])
                    if (folder/'RUN_CONFIG.json').exists():
                        if not self.resume_existing:
                            raise RuntimeError('Interrupted runs require explicit resume, never restart')
                        latest = folder/'checkpoints/latest.pt'
                        state = inspect_full_checkpoint(latest, method, self.plan['runtime']['world_size'])
                        if state['config_hash'] != scientific_hash(config):
                            raise RuntimeError('Resume science/configuration differs')
                        config['resume_checkpoint'] = str(latest)
                    path = self.save_config(config)
                    self.run_command(method, self.launcher('training.runner', '--config', path,
                                     '--factory', 'runtime.factories:training_components'))
                    done = self.completed_summary(method, required=True, expected_config=config)
                atomic_json(self.reports/f'{method}_COMPLETE.json', done)
                self.event('group_complete', method=method, best_psnr=done['best_psnr'])
            self.update('BOTH_AUTHORIZED_GROUPS_COMPLETE', direct_R0='DEFERRED', per_group_cap=10000,
                        test_evaluation='NOT_RUN', final_refinement='PENDING_CONFIGURATION')
            return 0
        except BaseException as exc:
            self.update('FAILED_REQUIRES_ATTENTION', error=repr(exc), traceback=traceback.format_exc(),
                        automatic_retry=False)
            return 1
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

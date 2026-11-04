#!/usr/bin/env python3
"""One explicit HCCL connection-timeout recovery, reusing checked smoke evidence.

No training mathematics or selection rules change. The existing selector still
owns ranking, confirmation, resume, gradient gates, and the global smoke budget.
"""
import copy
import fcntl
from pathlib import Path
import traceback

from run_stage2_smokes import (ROOT, REPORT_NAME, LEDGER_NAME, SelectionDriver,
                              evidence, memory_failure_reason, now, read,
                              require, sha256, write)

ARCHIVE = 'artifacts/STAGE2_RUNTIME_SELECTION_before_hccl_repair_01.json'
FAILED = 'stage2_performance_mb4_01'


def checked(root, record):
    path = Path(record['path']).resolve()
    require(path.is_relative_to(root.resolve()), 'Reuse evidence leaves project')
    require(sha256(path) == record['sha256'], f'Reuse evidence changed: {path}')
    return read(path) if path.suffix == '.json' else path.read_text()


def connection_timeout_only(root, attempt):
    guard = checked(root, attempt['guard_evidence'])
    require(guard.get('status') == 'failed' and guard.get('own_npu_processes_released') is True,
            'Failed attempt resources are not confirmed released')
    require(not guard.get('cleanup') or all('cleanup_error' not in x for x in guard['cleanup']),
            'Failed attempt cleanup is unresolved')
    require(guard.get('reason') == 'Worker exited with return code 1', 'Unexpected guard failure')
    output = root / 'artifacts' / attempt['id']
    errors = [read(p) for p in output.glob('error_rank*.json')]
    require(errors and all('HcclAllreduce' in x.get('error', '')
                          and 'Getting socket times out' in x.get('error', '')
                          and x.get('optimizer_step_attempts_this_invocation') == 0 for x in errors),
            'Only the inspected first-collective socket timeout is eligible')
    log = (root / 'runs' / attempt['id'] / 'worker.log').read_text()
    require(not any(x in log for x in ['FloatingPointError:', 'out of memory', 'AssertionError:']),
            'Worker log contains a non-transport failure')


class RecoveryDriver(SelectionDriver):
    def __init__(self, root, previous):
        initial = Path(previous['initial_ddp_normalization']['path'])
        super().__init__(root, 'hccl02', initial)
        self.previous = previous
        self.report.update(maximum_new_smoke_reservation=15,
                           recovery={'prior_report': evidence(self.root / ARCHIVE),
                                     'failed_attempt': FAILED, 'retry_number': 1, 'retry_limit': 2,
                                     'change': 'HCCL_CONNECT_TIMEOUT=3600; tensor programs unchanged',
                                     'environment_script': evidence(self.root / 'code/run_distributed_env.sh'),
                                     'recovery_source': evidence(Path(__file__).resolve())})

    def verify_prerequisites(self):
        super().verify_prerequisites()
        require(self.previous['status'] == 'blocked', 'Previous run is not the inspected stopped selection')
        checked(self.root, self.previous['driver_source'])
        checked(self.root, self.previous['controller_source'])
        require(self.previous['candidate_configs'] == self.report['candidate_configs'], 'Candidate configs changed')
        require(self.previous['initial_ddp_normalization'] == self.report['initial_ddp_normalization'],
                'Initial DDP evidence changed')
        require('export HCCL_CONNECT_TIMEOUT=3600' in (self.root / 'code/run_distributed_env.sh').read_text(),
                'Connection timeout repair is absent')
        ledger = read(self.root / LEDGER_NAME)
        require(ledger['conservative_consumed_updates'] + 15 <= 60, 'Insufficient remaining smoke budget')
        attempts = self.previous['attempts']
        require(len(attempts) == 7 and attempts[-1]['id'] == FAILED, 'Unexpected prior attempt history')
        connection_timeout_only(self.root, attempts[-1])

    def launch_smoke(self, mode, mb, label, measure):
        old_id = f'stage2_{label}_mb{mb}_01'
        matches = [a for a in self.previous['attempts'] if a['id'] == old_id]
        if not matches:
            return super().launch_smoke(mode, mb, label, measure)
        self.verify_current_code()
        attempt = copy.deepcopy(matches[0])
        ledger = read(self.root / LEDGER_NAME)
        require(next(a for a in ledger['attempts'] if a['id'] == old_id) == attempt['ledger_row'],
                'Prior durable smoke ledger row changed')
        guard = checked(self.root, attempt['guard_evidence'])
        if old_id == FAILED:
            connection_timeout_only(self.root, attempt)
            attempt['repair_action'] = 'Preserved failure; one fresh guarded retry with longer connection wait'
            self.report['attempts'].append(attempt)
            self.persist()
            return super().launch_smoke(mode, mb, label, measure)
        attempt['reused_without_new_optimizer_updates'] = True
        output = self.root / 'artifacts' / old_id
        if attempt['status'] == 'passed':
            checked(self.root, attempt['result_evidence'])
            for record in attempt['rank_log_evidence']:
                checked(self.root, record)
            require(guard['status'] == 'completed' and guard['own_npu_processes_released'] is True,
                    'Prior successful guard did not complete and release')
            result, logs = self.validate_success(output, mode, mb, measure)
            answer = {'passed': True, 'result': result, 'logs': logs, 'attempt': attempt}
        else:
            require(mode == 'memory-smoke' and mb == 8, 'Unexpected failure cannot be reused')
            ranks = [read(p) for p in sorted(output.glob('error_rank*.json'))]
            reason = memory_failure_reason(guard, ranks, (self.root / 'runs' / old_id / 'worker.log').read_text())
            require(reason == attempt.get('candidate_exclusion_reason'), 'OOM exclusion evidence changed')
            require(reason is not None, 'Prior failure is not an explicit qualified OOM')
            answer = {'passed': False, 'reason': reason, 'attempt': attempt}
        self.report['attempts'].append(attempt)
        self.persist()
        return answer


def main():
    with (ROOT / '.stage2_selection_driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked()


def run_locked():
    require(not (ROOT / 'STOP_REQUESTED').exists(), 'Project stop marker is present')
    require(not (ROOT / ARCHIVE).exists(), 'This explicit recovery has already been prepared')
    previous = read(ROOT / REPORT_NAME)
    write(ROOT / ARCHIVE, previous)
    driver = RecoveryDriver(ROOT, previous)
    try:
        driver.run()
        repair_path = ROOT / 'artifacts/STAGE2_HCCL_TIMEOUT_REPAIR.json'
        repair = read(repair_path)
        repair.update(status='verified_by_completed_runtime_selection', verified_utc=now(),
                      retest=evidence(ROOT / 'artifacts/stage2_performance_mb4_hccl02/results.json'),
                      runtime_selection=evidence(ROOT / REPORT_NAME))
        write(repair_path, repair)
    except BaseException as exc:
        driver.report.update(status='blocked', error=repr(exc), traceback=traceback.format_exc(),
                             completed_utc=now(), formal_training_started=False)
        driver.persist()
        raise


if __name__ == '__main__':
    main()

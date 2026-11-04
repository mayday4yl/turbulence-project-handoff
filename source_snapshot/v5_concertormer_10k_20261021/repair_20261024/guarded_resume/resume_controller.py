"""Resume through existing controller protections with an explicit recovery gate.

Never launches from the proposed policy alone. Root must provide a separate
accepted evidence document after the current checkpoint's diagnosis passes.
"""
import argparse
import json
from pathlib import Path
import time

from optimizations.b_utilization_20261023.resume_controller import BResumeWorkflow
from alignment_runtime.controller import read_json, file_sha, atomic_json, json_rows


class GuardAbortNoRetry(FloatingPointError):
    """Intentionally not RuntimeError: inherited transient retry must not catch it."""


def guard_offsets(folder):
    return {path.name: path.stat().st_size for path in Path(folder).glob('guard_rank*.jsonl')}


def new_guard_aborts(folder, offsets):
    found = []
    for path in Path(folder).glob('guard_rank*.jsonl'):
        with path.open('rb') as stream:
            stream.seek(offsets.get(path.name, 0))
            for line in stream:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if row.get('event') == 'gradient_guard_abort':
                    found.append({'file': str(path), 'record': row})
    return found


def guard_abort_preempts_retry(folder, offsets, receipt_path, *, stage, original_error):
    found = new_guard_aborts(folder, offsets)
    if not found:
        return
    atomic_json(receipt_path, {'status': 'GUARD_ABORT_NO_AUTOMATIC_RETRY',
        'stage': stage, 'unix_time': time.time(), 'original_error': str(original_error),
        'new_guard_aborts': found, 'automatic_retry_allowed': False,
        'basis': 'Only records appended during this child invocation; prior timeout text cannot override guard failure'})
    raise GuardAbortNoRetry('gradient guard abort: automatic transient retry prohibited; '
                            'see ' + str(receipt_path)) from original_error


class GuardedResumeWorkflow(BResumeWorkflow):
    def __init__(self, plan_path, authorization_path, policy_path, acceptance_path):
        super().__init__(plan_path, authorization_path)
        self.policy_path = Path(policy_path).resolve()
        self.acceptance_path = Path(acceptance_path).resolve()
        self.policy_sha = file_sha(self.policy_path)
        self.acceptance_sha = file_sha(self.acceptance_path)

    def verify_frozen(self):
        super().verify_frozen()
        folder = self.root / 'repair_20261024/guarded_resume'
        for name, digest in read_json(folder / 'MANIFEST.json')['files'].items():
            if file_sha(folder / name) != digest:
                raise RuntimeError('Gradient guard extension identity mismatch: ' + name)
        if file_sha(self.policy_path) != self.policy_sha or file_sha(self.acceptance_path) != self.acceptance_sha:
            raise RuntimeError('Guard policy or recovery acceptance changed')
        policy, acceptance = read_json(self.policy_path), read_json(self.acceptance_path)
        if (policy.get('status') != 'ACTIVE_PAUSE_ONLY_MONITOR' or
                acceptance.get('status') != 'ACCEPTED_FOR_GUARDED_RESUME' or
                acceptance.get('guard_policy_sha256') != self.policy_sha or
                acceptance.get('extension_manifest_sha256') != file_sha(folder / 'MANIFEST.json') or
                acceptance.get('method_changed') is not False or
                acceptance.get('optimizer_scheduler_rng_sampler_reset') is not False or
                not acceptance.get('recovery_evidence')):
            raise RuntimeError('Cannot resume without separately reviewed and bound recovery evidence')
        for item in acceptance['recovery_evidence']:
            if file_sha(Path(item['path'])) != item['sha256']:
                raise RuntimeError('Recovery evidence changed: ' + item['path'])
        if file_sha(Path(acceptance['resume_anchor_checkpoint'])) != acceptance['resume_anchor_checkpoint_sha256']:
            raise RuntimeError('Accepted recovery anchor checkpoint changed')

    def distributed(self, module, *args):
        if module == 'alignment_training.runner':
            config = read_json(args[args.index('--config') + 1])
            if config['method'] == 'B':
                module = 'repair_20261024.guarded_resume.guarded_runner'
                args = (*args, '--guard-policy', str(self.policy_path),
                        '--resume-acceptance', str(self.acceptance_path))
                self.event('pause_only_gradient_guard_selected',
                    policy_path=str(self.policy_path), policy_sha256=self.policy_sha,
                    recovery_acceptance=str(self.acceptance_path),
                    recovery_acceptance_sha256=self.acceptance_sha,
                    loss_or_gradient_modified=False, samples_skipped=False)
        return super().distributed(module, *args)

    def run_commands(self, stage, commands, *, npu=True, monitor_method=None):
        folder = self.run_root / 'B/gradient_guard'
        offsets = guard_offsets(folder) if monitor_method == 'B' else None
        try:
            return super().run_commands(stage, commands, npu=npu, monitor_method=monitor_method)
        except RuntimeError as exc:
            if offsets is not None:
                receipt = self.reports / 'guarded_resume_20261024' / f'GUARD_ABORT_{time.time_ns()}.json'
                guard_abort_preempts_retry(folder, offsets, receipt,
                                          stage=stage, original_error=exc)
            # Ordinary I/O/device failures with no newly appended guard abort
            # continue through the original bounded transient recovery policy.
            raise

    def archive_uncommitted_tail(self, stage, checkpoint, reason):
        # Capture original attempted cost *before* the base class archives and
        # truncates the canonical logs. This cost is additional to the resumed
        # checkpoint's counters; no numeric budget is increased.
        run = self.run_root / stage
        status = read_json(run / 'STATUS.json') if (run / 'STATUS.json').exists() else {}
        rows = json_rows(run / 'metrics.jsonl')
        candidates = [status, *[row for row in rows if row.get('event') == 'train_log']]
        latest = max(candidates, key=lambda row: row.get('update', row.get('completed_update', 0)))
        cutoff = checkpoint['completed_update']
        observed = latest.get('update', latest.get('completed_update', 0))
        after, before = latest.get('counters', {}), checkpoint.get('counters', {})
        for event in json_rows(run / 'checkpoint_events.jsonl'):
            if (event.get('completed_update') == cutoff and
                    event.get('config_hash') == checkpoint['config_hash']):
                before = event['counters_after_save']
        cost = {'stage': stage, 'recorded_unix_time': time.time(),
            'resume_checkpoint_sha256': checkpoint['sha256'], 'durable_update': cutoff,
            'last_recorded_attempt_update': observed,
            'discarded_attempted_updates': max(0, observed - cutoff),
            'discarded_attempted_sample_exposures': max(0, observed - cutoff) * 48,
            'discarded_attempted_network_forward_exposures': max(0, observed - cutoff) * 48 * (4 if stage == 'B' else 1),
            'additional_recorded_training_seconds': max(0., after.get('training_seconds', 0.) - before.get('training_seconds', 0.)),
            'additional_recorded_active_run_seconds': max(0., after.get('run_elapsed_seconds', 0.) - before.get('run_elapsed_seconds', 0.)),
            'in_flight_partial_update_cost': 'unknown; not claimed as zero',
            'counter_cutoff': 'matching checkpoint event counters_after_save when available',
            'budget_extended': False, 'count_as_retained_optimizer_updates': False,
            'reason': reason, 'prior_status': status}
        audit = self.reports / 'guarded_resume_20261024'
        audit.mkdir(parents=True, exist_ok=True)
        target = audit / f'UNCOMMITTED_COST_{stage}_{time.time_ns()}.json'
        atomic_json(target, cost)
        super().archive_uncommitted_tail(stage, checkpoint, reason)
        self.event('discarded_attempt_cost_preserved', cost_receipt=str(target),
                   discarded_attempted_updates=cost['discarded_attempted_updates'])

    def final_report(self):
        result = super().final_report()
        receipts = sorted((self.reports / 'guarded_resume_20261024').glob('UNCOMMITTED_COST_*.json'))
        result['additional_discarded_attempt_cost_receipts'] = [str(p) for p in receipts]
        result['additional_recorded_discarded_attempt_training_seconds'] = sum(
            read_json(p)['additional_recorded_training_seconds'] for p in receipts)
        # Overall workflow wall already spans interruptions; do not add these
        # seconds to it a second time. Detail accounting is reported separately.
        atomic_json(self.reports / 'FINAL_RESULTS.json', result)
        with (self.reports / 'FINAL_REPORT_ZH.md').open('a') as stream:
            stream.write('\n异常暂停后从完整 checkpoint 恢复；未提交训练尾部的日志和额外计算成本单独保留，'
                         '不计作保留模型的 optimizer updates，也不重复计入总墙钟。见 guarded_resume_20261024。\n')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--guard-policy', required=True)
    parser.add_argument('--resume-acceptance', required=True)
    args = parser.parse_args()
    raise SystemExit(GuardedResumeWorkflow(args.plan, args.authorization,
        args.guard_policy, args.resume_acceptance).run())

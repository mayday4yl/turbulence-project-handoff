"""Run only the user-confirmed D0 then R0 phase; always stop for paired review."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.authorization import require_formal_training_confirmation, require_stage_confirmation
from runtime.controller import file_sha, read_json, validate_plan
from runtime.staged import StagedWorkflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage-target', type=int, required=True, choices=(2000, 5000, 10000))
    parser.add_argument('--resume-existing', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = validate_plan(read_json(args.plan))
    if plan.get('stage_boundaries') != [2000, 5000, 10000] or plan.get('max_updates', 10000) != 10000:
        raise ValueError('The staged plan must keep a 10k scheduler with 2k/5k/10k pause boundaries')
    if args.validate_only:
        print('STAGED_PLAN_SCHEMA_VALID; accelerator_jobs_started=0')
        return 0
    confirmation = require_formal_training_confirmation()
    require_stage_confirmation(confirmation, args.stage_target, ['D0', 'R0'])
    if confirmation['approved_execution_plan_sha256'] != file_sha(args.plan):
        raise RuntimeError('The selected plan differs from the plan confirmed by the user')
    return StagedWorkflow(args.plan, stage_target=args.stage_target,
                          resume_existing=args.resume_existing).run()


if __name__ == '__main__':
    raise SystemExit(main())

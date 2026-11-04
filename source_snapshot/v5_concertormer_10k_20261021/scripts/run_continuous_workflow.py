"""Start only the explicitly approved finite V5 continuous execution plan."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.authorization import require_formal_training_confirmation
from runtime.controller import read_json, file_sha, validate_plan
from runtime.continuous import ContinuousWorkflow, SEQUENCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--resume-existing', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    plan = validate_plan(read_json(args.plan))
    if plan.get('execution_sequence') != SEQUENCE or plan.get('max_updates') != 10000:
        raise RuntimeError('Plan must match the approved execution order and 10k cap')
    if args.validate_only:
        print('CONTINUOUS_PLAN_SCHEMA_VALID; accelerator_jobs_started=0')
        return 0
    approval = require_formal_training_confirmation()
    if approval.get('approved_execution_sequence') != SEQUENCE or approval['approved_execution_plan_sha256'] != file_sha(args.plan):
        raise RuntimeError('This exact continuous execution plan has not been approved')
    return ContinuousWorkflow(args.plan, resume_existing=args.resume_existing).run()


if __name__ == '__main__':
    raise SystemExit(main())

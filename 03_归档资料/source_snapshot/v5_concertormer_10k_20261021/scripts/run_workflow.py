"""Run the authorized workflow from a frozen plan, or validate its schema only."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.controller import Workflow, read_json, validate_plan
from runtime.authorization import require_formal_training_confirmation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(ROOT / 'protocol/EXECUTION_PLAN.json'))
    parser.add_argument('--resume-existing', action='store_true', help='Explicitly resume incomplete runs from their verified full latest checkpoint; never retry a failure within one invocation')
    parser.add_argument('--validate-only', action='store_true', help='Pure schema check; does not initialize an accelerator, wait, or launch any process')
    args = parser.parse_args()
    if args.validate_only:
        validate_plan(read_json(args.plan))
        print(json.dumps({'status': 'PLAN_SCHEMA_VALID', 'plan': args.plan, 'accelerator_jobs_started': 0}))
        return 0
    confirmation = require_formal_training_confirmation()
    from runtime.controller import file_sha
    if confirmation['approved_execution_plan_sha256'] != file_sha(args.plan):
        raise RuntimeError('The execution plan differs from the settings confirmed by the user')
    return Workflow(args.plan, resume_existing=args.resume_existing).run()


if __name__ == '__main__':
    raise SystemExit(main())

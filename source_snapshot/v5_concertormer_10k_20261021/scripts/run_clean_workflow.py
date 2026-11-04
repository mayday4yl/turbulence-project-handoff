import argparse
from runtime.authorization import require_formal_training_confirmation
from runtime.controller import file_sha, read_json, validate_plan
from runtime.clean_workflow import CleanWorkflow, SEQUENCE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--resume-existing', action='store_true')
    a = p.parse_args()
    plan = validate_plan(read_json(a.plan))
    authorization = require_formal_training_confirmation()
    if (plan.get('execution_sequence') != SEQUENCE or authorization.get('approved_execution_sequence') != SEQUENCE
            or authorization['approved_execution_plan_sha256'] != file_sha(a.plan)):
        raise RuntimeError('Exact clean-only execution sequence is not authorized')
    return CleanWorkflow(a.plan, resume_existing=a.resume_existing).run()


if __name__ == '__main__':
    raise SystemExit(main())

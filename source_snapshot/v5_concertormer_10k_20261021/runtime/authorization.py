"""Honor the user's explicit hold on formal training after runtime selection."""
import json
from pathlib import Path

AUTHORIZATION = Path(__file__).resolve().parents[1] / "protocol/FORMAL_TRAINING_AUTHORIZATION.json"


def require_formal_training_confirmation(path=AUTHORIZATION):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("Formal training requires the user's confirmation of final settings; authorization record missing")
    value = json.loads(path.read_text())
    if (value.get("formal_training_approved") is not True or
            not value.get("approval_user_message") or
            not value.get("approved_execution_plan_sha256")):
        raise RuntimeError("Formal training is held: user must confirm the final training settings before launch")
    return value


def require_stage_confirmation(value, target, methods, *, worker=False):
    """A 2k approval does not authorize a later 5k/10k phase or D0Init."""
    if value.get('approved_execution_sequence'):
        allowed = {(phase['method'], phase['target']) for phase in value['approved_execution_sequence']}
        if target not in (2000, 5000, 10000) or any((m, target) not in allowed for m in methods):
            raise RuntimeError('This method/endpoint is outside the user-approved execution sequence')
        return
    if target not in (2000, 5000, 10000) or value.get('approved_stage_target') != target:
        raise RuntimeError('This exact stage endpoint has not been confirmed by the user')
    approved = value.get('approved_stage_methods', [])
    if (not worker and approved != list(methods)) or (worker and any(m not in approved for m in methods)):
        raise RuntimeError('These stage methods have not been confirmed by the user')

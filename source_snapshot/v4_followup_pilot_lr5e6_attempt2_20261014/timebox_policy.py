"""Pure time-window construction for the user-authorized new attempt."""
from datetime import datetime, timezone, timedelta

def make_timebox(config, offset, start):
    policy=config['timebox_policy']
    assert policy['total_seconds']==21600
    assert 0<policy['training_seconds']<policy['gpu_seconds']<policy['total_seconds']
    local=lambda seconds:datetime.fromtimestamp(start+seconds-offset,timezone(timedelta(hours=8))).isoformat()
    return {'first_gpu_gate_start_server':start,'server_minus_local_seconds':offset,
            'training_deadline_server':start+policy['training_seconds'],
            'gpu_deadline_server':start+policy['gpu_seconds'],
            'final_deadline_server':start+policy['total_seconds'],
            'local_actual_start':local(0),'local_training_stop':local(policy['training_seconds']),
            'local_gpu_stop':local(policy['gpu_seconds']),'local_final_deadline':local(policy['total_seconds']),
            'user_authorized_new_window':True,'original_timebox_unchanged':True}

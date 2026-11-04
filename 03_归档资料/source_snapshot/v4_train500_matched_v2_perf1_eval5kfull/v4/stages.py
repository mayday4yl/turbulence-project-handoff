"""Execution-only common milestones and explicit user stop controls."""
import json,time
from pathlib import Path
from .runtime import save_json,sha256

NODES=(5000,7500,10000)
GROUPS=('D0','R0','R08','L08')

def read_control(path):
    value=json.loads(Path(path).read_text())
    if value.get('stop_at_common_update') not in (None,5000,7500,10000):raise ValueError('Invalid stop node')
    if value.get('authorized_default_endpoint')!=10000:raise ValueError('Maximum authorized budget changed')
    return value

def late_stop(control,stage_root):
    target=control['stop_at_common_update']
    return bool(control['stop_at_next_boundary'] or (target is not None and
                (Path(stage_root)/f'U{target:05d}'/'RELEASE.json').exists()))

class StageBarrier:
    def __init__(self,root,group,control_path,pause_path):
        self.root=Path(root);self.group=group;self.control_path=Path(control_path);self.pause_path=Path(pause_path)
    def wait(self,update,checkpoint_path,evaluation_commit,mutex):
        assert update in NODES
        # Release evaluation first, then acknowledge a parked peer. No eval lease held here.
        mutex.park(update)
        directory=self.root/f'U{update:05d}';directory.mkdir(parents=True,exist_ok=True)
        save_json(directory/f'{self.group}.READY.json',{'group':self.group,'update':update,'exposures':32*update,
                  'checkpoint_path':str(checkpoint_path),'checkpoint_sha256':sha256(checkpoint_path),
                  'evaluation_commit':str(evaluation_commit),'evaluation_commit_sha256':sha256(evaluation_commit),
                  'ready_at':time.time()})
        while not (directory/'RELEASE.json').exists():
            if self.pause_path.exists():raise RuntimeError('Global pause during stage wait')
            if late_stop(read_control(self.control_path),self.root):return 'late_stop'
            time.sleep(.5)
        release=json.loads((directory/'RELEASE.json').read_text())
        assert release['update']==update and release['all_four_complete']
        manifest=Path(release['delivery_manifest'])
        if sha256(manifest)!=release['delivery_manifest_sha256']:raise ValueError('Stage delivery changed after release')
        # A stop can be requested concurrently with release; never step past that boundary.
        c=read_control(self.control_path)
        if c['stop_at_next_boundary'] or c['stop_at_common_update']==update:return 'stop'
        return release['action']


def verify_budget_scope(root):
    root=Path(root)
    spec=json.loads((root/'protocol/TRAINING_PLAN_SPEC.json').read_text())
    auth=json.loads((root/'protocol/AUTHORIZATION.json').read_text())
    expected={g:10000 for g in GROUPS}
    if spec['budget']['stage1_updates_per_group']!=expected:
        raise RuntimeError('BUDGET_CONFLICT: original approved scientific plan is not four-arm10000; STOP_AND_REPORT')
    if auth['updates_per_method']!=10000 or set(auth['methods'])!=set(GROUPS):
        raise RuntimeError('BUDGET_CONFLICT: authorization differs; STOP_AND_REPORT')
    frozen_path=root/'protocol/FROZEN.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        if frozen.get('updates_per_group')!=expected or frozen.get('updates')!=10000:
            raise RuntimeError('BUDGET_CONFLICT: existing runtime freeze differs; supplement cannot override it')
    return expected

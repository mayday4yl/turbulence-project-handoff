"""Durable CPU interface acceptance; every AdamW update is prohibited."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from v031.runtime import save_json,sha256
from v031.scheduler import lr_for_update
root=Path(__file__).resolve().parents[1]
suite=unittest.defaultTestLoader.discover(str(root/'tests'))
stream=io.StringIO()
with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('Stage1 prohibits optimizer.step')):
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
log=stream.getvalue();print(log)
(root/'artifacts/cpu_acceptance.log').write_text(log)
save_json(root/'artifacts/cpu_acceptance.json',{
 'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
 'restoration_optimizer_updates':0,'adamw_step_guard':'raises unconditionally',
 'LR':{str(n):lr_for_update(n) for n in [0,1,50,500,1000,1001,1500]},
 'checkpoint_resume_scope':'CPU synthetic moments and virtual counters at 500/1000; exact state and LR replay to 2000, no training updates',
 'test_sha256':{p.name:sha256(p) for p in (root/'tests').glob('test_*.py')},
 'code_sha256':{p.name:sha256(p) for p in (root/'code/v031').glob('*.py')}})
raise SystemExit(0 if result.wasSuccessful() else 1)

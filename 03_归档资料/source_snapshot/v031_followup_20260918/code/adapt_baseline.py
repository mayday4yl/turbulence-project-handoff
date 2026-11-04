"""Reproducible, isolated adaptation; never writes the old experiment."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'shared/baseline_code'
DST=ROOT/'code'
(DST/'v031').mkdir(exist_ok=True)
for p in (SRC/'v031').glob('*.py'):
 shutil.copyfile(p,DST/'v031'/p.name)
def replace(text,old,new,count=1):
 actual=text.count(old)
 if actual!=count: raise RuntimeError(f'Expected {count}, found {actual}: {old[:90]}')
 return text.replace(old,new)
t=(SRC/'v031/training.py').read_text()
t=replace(t,'from .core import RolloutModule, ddp_loss_scale, severity_path','from .core import ddp_loss_scale, severity_path\nfrom .followup import RolloutModule')
t=replace(t,'FORMAT = "v031_distributed_update_boundary_v1"','FORMAT = "v031_followup_raw_boundary_v1"')
a=t.index('class FP32EMA:');b=t.index('def empty_counts()',a)
t=t[:a]+t[b:]
t=replace(t,'def empty_counts() -> dict[str, Any]:\n    return {"paths": 0,','def empty_counts(task="T2_FEEDBACK_R") -> dict[str, Any]:\n    return {"task": task, "LR_simulator_renders": 0, "transition_simulator_renders": 0, "training_refinement_iterations": 0, "paths": 0,')
t=replace(t,'        path = severity_path(initial)','        path = (initial,) if counts["task"] == "T1_DIRECT" else severity_path(initial)\n        counts["LR_simulator_renders"] += 0 if counts["task"] == "T1_DIRECT" else len(path)')
t=replace(t,'self.counts = empty_counts()','self.task = network.followup_task\n        self.counts = empty_counts(self.task)')
t=replace(t,'        self.ema = FP32EMA(network)','        self.ema_enabled = False')
t=replace(t,'        if self.poisoned or not self.at_boundary:','        if self.completed_updates >= 1000:\n            raise RuntimeError("Followup update limit 1000 reached")\n        if self.poisoned or not self.at_boundary:')
t=replace(t,'        self.ema.update(self.network)\n','')
t=replace(t,'update_counts = empty_counts()','update_counts = empty_counts(self.task)')
t=replace(t,'1.0 / (GLOBAL_BATCH * g)','1.0 / (GLOBAL_BATCH * (1 if self.task == "T1_DIRECT" else g))')
t=replace(t,'0.2 / (GLOBAL_BATCH * g)','(0.0 if self.task == "T1_DIRECT" else 0.2 / (GLOBAL_BATCH * g))')
t=replace(t,'1.0 / (GLOBAL_BATCH * int(record["initial_G"]))','1.0 / (GLOBAL_BATCH * (1 if self.task == "T1_DIRECT" else int(record["initial_G"])))')
t=replace(t,'"lambda_R": 0.2} for record','"lambda_R": (0.0 if self.task == "T1_DIRECT" else 0.2)} for record')
t=replace(t,'"ema": _cpu_snapshot(engine.ema.state_dict()),','"ema_enabled": False, "task": engine.task,')
t=replace(t,'payload["ema"]','None',count=4)
t=replace(t,'payload["completed_updates"] != n or None["updates"] != n','payload["completed_updates"] != n or payload.get("ema_enabled") is not False')
t=replace(t,'"microbatch": engine.microbatch, "effective_batch": GLOBAL_BATCH,\n                "formal_training"','"microbatch": engine.microbatch, "effective_batch": GLOBAL_BATCH,\n                "task": engine.task, "ema_enabled": False, "formal_training"')
t=replace(t,'expected_counts = empty_counts()','expected_counts = empty_counts(engine.task)')
t=replace(t,'        engine.ema.load_state_dict(None)\n','') if 'engine.ema.load_state_dict(None)' in t else replace(t,'        engine.ema.load_state_dict(payload["ema"])\n','')
t=replace(t,'engine.ema.state_dict())','None)')
(DST/'v031/training.py').write_text(t)
w=(SRC/'train_worker.py').read_text()
w=replace(w,'"clip_grad_norm": 1.0, "ema_decay": 0.996,','"clip_grad_norm": 1.0, "ema_enabled": False,')
w=replace(w,'             "ema_buffer_rule": "copy_after_each_update", "precision": "fp32",','             "precision": "fp32",')
w=replace(w,'config.get("lambda_R") != 0.2','config.get("lambda_R") != (0.0 if config["task"] == "T1_DIRECT" else 0.2)')
w=replace(w,'config.get("mode") != "AUTO_TO_1000"','config.get("mode") != "FOLLOWUP_TO_1000"')
w=replace(w,'["stage1_passed", "training_config_frozen", "inference_policy_frozen"]','["targeted_acceptance_passed", "training_config_frozen"]')
w=replace(w,'if calls > 60:','if calls > 30:')
a=w.index('    network = DualHeadUNet(**config["network"]).to(device)');b=w.index('    return TrainingEngine(',a)
w=w[:a]+'''    from v031.followup import CleanOnly
    network = DualHeadUNet(**config["network"])
    if state_hash(network) != config["expected_full_initial_network_sha256"]:
        raise ValueError("Full common constructor initialization mismatch")
    shared = torch.load(config["initial_state_path"], map_location="cpu", weights_only=True)
    network.load_state_dict(shared, strict=True)
    if state_hash(network) != config["expected_full_initial_network_sha256"]:
        raise ValueError("Shared saved initialization mismatch")
    if config["task"] == "T1_DIRECT":
        network = CleanOnly(network)
        simulator = torch.nn.Identity().to(device).eval().requires_grad_(False)
    else:
        simulator = PiRNP2SSimulator(config["asset_dir"], **config["physics"], device=device)
        simulator.eval().requires_grad_(False)
    network.followup_task = config["task"]
    network = network.to(device)
'''+w[b:]
w=replace(w,'            "ema": _cpu_snapshot(engine.ema.state_dict()),','            "ema_enabled": False,')
w=replace(w,'"ema_allocated": True','"ema_enabled": False')
w=w.replace('AdamW/EMA','AdamW').replace('optimizer, EMA','optimizer')
(DST/'train_worker.py').write_text(w)
for name in ('run_distributed_env.sh','run_env.sh','guard_distributed.py','guard.py'):
 t=(SRC/name).read_text().replace('/data2/yanglin/v031_overfit100_g4321_n4','/data2/yanglin/v031_followup_20260918')
 (DST/name).write_text(t)
print('Adapted training and checkpoint only in',DST)

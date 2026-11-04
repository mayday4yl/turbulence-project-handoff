"""CPU controller tests using tiny checkpoint fixtures; no device operations."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest import mock

import torch

from alignment_runtime.controller import (AlignmentWorkflow, inspect_checkpoint,
    json_rows, transient_failure, validate_plan)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def checkpoint(path, method="A", update=250, *, best_update=None, best_psnr=21.):
    cap, warmup, objective = {"D0": (10000, 300, "clean_only"),
        "A": (6000, 300, "precomputed_current_g"), "B": (4000, 100, "recursive_alignment")}[method]
    if best_update is None and update:
        best_update = update
    state = {"schema": "v5_full_training_checkpoint_v1" if method == "D0" else "v5_alignment_training_checkpoint_v1",
        "method": method, "model": {"toy": torch.tensor(.1)}, "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_update": update, "max_updates": cap, "warmup_updates": warmup},
        "sampler": {"completed_update": update}, "rank_states": [{"rng": {"rank": rank}} for rank in range(6)],
        "counters": {"optimizer_updates": update, "run_elapsed_seconds": 1.},
        "config": {"max_updates": cap, "warmup_updates": warmup, "objective": objective},
        "config_hash": "test_config_hash", "initial_identity": {"sha256": "test_initial"},
        "completed_update": update, "world_size": 6, "best_update": best_update,
        "best_psnr": best_psnr if update else -float("inf")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def workflow(folder):
    instance = object.__new__(AlignmentWorkflow)
    instance.root = Path(folder)
    instance.reports = Path(folder) / "reports"
    instance.run_root = Path(folder) / "runs"
    instance.reports.mkdir(parents=True)
    instance.run_root.mkdir()
    instance.children = []
    instance.stop_requested = False
    instance.plan = {"max_transient_retries": 0, "master_port": 29867,
        "D0_scientific_config_hash": "test_config_hash",
        "stages": {"A": {"max_updates": 6000}, "B": {"max_updates": 4000}}}
    return instance


class ControllerTests(unittest.TestCase):
    def test_plan_rejects_budget_or_semantic_changes(self):
        plan = json.loads((ROOT / "protocol/ALIGNMENT_A6K_B4K_20261022.json").read_text())
        validate_plan(plan)
        for stage, field, value in (("A", "max_updates", 10000), ("B", "warmup_updates", 300),
                                    ("A", "objective", "recursive_alignment")):
            wrong = deepcopy(plan)
            wrong["stages"][stage][field] = value
            with self.assertRaises(ValueError):
                validate_plan(wrong)

    def test_checkpoint_enforces_stage_scheduler_and_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            for method, cap in (("D0", 10000), ("A", 6000), ("B", 4000)):
                valid = checkpoint(path, method, cap)
                self.assertEqual(inspect_checkpoint(path, method)["completed_update"], cap)
                for field, value in (("max_updates", cap + 1), ("warmup_updates", 99)):
                    bad = deepcopy(valid)
                    bad["scheduler"][field] = value
                    torch.save(bad, path)
                    with self.assertRaises(RuntimeError):
                        inspect_checkpoint(path, method)
                bad = deepcopy(valid)
                bad["completed_update"] = cap + 1
                bad["scheduler"]["last_update"] = cap + 1
                bad["sampler"]["completed_update"] = cap + 1
                torch.save(bad, path)
                with self.assertRaises(RuntimeError):
                    inspect_checkpoint(path, method)

    def test_u0_recovery_is_json_safe_and_not_a_best(self):
        with tempfile.TemporaryDirectory() as folder:
            control = workflow(folder)
            run = control.run_root / "A"
            path = run / "checkpoints/latest.pt"
            checkpoint(path, update=0)
            evidence = inspect_checkpoint(path, "A")
            json.dumps(evidence, allow_nan=False)
            self.assertEqual(evidence["completed_update"], 0)
            with self.assertRaises(RuntimeError):
                inspect_checkpoint(path, "A", best=True)
            write_json(run / "STATUS.json", {"status": "FAILED", "completed_update": 3})
            write_rows(run / "metrics.jsonl", [{"event": "training_started", "completed_update": 0}])
            control.archive_uncommitted_tail("A", evidence, "Input/output error before U250")
            recovery = list((control.reports / "recovery/A").glob("*/RECOVERY.json"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(json.loads(recovery[0].read_text())["committed_update"], 0)
            self.assertEqual(json_rows(control.reports / "workflow_EVENTS.jsonl")[-1]["checkpoint"]["completed_update"], 0)

    def test_uncommitted_curves_are_archived_before_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            control = workflow(folder)
            run = control.run_root / "A"
            checkpoint(run / "checkpoints/latest.pt", update=250)
            for filename in ("metrics.jsonl", "validation_curve.jsonl", "val_a_curve.jsonl"):
                write_rows(run / filename, [{"update": 250, "psnr": 20.}, {"update": 500, "psnr": 21.}])
            write_rows(run / "checkpoint_events.jsonl", [{"completed_update": 0}, {"completed_update": 250}])
            write_json(run / "STATUS.json", {"status": "FAILED", "completed_update": 500})
            for directory, filename in (("validation", "update_000500_refinement_off.json"),
                                         ("validation_a", "update_000500.json")):
                write_json(run / directory / filename, {"failed_attempt_per_image_evidence": True})
            control.archive_uncommitted_tail("A", inspect_checkpoint(run / "checkpoints/latest.pt", "A"), "I/O failure")
            snap = next((control.reports / "recovery/A").iterdir())
            for filename in ("metrics.jsonl", "validation_curve.jsonl", "val_a_curve.jsonl"):
                self.assertEqual([r["update"] for r in json_rows(run / filename)], [250])
                self.assertEqual([r["update"] for r in json_rows(snap / filename)], [250, 500])
            for directory, filename in (("validation", "update_000500_refinement_off.json"),
                                         ("validation_a", "update_000500.json")):
                self.assertFalse((run / directory / filename).exists())
                self.assertTrue(json.loads((snap / directory / filename).read_text())["failed_attempt_per_image_evidence"])

    def test_restart_archives_failure_before_first_train_log(self):
        with tempfile.TemporaryDirectory() as folder:
            control = workflow(folder)
            run = control.run_root / "A"
            checkpoint(run / "checkpoints/latest.pt", update=0)
            write_json(run / "RUN_CONFIG.json", {})
            write_json(run / "STATUS.json", {"status": "FAILED", "completed_update": 3})
            write_rows(run / "metrics.jsonl", [{"event": "training_started", "completed_update": 0}])
            control.config = mock.Mock(return_value=(Path(folder) / "config.json", SimpleNamespace()))
            control.readiness = mock.Mock()
            control.progress_snapshot = mock.Mock()
            def finish(*args, **kwargs):
                checkpoint(run / "checkpoints/latest.pt", update=2000)
                write_json(run / "STATUS.json", {"status": "STAGE_COMPLETE_HANDOFF_READY", "completed_update": 2000})
            control.run_commands = mock.Mock(side_effect=finish)
            control.train_segment("A", Path(folder), Path(folder) / "initial.pt", Path(folder) / "init.json", 2000)
            receipts = list((control.reports / "recovery/A").glob("*/RECOVERY.json"))
            self.assertEqual(len(receipts), 1, "Failed U3 STATUS must survive recovery from U0")
            failed_status = json.loads((receipts[0].parent / "STATUS.json").read_text())
            self.assertEqual(failed_status["completed_update"], 3)
            self.assertEqual(control.config.call_args.kwargs["resume"], run / "checkpoints/latest.pt")

    def test_selection_uses_original_val_curve_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            control = workflow(folder)
            run = control.run_root / "A"
            checkpoint(run / "checkpoints/best.pt", update=5750, best_psnr=22.)
            checkpoint(run / "checkpoints/latest.pt", update=6000, best_update=5750, best_psnr=22.)
            write_json(run / "STATUS.json", {"status": "TRAINING_COMPLETE_STAGE_HANDOFF_READY", "method": "A",
                "completed_update": 6000, "best_update": 5750, "best_psnr": 22., "test_used": False})
            write_json(run / "RUN_CONFIG.json", {})
            write_rows(run / "validation_curve.jsonl", [{"update": update,
                "psnr": 22. if update == 5750 else 21.9, "count": 1000,
                "method": "A", "source_count": 250, "refinement": False, "inference_nfe": 4}
                for update in range(250, 6001, 250)])
            write_rows(run / "val_a_curve.jsonl", [{"update": 6000, "psnr": 99., "count": 1000,
                "refinement": False, "inference_nfe": 1}])
            for name in ("metrics.jsonl", "checkpoint_events.jsonl", "INTERVAL_HISTORY.jsonl"):
                write_rows(run / name, [])
            frozen = control.freeze_parent("A", run)
            record = json.loads((frozen / "IDENTITY.json").read_text())
            self.assertEqual(record["best_validation"]["update"], 5750)
            self.assertEqual(record["best_validation"]["psnr"], 22.)
            self.assertEqual((frozen / "best.pt").read_bytes(), (run / "checkpoints/best.pt").read_bytes())

    def test_scientific_failures_are_not_retried_as_transient(self):
        for text in ("CUDA out of memory", "nonfinite gradient", "hash mismatch", "checkpoint corrupt"):
            self.assertFalse(transient_failure(text))
        self.assertTrue(transient_failure("OSError: Input/output error"))
        self.assertTrue(transient_failure("HCCL timeout"))


if __name__ == "__main__":
    unittest.main()

"""CPU/stdlib validation adapter tests; real inference smoke is separate."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from v03.evaluate import (G_GRID, METRIC_DIRECTIONS, paired_delta, parse_ks,
                          select_val_records, summarize_rows, verify_val_source_split)


class ValidationContractTests(unittest.TestCase):
    def test_pairing_keeps_raw_delta_and_lower_is_better_direction(self):
        old = {"psnr": 20.0, "ssim": .6, "lpips": .4, "l1": .1}
        new = {"psnr": 21.0, "ssim": .7, "lpips": .3, "l1": .08}
        result = paired_delta(new, old)
        self.assertLess(result["raw_metric_K_minus_K0"]["lpips"], 0)
        self.assertTrue(all(value > 0 for value in result["improvement_positive"].values()))

    def test_validation_selection_is_order_independent_and_nested(self):
        rows = [{"key": f"val_{i:06d}"} for i in range(20)]
        selected = select_val_records(rows, 8)
        self.assertEqual(selected, select_val_records(rows[::-1], 8))
        self.assertEqual(selected[:4], select_val_records(rows, 4))
        with self.assertRaises(ValueError):
            select_val_records(rows, 0)
        with self.assertRaises(ValueError):
            select_val_records(rows, 21)

    def test_ks_reject_unapproved_values_or_unpaired_configuration(self):
        self.assertEqual(parse_ks("5,3,1,0"), (0, 1, 3, 5))
        for value in ("1,3", "0,2", "0,1,1", ""):
            with self.assertRaises(ValueError):
                parse_ks(value)

    def test_split_audit_never_needs_a_test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            for split, source in (("train", "source-a"), ("val", "source-b")):
                (root / "manifests" / f"{split}.jsonl").write_text(json.dumps(
                    {"key": split + "_1", "source_member": source, "split": split}) + "\n")
            result = verify_val_source_split(root)
            self.assertEqual(result["train_val_source_overlap"], 0)
            self.assertFalse(result["held_out_test_opened"])
            (root / "manifests/val.jsonl").write_text(json.dumps(
                {"key": "val_1", "source_member": "source-a"}) + "\n")
            with self.assertRaises(ValueError):
                verify_val_source_split(root)

    @staticmethod
    def rows():
        result = []
        for key in ("a", "b"):
            for g in G_GRID:
                baseline = {"psnr": 20.0 + g, "ssim": .6, "lpips": .4, "l1": .1}
                for k in (0, 1):
                    quality = {name: value + k * .01 * METRIC_DIRECTIONS[name]
                               for name, value in baseline.items()}
                    steps = [] if k == 0 else [{"next_state_mse": .02,
                        "observation_residual_before": .04, "observation_residual_after": .03}]
                    result.append({"key": key, "G": g, "requested_K": k,
                        "final_quality": quality, "delta_vs_K0": paired_delta(quality, baseline),
                        "steps": steps, "network_forwards": k+1,
                        "evaluation_wall_seconds": 1., "cache_reused": False,
                        "diagnostics": {"final_network_xi_observation_mse_full": .04,
                                        "final_network_xi_observation_mse_valid": .03}})
        return result

    def test_complete_coverage_and_macro_aggregation(self):
        rows = self.rows()
        value = summarize_rows(rows, ["a", "b"], [0, 1])
        self.assertEqual(value["row_count"], 24)
        self.assertAlmostEqual(value["per_k"]["0"]["macro_g"]["psnr"], 22.625)
        self.assertAlmostEqual(value["per_k"]["1"]["improvement_positive_macro_g"]["lpips"], .01)
        self.assertIsNone(value["per_k"]["0"]["all_paths"]["mechanism_diagnostics"]
                          ["next_state_mse_path_mean"])
        with self.assertRaises(ValueError):
            summarize_rows(rows[:-1], ["a", "b"], [0, 1])
        with self.assertRaises(ValueError):
            summarize_rows(rows + [rows[0]], ["a", "b"], [0, 1])

    def test_nonfinite_quality_cannot_be_accepted(self):
        rows = copy.deepcopy(self.rows())
        rows[0]["final_quality"]["psnr"] = float("nan")
        with self.assertRaises(FloatingPointError):
            summarize_rows(rows, ["a", "b"], [0, 1])

    def test_transition_diagnostics_are_path_weighted_not_transition_weighted(self):
        rows = self.rows()
        target = next(row for row in rows if row["key"] == "a" and row["G"] == .75
                      and row["requested_K"] == 1)
        target["steps"] = [{"next_state_mse": .1, "observation_residual_before": .4,
                             "observation_residual_after": .3}] * 3
        value = summarize_rows(rows, ["a", "b"], [0, 1])
        measured = value["per_k"]["1"]["per_g"]["0.75"]["mechanism_diagnostics"]
        self.assertAlmostEqual(measured["next_state_mse_path_mean"], .06)


if __name__ == "__main__":
    unittest.main()

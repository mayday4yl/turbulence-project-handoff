"""Local plan-only audit. No Torch, remote access, model inference or training."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "archive/matched_v2_before_reporting_addendum_20260930"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def best_update(scores, nodes, tolerance):
    if set(scores) != set(nodes) or any(not math.isfinite(scores[u]) for u in nodes):
        raise ValueError("All registered candidate nodes must be complete and finite")
    maximum = max(scores.values())
    return min(u for u in nodes if scores[u] >= maximum - tolerance)


def continuation_review(series_by_group, groups, rule):
    result = {}
    for group in groups:
        values = series_by_group.get(group, {})
        epochs = rule["prior_epochs"] + rule["last_epochs"]
        if any(e not in values or not math.isfinite(values[e]) for e in epochs):
            return {"status": rule["incomplete_status"], "group_results": {}}
        prior = statistics.mean(values[e] for e in rule["prior_epochs"])
        last = statistics.mean(values[e] for e in rule["last_epochs"])
        change = statistics.mean(values[e] for e in range(36, 41)) - statistics.mean(
            values[e] for e in range(31, 36)
        )
        xbar = statistics.mean(rule["last_epochs"])
        slope = sum((e - xbar) * (values[e] - last) for e in rule["last_epochs"]) / sum(
            (e - xbar) ** 2 for e in rule["last_epochs"]
        )
        decline = slope < 0 and change <= -rule["persistent_decline_halfwindow_threshold_db"]
        trigger = last - prior >= rule["improvement_threshold_db"] and not decline
        result[group] = {"A": prior, "B": last, "B_minus_A": last - prior,
                         "D": change, "beta": slope, "persistent_decline": decline,
                         "trigger": trigger}
    status = rule["triggered_status"] if any(r["trigger"] for r in result.values()) else rule["not_triggered_status"]
    return {"status": status, "group_results": result}


def main():
    spec = read_json(ROOT / "TRAINING_PLAN_SPEC.json")
    old = read_json(BASE / "TRAINING_PLAN_SPEC.json")
    old_check = read_json(BASE / "TRAINING_PLAN_CHECK.json")
    markdown = (ROOT / "TRAINING_PLAN.md").read_text()
    assertions = []

    def check(condition, description):
        if not condition:
            raise AssertionError(description)
        assertions.append(description)

    immutable = ["plan_id", "groups", "data", "batch", "optimizer", "EMA", "precision", "seeds",
                 "preflight", "new_paths_proposed_not_created", "parent_method_spec_sha256",
                 "parent_data_scope_audit_sha256", "planned_monitor_lists_sha256"]
    for key in immutable:
        check(spec[key] == old[key], f"unchanged: {key}")
    for key, value in old["learning_rate"].items():
        check(spec["learning_rate"][key] == value, f"unchanged LR field: {key}")
    for key in ["primary", "stage1_updates_per_group", "stage1_exposures_per_group", "stage1_epochs_per_group",
                "effective_batch", "same_ordered_start_stream", "same_LR_at_same_update", "reference_hours_cap",
                "compute_or_walltime_determines_updates", "cost_role", "stop"]:
        check(spec["budget"][key] == old["budget"][key], f"unchanged budget: {key}")
    for key, value in old["budget"]["extension"].items():
        check(spec["budget"]["extension"][key] == value, f"unchanged extension constraint: {key}")
    for name in ["METHOD_AND_DATA_SPEC.json", "DATA_SCOPE_AUDIT.json", "PLANNED_MONITOR_LISTS.json"]:
        check(digest(ROOT / name) == old_check["file_sha256"][name], f"file unchanged: {name}")
    check(10000 * 32 == 320000 == 40 * 8000, "matched budget arithmetic")
    check(not spec["formal_training_authorized_by_this_document"], "plan does not authorize launch")

    evaluation = spec["evaluation"]
    nodes = list(range(0, 10001, 250))
    best = evaluation["best_validation_capability"]
    check(evaluation["raw_eval_update_nodes"] == best["candidate_updates"] == nodes, "best-val uses all41 registered RAW nodes")
    check(evaluation["EMA_eval_update_nodes"] == list(range(0, 10001, 1250)), "EMA cadence unchanged")
    check(evaluation["full_train8000_nodes"] == [0, 5000, 7500, 10000], "full-train anchors unchanged")
    cases = {}
    flat = {u: 20.0 for u in nodes}
    check(best_update(flat, nodes, 1e-6) == 0, "flat best-val tie selects initialization")
    chain = dict(flat)
    chain[250], chain[500] = 20.0000008, 20.0000016
    check(best_update(chain, nodes, 1e-6) == 250, "global-max tolerance selection avoids nontransitive online tie rule")
    cases["best_val_global_tie"] = {"selected_update": 250, "same_checkpoint_all_metrics": True}
    for damaged in [{u: v for u, v in flat.items() if u != 0}, {**flat, 250: float("nan")}]:
        try:
            best_update(damaged, nodes, 1e-6)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete best-val candidates accepted")
    assertions.append("missing/nonfinite best-val node rejected")

    rule = evaluation["sufficiency_review"]
    check(rule["prior_epochs"] == list(range(21, 31)), "prior trigger window fixed")
    check(rule["last_epochs"] == list(range(31, 41)), "last trigger window fixed")
    check(rule["prior_updates"] == [e * 250 for e in rule["prior_epochs"]], "prior epoch/update mapping")
    check(rule["last_updates"] == [e * 250 for e in rule["last_epochs"]], "last epoch/update mapping")
    check(rule["improvement_threshold_db"] == .05 and rule["persistent_decline_halfwindow_threshold_db"] == .02,
          "trigger thresholds preset at0.05 and0.02dB")
    groups = list(spec["groups"])
    base = {e: 20.0 for e in range(41)}
    curves = {
        "flat": dict(base),
        "rising": {**base, **{e: 20.06 + .002 * (e - 31) for e in range(31, 41)}},
        "persistent_decline": {**base, **{e: 20.12 - .005 * (e - 31) for e in range(31, 41)}},
        "tiny_pullback": {**base, **{e: 20.06 - .001 * (e - 31) for e in range(31, 41)}},
        "threshold_equal": {**base, **{e: 20.05 for e in range(31, 41)}},
        "below_threshold": {**base, **{e: 20.04 for e in range(31, 41)}}
    }
    expected = {"flat": False, "rising": True, "persistent_decline": False, "tiny_pullback": True,
                "threshold_equal": True, "below_threshold": False}
    for name, curve in curves.items():
        result = continuation_review({g: curve for g in groups}, groups, rule)
        wanted = rule["triggered_status"] if expected[name] else rule["not_triggered_status"]
        check(result["status"] == wanted, f"synthetic trigger: {name}")
        cases[name] = result
    only_d0 = {g: curves["flat"] for g in groups}
    only_d0["D0"] = curves["rising"]
    check(continuation_review(only_d0, groups, rule)["status"] == "TRAINING_NOT_SATURATED",
          "any of all four methods can trigger; not restricted to recursive methods")
    incomplete = {**only_d0, "R08": {21: 20.0}}
    check(continuation_review(incomplete, groups, rule)["status"] == "REVIEW_INCOMPLETE", "incomplete group blocks overall review")
    check(not spec["budget"]["extension"]["automatic"], "trigger never auto-starts +5000")

    progression = evaluation["progression"]
    check(progression["G"] == [2, 3, 4] and progression["overall_starts"] == 600, "progression Overall excludes G1")
    means = {g: statistics.mean([float(g - 1)] * 200) for g in progression["G"]}
    check(statistics.mean(means.values()) == 2.0, "synthetic equal-G first-final aggregation")
    contrasts = ["R0-D0", "R08-R0", "R08-D0", "L08-D0", "R08-L08"]
    check(evaluation["primary_pairs"] == contrasts, "all five mandatory contrasts")
    win_deltas = [0.1] * 60 + [0.0] * 20 + [-0.1] * 20
    check(sum(d > 1e-6 for d in win_deltas) / 100 == .6, "source win-rate retains ties in denominator")
    check(evaluation["bootstrap"]["unit"] == "source cluster" and evaluation["bootstrap"]["replicates"] == 10000,
          "bootstrap remains source-cluster10000")

    mutex = spec["resource"]["evaluation_mutex"]
    check(mutex["per_card_max_evaluators"] == 1 and not mutex["peer_training_allowed_during_eval"],
          "mutex forbids both peer evaluation and training during lease")
    check(mutex["grant_requires_live_peer_pause_ACK"] and mutex["paused_peer_resident_model_optimizer_in_memory_acceptance"],
          "mutex requires boundary ACK and accounts for paused-peer memory")
    art = spec["artifacts"]
    check(not art["full_image_tensor_archive"]["required"] and not art["full_image_tensor_archive"]["default_enabled"],
          "full55GB tensor archive not required or enabled by default")
    check(art["qualitative"]["layouts"] == ["vertical", "horizontal"] and art["qualitative"]["starts"] == 40,
          "fixed40 qualitative includes both layouts")
    figures = [x["stem"] for x in spec["deliverables"]["main_figures"]]
    tables = [x["stem"] for x in spec["deliverables"]["main_tables"]]
    check(figures == ["train_val_psnr_curve", "train_val_loss_curve", "progression_by_G"], "three exact main figure names")
    check(tables == ["main_performance_table", "paired_ablation_table", "progression_by_G_table"], "three exact main table names")
    for token in figures + tables + ["best_validation_capability_table", "TRAINING_NOT_SATURATED", "EXTENSION_TRIGGER_NOT_MET", "REVIEW_INCOMPLETE"]:
        check(token in markdown, f"Markdown contains required artifact/status: {token}")
    for stale in ["本版不新增自动数值plateau", "全量浮点输出只在U=10,000保存", "一组评估/另一组训练的重叠"]:
        check(stale not in markdown, f"stale conflicting text removed: {stale}")

    files = ["METHOD_AND_DATA_SPEC.json", "DATA_SCOPE_AUDIT.json", "PLANNED_MONITOR_LISTS.json",
             "TRAINING_PLAN.md", "TRAINING_PLAN_SPEC.json", "check_training_plan.py"]
    result = {
        "status": "PASS_LOCAL_PLAN_INVARIANTS_AND_SYNTHETIC_RULE_CHECKS_ONLY",
        "plan_id": spec["plan_id"], "revision_id": spec["revision_id"],
        "runtime_training_verified": False, "new_training_optimizer_steps": 0, "remote_access_performed": False,
        "numerical_evaluation_repeat_check_executed": False, "GPU_mutex_runtime_verified": False,
        "training_sufficiency_established": False, "formal_training_started": False,
        "assertion_count": len(assertions), "checks": assertions, "synthetic_cases": cases,
        "file_sha256": {name: digest(ROOT / name) for name in files},
        "prior_matched_v2_files_sha256": {name: digest(BASE / name) for name in ["TRAINING_PLAN.md", "TRAINING_PLAN_SPEC.json", "TRAINING_PLAN_CHECK.json"]},
        "reproduce": "python3 'output/v4_linear_path_design_20260929/check_training_plan.py'",
        "source_counts": old_check["source_counts"], "monitor_counts": old_check["monitor_counts"],
        "visual_counts": old_check["visual_counts"],
        "limitation": "No implementation, GPU numerical stability, mutex behavior, metric generation, model forward/backward or training has been tested by this local plan check."
    }
    (ROOT / "TRAINING_PLAN_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "assertion_count", "new_training_optimizer_steps", "formal_training_started"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write an evidence-only update500/1000 report using the Python standard library.

Exit 0 requires a complete formal training log/checkpoint and identity-consistent
1600-start raw/EMA x unrefined/refined evaluation. Incomplete evidence still
produces a report, with explicit gaps, and exits 2. No torch or NPU is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, errors, label):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
        return {}


def read_jsonl(path, errors, label):
    rows = []
    try:
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected JSON object")
                    rows.append(row)
                except ValueError as exc:
                    errors.append(f"{label}:{line_number}: {exc}")
    except OSError as exc:
        errors.append(f"{label}: {exc}")
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    return statistics.fmean(values) if values else None


def fmt(value):
    if value is None:
        return "未记录/不适用"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "非有限"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(fmt(value) for value in row) + " |" for row in rows]


def metric_rows(records):
    curves, pairs, refinement = [], [], []
    for g in (4, 3, 2, 1):
        subset = [row for row in records if row.get("initial_G") == g]
        for k in range(1, g + 1):
            predictions = [row["predictions"][k - 1] for row in subset if len(row.get("predictions", [])) >= k]
            mses = [p["clean_mse"] for p in predictions if finite(p.get("clean_mse"))]
            residuals = [p["network_redegradation_mse"] for p in predictions if finite(p.get("network_redegradation_mse"))]
            psnrs = [p["clean_psnr_db"] for p in predictions if finite(p.get("clean_psnr_db"))]
            zeros = sum(p.get("zero_mse") is True for p in predictions)
            curves.append({"G": g, "K": k, "count": len(predictions), "mse": mean(mses),
                           "psnr": "+∞（含精确零 MSE）" if zeros else mean(psnrs), "zero_count": zeros,
                           "network_residual": mean(residuals)})
        if g > 1:
            gains = [row.get("full_minus_one_shot") for row in subset]
            gains = [gain for gain in gains if isinstance(gain, dict)]
            psnr_gain = [gain["psnr_gain_db"] for gain in gains if finite(gain.get("psnr_gain_db"))]
            outcomes = Counter(gain.get("outcome") for gain in gains)
            pairs.append({"G": g, "count": len(gains), "gain": mean(psnr_gain) if len(psnr_gain) == len(gains) else None,
                          "nonfinite_gain": len(gains) - len(psnr_gain),
                          "mse_gain": mean([gain["mse_gain"] for gain in gains if finite(gain.get("mse_gain"))]),
                          "win": outcomes["win"], "tie": outcomes["tie"], "loss": outcomes["loss"]})
        transitions = [(step["refinement"], row.get("batch_size")) for row in subset
                       for step in row.get("predictions", []) if isinstance(step.get("refinement"), dict)]
        if transitions:
            choices = Counter(t.get("initialization_source") for t, _ in transitions)
            before = [t["selected_initial_objective"] for t, _ in transitions if finite(t.get("selected_initial_objective"))]
            after = [t["best_objective"] for t, _ in transitions if finite(t.get("best_objective"))]
            solver_seconds = [t["batch_solver_seconds"] / size for t, size in transitions
                              if finite(t.get("batch_solver_seconds")) and isinstance(size, int) and size > 0]
            refinement.append({"G": g, "count": len(transitions), "network": choices["network"],
                               "previous": choices["previous_refined"], "before": mean(before), "after": mean(after),
                               "solver_seconds": sum(solver_seconds) if len(solver_seconds) == len(transitions) else None,
                               "finite": all(t.get("finite") is True for t, _ in transitions),
                               "connected": all(t.get("gradient_connected") is True for t, _ in transitions),
                               "z_updates": sum(t.get("nonzero_zernike_update") is True for t, _ in transitions),
                               "t_updates": sum(t.get("nonzero_tilt_update") is True for t, _ in transitions)})
    return curves, pairs, refinement


def inspect_evaluation(root, update, checkpoint_hash, authoritative, errors):
    groups, summaries, identities = {}, {}, []
    expected_ids = {row["start_id"] for row in authoritative}
    original = {row["start_id"]: row for row in authoritative}
    for weights in ("raw", "ema"):
        folder = root / "artifacts/evaluation" / f"update_{update:06d}" / weights
        summary = read_json(folder / "summary.json", errors, f"{weights} summary")
        progress = read_json(folder / "progress.json", errors, f"{weights} progress")
        summaries[weights] = summary
        identity = summary.get("identity", {})
        identities.append(identity)
        for label, record in (("summary", summary), ("progress", progress)):
            if record.get("status") != "complete":
                errors.append(f"{weights}/{label} status is not complete")
        if identity != progress.get("identity"):
            errors.append(f"{weights} summary/progress identity mismatch")
        if (identity.get("global_update") != update or identity.get("weights") != weights
                or identity.get("checkpoint_sha256") != checkpoint_hash):
            errors.append(f"{weights} checkpoint/update/weights identity mismatch")
        if summary.get("start_count_per_mode") != 1600:
            errors.append(f"{weights} summary does not declare 1600 starts per mode")
        if not summary.get("network_unchanged") or not summary.get("simulator_unchanged"):
            errors.append(f"{weights} unchanged-network/simulator evidence is absent")
        for mode in ("unrefined", "refined"):
            label = f"{weights}/{mode}"
            records = read_jsonl(folder / mode / "per_start.jsonl", errors, label)
            groups[(weights, mode)] = records
            ids = [row.get("start_id") for row in records]
            if len(records) != 1600 or len(set(ids)) != 1600 or set(ids) != expected_ids:
                errors.append(f"{label} coverage is not exactly 1600 unique authoritative starts")
            severities = Counter(row.get("initial_G") for row in records)
            if severities != Counter({1: 400, 2: 400, 3: 400, 4: 400}):
                errors.append(f"{label} does not have 400 starts at each G")
            invalid = []
            for row in records:
                g = row.get("initial_G")
                source = original.get(row.get("start_id"), {})
                steps = row.get("predictions", [])
                valid = (g in (1, 2, 3, 4) and len(steps) == g and row.get("mode") == mode
                         and row.get("weights") == weights and row.get("global_update") == update
                         and source.get("initial_G") == g and source.get("trajectory_id") == row.get("trajectory_id"))
                for index, step in enumerate(steps):
                    mse, psnr = step.get("clean_mse"), step.get("clean_psnr_db")
                    valid = valid and step.get("K") == index + 1 and step.get("current_g") == g - index
                    valid = valid and finite(mse) and mse >= 0 and finite(step.get("network_redegradation_mse"))
                    valid = valid and ((mse == 0 and step.get("zero_mse") is True and psnr is None)
                                       or (mse > 0 and step.get("zero_mse") is False and finite(psnr)
                                           and math.isclose(psnr, -10 * math.log10(mse), rel_tol=1e-9, abs_tol=1e-8)))
                    ref = step.get("refinement")
                    if mode == "refined" and index < g - 1:
                        valid = valid and isinstance(ref, dict)
                        if isinstance(ref, dict):
                            j0, best = ref.get("selected_initial_objective"), ref.get("best_objective")
                            valid = valid and finite(j0) and finite(best) and 0 <= best <= j0
                            valid = valid and ref.get("finite") is True and ref.get("gradient_connected") is True
                    else:
                        valid = valid and ref is None
                pair = row.get("full_minus_one_shot")
                valid = valid and ((g == 1 and pair is None) or
                                   (g != 1 and isinstance(pair, dict) and pair.get("outcome") in ("win", "tie", "loss")))
                if not valid:
                    invalid.append(row.get("start_id"))
            if invalid:
                errors.append(f"{label} invalid prefix/metric/refinement records: {len(invalid)}; first={invalid[:3]}")
    if len(identities) == 2:
        for key in ("training_identity", "manifest_sha256", "calibration_identity", "numeric_config", "tie_psnr_db"):
            if identities[0].get(key) != identities[1].get(key):
                errors.append(f"raw/EMA differ in {key}")
    return groups, summaries


def window_rows(records, update):
    ranges = [(1, min(50, update)), (max(1, update - 99), update), (max(1, update - 49), update)]
    if update == 1000:
        ranges.insert(1, (451, 500))
    rows = []
    for low, high in ranges:
        subset = [row for row in records if low <= row.get("global_update", -1) <= high]
        values = {key: [row[key] for row in subset if finite(row.get(key))]
                  for key in ("loss", "LC", "LR", "preclip_global_norm", "lr_used", "seconds")}
        rows.append([f"{low}–{high}", len(subset), mean(values["loss"]), mean(values["LC"]), mean(values["LR"]),
                     mean(values["preclip_global_norm"]), max(values["preclip_global_norm"], default=None),
                     min(values["lr_used"], default=None), max(values["lr_used"], default=None),
                     statistics.median(values["seconds"]) if values["seconds"] else None])
    return rows


def build_report(root, update):
    errors, training_errors, evaluation_errors = [], [], []
    config_path = root / "configs/train_v031.json"
    config = read_json(config_path, errors, "training config")
    dataset_root = Path(config.get("dataset_root", "/missing-v031-dataset"))
    manifest_path = dataset_root / "start_manifest.jsonl"
    authoritative = read_jsonl(manifest_path, errors, "authoritative starts")
    if len(authoritative) != 1600 or len({r.get("start_id") for r in authoritative}) != 1600:
        errors.append("Authoritative manifest does not contain 1600 unique starts")
    checkpoint = root / "runs/formal" / f"checkpoint_{update:06d}.pt"
    checkpoint_hash = sha256(checkpoint) if checkpoint.is_file() else None
    if checkpoint_hash is None:
        training_errors.append(f"Missing checkpoint: {checkpoint}")
    log_path = root / "runs/formal/updates_rank0.jsonl"
    log = read_jsonl(log_path, training_errors, "formal rank0 update log")
    records = [row for row in log if isinstance(row.get("global_update"), int)
               and 1 <= row["global_update"] <= update]
    counts = Counter(row["global_update"] for row in records)
    if set(counts) != set(range(1, update + 1)) or any(count != 1 for count in counts.values()):
        training_errors.append("Formal update log is not unique/contiguous through requested update")
    nonfinite = [row["global_update"] for row in records if row.get("formal") is not True
                 or row.get("smoke_optimizer_updates") != 0 or row.get("finite_loss") is not True
                 or row.get("finite_gradients") is not True or not all(finite(row.get(key))
                    for key in ("loss", "LC", "LR", "preclip_global_norm", "lr_used", "seconds"))]
    if nonfinite:
        training_errors.append(f"Missing/nonfinite/nonformal training evidence at updates: {nonfinite[:10]}")
    records.sort(key=lambda row: row["global_update"])
    latest = records[-1] if records else {}
    groups, summaries = inspect_evaluation(root, update, checkpoint_hash, authoritative, evaluation_errors)
    calibration_path = root / "artifacts/calibration_update100/calibration.json"
    calibration = read_json(calibration_path, evaluation_errors, "update100 calibration")
    if calibration.get("status") != "complete":
        evaluation_errors.append("Update100 calibration is not complete")
    for weights, summary in summaries.items():
        identity = summary.get("identity", {})
        if identity.get("calibration_identity") != calibration.get("identity") or identity.get("numeric_config") != calibration.get("selected"):
            evaluation_errors.append(f"{weights} numeric configuration differs from frozen update100 calibration")
        if manifest_path.exists() and identity.get("manifest_sha256") != sha256(manifest_path):
            evaluation_errors.append(f"{weights} evaluation manifest is not the authoritative full manifest")
        if identity.get("tie_psnr_db") != config.get("evaluation", {}).get("tie_psnr_db"):
            evaluation_errors.append(f"{weights} tie threshold differs from frozen training config")
    training_complete = not training_errors and not errors
    evaluation_complete = not evaluation_errors and not errors
    complete = training_complete and evaluation_complete
    lines = [f"# V0.31 update {update} 证据报告", "",
             f"恢复网络训练完成到 update {update}：**{'是' if training_complete else '未通过完整性核验'}**。",
             f"该检查点完整评估完成：**{'是' if evaluation_complete else '未通过完整性核验'}**。", "",
             "两项状态分别判断。以下只汇总已有日志与逐起点记录；全量集合属于训练数据上的 overfit 诊断。",
             "达到预算不证明收敛，多步收益或其变化也不证明未见数据泛化。", "",
             f"- Checkpoint：`{checkpoint}`", f"- Checkpoint SHA-256：`{checkpoint_hash or '缺失'}`",
             f"- 固定 PSNR 配对 tie 容差：`{config.get('evaluation', {}).get('tie_psnr_db', '缺失')}` dB。",
             "- 指标来自原始 float RGB；单图全图 mean MSE、data range=1 的单图 PSNR 后取平均，精确零 MSE 单列。", ""]
    if errors or training_errors or evaluation_errors:
        lines += ["## 尚未通过的证据检查", ""] + [f"- {error}" for error in errors + training_errors + evaluation_errors] + [""]
    lines += ["## 训练日志", "", f"已读取请求范围内 {len(records)} 条记录；finite loss/gradient 与必要数值字段全部通过：{fmt(not nonfinite and bool(records))}。",
              "表中 LR 是重退化损失；学习率列指恢复网络 AdamW 的 lr_used。", ""]
    lines += table(["update 窗口", "记录数", "loss 均值", "LC 均值", "LR 均值", "clip 前 norm 均值", "norm 最大值", "最小学习率", "最大学习率", "中位秒/update"], window_rows(records, update))
    observed_seconds = [r["seconds"] for r in records if finite(r.get("seconds"))]
    peaks = [r.get("memory", {}).get("peak_allocated_bytes") for r in records]
    peaks = [x for x in peaks if finite(x)]
    lines += ["", f"日志记录的 update 总耗时：{fmt(sum(observed_seconds) if observed_seconds else None)} 秒；rank0 记录的 peak allocated 最大值：{fmt(max(peaks, default=None))} bytes。",
              "update 计时采用分布式最慢 rank；上述峰值只依据 rank0 日志，不冒称四卡总体峰值。", ""]
    cumulative = latest.get("cumulative_counts", {})
    lines += table(["累计统计", "日志值"], [[key, cumulative.get(key)] for key in ("paths", "predictions", "true_inputs", "predicted_inputs")])
    lines += [""] + table(["G / 当前 g", "原始 G 路径数", "当前 g 预测数", "真实输入", "预测输入"], [
        [g, cumulative.get("by_initial_G", {}).get(str(g)),
         cumulative.get("by_current_g", {}).get(str(g), {}).get("predictions"),
         cumulative.get("by_current_g", {}).get(str(g), {}).get("true_inputs"),
         cumulative.get("by_current_g", {}).get(str(g), {}).get("predicted_inputs")] for g in (4, 3, 2, 1)])
    lines += ["", "## 校准与评估配置", "", f"固定数值配置：`{json.dumps(calibration.get('selected'), ensure_ascii=False)}`。",
              f"calibration_low_signal：{fmt(calibration.get('calibration_low_signal'))}；校准 checkpoint SHA-256：`{calibration.get('identity', {}).get('checkpoint_sha256', '缺失')}`。",
              "数值配置只按 update100 的固定原始观测残差问题选择；不能将其解释为图像质量最优。", ""]
    lines += table(["权重", "summary 状态", "每模式起点数", "rollout 秒", "batch wall 秒", "完成进程秒", "peak bytes", "降批事件数"], [
        [weights, summary.get("status"), summary.get("start_count_per_mode"), summary.get("measured_rollout_seconds"),
         summary.get("measured_batch_wall_seconds"), summary.get("completing_process_seconds"),
         summary.get("peak_allocated_bytes"), len(summary.get("batch_size_events", []))]
        for weights, summary in summaries.items()])
    aggregates = {}
    for weights in ("raw", "ema"):
        for mode in ("unrefined", "refined"):
            rows = groups[(weights, mode)]
            curves, pairs, refinement = metric_rows(rows)
            aggregates[(weights, mode)] = (curves, pairs, refinement)
            lines += ["", f"## {weights} / {mode}", "", f"读取 {len(rows)} 个起点；下表固定报告全部合法前缀，没有事后选择 best K。", ""]
            lines += table(["原始 G", "K", "起点数", "clean MSE", "平均单图 PSNR dB", "精确零 MSE 数", "网络预测 xi 重退化 MSE"], [
                [r["G"], r["K"], r["count"], r["mse"], r["psnr"], r["zero_count"], r["network_residual"]] for r in curves])
            lines += ["", "同一原始 Y/G 的完整路径相对 K=1：", ""]
            lines += table(["G", "配对数", "PSNR gain dB", "MSE 改善", "胜", "平", "负", "非有限 gain 数"], [
                [r["G"], r["count"], r["gain"], r["mse_gain"], r["win"], r["tie"], r["loss"], r["nonfinite_gain"]] for r in pairs])
            lines += ["", "G=1 仅有 K=1，单独作为单步结果，不计算多步胜/平/负。"]
            if refinement:
                lines += ["", "以下 objective 是择优初始化/xi 优化残差，与上面的网络预测 xi 重退化误差分开记录。", ""]
                lines += table(["G", "transition 数", "选网络", "继承 previous", "优化前 J 均值", "best J 均值", "solver 秒", "有限", "连接", "Z 非零更新数", "tilt 非零更新数"], [
                    [r["G"], r["count"], r["network"], r["previous"], r["before"], r["after"], r["solver_seconds"],
                     r["finite"], r["connected"], r["z_updates"], r["t_updates"]] for r in refinement])
                lines += ["", "solver 秒按每条记录的 batch_size 去除同批逐样本重复计时；返回 best 包含初始化，不增本身不证明有效改善。"]
    if update == 1000:
        lines += ["", "## update500 → update1000 的现有证据", ""]
        trend = []
        for weights in ("raw", "ema"):
            prior_path = root / "artifacts/evaluation/update_000500" / weights / "summary.json"
            if not prior_path.exists():
                lines.append(f"{weights} 的 update500 summary 未找到，未推断该组趋势。")
                continue
            prior_errors = []
            prior = read_json(prior_path, prior_errors, f"{weights} update500")
            current_identity = summaries[weights].get("identity", {})
            prior_identity = prior.get("identity", {})
            compatible = (prior.get("status") == "complete" and prior.get("start_count_per_mode") == 1600
                          and prior_identity.get("global_update") == 500 and prior_identity.get("weights") == weights
                          and all(prior_identity.get(k) == current_identity.get(k) for k in
                                  ("training_identity", "manifest_sha256", "calibration_identity", "numeric_config", "tie_psnr_db")))
            if not compatible:
                lines.append(f"{weights} 的 update500 证据不完整或身份不兼容，未计算跨检查点趋势。")
                continue
            for mode in ("unrefined", "refined"):
                old = prior.get("modes", {}).get(mode, {}).get("G_curves", [])
                for row in aggregates[(weights, mode)][0]:
                    if row["K"] not in (1, row["G"]):
                        continue
                    match = next((x for x in old if x.get("initial_G") == row["G"] and x.get("K") == row["K"]), None)
                    if match is None:
                        continue
                    old_mse, old_psnr = match.get("mean_clean_mse"), match.get("mean_clean_psnr_db")
                    trend.append([weights, mode, row["G"], row["K"], old_mse, row["mse"],
                                  old_mse - row["mse"] if finite(old_mse) and finite(row["mse"]) else None,
                                  row["psnr"] - old_psnr if finite(row["psnr"]) and finite(old_psnr) else None])
        if trend:
            lines += [""] + table(["权重", "模式", "G", "K", "500 MSE", "1000 MSE", "MSE 改善", "PSNR 变化 dB"], trend)
        lines += ["", "正 MSE 改善/PSNR 变化表示这些已记录训练样本在两个固定检查点间改善；负值表示退化。该比较不判定收敛，也不授权增加训练预算。"]
    lines += ["", "## 证据位置与指纹", ""]
    evidence = [config_path, log_path, checkpoint, calibration_path, manifest_path]
    for weights in ("raw", "ema"):
        folder = root / "artifacts/evaluation" / f"update_{update:06d}" / weights
        evidence += [folder / "summary.json", folder / "progress.json"] + [folder / mode / "per_start.jsonl" for mode in ("unrefined", "refined")]
    for path in evidence:
        digest = checkpoint_hash if path == checkpoint else sha256(path) if path.is_file() else "缺失"
        lines.append(f"- `{path}` — SHA-256 `{digest}`")
    lines += ["", f"报告完整性状态：`{'complete' if complete else 'incomplete'}`。"]
    output = root / "artifacts" / f"V031_UPDATE{update}_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    return {"status": "complete" if complete else "incomplete", "training_complete": training_complete,
            "evaluation_complete": evaluation_complete, "report": str(output),
            "errors": errors + training_errors + evaluation_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--update", required=True, type=int, choices=(500, 1000))
    args = parser.parse_args()
    result = build_report(args.root.resolve(), args.update)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["status"] == "complete" else 2)


if __name__ == "__main__":
    main()

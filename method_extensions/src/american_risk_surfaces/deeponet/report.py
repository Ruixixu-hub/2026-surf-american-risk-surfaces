"""Layered decisions, figures, and Chinese plain-language synthesis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.deeponet.protocol import (
    REPORTS_DIR,
    RESULTS_DIR,
    SOURCE_SNAPSHOT_DIR,
    VALIDATION_CACHE_DIR,
    protocol_hash,
)
from american_risk_surfaces.deeponet.study import (
    DEVELOPMENT_DIR,
    FIVE_SEED_DIR,
    HELDOUT_DIR,
    RUNTIME_DIR,
    SYNTHESIS_DIR,
    rows_pass_approximate_gate,
)


def synthesize_deeponet() -> dict[str, object]:
    SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = FIVE_SEED_DIR / "validation_decision.json"
    if not validation_path.exists():
        development_path = DEVELOPMENT_DIR / "development_decision.json"
        development = {"status": "NOT_RUN_FOR_CURRENT_PROTOCOL"}
        if development_path.exists():
            candidate = json.loads(development_path.read_text(encoding="utf-8"))
            if candidate.get("protocol_hash") == protocol_hash():
                development = candidate
        decision = {
            "status": "INCOMPLETE",
            "reason": "formal five-seed validation has not been completed",
            "protocol_hash": protocol_hash(),
            "development": development,
            "formal_results_available": False,
            "smoke_results_excluded": True,
            "next_method": "finish registered Windows CUDA development and validation",
        }
        return _write_outputs(decision)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    heldout_metrics = HELDOUT_DIR / "heldout_metrics.csv"
    if not heldout_metrics.exists():
        decision = {
            "status": (
                "STOP_BEFORE_HELDOUT"
                if validation["status"] == "STOP_BEFORE_HELDOUT"
                else "AWAITING_HELDOUT"
            ),
            "reason": "validation gate did not open heldout" if validation["status"] == "STOP_BEFORE_HELDOUT" else "validation passed but heldout scoring is incomplete",
            "protocol_hash": protocol_hash(),
            "validation": validation,
            "heldout_read": False,
            "next_method": "stop global operator route" if validation["status"] == "STOP_BEFORE_HELDOUT" else "complete one-time heldout prediction and scoring",
        }
        return _write_outputs(decision)
    rows = _read_csv(heldout_metrics)
    runtime_rows = _read_csv(RUNTIME_DIR / "runtime_samples_cuda.csv")
    cpu_runtime_rows = _read_csv(RUNTIME_DIR / "runtime_samples_cpu.csv")
    offline = _offline_cost_summary()
    families = {}
    for family in ("put", "call"):
        learned = [
            row for row in rows if row["option_type"] == family and int(row["seed"]) >= 0
        ]
        gate = rows_pass_approximate_gate(learned, multi_seed=True)
        status = "GO_APPROXIMATE_DEEPONET" if gate["passes"] else (
            "STOP_STRUCTURE"
            if learned and max(float(row["reduction_price_rmse"]) for row in learned) <= 1.5 * 4.94989e-4
            else "STOP_PRICE_MAPPING"
        )
        runtime = (
            _runtime_decisions(runtime_rows, family, offline[family])
            if gate["passes"] else {}
        )
        if cpu_runtime_rows:
            runtime["cpu_only_runtime"] = _runtime_medians(cpu_runtime_rows, family)
        families[family] = {"status": status, "gate": gate, **runtime}
    passing = [key for key, item in families.items() if item["status"] == "GO_APPROXIMATE_DEEPONET"]
    decision = {
        "status": "GO_APPROXIMATE_DEEPONET" if len(passing) == 2 else "PARTIAL_GO" if passing else "STOP_DEEPONET",
        "protocol_hash": protocol_hash(), "families": families,
        "offline_cost": offline,
        "next_method": "multi-fidelity, then UQ/OOD/Policy fallback" if passing else "stop the global operator route; keep FNO deferred",
    }
    _write_pareto_figures(rows, runtime_rows)
    return _write_outputs(decision)


def _runtime_decisions(rows, family, offline):
    selected = [row for row in rows if row["option_type"] == family]
    medians = {}
    p95 = {}
    for arm in {row["arm"] for row in selected}:
        values = np.asarray([float(row["seconds"]) for row in selected if row["arm"] == arm])
        medians[arm] = float(np.median(values))
        p95[arm] = float(np.quantile(values, 0.95))
    result = {"runtime_median": medians, "runtime_p95": p95}
    if "deeponet_safe" in medians:
        for baseline, status in (("psor", "GO_BEATS_CN_PSOR"), ("policy", "GO_BEATS_CN_POLICY")):
            saving = medians.get(baseline, 0.0) - medians["deeponet_safe"]
            break_even = (
                float(offline["total_seconds"]) / saving
                if saving > 0.0 else float("inf")
            )
            result[f"break_even_queries_vs_{baseline}"] = break_even
            result[status] = bool(
                baseline in medians
                and medians["deeponet_safe"] <= 0.8 * medians[baseline]
                and p95["deeponet_safe"] <= p95[baseline]
                and break_even <= 10000.0
            )
    if "hybrid" in medians:
        strict = [row for row in selected if row["arm"] == "hybrid"]
        exact = all(
            str(row["converged"]).lower() == "true"
            and float(row["max_lcp_residual"]) <= 1e-12
            and float(row["max_solution_difference_vs_policy"]) <= 1e-12
            for row in strict
        )
        for baseline, status in (("psor", "GO_EXACT_HYBRID_PSOR"), ("policy", "GO_EXACT_HYBRID_POLICY")):
            result[status] = bool(
                exact and baseline in medians
                and medians["hybrid"] <= 0.8 * medians[baseline]
                and p95["hybrid"] <= p95[baseline]
            )
    return result


def _runtime_medians(rows, family):
    selected = [row for row in rows if row["option_type"] == family]
    return {
        arm: float(np.median([
            float(row["seconds"]) for row in selected if row["arm"] == arm
        ]))
        for arm in sorted({row["arm"] for row in selected})
    }


def _offline_cost_summary():
    """Account for all persisted offline costs available to this study."""

    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None
    output = {}
    for family in ("put", "call"):
        snapshot_seconds = 0.0
        for path in (SOURCE_SNAPSHOT_DIR / family).glob("*.npz"):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))
            snapshot_seconds += float(metadata.get("generation_seconds", 0.0))
        training_seconds = 0.0
        checkpoint_paths = [
            *sorted((DEVELOPMENT_DIR / family).glob("*/checkpoint.pt")),
            *sorted((FIVE_SEED_DIR / family).glob("*/checkpoint.pt")),
        ]
        if torch is not None:
            for path in checkpoint_paths:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                training_seconds += float(payload.get("training_seconds", 0.0))
        reference_seconds = 0.0
        for path in VALIDATION_CACHE_DIR.glob(f"{family}_*.npz"):
            with np.load(path, allow_pickle=False) as data:
                reference_seconds += float(data["cn_seconds"])
                reference_seconds += float(data["high_reference_seconds"])
        total = snapshot_seconds + training_seconds + reference_seconds
        output[family] = {
            "snapshot_generation_seconds": snapshot_seconds,
            "development_and_formal_training_seconds": training_seconds,
            "validation_reference_seconds": reference_seconds,
            "total_seconds": total,
            "scope": (
                "persisted snapshot generation + all DeepONet development/formal "
                "checkpoint training + validation CN/DIRK reference generation"
            ),
        }
    return output


def _write_pareto_figures(metric_rows, runtime_rows):
    """Write dependency-free SVG summaries when formal heldout/timing exists."""

    if not metric_rows or not runtime_rows:
        return
    figure_dir = SYNTHESIS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for family, color in (("put", "#1f77b4"), ("call", "#d95f02")):
        learned = [
            row for row in metric_rows
            if row["option_type"] == family and int(row["seed"]) >= 0
        ]
        timing = [row for row in runtime_rows if row["option_type"] == family]
        if not learned or not timing:
            continue
        price = float(np.median([float(row["reduction_price_rmse"]) for row in learned]))
        boundary_values = [
            float(row["reduction_boundary_conditional_mae"]) for row in learned
            if np.isfinite(float(row["reduction_boundary_conditional_mae"]))
        ]
        boundary = float(np.median(boundary_values)) if boundary_values else float("nan")
        f1 = float(np.median([float(row["reduction_exercise_f1"]) for row in learned]))
        lcp = float(np.median([
            float(row["normalized_full_lcp_residual_p95"]) for row in learned
        ]))
        arms = {}
        for arm in ("psor", "policy", "deeponet_safe", "hybrid"):
            values = [float(row["seconds"]) for row in timing if row["arm"] == arm]
            if values:
                arms[arm] = float(np.median(values))
        quality_rows = [
            ("median price RMSE", price),
            ("median boundary MAE", boundary),
            ("1 - median exercise F1", 1.0 - f1),
            ("median normalized LCP p95", lcp),
        ]
        _simple_svg(
            figure_dir / f"{family}_quality_summary.svg",
            f"{family}: heldout DeepONet quality (lower is better)", quality_rows, color,
        )
        _simple_svg(
            figure_dir / f"{family}_runtime_summary.svg",
            f"{family}: same-machine safe latency", list(arms.items()), color,
        )


def _simple_svg(path, title, rows, color):
    finite = [(label, value) for label, value in rows if np.isfinite(value)]
    if not finite:
        return
    width, height = 800, 90 + 55 * len(finite)
    maximum = max(value for _, value in finite)
    scale = 570.0 / max(maximum, 1e-15)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="32" font-size="19" font-family="Arial">{title}</text>',
    ]
    for index, (label, value) in enumerate(finite):
        y = 60 + 55 * index
        bar = max(1.0, value * scale)
        parts.extend([
            f'<text x="20" y="{y+21}" font-size="13" font-family="Arial">{label}</text>',
            f'<rect x="190" y="{y+5}" width="{bar:.2f}" height="24" fill="{color}" opacity="0.8"/>',
            f'<text x="{min(775.0, 200+bar):.2f}" y="{y+22}" font-size="12" font-family="Arial">{value:.6g}</text>',
        ])
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_outputs(decision):
    path = SYNTHESIS_DIR / "method_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    status = decision["status"]
    report = f"""# Positive-Premium DeepONet 实验结论

## 当前状态

**{status}**

## 诚实解释

{_plain_reason(decision)}

DeepONet 是参数条件的 branch--trunk solution operator。本项目只在完整五-seed validation 通过后才允许打开 test/stress；未运行的 CUDA 任务、held-out、速度或 hybrid 不会被写成已完成结果。

## 下一步

{decision.get('next_method', 'follow the frozen stage gate')}
"""
    (REPORTS_DIR / "positive_premium_deeponet_结论_CN.md").write_text(report, encoding="utf-8")
    return decision


def _plain_reason(decision):
    if decision["status"] == "INCOMPLETE":
        return "代码和协议已就绪，但正式 Windows CUDA development 与五-seed validation 尚未完成，因此现在没有 DeepONet 性能结论。"
    if decision["status"] == "STOP_BEFORE_HELDOUT":
        return "正式 validation 未通过联合价格、边界、Greek、exercise-set 和 LCP gate，所以 held-out 继续封存。"
    if decision["status"] == "AWAITING_HELDOUT":
        return "Validation 已允许继续，但一次性 held-out 预测和评分尚未完成。"
    if decision["status"] in {"GO_APPROXIMATE_DEEPONET", "PARTIAL_GO"}:
        return "至少一个期权族通过了 held-out 联合质量门；速度与 exact hybrid 是否胜出必须读取各自的分层状态。"
    return "DeepONet 没有在 held-out 联合门上成为合格风险面代理，不能声称超过经典 benchmark。"


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

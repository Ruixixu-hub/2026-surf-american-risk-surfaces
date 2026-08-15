"""Synthesis tables, plots, and plain-language Chinese decision report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from american_risk_surfaces.basis_operator.protocol import REPORTS_DIR, RESULTS_DIR, protocol_hash


def synthesize_basis_operator() -> dict[str, object]:
    representation = json.loads(
        (RESULTS_DIR / "03_representation_ceiling/representation_decision.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (RESULTS_DIR / "05_five_seed_validation/validation_decision.json").read_text(encoding="utf-8")
    )
    heldout_status_path = RESULTS_DIR / "06_heldout/prediction_status.json"
    heldout = json.loads(heldout_status_path.read_text(encoding="utf-8")) if heldout_status_path.exists() else {"status": "NOT_RUN"}
    with (RESULTS_DIR / "05_five_seed_validation/five_seed_validation_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metrics = list(csv.DictReader(handle))
    family_summary = {}
    for family in ("put", "call"):
        rows = [row for row in metrics if row["option_type"] == family]
        regime_ids = sorted({row["regime_id"] for row in rows})
        median_price = {
            regime_id: float(np.median([
                float(row["reduction_price_rmse"]) for row in rows if row["regime_id"] == regime_id
            ])) for regime_id in regime_ids
        }
        median_boundary = [
            float(np.median([
                float(row["reduction_boundary_conditional_mae"])
                for row in rows if row["regime_id"] == regime_id
                and np.isfinite(float(row["reduction_boundary_conditional_mae"]))
            ]))
            for regime_id in regime_ids
            if any(
                np.isfinite(float(row["reduction_boundary_conditional_mae"]))
                for row in rows if row["regime_id"] == regime_id
            )
        ]
        lcp_ratios = [
            float(row["normalized_full_lcp_residual_p95"]) / max(float(row["oracle_lcp_p95"]), 1e-15)
            for row in rows
        ]
        family_summary[family] = {
            "representation_min_passing_modes": min(representation["families"][family]["passing_modes"]),
            "representation_32_mode_worst_rmse": next(
                item["worst_reduction_rmse"]
                for item in representation["families"][family]["ladder"] if item["modes"] == 32
            ),
            "selected_modes": validation["families"][family]["selected_modes"],
            "five_seed_worst_seed_median_price_rmse": max(median_price.values()),
            "five_seed_worst_seed_median_boundary_mae": max(median_boundary) if median_boundary else None,
            "median_lcp_p95_ratio_to_oracle": float(np.median(lcp_ratios)),
            "max_gate_ratio": validation["families"][family]["approximate_gate"]["max_gate_ratio"],
            "price_gate_passes": validation["families"][family]["price_gate_passes"],
            "status": validation["families"][family]["status"],
        }
    final_status = (
        "STOP_BASIS_OPERATOR" if validation["status"] == "STOP_BEFORE_HELDOUT"
        else validation["status"]
    )
    decision = {
        "protocol_hash": protocol_hash(),
        "method": "P2 structure-aware hard-positive POD basis operator",
        "status": final_status,
        "families": family_summary,
        "heldout": heldout,
        "claims_not_permitted": [
            "GO_APPROXIMATE_SURROGATE", "GO_BEATS_CN_PSOR", "GO_BEATS_CN_POLICY",
            "GO_EXACT_HYBRID_PSOR", "GO_EXACT_HYBRID_POLICY",
        ],
        "next_method": "positive-premium DeepONet",
        "reason": "POD representation and price mapping passed, but boundary/Greek/LCP structure gates failed on validation",
    }
    output = RESULTS_DIR / "07_synthesis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "method_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(decision)
    _plot_representation()
    return decision


def _write_report(decision: dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    put = decision["families"]["put"]
    call = decision["families"]["call"]
    text = f"""# Positive-Premium Basis Operator 实验结论

## 一句话结论

本方法最终状态为 **{decision['status']}**。POD basis 能很精确地表示价格面，12-mode P2 的五-seed 价格门也通过；但是行权边界、Greeks 或完整 LCP 结构没有达到预先冻结的要求，因此不允许打开 test/stress，也不能声称它优于 CN+PSOR 或 CN+Policy Iteration。

## 主要数据

| 期权族 | 8-mode POD 表示门 | 32-mode oracle 最差 RMSE | 12-mode P2 五-seed最差价格 RMSE | 价格门 | 最终 family 状态 |
|---|---:|---:|---:|---|---|
| American put | 通过 | {put['representation_32_mode_worst_rmse']:.6g} | {put['five_seed_worst_seed_median_price_rmse']:.6g} | {'通过' if put['price_gate_passes'] else '失败'} | {put['status']} |
| dividend American call | 通过 | {call['representation_32_mode_worst_rmse']:.6g} | {call['five_seed_worst_seed_median_price_rmse']:.6g} | {'通过' if call['price_gate_passes'] else '失败'} | {call['status']} |

冻结的价格门是 RMSE ≤ 4.94989e-4。put 与 dividend call 均满足，但完整 gate 的最大超限倍数分别为 {put['max_gate_ratio']:.3f} 和 {call['max_gate_ratio']:.3f}，说明仅凭价格 RMSE 会高估该方法对风险面的质量。

## 这代表什么

1. 失败原因不是“POD basis 不够低秩”：oracle projection 在 8 modes 已过门，32 modes 的误差更低。
2. 小网络确实能把总体价格学到门槛以内，但不能同时可靠重现移动行权边界、Delta/Gamma 与 LCP 互补结构。
3. 因 validation 失败，held-out 文件继续封存；没有运行速度竞争和 exact hybrid。因此不存在“超过 CN+PSOR/Policy”的合规结论。
4. 预注册路线要求停止继续增加 POD modes。下一方法是 positive-premium DeepONet，用更灵活的分支/主干表示检验全局 coefficient map 是否是瓶颈。
"""
    (REPORTS_DIR / "positive_premium_basis_operator_结论_CN.md").write_text(text, encoding="utf-8")


def _plot_representation() -> None:
    """Write a dependency-free SVG (the macOS plotting backend is not stable here)."""

    path = RESULTS_DIR / "03_representation_ceiling/oracle_ceiling_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    width, height = 760, 470
    left, right, top, bottom = 90, 30, 35, 70
    x_min, x_max = 4.0, 32.0
    y_min, y_max = 1e-5, 2e-3
    def x_coordinate(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)
    def y_coordinate(value: float) -> float:
        log_value = np.log10(max(value, y_min))
        return top + (np.log10(y_max) - log_value) / (np.log10(y_max) - np.log10(y_min)) * (height - top - bottom)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
    ]
    for mode in (4, 8, 12, 16, 24, 32):
        x = x_coordinate(mode)
        elements.extend([
            f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+6}" stroke="black"/>',
            f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="13">{mode}</text>',
        ])
    for value, label in ((1e-5, "1e-5"), (1e-4, "1e-4"), (1e-3, "1e-3")):
        y = y_coordinate(value)
        elements.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>',
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="13">{label}</text>',
        ])
    gate_y = y_coordinate(4.94989e-4)
    elements.append(
        f'<line x1="{left}" y1="{gate_y:.1f}" x2="{width-right}" y2="{gate_y:.1f}" stroke="black" stroke-dasharray="6,5"/>'
    )
    for family, color in (("put", "#1f77b4"), ("call", "#d95f02")):
        dimensions = sorted({int(row["modes"]) for row in rows if row["option_type"] == family})
        worst = [
            max(float(row["reduction_price_rmse"]) for row in rows
                if row["option_type"] == family and int(row["modes"]) == mode and row["projection"] == "hard")
            for mode in dimensions
        ]
        points = " ".join(
            f"{x_coordinate(mode):.1f},{y_coordinate(value):.1f}"
            for mode, value in zip(dimensions, worst)
        )
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        elements.extend(
            f'<circle cx="{x_coordinate(mode):.1f}" cy="{y_coordinate(value):.1f}" r="4" fill="{color}"/>'
            for mode, value in zip(dimensions, worst)
        )
    elements.extend([
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="15">POD modes</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {height/2})">Worst validation RMSE vs CN+Policy</text>',
        f'<circle cx="{width-200}" cy="28" r="4" fill="#1f77b4"/><text x="{width-190}" y="33" font-size="13">put</text>',
        f'<circle cx="{width-140}" cy="28" r="4" fill="#d95f02"/><text x="{width-130}" y="33" font-size="13">call</text>',
        f'<line x1="{width-75}" y1="28" x2="{width-50}" y2="28" stroke="black" stroke-dasharray="5,4"/><text x="{width-45}" y="33" font-size="13">gate</text>',
        '</svg>',
    ])
    output = RESULTS_DIR / "07_synthesis/representation_ceiling.svg"
    output.write_text("\n".join(elements), encoding="utf-8")

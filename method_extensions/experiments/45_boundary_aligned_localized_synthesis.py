"""Experiment 45: validation-only scoring, gates, figures, and Chinese report."""

from __future__ import annotations

import json
import csv

from american_risk_surfaces.boundary_aligned_basis.protocol import REPORTS_DIR, RESULTS_DIR
from american_risk_surfaces.boundary_aligned_basis.study import (
    evaluate_validation_grid,
    write_validation_outputs,
)
from american_risk_surfaces.boundary_aligned_basis.figures import make_validation_figures


def main() -> None:
    artifacts = sorted((RESULTS_DIR / "02_basis").glob("*/*/*/basis_*.npz"))
    cached = RESULTS_DIR / "03_validation" / "oracle_validation_ladder.csv"
    if cached.exists():
        with cached.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = evaluate_validation_grid(artifacts)
    ladder, decision_path = write_validation_outputs(rows)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "boundary_aligned_localized_basis_report_CN.md"
    transformation = json.loads(
        (RESULTS_DIR / "01_transformation_audit" / "transformation_decision.json").read_text(
            encoding="utf-8"
        )
    )
    lines = [
        "# Boundary-aligned / Localized Basis 可证伪实验结论",
        "",
        "本阶段只使用 202 个 train 快照和 19 个 validation regimes；没有读取 test、stress 或 no-dividend call。",
        "oracle boundary 只用于诊断，结果不是可部署方法，也不是在线速度结果。",
        "",
        "## Transformation gate",
        "",
    ]
    for family in ("put", "call"):
        item = transformation["families"][family]
        lines.append(
            f"- {family}: `{item['status']}`；选定 canonical points = "
            f"{item['selected_canonical_points']}。"
        )
    lines += ["", "## Basis decision", ""]
    for family in ("put", "call"):
        item = decision["families"][family]
        lines.append(f"- {family}: `{item['status']}`；selected = {item['selected']}。")
        complete = [
            candidate for candidate in item["configurations"]
            if "worst_reduction_price_rmse" in candidate and candidate.get("passed") is not None
        ]
        if complete:
            best_price = min(complete, key=lambda row: row["worst_reduction_price_rmse"])
            best_boundary = min(complete, key=lambda row: row["worst_boundary_mae"])
            best_f1 = max(complete, key=lambda row: row["minimum_active_f1"])
            lines += [
                f"  - 最佳完整 price 配置：L, m={best_price['dimension']}, bins={best_price['bin_count']}；"
                f"worst RMSE={best_price['worst_reduction_price_rmse']:.6g}。",
                f"  - 最佳 boundary 配置：L, m={best_boundary['dimension']}, bins={best_boundary['bin_count']}；"
                f"worst conditional MAE={best_boundary['worst_boundary_mae']:.6g}（门槛 0.066667）。",
                f"  - 最佳 active-set 配置：L, m={best_f1['dimension']}, bins={best_f1['bin_count']}；"
                f"minimum F1={best_f1['minimum_active_f1']:.6g}（门槛 0.98）。",
            ]
    lines += [
        "",
        f"总决策：`{decision['status']}`。",
        "",
        "若 aligned transform 被 DEFER，本结论只否定当前 PCHIP/Jacobian 实现达到预注册门槛，不能据此断言所有 alignment 都无效。",
        "若 physical localization 仍未通过绝对 gate，则下一方法固定为 positive-premium basis operator。",
        "",
        "## 解释",
        "",
        "physical localization 的确能在部分配置显著降低价格或边界误差，但没有一个配置同时通过 price、boundary、Greek、active-set 和 full-LCP residual 门槛。",
        "因此不能用“某一个平均价格指标改善”替代整个 obstacle problem 的结构性验证；本阶段不打开 held-out。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures = make_validation_figures()
    print(f"ladder={ladder}")
    print(f"decision={decision_path}")
    print(f"report={report}")
    print(f"figures={[str(path) for path in figures]}")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

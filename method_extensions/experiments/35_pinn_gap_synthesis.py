"""Experiment 35: synthesize pre-registered Arm C/D/E decisions and next route."""

from __future__ import annotations

import json
from pathlib import Path

from american_risk_surfaces.pinn.protocol import REPORTS_DIR, RESULTS_DIR


def synthesize() -> dict[str, object]:
    decisions = {
        "architecture": _load(RESULTS_DIR / "03_validation_gates" / "arm_c_architecture_decision.json"),
        "ablations": _load(RESULTS_DIR / "03_validation_gates" / "arm_d_ablation_decision.json"),
        "validation_d": _load(RESULTS_DIR / "03_validation_gates" / "arm_d_validation_decision.json"),
        "heldout_d": _load(RESULTS_DIR / "04_heldout" / "arm_d_heldout_decision.json"),
        "arm_e": _load(RESULTS_DIR / "05_arm_e" / "arm_e_decision.json"),
        "tiny_smoke": _load(RESULTS_DIR / "06_tiny_smoke" / "smoke_comparison.json"),
    }
    heldout = decisions["heldout_d"]
    hybrid = decisions["arm_e"]
    if heldout.get("status") == "GO":
        next_route = "GO_PARAMETRIC_ETC_FB_PINN"
        explanation = "Arm D passed accuracy/structure; parametric PINN is allowed even if single-instance speed loses."
    elif heldout.get("status") == "STOP":
        next_route = "STOP_PARAMETRIC_AND_HIGH_DIMENSIONAL_EXTENSION"
        explanation = "Arm D failed the accuracy/structure gate; diagnose before any dimensional expansion."
    else:
        next_route = "DEFER_UNTIL_HELDOUT_COMPLETE"
        explanation = "Formal held-out evidence is incomplete."
    summary = {"decisions": decisions, "next_route": next_route, "explanation": explanation}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pinn_method_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = [
        "# SURF PINN Arm C/D/E implementation report",
        "",
        f"- Arm D validation: **{decisions['validation_d'].get('status', 'DEFER')}**",
        f"- Arm D held-out: **{heldout.get('status', 'DEFER')}**",
        f"- Arm E hybrid: **{hybrid.get('status', 'DEFER')}**",
        f"- Next route: **{next_route}**",
        "",
        explanation,
        "",
        "## Development evidence (not a formal result)",
        "",
        f"- Tiny Arm C median price RMSE: {decisions['tiny_smoke'].get('median_c_rmse', 'not run')}",
        f"- Tiny Arm D median price RMSE: {decisions['tiny_smoke'].get('median_d_rmse', 'not run')}",
        f"- Evidence scope: {decisions['tiny_smoke'].get('evidence_scope', 'not run')}",
        "",
        "Training time, steady-state online time, first-query cost, and failed seeds must be read together; no best-seed result is used.",
    ]
    (REPORTS_DIR / "implementation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "DEFER"}


if __name__ == "__main__":
    print(json.dumps(synthesize(), indent=2, sort_keys=True))

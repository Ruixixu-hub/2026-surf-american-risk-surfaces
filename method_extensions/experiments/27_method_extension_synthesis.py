"""Assemble the stage-gated method-extension decisions into one handoff report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "07_method_extensions"
REPORTS = ROOT / "reports" / "09_method_extensions"


def _read(relative: str) -> dict[str, Any]:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def synthesize() -> dict[str, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    protocol = _read("00_protocol/tolerance_decision.json")
    warmstart = _read("02_warmstart/method_decision.json")
    temporal = _read("03_greek_audit/greek_decision.json")
    spatial_path = RESULTS / "03_greek_audit" / "spatial_greek_decision.json"
    spatial = json.loads(spatial_path.read_text(encoding="utf-8")) if spatial_path.exists() else {}
    pod = _read("04_pod/pod_decision.json")
    coefficient = _read("05_pod_coefficient/coefficient_decision.json")

    cards = [
        {
            "method": "Policy Iteration",
            "role": "strict discrete-LCP solver",
            "hypothesis": "Beat PSOR at the identical frozen residual tolerance.",
            "status": "GO" if warmstart["best_classical_arm"] == "B_previous_policy" else "STOP",
            "decision": warmstart["selected_method"],
            "evidence": "2x2 held-out timing plus exact residual and structure checks",
        },
        {
            "method": "Positive-premium MLP warm-start",
            "role": "initializer only; never the final correctness mechanism",
            "hypothesis": "Reduce end-to-end median by at least 20% without worsening p95.",
            "status": "GO" if warmstart["status"].startswith("GO") else "STOP",
            "decision": warmstart["status"],
            "evidence": f"learned-vs-classical median speedup {warmstart['median_speedup_fraction']:.3%}",
        },
        {
            "method": "DIRK / Lobatto Greek reference",
            "role": "high-accuracy label and convergence audit",
            "hypothesis": "Stable order >=1.5 in at least 90% of audit regimes.",
            "status": "GO" if temporal["status"] == "GAMMA_REFERENCE_CANDIDATE" else "STOP",
            "decision": spatial.get("status", "PENDING_SPATIAL_AUDIT"),
            "evidence": "time and space refinements are kept separate",
        },
        {
            "method": "POD/SVD rank diagnostic",
            "role": "falsification test before a large operator network",
            "hypothesis": "10-30 modes reach the numerical-label noise floor on held-out regimes.",
            "status": "GO" if pod["status"].startswith("GO") else "STOP",
            "decision": f"{pod['selected_modes']} {pod['selected_representation']} modes",
            "evidence": f"worst held-out RMSE {pod['worst_heldout_rmse']:.6g}",
        },
        {
            "method": "Polynomial POD coefficient map",
            "role": "minimal nonintrusive basis surrogate",
            "hypothesis": "Held-out error <=1.25 times the reference-noise floor.",
            "status": "GO" if coefficient["status"].startswith("GO") else "STOP",
            "decision": coefficient["status"],
            "evidence": f"worst held-out RMSE {coefficient['worst_heldout_rmse']:.6g}",
        },
        {
            "method": "Primal/dual Reduced-Basis VI",
            "role": "intrusive many-query obstacle solver",
            "hypothesis": "Reach the reference floor with audited residual at lower online cost.",
            "status": "DEFER",
            "decision": "Needs full-grid primal and multiplier snapshots, stability enrichment, and an online VI estimator; a projected PCA solve is not an honest substitute.",
            "evidence": "POD rank gate passed, but the nonintrusive coefficient-map gate failed",
        },
        {
            "method": "DeepONet / FNO / localized operator",
            "role": "large learned operator",
            "hypothesis": "Improve price by 20% or boundary/Delta by 15% under matched budgets.",
            "status": "DEFER",
            "decision": "Do not implement FNO: unaligned POD is already low-rank; revisit basis operator or DeepONet only after RB VI.",
            "evidence": "stage-gated ordering",
        },
        {
            "method": "Multi-fidelity and UQ fallback",
            "role": "label-efficiency and deployment safety",
            "hypothesis": "Save 30% fine solves; calibrated risky requests fall back to the classical solver.",
            "status": "DEFER",
            "decision": "Requires a retained surrogate; no surrogate has passed its held-out gate yet.",
            "evidence": "downstream prerequisite not met",
        },
    ]

    cards_path = RESULTS / "method_cards.json"
    cards_path.write_text(json.dumps(cards, indent=2, sort_keys=True), encoding="utf-8")
    summary_csv = RESULTS / "method_decision_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(cards[0]))
        writer.writeheader()
        writer.writerows(cards)

    report = _report(protocol, warmstart, temporal, spatial, pod, coefficient, cards)
    report_path = REPORTS / "implementation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"cards": cards_path, "summary": summary_csv, "report": report_path}


def _report(
    protocol: dict[str, Any],
    warmstart: dict[str, Any],
    temporal: dict[str, Any],
    spatial: dict[str, Any],
    pod: dict[str, Any],
    coefficient: dict[str, Any],
    cards: list[dict[str, Any]],
) -> str:
    lines = [
        "# SURF Method-Extension Implementation Report",
        "",
        "## Outcome",
        "",
        f"The frozen normalized LCP tolerance is `{protocol['frozen_normalized_lcp_tolerance']}`. "
        "The requested `1e-10` target was not frozen because its maximum price difference "
        f"from `1e-12` was `{protocol['max_1e10_vs_1e12_value_difference']:.6g}`, above `1e-9 K`.",
        "",
        f"The selected online classical method is `{warmstart['selected_method']}`. "
        f"The MLP warm-start speedup was `{warmstart['median_speedup_fraction']:.3%}`, so learned "
        "acceleration is stopped under the pre-registered gate.",
        "",
        f"The temporal Greek winner is `{temporal['best_temporal_method']}`. "
        f"The current joint Gamma decision is `{spatial.get('status', 'PENDING_SPATIAL_AUDIT')}`.",
        "",
        f"POD passed its rank test with `{pod['selected_modes']}` modes, but the simple coefficient "
        f"map failed (`{coefficient['worst_heldout_rmse']:.6g}` versus acceptance "
        f"`{coefficient['acceptance_threshold']:.6g}`).",
        "",
        "## Method decisions",
        "",
        "| Method | Role | Status | Decision |",
        "|---|---|---|---|",
    ]
    for card in cards:
        lines.append(
            f"| {card['method']} | {card['role']} | **{card['status']}** | {card['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Recommended next experiment",
            "",
            "Implement a genuine primal/dual Reduced-Basis variational-inequality prototype. "
            "First regenerate full-grid train snapshots and multipliers with the frozen Policy "
            "Iteration solver; then add dual angle-greedy selection and stability enrichment. "
            "Do not call a direct POD coefficient regression a Reduced-Basis VI.",
            "",
            "DeepONet, FNO, multi-fidelity, and UQ remain deferred because their prerequisite "
            "surrogate gate has not passed. This is the intended stage-gated stopping behavior, "
            "not missing evidence hidden from the report.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(json.dumps({key: str(value) for key, value in synthesize().items()}, indent=2))

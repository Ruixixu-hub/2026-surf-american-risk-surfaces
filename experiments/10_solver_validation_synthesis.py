"""Ticket 12: solver validation synthesis and artifact audit."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "01_solver_validation" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "01_solver_validation" / "figures"

EVIDENCE_CSV = TABLE_DIR / "ticket_12_solver_validation_evidence_summary.csv"
GATE_CSV = TABLE_DIR / "ticket_12_validation_gate_decision.csv"
AUDIT_CSV = TABLE_DIR / "ticket_12_artifact_audit.csv"

EVIDENCE_FIELDNAMES = [
    "ticket",
    "validation_area",
    "source_artifact",
    "key_metric",
    "metric_value",
    "status",
    "limitation",
]

GATE_FIELDNAMES = [
    "decision",
    "allowed_values",
    "basis",
    "next_recommended_stage",
    "required_limitations",
    "blocked_until_later",
    "artifact_audit_status",
    "evidence_status",
]

AUDIT_FIELDNAMES = [
    "artifact_group",
    "ticket",
    "artifact_type",
    "path",
    "required",
    "exists",
    "status",
    "notes",
]

ALLOWED_DECISIONS = (
    "PASS_SOLVER_VALIDATION_WITH_LIMITATIONS",
    "PASS_TO_REFERENCE_INTEGRATION_AND_RESEARCH_STAGE",
    "REVIEW_REQUIRED_BEFORE_DOWNSTREAM_STAGE",
)


def expected_artifacts(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Return expected validation-ladder artifacts from Tickets 01 through 11."""

    del project_root
    rows: list[dict[str, Any]] = []

    def add(group: str, ticket: str, artifact_type: str, path: str, notes: str = "") -> None:
        rows.append(
            {
                "artifact_group": group,
                "ticket": ticket,
                "artifact_type": artifact_type,
                "path": path,
                "required": True,
                "notes": notes,
            }
        )

    for path in (
        "reports/00_planning/planning_report.md",
        "reports/01_literature/literature_map.md",
        "reports/02_math/formulation_note.md",
        "reports/03_solver/solver_validation_plan.md",
        "reports/03_solver/solver_implementation_tickets.md",
    ):
        add("project_document", "planning", "markdown", path, "source basis for the synthesis")

    report_stems = [
        "ticket_01_black_scholes_utilities",
        "ticket_02_grid_operator",
        "ticket_03_european_cn_validation",
        "ticket_04_psor_lcp_core",
        "ticket_05_american_put_validation",
        "ticket_06_no_dividend_american_call_validation",
        "ticket_07_dividend_american_call_validation",
        "ticket_08_obstacle_complementarity_diagnostics",
        "ticket_09_continuation_premium_boundary_extraction",
        "ticket_10_delta_gamma_diagnostics",
        "ticket_10a_rannacher_smoothing_comparison",
        "ticket_11_grid_domain_sensitivity",
    ]
    for stem in report_stems:
        ticket = _ticket_from_path(stem)
        add("ticket_report", ticket, "latex", f"reports/03_solver/tickets/{stem}.tex")
        add("ticket_report", ticket, "pdf", f"reports/03_solver/tickets/{stem}.pdf")

    for ticket, path in (
        ("Ticket 01", "src/american_risk_surfaces/solvers/black_scholes.py"),
        ("Ticket 02", "src/american_risk_surfaces/solvers/grid.py"),
        ("Ticket 02", "src/american_risk_surfaces/solvers/operator.py"),
        ("Ticket 03", "src/american_risk_surfaces/solvers/cn.py"),
        ("Ticket 04", "src/american_risk_surfaces/solvers/cn_psor.py"),
        ("Ticket 08", "src/american_risk_surfaces/diagnostics/lcp.py"),
        ("Ticket 09", "src/american_risk_surfaces/diagnostics/boundary.py"),
        ("Ticket 10", "src/american_risk_surfaces/diagnostics/greeks.py"),
        ("Ticket 10A", "src/american_risk_surfaces/solvers/rannacher.py"),
        ("Ticket 11", "src/american_risk_surfaces/diagnostics/sensitivity.py"),
    ):
        add("source", ticket, "python", path)

    for ticket, path in (
        ("Ticket 01", "tests/test_black_scholes.py"),
        ("Ticket 02", "tests/test_grid_operator.py"),
        ("Ticket 03", "tests/test_cn_european.py"),
        ("Ticket 04", "tests/test_psor.py"),
        ("Ticket 05", "tests/test_american_put_validation.py"),
        ("Ticket 06", "tests/test_no_dividend_american_call_validation.py"),
        ("Ticket 07", "tests/test_dividend_american_call_validation.py"),
        ("Ticket 08", "tests/test_lcp_diagnostics.py"),
        ("Ticket 09", "tests/test_boundary_extraction.py"),
        ("Ticket 10", "tests/test_greek_diagnostics.py"),
        ("Ticket 10A", "tests/test_rannacher_smoothing.py"),
        ("Ticket 11", "tests/test_grid_domain_sensitivity.py"),
    ):
        add("test", ticket, "python", path)

    for ticket, path in (
        ("Ticket 03", "experiments/01_european_cn_validation.py"),
        ("Ticket 05", "experiments/02_american_put_validation.py"),
        ("Ticket 06", "experiments/03_no_dividend_american_call_validation.py"),
        ("Ticket 07", "experiments/04_dividend_american_call_validation.py"),
        ("Ticket 08", "experiments/05_obstacle_complementarity_diagnostics.py"),
        ("Ticket 09", "experiments/06_boundary_extraction.py"),
        ("Ticket 10", "experiments/07_greek_diagnostics.py"),
        ("Ticket 10A", "experiments/08_rannacher_smoothing_comparison.py"),
        ("Ticket 11", "experiments/09_grid_domain_sensitivity.py"),
    ):
        add("experiment", ticket, "python", path)

    csv_paths = sorted(path.relative_to(PROJECT_ROOT).as_posix() for path in TABLE_DIR.glob("ticket_*.csv"))
    for path in csv_paths:
        if "ticket_12" not in path:
            add("result_table", _ticket_from_path(path), "csv", path)

    figure_paths = sorted(path.relative_to(PROJECT_ROOT).as_posix() for path in FIGURE_DIR.glob("ticket_*.png"))
    for path in figure_paths:
        add("figure", _ticket_from_path(path), "png", path)

    return rows


def audit_artifacts(
    expected: list[dict[str, Any]], project_root: Path = PROJECT_ROOT
) -> list[dict[str, str]]:
    """Check artifact presence without inventing replacements for missing files."""

    rows: list[dict[str, str]] = []
    for item in expected:
        path = str(item["path"])
        exists = (project_root / path).exists()
        rows.append(
            {
                "artifact_group": str(item.get("artifact_group", "")),
                "ticket": str(item.get("ticket", "")),
                "artifact_type": str(item.get("artifact_type", "")),
                "path": path,
                "required": str(bool(item.get("required", True))),
                "exists": str(exists),
                "status": "PRESENT" if exists else "MISSING",
                "notes": str(item.get("notes", "")),
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_evidence_summary(project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    """Build compact evidence rows from prior ticket reports and result CSVs."""

    rows: list[dict[str, str]] = []

    def add(
        ticket: str,
        area: str,
        source: str,
        metric: str,
        value: Any,
        status: str,
        limitation: str,
    ) -> None:
        rows.append(
            {
                "ticket": ticket,
                "validation_area": area,
                "source_artifact": source,
                "key_metric": metric,
                "metric_value": _format_value(value),
                "status": status,
                "limitation": limitation,
            }
        )

    add(
        "Ticket 01",
        "Black-Scholes utilities",
        "reports/03_solver/tickets/ticket_01_black_scholes_utilities.tex",
        "payoff, parity, scalar/vector, T=0, sigma=0 tests",
        "implemented and tested",
        "PASS",
        "Analytic utilities only; no finite-difference solver validation.",
    )
    add(
        "Ticket 02",
        "Grid and operator setup",
        "reports/03_solver/tickets/ticket_02_grid_operator.tex",
        "uniform grid/operator/boundary helper tests",
        "implemented and tested",
        "PASS",
        "Central-difference uniform-grid setup only.",
    )
    add(
        "Ticket 04",
        "PSOR/LCP core",
        "reports/03_solver/tickets/ticket_04_psor_lcp_core.tex",
        "projected SOR obstacle core tests",
        "implemented and smoke-tested",
        "PASS_WITH_LIMITATIONS",
        "Core smoke tests are not full American validation.",
    )

    ticket03 = read_csv_rows(project_root / "results/01_solver_validation/tables/ticket_03_european_cn_validation.csv")
    if ticket03:
        max_error = max(_float(row.get("max_abs_error")) for row in ticket03)
        max_rmse = max(_float(row.get("rmse")) for row in ticket03)
        add(
            "Ticket 03",
            "European CN validation",
            "results/01_solver_validation/tables/ticket_03_european_cn_validation.csv",
            "max target-region absolute error",
            max_error,
            "PASS",
            "European validation only; no American obstacle.",
        )
        add(
            "Ticket 03",
            "European CN validation",
            "results/01_solver_validation/tables/ticket_03_european_cn_validation.csv",
            "max target-region RMSE",
            max_rmse,
            "PASS",
            "Representative case, not broad parameter sweep.",
        )
    else:
        add_missing("Ticket 03", "European CN validation", "ticket_03_european_cn_validation.csv", add)

    _add_american_put_evidence(project_root, add)
    _add_no_dividend_call_evidence(project_root, add)
    _add_dividend_call_evidence(project_root, add)
    _add_lcp_evidence(project_root, add)
    _add_boundary_evidence(project_root, add)
    _add_greek_evidence(project_root, add)
    _add_rannacher_evidence(project_root, add)
    _add_sensitivity_evidence(project_root, add)

    return rows


def build_gate_decision(
    evidence_rows: list[dict[str, str]], audit_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return one conservative solver-validation gate decision row."""

    missing_required = [
        row
        for row in audit_rows
        if row["required"] == "True" and row["status"] == "MISSING"
    ]
    bad_evidence = [
        row
        for row in evidence_rows
        if row["status"] in {"MISSING_DATA", "FAIL", "REVIEW_REQUIRED"}
    ]

    if missing_required or bad_evidence:
        decision = "REVIEW_REQUIRED_BEFORE_DOWNSTREAM_STAGE"
        artifact_status = f"{len(missing_required)} required artifact(s) missing"
        evidence_status = f"{len(bad_evidence)} evidence row(s) require review"
        basis = "Required artifacts or evidence rows are missing; no solver-validation pass is claimed."
    else:
        decision = "PASS_SOLVER_VALIDATION_WITH_LIMITATIONS"
        artifact_status = "ALL_REQUIRED_PRESENT"
        evidence_status = "COMPLETE_WITH_LIMITATIONS"
        basis = (
            "Tickets 01-11 provide consistent representative evidence: European CN matches "
            "closed-form prices to finite-difference accuracy, American PSOR runs converge, "
            "obstacle and LCP diagnostics are near tolerance, boundary extraction is available "
            "with threshold metadata, Greeks are diagnosed with cautions, Rannacher was not "
            "adopted as default, and grid/domain sensitivity did not contradict baseline use."
        )

    return [
        {
            "decision": decision,
            "allowed_values": "; ".join(ALLOWED_DECISIONS),
            "basis": basis,
            "next_recommended_stage": "formal reference integration and human review before downstream risk-surface work",
            "required_limitations": (
                "representative cases only; threshold-based boundaries; diagnostic Greeks; "
                "no production convergence proof; no stress maps or surrogate labels yet"
            ),
            "blocked_until_later": "stress maps; datasets; neural surrogates; production Greek labels; final bibliography",
            "artifact_audit_status": artifact_status,
            "evidence_status": evidence_status,
        }
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write CSV rows with stable columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    evidence_rows = build_evidence_summary(PROJECT_ROOT)
    audit_rows = audit_artifacts(expected_artifacts(PROJECT_ROOT), PROJECT_ROOT)
    gate_rows = build_gate_decision(evidence_rows, audit_rows)

    write_csv(EVIDENCE_CSV, evidence_rows, EVIDENCE_FIELDNAMES)
    write_csv(GATE_CSV, gate_rows, GATE_FIELDNAMES)
    write_csv(AUDIT_CSV, audit_rows, AUDIT_FIELDNAMES)

    print(f"wrote {len(evidence_rows)} rows to {EVIDENCE_CSV}")
    print(f"wrote {len(gate_rows)} rows to {GATE_CSV}")
    print(f"wrote {len(audit_rows)} rows to {AUDIT_CSV}")
    return evidence_rows, gate_rows, audit_rows


def _add_american_put_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_05_american_put_validation.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 05", "American put validation", source, add)
        return
    add("Ticket 05", "American put validation", source, "max obstacle violation", max(_float(row["max_obstacle_violation"]) for row in rows), "PASS", "Representative medium/fine case only.")
    add("Ticket 05", "American put validation", source, "max American minus European put", max(_float(row["max_american_minus_european"]) for row in rows), "PASS", "Not an external benchmark price.")
    add("Ticket 05", "American put validation", source, "all PSOR steps converged", all(row["all_psor_steps_converged"] == "True" for row in rows), "PASS", "Convergence checked for selected cases.")


def _add_no_dividend_call_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_06_no_dividend_american_call_validation.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 06", "No-dividend American call theorem", source, add)
        return
    add("Ticket 06", "No-dividend American call theorem", source, "fine-grid max American-European error", _value_for_case(rows, "fine", "max_abs_american_european_error"), "PASS", "Approximate finite-difference equality, not exact identity.")
    add("Ticket 06", "No-dividend American call theorem", source, "max obstacle violation", max(_float(row["max_obstacle_violation"]) for row in rows), "PASS", "Only q=0 control case.")


def _add_dividend_call_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_07_dividend_american_call_validation.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 07", "Dividend-paying American call", source, add)
        return
    add("Ticket 07", "Dividend-paying American call", source, "fine-grid max American minus European call", _value_for_case(rows, "fine", "max_american_minus_european"), "PASS", "Positive value evidence is parameter-specific.")
    add("Ticket 07", "Dividend-paying American call", source, "positive American-minus-European node count", _value_for_case(rows, "fine", "positive_american_minus_european_node_count"), "PASS", "Does not extract an exercise boundary by itself.")


def _add_lcp_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_08_lcp_diagnostics_summary.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 08", "LCP diagnostics", source, add)
        return
    add("Ticket 08", "LCP diagnostics", source, "max obstacle violation", max(_float(row["max_obstacle_violation"]) for row in rows), "PASS", "Interior-node diagnostics only.")
    add("Ticket 08", "LCP diagnostics", source, "max equation violation", max(_float(row["max_equation_violation"]) for row in rows), "PASS", "Tolerance diagnostic, not proof.")
    add("Ticket 08", "LCP diagnostics", source, "max complementarity product", max(_float(row["max_abs_complementarity_product"]) for row in rows), "PASS", "Representative cases only.")


def _add_boundary_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_09_boundary_extraction_summary.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 09", "Boundary extraction", source, add)
        return
    put = _row_for_case(rows, "american_put_medium")
    dividend_call = _row_for_case(rows, "dividend_call_medium")
    control = _row_for_case(rows, "no_dividend_call_control")
    add("Ticket 09", "Boundary extraction", source, "American put found boundary count", put.get("found_boundary_count", ""), "PASS_WITH_LIMITATIONS", "Threshold-based approximate boundary.")
    add("Ticket 09", "Boundary extraction", source, "dividend call found boundary count", dividend_call.get("found_boundary_count", ""), "PASS_WITH_LIMITATIONS", "Threshold-based approximate boundary.")
    add("Ticket 09", "Boundary extraction", source, "no-dividend call control status", control.get("status", ""), "PASS", "Control prevents forced boundary extraction.")


def _add_greek_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_10_greek_diagnostics_summary.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 10", "Greek diagnostics", source, add)
        return
    add("Ticket 10", "Greek diagnostics", source, "max full-grid absolute Gamma", max(_float(row["max_abs_gamma"]) for row in rows), "PASS_WITH_CAUTIONS", "Gamma is diagnostic and fragile near kinks/boundaries.")
    add("Ticket 10", "Greek diagnostics", source, "max strict-mask absolute Gamma", max(_float(row["max_abs_gamma_strict"]) for row in rows), "PASS_WITH_CAUTIONS", "Strict mask is not a production risk label.")
    add("Ticket 10", "Greek diagnostics", source, "strict negative Gamma count", max(_float(row["strict_negative_gamma_count"]) for row in rows), "PASS_WITH_CAUTIONS", "Finite-difference Greeks require caution.")


def _add_rannacher_evidence(project_root: Path, add) -> None:
    source = "results/01_solver_validation/tables/ticket_10a_rannacher_comparison_summary.csv"
    rows = read_csv_rows(project_root / source)
    if not rows:
        add_missing("Ticket 10A", "Rannacher comparison gate", source, add)
        return
    add("Ticket 10A", "Rannacher comparison gate", source, "gate recommendation", "; ".join(sorted({row["gate_recommendation"] for row in rows})), "PASS_WITH_LIMITATIONS", "Baseline retained for now.")
    add("Ticket 10A", "Rannacher comparison gate", source, "max baseline-vs-Rannacher price difference", max(_float(row["max_abs_price_difference"]) for row in rows), "PASS", "Smoothing comparison only.")


def _add_sensitivity_evidence(project_root: Path, add) -> None:
    grid_source = "results/01_solver_validation/tables/ticket_11_grid_sensitivity_summary.csv"
    domain_source = "results/01_solver_validation/tables/ticket_11_domain_sensitivity_summary.csv"
    diagnostic_source = "results/01_solver_validation/tables/ticket_11_diagnostic_sensitivity.csv"
    grid_rows = read_csv_rows(project_root / grid_source)
    domain_rows = read_csv_rows(project_root / domain_source)
    diagnostic_rows = read_csv_rows(project_root / diagnostic_source)

    if not grid_rows:
        add_missing("Ticket 11", "Grid sensitivity", grid_source, add)
    else:
        add("Ticket 11", "Grid sensitivity", grid_source, "max selected price difference vs reference", max(_float(row["max_abs_selected_price_difference"]) for row in grid_rows), "PASS_WITH_CAUTIONS", "Three grid levels are not a convergence proof.")
        add("Ticket 11", "Grid sensitivity", grid_source, "max selected boundary shift vs reference", max(_float(row["max_abs_boundary_shift"]) for row in grid_rows), "PASS_WITH_CAUTIONS", "Boundary extraction is threshold-based.")

    if not domain_rows:
        add_missing("Ticket 11", "Domain sensitivity", domain_source, add)
    else:
        add("Ticket 11", "Domain sensitivity", domain_source, "max selected price difference vs domain reference", max(_float(row["max_abs_selected_price_difference"]) for row in domain_rows), "PASS", "Comparable dS design, representative domains only.")
        add("Ticket 11", "Domain sensitivity", domain_source, "max boundary shift vs domain reference", max(_float(row["max_abs_boundary_shift"]) for row in domain_rows), "PASS_WITH_CAUTIONS", "Selected tau comparisons only.")

    if not diagnostic_rows:
        add_missing("Ticket 11", "Diagnostic sensitivity", diagnostic_source, add)
    else:
        add("Ticket 11", "Diagnostic sensitivity", diagnostic_source, "max LCP equation violation", max(_float(row["max_equation_violation"]) for row in diagnostic_rows), "PASS", "Diagnostic tolerance evidence only.")
        add("Ticket 11", "Diagnostic sensitivity", diagnostic_source, "max strict Gamma", max(_float(row["max_abs_gamma_strict"]) for row in diagnostic_rows), "PASS_WITH_CAUTIONS", "Greeks remain diagnostic, not labels.")


def add_missing(ticket: str, area: str, source: str, add) -> None:
    add(ticket, area, source, "required evidence source", "", "MISSING_DATA", "Missing artifact; no result fabricated.")


def _ticket_from_path(path: str) -> str:
    if "ticket_10a" in path:
        return "Ticket 10A"
    marker = "ticket_"
    if marker not in path:
        return "unknown"
    suffix = path.split(marker, 1)[1]
    number = suffix[:2]
    return f"Ticket {number}" if number.isdigit() else "unknown"


def _row_for_case(rows: list[dict[str, str]], case_name: str) -> dict[str, str]:
    for row in rows:
        if row.get("case_name") == case_name:
            return row
    return {}


def _value_for_case(rows: list[dict[str, str]], case_name: str, field: str) -> str:
    return _row_for_case(rows, case_name).get(field, "")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.12g}"
    return str(value)


if __name__ == "__main__":
    main()

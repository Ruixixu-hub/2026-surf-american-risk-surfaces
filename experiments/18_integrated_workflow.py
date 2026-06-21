"""Stage 6: integrated workflow and claim synthesis."""

from __future__ import annotations

import base64
import csv
import math
from pathlib import Path
from typing import NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "07_integrated"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "06_integrated_workflow"

REQUIRED_INPUT_CSVS = (
    Path("results/04_surrogate_dataset/v1_small_grid/v1_dataset_quality_summary.csv"),
    Path("results/05_surrogate_models/price_premium/surrogate_metrics_by_split.csv"),
    Path("results/05_surrogate_models/price_premium/surrogate_metrics_by_region.csv"),
    Path("results/05_surrogate_models/price_premium/obstacle_violation_summary.csv"),
    Path("results/05_surrogate_models/price_premium/model_run_manifest.csv"),
    Path("results/05_surrogate_models/boundary/boundary_metrics_by_split.csv"),
    Path("results/05_surrogate_models/boundary/boundary_metrics_by_option_type.csv"),
    Path("results/05_surrogate_models/boundary/boundary_metrics_by_regime.csv"),
    Path("results/05_surrogate_models/boundary/boundary_model_manifest.csv"),
    Path("results/05_surrogate_models/delta/delta_metrics_by_split.csv"),
    Path("results/05_surrogate_models/delta/delta_metrics_by_option_type.csv"),
    Path("results/05_surrogate_models/delta/delta_metrics_by_region.csv"),
    Path("results/05_surrogate_models/delta/delta_bounds_violation_summary.csv"),
    Path("results/05_surrogate_models/delta/delta_model_manifest.csv"),
)

REQUIRED_INPUT_REPORTS = (
    Path("reports/03_solver/tickets/ticket_12_solver_validation_synthesis.tex"),
    Path("reports/04_downstream/application_strength_gate.tex"),
    Path("reports/05_surrogate/v1_small_grid_dataset_report.tex"),
    Path("reports/06_surrogate/price_premium_surrogate_report.tex"),
    Path("reports/06_surrogate/boundary_diagnostic_report.tex"),
    Path("reports/06_surrogate/delta_diagnostic_report.tex"),
    Path("docs/student_handout_full_report.pdf"),
    Path("docs/Student_Methodology_FreeBoundary_Risk_Surfaces.pdf"),
)

COMPONENT_READINESS_FIELDNAMES = [
    "component",
    "input_artifact",
    "method",
    "key_metric",
    "status",
    "supported_use",
    "limitations",
]
CLAIM_EVIDENCE_FIELDNAMES = [
    "claim_id",
    "claim_text",
    "supporting_evidence",
    "required_citation_or_artifact",
    "status",
    "final_paper_use",
]
BLOCKED_CLAIMS_FIELDNAMES = [
    "blocked_claim",
    "reason_blocked",
    "evidence_gap",
    "possible_future_work",
]
FIGURE_TABLE_INVENTORY_FIELDNAMES = [
    "item_id",
    "item_type",
    "source_file",
    "recommended_final_paper_section",
    "purpose",
    "status",
]
FINAL_PAPER_SECTION_FIELDNAMES = [
    "section_number",
    "section_title",
    "purpose",
    "key_evidence",
    "figures_or_tables",
    "writing_status",
]
INTEGRATED_METRICS_FIELDNAMES = [
    "component",
    "metric_name",
    "split_or_region",
    "value",
    "interpretation",
    "source_file",
]
WORKFLOW_MANIFEST_FIELDNAMES = [
    "artifact",
    "artifact_type",
    "source_path",
    "included_in_stage6",
    "notes",
]

ALLOWED_DECISIONS = {
    "READY_FOR_FINAL_PAPER_DRAFTING",
    "REVIEW_REQUIRED_BEFORE_FINAL_PAPER",
}

_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class IntegratedWorkflowResult(NamedTuple):
    report_tex_path: Path
    final_paper_outline_path: Path
    report_csv_paths: tuple[Path, ...]
    result_csv_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    manifest_path: Path
    review_decision: str


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{path} has row-length drift at row {row_number}: {row[None]!r}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_component_readiness_matrix(project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    quality = _rows_by_metric(read_csv_rows(project_root / REQUIRED_INPUT_CSVS[0]))
    stage3_split = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[1])
    stage3_obstacle = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[3])
    stage4_split = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[5])
    stage5_split = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[9])
    stage5_bounds = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[12])

    pos_test = _find_row(stage3_split, model_name="positive_premium_mlp", split="test")
    pos_stress = _find_row(stage3_split, model_name="positive_premium_mlp", split="stress_holdout")
    pos_obstacle = _find_row(stage3_obstacle, model_name="positive_premium_mlp", split="test")
    boundary_test = _find_row(stage4_split, method_name="direct_boundary_head", split="test")
    boundary_stress = _find_row(stage4_split, method_name="direct_boundary_head", split="stress_holdout")
    delta_test = _find_row(stage5_split, method_name="supervised_delta_head", split="test")
    delta_stress = _find_row(stage5_split, method_name="supervised_delta_head", split="stress_holdout")
    delta_bounds = _find_row(stage5_bounds, method_name="supervised_delta_head", split="test")

    return [
        {
            "component": "CN/PSOR solver benchmark",
            "input_artifact": "reports/03_solver/tickets/ticket_12_solver_validation_synthesis.tex",
            "method": "Baseline American Crank-Nicolson / PSOR LCP solver",
            "key_metric": "PASS_SOLVER_VALIDATION_WITH_LIMITATIONS",
            "status": "READY_WITH_LIMITATIONS",
            "supported_use": "Finite-difference benchmark for one-dimensional Black-Scholes American option diagnostics.",
            "limitations": "Representative validation ladder, not production certification or analytical proof.",
        },
        {
            "component": "v1 small-grid dataset",
            "input_artifact": "results/04_surrogate_dataset/v1_small_grid/v1_dataset_quality_summary.csv",
            "method": "288 approved regimes sampled on S/K in [0.4,1.8]",
            "key_metric": (
                f"{quality['accepted_regimes']['metric_value']} accepted regimes; "
                f"{quality['accepted_rows']['metric_value']} rows; max obstacle "
                f"{quality['max_obstacle_violation']['metric_value']}"
            ),
            "status": "READY_WITH_LIMITATIONS",
            "supported_use": "Research surrogate diagnostics on the approved v1 regime grid.",
            "limitations": "Small-grid research dataset, not production data and not broad extrapolation evidence.",
        },
        {
            "component": "price/premium surrogate",
            "input_artifact": "results/05_surrogate_models/price_premium/surrogate_metrics_by_split.csv",
            "method": "Positive-premium MLP with analytic payoff reconstruction",
            "key_metric": (
                f"test value RMSE {_fmt(pos_test['value_rmse'])}; stress RMSE {_fmt(pos_stress['value_rmse'])}; "
                f"test obstacle rate {_fmt(pos_obstacle['obstacle_violation_rate'])}"
            ),
            "status": "READY_WITH_LIMITATIONS",
            "supported_use": "Obstacle-aligned price and continuation-premium research surrogate.",
            "limitations": "Does not by itself provide reliable boundary or Greek behavior.",
        },
        {
            "component": "boundary diagnostic component",
            "input_artifact": "results/05_surrogate_models/boundary/boundary_metrics_by_split.csv",
            "method": "Direct boundary-head diagnostic baseline",
            "key_metric": f"test RMSE {_fmt(boundary_test['boundary_rmse'])}; stress RMSE {_fmt(boundary_stress['boundary_rmse'])}",
            "status": "READY_WITH_LIMITATIONS",
            "supported_use": "Threshold-label boundary diagnostics inside the tested envelope.",
            "limitations": "Premium-implied boundary was not reliable; labels are threshold-based, not exact free-boundary truth.",
        },
        {
            "component": "Delta diagnostic component",
            "input_artifact": "results/05_surrogate_models/delta/delta_metrics_by_split.csv",
            "method": "Supervised bounded Delta-head diagnostic baseline",
            "key_metric": (
                f"test RMSE {_fmt(delta_test['delta_rmse'])}; stress RMSE {_fmt(delta_stress['delta_rmse'])}; "
                f"test bounds rate {_fmt(delta_bounds['bounds_violation_rate'])}"
            ),
            "status": "READY_WITH_LIMITATIONS",
            "supported_use": "Finite-difference Delta diagnostic head with mask and bounds checks.",
            "limitations": "Delta labels are numerical diagnostics, not production Greeks.",
        },
        {
            "component": "Gamma component",
            "input_artifact": "results/04_surrogate_dataset/v1_small_grid/schema_snapshot.csv",
            "method": "No Gamma head trained",
            "key_metric": "Gamma remains blocked",
            "status": "BLOCKED",
            "supported_use": "Diagnostic masks and cautions only.",
            "limitations": "No Gamma surrogate or production Gamma claim is supported.",
        },
    ]


def build_claim_evidence_matrix() -> list[dict[str, str]]:
    return [
        {
            "claim_id": "C01",
            "claim_text": "A validated baseline CN/PSOR benchmark can generate consistent American-option diagnostic surfaces within the approved one-dimensional Black-Scholes grid.",
            "supporting_evidence": "Ticket 12 gate plus v1 dataset LCP diagnostics: 288 accepted regimes, zero max obstacle violation.",
            "required_citation_or_artifact": "ticket_12_solver_validation_synthesis.tex; v1_dataset_quality_summary.csv",
            "status": "SUPPORTED_WITH_LIMITATIONS",
            "final_paper_use": "Benchmark and dataset construction sections.",
        },
        {
            "claim_id": "C02",
            "claim_text": "Positive continuation-premium prediction is better aligned with the American payoff obstacle than direct value prediction.",
            "supporting_evidence": "Stage 3 positive-premium model has lower held-out value RMSE and zero obstacle and negative-premium rates.",
            "required_citation_or_artifact": "surrogate_metrics_by_split.csv; obstacle_violation_summary.csv",
            "status": "SUPPORTED_WITH_LIMITATIONS",
            "final_paper_use": "Price / premium surrogate section.",
        },
        {
            "claim_id": "C03",
            "claim_text": "Boundary behavior should be handled by a boundary-focused diagnostic component rather than only thresholding price/premium predictions.",
            "supporting_evidence": "Stage 4 premium-implied boundary lacks comparable error points, while direct boundary head passes validation/test/stress diagnostics.",
            "required_citation_or_artifact": "boundary_metrics_by_split.csv; boundary_diagnostic_report.tex",
            "status": "SUPPORTED_WITH_LIMITATIONS",
            "final_paper_use": "Boundary diagnostic component section.",
        },
        {
            "claim_id": "C04",
            "claim_text": "Delta behavior should be handled with a Delta-focused diagnostic component rather than assumed from price accuracy.",
            "supporting_evidence": "Stage 5 supervised Delta head passes RMSE and bounds/sign checks; direct price-implied Delta fails bounds/sign checks.",
            "required_citation_or_artifact": "delta_metrics_by_split.csv; delta_bounds_violation_summary.csv",
            "status": "SUPPORTED_WITH_LIMITATIONS",
            "final_paper_use": "Delta diagnostic component section.",
        },
        {
            "claim_id": "C05",
            "claim_text": "The project contribution is a modular research workflow, not a single universal neural solver.",
            "supporting_evidence": "Stages 3-5 retain separate price/premium, boundary, and Delta components, with Gamma blocked.",
            "required_citation_or_artifact": "price_premium_surrogate_report.tex; boundary_diagnostic_report.tex; delta_diagnostic_report.tex",
            "status": "SUPPORTED_WITH_LIMITATIONS",
            "final_paper_use": "Integrated workflow and conclusion sections.",
        },
    ]


def build_blocked_claims_matrix() -> list[dict[str, str]]:
    return [
        {
            "blocked_claim": "Production pricing or production risk-system readiness",
            "reason_blocked": "The workflow is validated as a research diagnostic pipeline only.",
            "evidence_gap": "No market calibration, production controls, or operational validation.",
            "possible_future_work": "Add calibration, monitoring, runtime SLAs, and independent production QA.",
        },
        {
            "blocked_claim": "Exact analytical free-boundary accuracy",
            "reason_blocked": "Boundary labels are threshold-based CN/PSOR diagnostics.",
            "evidence_gap": "No analytical free-boundary reference across the full parameter grid.",
            "possible_future_work": "Compare against high-resolution references and alternative boundary methods.",
        },
        {
            "blocked_claim": "Production Greek reliability",
            "reason_blocked": "Delta is a finite-difference diagnostic label and Gamma remains fragile.",
            "evidence_gap": "No production Greek validation, hedging backtest, or Gamma head.",
            "possible_future_work": "Add higher-grid Greek confirmation, hedging diagnostics, and uncertainty checks.",
        },
        {
            "blocked_claim": "Gamma-head or Gamma-surrogate claims",
            "reason_blocked": "No Gamma head is trained in the compressed roadmap.",
            "evidence_gap": "Gamma instability near kinks and boundaries has not been repaired.",
            "possible_future_work": "Design a separate masked Gamma diagnostic stage after human review.",
        },
        {
            "blocked_claim": "Broad extrapolation outside the approved v1 grid",
            "reason_blocked": "The dataset uses a fixed small Cartesian grid and regime-level splits.",
            "evidence_gap": "No larger envelope, active learning, or out-of-distribution campaign.",
            "possible_future_work": "Run a larger-regime envelope study with PSOR spot checks.",
        },
        {
            "blocked_claim": "One surrogate model solves price, boundary, and Greeks together",
            "reason_blocked": "Stages 4 and 5 show separate boundary and Delta components are needed.",
            "evidence_gap": "Single-model boundary and Greek reliability is not demonstrated.",
            "possible_future_work": "Study integrated multi-head architectures only after preserving component diagnostics.",
        },
    ]


def build_integrated_metrics_summary(project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    quality = _rows_by_metric(read_csv_rows(project_root / REQUIRED_INPUT_CSVS[0]))
    stage3_split = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[1])
    obstacle = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[3])
    boundary = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[5])
    delta = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[9])
    delta_bounds = read_csv_rows(project_root / REQUIRED_INPUT_CSVS[12])

    rows: list[dict[str, str]] = [
        _metric_row("solver", "validation_gate_decision", "overall", "PASS_SOLVER_VALIDATION_WITH_LIMITATIONS", "Solver validation passed with explicit limitations.", "reports/03_solver/tickets/ticket_12_solver_validation_synthesis.tex"),
    ]
    for metric in (
        "accepted_regimes",
        "accepted_rows",
        "max_obstacle_violation",
        "max_equation_violation",
        "max_abs_complementarity_product",
    ):
        rows.append(
            _metric_row(
                "dataset",
                metric,
                "v1_small_grid",
                quality[metric]["metric_value"],
                quality[metric]["notes"],
                str(REQUIRED_INPUT_CSVS[0]),
            )
        )
    for split in ("validation", "test", "stress_holdout"):
        pos = _find_row(stage3_split, model_name="positive_premium_mlp", split=split)
        rows.append(_metric_row("price/premium", "positive_premium_value_rmse", split, pos["value_rmse"], "Primary Stage 3 value error.", str(REQUIRED_INPUT_CSVS[1])))
        obs = _find_row(obstacle, model_name="positive_premium_mlp", split=split)
        rows.append(_metric_row("price/premium", "positive_premium_obstacle_rate", split, obs["obstacle_violation_rate"], "Obstacle diagnostic for reconstructed value.", str(REQUIRED_INPUT_CSVS[3])))
    for split in ("validation", "test", "stress_holdout"):
        direct = _find_row(boundary, method_name="direct_boundary_head", split=split)
        rows.append(_metric_row("boundary", "direct_boundary_head_rmse", split, direct["boundary_rmse"], "Direct boundary-head diagnostic error.", str(REQUIRED_INPUT_CSVS[5])))
        implied = _find_row(boundary, method_name="premium_implied_boundary", split=split)
        rows.append(_metric_row("boundary", "premium_implied_boundary_rmse", split, implied["boundary_rmse"], "NA means no comparable boundary-error points; not a zero-error result.", str(REQUIRED_INPUT_CSVS[5])))
    for split in ("validation", "test", "stress_holdout"):
        supervised = _find_row(delta, method_name="supervised_delta_head", split=split)
        rows.append(_metric_row("Delta", "supervised_delta_head_rmse", split, supervised["delta_rmse"], "Supervised finite-difference Delta diagnostic error.", str(REQUIRED_INPUT_CSVS[9])))
        bounds = _find_row(delta_bounds, method_name="supervised_delta_head", split=split)
        rows.append(_metric_row("Delta", "supervised_delta_head_bounds_rate", split, bounds["bounds_violation_rate"], "Bounds violation diagnostic for Delta head.", str(REQUIRED_INPUT_CSVS[12])))
    rows.append(_metric_row("Gamma", "gamma_head_status", "overall", "BLOCKED", "No Gamma head is trained or claimed.", "reports/06_surrogate/delta_diagnostic_report.tex"))
    return rows


def build_figure_table_inventory() -> list[dict[str, str]]:
    return [
        _inventory("F01", "figure", "results/05_surrogate_models/price_premium/figures/error_by_split.png", "Price / premium surrogate", "Compare direct value and positive-premium held-out errors.", "recommended"),
        _inventory("F02", "figure", "results/05_surrogate_models/price_premium/figures/obstacle_violation_comparison.png", "Price / premium surrogate", "Show obstacle and negative-premium behavior.", "recommended"),
        _inventory("F03", "figure", "results/05_surrogate_models/boundary/figures/boundary_error_by_split.png", "Boundary diagnostic component", "Show direct boundary-head diagnostics and premium-implied limitations.", "recommended"),
        _inventory("F04", "figure", "results/05_surrogate_models/boundary/figures/sample_put_boundary_curves.png", "Boundary diagnostic component", "Show representative put boundary curves.", "optional"),
        _inventory("F05", "figure", "results/05_surrogate_models/delta/figures/delta_error_by_split.png", "Delta diagnostic component", "Show Delta error across held-out splits.", "recommended"),
        _inventory("F06", "figure", "results/05_surrogate_models/delta/figures/delta_bounds_violation_comparison.png", "Delta diagnostic component", "Show bounds/sign diagnostics.", "recommended"),
        _inventory("T01", "table", "reports/07_integrated/component_readiness_matrix.csv", "Integrated workflow", "Summarize component readiness.", "include"),
        _inventory("T02", "table", "reports/07_integrated/claim_evidence_matrix.csv", "Integrated workflow", "Map final claims to evidence.", "include"),
        _inventory("T03", "table", "reports/07_integrated/blocked_claims_matrix.csv", "Limitations", "Prevent overclaiming.", "include"),
    ]


def build_final_paper_section_plan() -> list[dict[str, str]]:
    sections = [
        ("1", "Introduction", "Motivate American option risk surfaces and the free-boundary challenge.", "Planning report and methodology guide.", "Integrated workflow diagram"),
        ("2", "Research question", "State the modular structure-preserving surrogate question.", "Application strength gate and Stage 6 claim matrix.", "Claim evidence matrix"),
        ("3", "Background and American option formulation", "Define payoff obstacle, LCP, boundary, Delta, and Gamma caveats.", "Formulation note and references.", "Mathematical formulation table"),
        ("4", "Numerical benchmark: CN/PSOR and LCP validation", "Document the validated finite-difference benchmark.", "Ticket 12 synthesis and v1 QA.", "Solver validation table"),
        ("5", "Dataset construction", "Explain v1 regimes, splits, masks, and diagnostics.", "v1 small-grid report.", "Dataset QA table"),
        ("6", "Price / premium surrogate", "Show why positive premium is the retained price component.", "Stage 3 metrics and obstacle summary.", "Price and obstacle figures"),
        ("7", "Boundary diagnostic component", "Explain why direct boundary diagnostics are needed.", "Stage 4 boundary metrics.", "Boundary error and curve figures"),
        ("8", "Delta diagnostic component", "Explain why Delta needs a supervised diagnostic head.", "Stage 5 metrics and bounds summary.", "Delta error and bounds figures"),
        ("9", "Integrated workflow", "Assemble the component workflow and claim matrix.", "Stage 6 readiness matrices.", "Component overview and evidence map"),
        ("10", "Limitations", "State blocked claims and scope boundaries.", "Blocked claims matrix.", "Blocked claims table"),
        ("11", "Conclusion", "Summarize the cautious research contribution.", "Supported claim matrix.", "Final claims status"),
        ("12", "Future work", "Describe larger regimes, active learning, Gamma diagnostics, and production gaps.", "Handouts and blocked claims.", "Future work table"),
    ]
    return [
        {
            "section_number": number,
            "section_title": title,
            "purpose": purpose,
            "key_evidence": evidence,
            "figures_or_tables": figures,
            "writing_status": "planned_not_written",
        }
        for number, title, purpose, evidence, figures in sections
    ]


def stage6_review_decision(
    component_rows: list[dict[str, str]],
    claim_rows: list[dict[str, str]],
    blocked_rows: list[dict[str, str]],
) -> str:
    if not component_rows or not claim_rows or not blocked_rows:
        return "REVIEW_REQUIRED_BEFORE_FINAL_PAPER"
    required_components = [row for row in component_rows if row["component"] != "Gamma component"]
    if any(row["status"].startswith("REVIEW") or row["status"] == "MISSING" for row in required_components):
        return "REVIEW_REQUIRED_BEFORE_FINAL_PAPER"
    if not any(row["status"] == "BLOCKED" and row["component"] == "Gamma component" for row in component_rows):
        return "REVIEW_REQUIRED_BEFORE_FINAL_PAPER"
    if any(row["status"] not in {"SUPPORTED_WITH_LIMITATIONS", "SUPPORTED_DIAGNOSTIC_ONLY"} for row in claim_rows):
        return "REVIEW_REQUIRED_BEFORE_FINAL_PAPER"
    return "READY_FOR_FINAL_PAPER_DRAFTING"


def run_integrated_workflow(
    *,
    project_root: Path = PROJECT_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    create_figures: bool = True,
) -> IntegratedWorkflowResult:
    _validate_required_inputs(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    component_rows = build_component_readiness_matrix(project_root)
    claim_rows = build_claim_evidence_matrix()
    blocked_rows = build_blocked_claims_matrix()
    inventory_rows = build_figure_table_inventory()
    section_rows = build_final_paper_section_plan()
    metrics_rows = build_integrated_metrics_summary(project_root)
    decision = stage6_review_decision(component_rows, claim_rows, blocked_rows)

    report_csv_paths = (
        report_dir / "component_readiness_matrix.csv",
        report_dir / "claim_evidence_matrix.csv",
        report_dir / "blocked_claims_matrix.csv",
        report_dir / "figure_table_inventory.csv",
        report_dir / "final_paper_section_plan.csv",
    )
    result_csv_paths = (results_dir / "integrated_metrics_summary.csv",)

    write_csv(report_csv_paths[0], component_rows, COMPONENT_READINESS_FIELDNAMES)
    write_csv(report_csv_paths[1], claim_rows, CLAIM_EVIDENCE_FIELDNAMES)
    write_csv(report_csv_paths[2], blocked_rows, BLOCKED_CLAIMS_FIELDNAMES)
    write_csv(report_csv_paths[3], inventory_rows, FIGURE_TABLE_INVENTORY_FIELDNAMES)
    write_csv(report_csv_paths[4], section_rows, FINAL_PAPER_SECTION_FIELDNAMES)
    write_csv(result_csv_paths[0], metrics_rows, INTEGRATED_METRICS_FIELDNAMES)

    figure_paths = (
        figures_dir / "component_readiness_overview.png",
        figures_dir / "workflow_evidence_map.png",
        figures_dir / "stress_holdout_summary.png",
        figures_dir / "final_claims_status.png",
    )
    if create_figures:
        _write_figures(figure_paths, component_rows, claim_rows, blocked_rows, metrics_rows)
    else:
        for path in figure_paths:
            _write_blank_png(path)

    outline_path = report_dir / "final_paper_outline.md"
    outline_path.write_text(_final_paper_outline_text(section_rows, claim_rows, blocked_rows), encoding="utf-8")

    report_tex_path = report_dir / "integrated_workflow_report.tex"
    report_tex_path.write_text(
        _report_tex(component_rows, claim_rows, blocked_rows, inventory_rows, section_rows, metrics_rows, figure_paths, decision),
        encoding="utf-8",
    )

    manifest_path = results_dir / "integrated_workflow_manifest.csv"
    manifest_rows = _manifest_rows(project_root, report_csv_paths, result_csv_paths, figure_paths, report_tex_path, outline_path, manifest_path)
    write_csv(manifest_path, manifest_rows, WORKFLOW_MANIFEST_FIELDNAMES)

    return IntegratedWorkflowResult(
        report_tex_path=report_tex_path,
        final_paper_outline_path=outline_path,
        report_csv_paths=report_csv_paths,
        result_csv_paths=(result_csv_paths[0], manifest_path),
        figure_paths=figure_paths,
        manifest_path=manifest_path,
        review_decision=decision,
    )


def main() -> IntegratedWorkflowResult:
    result = run_integrated_workflow()
    print(f"Stage 6 review decision: {result.review_decision}")
    print(f"Integrated workflow report: {result.report_tex_path}")
    print(f"Integrated outputs: {result.manifest_path.parent}")
    return result


def _validate_required_inputs(project_root: Path) -> None:
    for path in REQUIRED_INPUT_CSVS:
        read_csv_rows(project_root / path)
    for path in REQUIRED_INPUT_REPORTS:
        if not (project_root / path).exists():
            raise FileNotFoundError(project_root / path)


def _rows_by_metric(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["metric"]: row for row in rows}


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"missing row matching {criteria!r}")


def _fmt(value: str | float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    if abs(number) >= 1e-3:
        return f"{number:.6f}"
    return f"{number:.3e}"


def _metric_row(component: str, metric: str, split: str, value: str, interpretation: str, source: str) -> dict[str, str]:
    return {
        "component": component,
        "metric_name": metric,
        "split_or_region": split,
        "value": value,
        "interpretation": interpretation,
        "source_file": source,
    }


def _inventory(item_id: str, item_type: str, source: str, section: str, purpose: str, status: str) -> dict[str, str]:
    return {
        "item_id": item_id,
        "item_type": item_type,
        "source_file": source,
        "recommended_final_paper_section": section,
        "purpose": purpose,
        "status": status,
    }


def _write_figures(
    figure_paths: tuple[Path, ...],
    component_rows: list[dict[str, str]],
    claim_rows: list[dict[str, str]],
    blocked_rows: list[dict[str, str]],
    metrics_rows: list[dict[str, str]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        for path in figure_paths:
            _write_blank_png(path)
        return

    _figure_component_readiness(plt, figure_paths[0], component_rows)
    _figure_workflow_map(plt, figure_paths[1])
    _figure_stress_holdout(plt, figure_paths[2], metrics_rows)
    _figure_claims_status(plt, figure_paths[3], claim_rows, blocked_rows)


def _figure_component_readiness(plt, path: Path, component_rows: list[dict[str, str]]) -> None:
    labels = [row["component"].replace(" ", "\n") for row in component_rows]
    score = [0.0 if row["status"] == "BLOCKED" else 1.0 for row in component_rows]
    colors = ["#4c78a8" if row["status"] != "BLOCKED" else "#b75d69" for row in component_rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(labels)), score, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Readiness indicator")
    ax.set_title("Stage 6 component readiness overview")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    for index, row in enumerate(component_rows):
        ax.text(index, score[index] + 0.03, row["status"], ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_workflow_map(plt, path: Path) -> None:
    nodes = [
        ("CN/PSOR\nbenchmark", 0.07, 0.62),
        ("v1 dataset", 0.25, 0.62),
        ("price/premium\nsurrogate", 0.43, 0.62),
        ("boundary\ncomponent", 0.61, 0.76),
        ("Delta\ncomponent", 0.61, 0.48),
        ("final paper\nclaims", 0.80, 0.62),
        ("Gamma\nblocked", 0.43, 0.24),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    for label, x, y in nodes:
        face = "#f7f7f7" if "blocked" not in label else "#f7d8dc"
        ax.text(x, y, label, ha="center", va="center", fontsize=10, bbox={"boxstyle": "round,pad=0.35", "fc": face, "ec": "#555555"})
    arrows = [
        ((0.14, 0.62), (0.20, 0.62)),
        ((0.31, 0.62), (0.37, 0.62)),
        ((0.50, 0.64), (0.56, 0.73)),
        ((0.50, 0.60), (0.56, 0.51)),
        ((0.68, 0.74), (0.76, 0.65)),
        ((0.68, 0.50), (0.76, 0.59)),
        ((0.43, 0.55), (0.43, 0.31)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"})
    ax.set_title("Integrated evidence map: modular workflow, Gamma blocked", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_stress_holdout(plt, path: Path, metrics_rows: list[dict[str, str]]) -> None:
    items = [
        ("Price/premium\nvalue RMSE", _metric_value(metrics_rows, "price/premium", "positive_premium_value_rmse", "stress_holdout")),
        ("Boundary\nRMSE", _metric_value(metrics_rows, "boundary", "direct_boundary_head_rmse", "stress_holdout")),
        ("Delta\nRMSE", _metric_value(metrics_rows, "Delta", "supervised_delta_head_rmse", "stress_holdout")),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([label for label, _ in items], [value for _, value in items], color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.set_ylabel("Stress-holdout error")
    ax.set_title("Stress-holdout summary across retained components")
    for index, (_, value) in enumerate(items):
        ax.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_claims_status(plt, path: Path, claim_rows: list[dict[str, str]], blocked_rows: list[dict[str, str]]) -> None:
    labels = ["supported with limits", "blocked"]
    values = [len(claim_rows), len(blocked_rows)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(labels, values, color=["#54a24b", "#b75d69"])
    ax.set_ylabel("Claim count")
    ax.set_title("Final claim status")
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _metric_value(rows: list[dict[str, str]], component: str, metric: str, split: str) -> float:
    row = next(
        item
        for item in rows
        if item["component"] == component and item["metric_name"] == metric and item["split_or_region"] == split
    )
    return float(row["value"])


def _write_blank_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_BLANK_PNG)


def _manifest_rows(
    project_root: Path,
    report_csv_paths: tuple[Path, ...],
    result_csv_paths: tuple[Path, ...],
    figure_paths: tuple[Path, ...],
    report_tex_path: Path,
    outline_path: Path,
    manifest_path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in REQUIRED_INPUT_CSVS + REQUIRED_INPUT_REPORTS:
        rows.append(
            {
                "artifact": source.name,
                "artifact_type": "input",
                "source_path": str(source),
                "included_in_stage6": str((project_root / source).exists()),
                "notes": "Read as Stage 6 evidence source.",
            }
        )
    for path in report_csv_paths:
        rows.append(_output_manifest_row(path, "report_csv"))
    for path in result_csv_paths:
        rows.append(_output_manifest_row(path, "result_csv"))
    for path in figure_paths:
        rows.append(_output_manifest_row(path, "figure"))
    rows.append(_output_manifest_row(report_tex_path, "report_tex"))
    rows.append(_output_manifest_row(outline_path, "final_paper_outline"))
    rows.append(_output_manifest_row(manifest_path, "result_csv", included=True))
    return rows


def _output_manifest_row(path: Path, artifact_type: str, *, included: bool | None = None) -> dict[str, str]:
    return {
        "artifact": path.name,
        "artifact_type": artifact_type,
        "source_path": str(path),
        "included_in_stage6": str(path.exists() if included is None else included),
        "notes": "Generated by Stage 6 synthesis.",
    }


def _final_paper_outline_text(
    section_rows: list[dict[str, str]],
    claim_rows: list[dict[str, str]],
    blocked_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Final Paper Outline",
        "",
        "This outline is a Stage 6 preparation artifact. It is not the final paper.",
        "",
        "## Supported Story",
        "",
        "The paper should present a modular American-option risk-surface workflow: validated CN/PSOR benchmark, v1 research dataset, positive-premium price component, direct boundary diagnostic component, and supervised Delta diagnostic component. Gamma remains blocked.",
        "",
        "## Sections",
        "",
    ]
    for row in section_rows:
        lines.append(f"{row['section_number']}. **{row['section_title']}**")
        lines.append(f"   - Purpose: {row['purpose']}")
        lines.append(f"   - Evidence: {row['key_evidence']}")
        lines.append(f"   - Figures/tables: {row['figures_or_tables']}")
    lines.extend(["", "## Claims To Use", ""])
    for row in claim_rows:
        lines.append(f"- {row['claim_id']}: {row['claim_text']} ({row['status']})")
    lines.extend(["", "## Claims To Block", ""])
    for row in blocked_rows:
        lines.append(f"- {row['blocked_claim']}: {row['reason_blocked']}")
    lines.extend(["", "## Recommended Next Step", "", "Draft the final paper from these matrices without adding new model claims."])
    return "\n".join(lines) + "\n"


def _report_tex(
    component_rows: list[dict[str, str]],
    claim_rows: list[dict[str, str]],
    blocked_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
    section_rows: list[dict[str, str]],
    metrics_rows: list[dict[str, str]],
    figure_paths: tuple[Path, ...],
    decision: str,
) -> str:
    component_table = _latex_table(component_rows, ["component", "method", "key_metric", "status"], [0.19, 0.24, 0.31, 0.16])
    claim_table = _latex_table(claim_rows, ["claim_id", "claim_text", "status", "final_paper_use"], [0.09, 0.43, 0.20, 0.20])
    blocked_table = _latex_table(blocked_rows, ["blocked_claim", "reason_blocked", "possible_future_work"], [0.28, 0.33, 0.30])
    section_table = _latex_table(section_rows, ["section_number", "section_title", "purpose", "key_evidence"], [0.08, 0.20, 0.35, 0.28])
    metric_table = _latex_table(_headline_metric_rows(metrics_rows), ["component", "metric_name", "split_or_region", "value"], [0.18, 0.32, 0.20, 0.20])
    inventory_table = _latex_table(inventory_rows[:8], ["item_id", "item_type", "recommended_final_paper_section", "purpose"], [0.10, 0.13, 0.25, 0.42])
    figures = "\n".join(
        [
            "\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=0.82\\textwidth]{{{_latex_path(path)}}}\n"
            f"\\caption{{{_latex_escape(_figure_caption(path))}}}\n"
            "\\end{figure}"
            for path in figure_paths
        ]
    )
    return rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{fontspec}}
\IfFontExistsTF{{Times New Roman}}{{\setmainfont{{Times New Roman}}}}{{\setmainfont{{TeX Gyre Termes}}}}
\IfFontExistsTF{{Menlo}}{{\setmonofont{{Menlo}}}}{{\setmonofont{{Latin Modern Mono}}}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{enumitem}}
\usepackage{{graphicx}}
\usepackage{{fancyhdr}}
\usepackage[round,authoryear]{{natbib}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{xurl}}
\usepackage{{microtype}}

\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.65em}}
\setlist[itemize]{{leftmargin=1.4em}}
\newcommand{{\file}}[1]{{\path{{#1}}}}
\newcommand{{\code}}[1]{{\path{{#1}}}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}

\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{\small Stage 6 Integrated Workflow}}
\rhead{{\small Claim Synthesis}}
\cfoot{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\title{{\textbf{{Stage 6 Integrated Workflow Report}}\\
\large Claim Synthesis and Final Paper Preparation}}
\author{{Codex-assisted integrated workflow review}}
\date{{June 21, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report synthesizes the completed solver validation, v1 small-grid dataset, price/premium
surrogate, boundary diagnostic component, and Delta diagnostic component. It does not train a
new model, regenerate the dataset, introduce a Gamma head, or modify solver code. The Stage 6
review decision is \code{{{_latex_escape(decision)}}}.
\end{{abstract}}

\tableofcontents
\newpage

\section{{Purpose of Stage 6}}

Stage 6 is an integrated review and final-paper preparation pass. Its job is to connect each
technical component to a claim that can safely appear in a draft paper. The methodology guides
emphasize that American option risk-surface work should begin with a validated finite-difference
benchmark and only then add constrained, diagnostic neural components \citep{{brennan_schwartz_1977,wilmott_howison_dewynne_1995}}.

\section{{Compressed Roadmap Status}}

The compressed roadmap has completed five stages before this synthesis: v0 dry run, v1 dataset
QA, price/premium surrogate comparison, boundary diagnostics, and Delta diagnostics. Stage 6
does not add a sixth model; it audits the evidence produced by those stages.

\section{{Component Evidence Summary}}

{component_table}

\section{{Integrated Workflow Description}}

The retained workflow is modular. A validated CN/PSOR benchmark generates diagnostic surfaces.
The v1 small-grid dataset records regime-level splits, labels, masks, and diagnostics. The
positive-premium model handles price and continuation premium. A direct boundary-head diagnostic
component handles threshold-based free-boundary labels. A supervised Delta-head diagnostic
component handles finite-difference Delta labels. Gamma remains blocked rather than silently
converted into a model claim.

\section{{Headline Metrics}}

{metric_table}

\section{{Price/Premium Component Synthesis}}

The Stage 3 evidence supports the positive-premium representation. Predicting nonnegative
continuation premium and reconstructing value with analytic payoff aligns the surrogate with the
American obstacle. This supports a price and premium surface claim, but not a boundary or Greek
claim by itself.

\section{{Boundary Component Synthesis}}

The Stage 4 evidence is deliberately two-part. The premium-implied boundary method is not reliable
in this test because many rows have no comparable boundary-error points. The direct boundary-head
diagnostic baseline satisfies the boundary diagnostics and supports a separate boundary-focused
component. The boundary remains a threshold-based CN/PSOR diagnostic, not exact analytical truth.

\section{{Delta Component Synthesis}}

The Stage 5 evidence shows that good price or premium accuracy does not automatically imply
production-quality Delta. Price-implied Delta diagnostics are informative, but the supervised
Delta head is the retained diagnostic component because it satisfies the RMSE and bounds/sign
criteria. Delta labels remain finite-difference diagnostics.

\section{{Blocked Gamma and Production Claims}}

No Gamma head is trained in the compressed roadmap. Production pricing, exact free-boundary,
production Greek, broad extrapolation, and one-model-solves-all claims are blocked. This is a
strength of the synthesis: weak claims are removed rather than hidden.

\section{{Claim Evidence Matrix}}

{claim_table}

\section{{Blocked Claims Matrix}}

{blocked_table}

\section{{Final Paper Outline}}

{section_table}

\section{{Figure and Table Inventory}}

{inventory_table}

\section{{Stage 6 Figures}}

{figures}

\section{{Limitations}}

\begin{{itemize}}
  \item The integrated workflow is a synthesis of existing evidence, not a fresh PSOR benchmark run.
  \item The v1 dataset is limited to the approved small-grid Black-Scholes regime envelope.
  \item Boundary labels are threshold-based diagnostics.
  \item Delta labels are finite-difference diagnostics.
  \item Gamma is blocked from surrogate claims.
  \item The final paper still needs careful prose, citation integration, and human review.
\end{{itemize}}

\section{{Final Readiness Decision}}

\[
\boxed{{\text{{\code{{{_latex_escape(decision)}}}}}}}
\]

The evidence is strong enough to begin final paper drafting, provided the paper keeps the modular
claim structure and the blocked-claims language intact.

\section{{Recommended Next Step}}

Draft the final paper from \file{{reports/07_integrated/final_paper_outline.md}} and the Stage 6
matrices. Do not add a Gamma claim, production claim, or universal neural-superiority claim unless
new evidence is generated in a separate approved stage.

\bibliographystyle{{plainnat}}
\bibliography{{reports/03_solver/references}}

\end{{document}}
"""


def _headline_metric_rows(metrics_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    keep = {
        ("dataset", "accepted_regimes", "v1_small_grid"),
        ("dataset", "accepted_rows", "v1_small_grid"),
        ("dataset", "max_obstacle_violation", "v1_small_grid"),
        ("price/premium", "positive_premium_value_rmse", "test"),
        ("price/premium", "positive_premium_value_rmse", "stress_holdout"),
        ("boundary", "direct_boundary_head_rmse", "test"),
        ("boundary", "direct_boundary_head_rmse", "stress_holdout"),
        ("Delta", "supervised_delta_head_rmse", "test"),
        ("Delta", "supervised_delta_head_rmse", "stress_holdout"),
        ("Gamma", "gamma_head_status", "overall"),
    }
    for row in metrics_rows:
        if (row["component"], row["metric_name"], row["split_or_region"]) in keep:
            selected.append(row)
    return selected


def _latex_table(rows: list[dict[str, str]], columns: list[str], widths: list[float]) -> str:
    spec = " ".join(f"L{{{width:.2f}\\textwidth}}" for width in widths)
    header = " & ".join(_latex_escape(column.replace("_", " ").title()) for column in columns) + r" \\"
    body = "\n".join(
        " & ".join(_latex_escape(str(row.get(column, ""))) for column in columns) + r" \\"
        for row in rows
    )
    return (
        f"\\begin{{longtable}}{{{spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        "\\endhead\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{longtable}"
    )


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _figure_caption(path: Path) -> str:
    captions = {
        "component_readiness_overview.png": "Component readiness overview.",
        "workflow_evidence_map.png": "Workflow evidence map.",
        "stress_holdout_summary.png": "Stress-holdout summary across retained components.",
        "final_claims_status.png": "Final supported and blocked claim counts.",
    }
    return captions.get(path.name, path.stem.replace("_", " ").title())


if __name__ == "__main__":
    main()

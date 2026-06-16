"""Ticket 11: grid and domain sensitivity diagnostics experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.sensitivity import (
    BoundarySensitivityRow,
    DiagnosticSensitivityRow,
    SelectedSpotSensitivityRow,
    SensitivityComparisonSummary,
    SensitivityRunResult,
    boundary_shift_rows,
    diagnostic_row,
    domain_sensitivity_cases,
    grid_sensitivity_cases,
    run_sensitivity_case,
    selected_spot_rows,
    summarize_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

GRID_SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_11_grid_sensitivity_summary.csv"
)
DOMAIN_SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_11_domain_sensitivity_summary.csv"
)
SELECTED_SPOT_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_11_selected_spot_sensitivity.csv"
)
BOUNDARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_11_boundary_sensitivity.csv"
)
DIAGNOSTIC_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_11_diagnostic_sensitivity.csv"
)

PRICE_GRID_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_11_price_vs_grid_size.png"
)
PRICE_ERROR_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_11_price_error_vs_reference.png"
)
BOUNDARY_GRID_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_11_boundary_grid_comparison.png"
)
DOMAIN_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_11_domain_cutoff_comparison.png"
)
LCP_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_11_lcp_diagnostic_stability.png"
)
GAMMA_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_11_gamma_stability.png"
)

SUMMARY_FIELDNAMES = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "dS",
    "dtau",
    "reference_case_name",
    "reference_Smax",
    "reference_M",
    "reference_N",
    "all_psor_steps_converged",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_abs_selected_price_difference",
    "rmse_selected_price_difference",
    "boundary_found_count",
    "max_abs_boundary_shift",
    "mean_abs_boundary_shift",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "runtime_seconds",
    "price_vs_grid_figure_created",
    "price_error_figure_created",
    "boundary_figure_created",
    "domain_figure_created",
    "lcp_figure_created",
    "gamma_figure_created",
]

SELECTED_FIELDNAMES = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "reference_case_name",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "value",
    "reference_nearest_spot",
    "reference_actual_moneyness",
    "reference_value",
    "difference_vs_reference",
    "abs_difference_vs_reference",
    "relative_difference_vs_reference",
]

BOUNDARY_FIELDNAMES = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "reference_case_name",
    "target_tau_fraction",
    "nearest_tau",
    "boundary_found",
    "boundary_spot",
    "reference_nearest_tau",
    "reference_boundary_found",
    "reference_boundary_spot",
    "boundary_shift",
    "abs_boundary_shift",
    "boundary_status",
]

DIAGNOSTIC_FIELDNAMES = [
    "sensitivity_type",
    "case_name",
    "option_type",
    "Smax",
    "M",
    "N",
    "dS",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "strict_negative_gamma_count",
    "runtime_seconds",
]


def run_grid_sensitivity() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[SensitivityRunResult],
]:
    """Run fixed-domain grid sensitivity cases and return CSV-ready rows."""

    runs = [run_sensitivity_case(case) for case in grid_sensitivity_cases()]
    return _rows_for_runs(runs)


def run_domain_sensitivity() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[SensitivityRunResult],
]:
    """Run comparable-spacing domain sensitivity cases and return CSV-ready rows."""

    runs = [run_sensitivity_case(case) for case in domain_sensitivity_cases()]
    return _rows_for_runs(runs)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_price_vs_grid_size_figure(
    grid_runs: list[SensitivityRunResult],
    path: Path,
) -> bool:
    """Plot at-the-money final price versus grid size for grid sensitivity."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    for option_type in ("put", "call"):
        runs = _runs_for_option(grid_runs, option_type)
        x_values = [run.case.M for run in runs]
        y_values = [_selected_final_value(run, 1.0) for run in runs]
        axis.plot(x_values, y_values, marker="o", linewidth=1.3, label=option_type)
    axis.set_xlabel("Grid intervals M=N")
    axis.set_ylabel("Final value at S/K near 1.0")
    axis.set_title("Ticket 11 selected price versus grid size")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_price_error_figure(
    grid_rows: list[dict[str, str]],
    domain_rows: list[dict[str, str]],
    path: Path,
) -> bool:
    """Plot max selected price difference versus reference."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for axis, rows, title, x_key in (
        (axes[0][0], grid_rows, "Grid sensitivity", "M"),
        (axes[0][1], domain_rows, "Domain sensitivity", "Smax"),
    ):
        for option_type in ("put", "call"):
            option_rows = [row for row in rows if row["option_type"] == option_type]
            x_values = [float(row[x_key]) for row in option_rows]
            y_values = [float(row["max_abs_selected_price_difference"]) for row in option_rows]
            axis.plot(x_values, y_values, marker="o", linewidth=1.2, label=option_type)
        axis.set_title(title)
        axis.set_xlabel(x_key)
        axis.set_ylabel("Max selected price difference")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_boundary_grid_figure(
    grid_runs: list[SensitivityRunResult],
    path: Path,
) -> bool:
    """Plot boundary curves across grid levels."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for axis, option_type in zip(axes[0], ("put", "call")):
        for run in _runs_for_option(grid_runs, option_type):
            curve = run.boundary_curve
            if curve is None:
                continue
            tau = [point.tau for point in curve.points if point.boundary_found]
            boundary = [point.boundary_spot for point in curve.points if point.boundary_found]
            axis.plot(tau, boundary, linewidth=1.1, label=f"M={run.case.M}")
        axis.set_title(f"{option_type} boundary")
        axis.set_xlabel("Time to maturity tau")
        axis.set_ylabel("Approximate boundary spot")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_domain_cutoff_figure(
    domain_rows: list[dict[str, str]],
    path: Path,
) -> bool:
    """Plot selected price difference and boundary shift versus Smax."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for option_type in ("put", "call"):
        rows = [row for row in domain_rows if row["option_type"] == option_type]
        smax = [float(row["Smax"]) for row in rows]
        price_diff = [float(row["max_abs_selected_price_difference"]) for row in rows]
        boundary_shift = [_parse_float(row["max_abs_boundary_shift"]) for row in rows]
        axes[0][0].plot(smax, price_diff, marker="o", linewidth=1.2, label=option_type)
        axes[0][1].plot(smax, boundary_shift, marker="o", linewidth=1.2, label=option_type)
    axes[0][0].set_title("Target-region price sensitivity")
    axes[0][0].set_xlabel("Smax")
    axes[0][0].set_ylabel("Max selected price difference")
    axes[0][1].set_title("Selected boundary sensitivity")
    axes[0][1].set_xlabel("Smax")
    axes[0][1].set_ylabel("Max selected boundary shift")
    for axis in axes[0]:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_lcp_diagnostic_figure(
    diagnostic_rows: list[dict[str, str]],
    path: Path,
) -> bool:
    """Plot LCP diagnostic stability across all settings."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(diagnostic_rows))
    equation = [float(row["max_equation_violation"]) for row in diagnostic_rows]
    complementarity = [float(row["max_abs_complementarity_product"]) for row in diagnostic_rows]
    axis.plot(x, equation, marker="o", linewidth=1.1, label="equation violation")
    axis.plot(x, complementarity, marker="s", linewidth=1.1, label="complementarity")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [row["case_name"].replace("american_", "a_").replace("dividend_", "d_") for row in diagnostic_rows],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel("Diagnostic metric")
    axis.set_title("Ticket 11 LCP diagnostic stability")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_gamma_stability_figure(
    diagnostic_rows: list[dict[str, str]],
    path: Path,
) -> bool:
    """Plot full and strict Gamma diagnostic stability across all settings."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(diagnostic_rows))
    full_gamma = [float(row["max_abs_gamma"]) for row in diagnostic_rows]
    strict_gamma = [float(row["max_abs_gamma_strict"]) for row in diagnostic_rows]
    axis.plot(x, full_gamma, marker="o", linewidth=1.1, label="full max |Gamma|")
    axis.plot(x, strict_gamma, marker="s", linewidth=1.1, label="strict max |Gamma|")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [row["case_name"].replace("american_", "a_").replace("dividend_", "d_") for row in diagnostic_rows],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel("Max |Gamma|")
    axis.set_title("Ticket 11 Gamma diagnostic stability")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    """Run Ticket 11 sensitivity experiments and write CSV/figure artifacts."""

    grid_rows, grid_selected, grid_boundary, grid_diagnostic, grid_runs = run_grid_sensitivity()
    domain_rows, domain_selected, domain_boundary, domain_diagnostic, domain_runs = (
        run_domain_sensitivity()
    )
    selected_rows = grid_selected + domain_selected
    boundary_rows = grid_boundary + domain_boundary
    diagnostic_rows = grid_diagnostic + domain_diagnostic

    price_vs_grid_created = create_price_vs_grid_size_figure(grid_runs, PRICE_GRID_FIGURE)
    price_error_created = create_price_error_figure(
        grid_rows, domain_rows, PRICE_ERROR_FIGURE
    )
    boundary_created = create_boundary_grid_figure(grid_runs, BOUNDARY_GRID_FIGURE)
    domain_created = create_domain_cutoff_figure(domain_rows, DOMAIN_FIGURE)
    lcp_created = create_lcp_diagnostic_figure(diagnostic_rows, LCP_FIGURE)
    gamma_created = create_gamma_stability_figure(diagnostic_rows, GAMMA_FIGURE)
    figure_flags = {
        "price_vs_grid_figure_created": str(price_vs_grid_created),
        "price_error_figure_created": str(price_error_created),
        "boundary_figure_created": str(boundary_created),
        "domain_figure_created": str(domain_created),
        "lcp_figure_created": str(lcp_created),
        "gamma_figure_created": str(gamma_created),
    }
    grid_rows = [row | figure_flags for row in grid_rows]
    domain_rows = [row | figure_flags for row in domain_rows]

    write_csv(GRID_SUMMARY_CSV, grid_rows, SUMMARY_FIELDNAMES)
    write_csv(DOMAIN_SUMMARY_CSV, domain_rows, SUMMARY_FIELDNAMES)
    write_csv(SELECTED_SPOT_CSV, selected_rows, SELECTED_FIELDNAMES)
    write_csv(BOUNDARY_CSV, boundary_rows, BOUNDARY_FIELDNAMES)
    write_csv(DIAGNOSTIC_CSV, diagnostic_rows, DIAGNOSTIC_FIELDNAMES)

    metadata = {
        "grid_summary_csv": str(GRID_SUMMARY_CSV),
        "domain_summary_csv": str(DOMAIN_SUMMARY_CSV),
        "selected_spot_csv": str(SELECTED_SPOT_CSV),
        "boundary_csv": str(BOUNDARY_CSV),
        "diagnostic_csv": str(DIAGNOSTIC_CSV),
        "price_vs_grid_figure_path": str(PRICE_GRID_FIGURE),
        "price_error_figure_path": str(PRICE_ERROR_FIGURE),
        "boundary_figure_path": str(BOUNDARY_GRID_FIGURE),
        "domain_figure_path": str(DOMAIN_FIGURE),
        "lcp_figure_path": str(LCP_FIGURE),
        "gamma_figure_path": str(GAMMA_FIGURE),
    } | figure_flags
    print(f"wrote {len(grid_rows)} rows to {GRID_SUMMARY_CSV}")
    print(f"wrote {len(domain_rows)} rows to {DOMAIN_SUMMARY_CSV}")
    print(f"wrote {len(selected_rows)} rows to {SELECTED_SPOT_CSV}")
    print(f"wrote {len(boundary_rows)} rows to {BOUNDARY_CSV}")
    print(f"wrote {len(diagnostic_rows)} rows to {DIAGNOSTIC_CSV}")
    print(
        "figures_created="
        f"{price_vs_grid_created},{price_error_created},{boundary_created},"
        f"{domain_created},{lcp_created},{gamma_created}"
    )
    return grid_rows, domain_rows, selected_rows, boundary_rows, diagnostic_rows, metadata


def _rows_for_runs(
    runs: list[SensitivityRunResult],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[SensitivityRunResult],
]:
    references = _reference_runs_by_option(runs)
    summary_rows: list[dict[str, str]] = []
    selected_rows: list[dict[str, str]] = []
    boundary_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []
    for option_type, family_runs in _runs_by_option(runs).items():
        reference = references[option_type]
        summaries = summarize_comparison(option_type, family_runs, reference)
        summary_rows.extend(_summary_row(summary) for summary in summaries)
        for run in family_runs:
            selected_rows.extend(
                _selected_row(row) for row in selected_spot_rows(run, reference)
            )
            boundary_rows.extend(
                _boundary_row(row) for row in boundary_shift_rows(run, reference)
            )
            diagnostic_rows.append(_diagnostic_row(diagnostic_row(run)))
    return summary_rows, selected_rows, boundary_rows, diagnostic_rows, runs


def _summary_row(summary: SensitivityComparisonSummary) -> dict[str, str]:
    return {
        "sensitivity_type": summary.sensitivity_type,
        "case_name": summary.case_name,
        "option_type": summary.option_type,
        "K": _format_float(summary.K),
        "T": _format_float(summary.T),
        "r": _format_float(summary.r),
        "q": _format_float(summary.q),
        "sigma": _format_float(summary.sigma),
        "Smax": _format_float(summary.Smax),
        "M": str(summary.M),
        "N": str(summary.N),
        "dS": _format_float(summary.dS),
        "dtau": _format_float(summary.dtau),
        "reference_case_name": summary.reference_case_name,
        "reference_Smax": _format_float(summary.reference_Smax),
        "reference_M": str(summary.reference_M),
        "reference_N": str(summary.reference_N),
        "all_psor_steps_converged": str(summary.all_psor_steps_converged),
        "max_psor_iterations": str(summary.max_psor_iterations),
        "mean_psor_iterations": _format_float(summary.mean_psor_iterations),
        "max_final_update": _format_float(summary.max_final_update),
        "max_abs_selected_price_difference": _format_float(
            summary.max_abs_selected_price_difference
        ),
        "rmse_selected_price_difference": _format_float(
            summary.rmse_selected_price_difference
        ),
        "boundary_found_count": str(summary.boundary_found_count),
        "max_abs_boundary_shift": _format_float(summary.max_abs_boundary_shift),
        "mean_abs_boundary_shift": _format_float(summary.mean_abs_boundary_shift),
        "max_obstacle_violation": _format_float(summary.max_obstacle_violation),
        "max_equation_violation": _format_float(summary.max_equation_violation),
        "max_abs_complementarity_product": _format_float(
            summary.max_abs_complementarity_product
        ),
        "max_abs_gamma": _format_float(summary.max_abs_gamma),
        "max_abs_gamma_strict": _format_float(summary.max_abs_gamma_strict),
        "runtime_seconds": _format_float(summary.runtime_seconds),
        "price_vs_grid_figure_created": "False",
        "price_error_figure_created": "False",
        "boundary_figure_created": "False",
        "domain_figure_created": "False",
        "lcp_figure_created": "False",
        "gamma_figure_created": "False",
    }


def _selected_row(row: SelectedSpotSensitivityRow) -> dict[str, str]:
    return {
        "sensitivity_type": row.sensitivity_type,
        "case_name": row.case_name,
        "option_type": row.option_type,
        "reference_case_name": row.reference_case_name,
        "target_moneyness": _format_float(row.target_moneyness),
        "nearest_spot": _format_float(row.nearest_spot),
        "actual_moneyness": _format_float(row.actual_moneyness),
        "value": _format_float(row.value),
        "reference_nearest_spot": _format_float(row.reference_nearest_spot),
        "reference_actual_moneyness": _format_float(row.reference_actual_moneyness),
        "reference_value": _format_float(row.reference_value),
        "difference_vs_reference": _format_float(row.difference_vs_reference),
        "abs_difference_vs_reference": _format_float(row.abs_difference_vs_reference),
        "relative_difference_vs_reference": _format_float(
            row.relative_difference_vs_reference
        ),
    }


def _boundary_row(row: BoundarySensitivityRow) -> dict[str, str]:
    return {
        "sensitivity_type": row.sensitivity_type,
        "case_name": row.case_name,
        "option_type": row.option_type,
        "reference_case_name": row.reference_case_name,
        "target_tau_fraction": _format_float(row.target_tau_fraction),
        "nearest_tau": _format_float(row.nearest_tau),
        "boundary_found": str(row.boundary_found),
        "boundary_spot": _format_float(row.boundary_spot),
        "reference_nearest_tau": _format_float(row.reference_nearest_tau),
        "reference_boundary_found": str(row.reference_boundary_found),
        "reference_boundary_spot": _format_float(row.reference_boundary_spot),
        "boundary_shift": _format_float(row.boundary_shift),
        "abs_boundary_shift": _format_float(row.abs_boundary_shift),
        "boundary_status": row.boundary_status,
    }


def _diagnostic_row(row: DiagnosticSensitivityRow) -> dict[str, str]:
    return {
        "sensitivity_type": row.sensitivity_type,
        "case_name": row.case_name,
        "option_type": row.option_type,
        "Smax": _format_float(row.Smax),
        "M": str(row.M),
        "N": str(row.N),
        "dS": _format_float(row.dS),
        "all_psor_steps_converged": str(row.all_psor_steps_converged),
        "psor_step_count": str(row.psor_step_count),
        "max_psor_iterations": str(row.max_psor_iterations),
        "mean_psor_iterations": _format_float(row.mean_psor_iterations),
        "max_final_update": _format_float(row.max_final_update),
        "max_obstacle_violation": _format_float(row.max_obstacle_violation),
        "max_equation_violation": _format_float(row.max_equation_violation),
        "max_abs_complementarity_product": _format_float(
            row.max_abs_complementarity_product
        ),
        "max_abs_gamma": _format_float(row.max_abs_gamma),
        "max_abs_gamma_strict": _format_float(row.max_abs_gamma_strict),
        "boundary_near_node_count": str(row.boundary_near_node_count),
        "strict_negative_gamma_count": str(row.strict_negative_gamma_count),
        "runtime_seconds": _format_float(row.runtime_seconds),
    }


def _runs_by_option(
    runs: list[SensitivityRunResult],
) -> dict[str, list[SensitivityRunResult]]:
    grouped: dict[str, list[SensitivityRunResult]] = {"put": [], "call": []}
    for run in runs:
        grouped[run.case.option_type].append(run)
    return grouped


def _reference_runs_by_option(
    runs: list[SensitivityRunResult],
) -> dict[str, SensitivityRunResult]:
    references: dict[str, SensitivityRunResult] = {}
    for option_type, option_runs in _runs_by_option(runs).items():
        if not option_runs:
            continue
        if option_runs[0].case.sensitivity_type == "grid":
            references[option_type] = max(option_runs, key=lambda run: run.case.M)
        else:
            references[option_type] = max(option_runs, key=lambda run: run.case.Smax)
    return references


def _runs_for_option(
    runs: list[SensitivityRunResult],
    option_type: str,
) -> list[SensitivityRunResult]:
    return [run for run in runs if run.case.option_type == option_type]


def _selected_final_value(run: SensitivityRunResult, moneyness: float) -> float:
    target = run.result.K * moneyness
    index = int(np.argmin(np.abs(run.result.spot_grid - target)))
    return float(run.result.values[index])


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _format_float(value: Any) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.12g}"


def _parse_float(value: str) -> float:
    return float(value)


if __name__ == "__main__":
    main()

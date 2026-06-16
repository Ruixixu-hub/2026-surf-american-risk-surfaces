"""Ticket 10A: Rannacher smoothing comparison experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryCurve,
    extract_boundary_curve,
    summarize_boundary_curve,
)
from american_risk_surfaces.diagnostics.greeks import (
    GreekDiagnostics,
    diagnose_greek_result,
)
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    american_crank_nicolson_psor_price,
)
from american_risk_surfaces.solvers.rannacher import (
    RannacherCNPSORResult,
    rannacher_crank_nicolson_psor_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/fontconfig-cache")

SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10a_rannacher_comparison_summary.csv"
)
SELECTED_SPOTS_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10a_rannacher_selected_spots.csv"
)
GREEK_SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10a_rannacher_greek_summary.csv"
)
BOUNDARY_SUMMARY_CSV = PROJECT_ROOT / Path(
    "results/01_solver_validation/tables/ticket_10a_rannacher_boundary_summary.csv"
)
VALUE_PROFILE_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10a_value_profiles.png"
)
PRICE_DIFF_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10a_price_difference_profiles.png"
)
GAMMA_PROFILE_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10a_gamma_profiles.png"
)
GAMMA_FULL_STRICT_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10a_gamma_full_vs_strict.png"
)
BOUNDARY_COMPARISON_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10a_boundary_curve_comparison.png"
)
PSOR_ITERATIONS_FIGURE = PROJECT_ROOT / Path(
    "results/01_solver_validation/figures/ticket_10a_psor_iterations.png"
)

SELECTED_MONEYNESS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
SELECTED_TAU_FRACTIONS = (0.01, 0.5, 1.0)
DEFAULT_RANNACHER_SUBSTEPS = 2

DEFAULT_CASES = [
    {
        "case_name": "american_put_medium",
        "option_type": "put",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.02,
        "sigma": 0.2,
        "Smax": 4.0,
        "M": 120,
        "N": 120,
    },
    {
        "case_name": "dividend_call_medium",
        "option_type": "call",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.08,
        "sigma": 0.2,
        "Smax": 4.0,
        "M": 120,
        "N": 120,
    },
]

SUMMARY_FIELDNAMES = [
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
    "rannacher_substeps",
    "rannacher_substep_size",
    "baseline_converged",
    "rannacher_converged",
    "baseline_max_obstacle_violation",
    "rannacher_max_obstacle_violation",
    "max_abs_price_difference",
    "rmse_price_difference",
    "mean_price_difference",
    "max_price_difference_spot",
    "gate_recommendation",
    "value_figure_created",
    "price_difference_figure_created",
    "gamma_profile_figure_created",
    "gamma_full_strict_figure_created",
    "boundary_figure_created",
    "psor_figure_created",
]

SELECTED_FIELDNAMES = [
    "case_name",
    "option_type",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "payoff",
    "baseline_value",
    "rannacher_value",
    "rannacher_minus_baseline",
]

GREEK_FIELDNAMES = [
    "case_name",
    "option_type",
    "baseline_max_abs_gamma",
    "rannacher_max_abs_gamma",
    "baseline_max_abs_gamma_strict",
    "rannacher_max_abs_gamma_strict",
    "baseline_boundary_near_node_count",
    "rannacher_boundary_near_node_count",
    "baseline_strict_negative_gamma_count",
    "rannacher_strict_negative_gamma_count",
    "strict_gamma_change",
    "full_gamma_change",
]

BOUNDARY_FIELDNAMES = [
    "case_name",
    "option_type",
    "baseline_found_boundary_count",
    "rannacher_found_boundary_count",
    "matched_boundary_count",
    "max_abs_boundary_shift",
    "mean_abs_boundary_shift",
    "first_baseline_boundary_tau",
    "first_rannacher_boundary_tau",
    "boundary_status",
]


def run_comparison_cases(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] = DEFAULT_CASES,
    rannacher_substeps: int = DEFAULT_RANNACHER_SUBSTEPS,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    """Run baseline and Rannacher American cases and compare diagnostics."""

    bundles = []
    for case in cases:
        solver_kwargs = dict(case)
        case_name = str(solver_kwargs.pop("case_name"))
        baseline = american_crank_nicolson_psor_price(**solver_kwargs)
        smoothed_wrapper = rannacher_crank_nicolson_psor_price(
            **solver_kwargs,
            rannacher_substeps=rannacher_substeps,
        )
        smoothed = smoothed_wrapper.result
        baseline_curve = extract_boundary_curve(baseline, f"{case_name}_baseline")
        smoothed_curve = extract_boundary_curve(smoothed, f"{case_name}_rannacher")
        baseline_greeks = diagnose_greek_result(
            baseline, f"{case_name}_baseline", boundary_curve=baseline_curve
        )
        smoothed_greeks = diagnose_greek_result(
            smoothed, f"{case_name}_rannacher", boundary_curve=smoothed_curve
        )
        bundles.append(
            {
                "case_name": case_name,
                "baseline": baseline,
                "smoothed_wrapper": smoothed_wrapper,
                "smoothed": smoothed,
                "baseline_curve": baseline_curve,
                "smoothed_curve": smoothed_curve,
                "baseline_greeks": baseline_greeks,
                "smoothed_greeks": smoothed_greeks,
            }
        )

    greek_rows = [
        greek_summary_row(
            bundle["case_name"],
            bundle["baseline_greeks"],
            bundle["smoothed_greeks"],
        )
        for bundle in bundles
    ]
    boundary_rows = [
        boundary_summary_row(
            bundle["case_name"],
            bundle["baseline_curve"],
            bundle["smoothed_curve"],
        )
        for bundle in bundles
    ]

    value_figure_created = create_value_profiles_figure(bundles, VALUE_PROFILE_FIGURE)
    price_difference_figure_created = create_price_difference_figure(
        bundles, PRICE_DIFF_FIGURE
    )
    gamma_profile_figure_created = create_gamma_profiles_figure(
        bundles, GAMMA_PROFILE_FIGURE
    )
    gamma_full_strict_figure_created = create_gamma_full_strict_figure(
        bundles, GAMMA_FULL_STRICT_FIGURE
    )
    boundary_figure_created = create_boundary_comparison_figure(
        bundles, BOUNDARY_COMPARISON_FIGURE
    )
    psor_figure_created = create_psor_iterations_figure(bundles, PSOR_ITERATIONS_FIGURE)
    figure_flags = {
        "value_figure_created": str(value_figure_created),
        "price_difference_figure_created": str(price_difference_figure_created),
        "gamma_profile_figure_created": str(gamma_profile_figure_created),
        "gamma_full_strict_figure_created": str(gamma_full_strict_figure_created),
        "boundary_figure_created": str(boundary_figure_created),
        "psor_figure_created": str(psor_figure_created),
    }

    summary_rows = []
    selected_rows = []
    for bundle, greek_row, boundary_row in zip(bundles, greek_rows, boundary_rows):
        summary_rows.append(
            comparison_summary_row(
                bundle["case_name"],
                bundle["baseline"],
                bundle["smoothed_wrapper"],
                greek_row,
                boundary_row,
            )
            | figure_flags
        )
        selected_rows.extend(
            selected_spot_rows(
                bundle["case_name"],
                bundle["baseline"],
                bundle["smoothed"],
            )
        )

    metadata = {
        "summary_csv": str(SUMMARY_CSV),
        "selected_spots_csv": str(SELECTED_SPOTS_CSV),
        "greek_summary_csv": str(GREEK_SUMMARY_CSV),
        "boundary_summary_csv": str(BOUNDARY_SUMMARY_CSV),
        "value_figure_path": str(VALUE_PROFILE_FIGURE),
        "price_difference_figure_path": str(PRICE_DIFF_FIGURE),
        "gamma_profile_figure_path": str(GAMMA_PROFILE_FIGURE),
        "gamma_full_strict_figure_path": str(GAMMA_FULL_STRICT_FIGURE),
        "boundary_figure_path": str(BOUNDARY_COMPARISON_FIGURE),
        "psor_figure_path": str(PSOR_ITERATIONS_FIGURE),
        "value_figure_created": str(value_figure_created),
        "price_difference_figure_created": str(price_difference_figure_created),
        "gamma_profile_figure_created": str(gamma_profile_figure_created),
        "gamma_full_strict_figure_created": str(gamma_full_strict_figure_created),
        "boundary_figure_created": str(boundary_figure_created),
        "psor_figure_created": str(psor_figure_created),
    }
    return summary_rows, selected_rows, greek_rows, boundary_rows, metadata


def comparison_summary_row(
    case_name: str,
    baseline: AmericanCNPSORResult,
    smoothed_wrapper: RannacherCNPSORResult,
    greek_row: dict[str, str],
    boundary_row: dict[str, str],
) -> dict[str, str]:
    """Summarize price, convergence, obstacle, and gate evidence for one case."""

    smoothed = smoothed_wrapper.result
    difference = smoothed.values - baseline.values
    max_index = int(np.argmax(np.abs(difference)))
    max_abs_difference = float(np.max(np.abs(difference)))
    rmse_difference = float(np.sqrt(np.mean(difference**2)))
    mean_difference = float(np.mean(difference))
    gate = gate_recommendation(
        max_abs_difference,
        baseline.converged,
        smoothed.converged,
        greek_row,
        boundary_row,
    )
    return {
        "case_name": case_name,
        "option_type": baseline.option_type,
        "K": _format_float(baseline.K),
        "T": _format_float(baseline.T),
        "r": _format_float(baseline.r),
        "q": _format_float(baseline.q),
        "sigma": _format_float(baseline.sigma),
        "Smax": _format_float(baseline.Smax),
        "M": str(baseline.M),
        "N": str(baseline.N),
        "rannacher_substeps": str(smoothed_wrapper.metadata.rannacher_substeps),
        "rannacher_substep_size": _format_float(
            smoothed_wrapper.metadata.rannacher_substep_size
        ),
        "baseline_converged": str(baseline.converged),
        "rannacher_converged": str(smoothed.converged),
        "baseline_max_obstacle_violation": _format_float(
            baseline.max_obstacle_violation
        ),
        "rannacher_max_obstacle_violation": _format_float(
            smoothed.max_obstacle_violation
        ),
        "max_abs_price_difference": _format_float(max_abs_difference),
        "rmse_price_difference": _format_float(rmse_difference),
        "mean_price_difference": _format_float(mean_difference),
        "max_price_difference_spot": _format_float(baseline.spot_grid[max_index]),
        "gate_recommendation": gate,
        "value_figure_created": "False",
        "price_difference_figure_created": "False",
        "gamma_profile_figure_created": "False",
        "gamma_full_strict_figure_created": "False",
        "boundary_figure_created": "False",
        "psor_figure_created": "False",
    }


def selected_spot_rows(
    case_name: str,
    baseline: AmericanCNPSORResult,
    smoothed: AmericanCNPSORResult,
    selected_moneyness: tuple[float, ...] = SELECTED_MONEYNESS,
) -> list[dict[str, str]]:
    """Return final-time selected spot comparison rows."""

    rows = []
    for moneyness in selected_moneyness:
        target_spot = baseline.K * float(moneyness)
        index = int(np.argmin(np.abs(baseline.spot_grid - target_spot)))
        nearest_spot = float(baseline.spot_grid[index])
        rows.append(
            {
                "case_name": case_name,
                "option_type": baseline.option_type,
                "target_moneyness": _format_float(moneyness),
                "nearest_spot": _format_float(nearest_spot),
                "actual_moneyness": _format_float(nearest_spot / baseline.K),
                "payoff": _format_float(baseline.payoff[index]),
                "baseline_value": _format_float(baseline.values[index]),
                "rannacher_value": _format_float(smoothed.values[index]),
                "rannacher_minus_baseline": _format_float(
                    smoothed.values[index] - baseline.values[index]
                ),
            }
        )
    return rows


def greek_summary_row(
    case_name: str,
    baseline: GreekDiagnostics,
    smoothed: GreekDiagnostics,
) -> dict[str, str]:
    """Compare baseline and smoothed Gamma diagnostic summaries."""

    return {
        "case_name": case_name,
        "option_type": baseline.option_type,
        "baseline_max_abs_gamma": _format_float(baseline.summary.max_abs_gamma),
        "rannacher_max_abs_gamma": _format_float(smoothed.summary.max_abs_gamma),
        "baseline_max_abs_gamma_strict": _format_float(
            baseline.summary.max_abs_gamma_strict
        ),
        "rannacher_max_abs_gamma_strict": _format_float(
            smoothed.summary.max_abs_gamma_strict
        ),
        "baseline_boundary_near_node_count": str(
            baseline.summary.boundary_near_node_count
        ),
        "rannacher_boundary_near_node_count": str(
            smoothed.summary.boundary_near_node_count
        ),
        "baseline_strict_negative_gamma_count": str(
            baseline.summary.strict_negative_gamma_count
        ),
        "rannacher_strict_negative_gamma_count": str(
            smoothed.summary.strict_negative_gamma_count
        ),
        "strict_gamma_change": _format_float(
            smoothed.summary.max_abs_gamma_strict
            - baseline.summary.max_abs_gamma_strict
        ),
        "full_gamma_change": _format_float(
            smoothed.summary.max_abs_gamma - baseline.summary.max_abs_gamma
        ),
    }


def boundary_summary_row(
    case_name: str,
    baseline: BoundaryCurve,
    smoothed: BoundaryCurve,
) -> dict[str, str]:
    """Compare threshold-based baseline and smoothed boundary curves."""

    baseline_summary = summarize_boundary_curve(baseline)
    smoothed_summary = summarize_boundary_curve(smoothed)
    shifts = []
    for baseline_point, smoothed_point in zip(baseline.points, smoothed.points):
        if baseline_point.boundary_found and smoothed_point.boundary_found:
            shifts.append(abs(smoothed_point.boundary_spot - baseline_point.boundary_spot))
    if shifts:
        max_shift = float(np.max(shifts))
        mean_shift = float(np.mean(shifts))
        status = "COMPARABLE"
    else:
        max_shift = float("nan")
        mean_shift = float("nan")
        status = "NO_MATCHED_BOUNDARIES"
    return {
        "case_name": case_name,
        "option_type": baseline.option_type,
        "baseline_found_boundary_count": str(baseline_summary.found_boundary_count),
        "rannacher_found_boundary_count": str(smoothed_summary.found_boundary_count),
        "matched_boundary_count": str(len(shifts)),
        "max_abs_boundary_shift": _format_float(max_shift),
        "mean_abs_boundary_shift": _format_float(mean_shift),
        "first_baseline_boundary_tau": _format_float(
            baseline_summary.first_boundary_tau
        ),
        "first_rannacher_boundary_tau": _format_float(
            smoothed_summary.first_boundary_tau
        ),
        "boundary_status": status,
    }


def gate_recommendation(
    max_abs_price_difference: float,
    baseline_converged: bool,
    smoothed_converged: bool,
    greek_row: dict[str, str],
    boundary_row: dict[str, str],
) -> str:
    """Return a conservative gate recommendation from comparison evidence."""

    if not baseline_converged or not smoothed_converged:
        return "INCONCLUSIVE_NEEDS_GRID_SENSITIVITY"
    boundary_shift = _parse_float(boundary_row["max_abs_boundary_shift"])
    boundary_distorted = np.isfinite(boundary_shift) and boundary_shift > 0.1
    if max_abs_price_difference > 0.01 or boundary_distorted:
        return "INCONCLUSIVE_NEEDS_GRID_SENSITIVITY"

    baseline_strict = _parse_float(greek_row["baseline_max_abs_gamma_strict"])
    smoothed_strict = _parse_float(greek_row["rannacher_max_abs_gamma_strict"])
    baseline_full = _parse_float(greek_row["baseline_max_abs_gamma"])
    smoothed_full = _parse_float(greek_row["rannacher_max_abs_gamma"])
    strict_improves = (
        np.isfinite(baseline_strict)
        and np.isfinite(smoothed_strict)
        and smoothed_strict <= 0.95 * baseline_strict
    )
    full_improves = (
        np.isfinite(baseline_full)
        and np.isfinite(smoothed_full)
        and smoothed_full <= 0.95 * baseline_full
    )
    if strict_improves or full_improves:
        return "USE_RANNACHER_FOR_GREEK_DIAGNOSTICS"
    return "KEEP_BASELINE_FOR_NOW"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_value_profiles_figure(bundles: list[dict[str, Any]], path: Path) -> bool:
    """Plot baseline and Rannacher final values against payoff."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bundles), figsize=(11, 4), squeeze=False)
    for axis, bundle in zip(axes[0], bundles):
        baseline = bundle["baseline"]
        smoothed = bundle["smoothed"]
        moneyness = baseline.spot_grid / baseline.K
        axis.plot(moneyness, baseline.values, label="baseline CN/PSOR", linewidth=1.5)
        axis.plot(
            moneyness,
            smoothed.values,
            linestyle="--",
            label="Rannacher variant",
            linewidth=1.5,
        )
        axis.plot(moneyness, baseline.payoff, label="payoff", linewidth=1.0)
        axis.set_title(bundle["case_name"])
        axis.set_xlabel("S / K")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_price_difference_figure(bundles: list[dict[str, Any]], path: Path) -> bool:
    """Plot smoothed-minus-baseline final price differences."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bundles), figsize=(11, 4), squeeze=False)
    for axis, bundle in zip(axes[0], bundles):
        baseline = bundle["baseline"]
        smoothed = bundle["smoothed"]
        moneyness = baseline.spot_grid / baseline.K
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(moneyness, smoothed.values - baseline.values, linewidth=1.5)
        axis.set_title(bundle["case_name"])
        axis.set_xlabel("S / K")
        axis.set_ylabel("Rannacher - baseline")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_gamma_profiles_figure(bundles: list[dict[str, Any]], path: Path) -> bool:
    """Plot selected baseline and smoothed Gamma profiles."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bundles), figsize=(12, 4), squeeze=False)
    for axis, bundle in zip(axes[0], bundles):
        baseline = bundle["baseline_greeks"]
        smoothed = bundle["smoothed_greeks"]
        moneyness = baseline.spot_grid / baseline.summary.K
        for fraction in (0.01, 1.0):
            target_tau = baseline.summary.T * fraction
            time_index = int(np.argmin(np.abs(baseline.tau_grid - target_tau)))
            tau_text = f"tau={baseline.tau_grid[time_index]:.3f}"
            axis.plot(
                moneyness,
                baseline.arrays.gamma[time_index],
                linewidth=1.1,
                label=f"baseline {tau_text}",
            )
            axis.plot(
                moneyness,
                smoothed.arrays.gamma[time_index],
                linestyle="--",
                linewidth=1.1,
                label=f"Rannacher {tau_text}",
            )
        axis.set_title(bundle["case_name"])
        axis.set_xlabel("S / K")
        axis.set_ylabel("Gamma")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_gamma_full_strict_figure(bundles: list[dict[str, Any]], path: Path) -> bool:
    """Compare full-grid and strict-mask max absolute Gamma over time."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bundles), figsize=(12, 4), squeeze=False)
    for axis, bundle in zip(axes[0], bundles):
        baseline = bundle["baseline_greeks"]
        smoothed = bundle["smoothed_greeks"]
        tau = baseline.tau_grid
        axis.plot(
            tau,
            [row.max_abs_gamma for row in baseline.by_time_rows],
            label="baseline full",
            linewidth=1.1,
        )
        axis.plot(
            tau,
            [row.max_abs_gamma_strict for row in baseline.by_time_rows],
            label="baseline strict",
            linestyle="--",
            linewidth=1.1,
        )
        axis.plot(
            tau,
            [row.max_abs_gamma for row in smoothed.by_time_rows],
            label="Rannacher full",
            linewidth=1.1,
        )
        axis.plot(
            tau,
            [row.max_abs_gamma_strict for row in smoothed.by_time_rows],
            label="Rannacher strict",
            linestyle="--",
            linewidth=1.1,
        )
        axis.set_title(bundle["case_name"])
        axis.set_xlabel("Time to maturity tau")
        axis.set_ylabel("Max |Gamma|")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_boundary_comparison_figure(
    bundles: list[dict[str, Any]],
    path: Path,
) -> bool:
    """Plot threshold-based baseline and Rannacher boundary curves."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bundles), figsize=(11, 4), squeeze=False)
    for axis, bundle in zip(axes[0], bundles):
        for curve, label, marker in (
            (bundle["baseline_curve"], "baseline", "o"),
            (bundle["smoothed_curve"], "Rannacher", "x"),
        ):
            tau = [
                point.tau for point in curve.points if point.boundary_found
            ]
            spot = [
                point.boundary_spot for point in curve.points if point.boundary_found
            ]
            axis.plot(tau, spot, marker=marker, markersize=2.5, linewidth=1.1, label=label)
        axis.set_title(bundle["case_name"])
        axis.set_xlabel("Time to maturity tau")
        axis.set_ylabel("Approximate boundary spot")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_psor_iterations_figure(bundles: list[dict[str, Any]], path: Path) -> bool:
    """Plot baseline and smoothed PSOR iteration counts."""

    plt = _load_pyplot()
    if plt is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bundles), figsize=(12, 4), squeeze=False)
    for axis, bundle in zip(axes[0], bundles):
        baseline = bundle["baseline"]
        smoothed_wrapper = bundle["smoothed_wrapper"]
        baseline_iterations = [result.iterations for result in baseline.psor_results]
        smoothed_iterations = [
            result.iterations for result in smoothed_wrapper.result.psor_results
        ]
        axis.plot(baseline_iterations, label="baseline", linewidth=1.2)
        axis.plot(smoothed_iterations, label="Rannacher", linestyle="--", linewidth=1.2)
        if smoothed_wrapper.metadata.rannacher_psor_results:
            axis.axvline(
                len(smoothed_wrapper.metadata.rannacher_psor_results) - 0.5,
                color="black",
                linewidth=0.8,
                alpha=0.5,
            )
        axis.set_title(bundle["case_name"])
        axis.set_xlabel("Solve index")
        axis.set_ylabel("PSOR iterations")
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
    dict[str, str],
]:
    """Run the Ticket 10A comparison experiment and write artifacts."""

    summary_rows, selected_rows, greek_rows, boundary_rows, metadata = run_comparison_cases()
    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(SELECTED_SPOTS_CSV, selected_rows, SELECTED_FIELDNAMES)
    write_csv(GREEK_SUMMARY_CSV, greek_rows, GREEK_FIELDNAMES)
    write_csv(BOUNDARY_SUMMARY_CSV, boundary_rows, BOUNDARY_FIELDNAMES)
    print(f"wrote {len(summary_rows)} rows to {SUMMARY_CSV}")
    print(f"wrote {len(selected_rows)} rows to {SELECTED_SPOTS_CSV}")
    print(f"wrote {len(greek_rows)} rows to {GREEK_SUMMARY_CSV}")
    print(f"wrote {len(boundary_rows)} rows to {BOUNDARY_SUMMARY_CSV}")
    print(
        "figures_created="
        f"{metadata['value_figure_created']},"
        f"{metadata['price_difference_figure_created']},"
        f"{metadata['gamma_profile_figure_created']},"
        f"{metadata['gamma_full_strict_figure_created']},"
        f"{metadata['boundary_figure_created']},"
        f"{metadata['psor_figure_created']}"
    )
    return summary_rows, selected_rows, greek_rows, boundary_rows, metadata


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

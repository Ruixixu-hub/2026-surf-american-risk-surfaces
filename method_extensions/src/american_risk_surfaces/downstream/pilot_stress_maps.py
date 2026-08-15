"""Pilot 01: controlled pilot stress maps for validated American CN/PSOR solver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryCurve,
    BoundaryExtractionSummary,
    continuation_premium,
    extract_boundary_curve,
    summarize_boundary_curve,
)
from american_risk_surfaces.diagnostics.greeks import (
    GreekDiagnostics,
    diagnose_greek_result,
)
from american_risk_surfaces.diagnostics.lcp import LCPDiagnostics, diagnose_lcp_result
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    american_crank_nicolson_psor_price,
)

SOLVER_NAME = "american_crank_nicolson_psor_price"
SOLVER_VARIANT = "baseline_cn_psor"
PREMIUM_THRESHOLD = 1e-6
INTERPRETATION_MONEYNESS_BOUNDS = (0.4, 1.8)
SELECTED_TAU_FRACTIONS = (0.01, 0.25, 0.50, 0.75, 1.00)
SELECTED_MONEYNESS = (0.5, 0.8, 1.0, 1.2, 1.5, 1.8)
DOWNSTREAM_USE_STATUS = "pilot_diagnostic_only"

RUN_SUMMARY_FIELDNAMES = [
    "case_name",
    "case_family",
    "variation_name",
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
    "solver_name",
    "solver_variant",
    "premium_threshold",
    "interpretation_lower_moneyness",
    "interpretation_upper_moneyness",
    "is_base_case",
    "is_higher_grid_check",
    "runtime_seconds",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "boundary_found_count",
    "boundary_status",
    "greek_status",
    "acceptance_status",
    "downstream_use_status",
]

DIAGNOSTIC_SUMMARY_FIELDNAMES = [
    "case_name",
    "case_family",
    "variation_name",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "max_obstacle_violation",
    "min_value_gap",
    "max_equation_violation",
    "min_equation_gap",
    "max_abs_complementarity_product",
    "mean_max_abs_complementarity_product",
    "lcp_status",
    "finite_delta_count",
    "finite_gamma_count",
    "nonfinite_delta_count",
    "nonfinite_gamma_count",
    "max_abs_gamma",
    "max_abs_gamma_away_from_boundary",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "kink_near_node_count",
    "maturity_masked_node_count",
    "strict_interior_node_count",
    "strict_delta_lower_violation_count",
    "strict_delta_upper_violation_count",
    "strict_negative_gamma_count",
    "greek_status",
    "downstream_use_status",
]

BOUNDARY_SUMMARY_FIELDNAMES = [
    "case_name",
    "case_family",
    "variation_name",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "threshold",
    "search_direction",
    "total_time_rows",
    "positive_tau_rows",
    "found_boundary_count",
    "no_boundary_count",
    "maturity_ambiguous_count",
    "all_continuation_count",
    "all_exercise_count",
    "expected_exercise_side_absent_count",
    "no_clean_transition_count",
    "insufficient_interior_nodes_count",
    "first_boundary_tau",
    "last_boundary_tau",
    "min_boundary_spot",
    "max_boundary_spot",
    "status",
    "downstream_use_status",
]

SELECTED_SLICE_FIELDNAMES = [
    "case_name",
    "case_family",
    "variation_name",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "solver_name",
    "solver_variant",
    "premium_threshold",
    "target_tau_fraction",
    "target_tau",
    "nearest_tau",
    "time_index",
    "target_moneyness",
    "nearest_spot",
    "actual_moneyness",
    "value",
    "payoff",
    "continuation_premium",
    "exercise_indicator",
    "boundary_found_at_time",
    "boundary_spot_at_time",
    "boundary_distance",
    "delta",
    "gamma",
    "boundary_near",
    "kink_near",
    "maturity_row",
    "strict_interior",
    "downstream_use_status",
]

OUTPUT_MANIFEST_FIELDNAMES = [
    "output_id",
    "output_type",
    "case_name",
    "case_family",
    "variation_name",
    "path",
    "created",
    "description",
    "solver_name",
    "solver_variant",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "premium_threshold",
    "interpretation_lower_moneyness",
    "interpretation_upper_moneyness",
    "contains_greek_diagnostics",
    "contains_boundary_overlay",
    "downstream_use_status",
    "review_status",
]

__all__ = (
    "PilotCase",
    "PilotRunArtifacts",
    "SOLVER_NAME",
    "SOLVER_VARIANT",
    "PREMIUM_THRESHOLD",
    "INTERPRETATION_MONEYNESS_BOUNDS",
    "SELECTED_TAU_FRACTIONS",
    "SELECTED_MONEYNESS",
    "DOWNSTREAM_USE_STATUS",
    "RUN_SUMMARY_FIELDNAMES",
    "DIAGNOSTIC_SUMMARY_FIELDNAMES",
    "BOUNDARY_SUMMARY_FIELDNAMES",
    "SELECTED_SLICE_FIELDNAMES",
    "OUTPUT_MANIFEST_FIELDNAMES",
    "pilot_cases",
    "run_pilot_case",
    "continuation_premium_grid",
    "exercise_indicator",
    "selected_slice_rows",
    "run_summary_row",
    "diagnostic_summary_row",
    "boundary_summary_row",
    "manifest_row",
    "write_csv",
    "create_base_value_heatmaps",
    "create_base_premium_heatmaps",
    "create_base_indicator_boundary_maps",
    "create_base_selected_value_slices",
    "create_boundary_variation_comparison",
    "create_premium_slice_variation_comparison",
    "create_greek_diagnostic_slices",
)


@dataclass(frozen=True)
class PilotCase:
    """One controlled Pilot 01 stress-map case."""

    case_name: str
    case_family: str
    variation_name: str
    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int
    is_base_case: bool = False
    is_higher_grid_check: bool = False


@dataclass(frozen=True)
class PilotRunArtifacts:
    """Solver output and diagnostics for one controlled pilot run."""

    case: PilotCase
    result: AmericanCNPSORResult
    premium_grid: np.ndarray
    exercise_indicator_grid: np.ndarray
    boundary_curve: BoundaryCurve
    boundary_summary: BoundaryExtractionSummary
    lcp_diagnostics: LCPDiagnostics
    greek_diagnostics: GreekDiagnostics
    runtime_seconds: float
    dS: float
    dtau: float
    premium_threshold: float
    solver_name: str = SOLVER_NAME
    solver_variant: str = SOLVER_VARIANT


def pilot_cases(include_higher_grid_checks: bool = True) -> tuple[PilotCase, ...]:
    """Return the approved small Pilot 01 case set."""

    cases = [
        PilotCase(
            "american_put_base",
            "american_put",
            "base",
            "put",
            1.0,
            1.0,
            0.05,
            0.02,
            0.20,
            4.0,
            120,
            120,
            is_base_case=True,
        ),
        PilotCase(
            "american_put_sigma_040",
            "american_put",
            "sigma_040",
            "put",
            1.0,
            1.0,
            0.05,
            0.02,
            0.40,
            4.0,
            120,
            120,
        ),
        PilotCase(
            "american_put_r_003",
            "american_put",
            "r_003",
            "put",
            1.0,
            1.0,
            0.03,
            0.02,
            0.20,
            4.0,
            120,
            120,
        ),
        PilotCase(
            "american_put_q_000",
            "american_put",
            "q_000",
            "put",
            1.0,
            1.0,
            0.05,
            0.00,
            0.20,
            4.0,
            120,
            120,
        ),
        PilotCase(
            "dividend_call_base",
            "dividend_call",
            "base",
            "call",
            1.0,
            1.0,
            0.05,
            0.08,
            0.20,
            4.0,
            120,
            120,
            is_base_case=True,
        ),
        PilotCase(
            "dividend_call_sigma_040",
            "dividend_call",
            "sigma_040",
            "call",
            1.0,
            1.0,
            0.05,
            0.08,
            0.40,
            4.0,
            120,
            120,
        ),
        PilotCase(
            "dividend_call_r_003",
            "dividend_call",
            "r_003",
            "call",
            1.0,
            1.0,
            0.03,
            0.08,
            0.20,
            4.0,
            120,
            120,
        ),
        PilotCase(
            "dividend_call_q_003",
            "dividend_call",
            "q_003",
            "call",
            1.0,
            1.0,
            0.05,
            0.03,
            0.20,
            4.0,
            120,
            120,
        ),
        PilotCase(
            "dividend_call_q_010",
            "dividend_call",
            "q_010",
            "call",
            1.0,
            1.0,
            0.05,
            0.10,
            0.20,
            4.0,
            120,
            120,
        ),
    ]
    if include_higher_grid_checks:
        cases.extend(
            [
                PilotCase(
                    "american_put_high_grid_check",
                    "american_put",
                    "high_grid_check",
                    "put",
                    1.0,
                    1.0,
                    0.05,
                    0.02,
                    0.20,
                    4.0,
                    180,
                    180,
                    is_higher_grid_check=True,
                ),
                PilotCase(
                    "dividend_call_high_grid_check",
                    "dividend_call",
                    "high_grid_check",
                    "call",
                    1.0,
                    1.0,
                    0.05,
                    0.08,
                    0.20,
                    4.0,
                    180,
                    180,
                    is_higher_grid_check=True,
                ),
            ]
        )
    return tuple(cases)


def run_pilot_case(
    case: PilotCase,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> PilotRunArtifacts:
    """Run the baseline solver and attach Pilot 01 diagnostics."""

    validated = _validate_case(case)
    threshold = _validate_threshold(premium_threshold)
    start = perf_counter()
    result = american_crank_nicolson_psor_price(
        option_type=validated.option_type,
        K=validated.K,
        T=validated.T,
        r=validated.r,
        q=validated.q,
        sigma=validated.sigma,
        Smax=validated.Smax,
        M=validated.M,
        N=validated.N,
    )
    runtime_seconds = perf_counter() - start
    premium_grid = continuation_premium_grid(result)
    indicator = exercise_indicator(premium_grid, threshold=threshold)
    boundary_curve = extract_boundary_curve(result, validated.case_name, threshold=threshold)
    boundary_summary = summarize_boundary_curve(boundary_curve)
    lcp_diagnostics = diagnose_lcp_result(result, validated.case_name)
    greek_diagnostics = diagnose_greek_result(
        result,
        validated.case_name,
        boundary_curve=boundary_curve,
        boundary_threshold=threshold,
    )
    dS = _spacing(result.spot_grid)
    dtau = _spacing(result.tau_grid)
    return PilotRunArtifacts(
        case=validated,
        result=result,
        premium_grid=premium_grid,
        exercise_indicator_grid=indicator,
        boundary_curve=boundary_curve,
        boundary_summary=boundary_summary,
        lcp_diagnostics=lcp_diagnostics,
        greek_diagnostics=greek_diagnostics,
        runtime_seconds=float(runtime_seconds),
        dS=dS,
        dtau=dtau,
        premium_threshold=threshold,
    )


def continuation_premium_grid(result: Any) -> np.ndarray:
    """Return the Pilot 01 continuation premium grid, U minus payoff."""

    return continuation_premium(result.value_grid, result.payoff)


def exercise_indicator(premium_grid: Any, threshold: float = PREMIUM_THRESHOLD) -> np.ndarray:
    """Return 1 for exercise-like nodes and 0 for continuation-like nodes."""

    premium = np.asarray(premium_grid, dtype=float)
    if premium.ndim not in (1, 2):
        raise ValueError("premium_grid must be one- or two-dimensional.")
    if np.any(~np.isfinite(premium)):
        raise ValueError("premium_grid must contain finite values.")
    threshold_value = _validate_threshold(threshold)
    return (premium <= threshold_value).astype(int)


def selected_slice_rows(
    artifacts: PilotRunArtifacts,
    selected_tau_fractions: tuple[float, ...] = SELECTED_TAU_FRACTIONS,
    selected_moneyness: tuple[float, ...] = SELECTED_MONEYNESS,
) -> list[dict[str, str]]:
    """Return selected tau/moneyness rows for pilot interpretation."""

    if not isinstance(artifacts, PilotRunArtifacts):
        raise ValueError("artifacts must be a PilotRunArtifacts instance.")
    rows: list[dict[str, str]] = []
    result = artifacts.result
    case = artifacts.case
    for tau_fraction in selected_tau_fractions:
        fraction = _validate_fraction(tau_fraction)
        target_tau = result.T * fraction
        time_index = int(np.argmin(np.abs(result.tau_grid - target_tau)))
        nearest_tau = float(result.tau_grid[time_index])
        boundary_point = artifacts.boundary_curve.points[time_index]
        for moneyness in selected_moneyness:
            money = _validate_nonnegative("moneyness", moneyness)
            target_spot = case.K * money
            spot_index = int(np.argmin(np.abs(result.spot_grid - target_spot)))
            nearest_spot = float(result.spot_grid[spot_index])
            boundary_spot = (
                float(boundary_point.boundary_spot)
                if boundary_point.boundary_found
                else float("nan")
            )
            boundary_distance = (
                abs(nearest_spot - boundary_spot)
                if boundary_point.boundary_found
                else float("nan")
            )
            rows.append(
                _case_metadata(case)
                | {
                    "solver_name": SOLVER_NAME,
                    "solver_variant": SOLVER_VARIANT,
                    "premium_threshold": _format_float(artifacts.premium_threshold),
                    "target_tau_fraction": _format_float(fraction),
                    "target_tau": _format_float(target_tau),
                    "nearest_tau": _format_float(nearest_tau),
                    "time_index": str(time_index),
                    "target_moneyness": _format_float(money),
                    "nearest_spot": _format_float(nearest_spot),
                    "actual_moneyness": _format_float(nearest_spot / case.K),
                    "value": _format_float(result.value_grid[time_index, spot_index]),
                    "payoff": _format_float(result.payoff[spot_index]),
                    "continuation_premium": _format_float(
                        artifacts.premium_grid[time_index, spot_index]
                    ),
                    "exercise_indicator": str(
                        int(artifacts.exercise_indicator_grid[time_index, spot_index])
                    ),
                    "boundary_found_at_time": str(boundary_point.boundary_found),
                    "boundary_spot_at_time": _format_float(boundary_spot),
                    "boundary_distance": _format_float(boundary_distance),
                    "delta": _format_float(
                        artifacts.greek_diagnostics.arrays.delta[time_index, spot_index]
                    ),
                    "gamma": _format_float(
                        artifacts.greek_diagnostics.arrays.gamma[time_index, spot_index]
                    ),
                    "boundary_near": str(
                        bool(artifacts.greek_diagnostics.masks.boundary_near[time_index, spot_index])
                    ),
                    "kink_near": str(
                        bool(
                            artifacts.greek_diagnostics.masks.payoff_kink_near[
                                time_index, spot_index
                            ]
                        )
                    ),
                    "maturity_row": str(
                        bool(artifacts.greek_diagnostics.masks.maturity_row[time_index, spot_index])
                    ),
                    "strict_interior": str(
                        bool(
                            artifacts.greek_diagnostics.masks.strict_interior[
                                time_index, spot_index
                            ]
                        )
                    ),
                    "downstream_use_status": DOWNSTREAM_USE_STATUS,
                }
            )
    return rows


def run_summary_row(artifacts: PilotRunArtifacts) -> dict[str, str]:
    """Build a run-level CSV row."""

    if not isinstance(artifacts, PilotRunArtifacts):
        raise ValueError("artifacts must be a PilotRunArtifacts instance.")
    summary = artifacts.lcp_diagnostics.summary
    case = artifacts.case
    return (
        _case_metadata(case)
        | _grid_metadata(artifacts)
        | {
            "solver_name": SOLVER_NAME,
            "solver_variant": SOLVER_VARIANT,
            "premium_threshold": _format_float(artifacts.premium_threshold),
            "interpretation_lower_moneyness": _format_float(
                INTERPRETATION_MONEYNESS_BOUNDS[0]
            ),
            "interpretation_upper_moneyness": _format_float(
                INTERPRETATION_MONEYNESS_BOUNDS[1]
            ),
            "is_base_case": str(case.is_base_case),
            "is_higher_grid_check": str(case.is_higher_grid_check),
            "runtime_seconds": _format_float(artifacts.runtime_seconds),
            "all_psor_steps_converged": str(summary.all_psor_steps_converged),
            "psor_step_count": str(summary.psor_step_count),
            "max_psor_iterations": str(summary.max_psor_iterations),
            "mean_psor_iterations": _format_float(summary.mean_psor_iterations),
            "max_final_update": _format_float(summary.max_final_update),
            "max_obstacle_violation": _format_float(summary.max_obstacle_violation),
            "boundary_found_count": str(artifacts.boundary_summary.found_boundary_count),
            "boundary_status": artifacts.boundary_summary.status,
            "greek_status": artifacts.greek_diagnostics.summary.status,
            "acceptance_status": _acceptance_status(artifacts),
            "downstream_use_status": DOWNSTREAM_USE_STATUS,
        }
    )


def diagnostic_summary_row(artifacts: PilotRunArtifacts) -> dict[str, str]:
    """Build an LCP/Greek diagnostic CSV row."""

    if not isinstance(artifacts, PilotRunArtifacts):
        raise ValueError("artifacts must be a PilotRunArtifacts instance.")
    lcp = artifacts.lcp_diagnostics.summary
    greek = artifacts.greek_diagnostics.summary
    return _case_metadata(artifacts.case) | {
        "max_obstacle_violation": _format_float(lcp.max_obstacle_violation),
        "min_value_gap": _format_float(lcp.min_value_gap),
        "max_equation_violation": _format_float(lcp.max_equation_violation),
        "min_equation_gap": _format_float(lcp.min_equation_gap),
        "max_abs_complementarity_product": _format_float(
            lcp.max_abs_complementarity_product
        ),
        "mean_max_abs_complementarity_product": _format_float(
            lcp.mean_max_abs_complementarity_product
        ),
        "lcp_status": lcp.status,
        "finite_delta_count": str(greek.finite_delta_count),
        "finite_gamma_count": str(greek.finite_gamma_count),
        "nonfinite_delta_count": str(greek.nonfinite_delta_count),
        "nonfinite_gamma_count": str(greek.nonfinite_gamma_count),
        "max_abs_gamma": _format_float(greek.max_abs_gamma),
        "max_abs_gamma_away_from_boundary": _format_float(
            greek.max_abs_gamma_away_from_boundary
        ),
        "max_abs_gamma_strict": _format_float(greek.max_abs_gamma_strict),
        "boundary_near_node_count": str(greek.boundary_near_node_count),
        "kink_near_node_count": str(greek.kink_near_node_count),
        "maturity_masked_node_count": str(greek.maturity_masked_node_count),
        "strict_interior_node_count": str(greek.strict_interior_node_count),
        "strict_delta_lower_violation_count": str(
            greek.strict_delta_lower_violation_count
        ),
        "strict_delta_upper_violation_count": str(
            greek.strict_delta_upper_violation_count
        ),
        "strict_negative_gamma_count": str(greek.strict_negative_gamma_count),
        "greek_status": greek.status,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def boundary_summary_row(artifacts: PilotRunArtifacts) -> dict[str, str]:
    """Build a boundary metadata CSV row."""

    if not isinstance(artifacts, PilotRunArtifacts):
        raise ValueError("artifacts must be a PilotRunArtifacts instance.")
    summary = artifacts.boundary_summary
    return _case_metadata(artifacts.case) | {
        "threshold": _format_float(summary.threshold),
        "search_direction": summary.search_direction,
        "total_time_rows": str(summary.total_time_rows),
        "positive_tau_rows": str(summary.positive_tau_rows),
        "found_boundary_count": str(summary.found_boundary_count),
        "no_boundary_count": str(summary.no_boundary_count),
        "maturity_ambiguous_count": str(summary.maturity_ambiguous_count),
        "all_continuation_count": str(summary.all_continuation_count),
        "all_exercise_count": str(summary.all_exercise_count),
        "expected_exercise_side_absent_count": str(
            summary.expected_exercise_side_absent_count
        ),
        "no_clean_transition_count": str(summary.no_clean_transition_count),
        "insufficient_interior_nodes_count": str(
            summary.insufficient_interior_nodes_count
        ),
        "first_boundary_tau": _format_float(summary.first_boundary_tau),
        "last_boundary_tau": _format_float(summary.last_boundary_tau),
        "min_boundary_spot": _format_float(summary.min_boundary_spot),
        "max_boundary_spot": _format_float(summary.max_boundary_spot),
        "status": summary.status,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def manifest_row(
    path: Path,
    output_type: str,
    output_id: str,
    description: str,
    created: bool,
    artifacts: PilotRunArtifacts | None = None,
    contains_greek_diagnostics: bool = False,
    contains_boundary_overlay: bool = False,
) -> dict[str, str]:
    """Build one output-manifest row."""

    if artifacts is None:
        case_fields = {
            "case_name": "",
            "case_family": "",
            "variation_name": "",
            "K": "",
            "T": "",
            "r": "",
            "q": "",
            "sigma": "",
            "Smax": "",
            "M": "",
            "N": "",
        }
    else:
        case_fields = _case_metadata(artifacts.case)
    return {
        "output_id": output_id,
        "output_type": output_type,
        **case_fields,
        "path": str(path),
        "created": str(bool(created)),
        "description": description,
        "solver_name": SOLVER_NAME,
        "solver_variant": SOLVER_VARIANT,
        "premium_threshold": _format_float(PREMIUM_THRESHOLD),
        "interpretation_lower_moneyness": _format_float(
            INTERPRETATION_MONEYNESS_BOUNDS[0]
        ),
        "interpretation_upper_moneyness": _format_float(
            INTERPRETATION_MONEYNESS_BOUNDS[1]
        ),
        "contains_greek_diagnostics": str(bool(contains_greek_diagnostics)),
        "contains_boundary_overlay": str(bool(contains_boundary_overlay)),
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
        "review_status": "human_review_required_before_scaling",
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_base_value_heatmaps(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Create side-by-side base-case value heatmaps."""

    return _create_heatmap_pair(
        artifacts,
        path,
        data_getter=lambda item: item.result.value_grid,
        title_suffix="value",
        color_label="Option value",
    )


def create_base_premium_heatmaps(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Create side-by-side base-case continuation-premium heatmaps."""

    return _create_heatmap_pair(
        artifacts,
        path,
        data_getter=lambda item: item.premium_grid,
        title_suffix="continuation premium",
        color_label="U - payoff",
    )


def create_base_indicator_boundary_maps(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Create exercise/continuation indicator maps with boundary overlays."""

    plt = _load_pyplot()
    base_runs = _base_runs(artifacts)
    if plt is None or not base_runs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(base_runs), figsize=(6 * len(base_runs), 4), squeeze=False)
    for axis, item in zip(axes[0], base_runs):
        x, data = _interpretation_region(item, item.exercise_indicator_grid)
        image = axis.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[float(x[0] / item.case.K), float(x[-1] / item.case.K), 0.0, item.case.T],
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        _plot_boundary_overlay(axis, item)
        axis.set_title(f"{item.case.case_family}: exercise indicator")
        axis.set_xlabel("S/K")
        axis.set_ylabel("tau/T")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="1=exercise-like")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_base_selected_value_slices(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Plot selected value slices for the base put and call cases."""

    plt = _load_pyplot()
    base_runs = _base_runs(artifacts)
    if plt is None or not base_runs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(base_runs), figsize=(6 * len(base_runs), 4), squeeze=False)
    for axis, item in zip(axes[0], base_runs):
        x = item.result.spot_grid / item.case.K
        mask = _interpretation_mask(x)
        for fraction in SELECTED_TAU_FRACTIONS:
            time_index = _nearest_tau_index(item, fraction)
            axis.plot(
                x[mask],
                item.result.value_grid[time_index, mask],
                linewidth=1.0,
                label=f"tau/T={fraction:g}",
            )
        axis.plot(x[mask], item.result.payoff[mask], "k--", linewidth=0.9, label="payoff")
        axis.set_title(f"{item.case.case_family}: value slices")
        axis.set_xlabel("S/K")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_boundary_variation_comparison(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Plot approximate boundary curves across one-at-a-time variations."""

    plt = _load_pyplot()
    if plt is None or not artifacts:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    families = _families_in_order(artifacts)
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4), squeeze=False)
    for axis, family in zip(axes[0], families):
        for item in artifacts:
            if item.case.case_family != family:
                continue
            points = [point for point in item.boundary_curve.points if point.boundary_found]
            if not points:
                continue
            tau = np.array([point.tau / item.case.T for point in points], dtype=float)
            boundary = np.array([point.boundary_spot / item.case.K for point in points], dtype=float)
            axis.plot(tau, boundary, linewidth=1.0, label=item.case.variation_name)
        axis.set_title(f"{family}: threshold boundary")
        axis.set_xlabel("tau/T")
        axis.set_ylabel("boundary S/K")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_premium_slice_variation_comparison(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Plot mid-time continuation-premium slices across variations."""

    plt = _load_pyplot()
    if plt is None or not artifacts:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    families = _families_in_order(artifacts)
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4), squeeze=False)
    for axis, family in zip(axes[0], families):
        for item in artifacts:
            if item.case.case_family != family or item.case.is_higher_grid_check:
                continue
            x = item.result.spot_grid / item.case.K
            mask = _interpretation_mask(x)
            time_index = _nearest_tau_index(item, 0.5)
            axis.plot(
                x[mask],
                item.premium_grid[time_index, mask],
                linewidth=1.0,
                label=item.case.variation_name,
            )
        axis.axhline(item.premium_threshold, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"{family}: premium at tau/T near 0.5")
        axis.set_xlabel("S/K")
        axis.set_ylabel("U - payoff")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_greek_diagnostic_slices(
    artifacts: list[PilotRunArtifacts],
    path: Path,
) -> bool:
    """Plot base-case Gamma slices with kink and boundary caution markers."""

    plt = _load_pyplot()
    base_runs = _base_runs(artifacts)
    if plt is None or not base_runs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(base_runs), figsize=(6 * len(base_runs), 4), squeeze=False)
    for axis, item in zip(axes[0], base_runs):
        x = item.result.spot_grid / item.case.K
        mask = _interpretation_mask(x)
        for fraction in (0.25, 0.50, 1.00):
            time_index = _nearest_tau_index(item, fraction)
            gamma = item.greek_diagnostics.arrays.gamma[time_index]
            strict = item.greek_diagnostics.masks.strict_interior[time_index]
            axis.plot(
                x[mask],
                gamma[mask],
                linewidth=0.9,
                label=f"tau/T={fraction:g}",
            )
            axis.scatter(
                x[mask & strict],
                gamma[mask & strict],
                s=5,
                alpha=0.35,
            )
        axis.axvline(1.0, color="black", linewidth=0.8, linestyle="--", label="payoff kink")
        _plot_boundary_overlay(axis, item, normalized_tau=False, vertical_for_latest=True)
        axis.set_title(f"{item.case.case_family}: Gamma diagnostics")
        axis.set_xlabel("S/K")
        axis.set_ylabel("Gamma")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _create_heatmap_pair(
    artifacts: list[PilotRunArtifacts],
    path: Path,
    data_getter: Any,
    title_suffix: str,
    color_label: str,
) -> bool:
    plt = _load_pyplot()
    base_runs = _base_runs(artifacts)
    if plt is None or not base_runs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(base_runs), figsize=(6 * len(base_runs), 4), squeeze=False)
    for axis, item in zip(axes[0], base_runs):
        x, data = _interpretation_region(item, data_getter(item))
        image = axis.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[float(x[0] / item.case.K), float(x[-1] / item.case.K), 0.0, item.case.T],
        )
        axis.set_title(f"{item.case.case_family}: {title_suffix}")
        axis.set_xlabel("S/K")
        axis.set_ylabel("tau/T")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=color_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_boundary_overlay(
    axis: Any,
    item: PilotRunArtifacts,
    normalized_tau: bool = True,
    vertical_for_latest: bool = False,
) -> None:
    points = [point for point in item.boundary_curve.points if point.boundary_found]
    if not points:
        return
    if vertical_for_latest:
        latest = points[-1]
        axis.axvline(
            latest.boundary_spot / item.case.K,
            color="tab:red",
            linestyle=":",
            linewidth=0.9,
            label="latest boundary marker",
        )
        return
    tau = np.array([point.tau / item.case.T for point in points], dtype=float)
    boundary = np.array([point.boundary_spot / item.case.K for point in points], dtype=float)
    axis.plot(boundary, tau if normalized_tau else [point.tau for point in points], "r-", linewidth=1.0)


def _base_runs(artifacts: list[PilotRunArtifacts]) -> list[PilotRunArtifacts]:
    return [
        item
        for item in artifacts
        if item.case.is_base_case and not item.case.is_higher_grid_check
    ] or [
        item
        for item in artifacts
        if item.case.variation_name == "base" and not item.case.is_higher_grid_check
    ]


def _families_in_order(artifacts: list[PilotRunArtifacts]) -> list[str]:
    families: list[str] = []
    for item in artifacts:
        if item.case.case_family not in families:
            families.append(item.case.case_family)
    return families


def _interpretation_region(
    item: PilotRunArtifacts,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = item.result.spot_grid
    mask = _interpretation_mask(x / item.case.K)
    return x[mask], np.asarray(data)[:, mask]


def _interpretation_mask(moneyness: np.ndarray) -> np.ndarray:
    lower, upper = INTERPRETATION_MONEYNESS_BOUNDS
    return (moneyness >= lower) & (moneyness <= upper)


def _nearest_tau_index(item: PilotRunArtifacts, tau_fraction: float) -> int:
    target_tau = item.case.T * float(tau_fraction)
    return int(np.argmin(np.abs(item.result.tau_grid - target_tau)))


def _acceptance_status(artifacts: PilotRunArtifacts) -> str:
    lcp = artifacts.lcp_diagnostics.summary
    if (
        lcp.all_psor_steps_converged
        and lcp.max_obstacle_violation <= 1e-8
        and lcp.max_equation_violation <= 1e-6
        and lcp.max_abs_complementarity_product <= 1e-6
    ):
        return "ACCEPTABLE_FOR_PILOT_REVIEW"
    return "REVIEW_REQUIRED"


def _case_metadata(case: PilotCase) -> dict[str, str]:
    return {
        "case_name": case.case_name,
        "case_family": case.case_family,
        "variation_name": case.variation_name,
        "option_type": case.option_type,
        "K": _format_float(case.K),
        "T": _format_float(case.T),
        "r": _format_float(case.r),
        "q": _format_float(case.q),
        "sigma": _format_float(case.sigma),
        "Smax": _format_float(case.Smax),
        "M": str(case.M),
        "N": str(case.N),
    }


def _grid_metadata(artifacts: PilotRunArtifacts) -> dict[str, str]:
    return {
        "dS": _format_float(artifacts.dS),
        "dtau": _format_float(artifacts.dtau),
    }


def _validate_case(case: PilotCase) -> PilotCase:
    if not isinstance(case, PilotCase):
        raise ValueError("case must be a PilotCase.")
    if not case.case_name or not case.case_family or not case.variation_name:
        raise ValueError("case name, family, and variation must be nonempty.")
    if case.option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    for name in ("K", "T", "sigma", "Smax"):
        if _validate_positive(name, getattr(case, name)) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    for name in ("M", "N"):
        value = getattr(case, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3.")
    return case


def _validate_positive(name: str, value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric


def _validate_nonnegative(name: str, value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite.")
    return numeric


def _validate_threshold(value: float) -> float:
    numeric = _validate_nonnegative("threshold", value)
    if numeric == 0.0:
        raise ValueError("threshold must be positive.")
    return numeric


def _validate_fraction(value: float) -> float:
    fraction = float(value)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("selected tau fractions must be in [0, 1].")
    return fraction


def _spacing(grid: Any) -> float:
    values = np.asarray(grid, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(values[1] - values[0])


def _format_float(value: Any) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    if np.isposinf(numeric):
        return "inf"
    if np.isneginf(numeric):
        return "-inf"
    return f"{numeric:.12g}"


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt

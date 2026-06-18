"""Quantitative Risk Analytics: small controlled expansion after Pilot 01 stress maps."""

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
DOWNSTREAM_USE_STATUS = "analytics_diagnostic_only"

RUN_SUMMARY_FIELDNAMES = [
    "case_name",
    "case_family",
    "sweep_name",
    "stress_parameter",
    "stress_value",
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
    "is_heatmap_case",
    "runtime_seconds",
    "all_psor_steps_converged",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "max_obstacle_violation",
    "max_equation_violation",
    "max_abs_complementarity_product",
    "boundary_found_count",
    "boundary_status",
    "min_boundary_spot",
    "max_boundary_spot",
    "max_continuation_premium",
    "max_abs_gamma",
    "max_abs_gamma_strict",
    "boundary_near_node_count",
    "strict_negative_gamma_count",
    "acceptance_status",
    "downstream_use_status",
]

BOUNDARY_METRIC_FIELDNAMES = [
    "case_name",
    "case_family",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "target_tau_fraction",
    "target_tau",
    "nearest_tau",
    "time_index",
    "boundary_found",
    "boundary_spot",
    "boundary_moneyness",
    "no_boundary_reason",
    "min_boundary_spot",
    "max_boundary_spot",
    "max_continuation_premium_at_tau",
    "mean_continuation_premium_at_tau_interpretation_region",
    "downstream_use_status",
]

GREEK_METRIC_FIELDNAMES = [
    "case_name",
    "case_family",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
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
    "strict_negative_gamma_count",
    "greek_status",
    "downstream_use_status",
]

LCP_METRIC_FIELDNAMES = [
    "case_name",
    "case_family",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "lcp_status",
    "all_psor_steps_converged",
    "max_obstacle_violation",
    "min_value_gap",
    "max_equation_violation",
    "min_equation_gap",
    "max_abs_complementarity_product",
    "mean_max_abs_complementarity_product",
    "max_exercise_like_node_count",
    "max_continuation_like_node_count",
    "max_ambiguous_node_count",
    "downstream_use_status",
]

RUNTIME_ITERATION_FIELDNAMES = [
    "case_name",
    "case_family",
    "sweep_name",
    "stress_parameter",
    "stress_value",
    "option_type",
    "K",
    "T",
    "r",
    "q",
    "sigma",
    "Smax",
    "M",
    "N",
    "runtime_seconds",
    "psor_step_count",
    "max_psor_iterations",
    "mean_psor_iterations",
    "max_final_update",
    "all_psor_steps_converged",
    "downstream_use_status",
]

OUTPUT_MANIFEST_FIELDNAMES = [
    "output_id",
    "output_type",
    "path",
    "created",
    "description",
    "solver_name",
    "solver_variant",
    "case_count",
    "contains_boundary_metrics",
    "contains_greek_metrics",
    "contains_lcp_metrics",
    "downstream_use_status",
    "review_status",
]

__all__ = (
    "RiskAnalyticsCase",
    "RiskAnalyticsRun",
    "DirectionCheckResult",
    "SOLVER_NAME",
    "SOLVER_VARIANT",
    "PREMIUM_THRESHOLD",
    "INTERPRETATION_MONEYNESS_BOUNDS",
    "SELECTED_TAU_FRACTIONS",
    "DOWNSTREAM_USE_STATUS",
    "RUN_SUMMARY_FIELDNAMES",
    "BOUNDARY_METRIC_FIELDNAMES",
    "GREEK_METRIC_FIELDNAMES",
    "LCP_METRIC_FIELDNAMES",
    "RUNTIME_ITERATION_FIELDNAMES",
    "OUTPUT_MANIFEST_FIELDNAMES",
    "risk_analytics_cases",
    "run_risk_analytics_case",
    "run_risk_analytics_cases",
    "continuation_premium_grid",
    "boundary_metric_rows",
    "run_summary_row",
    "greek_metric_row",
    "lcp_metric_row",
    "runtime_iteration_row",
    "manifest_row",
    "monotone_direction",
    "write_csv",
    "create_put_boundary_vs_volatility_figure",
    "create_call_boundary_vs_dividend_yield_figure",
    "create_put_vol_boundary_curves_figure",
    "create_call_q_boundary_curves_figure",
    "create_gamma_concentration_figure",
    "create_psor_runtime_iterations_figure",
    "create_lcp_stability_figure",
    "create_call_q_sigma_boundary_heatmap_figure",
)


@dataclass(frozen=True)
class RiskAnalyticsCase:
    """One controlled quantitative risk-analytics case."""

    case_name: str
    case_family: str
    sweep_name: str
    stress_parameter: str
    stress_value: float
    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int
    is_heatmap_case: bool = False


@dataclass(frozen=True)
class RiskAnalyticsRun:
    """Solver result and attached analytics diagnostics for one case row."""

    case: RiskAnalyticsCase
    result: AmericanCNPSORResult
    premium_grid: np.ndarray
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


@dataclass(frozen=True)
class DirectionCheckResult:
    """Small qualitative direction check for one ordered stress sequence."""

    sweep_name: str
    metric_name: str
    expected_direction: str
    observed_direction: str
    status: str


def risk_analytics_cases(include_q_sigma_heatmap: bool = True) -> tuple[RiskAnalyticsCase, ...]:
    """Return the approved small quantitative risk-analytics case set."""

    cases: list[RiskAnalyticsCase] = []
    cases.extend(
        _sweep_cases(
            "american_put",
            "put_volatility_sweep",
            "sigma",
            (0.20, 0.40, 0.60),
            option_type="put",
            r=0.05,
            q=0.02,
            sigma=0.20,
        )
    )
    cases.extend(
        _sweep_cases(
            "american_put",
            "put_dividend_yield_sweep",
            "q",
            (0.00, 0.02, 0.04),
            option_type="put",
            r=0.05,
            q=0.02,
            sigma=0.20,
        )
    )
    cases.extend(
        _sweep_cases(
            "american_put",
            "put_interest_rate_sweep",
            "r",
            (0.03, 0.05, 0.08),
            option_type="put",
            r=0.05,
            q=0.02,
            sigma=0.20,
        )
    )
    cases.extend(
        _sweep_cases(
            "dividend_call",
            "call_dividend_yield_sweep",
            "q",
            (0.03, 0.08, 0.10, 0.14),
            option_type="call",
            r=0.05,
            q=0.08,
            sigma=0.20,
        )
    )
    cases.extend(
        _sweep_cases(
            "dividend_call",
            "call_volatility_sweep",
            "sigma",
            (0.20, 0.40, 0.60),
            option_type="call",
            r=0.05,
            q=0.08,
            sigma=0.20,
        )
    )
    cases.extend(
        _sweep_cases(
            "dividend_call",
            "call_interest_rate_sweep",
            "r",
            (0.03, 0.05, 0.08),
            option_type="call",
            r=0.05,
            q=0.08,
            sigma=0.20,
        )
    )
    if include_q_sigma_heatmap:
        for q_value in (0.03, 0.08, 0.10, 0.14):
            for sigma_value in (0.20, 0.40, 0.60):
                cases.append(
                    RiskAnalyticsCase(
                        case_name=(
                            "call_heatmap_q_"
                            f"{_value_token(q_value)}_sigma_{_value_token(sigma_value)}"
                        ),
                        case_family="dividend_call",
                        sweep_name="call_q_sigma_heatmap",
                        stress_parameter="q_sigma",
                        stress_value=float(q_value),
                        option_type="call",
                        K=1.0,
                        T=1.0,
                        r=0.05,
                        q=float(q_value),
                        sigma=float(sigma_value),
                        Smax=4.0,
                        M=120,
                        N=120,
                        is_heatmap_case=True,
                    )
                )
    return tuple(cases)


def run_risk_analytics_case(
    case: RiskAnalyticsCase,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> RiskAnalyticsRun:
    """Run the baseline solver and attach quantitative risk analytics diagnostics."""

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
    return _build_run_from_result(validated, result, runtime_seconds, threshold)


def run_risk_analytics_cases(
    cases: tuple[RiskAnalyticsCase, ...],
    premium_threshold: float = PREMIUM_THRESHOLD,
    cache_duplicate_parameters: bool = True,
) -> list[RiskAnalyticsRun]:
    """Run a batch of analytics cases, optionally reusing duplicate parameter solves."""

    threshold = _validate_threshold(premium_threshold)
    runs: list[RiskAnalyticsRun] = []
    cache: dict[tuple[Any, ...], tuple[AmericanCNPSORResult, float]] = {}
    for case in cases:
        validated = _validate_case(case)
        key = _case_parameter_key(validated)
        if cache_duplicate_parameters and key in cache:
            result, runtime_seconds = cache[key]
            runs.append(_build_run_from_result(validated, result, runtime_seconds, threshold))
            continue
        run = run_risk_analytics_case(validated, premium_threshold=threshold)
        if cache_duplicate_parameters:
            cache[key] = (run.result, run.runtime_seconds)
        runs.append(run)
    return runs


def continuation_premium_grid(result: Any) -> np.ndarray:
    """Return the analytics continuation premium grid, U minus payoff."""

    return continuation_premium(result.value_grid, result.payoff)


def boundary_metric_rows(
    run: RiskAnalyticsRun,
    selected_tau_fractions: tuple[float, ...] = SELECTED_TAU_FRACTIONS,
) -> list[dict[str, str]]:
    """Build selected-time boundary and continuation-premium metric rows."""

    _validate_run(run)
    rows: list[dict[str, str]] = []
    for fraction in selected_tau_fractions:
        frac = _validate_fraction(fraction)
        time_index = _nearest_tau_index(run, frac)
        target_tau = run.case.T * frac
        nearest_tau = float(run.result.tau_grid[time_index])
        point = run.boundary_curve.points[time_index]
        premium_row = run.premium_grid[time_index]
        region = _interpretation_mask(run.result.spot_grid / run.case.K)
        boundary_spot = float(point.boundary_spot) if point.boundary_found else float("nan")
        rows.append(
            _case_metadata(run.case)
            | {
                "target_tau_fraction": _format_float(frac),
                "target_tau": _format_float(target_tau),
                "nearest_tau": _format_float(nearest_tau),
                "time_index": str(time_index),
                "boundary_found": str(bool(point.boundary_found)),
                "boundary_spot": _format_float(boundary_spot),
                "boundary_moneyness": _format_float(boundary_spot / run.case.K),
                "no_boundary_reason": point.no_boundary_reason,
                "min_boundary_spot": _format_float(run.boundary_summary.min_boundary_spot),
                "max_boundary_spot": _format_float(run.boundary_summary.max_boundary_spot),
                "max_continuation_premium_at_tau": _format_float(np.max(premium_row)),
                "mean_continuation_premium_at_tau_interpretation_region": _format_float(
                    np.mean(premium_row[region])
                ),
                "downstream_use_status": DOWNSTREAM_USE_STATUS,
            }
        )
    return rows


def run_summary_row(run: RiskAnalyticsRun) -> dict[str, str]:
    """Build one run-level analytics summary row."""

    _validate_run(run)
    lcp = run.lcp_diagnostics.summary
    greek = run.greek_diagnostics.summary
    boundary = run.boundary_summary
    return (
        _case_metadata(run.case)
        | _grid_metadata(run)
        | {
            "solver_name": run.solver_name,
            "solver_variant": run.solver_variant,
            "premium_threshold": _format_float(run.premium_threshold),
            "interpretation_lower_moneyness": _format_float(
                INTERPRETATION_MONEYNESS_BOUNDS[0]
            ),
            "interpretation_upper_moneyness": _format_float(
                INTERPRETATION_MONEYNESS_BOUNDS[1]
            ),
            "is_heatmap_case": str(run.case.is_heatmap_case),
            "runtime_seconds": _format_float(run.runtime_seconds),
            "all_psor_steps_converged": str(lcp.all_psor_steps_converged),
            "psor_step_count": str(lcp.psor_step_count),
            "max_psor_iterations": str(lcp.max_psor_iterations),
            "mean_psor_iterations": _format_float(lcp.mean_psor_iterations),
            "max_final_update": _format_float(lcp.max_final_update),
            "max_obstacle_violation": _format_float(lcp.max_obstacle_violation),
            "max_equation_violation": _format_float(lcp.max_equation_violation),
            "max_abs_complementarity_product": _format_float(
                lcp.max_abs_complementarity_product
            ),
            "boundary_found_count": str(boundary.found_boundary_count),
            "boundary_status": boundary.status,
            "min_boundary_spot": _format_float(boundary.min_boundary_spot),
            "max_boundary_spot": _format_float(boundary.max_boundary_spot),
            "max_continuation_premium": _format_float(np.max(run.premium_grid)),
            "max_abs_gamma": _format_float(greek.max_abs_gamma),
            "max_abs_gamma_strict": _format_float(greek.max_abs_gamma_strict),
            "boundary_near_node_count": str(greek.boundary_near_node_count),
            "strict_negative_gamma_count": str(greek.strict_negative_gamma_count),
            "acceptance_status": _acceptance_status(run),
            "downstream_use_status": DOWNSTREAM_USE_STATUS,
        }
    )


def greek_metric_row(run: RiskAnalyticsRun) -> dict[str, str]:
    """Build one Greek diagnostic analytics row."""

    _validate_run(run)
    greek = run.greek_diagnostics.summary
    return _case_metadata(run.case) | {
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
        "strict_negative_gamma_count": str(greek.strict_negative_gamma_count),
        "greek_status": greek.status,
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def lcp_metric_row(run: RiskAnalyticsRun) -> dict[str, str]:
    """Build one LCP diagnostic analytics row."""

    _validate_run(run)
    lcp = run.lcp_diagnostics.summary
    return _case_metadata(run.case) | {
        "lcp_status": lcp.status,
        "all_psor_steps_converged": str(lcp.all_psor_steps_converged),
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
        "max_exercise_like_node_count": str(lcp.max_exercise_like_node_count),
        "max_continuation_like_node_count": str(lcp.max_continuation_like_node_count),
        "max_ambiguous_node_count": str(lcp.max_ambiguous_node_count),
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def runtime_iteration_row(run: RiskAnalyticsRun) -> dict[str, str]:
    """Build one runtime and PSOR iteration analytics row."""

    _validate_run(run)
    lcp = run.lcp_diagnostics.summary
    return _case_metadata(run.case) | {
        "runtime_seconds": _format_float(run.runtime_seconds),
        "psor_step_count": str(lcp.psor_step_count),
        "max_psor_iterations": str(lcp.max_psor_iterations),
        "mean_psor_iterations": _format_float(lcp.mean_psor_iterations),
        "max_final_update": _format_float(lcp.max_final_update),
        "all_psor_steps_converged": str(lcp.all_psor_steps_converged),
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
    }


def manifest_row(
    path: Path,
    output_type: str,
    output_id: str,
    description: str,
    created: bool,
    case_count: int,
    contains_boundary_metrics: bool = False,
    contains_greek_metrics: bool = False,
    contains_lcp_metrics: bool = False,
) -> dict[str, str]:
    """Build one analytics output-manifest row."""

    return {
        "output_id": output_id,
        "output_type": output_type,
        "path": str(path),
        "created": str(bool(created)),
        "description": description,
        "solver_name": SOLVER_NAME,
        "solver_variant": SOLVER_VARIANT,
        "case_count": str(int(case_count)),
        "contains_boundary_metrics": str(bool(contains_boundary_metrics)),
        "contains_greek_metrics": str(bool(contains_greek_metrics)),
        "contains_lcp_metrics": str(bool(contains_lcp_metrics)),
        "downstream_use_status": DOWNSTREAM_USE_STATUS,
        "review_status": "human_review_required_before_scaling",
    }


def monotone_direction(values: Any, tolerance: float = 1e-10) -> str:
    """Classify a short numeric sequence as increasing, decreasing, flat, or mixed."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("values must be a one-dimensional sequence with at least two entries.")
    if np.any(~np.isfinite(array)):
        raise ValueError("values must contain finite entries.")
    tol = _validate_nonnegative("tolerance", tolerance)
    differences = np.diff(array)
    if np.all(np.abs(differences) <= tol):
        return "flat"
    if np.all(differences >= -tol) and np.any(differences > tol):
        return "increasing"
    if np.all(differences <= tol) and np.any(differences < -tol):
        return "decreasing"
    return "mixed"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_put_boundary_vs_volatility_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot put boundary moneyness at final tau against volatility."""

    return _create_boundary_scalar_figure(
        runs,
        path,
        sweep_name="put_volatility_sweep",
        title="American put boundary versus volatility",
        x_label="sigma",
        expected="expected lower boundary as volatility rises",
    )


def create_call_boundary_vs_dividend_yield_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot dividend-call boundary moneyness at final tau against dividend yield."""

    return _create_boundary_scalar_figure(
        runs,
        path,
        sweep_name="call_dividend_yield_sweep",
        title="Dividend-call boundary versus dividend yield",
        x_label="q",
        expected="expected lower boundary as q rises",
    )


def create_put_vol_boundary_curves_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot full threshold boundary curves across the put volatility sweep."""

    return _create_boundary_curves_figure(
        runs,
        path,
        sweep_name="put_volatility_sweep",
        title="American put boundary curves across volatility",
    )


def create_call_q_boundary_curves_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot full threshold boundary curves across the call dividend-yield sweep."""

    return _create_boundary_curves_figure(
        runs,
        path,
        sweep_name="call_dividend_yield_sweep",
        title="Dividend-call boundary curves across dividend yield",
    )


def create_gamma_concentration_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot full and strict-mask Gamma concentration for key stress sweeps."""

    plt = _load_pyplot()
    if plt is None:
        return False
    groups = [
        ("put_volatility_sweep", "Put volatility"),
        ("call_dividend_yield_sweep", "Call dividend yield"),
    ]
    selected = [_sorted_sweep_runs(runs, sweep_name) for sweep_name, _ in groups]
    if not any(selected):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
    created = False
    for axis, (sweep_name, title), items in zip(axes[0], groups, selected):
        if not items:
            axis.set_axis_off()
            continue
        x = np.array([item.case.stress_value for item in items], dtype=float)
        full = np.array([item.greek_diagnostics.summary.max_abs_gamma for item in items])
        strict = np.array(
            [item.greek_diagnostics.summary.max_abs_gamma_strict for item in items]
        )
        axis.plot(x, full, marker="o", label="full max |Gamma|")
        axis.plot(x, strict, marker="s", label="strict-mask max |Gamma|")
        axis.set_title(title)
        axis.set_xlabel(items[0].case.stress_parameter)
        axis.set_ylabel("Gamma magnitude")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        created = True
    if not created:
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_psor_runtime_iterations_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot PSOR effort and runtime across non-heatmap analytics sweeps."""

    plt = _load_pyplot()
    regular = [item for item in runs if not item.case.is_heatmap_case]
    if plt is None or not regular:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    labels = [item.case.case_name.replace("_", "\n") for item in regular]
    x = np.arange(len(regular))
    axes[0, 0].plot(
        x,
        [item.lcp_diagnostics.summary.mean_psor_iterations for item in regular],
        marker="o",
        label="mean iterations",
    )
    axes[0, 0].plot(
        x,
        [item.lcp_diagnostics.summary.max_psor_iterations for item in regular],
        marker="s",
        label="max iterations",
    )
    axes[0, 0].set_title("PSOR iterations across analytics cases")
    axes[0, 0].set_ylabel("iterations")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(x, [item.runtime_seconds for item in regular], marker="o")
    axes[0, 1].set_title("Runtime across analytics cases")
    axes[0, 1].set_ylabel("seconds")
    for axis in axes[0]:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_lcp_stability_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Plot LCP diagnostic stability across all analytics runs."""

    plt = _load_pyplot()
    regular = [item for item in runs if not item.case.is_heatmap_case]
    if plt is None or not regular:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [item.case.case_name.replace("_", "\n") for item in regular]
    x = np.arange(len(regular))
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.semilogy(
        x,
        [max(item.lcp_diagnostics.summary.max_equation_violation, 1e-18) for item in regular],
        marker="o",
        label="max equation violation",
    )
    axis.semilogy(
        x,
        [
            max(item.lcp_diagnostics.summary.max_abs_complementarity_product, 1e-18)
            for item in regular
        ],
        marker="s",
        label="max |complementarity product|",
    )
    axis.set_title("LCP diagnostic stability across analytics runs")
    axis.set_ylabel("diagnostic magnitude")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_call_q_sigma_boundary_heatmap_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
) -> bool:
    """Create a small aggregate dividend-call boundary heatmap over q and sigma."""

    plt = _load_pyplot()
    items = [item for item in runs if item.case.sweep_name == "call_q_sigma_heatmap"]
    if plt is None or not items:
        return False
    q_values = sorted({item.case.q for item in items})
    sigma_values = sorted({item.case.sigma for item in items})
    if len(q_values) < 2 or len(sigma_values) < 2:
        return False
    matrix = np.full((len(sigma_values), len(q_values)), np.nan, dtype=float)
    for item in items:
        row = sigma_values.index(item.case.sigma)
        col = q_values.index(item.case.q)
        matrix[row, col] = boundary_moneyness_at_tau(item, 1.0)
    if np.all(np.isnan(matrix)):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(matrix, origin="lower", aspect="auto")
    axis.set_title("Dividend-call boundary S/K over q and sigma")
    axis.set_xlabel("q")
    axis.set_ylabel("sigma")
    axis.set_xticks(np.arange(len(q_values)))
    axis.set_xticklabels([_format_float(value) for value in q_values])
    axis.set_yticks(np.arange(len(sigma_values)))
    axis.set_yticklabels([_format_float(value) for value in sigma_values])
    fig.colorbar(image, ax=axis, label="boundary S/K at tau/T=1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def boundary_moneyness_at_tau(run: RiskAnalyticsRun, tau_fraction: float) -> float:
    """Return boundary moneyness at the nearest selected tau fraction."""

    _validate_run(run)
    index = _nearest_tau_index(run, _validate_fraction(tau_fraction))
    point = run.boundary_curve.points[index]
    if not point.boundary_found:
        return float("nan")
    return float(point.boundary_spot / run.case.K)


def _build_run_from_result(
    case: RiskAnalyticsCase,
    result: AmericanCNPSORResult,
    runtime_seconds: float,
    threshold: float,
) -> RiskAnalyticsRun:
    premium_grid = continuation_premium_grid(result)
    boundary_curve = extract_boundary_curve(result, case.case_name, threshold=threshold)
    boundary_summary = summarize_boundary_curve(boundary_curve)
    lcp_diagnostics = diagnose_lcp_result(result, case.case_name)
    greek_diagnostics = diagnose_greek_result(
        result,
        case.case_name,
        boundary_curve=boundary_curve,
        boundary_threshold=threshold,
    )
    return RiskAnalyticsRun(
        case=case,
        result=result,
        premium_grid=premium_grid,
        boundary_curve=boundary_curve,
        boundary_summary=boundary_summary,
        lcp_diagnostics=lcp_diagnostics,
        greek_diagnostics=greek_diagnostics,
        runtime_seconds=float(runtime_seconds),
        dS=_spacing(result.spot_grid),
        dtau=_spacing(result.tau_grid),
        premium_threshold=threshold,
    )


def _sweep_cases(
    case_family: str,
    sweep_name: str,
    stress_parameter: str,
    stress_values: tuple[float, ...],
    option_type: str,
    r: float,
    q: float,
    sigma: float,
) -> list[RiskAnalyticsCase]:
    cases: list[RiskAnalyticsCase] = []
    for value in stress_values:
        params = {"r": r, "q": q, "sigma": sigma}
        params[stress_parameter] = float(value)
        cases.append(
            RiskAnalyticsCase(
                case_name=f"{case_family}_{stress_parameter}_{_value_token(value)}",
                case_family=case_family,
                sweep_name=sweep_name,
                stress_parameter=stress_parameter,
                stress_value=float(value),
                option_type=option_type,
                K=1.0,
                T=1.0,
                r=float(params["r"]),
                q=float(params["q"]),
                sigma=float(params["sigma"]),
                Smax=4.0,
                M=120,
                N=120,
            )
        )
    return cases


def _create_boundary_scalar_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
    sweep_name: str,
    title: str,
    x_label: str,
    expected: str,
) -> bool:
    plt = _load_pyplot()
    items = _sorted_sweep_runs(runs, sweep_name)
    if plt is None or len(items) < 2:
        return False
    x = np.array([item.case.stress_value for item in items], dtype=float)
    y = np.array([boundary_moneyness_at_tau(item, 1.0) for item in items], dtype=float)
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 1:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(x[finite], y[finite], marker="o")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("boundary S/K at tau/T=1")
    axis.grid(True, alpha=0.25)
    axis.text(
        0.02,
        0.02,
        expected,
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _create_boundary_curves_figure(
    runs: list[RiskAnalyticsRun],
    path: Path,
    sweep_name: str,
    title: str,
) -> bool:
    plt = _load_pyplot()
    items = _sorted_sweep_runs(runs, sweep_name)
    if plt is None or not items:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    created = False
    for item in items:
        points = [point for point in item.boundary_curve.points if point.boundary_found]
        if not points:
            continue
        tau = np.array([point.tau / item.case.T for point in points], dtype=float)
        boundary = np.array([point.boundary_spot / item.case.K for point in points])
        axis.plot(
            tau,
            boundary,
            linewidth=1.0,
            label=f"{item.case.stress_parameter}={item.case.stress_value:g}",
        )
        created = True
    if not created:
        plt.close(fig)
        return False
    axis.set_title(title)
    axis.set_xlabel("tau/T")
    axis.set_ylabel("boundary S/K")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _sorted_sweep_runs(runs: list[RiskAnalyticsRun], sweep_name: str) -> list[RiskAnalyticsRun]:
    return sorted(
        [item for item in runs if item.case.sweep_name == sweep_name],
        key=lambda item: item.case.stress_value,
    )


def _case_parameter_key(case: RiskAnalyticsCase) -> tuple[Any, ...]:
    return (
        case.option_type,
        float(case.K),
        float(case.T),
        float(case.r),
        float(case.q),
        float(case.sigma),
        float(case.Smax),
        int(case.M),
        int(case.N),
    )


def _nearest_tau_index(run: RiskAnalyticsRun, tau_fraction: float) -> int:
    target_tau = run.case.T * float(tau_fraction)
    return int(np.argmin(np.abs(run.result.tau_grid - target_tau)))


def _interpretation_mask(moneyness: np.ndarray) -> np.ndarray:
    lower, upper = INTERPRETATION_MONEYNESS_BOUNDS
    return (moneyness >= lower) & (moneyness <= upper)


def _acceptance_status(run: RiskAnalyticsRun) -> str:
    lcp = run.lcp_diagnostics.summary
    if (
        lcp.all_psor_steps_converged
        and lcp.max_obstacle_violation <= 1e-8
        and lcp.max_equation_violation <= 1e-6
        and lcp.max_abs_complementarity_product <= 1e-6
    ):
        return "ACCEPTABLE_FOR_ANALYTICS_REVIEW"
    return "REVIEW_REQUIRED"


def _case_metadata(case: RiskAnalyticsCase) -> dict[str, str]:
    return {
        "case_name": case.case_name,
        "case_family": case.case_family,
        "sweep_name": case.sweep_name,
        "stress_parameter": case.stress_parameter,
        "stress_value": _format_float(case.stress_value),
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


def _grid_metadata(run: RiskAnalyticsRun) -> dict[str, str]:
    return {
        "dS": _format_float(run.dS),
        "dtau": _format_float(run.dtau),
    }


def _validate_case(case: RiskAnalyticsCase) -> RiskAnalyticsCase:
    if not isinstance(case, RiskAnalyticsCase):
        raise ValueError("case must be a RiskAnalyticsCase.")
    if not case.case_name or not case.case_family or not case.sweep_name:
        raise ValueError("case name, family, and sweep name must be nonempty.")
    if not case.stress_parameter:
        raise ValueError("stress_parameter must be nonempty.")
    if case.option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    for name in ("K", "T", "sigma", "Smax"):
        _validate_positive(name, getattr(case, name))
    for name in ("r", "q", "stress_value"):
        _validate_nonnegative(name, getattr(case, name))
    for name in ("M", "N"):
        value = getattr(case, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3.")
    return case


def _validate_run(run: RiskAnalyticsRun) -> None:
    if not isinstance(run, RiskAnalyticsRun):
        raise ValueError("run must be a RiskAnalyticsRun instance.")


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


def _value_token(value: float) -> str:
    return f"{int(round(float(value) * 1000)):03d}"


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

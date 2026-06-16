"""Ticket 11: grid and domain sensitivity diagnostics for American CN/PSOR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryCurve,
    BoundaryExtractionSummary,
    extract_boundary_curve,
    summarize_boundary_curve,
)
from american_risk_surfaces.diagnostics.greeks import GreekDiagnostics, diagnose_greek_result
from american_risk_surfaces.diagnostics.lcp import LCPDiagnostics, diagnose_lcp_result
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    american_crank_nicolson_psor_price,
)

__all__ = (
    "SensitivityCase",
    "SensitivityRunResult",
    "SelectedSpotSensitivityRow",
    "BoundarySensitivityRow",
    "DiagnosticSensitivityRow",
    "SensitivityComparisonSummary",
    "grid_sensitivity_cases",
    "domain_sensitivity_cases",
    "run_sensitivity_case",
    "nearest_spot_index",
    "selected_spot_rows",
    "boundary_shift_rows",
    "diagnostic_row",
    "summarize_comparison",
)


@dataclass(frozen=True)
class SensitivityCase:
    """One baseline-solver sensitivity configuration."""

    sensitivity_type: str
    case_name: str
    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int


@dataclass(frozen=True)
class SensitivityRunResult:
    """Solver output plus diagnostics for one sensitivity configuration."""

    case: SensitivityCase
    result: Any
    boundary_curve: BoundaryCurve | None
    boundary_summary: BoundaryExtractionSummary | None
    lcp_diagnostics: LCPDiagnostics | None
    greek_diagnostics: GreekDiagnostics | None
    runtime_seconds: float
    dS: float
    dtau: float


@dataclass(frozen=True)
class SelectedSpotSensitivityRow:
    """Selected-spot price comparison against a reference run."""

    sensitivity_type: str
    case_name: str
    option_type: str
    reference_case_name: str
    target_moneyness: float
    nearest_spot: float
    actual_moneyness: float
    value: float
    reference_nearest_spot: float
    reference_actual_moneyness: float
    reference_value: float
    difference_vs_reference: float
    abs_difference_vs_reference: float
    relative_difference_vs_reference: float


@dataclass(frozen=True)
class BoundarySensitivityRow:
    """Selected-time boundary comparison against a reference run."""

    sensitivity_type: str
    case_name: str
    option_type: str
    reference_case_name: str
    target_tau_fraction: float
    nearest_tau: float
    boundary_found: bool
    boundary_spot: float
    reference_nearest_tau: float
    reference_boundary_found: bool
    reference_boundary_spot: float
    boundary_shift: float
    abs_boundary_shift: float
    boundary_status: str


@dataclass(frozen=True)
class DiagnosticSensitivityRow:
    """LCP, Greek, and PSOR diagnostic summary for one sensitivity run."""

    sensitivity_type: str
    case_name: str
    option_type: str
    Smax: float
    M: int
    N: int
    dS: float
    all_psor_steps_converged: bool
    psor_step_count: int
    max_psor_iterations: int
    mean_psor_iterations: float
    max_final_update: float
    max_obstacle_violation: float
    max_equation_violation: float
    max_abs_complementarity_product: float
    max_abs_gamma: float
    max_abs_gamma_strict: float
    boundary_near_node_count: int
    strict_negative_gamma_count: int
    runtime_seconds: float


@dataclass(frozen=True)
class SensitivityComparisonSummary:
    """Run-level comparison summary against the selected reference configuration."""

    sensitivity_type: str
    case_name: str
    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int
    dS: float
    dtau: float
    reference_case_name: str
    reference_Smax: float
    reference_M: int
    reference_N: int
    all_psor_steps_converged: bool
    max_psor_iterations: int
    mean_psor_iterations: float
    max_final_update: float
    max_abs_selected_price_difference: float
    rmse_selected_price_difference: float
    boundary_found_count: int
    max_abs_boundary_shift: float
    mean_abs_boundary_shift: float
    max_obstacle_violation: float
    max_equation_violation: float
    max_abs_complementarity_product: float
    max_abs_gamma: float
    max_abs_gamma_strict: float
    runtime_seconds: float


def grid_sensitivity_cases() -> tuple[SensitivityCase, ...]:
    """Return fixed-domain coarse/medium/fine baseline sensitivity cases."""

    cases: list[SensitivityCase] = []
    for family, option_type, q in _families():
        for grid in (80, 120, 180):
            cases.append(
                SensitivityCase(
                    sensitivity_type="grid",
                    case_name=f"{family}_grid_{grid}",
                    option_type=option_type,
                    K=1.0,
                    T=1.0,
                    r=0.05,
                    q=q,
                    sigma=0.2,
                    Smax=4.0,
                    M=grid,
                    N=grid,
                )
            )
    return tuple(cases)


def domain_sensitivity_cases() -> tuple[SensitivityCase, ...]:
    """Return domain-cutoff cases with comparable spot spacing."""

    cases: list[SensitivityCase] = []
    for family, option_type, q in _families():
        for Smax, M in ((4.0, 120), (5.0, 150), (6.0, 180)):
            cases.append(
                SensitivityCase(
                    sensitivity_type="domain",
                    case_name=f"{family}_domain_smax{int(Smax)}",
                    option_type=option_type,
                    K=1.0,
                    T=1.0,
                    r=0.05,
                    q=q,
                    sigma=0.2,
                    Smax=Smax,
                    M=M,
                    N=120,
                )
            )
    return tuple(cases)


def run_sensitivity_case(case: SensitivityCase) -> SensitivityRunResult:
    """Run the baseline American CN/PSOR solver and attach diagnostics."""

    validated_case = _validate_case(case)
    start = perf_counter()
    result = american_crank_nicolson_psor_price(
        option_type=validated_case.option_type,
        K=validated_case.K,
        T=validated_case.T,
        r=validated_case.r,
        q=validated_case.q,
        sigma=validated_case.sigma,
        Smax=validated_case.Smax,
        M=validated_case.M,
        N=validated_case.N,
    )
    runtime_seconds = float(perf_counter() - start)
    boundary_curve = extract_boundary_curve(result, validated_case.case_name)
    boundary_summary = summarize_boundary_curve(boundary_curve)
    lcp = diagnose_lcp_result(result, validated_case.case_name)
    greeks = diagnose_greek_result(
        result,
        validated_case.case_name,
        boundary_curve=boundary_curve,
    )
    dS = _grid_spacing(result.spot_grid)
    dtau = _grid_spacing(result.tau_grid)
    return SensitivityRunResult(
        case=validated_case,
        result=result,
        boundary_curve=boundary_curve,
        boundary_summary=boundary_summary,
        lcp_diagnostics=lcp,
        greek_diagnostics=greeks,
        runtime_seconds=runtime_seconds,
        dS=dS,
        dtau=dtau,
    )


def nearest_spot_index(spot_grid: Any, target_spot: float) -> int:
    """Return the nearest spot-grid index to a target spot."""

    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1 or len(spots) == 0:
        raise ValueError("spot_grid must be a nonempty one-dimensional array.")
    if np.any(~np.isfinite(spots)):
        raise ValueError("spot_grid must contain finite values.")
    target = float(target_spot)
    if not np.isfinite(target):
        raise ValueError("target_spot must be finite.")
    return int(np.argmin(np.abs(spots - target)))


def selected_spot_rows(
    run: SensitivityRunResult,
    reference_run: SensitivityRunResult,
    selected_moneyness: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
) -> list[SelectedSpotSensitivityRow]:
    """Compare selected final-time prices with a reference run."""

    rows: list[SelectedSpotSensitivityRow] = []
    for moneyness in selected_moneyness:
        money = float(moneyness)
        if money < 0.0:
            raise ValueError("selected moneyness values must be nonnegative.")
        spot = float(run.result.K) * money
        reference_spot = float(reference_run.result.K) * money
        index = nearest_spot_index(run.result.spot_grid, spot)
        reference_index = nearest_spot_index(reference_run.result.spot_grid, reference_spot)
        value = float(run.result.values[index])
        reference_value = float(reference_run.result.values[reference_index])
        difference = value - reference_value
        rows.append(
            SelectedSpotSensitivityRow(
                sensitivity_type=run.case.sensitivity_type,
                case_name=run.case.case_name,
                option_type=run.case.option_type,
                reference_case_name=reference_run.case.case_name,
                target_moneyness=money,
                nearest_spot=float(run.result.spot_grid[index]),
                actual_moneyness=float(run.result.spot_grid[index] / run.result.K),
                value=value,
                reference_nearest_spot=float(reference_run.result.spot_grid[reference_index]),
                reference_actual_moneyness=float(
                    reference_run.result.spot_grid[reference_index] / reference_run.result.K
                ),
                reference_value=reference_value,
                difference_vs_reference=difference,
                abs_difference_vs_reference=abs(difference),
                relative_difference_vs_reference=_relative_difference(
                    difference, reference_value
                ),
            )
        )
    return rows


def boundary_shift_rows(
    run: SensitivityRunResult,
    reference_run: SensitivityRunResult,
    selected_tau_fractions: tuple[float, ...] = (0.01, 0.5, 1.0),
) -> list[BoundarySensitivityRow]:
    """Compare selected boundary points with a reference boundary curve."""

    if run.boundary_curve is None or reference_run.boundary_curve is None:
        raise ValueError("both runs must include boundary curves.")
    rows: list[BoundarySensitivityRow] = []
    for fraction in selected_tau_fractions:
        frac = float(fraction)
        if frac < 0.0 or frac > 1.0:
            raise ValueError("selected tau fractions must be in [0, 1].")
        point = _nearest_boundary_point(run.boundary_curve, run.result.T * frac)
        reference_point = _nearest_boundary_point(
            reference_run.boundary_curve, reference_run.result.T * frac
        )
        if point.boundary_found and reference_point.boundary_found:
            shift = float(point.boundary_spot - reference_point.boundary_spot)
            abs_shift = abs(shift)
            status = "matched"
        else:
            shift = float("nan")
            abs_shift = float("nan")
            status = "unmatched"
        rows.append(
            BoundarySensitivityRow(
                sensitivity_type=run.case.sensitivity_type,
                case_name=run.case.case_name,
                option_type=run.case.option_type,
                reference_case_name=reference_run.case.case_name,
                target_tau_fraction=frac,
                nearest_tau=float(point.tau),
                boundary_found=bool(point.boundary_found),
                boundary_spot=float(point.boundary_spot),
                reference_nearest_tau=float(reference_point.tau),
                reference_boundary_found=bool(reference_point.boundary_found),
                reference_boundary_spot=float(reference_point.boundary_spot),
                boundary_shift=shift,
                abs_boundary_shift=abs_shift,
                boundary_status=status,
            )
        )
    return rows


def diagnostic_row(run: SensitivityRunResult) -> DiagnosticSensitivityRow:
    """Return LCP, PSOR, and Greek diagnostics for one sensitivity run."""

    if run.lcp_diagnostics is None or run.greek_diagnostics is None:
        raise ValueError("run must include LCP and Greek diagnostics.")
    lcp = run.lcp_diagnostics.summary
    greeks = run.greek_diagnostics.summary
    return DiagnosticSensitivityRow(
        sensitivity_type=run.case.sensitivity_type,
        case_name=run.case.case_name,
        option_type=run.case.option_type,
        Smax=run.case.Smax,
        M=run.case.M,
        N=run.case.N,
        dS=run.dS,
        all_psor_steps_converged=bool(lcp.all_psor_steps_converged),
        psor_step_count=int(lcp.psor_step_count),
        max_psor_iterations=int(lcp.max_psor_iterations),
        mean_psor_iterations=float(lcp.mean_psor_iterations),
        max_final_update=float(lcp.max_final_update),
        max_obstacle_violation=float(lcp.max_obstacle_violation),
        max_equation_violation=float(lcp.max_equation_violation),
        max_abs_complementarity_product=float(lcp.max_abs_complementarity_product),
        max_abs_gamma=float(greeks.max_abs_gamma),
        max_abs_gamma_strict=float(greeks.max_abs_gamma_strict),
        boundary_near_node_count=int(greeks.boundary_near_node_count),
        strict_negative_gamma_count=int(greeks.strict_negative_gamma_count),
        runtime_seconds=float(run.runtime_seconds),
    )


def summarize_comparison(
    group_name: str,
    runs: list[SensitivityRunResult] | tuple[SensitivityRunResult, ...],
    reference_run: SensitivityRunResult,
) -> list[SensitivityComparisonSummary]:
    """Summarize each run in a grid/domain family against a reference run."""

    if not isinstance(group_name, str) or not group_name:
        raise ValueError("group_name must be a nonempty string.")
    summaries: list[SensitivityComparisonSummary] = []
    for run in runs:
        selected_rows = selected_spot_rows(run, reference_run)
        boundary_rows = boundary_shift_rows(run, reference_run)
        diag = diagnostic_row(run)
        selected_abs = np.array(
            [row.abs_difference_vs_reference for row in selected_rows], dtype=float
        )
        selected_diff = np.array(
            [row.difference_vs_reference for row in selected_rows], dtype=float
        )
        boundary_abs = np.array(
            [
                row.abs_boundary_shift
                for row in boundary_rows
                if np.isfinite(row.abs_boundary_shift)
            ],
            dtype=float,
        )
        summaries.append(
            SensitivityComparisonSummary(
                sensitivity_type=run.case.sensitivity_type,
                case_name=run.case.case_name,
                option_type=run.case.option_type,
                K=run.case.K,
                T=run.case.T,
                r=run.case.r,
                q=run.case.q,
                sigma=run.case.sigma,
                Smax=run.case.Smax,
                M=run.case.M,
                N=run.case.N,
                dS=run.dS,
                dtau=run.dtau,
                reference_case_name=reference_run.case.case_name,
                reference_Smax=reference_run.case.Smax,
                reference_M=reference_run.case.M,
                reference_N=reference_run.case.N,
                all_psor_steps_converged=diag.all_psor_steps_converged,
                max_psor_iterations=diag.max_psor_iterations,
                mean_psor_iterations=diag.mean_psor_iterations,
                max_final_update=diag.max_final_update,
                max_abs_selected_price_difference=float(np.max(selected_abs)),
                rmse_selected_price_difference=float(
                    np.sqrt(np.mean(selected_diff**2))
                ),
                boundary_found_count=(
                    run.boundary_summary.found_boundary_count
                    if run.boundary_summary is not None
                    else 0
                ),
                max_abs_boundary_shift=_nan_if_empty(boundary_abs, "max"),
                mean_abs_boundary_shift=_nan_if_empty(boundary_abs, "mean"),
                max_obstacle_violation=diag.max_obstacle_violation,
                max_equation_violation=diag.max_equation_violation,
                max_abs_complementarity_product=diag.max_abs_complementarity_product,
                max_abs_gamma=diag.max_abs_gamma,
                max_abs_gamma_strict=diag.max_abs_gamma_strict,
                runtime_seconds=diag.runtime_seconds,
            )
        )
    return summaries


def _families() -> tuple[tuple[str, str, float], ...]:
    return (
        ("american_put", "put", 0.02),
        ("dividend_call", "call", 0.08),
    )


def _validate_case(case: SensitivityCase) -> SensitivityCase:
    if not isinstance(case, SensitivityCase):
        raise ValueError("case must be a SensitivityCase.")
    if case.sensitivity_type not in {"grid", "domain"}:
        raise ValueError("sensitivity_type must be 'grid' or 'domain'.")
    if not case.case_name:
        raise ValueError("case_name must be nonempty.")
    if case.option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    if case.K <= 0.0 or case.T <= 0.0 or case.Smax <= 0.0:
        raise ValueError("K, T, and Smax must be positive.")
    if case.sigma < 0.0:
        raise ValueError("sigma must be nonnegative.")
    if case.M < 2 or case.N < 1:
        raise ValueError("M must be at least 2 and N must be at least 1.")
    return case


def _grid_spacing(grid: Any) -> float:
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("grid must be one-dimensional with at least two nodes.")
    return float(values[1] - values[0])


def _nearest_boundary_point(curve: BoundaryCurve, target_tau: float):
    index = int(np.argmin(np.abs(curve.tau_grid - float(target_tau))))
    return curve.points[index]


def _relative_difference(difference: float, reference_value: float) -> float:
    denominator = abs(float(reference_value))
    if denominator <= 1e-14:
        return float("nan")
    return float(difference / denominator)


def _nan_if_empty(values: np.ndarray, method: str) -> float:
    if values.size == 0:
        return float("nan")
    if method == "max":
        return float(np.max(values))
    return float(np.mean(values))

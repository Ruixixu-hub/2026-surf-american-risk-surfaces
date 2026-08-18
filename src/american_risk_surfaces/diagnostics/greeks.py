"""Ticket 10: Delta and Gamma diagnostics for American option value grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from american_risk_surfaces.diagnostics.boundary import (
    BoundaryCurve,
    extract_boundary_curve,
)
from american_risk_surfaces.solvers.cn_psor import AmericanCNPSORResult

__all__ = (
    "GreekArrays",
    "GreekDiagnosticMasks",
    "GreekTimeDiagnosticRow",
    "GreekDiagnosticSummary",
    "GreekDiagnostics",
    "finite_difference_delta",
    "finite_difference_gamma",
    "finite_difference_delta_nonuniform",
    "finite_difference_gamma_nonuniform",
    "payoff_kink_mask",
    "boundary_near_mask",
    "maturity_row_mask",
    "strict_greek_mask",
    "diagnose_greek_result",
    "selected_greek_profile_rows",
    "greek_by_time_rows",
)


@dataclass(frozen=True)
class GreekArrays:
    """Finite-difference Delta and Gamma arrays with finite-value masks."""

    delta: np.ndarray
    gamma: np.ndarray
    finite_delta_mask: np.ndarray
    finite_gamma_mask: np.ndarray


@dataclass(frozen=True)
class GreekDiagnosticMasks:
    """Caution masks for Greek diagnostics."""

    boundary_near: np.ndarray
    payoff_kink_near: np.ndarray
    maturity_row: np.ndarray
    strict_interior: np.ndarray


@dataclass(frozen=True)
class GreekTimeDiagnosticRow:
    """One time-row summary for Delta/Gamma diagnostics."""

    time_index: int
    tau: float
    finite_delta_count: int
    finite_gamma_count: int
    boundary_near_node_count: int
    kink_near_node_count: int
    strict_interior_count: int
    max_abs_gamma: float
    max_abs_gamma_away_from_boundary: float
    max_abs_gamma_strict: float
    min_delta_strict: float
    max_delta_strict: float
    strict_delta_lower_violation_count: int
    strict_delta_upper_violation_count: int
    strict_negative_gamma_count: int
    warning_flag: str


@dataclass(frozen=True)
class GreekDiagnosticSummary:
    """Whole-case Delta/Gamma diagnostic summary."""

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
    kink_band_steps: int
    boundary_band_steps: int
    delta_bound_tolerance: float
    gamma_negative_tolerance: float
    finite_delta_count: int
    finite_gamma_count: int
    nonfinite_delta_count: int
    nonfinite_gamma_count: int
    min_delta: float
    max_delta: float
    min_gamma: float
    max_gamma: float
    max_abs_gamma: float
    max_abs_gamma_away_from_boundary: float
    max_abs_gamma_strict: float
    boundary_near_node_count: int
    kink_near_node_count: int
    maturity_masked_node_count: int
    strict_interior_node_count: int
    strict_delta_lower_violation_count: int
    strict_delta_upper_violation_count: int
    strict_negative_gamma_count: int
    status: str


@dataclass(frozen=True)
class GreekDiagnostics:
    """Greek diagnostic bundle for one American CN/PSOR result."""

    case_name: str
    option_type: str
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    value_grid: np.ndarray
    arrays: GreekArrays
    masks: GreekDiagnosticMasks
    summary: GreekDiagnosticSummary
    by_time_rows: tuple[GreekTimeDiagnosticRow, ...]
    result_shape: tuple[int, int]


def finite_difference_delta(spot_grid: Any, values: Any) -> np.ndarray:
    """Compute central finite-difference Delta on a uniform spot grid.

    Delta is the sensitivity of option value to spot. Boundary-node values are
    returned as ``NaN`` because this diagnostic intentionally avoids presenting
    endpoint one-sided derivatives as equally reliable Greeks.
    """

    spots, dS = _validated_spot_grid(spot_grid)
    value_array = _validated_values(values, len(spots))
    delta = np.full(value_array.shape, np.nan, dtype=float)
    if value_array.ndim == 1:
        delta[1:-1] = (value_array[2:] - value_array[:-2]) / (2.0 * dS)
    else:
        delta[:, 1:-1] = (value_array[:, 2:] - value_array[:, :-2]) / (2.0 * dS)
    return delta


def finite_difference_gamma(spot_grid: Any, values: Any) -> np.ndarray:
    """Compute central second-difference Gamma on a uniform spot grid.

    Gamma is the curvature of option value with respect to spot. Boundary-node
    values are returned as ``NaN`` for the same reliability reason used for
    Delta.
    """

    spots, dS = _validated_spot_grid(spot_grid)
    value_array = _validated_values(values, len(spots))
    gamma = np.full(value_array.shape, np.nan, dtype=float)
    if value_array.ndim == 1:
        gamma[1:-1] = (
            value_array[2:] - 2.0 * value_array[1:-1] + value_array[:-2]
        ) / (dS * dS)
    else:
        gamma[:, 1:-1] = (
            value_array[:, 2:]
            - 2.0 * value_array[:, 1:-1]
            + value_array[:, :-2]
        ) / (dS * dS)
    return gamma


def finite_difference_delta_nonuniform(spot_grid: Any, values: Any) -> np.ndarray:
    """Compute a three-point first derivative on a nonuniform spot grid."""

    spots = _validated_strict_spot_grid(spot_grid)
    array = _validated_values(values, len(spots))
    left = spots[1:-1] - spots[:-2]
    right = spots[2:] - spots[1:-1]
    a = -right / (left * (left + right))
    b = (right - left) / (left * right)
    c = left / (right * (left + right))
    result = np.full(array.shape, np.nan, dtype=float)
    if array.ndim == 1:
        result[1:-1] = a * array[:-2] + b * array[1:-1] + c * array[2:]
    else:
        result[:, 1:-1] = a * array[:, :-2] + b * array[:, 1:-1] + c * array[:, 2:]
    return result


def finite_difference_gamma_nonuniform(spot_grid: Any, values: Any) -> np.ndarray:
    """Compute a three-point second derivative on a nonuniform spot grid."""

    spots = _validated_strict_spot_grid(spot_grid)
    array = _validated_values(values, len(spots))
    left = spots[1:-1] - spots[:-2]
    right = spots[2:] - spots[1:-1]
    a = 2.0 / (left * (left + right))
    b = -2.0 / (left * right)
    c = 2.0 / (right * (left + right))
    result = np.full(array.shape, np.nan, dtype=float)
    if array.ndim == 1:
        result[1:-1] = a * array[:-2] + b * array[1:-1] + c * array[2:]
    else:
        result[:, 1:-1] = a * array[:, :-2] + b * array[:, 1:-1] + c * array[:, 2:]
    return result


def payoff_kink_mask(
    spot_grid: Any,
    tau_grid: Any,
    K: float,
    kink_band_steps: int = 2,
) -> np.ndarray:
    """Mark nodes near the payoff kink at ``S = K`` for every time row."""

    spots, _ = _validated_spot_grid(spot_grid)
    taus = _validated_tau_grid(tau_grid)
    strike = _validated_positive_float("K", K)
    band = _validated_nonnegative_int("kink_band_steps", kink_band_steps)
    nearest = int(np.argmin(np.abs(spots - strike)))
    start = max(0, nearest - band)
    stop = min(len(spots), nearest + band + 1)
    mask = np.zeros((len(taus), len(spots)), dtype=bool)
    mask[:, start:stop] = True
    return mask


def boundary_near_mask(
    boundary_curve: BoundaryCurve,
    spot_grid: Any,
    tau_grid: Any,
    boundary_band_steps: int = 2,
    spot_width: float | None = None,
) -> np.ndarray:
    """Mark nodes near extracted Ticket 09 boundary points.

    A missing boundary point leaves that time row unmarked. The mask is
    diagnostic context only; it is not a statement that the boundary is exact.
    """

    if not isinstance(boundary_curve, BoundaryCurve):
        raise ValueError("boundary_curve must be a BoundaryCurve.")
    spots, dS = _validated_spot_grid(spot_grid)
    taus = _validated_tau_grid(tau_grid)
    if len(boundary_curve.points) != len(taus):
        raise ValueError("boundary_curve point count must match tau_grid length.")
    if len(boundary_curve.spot_grid) != len(spots) or not np.allclose(
        boundary_curve.spot_grid, spots
    ):
        raise ValueError("boundary_curve spot_grid must match spot_grid.")
    if len(boundary_curve.tau_grid) != len(taus) or not np.allclose(
        boundary_curve.tau_grid, taus
    ):
        raise ValueError("boundary_curve tau_grid must match tau_grid.")

    band = _validated_nonnegative_int("boundary_band_steps", boundary_band_steps)
    if spot_width is None:
        width = band * dS
    else:
        width = _validated_nonnegative_float("spot_width", spot_width)

    mask = np.zeros((len(taus), len(spots)), dtype=bool)
    for point in boundary_curve.points:
        if point.boundary_found and np.isfinite(point.boundary_spot):
            mask[point.time_index] = np.abs(spots - point.boundary_spot) <= width + 1e-14
    return mask


def maturity_row_mask(spot_grid: Any, tau_grid: Any) -> np.ndarray:
    """Mark the maturity row where payoff kinks make Gamma especially fragile."""

    spots, _ = _validated_spot_grid(spot_grid)
    taus = _validated_tau_grid(tau_grid)
    mask = np.zeros((len(taus), len(spots)), dtype=bool)
    mask[np.isclose(taus, 0.0), :] = True
    return mask


def strict_greek_mask(
    greek_arrays: GreekArrays,
    masks: GreekDiagnosticMasks,
) -> np.ndarray:
    """Return nodes used for headline strict-interior Greek summaries."""

    if not isinstance(greek_arrays, GreekArrays):
        raise ValueError("greek_arrays must be a GreekArrays instance.")
    if not isinstance(masks, GreekDiagnosticMasks):
        raise ValueError("masks must be a GreekDiagnosticMasks instance.")
    shape = greek_arrays.delta.shape
    for name, mask in (
        ("finite_delta_mask", greek_arrays.finite_delta_mask),
        ("finite_gamma_mask", greek_arrays.finite_gamma_mask),
        ("boundary_near", masks.boundary_near),
        ("payoff_kink_near", masks.payoff_kink_near),
        ("maturity_row", masks.maturity_row),
    ):
        if mask.shape != shape:
            raise ValueError(f"{name} mask must match Delta/Gamma shape.")
    return (
        greek_arrays.finite_delta_mask
        & greek_arrays.finite_gamma_mask
        & ~masks.boundary_near
        & ~masks.payoff_kink_near
        & ~masks.maturity_row
    )


def diagnose_greek_result(
    result: AmericanCNPSORResult,
    case_name: str,
    boundary_curve: BoundaryCurve | None = None,
    boundary_threshold: float = 1e-6,
    kink_band_steps: int = 2,
    boundary_band_steps: int = 2,
    delta_bound_tolerance: float = 5e-3,
    gamma_negative_tolerance: float = 1e-6,
) -> GreekDiagnostics:
    """Compute Delta/Gamma diagnostics for an American CN/PSOR value grid."""

    _validate_american_result(result)
    if not isinstance(case_name, str) or not case_name:
        raise ValueError("case_name must be a nonempty string.")
    kink_band = _validated_nonnegative_int("kink_band_steps", kink_band_steps)
    boundary_band = _validated_nonnegative_int("boundary_band_steps", boundary_band_steps)
    delta_tol = _validated_nonnegative_float(
        "delta_bound_tolerance", delta_bound_tolerance
    )
    gamma_tol = _validated_nonnegative_float(
        "gamma_negative_tolerance", gamma_negative_tolerance
    )

    if boundary_curve is None:
        boundary_curve = extract_boundary_curve(
            result, case_name, threshold=boundary_threshold
        )
    elif not isinstance(boundary_curve, BoundaryCurve):
        raise ValueError("boundary_curve must be a BoundaryCurve.")

    spot_grid = np.asarray(result.spot_grid, dtype=float)
    tau_grid = np.asarray(result.tau_grid, dtype=float)
    value_grid = np.asarray(result.value_grid, dtype=float)
    delta = finite_difference_delta(spot_grid, value_grid)
    gamma = finite_difference_gamma(spot_grid, value_grid)
    arrays = GreekArrays(
        delta=delta,
        gamma=gamma,
        finite_delta_mask=np.isfinite(delta),
        finite_gamma_mask=np.isfinite(gamma),
    )
    preliminary_masks = GreekDiagnosticMasks(
        boundary_near=boundary_near_mask(
            boundary_curve,
            spot_grid,
            tau_grid,
            boundary_band_steps=boundary_band,
        ),
        payoff_kink_near=payoff_kink_mask(
            spot_grid,
            tau_grid,
            result.K,
            kink_band_steps=kink_band,
        ),
        maturity_row=maturity_row_mask(spot_grid, tau_grid),
        strict_interior=np.zeros_like(value_grid, dtype=bool),
    )
    strict_mask = strict_greek_mask(arrays, preliminary_masks)
    masks = GreekDiagnosticMasks(
        boundary_near=preliminary_masks.boundary_near,
        payoff_kink_near=preliminary_masks.payoff_kink_near,
        maturity_row=preliminary_masks.maturity_row,
        strict_interior=strict_mask,
    )

    lower_bound, upper_bound = _delta_bounds(result.option_type)
    by_time = tuple(
        _summarize_time_row(
            time_index,
            float(tau),
            arrays,
            masks,
            lower_bound,
            upper_bound,
            delta_tol,
            gamma_tol,
        )
        for time_index, tau in enumerate(tau_grid)
    )
    summary = _summarize_case(
        result,
        case_name,
        arrays,
        masks,
        by_time,
        kink_band,
        boundary_band,
        delta_tol,
        gamma_tol,
    )
    return GreekDiagnostics(
        case_name=case_name,
        option_type=result.option_type,
        spot_grid=spot_grid.copy(),
        tau_grid=tau_grid.copy(),
        value_grid=value_grid.copy(),
        arrays=arrays,
        masks=masks,
        summary=summary,
        by_time_rows=by_time,
        result_shape=value_grid.shape,
    )


def selected_greek_profile_rows(
    diagnostics: GreekDiagnostics,
    selected_tau_fractions: tuple[float, ...] = (0.01, 0.5, 1.0),
    selected_moneyness: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
) -> list[dict[str, float | int | bool | str]]:
    """Return selected Delta/Gamma rows for report and CSV output."""

    if not isinstance(diagnostics, GreekDiagnostics):
        raise ValueError("diagnostics must be a GreekDiagnostics instance.")
    rows: list[dict[str, float | int | bool | str]] = []
    for fraction in selected_tau_fractions:
        frac = _validated_fraction(fraction)
        target_tau = diagnostics.summary.T * frac
        time_index = int(np.argmin(np.abs(diagnostics.tau_grid - target_tau)))
        nearest_tau = float(diagnostics.tau_grid[time_index])
        for moneyness in selected_moneyness:
            money = _validated_nonnegative_float("selected moneyness", moneyness)
            target_spot = diagnostics.summary.K * money
            spot_index = int(np.argmin(np.abs(diagnostics.spot_grid - target_spot)))
            nearest_spot = float(diagnostics.spot_grid[spot_index])
            rows.append(
                {
                    "case_name": diagnostics.case_name,
                    "option_type": diagnostics.option_type,
                    "target_tau_fraction": frac,
                    "target_tau": float(target_tau),
                    "nearest_tau": nearest_tau,
                    "time_index": time_index,
                    "target_moneyness": money,
                    "nearest_spot": nearest_spot,
                    "actual_moneyness": nearest_spot / diagnostics.summary.K,
                    "value": float(diagnostics.value_grid[time_index, spot_index]),
                    "delta": float(diagnostics.arrays.delta[time_index, spot_index]),
                    "gamma": float(diagnostics.arrays.gamma[time_index, spot_index]),
                    "boundary_near": bool(
                        diagnostics.masks.boundary_near[time_index, spot_index]
                    ),
                    "kink_near": bool(
                        diagnostics.masks.payoff_kink_near[time_index, spot_index]
                    ),
                    "maturity_row": bool(
                        diagnostics.masks.maturity_row[time_index, spot_index]
                    ),
                    "strict_interior": bool(
                        diagnostics.masks.strict_interior[time_index, spot_index]
                    ),
                }
            )
    return rows


def greek_by_time_rows(diagnostics: GreekDiagnostics) -> list[GreekTimeDiagnosticRow]:
    """Return by-time Greek diagnostic rows."""

    if not isinstance(diagnostics, GreekDiagnostics):
        raise ValueError("diagnostics must be a GreekDiagnostics instance.")
    return list(diagnostics.by_time_rows)


def _summarize_time_row(
    time_index: int,
    tau: float,
    arrays: GreekArrays,
    masks: GreekDiagnosticMasks,
    lower_bound: float,
    upper_bound: float,
    delta_bound_tolerance: float,
    gamma_negative_tolerance: float,
) -> GreekTimeDiagnosticRow:
    delta_row = arrays.delta[time_index]
    gamma_row = arrays.gamma[time_index]
    finite_delta = arrays.finite_delta_mask[time_index]
    finite_gamma = arrays.finite_gamma_mask[time_index]
    boundary_near = masks.boundary_near[time_index]
    kink_near = masks.payoff_kink_near[time_index]
    strict = masks.strict_interior[time_index]
    away_from_boundary = finite_gamma & ~boundary_near

    lower_violations = strict & (delta_row < lower_bound - delta_bound_tolerance)
    upper_violations = strict & (delta_row > upper_bound + delta_bound_tolerance)
    negative_gamma = strict & (gamma_row < -gamma_negative_tolerance)
    warnings: list[str] = []
    if int(np.count_nonzero(lower_violations | upper_violations)) > 0:
        warnings.append("strict_delta_bound_violation")
    if int(np.count_nonzero(negative_gamma)) > 0:
        warnings.append("strict_negative_gamma")
    if int(np.count_nonzero(strict)) == 0:
        warnings.append("no_strict_nodes")

    return GreekTimeDiagnosticRow(
        time_index=time_index,
        tau=tau,
        finite_delta_count=int(np.count_nonzero(finite_delta)),
        finite_gamma_count=int(np.count_nonzero(finite_gamma)),
        boundary_near_node_count=int(np.count_nonzero(boundary_near)),
        kink_near_node_count=int(np.count_nonzero(kink_near)),
        strict_interior_count=int(np.count_nonzero(strict)),
        max_abs_gamma=_nanmax_abs(gamma_row[finite_gamma]),
        max_abs_gamma_away_from_boundary=_nanmax_abs(gamma_row[away_from_boundary]),
        max_abs_gamma_strict=_nanmax_abs(gamma_row[strict]),
        min_delta_strict=_nanmin(delta_row[strict]),
        max_delta_strict=_nanmax(delta_row[strict]),
        strict_delta_lower_violation_count=int(np.count_nonzero(lower_violations)),
        strict_delta_upper_violation_count=int(np.count_nonzero(upper_violations)),
        strict_negative_gamma_count=int(np.count_nonzero(negative_gamma)),
        warning_flag=";".join(warnings) if warnings else "ok",
    )


def _summarize_case(
    result: AmericanCNPSORResult,
    case_name: str,
    arrays: GreekArrays,
    masks: GreekDiagnosticMasks,
    by_time: tuple[GreekTimeDiagnosticRow, ...],
    kink_band_steps: int,
    boundary_band_steps: int,
    delta_bound_tolerance: float,
    gamma_negative_tolerance: float,
) -> GreekDiagnosticSummary:
    finite_delta = arrays.finite_delta_mask
    finite_gamma = arrays.finite_gamma_mask
    away_from_boundary = finite_gamma & ~masks.boundary_near
    strict = masks.strict_interior
    strict_delta_lower = sum(row.strict_delta_lower_violation_count for row in by_time)
    strict_delta_upper = sum(row.strict_delta_upper_violation_count for row in by_time)
    strict_negative_gamma = sum(row.strict_negative_gamma_count for row in by_time)
    strict_count = int(np.count_nonzero(strict))
    status = "PASS_WITH_CAUTIONS"
    if strict_count == 0 or not result.converged:
        status = "REVIEW"

    return GreekDiagnosticSummary(
        case_name=case_name,
        option_type=result.option_type,
        K=float(result.K),
        T=float(result.T),
        r=float(result.r),
        q=float(result.q),
        sigma=float(result.sigma),
        Smax=float(result.Smax),
        M=int(result.M),
        N=int(result.N),
        kink_band_steps=kink_band_steps,
        boundary_band_steps=boundary_band_steps,
        delta_bound_tolerance=delta_bound_tolerance,
        gamma_negative_tolerance=gamma_negative_tolerance,
        finite_delta_count=int(np.count_nonzero(finite_delta)),
        finite_gamma_count=int(np.count_nonzero(finite_gamma)),
        nonfinite_delta_count=int(arrays.delta.size - np.count_nonzero(finite_delta)),
        nonfinite_gamma_count=int(arrays.gamma.size - np.count_nonzero(finite_gamma)),
        min_delta=_nanmin(arrays.delta[finite_delta]),
        max_delta=_nanmax(arrays.delta[finite_delta]),
        min_gamma=_nanmin(arrays.gamma[finite_gamma]),
        max_gamma=_nanmax(arrays.gamma[finite_gamma]),
        max_abs_gamma=_nanmax_abs(arrays.gamma[finite_gamma]),
        max_abs_gamma_away_from_boundary=_nanmax_abs(arrays.gamma[away_from_boundary]),
        max_abs_gamma_strict=_nanmax_abs(arrays.gamma[strict]),
        boundary_near_node_count=int(np.count_nonzero(masks.boundary_near)),
        kink_near_node_count=int(np.count_nonzero(masks.payoff_kink_near)),
        maturity_masked_node_count=int(np.count_nonzero(masks.maturity_row)),
        strict_interior_node_count=strict_count,
        strict_delta_lower_violation_count=int(strict_delta_lower),
        strict_delta_upper_violation_count=int(strict_delta_upper),
        strict_negative_gamma_count=int(strict_negative_gamma),
        status=status,
    )


def _validated_spot_grid(spot_grid: Any) -> tuple[np.ndarray, float]:
    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1:
        raise ValueError("spot_grid must be one-dimensional.")
    if len(spots) < 3:
        raise ValueError("spot_grid must contain at least three nodes.")
    if np.any(~np.isfinite(spots)):
        raise ValueError("spot_grid must contain finite values.")
    spacing = np.diff(spots)
    if np.any(spacing <= 0.0):
        raise ValueError("spot_grid must be strictly increasing.")
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("spot_grid must be uniformly spaced.")
    return spots, float(spacing[0])


def _validated_strict_spot_grid(spot_grid: Any) -> np.ndarray:
    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1 or len(spots) < 3:
        raise ValueError("spot_grid must be one-dimensional with at least three nodes.")
    if np.any(~np.isfinite(spots)) or np.any(np.diff(spots) <= 0.0):
        raise ValueError("spot_grid must be finite and strictly increasing.")
    return spots


def _validated_values(values: Any, spot_count: int) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim not in (1, 2):
        raise ValueError("values must be one- or two-dimensional.")
    if value_array.shape[-1] != spot_count:
        raise ValueError("the last values dimension must match spot_grid length.")
    if np.any(~np.isfinite(value_array)):
        raise ValueError("values must contain finite values.")
    return value_array


def _validated_tau_grid(tau_grid: Any) -> np.ndarray:
    taus = np.asarray(tau_grid, dtype=float)
    if taus.ndim != 1:
        raise ValueError("tau_grid must be one-dimensional.")
    if len(taus) < 1:
        raise ValueError("tau_grid must contain at least one node.")
    if np.any(~np.isfinite(taus)):
        raise ValueError("tau_grid must contain finite values.")
    if np.any(taus < 0.0):
        raise ValueError("tau_grid must be nonnegative.")
    if len(taus) > 1 and np.any(np.diff(taus) < 0.0):
        raise ValueError("tau_grid must be nondecreasing.")
    return taus


def _validate_american_result(result: Any) -> None:
    if not isinstance(result, AmericanCNPSORResult):
        raise ValueError("result must be an AmericanCNPSORResult.")
    if result.option_type not in {"put", "call"}:
        raise ValueError("result option_type must be 'put' or 'call'.")
    spot_grid = np.asarray(result.spot_grid, dtype=float)
    tau_grid = np.asarray(result.tau_grid, dtype=float)
    payoff = np.asarray(result.payoff, dtype=float)
    value_grid = np.asarray(result.value_grid, dtype=float)
    _validated_spot_grid(spot_grid)
    _validated_tau_grid(tau_grid)
    if payoff.ndim != 1 or payoff.shape != spot_grid.shape:
        raise ValueError("result payoff must be one-dimensional and match spot grid.")
    if value_grid.ndim != 2 or value_grid.shape != (len(tau_grid), len(spot_grid)):
        raise ValueError("result value_grid shape must match tau and spot grids.")
    if np.any(~np.isfinite(payoff)) or np.any(~np.isfinite(value_grid)):
        raise ValueError("result payoff and value_grid must contain finite values.")


def _validated_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validated_nonnegative_float(name: str, value: float) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return numeric


def _validated_positive_float(name: str, value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _validated_fraction(value: float) -> float:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError("selected tau fractions must be in [0, 1].")
    return numeric


def _delta_bounds(option_type: str) -> tuple[float, float]:
    if option_type == "put":
        return -1.0, 0.0
    if option_type == "call":
        return 0.0, 1.0
    raise ValueError("option_type must be 'put' or 'call'.")


def _nanmin(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else float("nan")


def _nanmax_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else float("nan")

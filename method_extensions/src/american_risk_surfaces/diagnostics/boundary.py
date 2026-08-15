"""Ticket 09: continuation premium and free-boundary extraction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from american_risk_surfaces.solvers.cn_psor import AmericanCNPSORResult

__all__ = (
    "BoundaryPoint",
    "BoundaryCurve",
    "BoundaryExtractionSummary",
    "continuation_premium",
    "linear_interpolate_threshold_crossing",
    "extract_boundary_at_time",
    "extract_boundary_curve",
    "summarize_boundary_curve",
    "selected_time_profile_rows",
)


@dataclass(frozen=True)
class BoundaryPoint:
    """Approximate boundary extraction metadata for one time level."""

    time_index: int
    tau: float
    boundary_found: bool
    boundary_spot: float
    threshold: float
    search_direction: str
    extraction_method: str
    no_boundary_reason: str
    exercise_like_node_count: int
    continuation_like_node_count: int


@dataclass(frozen=True)
class BoundaryCurve:
    """Continuation-premium boundary extraction bundle for one solver result."""

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
    threshold: float
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    payoff: np.ndarray
    value_grid: np.ndarray
    premium_grid: np.ndarray
    points: tuple[BoundaryPoint, ...]


@dataclass(frozen=True)
class BoundaryExtractionSummary:
    """Whole-curve boundary extraction summary and no-boundary reason counts."""

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
    threshold: float
    search_direction: str
    total_time_rows: int
    positive_tau_rows: int
    found_boundary_count: int
    no_boundary_count: int
    maturity_ambiguous_count: int
    all_continuation_count: int
    all_exercise_count: int
    expected_exercise_side_absent_count: int
    no_clean_transition_count: int
    insufficient_interior_nodes_count: int
    first_boundary_tau: float
    last_boundary_tau: float
    min_boundary_spot: float
    max_boundary_spot: float
    status: str


def continuation_premium(values: Any, payoff: Any) -> np.ndarray:
    """Compute continuation premium, defined as American value minus payoff."""

    value_array = np.asarray(values, dtype=float)
    payoff_array = np.asarray(payoff, dtype=float)
    if payoff_array.ndim != 1:
        raise ValueError("payoff must be one-dimensional.")
    if value_array.ndim not in (1, 2):
        raise ValueError("values must be one- or two-dimensional.")
    if value_array.shape[-1] != len(payoff_array):
        raise ValueError("the last values dimension must match payoff length.")
    return value_array - payoff_array


def linear_interpolate_threshold_crossing(
    left_spot: float,
    left_premium: float,
    right_spot: float,
    right_premium: float,
    threshold: float = 1e-6,
) -> float:
    """Linearly estimate where premium crosses a threshold between two nodes."""

    left_s = float(left_spot)
    right_s = float(right_spot)
    left_p = float(left_premium)
    right_p = float(right_premium)
    thresh = _validate_threshold(threshold)
    if not right_s > left_s:
        raise ValueError("right_spot must be greater than left_spot.")
    if left_p == right_p:
        raise ValueError("left_premium and right_premium must differ for interpolation.")
    lower_p = min(left_p, right_p)
    upper_p = max(left_p, right_p)
    if thresh < lower_p or thresh > upper_p:
        raise ValueError("threshold must lie between the two premium values.")

    weight = (thresh - left_p) / (right_p - left_p)
    return float(left_s + weight * (right_s - left_s))


def extract_boundary_at_time(
    spot_grid: Any,
    premium_row: Any,
    option_type: str,
    tau: float,
    time_index: int,
    threshold: float = 1e-6,
) -> BoundaryPoint:
    """Extract one threshold-based boundary point from a continuation-premium row."""

    spots, premium = _validated_spot_and_premium(spot_grid, premium_row)
    option = _validated_option_type(option_type)
    tau_value = _validated_nonnegative_float("tau", tau)
    index = _validated_time_index(time_index)
    thresh = _validate_threshold(threshold)
    search_direction = _search_direction(option)

    interior_spots = spots[1:-1]
    interior_premium = premium[1:-1]
    if len(interior_spots) < 2:
        return _missing_point(
            index,
            tau_value,
            thresh,
            search_direction,
            "insufficient_interior_nodes",
            0,
            len(interior_spots),
        )

    exercise_like = interior_premium <= thresh
    exercise_count = int(np.count_nonzero(exercise_like))
    continuation_count = int(len(exercise_like) - exercise_count)
    if index == 0 or tau_value == 0.0:
        return _missing_point(
            index,
            tau_value,
            thresh,
            search_direction,
            "maturity_row_ambiguous",
            exercise_count,
            continuation_count,
        )
    if exercise_count == 0:
        return _missing_point(
            index,
            tau_value,
            thresh,
            search_direction,
            "all_continuation_like",
            exercise_count,
            continuation_count,
        )
    if continuation_count == 0:
        return _missing_point(
            index,
            tau_value,
            thresh,
            search_direction,
            "all_exercise_like",
            exercise_count,
            continuation_count,
        )

    if option == "put":
        return _extract_put_boundary(
            interior_spots,
            interior_premium,
            exercise_like,
            index,
            tau_value,
            thresh,
            exercise_count,
            continuation_count,
        )
    return _extract_call_boundary(
        interior_spots,
        interior_premium,
        exercise_like,
        index,
        tau_value,
        thresh,
        exercise_count,
        continuation_count,
    )


def extract_boundary_curve(
    result: AmericanCNPSORResult,
    case_name: str,
    threshold: float = 1e-6,
) -> BoundaryCurve:
    """Extract a continuation-premium boundary curve from a stored American result."""

    _validate_american_result(result)
    if not isinstance(case_name, str) or not case_name:
        raise ValueError("case_name must be a nonempty string.")
    thresh = _validate_threshold(threshold)
    premium_grid = continuation_premium(result.value_grid, result.payoff)
    points = tuple(
        extract_boundary_at_time(
            result.spot_grid,
            premium_grid[time_index],
            result.option_type,
            float(tau),
            time_index,
            threshold=thresh,
        )
        for time_index, tau in enumerate(result.tau_grid)
    )
    return BoundaryCurve(
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
        threshold=thresh,
        spot_grid=np.asarray(result.spot_grid, dtype=float).copy(),
        tau_grid=np.asarray(result.tau_grid, dtype=float).copy(),
        payoff=np.asarray(result.payoff, dtype=float).copy(),
        value_grid=np.asarray(result.value_grid, dtype=float).copy(),
        premium_grid=np.asarray(premium_grid, dtype=float).copy(),
        points=points,
    )


def summarize_boundary_curve(curve: BoundaryCurve) -> BoundaryExtractionSummary:
    """Summarize a boundary curve and its no-boundary metadata."""

    if not isinstance(curve, BoundaryCurve):
        raise ValueError("curve must be a BoundaryCurve.")
    found_points = [point for point in curve.points if point.boundary_found]
    no_boundary_points = [point for point in curve.points if not point.boundary_found]
    positive_tau_rows = int(np.count_nonzero(curve.tau_grid > 0.0))
    reasons = [point.no_boundary_reason for point in no_boundary_points]

    if found_points:
        first_boundary_tau = float(found_points[0].tau)
        last_boundary_tau = float(found_points[-1].tau)
        boundary_spots = np.array([point.boundary_spot for point in found_points], dtype=float)
        min_boundary_spot = float(np.min(boundary_spots))
        max_boundary_spot = float(np.max(boundary_spots))
        status = "BOUNDARIES_FOUND"
    else:
        first_boundary_tau = float("nan")
        last_boundary_tau = float("nan")
        min_boundary_spot = float("nan")
        max_boundary_spot = float("nan")
        status = "NO_BOUNDARY_FOUND"

    return BoundaryExtractionSummary(
        case_name=curve.case_name,
        option_type=curve.option_type,
        K=curve.K,
        T=curve.T,
        r=curve.r,
        q=curve.q,
        sigma=curve.sigma,
        Smax=curve.Smax,
        M=curve.M,
        N=curve.N,
        threshold=curve.threshold,
        search_direction=_search_direction(curve.option_type),
        total_time_rows=len(curve.points),
        positive_tau_rows=positive_tau_rows,
        found_boundary_count=len(found_points),
        no_boundary_count=len(no_boundary_points),
        maturity_ambiguous_count=reasons.count("maturity_row_ambiguous"),
        all_continuation_count=reasons.count("all_continuation_like"),
        all_exercise_count=reasons.count("all_exercise_like"),
        expected_exercise_side_absent_count=reasons.count("expected_exercise_side_absent"),
        no_clean_transition_count=reasons.count("no_clean_transition"),
        insufficient_interior_nodes_count=reasons.count("insufficient_interior_nodes"),
        first_boundary_tau=first_boundary_tau,
        last_boundary_tau=last_boundary_tau,
        min_boundary_spot=min_boundary_spot,
        max_boundary_spot=max_boundary_spot,
        status=status,
    )


def selected_time_profile_rows(
    curve: BoundaryCurve,
    selected_tau_fractions: tuple[float, ...] = (0.01, 0.5, 1.0),
) -> list[dict[str, float | int | str]]:
    """Return full-grid premium profile rows at selected time-to-maturity fractions."""

    if not isinstance(curve, BoundaryCurve):
        raise ValueError("curve must be a BoundaryCurve.")
    rows: list[dict[str, float | int | str]] = []
    for fraction in selected_tau_fractions:
        frac = float(fraction)
        if frac < 0.0 or frac > 1.0:
            raise ValueError("selected tau fractions must be in [0, 1].")
        target_tau = curve.T * frac
        time_index = int(np.argmin(np.abs(curve.tau_grid - target_tau)))
        nearest_tau = float(curve.tau_grid[time_index])
        for spot, value, payoff, premium in zip(
            curve.spot_grid,
            curve.value_grid[time_index],
            curve.payoff,
            curve.premium_grid[time_index],
        ):
            rows.append(
                {
                    "case_name": curve.case_name,
                    "option_type": curve.option_type,
                    "target_tau_fraction": frac,
                    "target_tau": float(target_tau),
                    "nearest_tau": nearest_tau,
                    "time_index": time_index,
                    "spot": float(spot),
                    "moneyness": float(spot / curve.K),
                    "value": float(value),
                    "payoff": float(payoff),
                    "premium": float(premium),
                    "threshold": curve.threshold,
                    "premium_class": (
                        "exercise_like"
                        if float(premium) <= curve.threshold
                        else "continuation_like"
                    ),
                }
            )
    return rows


def _extract_put_boundary(
    spots: np.ndarray,
    premium: np.ndarray,
    exercise_like: np.ndarray,
    time_index: int,
    tau: float,
    threshold: float,
    exercise_count: int,
    continuation_count: int,
) -> BoundaryPoint:
    runs = _boolean_runs(exercise_like)
    if not runs[0][0]:
        return _missing_point(
            time_index,
            tau,
            threshold,
            "low_to_high",
            "expected_exercise_side_absent",
            exercise_count,
            continuation_count,
        )
    if len(runs) < 2 or runs[1][0]:
        return _missing_point(
            time_index,
            tau,
            threshold,
            "low_to_high",
            "no_clean_transition",
            exercise_count,
            continuation_count,
        )
    continuation_run = runs[1]
    if continuation_run[3] < 2:
        return _missing_point(
            time_index,
            tau,
            threshold,
            "low_to_high",
            "no_clean_transition",
            exercise_count,
            continuation_count,
        )
    transition = int(runs[0][2])
    boundary = linear_interpolate_threshold_crossing(
        spots[transition],
        premium[transition],
        spots[transition + 1],
        premium[transition + 1],
        threshold,
    )
    return BoundaryPoint(
        time_index=time_index,
        tau=tau,
        boundary_found=True,
        boundary_spot=boundary,
        threshold=threshold,
        search_direction="low_to_high",
        extraction_method="linear_threshold_crossing",
        no_boundary_reason="",
        exercise_like_node_count=exercise_count,
        continuation_like_node_count=continuation_count,
    )


def _extract_call_boundary(
    spots: np.ndarray,
    premium: np.ndarray,
    exercise_like: np.ndarray,
    time_index: int,
    tau: float,
    threshold: float,
    exercise_count: int,
    continuation_count: int,
) -> BoundaryPoint:
    runs = _boolean_runs(exercise_like)
    if not runs[-1][0]:
        return _missing_point(
            time_index,
            tau,
            threshold,
            "high_to_low",
            "expected_exercise_side_absent",
            exercise_count,
            continuation_count,
        )
    if len(runs) < 2 or runs[-2][0]:
        return _missing_point(
            time_index,
            tau,
            threshold,
            "high_to_low",
            "no_clean_transition",
            exercise_count,
            continuation_count,
        )
    continuation_run = runs[-2]
    if continuation_run[3] < 2:
        return _missing_point(
            time_index,
            tau,
            threshold,
            "high_to_low",
            "no_clean_transition",
            exercise_count,
            continuation_count,
        )
    transition = int(continuation_run[2])
    boundary = linear_interpolate_threshold_crossing(
        spots[transition],
        premium[transition],
        spots[transition + 1],
        premium[transition + 1],
        threshold,
    )
    return BoundaryPoint(
        time_index=time_index,
        tau=tau,
        boundary_found=True,
        boundary_spot=boundary,
        threshold=threshold,
        search_direction="high_to_low",
        extraction_method="linear_threshold_crossing",
        no_boundary_reason="",
        exercise_like_node_count=exercise_count,
        continuation_like_node_count=continuation_count,
    )


def _missing_point(
    time_index: int,
    tau: float,
    threshold: float,
    search_direction: str,
    reason: str,
    exercise_count: int,
    continuation_count: int,
) -> BoundaryPoint:
    return BoundaryPoint(
        time_index=time_index,
        tau=tau,
        boundary_found=False,
        boundary_spot=float("nan"),
        threshold=threshold,
        search_direction=search_direction,
        extraction_method="none",
        no_boundary_reason=reason,
        exercise_like_node_count=exercise_count,
        continuation_like_node_count=continuation_count,
    )


def _boolean_runs(mask: np.ndarray) -> list[tuple[bool, int, int, int]]:
    runs: list[tuple[bool, int, int, int]] = []
    start = 0
    current = bool(mask[0])
    for index, value in enumerate(mask[1:], start=1):
        flag = bool(value)
        if flag != current:
            runs.append((current, start, index - 1, index - start))
            start = index
            current = flag
    runs.append((current, start, len(mask) - 1, len(mask) - start))
    return runs


def _validated_spot_and_premium(spot_grid: Any, premium_row: Any) -> tuple[np.ndarray, np.ndarray]:
    spots = np.asarray(spot_grid, dtype=float)
    premium = np.asarray(premium_row, dtype=float)
    if spots.ndim != 1:
        raise ValueError("spot_grid must be one-dimensional.")
    if premium.ndim != 1:
        raise ValueError("premium_row must be one-dimensional.")
    if len(spots) < 3:
        raise ValueError("spot_grid must contain at least three nodes.")
    if spots.shape != premium.shape:
        raise ValueError("spot_grid and premium_row must have matching shapes.")
    if np.any(~np.isfinite(spots)) or np.any(~np.isfinite(premium)):
        raise ValueError("spot_grid and premium_row must contain finite values.")
    if np.any(np.diff(spots) <= 0.0):
        raise ValueError("spot_grid must be strictly increasing.")
    return spots, premium


def _validated_option_type(option_type: str) -> str:
    if option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    return option_type


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if value < 0.0:
        raise ValueError("threshold must be nonnegative.")
    return value


def _validated_nonnegative_float(name: str, value: float) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return numeric


def _validated_time_index(time_index: int) -> int:
    if isinstance(time_index, bool) or not isinstance(time_index, int):
        raise ValueError("time_index must be an integer.")
    if time_index < 0:
        raise ValueError("time_index must be nonnegative.")
    return time_index


def _validate_american_result(result: Any) -> None:
    if not isinstance(result, AmericanCNPSORResult):
        raise ValueError("result must be an AmericanCNPSORResult.")
    if result.option_type not in {"put", "call"}:
        raise ValueError("result option_type must be 'put' or 'call'.")
    spot_grid = np.asarray(result.spot_grid, dtype=float)
    tau_grid = np.asarray(result.tau_grid, dtype=float)
    payoff = np.asarray(result.payoff, dtype=float)
    value_grid = np.asarray(result.value_grid, dtype=float)
    if spot_grid.ndim != 1 or tau_grid.ndim != 1 or payoff.ndim != 1:
        raise ValueError("result grids and payoff must be one-dimensional.")
    if value_grid.ndim != 2:
        raise ValueError("result value_grid must be two-dimensional.")
    if value_grid.shape != (len(tau_grid), len(spot_grid)):
        raise ValueError("result value_grid shape must match tau and spot grids.")
    if payoff.shape != spot_grid.shape:
        raise ValueError("result payoff shape must match spot grid.")


def _search_direction(option_type: str) -> str:
    return "low_to_high" if option_type == "put" else "high_to_low"

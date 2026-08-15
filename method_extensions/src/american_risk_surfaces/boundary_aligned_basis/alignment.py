"""Pairing-aware piecewise boundary-to-strike maps using shape-preserving PCHIP."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
from american_risk_surfaces.reduced_order.types import RBFOMSnapshot
from american_risk_surfaces.boundary_aligned_basis.types import (
    BoundaryAlignmentConfig,
    BoundaryAlignmentMap,
)


def extract_oracle_boundary_path(
    snapshot: RBFOMSnapshot,
    threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    premium = snapshot.value_grid - snapshot.payoff[None, :]
    locations = np.full(len(snapshot.tau_grid), np.nan, dtype=float)
    found = np.zeros(len(snapshot.tau_grid), dtype=bool)
    for index, tau in enumerate(snapshot.tau_grid):
        point = extract_boundary_at_time(
            snapshot.spot_grid,
            premium[index],
            snapshot.option_type,
            float(tau),
            index,
            threshold=float(threshold),
        )
        if point.boundary_found:
            locations[index] = point.boundary_spot
            found[index] = True
    return locations, found


def build_boundary_alignment_map(
    boundary: float | None,
    config: BoundaryAlignmentConfig,
    *,
    physical_grid: np.ndarray | None = None,
    strike: float = 1.0,
) -> BoundaryAlignmentMap:
    spots = (
        np.linspace(0.0, 4.0, 121)
        if physical_grid is None
        else np.asarray(physical_grid, dtype=float)
    )
    if spots.ndim != 1 or len(spots) < 3 or np.any(np.diff(spots) <= 0.0):
        raise ValueError("physical_grid must be strictly increasing")
    if config.interpolation != "pchip":
        raise ValueError("only PCHIP is frozen for this experiment")
    canonical = np.linspace(float(spots[0]), float(spots[-1]), config.canonical_points)
    found = boundary is not None and np.isfinite(float(boundary))
    if not found:
        physical_at = canonical.copy()
        jacobian = np.ones_like(canonical)
        location = float("nan")
    else:
        location = float(boundary)
        if not spots[0] < location < spots[-1]:
            raise ValueError("boundary must be strictly inside the spatial domain")
        left = canonical <= strike
        physical_at = np.empty_like(canonical)
        jacobian = np.empty_like(canonical)
        physical_at[left] = location * canonical[left] / strike
        jacobian[left] = location / strike
        physical_at[~left] = location + (spots[-1] - location) * (
            (canonical[~left] - strike) / (spots[-1] - strike)
        )
        jacobian[~left] = (spots[-1] - location) / (spots[-1] - strike)
    if not np.all(np.diff(physical_at) > 0.0):
        raise RuntimeError("alignment map is not strictly monotone")
    return BoundaryAlignmentMap(
        location,
        spots.copy(),
        canonical,
        physical_at,
        jacobian,
        bool(found),
    )


def align_primal_state(state: np.ndarray, mapping: BoundaryAlignmentMap) -> np.ndarray:
    full = _with_zero_boundaries(state, len(mapping.physical_grid))
    values = PchipInterpolator(mapping.physical_grid, full, extrapolate=False)(
        mapping.physical_at_canonical
    )
    return np.asarray(values[1:-1], dtype=float)


def inverse_align_primal_state(state: np.ndarray, mapping: BoundaryAlignmentMap) -> np.ndarray:
    full = _with_zero_boundaries(state, len(mapping.canonical_grid))
    canonical_at_physical = _canonical_at_physical(mapping)
    values = PchipInterpolator(mapping.canonical_grid, full, extrapolate=False)(
        canonical_at_physical
    )
    return np.asarray(values[1:-1], dtype=float)


def align_dual_multiplier(multiplier: np.ndarray, mapping: BoundaryAlignmentMap) -> np.ndarray:
    physical = _validated_nonnegative_with_boundaries(multiplier, len(mapping.physical_grid))
    interpolated = PchipInterpolator(mapping.physical_grid, physical, extrapolate=False)(
        mapping.physical_at_canonical
    )
    delta_y = float(mapping.canonical_grid[1] - mapping.canonical_grid[0])
    delta_s = float(mapping.physical_grid[1] - mapping.physical_grid[0])
    transformed = (delta_y / delta_s) * mapping.jacobian * interpolated
    return _clip_roundoff(transformed[1:-1])


def inverse_align_dual_multiplier(multiplier: np.ndarray, mapping: BoundaryAlignmentMap) -> np.ndarray:
    canonical = _validated_nonnegative_with_boundaries(multiplier, len(mapping.canonical_grid))
    canonical_at_physical = _canonical_at_physical(mapping)
    interpolated = PchipInterpolator(mapping.canonical_grid, canonical, extrapolate=False)(
        canonical_at_physical
    )
    jacobian_at_physical = np.where(
        canonical_at_physical <= 1.0,
        mapping.boundary_spot if mapping.boundary_found else 1.0,
        ((mapping.physical_grid[-1] - mapping.boundary_spot) / 3.0)
        if mapping.boundary_found
        else 1.0,
    )
    delta_y = float(mapping.canonical_grid[1] - mapping.canonical_grid[0])
    delta_s = float(mapping.physical_grid[1] - mapping.physical_grid[0])
    restored = (delta_s / delta_y) * interpolated / jacobian_at_physical
    return _clip_roundoff(restored[1:-1])


def pairing_relative_error(
    physical_state: np.ndarray,
    physical_multiplier: np.ndarray,
    aligned_state: np.ndarray,
    aligned_multiplier: np.ndarray,
) -> float:
    original = float(np.dot(physical_state, physical_multiplier))
    transformed = float(np.dot(aligned_state, aligned_multiplier))
    return abs(transformed - original) / max(abs(original), abs(transformed), 1e-12)


def _canonical_at_physical(mapping: BoundaryAlignmentMap) -> np.ndarray:
    if not mapping.boundary_found:
        return mapping.physical_grid.copy()
    spots = mapping.physical_grid
    boundary = mapping.boundary_spot
    result = np.empty_like(spots)
    left = spots <= boundary
    result[left] = spots[left] / boundary
    result[~left] = 1.0 + 3.0 * (spots[~left] - boundary) / (spots[-1] - boundary)
    return result


def _with_zero_boundaries(values: np.ndarray, full_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (full_length,):
        result = array.copy()
    elif array.shape == (full_length - 2,):
        result = np.zeros(full_length, dtype=float)
        result[1:-1] = array
    else:
        raise ValueError("array length does not match the map grid")
    result[0] = 0.0
    result[-1] = 0.0
    return result


def _validated_nonnegative_with_boundaries(values: np.ndarray, full_length: int) -> np.ndarray:
    result = _with_zero_boundaries(values, full_length)
    if float(np.min(result)) < -1e-14:
        raise ValueError("multiplier contains a negative value below -1e-14")
    result[result < 0.0] = 0.0
    return result


def sanitize_snapshot_multiplier(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply the frozen -1e-14 rule without silently changing source data."""

    array = np.asarray(values, dtype=float).copy()
    if array.ndim == 1:
        array = array[None, :]
        squeeze = True
    elif array.ndim == 2:
        squeeze = False
    else:
        raise ValueError("multiplier must be one- or two-dimensional")
    correction = 0.0
    for row in array:
        floor = float(np.min(row))
        if floor < -1e-14:
            raise ValueError(
                f"source multiplier minimum {floor:.3e} is below the frozen -1e-14 rule"
            )
        row[row < 0.0] = 0.0
    return (array[0] if squeeze else array), correction


def _clip_roundoff(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if float(np.min(result)) < -1e-14:
        raise RuntimeError("PCHIP multiplier transform broke nonnegativity")
    result[result < 0.0] = 0.0
    return result

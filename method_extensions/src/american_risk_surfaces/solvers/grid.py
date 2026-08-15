"""Ticket 02 finite-difference grid helpers for option-pricing validation."""

from __future__ import annotations

import numpy as np


def uniform_spot_grid(Smax: float, M: int) -> tuple[np.ndarray, float]:
    """Return a uniform spot grid over ``[0, Smax]``.

    ``M`` is the number of spot intervals, so the returned grid has ``M + 1``
    points: ``S_i = i * dS`` for ``i = 0, ..., M``. Boundary nodes are included
    at indices ``0`` and ``M``.
    """

    domain_max = float(Smax)
    intervals = _validate_interval_count("M", M, minimum=2)
    if domain_max <= 0.0:
        raise ValueError("Smax must be positive.")

    dS = domain_max / intervals
    return np.linspace(0.0, domain_max, intervals + 1), dS


def uniform_tau_grid(T: float, N: int) -> tuple[np.ndarray, float]:
    """Return a uniform time-to-maturity grid over ``[0, T]``.

    ``N`` is the number of time intervals, so the returned grid has ``N + 1``
    points: ``tau_n = n * dtau`` for ``n = 0, ..., N``. ``T = 0`` is allowed
    and represents the maturity-only grid.
    """

    maturity = float(T)
    intervals = _validate_interval_count("N", N, minimum=1)
    if maturity < 0.0:
        raise ValueError("T must be nonnegative.")

    dtau = maturity / intervals
    return np.linspace(0.0, maturity, intervals + 1), dtau


def sinh_spot_grid(
    Smax: float,
    K: float,
    M: int,
    concentration_width: float | None = None,
) -> np.ndarray:
    """Return a monotone sinh grid concentrated around the strike.

    The grid includes ``0``, ``K``, and ``Smax`` exactly. The two sinh
    coordinate pieces use interval counts proportional to their transformed
    lengths, avoiding an arbitrary 50/50 split when ``K`` is off-center.
    """

    domain_max = float(Smax)
    strike = float(K)
    intervals = _validate_interval_count("M", M, minimum=4)
    if not 0.0 < strike < domain_max:
        raise ValueError("K must lie strictly inside (0, Smax).")
    width = 0.1 * strike if concentration_width is None else float(concentration_width)
    if width <= 0.0:
        raise ValueError("concentration_width must be positive.")
    left_coordinate = float(np.arcsinh(-strike / width))
    right_coordinate = float(np.arcsinh((domain_max - strike) / width))
    transformed_fraction = -left_coordinate / (right_coordinate - left_coordinate)
    left_intervals = min(max(int(round(intervals * transformed_fraction)), 2), intervals - 2)
    right_intervals = intervals - left_intervals
    left = strike + width * np.sinh(
        np.linspace(left_coordinate, 0.0, left_intervals + 1)
    )
    right = strike + width * np.sinh(
        np.linspace(0.0, right_coordinate, right_intervals + 1)
    )
    grid = np.concatenate([left, right[1:]])
    grid[0], grid[left_intervals], grid[-1] = 0.0, strike, domain_max
    if len(grid) != intervals + 1 or np.any(np.diff(grid) <= 0.0):
        raise RuntimeError("failed to construct a valid sinh spot grid.")
    return grid


def interior_indices(M: int) -> np.ndarray:
    """Return spot-grid interior indices, excluding boundary nodes.

    For a spot grid with indices ``0, ..., M``, the boundaries are ``0`` and
    ``M``. The finite-difference PDE operator is built on ``1, ..., M - 1``.
    """

    intervals = _validate_interval_count("M", M, minimum=2)
    return np.arange(1, intervals)


def _validate_interval_count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value

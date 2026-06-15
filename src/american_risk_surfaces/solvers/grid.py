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

"""Ticket 02 Black-Scholes finite-difference spatial operator helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlackScholesOperatorCoefficients:
    """Tridiagonal finite-difference coefficients on interior spot nodes.

    The arrays represent the lower, diagonal, and upper weights for applying
    the Black-Scholes spatial operator to a full value vector. They are defined
    only on interior nodes; boundary values are supplied by the caller.
    """

    lower: np.ndarray
    diagonal: np.ndarray
    upper: np.ndarray
    interior_spots: np.ndarray


def black_scholes_operator_coefficients(
    spot_grid: Any,
    dS: float,
    r: float,
    q: float,
    sigma: float,
) -> BlackScholesOperatorCoefficients:
    """Build central-difference coefficients for the Black-Scholes operator.

    The continuous operator is
    ``L U = 0.5*sigma^2*S^2*U_SS + (r-q)*S*U_S - r*U``.
    The returned arrays contain the lower, diagonal, and upper coefficients for
    each interior spot node.
    """

    spacing = float(dS)
    volatility = float(sigma)
    rate = float(r)
    dividend = float(q)

    if spacing <= 0.0:
        raise ValueError("dS must be positive.")
    if volatility < 0.0:
        raise ValueError("sigma must be nonnegative.")

    spots = _validated_spot_grid(spot_grid, spacing)
    interior = spots[1:-1]
    diffusion = 0.5 * volatility**2 * interior**2 / spacing**2
    drift = 0.5 * (rate - dividend) * interior / spacing

    lower = diffusion - drift
    diagonal = -2.0 * diffusion - rate
    upper = diffusion + drift

    return BlackScholesOperatorCoefficients(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        interior_spots=interior,
    )


def black_scholes_operator_coefficients_nonuniform(
    spot_grid: Any,
    r: float,
    q: float,
    sigma: float,
) -> BlackScholesOperatorCoefficients:
    """Build second-order three-point coefficients on a nonuniform grid."""

    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1 or len(spots) < 3:
        raise ValueError("spot_grid must be one-dimensional with at least three nodes.")
    if spots[0] < 0.0 or np.any(np.diff(spots) <= 0.0):
        raise ValueError("spot_grid must be nonnegative and strictly increasing.")
    volatility = float(sigma)
    if volatility < 0.0:
        raise ValueError("sigma must be nonnegative.")
    rate, dividend = float(r), float(q)
    interior = spots[1:-1]
    left_width = interior - spots[:-2]
    right_width = spots[2:] - interior
    first_left = -right_width / (left_width * (left_width + right_width))
    first_diagonal = (right_width - left_width) / (left_width * right_width)
    first_right = left_width / (right_width * (left_width + right_width))
    second_left = 2.0 / (left_width * (left_width + right_width))
    second_diagonal = -2.0 / (left_width * right_width)
    second_right = 2.0 / (right_width * (left_width + right_width))
    diffusion = 0.5 * volatility**2 * interior**2
    drift = (rate - dividend) * interior
    return BlackScholesOperatorCoefficients(
        lower=diffusion * second_left + drift * first_left,
        diagonal=diffusion * second_diagonal + drift * first_diagonal - rate,
        upper=diffusion * second_right + drift * first_right,
        interior_spots=interior,
    )


def apply_black_scholes_operator(
    values: Any,
    coefficients: BlackScholesOperatorCoefficients,
) -> np.ndarray:
    """Apply the discrete Black-Scholes spatial operator to interior nodes.

    ``values`` must be a full value vector including the lower and upper
    boundary nodes. The returned array has one entry per interior node.
    """

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    expected_length = len(coefficients.interior_spots) + 2
    if len(vector) != expected_length:
        raise ValueError("values length must match the coefficient grid length.")

    return (
        coefficients.lower * vector[:-2]
        + coefficients.diagonal * vector[1:-1]
        + coefficients.upper * vector[2:]
    )


def european_put_boundaries(K: float, tau: float, r: float) -> tuple[float, float]:
    """Return European put boundary values at ``S = 0`` and ``S = Smax``."""

    strike = _validated_strike(K)
    maturity = _validated_tau(tau)
    return strike * math.exp(-float(r) * maturity), 0.0


def american_put_boundaries(K: float, tau: float) -> tuple[float, float]:
    """Return American put boundary values at ``S = 0`` and ``S = Smax``."""

    strike = _validated_strike(K)
    _validated_tau(tau)
    return strike, 0.0


def european_call_boundaries(
    Smax: float,
    K: float,
    tau: float,
    r: float,
    q: float,
) -> tuple[float, float]:
    """Return European call boundary values at ``S = 0`` and ``S = Smax``."""

    domain_max = _validated_smax(Smax)
    strike = _validated_strike(K)
    maturity = _validated_tau(tau)
    discounted_stock = domain_max * math.exp(-float(q) * maturity)
    discounted_strike = strike * math.exp(-float(r) * maturity)
    return 0.0, max(discounted_stock - discounted_strike, 0.0)


def american_call_boundaries(
    Smax: float,
    K: float,
    tau: float,
    r: float,
    q: float,
) -> tuple[float, float]:
    """Return American call boundary values at ``S = 0`` and ``S = Smax``."""

    domain_max = _validated_smax(Smax)
    strike = _validated_strike(K)
    maturity = _validated_tau(tau)
    immediate_exercise = domain_max - strike
    discounted_stock = domain_max * math.exp(-float(q) * maturity)
    discounted_strike = strike * math.exp(-float(r) * maturity)
    return 0.0, max(0.0, immediate_exercise, discounted_stock - discounted_strike)


def _validated_spot_grid(spot_grid: Any, dS: float) -> np.ndarray:
    spots = np.asarray(spot_grid, dtype=float)
    if spots.ndim != 1:
        raise ValueError("spot_grid must be one-dimensional.")
    if len(spots) < 3:
        raise ValueError("spot_grid must include at least three nodes.")
    if np.any(spots < 0.0):
        raise ValueError("spot_grid must be nonnegative.")
    spacings = np.diff(spots)
    if np.any(spacings <= 0.0):
        raise ValueError("spot_grid must be strictly increasing.")
    if not np.allclose(spacings, dS, rtol=1e-12, atol=1e-12):
        raise ValueError("spot_grid spacing must be uniform and match dS.")
    return spots


def _validated_smax(Smax: float) -> float:
    domain_max = float(Smax)
    if domain_max <= 0.0:
        raise ValueError("Smax must be positive.")
    return domain_max


def _validated_strike(K: float) -> float:
    strike = float(K)
    if strike <= 0.0:
        raise ValueError("K must be positive.")
    return strike


def _validated_tau(tau: float) -> float:
    maturity = float(tau)
    if maturity < 0.0:
        raise ValueError("tau must be nonnegative.")
    return maturity

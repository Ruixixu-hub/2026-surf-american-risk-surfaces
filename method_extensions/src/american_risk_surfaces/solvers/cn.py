"""Ticket 03: European Crank-Nicolson validation solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from american_risk_surfaces.solvers.black_scholes import (
    call_payoff,
    european_call_price,
    european_put_price,
    put_payoff,
)
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid
from american_risk_surfaces.solvers.operator import (
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
    european_call_boundaries,
    european_put_boundaries,
)


@dataclass(frozen=True)
class ValidationMetrics:
    """European validation errors over the chosen target moneyness region."""

    max_abs_error: float
    rmse: float
    max_error_spot: float
    target_lower: float
    target_upper: float


@dataclass(frozen=True)
class EuropeanCNResult:
    """Result bundle for one European Crank-Nicolson validation run."""

    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    values: np.ndarray
    closed_form_values: np.ndarray
    errors: np.ndarray
    metrics: ValidationMetrics


def solve_tridiagonal(lower: Any, diagonal: Any, upper: Any, rhs: Any) -> np.ndarray:
    """Solve a tridiagonal linear system using the Thomas algorithm.

    ``diagonal`` and ``rhs`` have length ``n``. ``lower`` and ``upper`` have
    length ``n - 1`` and represent the subdiagonal and superdiagonal.
    """

    lower_values = np.asarray(lower, dtype=float)
    diagonal_values = np.asarray(diagonal, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    rhs_values = np.asarray(rhs, dtype=float)

    if diagonal_values.ndim != 1 or rhs_values.ndim != 1:
        raise ValueError("diagonal and rhs must be one-dimensional.")
    if lower_values.ndim != 1 or upper_values.ndim != 1:
        raise ValueError("lower and upper must be one-dimensional.")
    if len(diagonal_values) == 0:
        raise ValueError("diagonal must not be empty.")
    if len(rhs_values) != len(diagonal_values):
        raise ValueError("rhs length must match diagonal length.")
    if len(lower_values) != len(diagonal_values) - 1:
        raise ValueError("lower length must be one less than diagonal length.")
    if len(upper_values) != len(diagonal_values) - 1:
        raise ValueError("upper length must be one less than diagonal length.")

    pivot_tolerance = 1e-14
    diag = diagonal_values.copy()
    superdiag = upper_values.copy()
    solution_rhs = rhs_values.copy()

    if len(diag) == 1:
        if abs(diag[0]) <= pivot_tolerance:
            raise ValueError("tridiagonal system has a zero pivot.")
        return solution_rhs / diag

    for row in range(1, len(diag)):
        previous_pivot = diag[row - 1]
        if abs(previous_pivot) <= pivot_tolerance:
            raise ValueError("tridiagonal system has a zero pivot.")
        factor = lower_values[row - 1] / previous_pivot
        diag[row] -= factor * superdiag[row - 1]
        solution_rhs[row] -= factor * solution_rhs[row - 1]

    if abs(diag[-1]) <= pivot_tolerance:
        raise ValueError("tridiagonal system has a zero pivot.")

    solution = np.empty_like(solution_rhs)
    solution[-1] = solution_rhs[-1] / diag[-1]
    for row in range(len(diag) - 2, -1, -1):
        solution[row] = (solution_rhs[row] - superdiag[row] * solution[row + 1]) / diag[row]

    return solution


def target_region_error_metrics(
    spot_grid: Any,
    numerical: Any,
    reference: Any,
    K: float,
    lower_moneyness: float = 0.4,
    upper_moneyness: float = 1.8,
) -> ValidationMetrics:
    """Compute max absolute error and RMSE over a target ``S/K`` region."""

    spots = np.asarray(spot_grid, dtype=float)
    numerical_values = np.asarray(numerical, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    strike = _validate_positive("K", K)
    lower_target = float(lower_moneyness)
    upper_target = float(upper_moneyness)

    if spots.ndim != 1:
        raise ValueError("spot_grid must be one-dimensional.")
    if numerical_values.shape != spots.shape or reference_values.shape != spots.shape:
        raise ValueError("numerical and reference values must match spot_grid shape.")
    if lower_target <= 0.0:
        raise ValueError("lower_moneyness must be positive.")
    if upper_target < lower_target:
        raise ValueError("upper_moneyness must be at least lower_moneyness.")

    moneyness = spots / strike
    target_mask = (moneyness >= lower_target) & (moneyness <= upper_target)
    if not np.any(target_mask):
        raise ValueError("target moneyness region contains no grid points.")

    target_spots = spots[target_mask]
    target_errors = numerical_values[target_mask] - reference_values[target_mask]
    target_abs_errors = np.abs(target_errors)
    max_error_index = int(np.argmax(target_abs_errors))

    return ValidationMetrics(
        max_abs_error=float(target_abs_errors[max_error_index]),
        rmse=float(math.sqrt(np.mean(target_errors**2))),
        max_error_spot=float(target_spots[max_error_index]),
        target_lower=lower_target,
        target_upper=upper_target,
    )


def european_crank_nicolson_price(
    option_type: str,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    Smax: float,
    M: int,
    N: int,
) -> EuropeanCNResult:
    """Price a European call or put with Crank-Nicolson in time-to-maturity.

    This is a validation-mode solver only. It advances the European continuation
    PDE and compares the final grid values with the Ticket 1 closed-form
    Black-Scholes utilities.
    """

    option = _validated_option_type(option_type)
    strike = _validate_positive("K", K)
    rate = float(r)
    dividend = float(q)
    volatility = _validate_nonnegative("sigma", sigma)
    domain_max = float(Smax)

    spot_grid, dS = uniform_spot_grid(domain_max, M)
    tau_grid, dtau = uniform_tau_grid(T, N)
    intervals = len(spot_grid) - 1
    time_steps = len(tau_grid) - 1

    payoff_function, price_function, boundary_function = _option_helpers(
        option, domain_max, strike, rate, dividend, volatility
    )

    values = np.asarray(payoff_function(spot_grid, strike), dtype=float)
    if dtau > 0.0:
        coefficients = black_scholes_operator_coefficients(
            spot_grid, dS=dS, r=rate, q=dividend, sigma=volatility
        )
        half_step = 0.5 * dtau
        lhs_lower = -half_step * coefficients.lower[1:]
        lhs_diagonal = 1.0 - half_step * coefficients.diagonal
        lhs_upper = -half_step * coefficients.upper[:-1]

        for step in range(time_steps):
            old_tau = float(tau_grid[step])
            new_tau = float(tau_grid[step + 1])
            old_lower, old_upper = boundary_function(old_tau)
            new_lower, new_upper = boundary_function(new_tau)

            values[0] = old_lower
            values[-1] = old_upper
            rhs = values[1:-1] + half_step * apply_black_scholes_operator(values, coefficients)
            rhs[0] += half_step * coefficients.lower[0] * new_lower
            rhs[-1] += half_step * coefficients.upper[-1] * new_upper

            next_values = np.empty_like(values)
            next_values[0] = new_lower
            next_values[-1] = new_upper
            next_values[1:-1] = solve_tridiagonal(lhs_lower, lhs_diagonal, lhs_upper, rhs)
            values = next_values

    closed_form_values = np.asarray(
        price_function(spot_grid, K=strike, T=float(T), r=rate, q=dividend, sigma=volatility),
        dtype=float,
    )
    errors = values - closed_form_values
    metrics = target_region_error_metrics(spot_grid, values, closed_form_values, K=strike)

    return EuropeanCNResult(
        option_type=option,
        K=strike,
        T=float(T),
        r=rate,
        q=dividend,
        sigma=volatility,
        Smax=domain_max,
        M=intervals,
        N=time_steps,
        spot_grid=spot_grid,
        tau_grid=tau_grid,
        values=values,
        closed_form_values=closed_form_values,
        errors=errors,
        metrics=metrics,
    )


def _validated_option_type(option_type: str) -> str:
    if not isinstance(option_type, str):
        raise ValueError("option_type must be 'call' or 'put'.")
    option = option_type.lower()
    if option not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    return option


def _validate_positive(name: str, value: float) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _option_helpers(
    option_type: str,
    Smax: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[[float], tuple[float, float]]]:
    if option_type == "call":
        return (
            call_payoff,
            european_call_price,
            lambda tau: european_call_boundaries(Smax=Smax, K=K, tau=tau, r=r, q=q),
        )

    return (
        put_payoff,
        european_put_price,
        lambda tau: european_put_boundaries(K=K, tau=tau, r=r),
    )

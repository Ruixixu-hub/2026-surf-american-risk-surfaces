"""Ticket 04: PSOR/LCP core for American option obstacle enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid
from american_risk_surfaces.solvers.operator import (
    american_call_boundaries as _american_call_boundaries,
    american_put_boundaries as _american_put_boundaries,
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
)

__all__ = (
    "PSORResult",
    "AmericanCNPSORResult",
    "psor_lcp_solve",
    "american_crank_nicolson_psor_price",
)


@dataclass(frozen=True)
class PSORResult:
    """Projected SOR result and convergence metadata for one LCP solve."""

    solution: np.ndarray
    converged: bool
    iterations: int
    final_update: float
    tolerance: float
    omega: float
    max_iter: int


@dataclass(frozen=True)
class AmericanCNPSORResult:
    """American CN/PSOR result over a full time-to-maturity grid."""

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
    payoff: np.ndarray
    value_grid: np.ndarray
    values: np.ndarray
    psor_results: tuple[PSORResult, ...]
    converged: bool
    max_obstacle_violation: float


def psor_lcp_solve(
    lower: Any,
    diagonal: Any,
    upper: Any,
    rhs: Any,
    payoff: Any,
    initial: Any = None,
    omega: float = 1.2,
    tolerance: float = 1e-8,
    max_iter: int = 10000,
) -> PSORResult:
    """Solve a tridiagonal LCP with projected SOR on interior nodes only.

    The LCP is ``A u >= b``, ``u >= payoff``, and
    ``(u - payoff)^T (A u - b) = 0``. The update first computes the SOR
    candidate for the linear row and then projects it back above payoff.
    """

    lower_values, diagonal_values, upper_values, rhs_values, payoff_values = _validated_lcp_arrays(
        lower, diagonal, upper, rhs, payoff
    )
    relaxation = _validated_omega(omega)
    tol = _validated_tolerance(tolerance)
    iterations_allowed = _validated_max_iter(max_iter)

    if initial is None:
        solution = payoff_values.copy()
    else:
        solution = np.asarray(initial, dtype=float)
        if solution.ndim != 1 or solution.shape != payoff_values.shape:
            raise ValueError("initial must be one-dimensional and match payoff shape.")
        solution = np.maximum(solution.copy(), payoff_values)

    pivot_tolerance = 1e-14
    if np.any(np.abs(diagonal_values) <= pivot_tolerance):
        raise ValueError("diagonal entries must be nonzero.")

    final_update = float("inf")
    iterations_used = 0
    for iteration in range(1, iterations_allowed + 1):
        max_update = 0.0
        for row in range(len(diagonal_values)):
            previous_value = solution[row]
            row_rhs = rhs_values[row]
            if row > 0:
                row_rhs -= lower_values[row - 1] * solution[row - 1]
            if row < len(diagonal_values) - 1:
                row_rhs -= upper_values[row] * solution[row + 1]

            gauss_seidel_candidate = row_rhs / diagonal_values[row]
            relaxed_candidate = previous_value + relaxation * (
                gauss_seidel_candidate - previous_value
            )
            solution[row] = max(payoff_values[row], relaxed_candidate)
            max_update = max(max_update, abs(solution[row] - previous_value))

        final_update = float(max_update)
        iterations_used = iteration
        if final_update <= tol:
            return PSORResult(
                solution=solution.copy(),
                converged=True,
                iterations=iterations_used,
                final_update=final_update,
                tolerance=tol,
                omega=relaxation,
                max_iter=iterations_allowed,
            )

    return PSORResult(
        solution=solution.copy(),
        converged=False,
        iterations=iterations_used,
        final_update=final_update,
        tolerance=tol,
        omega=relaxation,
        max_iter=iterations_allowed,
    )


def american_crank_nicolson_psor_price(
    option_type: str,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    Smax: float,
    M: int,
    N: int,
    omega: float = 1.2,
    tolerance: float = 1e-8,
    max_iter: int = 10000,
) -> AmericanCNPSORResult:
    """Price an American put or dividend-paying call with CN/PSOR.

    This is a reusable core solver, not a full validation experiment. It stores
    the full value grid so later tickets can compute validation diagnostics.
    """

    option = _validated_option_type(option_type)
    strike = _validate_positive("K", K)
    rate = float(r)
    dividend = float(q)
    volatility = _validate_nonnegative("sigma", sigma)
    domain_max = float(Smax)
    relaxation = _validated_omega(omega)
    tol = _validated_tolerance(tolerance)
    iterations_allowed = _validated_max_iter(max_iter)

    spot_grid, dS = uniform_spot_grid(domain_max, M)
    tau_grid, dtau = uniform_tau_grid(T, N)
    intervals = len(spot_grid) - 1
    time_steps = len(tau_grid) - 1
    payoff_function, boundary_function = _option_helpers(option, domain_max, strike, rate, dividend)

    payoff = np.asarray(payoff_function(spot_grid, strike), dtype=float)
    value_grid = np.empty((len(tau_grid), len(spot_grid)), dtype=float)
    values = payoff.copy()
    value_grid[0] = values
    psor_results: list[PSORResult] = []

    if dtau > 0.0:
        coefficients = black_scholes_operator_coefficients(
            spot_grid, dS=dS, r=rate, q=dividend, sigma=volatility
        )
        half_step = 0.5 * dtau
        lhs_lower = -half_step * coefficients.lower[1:]
        lhs_diagonal = 1.0 - half_step * coefficients.diagonal
        lhs_upper = -half_step * coefficients.upper[:-1]
        interior_payoff = payoff[1:-1]

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

            psor_result = psor_lcp_solve(
                lhs_lower,
                lhs_diagonal,
                lhs_upper,
                rhs,
                interior_payoff,
                initial=values[1:-1],
                omega=relaxation,
                tolerance=tol,
                max_iter=iterations_allowed,
            )
            psor_results.append(psor_result)

            next_values = np.empty_like(values)
            next_values[0] = new_lower
            next_values[-1] = new_upper
            next_values[1:-1] = psor_result.solution
            values = next_values
            value_grid[step + 1] = values
    else:
        value_grid[:] = payoff

    max_obstacle_violation = float(np.max(np.maximum(payoff[np.newaxis, :] - value_grid, 0.0)))
    all_steps_converged = all(result.converged for result in psor_results)

    return AmericanCNPSORResult(
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
        payoff=payoff,
        value_grid=value_grid,
        values=values.copy(),
        psor_results=tuple(psor_results),
        converged=all_steps_converged,
        max_obstacle_violation=max_obstacle_violation,
    )


def _validated_lcp_arrays(
    lower: Any,
    diagonal: Any,
    upper: Any,
    rhs: Any,
    payoff: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_values = np.asarray(lower, dtype=float)
    diagonal_values = np.asarray(diagonal, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    rhs_values = np.asarray(rhs, dtype=float)
    payoff_values = np.asarray(payoff, dtype=float)

    arrays = {
        "lower": lower_values,
        "diagonal": diagonal_values,
        "upper": upper_values,
        "rhs": rhs_values,
        "payoff": payoff_values,
    }
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

    system_size = len(diagonal_values)
    if system_size == 0:
        raise ValueError("diagonal must not be empty.")
    if len(lower_values) != system_size - 1:
        raise ValueError("lower length must be one less than diagonal length.")
    if len(upper_values) != system_size - 1:
        raise ValueError("upper length must be one less than diagonal length.")
    if len(rhs_values) != system_size:
        raise ValueError("rhs length must match diagonal length.")
    if len(payoff_values) != system_size:
        raise ValueError("payoff length must match diagonal length.")

    return lower_values, diagonal_values, upper_values, rhs_values, payoff_values


def _validated_option_type(option_type: str) -> str:
    if not isinstance(option_type, str):
        raise ValueError("option_type must be 'put' or 'call'.")
    option = option_type.lower()
    if option not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    return option


def _validated_omega(omega: float) -> float:
    relaxation = float(omega)
    if relaxation <= 0.0 or relaxation >= 2.0:
        raise ValueError("omega must satisfy 0 < omega < 2.")
    return relaxation


def _validated_tolerance(tolerance: float) -> float:
    tol = float(tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be positive.")
    return tol


def _validated_max_iter(max_iter: int) -> int:
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise ValueError("max_iter must be an integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    return max_iter


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
) -> tuple[Callable[..., Any], Callable[[float], tuple[float, float]]]:
    if option_type == "call":
        return (
            call_payoff,
            lambda tau: _american_call_boundaries(Smax=Smax, K=K, tau=tau, r=r, q=q),
        )

    return (
        put_payoff,
        lambda tau: _american_put_boundaries(K=K, tau=tau),
    )

"""Ticket 10A: Rannacher smoothing comparison for American CN/PSOR diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.cn_psor import (
    AmericanCNPSORResult,
    PSORResult,
    american_crank_nicolson_psor_price,
    psor_lcp_solve,
)
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid
from american_risk_surfaces.solvers.operator import (
    BlackScholesOperatorCoefficients,
    american_call_boundaries as _american_call_boundaries,
    american_put_boundaries as _american_put_boundaries,
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
)

__all__ = (
    "RannacherSmoothingMetadata",
    "RannacherCNPSORResult",
    "backward_euler_psor_step",
    "rannacher_crank_nicolson_psor_price",
)


@dataclass(frozen=True)
class RannacherSmoothingMetadata:
    """Metadata separating startup backward-Euler solves from later CN solves."""

    enabled: bool
    rannacher_substeps: int
    standard_dtau: float
    rannacher_substep_size: float
    rannacher_psor_results: tuple[PSORResult, ...]
    cn_psor_results: tuple[PSORResult, ...]
    all_rannacher_steps_converged: bool
    all_cn_steps_converged: bool
    all_steps_converged: bool
    max_obstacle_violation: float


@dataclass(frozen=True)
class RannacherCNPSORResult:
    """A named Rannacher variant result plus its smoothing metadata."""

    result: AmericanCNPSORResult
    metadata: RannacherSmoothingMetadata


def backward_euler_psor_step(
    values: Any,
    payoff: Any,
    coefficients: BlackScholesOperatorCoefficients,
    dtau: float,
    new_lower_boundary: float,
    new_upper_boundary: float,
    omega: float = 1.2,
    tolerance: float = 1e-8,
    max_iter: int = 10000,
) -> tuple[np.ndarray, PSORResult]:
    """Advance one American LCP step with backward Euler and PSOR.

    The step solves ``(I - dtau L_h) u >= old_values`` on interior nodes, with
    projection above the payoff. Boundary values are supplied at the new time
    level, matching the implicit nature of backward Euler.
    """

    old_values = _validated_full_vector("values", values, coefficients)
    payoff_values = _validated_full_vector("payoff", payoff, coefficients)
    step_size = _validated_positive_float("dtau", dtau)

    lhs_lower = -step_size * coefficients.lower[1:]
    lhs_diagonal = 1.0 - step_size * coefficients.diagonal
    lhs_upper = -step_size * coefficients.upper[:-1]
    rhs = old_values[1:-1].copy()
    rhs[0] += step_size * coefficients.lower[0] * float(new_lower_boundary)
    rhs[-1] += step_size * coefficients.upper[-1] * float(new_upper_boundary)

    psor_result = psor_lcp_solve(
        lhs_lower,
        lhs_diagonal,
        lhs_upper,
        rhs,
        payoff_values[1:-1],
        initial=old_values[1:-1],
        omega=omega,
        tolerance=tolerance,
        max_iter=max_iter,
    )

    next_values = np.empty_like(old_values)
    next_values[0] = float(new_lower_boundary)
    next_values[-1] = float(new_upper_boundary)
    next_values[1:-1] = psor_result.solution
    return next_values, psor_result


def rannacher_crank_nicolson_psor_price(
    option_type: str,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    Smax: float,
    M: int,
    N: int,
    rannacher_substeps: int = 2,
    omega: float = 1.2,
    tolerance: float = 1e-8,
    max_iter: int = 10000,
) -> RannacherCNPSORResult:
    """Price an American option with a separate Rannacher CN/PSOR variant.

    With the default setting, the first ordinary CN time interval is replaced
    by two equal backward-Euler substeps. Later intervals use the same CN/PSOR
    matrix and boundary adjustment logic as the Ticket 04 baseline solver.
    """

    substeps = _validated_substeps(rannacher_substeps)
    if substeps == 0:
        baseline = american_crank_nicolson_psor_price(
            option_type,
            K,
            T,
            r,
            q,
            sigma,
            Smax,
            M,
            N,
            omega=omega,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        standard_dtau = _standard_dtau_from_grid(baseline.tau_grid)
        metadata = RannacherSmoothingMetadata(
            enabled=False,
            rannacher_substeps=0,
            standard_dtau=standard_dtau,
            rannacher_substep_size=0.0,
            rannacher_psor_results=(),
            cn_psor_results=baseline.psor_results,
            all_rannacher_steps_converged=True,
            all_cn_steps_converged=baseline.converged,
            all_steps_converged=baseline.converged,
            max_obstacle_violation=baseline.max_obstacle_violation,
        )
        return RannacherCNPSORResult(result=baseline, metadata=metadata)

    option = _validated_option_type(option_type)
    strike = _validated_positive_float("K", K)
    rate = float(r)
    dividend = float(q)
    volatility = _validated_nonnegative_float("sigma", sigma)
    domain_max = float(Smax)

    spot_grid, dS = uniform_spot_grid(domain_max, M)
    tau_grid, dtau = uniform_tau_grid(T, N)
    intervals = len(spot_grid) - 1
    time_steps = len(tau_grid) - 1
    payoff_function, boundary_function = _option_helpers(
        option, domain_max, strike, rate, dividend
    )

    payoff = np.asarray(payoff_function(spot_grid, strike), dtype=float)
    values = payoff.copy()
    value_grid = np.empty((len(tau_grid), len(spot_grid)), dtype=float)
    value_grid[0] = values
    rannacher_results: list[PSORResult] = []
    cn_results: list[PSORResult] = []

    if dtau > 0.0:
        coefficients = black_scholes_operator_coefficients(
            spot_grid, dS=dS, r=rate, q=dividend, sigma=volatility
        )
        substep_size = dtau / substeps
        for substep in range(substeps):
            new_tau = (substep + 1) * substep_size
            new_lower, new_upper = boundary_function(new_tau)
            values, psor_result = backward_euler_psor_step(
                values,
                payoff,
                coefficients,
                substep_size,
                new_lower,
                new_upper,
                omega=omega,
                tolerance=tolerance,
                max_iter=max_iter,
            )
            rannacher_results.append(psor_result)
        value_grid[1] = values

        half_step = 0.5 * dtau
        lhs_lower = -half_step * coefficients.lower[1:]
        lhs_diagonal = 1.0 - half_step * coefficients.diagonal
        lhs_upper = -half_step * coefficients.upper[:-1]
        interior_payoff = payoff[1:-1]

        for step in range(1, time_steps):
            old_tau = float(tau_grid[step])
            new_tau = float(tau_grid[step + 1])
            old_lower, old_upper = boundary_function(old_tau)
            new_lower, new_upper = boundary_function(new_tau)

            values[0] = old_lower
            values[-1] = old_upper
            rhs = values[1:-1] + half_step * apply_black_scholes_operator(
                values, coefficients
            )
            rhs[0] += half_step * coefficients.lower[0] * new_lower
            rhs[-1] += half_step * coefficients.upper[-1] * new_upper

            psor_result = psor_lcp_solve(
                lhs_lower,
                lhs_diagonal,
                lhs_upper,
                rhs,
                interior_payoff,
                initial=values[1:-1],
                omega=omega,
                tolerance=tolerance,
                max_iter=max_iter,
            )
            cn_results.append(psor_result)

            next_values = np.empty_like(values)
            next_values[0] = new_lower
            next_values[-1] = new_upper
            next_values[1:-1] = psor_result.solution
            values = next_values
            value_grid[step + 1] = values
    else:
        substep_size = 0.0
        value_grid[:] = payoff

    psor_results = tuple(rannacher_results + cn_results)
    max_obstacle_violation = float(
        np.max(np.maximum(payoff[np.newaxis, :] - value_grid, 0.0))
    )
    all_rannacher_converged = all(result.converged for result in rannacher_results)
    all_cn_converged = all(result.converged for result in cn_results)
    all_steps_converged = all_rannacher_converged and all_cn_converged

    result = AmericanCNPSORResult(
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
        psor_results=psor_results,
        converged=all_steps_converged,
        max_obstacle_violation=max_obstacle_violation,
    )
    metadata = RannacherSmoothingMetadata(
        enabled=dtau > 0.0,
        rannacher_substeps=substeps,
        standard_dtau=float(dtau),
        rannacher_substep_size=float(substep_size),
        rannacher_psor_results=tuple(rannacher_results),
        cn_psor_results=tuple(cn_results),
        all_rannacher_steps_converged=all_rannacher_converged,
        all_cn_steps_converged=all_cn_converged,
        all_steps_converged=all_steps_converged,
        max_obstacle_violation=max_obstacle_violation,
    )
    return RannacherCNPSORResult(result=result, metadata=metadata)


def _validated_full_vector(
    name: str,
    values: Any,
    coefficients: BlackScholesOperatorCoefficients,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    expected_length = len(coefficients.interior_spots) + 2
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match the coefficient grid length.")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    return vector.copy()


def _validated_option_type(option_type: str) -> str:
    if not isinstance(option_type, str):
        raise ValueError("option_type must be 'put' or 'call'.")
    option = option_type.lower()
    if option not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'.")
    return option


def _validated_substeps(rannacher_substeps: int) -> int:
    if isinstance(rannacher_substeps, bool) or not isinstance(rannacher_substeps, int):
        raise ValueError("rannacher_substeps must be an integer.")
    if rannacher_substeps < 0:
        raise ValueError("rannacher_substeps must be nonnegative.")
    return rannacher_substeps


def _validated_positive_float(name: str, value: float) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validated_nonnegative_float(name: str, value: float) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _standard_dtau_from_grid(tau_grid: np.ndarray) -> float:
    taus = np.asarray(tau_grid, dtype=float)
    if len(taus) < 2:
        return 0.0
    return float(taus[1] - taus[0])


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

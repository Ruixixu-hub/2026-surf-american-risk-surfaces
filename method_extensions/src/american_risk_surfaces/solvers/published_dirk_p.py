"""Published DIRK-P framework from in 't Hout's American-Greeks paper.

This module intentionally keeps the paper's finite penalty parameters and
stopping rule separate from SURF's exact, residual-controlled LCP solvers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.cn import solve_tridiagonal
from american_risk_surfaces.solvers.greek_integrators import quadratic_tau_grid
from american_risk_surfaces.solvers.grid import inthout_published_spot_grid
from american_risk_surfaces.solvers.lcp import (
    LCPResidual,
    TridiagonalLCP,
    compute_lcp_residual,
)
from american_risk_surfaces.solvers.operator import (
    american_call_boundaries,
    american_put_boundaries,
    apply_black_scholes_operator,
    black_scholes_operator_coefficients_nonuniform,
)


PUBLISHED_DIRK_THETA = 1.0 - 0.5 * math.sqrt(2.0)
PUBLISHED_PENALTY_LARGE = 1.0e7
PUBLISHED_PENALTY_TOLERANCE = 1.0e-7
PUBLISHED_DAMPING_STEPS = 2
PUBLISHED_PENALTY_MAX_ITER = 100

__all__ = (
    "PUBLISHED_DIRK_THETA",
    "PUBLISHED_PENALTY_LARGE",
    "PUBLISHED_PENALTY_TOLERANCE",
    "PUBLISHED_DAMPING_STEPS",
    "PUBLISHED_PENALTY_MAX_ITER",
    "PublishedPenaltyStageResult",
    "PublishedDIRKPResult",
    "published_penalty_lcp_solve",
    "american_published_dirk_p_price",
)


@dataclass(frozen=True)
class PublishedPenaltyStageResult:
    """One paper-style penalty iteration, with an independent LCP audit."""

    solution: np.ndarray
    converged: bool
    iterations: int
    final_relative_update: float
    stopping_reason: str
    active_set_changes: tuple[int, ...]
    active_node_count: int
    residual: LCPResidual
    elapsed_seconds: float


@dataclass(frozen=True)
class PublishedDIRKPResult:
    """Full trajectory produced by the published DIRKa-P framework."""

    config: AmericanLCPConfig
    method: str
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    payoff: np.ndarray
    value_grid: np.ndarray
    values: np.ndarray
    stage_results: tuple[tuple[PublishedPenaltyStageResult, ...], ...]
    converged: bool
    max_obstacle_violation: float
    max_normalized_obstacle_violation: float
    max_normalized_equation_violation: float
    max_normalized_complementarity: float
    max_normalized_lcp_residual: float
    total_penalty_iterations: int
    maximum_stage_iterations: int
    total_seconds: float
    time_grid: str
    spatial_grid: str
    penalty_large: float
    penalty_tolerance: float
    damping_steps: int


def published_penalty_lcp_solve(
    system: TridiagonalLCP,
    initial: Any,
    *,
    large: float = PUBLISHED_PENALTY_LARGE,
    tolerance: float = PUBLISHED_PENALTY_TOLERANCE,
    max_iter: int = PUBLISHED_PENALTY_MAX_ITER,
) -> PublishedPenaltyStageResult:
    """Apply equations (3.10)--(3.11) to one tridiagonal stage LCP.

    ``large`` is added directly to active diagonal entries, as in the paper;
    it is not multiplied by the time step.  Convergence follows the published
    relative-update or unchanged-penalty-matrix rule.  The returned LCP
    residual is a separate SURF audit and does not alter the paper iteration.
    """

    if not isinstance(system, TridiagonalLCP):
        raise ValueError("system must be a TridiagonalLCP.")
    penalty = float(large)
    tol = float(tolerance)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("large must be positive and finite.")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be positive and finite.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    current = np.asarray(initial, dtype=float)
    if current.ndim != 1 or len(current) != system.size:
        raise ValueError("initial must match the LCP size.")
    if not np.all(np.isfinite(current)):
        raise ValueError("initial must contain only finite values.")
    current = current.copy()
    active = current < system.obstacle
    changes: list[int] = []
    final_update = float("inf")
    started = perf_counter()

    for iteration in range(1, max_iter + 1):
        penalty_diagonal = np.where(active, penalty, 0.0)
        candidate = solve_tridiagonal(
            system.lower,
            system.diagonal + penalty_diagonal,
            system.upper,
            system.rhs + penalty_diagonal * system.obstacle,
        )
        if not np.all(np.isfinite(candidate)):
            residual = compute_lcp_residual(system, current)
            return PublishedPenaltyStageResult(
                solution=current.copy(),
                converged=False,
                iterations=iteration,
                final_relative_update=float("inf"),
                stopping_reason="nonfinite_linear_solve",
                active_set_changes=tuple(changes),
                active_node_count=int(np.count_nonzero(active)),
                residual=residual,
                elapsed_seconds=float(perf_counter() - started),
            )
        scale = np.maximum(1.0, np.abs(candidate))
        final_update = float(np.max(np.abs(candidate - current) / scale))
        candidate_active = candidate < system.obstacle
        changes.append(int(np.count_nonzero(candidate_active != active)))
        active_unchanged = bool(np.array_equal(candidate_active, active))
        current = candidate
        active = candidate_active
        if final_update < tol or active_unchanged:
            residual = compute_lcp_residual(system, current)
            return PublishedPenaltyStageResult(
                solution=current.copy(),
                converged=True,
                iterations=iteration,
                final_relative_update=final_update,
                stopping_reason=(
                    "relative_update" if final_update < tol else "penalty_matrix_unchanged"
                ),
                active_set_changes=tuple(changes),
                active_node_count=int(np.count_nonzero(active)),
                residual=residual,
                elapsed_seconds=float(perf_counter() - started),
            )

    residual = compute_lcp_residual(system, current)
    return PublishedPenaltyStageResult(
        solution=current.copy(),
        converged=False,
        iterations=max_iter,
        final_relative_update=final_update,
        stopping_reason="iteration_guard_exhausted",
        active_set_changes=tuple(changes),
        active_node_count=int(np.count_nonzero(active)),
        residual=residual,
        elapsed_seconds=float(perf_counter() - started),
    )


def american_published_dirk_p_price(
    config: AmericanLCPConfig,
    *,
    theta: float = PUBLISHED_DIRK_THETA,
    penalty_large: float = PUBLISHED_PENALTY_LARGE,
    penalty_tolerance: float = PUBLISHED_PENALTY_TOLERANCE,
    penalty_max_iter: int = PUBLISHED_PENALTY_MAX_ITER,
    damping_steps: int = PUBLISHED_DAMPING_STEPS,
) -> PublishedDIRKPResult:
    """Run the published one-dimensional DIRKa-P numerical framework.

    The method uses the paper's quadratic time grid, first-two-step BE-P
    damping, nonuniform spatial transformation and central finite differences.
    SURF's option-specific exact boundaries extend the paper's put setup to
    dividend-paying and no-dividend calls.
    """

    if not isinstance(config, AmericanLCPConfig):
        raise ValueError("config must be an AmericanLCPConfig.")
    if not np.isclose(float(theta), PUBLISHED_DIRK_THETA, rtol=0.0, atol=1e-15):
        raise ValueError("published DIRK-P requires theta = 1 - sqrt(2)/2.")
    if not np.isclose(float(penalty_large), PUBLISHED_PENALTY_LARGE):
        raise ValueError("published DIRK-P requires Large = 1e7.")
    if not np.isclose(float(penalty_tolerance), PUBLISHED_PENALTY_TOLERANCE):
        raise ValueError("published DIRK-P requires tol = 1e-7.")
    if damping_steps != PUBLISHED_DAMPING_STEPS:
        raise ValueError("published DIRK-P requires two BE-P damping steps.")

    started = perf_counter()
    spots = inthout_published_spot_grid(config.Smax, config.K, config.M)
    coefficients = black_scholes_operator_coefficients_nonuniform(
        spots, r=config.r, q=config.q, sigma=config.sigma
    )
    taus = quadratic_tau_grid(config.T, config.N)
    payoff_function, boundary_function = _option_helpers(config)
    payoff = np.asarray(payoff_function(spots, config.K), dtype=float)
    values = payoff.copy()
    value_grid = np.empty((config.N + 1, config.M + 1), dtype=float)
    value_grid[0] = values
    all_stage_results: list[tuple[PublishedPenaltyStageResult, ...]] = []

    for step in range(1, config.N + 1):
        old_tau = float(taus[step - 1])
        new_tau = float(taus[step])
        dt = new_tau - old_tau
        old_lower, old_upper = boundary_function(old_tau)
        new_lower, new_upper = boundary_function(new_tau)
        values[0], values[-1] = old_lower, old_upper
        if step <= damping_steps:
            values, stage = _published_be_p_step(
                values,
                payoff,
                coefficients,
                dt,
                new_lower,
                new_upper,
                penalty_large,
                penalty_tolerance,
                penalty_max_iter,
            )
            stages = (stage,)
        else:
            values, stages = _published_dirk_p_step(
                values,
                payoff,
                coefficients,
                dt,
                new_lower,
                new_upper,
                float(theta),
                penalty_large,
                penalty_tolerance,
                penalty_max_iter,
            )
        all_stage_results.append(stages)
        value_grid[step] = values

    flat_stages = [stage for stages in all_stage_results for stage in stages]
    residuals = [stage.residual for stage in flat_stages]
    max_obstacle = float(np.max(np.maximum(payoff[np.newaxis, :] - value_grid, 0.0)))
    return PublishedDIRKPResult(
        config=config,
        method="published_dirka_p",
        spot_grid=spots,
        tau_grid=taus,
        payoff=payoff,
        value_grid=value_grid,
        values=values.copy(),
        stage_results=tuple(all_stage_results),
        converged=all(stage.converged for stage in flat_stages),
        max_obstacle_violation=max_obstacle,
        max_normalized_obstacle_violation=max(
            (item.normalized_obstacle_violation for item in residuals), default=0.0
        ),
        max_normalized_equation_violation=max(
            (item.normalized_equation_violation for item in residuals), default=0.0
        ),
        max_normalized_complementarity=max(
            (item.normalized_complementarity for item in residuals), default=0.0
        ),
        max_normalized_lcp_residual=max(
            (item.normalized_lcp_residual for item in residuals), default=0.0
        ),
        total_penalty_iterations=sum(stage.iterations for stage in flat_stages),
        maximum_stage_iterations=max((stage.iterations for stage in flat_stages), default=0),
        total_seconds=float(perf_counter() - started),
        time_grid="quadratic_published",
        spatial_grid="inthout_uniform_to_2K_then_sinh_d_K_over_10",
        penalty_large=float(penalty_large),
        penalty_tolerance=float(penalty_tolerance),
        damping_steps=int(damping_steps),
    )


def _published_be_p_step(
    old_values: np.ndarray,
    payoff: np.ndarray,
    coefficients: Any,
    dt: float,
    new_lower: float,
    new_upper: float,
    large: float,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, PublishedPenaltyStageResult]:
    rhs = old_values[1:-1].copy()
    rhs[0] += dt * coefficients.lower[0] * new_lower
    rhs[-1] += dt * coefficients.upper[-1] * new_upper
    system = TridiagonalLCP(
        lower=-dt * coefficients.lower[1:],
        diagonal=1.0 - dt * coefficients.diagonal,
        upper=-dt * coefficients.upper[:-1],
        rhs=rhs,
        obstacle=payoff[1:-1],
    )
    result = published_penalty_lcp_solve(
        system,
        old_values[1:-1],
        large=large,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    values = _with_boundaries(result.solution, new_lower, new_upper)
    return values, result


def _published_dirk_p_step(
    old_values: np.ndarray,
    payoff: np.ndarray,
    coefficients: Any,
    dt: float,
    new_lower: float,
    new_upper: float,
    theta: float,
    large: float,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, tuple[PublishedPenaltyStageResult, PublishedPenaltyStageResult]]:
    lhs_lower = -theta * dt * coefficients.lower[1:]
    lhs_diagonal = 1.0 - theta * dt * coefficients.diagonal
    lhs_upper = -theta * dt * coefficients.upper[:-1]
    old_operator = apply_black_scholes_operator(old_values, coefficients)

    rhs_y = old_values[1:-1] + (1.0 - theta) * dt * old_operator
    rhs_y[0] += theta * dt * coefficients.lower[0] * new_lower
    rhs_y[-1] += theta * dt * coefficients.upper[-1] * new_upper
    system_y = TridiagonalLCP(
        lhs_lower, lhs_diagonal, lhs_upper, rhs_y, payoff[1:-1]
    )
    result_y = published_penalty_lcp_solve(
        system_y,
        old_values[1:-1],
        large=large,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    y_values = _with_boundaries(result_y.solution, new_lower, new_upper)

    rhs_z = (
        old_values[1:-1]
        + 0.5 * dt * old_operator
        + (0.5 - theta) * dt * apply_black_scholes_operator(y_values, coefficients)
    )
    rhs_z[0] += theta * dt * coefficients.lower[0] * new_lower
    rhs_z[-1] += theta * dt * coefficients.upper[-1] * new_upper
    system_z = TridiagonalLCP(
        lhs_lower, lhs_diagonal, lhs_upper, rhs_z, payoff[1:-1]
    )
    result_z = published_penalty_lcp_solve(
        system_z,
        old_values[1:-1],
        large=large,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    values = _with_boundaries(result_z.solution, new_lower, new_upper)
    return values, (result_y, result_z)


def _with_boundaries(
    interior: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    values = np.empty(len(interior) + 2, dtype=float)
    values[0], values[-1] = float(lower), float(upper)
    values[1:-1] = interior
    return values


def _option_helpers(
    config: AmericanLCPConfig,
) -> tuple[Callable[..., Any], Callable[[float], tuple[float, float]]]:
    if config.option_type == "call":
        return call_payoff, lambda tau: american_call_boundaries(
            Smax=config.Smax, K=config.K, tau=tau, r=config.r, q=config.q
        )
    return put_payoff, lambda tau: american_put_boundaries(K=config.K, tau=tau)

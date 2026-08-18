"""Greek-focused DIRK and Lobatto time integrators for American options."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.cn_psor import AmericanCNPSORResult, PSORResult
from american_risk_surfaces.solvers.grid import uniform_spot_grid
from american_risk_surfaces.solvers.lcp import LCPSolveResult, TridiagonalLCP
from american_risk_surfaces.solvers.operator import (
    american_call_boundaries,
    american_put_boundaries,
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
    black_scholes_operator_coefficients_nonuniform,
)
from american_risk_surfaces.solvers.policy_iteration import policy_iteration_lcp_solve
from american_risk_surfaces.solvers.projected_lu import (
    ProjectedLUEligibility,
    ProjectedLUFactorization,
    audit_projected_lu_eligibility,
    factorize_projected_lu,
    projected_lu_lcp_solve,
)


DIRK_THETA = 1.0 - 0.5 * math.sqrt(2.0)

__all__ = (
    "DIRK_THETA",
    "AmericanGreekIntegratorResult",
    "PenaltyStageResult",
    "ProjectedLUDIRKStageAudit",
    "american_theta_policy_price",
    "american_dirk_policy_price",
    "american_dirk_projected_lu_price",
    "american_lobatto_penalty_price",
    "as_legacy_integrator_result",
    "quadratic_tau_grid",
)


@dataclass(frozen=True)
class PenaltyStageResult:
    converged: bool
    iterations: int
    final_relative_update: float
    active_set_changes: tuple[int, ...]


@dataclass(frozen=True)
class ProjectedLUDIRKStageAudit:
    """Structural and timing evidence for one Projected-LU implicit stage."""

    step_index: int
    stage_name: str
    dt: float
    mode: str
    matrix_sha256: str
    factorization_seconds: float
    precheck_seconds: float
    postcheck_seconds: float
    pre_eligibility: ProjectedLUEligibility
    post_eligibility: ProjectedLUEligibility
    result: LCPSolveResult


@dataclass(frozen=True)
class AmericanGreekIntegratorResult:
    config: AmericanLCPConfig
    method: str
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    payoff: np.ndarray
    value_grid: np.ndarray
    values: np.ndarray
    stage_results: tuple[tuple[LCPSolveResult | PenaltyStageResult, ...], ...]
    converged: bool
    max_obstacle_violation: float
    total_seconds: float
    time_grid: str
    projected_lu_stage_audits: tuple[ProjectedLUDIRKStageAudit, ...] = ()


def quadratic_tau_grid(T: float, N: int) -> np.ndarray:
    if T < 0.0:
        raise ValueError("T must be nonnegative.")
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer.")
    index = np.arange(N + 1, dtype=float)
    return (index / N) ** 2 * float(T)


def american_dirk_policy_price(
    config: AmericanLCPConfig,
    *,
    theta: float = DIRK_THETA,
    quadratic_time: bool = True,
    damping_steps: int = 2,
    spot_grid: np.ndarray | None = None,
) -> AmericanGreekIntegratorResult:
    """Second-order Cash DIRK with exact policy-iteration LCP stages."""

    return _run_integrator(
        config,
        method="dirk_policy",
        theta=float(theta),
        quadratic_time=quadratic_time,
        damping_steps=damping_steps,
        spot_grid=spot_grid,
        lcp_solver="policy_iteration",
    )


def american_dirk_projected_lu_price(
    config: AmericanLCPConfig,
    *,
    theta: float = DIRK_THETA,
    quadratic_time: bool = True,
    damping_steps: int = 2,
    spot_grid: np.ndarray | None = None,
) -> AmericanGreekIntegratorResult:
    """The frozen Cash-DIRK path with option-directed Projected-LU stages."""

    return _run_integrator(
        config,
        method="dirk_projected_lu",
        theta=float(theta),
        quadratic_time=quadratic_time,
        damping_steps=damping_steps,
        spot_grid=spot_grid,
        lcp_solver="projected_lu_single",
    )


def american_theta_policy_price(
    config: AmericanLCPConfig,
    *,
    theta: float = 0.5,
    quadratic_time: bool = True,
    damping_steps: int = 0,
    spot_grid: np.ndarray | None = None,
) -> AmericanGreekIntegratorResult:
    """Theta method with exact policy-iteration stages and optional damping."""

    method = "cn_policy" if theta == 0.5 and damping_steps == 0 else "rannacher_cn_policy"
    return _run_integrator(
        config,
        method=method,
        theta=float(theta),
        quadratic_time=quadratic_time,
        damping_steps=damping_steps,
        spot_grid=spot_grid,
        lcp_solver="policy_iteration",
    )


def american_lobatto_penalty_price(
    config: AmericanLCPConfig,
    *,
    quadratic_time: bool = True,
    damping_steps: int = 2,
    penalty: float = 1e10,
    penalty_tolerance: float = 1e-10,
    penalty_max_iter: int = 100,
    spot_grid: np.ndarray | None = None,
) -> AmericanGreekIntegratorResult:
    """Two-stage Lobatto IIIC using the published coupled penalty iteration."""

    if penalty <= 0.0 or penalty_tolerance <= 0.0 or penalty_max_iter < 1:
        raise ValueError("penalty controls must be positive.")
    return _run_integrator(
        config,
        method="lobatto_penalty",
        theta=0.5,
        quadratic_time=quadratic_time,
        damping_steps=damping_steps,
        penalty=penalty,
        penalty_tolerance=penalty_tolerance,
        penalty_max_iter=penalty_max_iter,
        spot_grid=spot_grid,
        lcp_solver="policy_iteration",
    )


def _run_integrator(
    config: AmericanLCPConfig,
    *,
    method: Literal[
        "cn_policy",
        "rannacher_cn_policy",
        "dirk_policy",
        "dirk_projected_lu",
        "lobatto_penalty",
    ],
    theta: float,
    quadratic_time: bool,
    damping_steps: int,
    penalty: float = 1e10,
    penalty_tolerance: float = 1e-10,
    penalty_max_iter: int = 100,
    spot_grid: np.ndarray | None = None,
    lcp_solver: Literal["policy_iteration", "projected_lu_single"] = "policy_iteration",
) -> AmericanGreekIntegratorResult:
    if not isinstance(config, AmericanLCPConfig):
        raise ValueError("config must be an AmericanLCPConfig.")
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in (0, 1].")
    if damping_steps < 0:
        raise ValueError("damping_steps must be nonnegative.")
    if lcp_solver not in {"policy_iteration", "projected_lu_single"}:
        raise ValueError("unsupported LCP stage solver")
    if method == "dirk_projected_lu" and lcp_solver != "projected_lu_single":
        raise ValueError("dirk_projected_lu requires Projected LU")
    started = perf_counter()
    if spot_grid is None:
        spots, dS = uniform_spot_grid(config.Smax, config.M)
        coefficients = black_scholes_operator_coefficients(
            spots, dS=dS, r=config.r, q=config.q, sigma=config.sigma
        )
    else:
        spots = np.asarray(spot_grid, dtype=float)
        if spots.ndim != 1 or len(spots) != config.M + 1:
            raise ValueError("spot_grid must contain config.M + 1 nodes.")
        if not np.isclose(spots[0], 0.0) or not np.isclose(spots[-1], config.Smax):
            raise ValueError("spot_grid boundaries must be 0 and config.Smax.")
        coefficients = black_scholes_operator_coefficients_nonuniform(
            spots, r=config.r, q=config.q, sigma=config.sigma
        )
    taus = quadratic_tau_grid(config.T, config.N) if quadratic_time else np.linspace(0.0, config.T, config.N + 1)
    payoff_function, boundary_function = _option_helpers(config)
    payoff = np.asarray(payoff_function(spots, config.K), dtype=float)
    values = payoff.copy()
    value_grid = np.empty((config.N + 1, config.M + 1), dtype=float)
    value_grid[0] = values
    all_stage_results: list[tuple[LCPSolveResult | PenaltyStageResult, ...]] = []
    projected_lu_audits: list[ProjectedLUDIRKStageAudit] = []
    for step in range(1, config.N + 1):
        old_tau = float(taus[step - 1])
        new_tau = float(taus[step])
        dt = new_tau - old_tau
        old_lower, old_upper = boundary_function(old_tau)
        new_lower, new_upper = boundary_function(new_tau)
        values[0] = old_lower
        values[-1] = old_upper
        if step <= damping_steps:
            values, stage, audit = _backward_euler_stage(
                values,
                payoff,
                coefficients,
                dt,
                new_lower,
                new_upper,
                config,
                lcp_solver=lcp_solver,
                step_index=step,
            )
            if audit is not None:
                projected_lu_audits.append(audit)
            stage_results: tuple[LCPSolveResult | PenaltyStageResult, ...] = (stage,)
        elif method in {"cn_policy", "rannacher_cn_policy"}:
            values, stage, audit = _theta_step(
                values,
                payoff,
                coefficients,
                dt,
                new_lower,
                new_upper,
                config,
                theta,
                lcp_solver=lcp_solver,
                step_index=step,
            )
            if audit is not None:
                projected_lu_audits.append(audit)
            stage_results = (stage,)
        elif method in {"dirk_policy", "dirk_projected_lu"}:
            values, stages, audits = _dirk_step(
                values,
                payoff,
                coefficients,
                dt,
                new_lower,
                new_upper,
                config,
                theta,
                lcp_solver=lcp_solver,
                step_index=step,
            )
            projected_lu_audits.extend(audits)
            stage_results = stages
        else:
            values, stage = _lobatto_penalty_step(
                values,
                payoff,
                coefficients,
                dt,
                new_lower,
                new_upper,
                penalty,
                penalty_tolerance,
                penalty_max_iter,
            )
            stage_results = (stage,)
        all_stage_results.append(stage_results)
        value_grid[step] = values

    converged = all(
        stage.converged for stages in all_stage_results for stage in stages
    )
    max_obstacle = float(np.max(np.maximum(payoff[np.newaxis, :] - value_grid, 0.0)))
    return AmericanGreekIntegratorResult(
        config=config,
        method=method,
        spot_grid=spots,
        tau_grid=taus,
        payoff=payoff,
        value_grid=value_grid,
        values=values.copy(),
        stage_results=tuple(all_stage_results),
        converged=converged,
        max_obstacle_violation=max_obstacle,
        total_seconds=float(perf_counter() - started),
        time_grid="quadratic" if quadratic_time else "uniform",
        projected_lu_stage_audits=tuple(projected_lu_audits),
    )


def _backward_euler_stage(
    old_values: np.ndarray,
    payoff: np.ndarray,
    coefficients: Any,
    dt: float,
    new_lower: float,
    new_upper: float,
    config: AmericanLCPConfig,
    *,
    lcp_solver: Literal["policy_iteration", "projected_lu_single"],
    step_index: int,
) -> tuple[np.ndarray, LCPSolveResult, ProjectedLUDIRKStageAudit | None]:
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
    result, audit = _solve_lcp_stage(
        system,
        initial=old_values[1:-1],
        config=config,
        lcp_solver=lcp_solver,
        step_index=step_index,
        stage_name="backward_euler_damping",
        dt=dt,
    )
    values = np.empty_like(old_values)
    values[0] = new_lower
    values[-1] = new_upper
    values[1:-1] = result.solution
    return values, result, audit


def _theta_step(
    old_values: np.ndarray,
    payoff: np.ndarray,
    coefficients: Any,
    dt: float,
    new_lower: float,
    new_upper: float,
    config: AmericanLCPConfig,
    theta: float,
    *,
    lcp_solver: Literal["policy_iteration", "projected_lu_single"],
    step_index: int,
) -> tuple[np.ndarray, LCPSolveResult, ProjectedLUDIRKStageAudit | None]:
    rhs = old_values[1:-1] + (1.0 - theta) * dt * apply_black_scholes_operator(
        old_values, coefficients
    )
    rhs[0] += theta * dt * coefficients.lower[0] * new_lower
    rhs[-1] += theta * dt * coefficients.upper[-1] * new_upper
    system = TridiagonalLCP(
        lower=-theta * dt * coefficients.lower[1:],
        diagonal=1.0 - theta * dt * coefficients.diagonal,
        upper=-theta * dt * coefficients.upper[:-1],
        rhs=rhs,
        obstacle=payoff[1:-1],
    )
    result, audit = _solve_lcp_stage(
        system,
        initial=old_values[1:-1],
        config=config,
        lcp_solver=lcp_solver,
        step_index=step_index,
        stage_name="theta",
        dt=dt,
    )
    values = np.empty_like(old_values)
    values[0] = new_lower
    values[-1] = new_upper
    values[1:-1] = result.solution
    return values, result, audit


def _dirk_step(
    old_values: np.ndarray,
    payoff: np.ndarray,
    coefficients: Any,
    dt: float,
    new_lower: float,
    new_upper: float,
    config: AmericanLCPConfig,
    theta: float,
    *,
    lcp_solver: Literal["policy_iteration", "projected_lu_single"],
    step_index: int,
) -> tuple[
    np.ndarray,
    tuple[LCPSolveResult, LCPSolveResult],
    tuple[ProjectedLUDIRKStageAudit, ...],
]:
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
    factorization: ProjectedLUFactorization | None = None
    pre_eligibility: ProjectedLUEligibility | None = None
    factorization_seconds = 0.0
    precheck_seconds = 0.0
    if lcp_solver == "projected_lu_single":
        (
            factorization,
            pre_eligibility,
            factorization_seconds,
            precheck_seconds,
        ) = _prepare_projected_lu(system_y)
    result_y, audit_y = _solve_lcp_stage(
        system_y,
        initial=old_values[1:-1],
        config=config,
        lcp_solver=lcp_solver,
        step_index=step_index,
        stage_name="dirk_y",
        dt=dt,
        factorization=factorization,
        pre_eligibility=pre_eligibility,
        factorization_seconds=factorization_seconds,
        precheck_seconds=precheck_seconds,
    )
    y_values = np.empty_like(old_values)
    y_values[0] = new_lower
    y_values[-1] = new_upper
    y_values[1:-1] = result_y.solution

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
    result_z, audit_z = _solve_lcp_stage(
        system_z,
        initial=result_y.solution,
        config=config,
        lcp_solver=lcp_solver,
        step_index=step_index,
        stage_name="dirk_z",
        dt=dt,
        factorization=factorization,
        pre_eligibility=pre_eligibility,
    )
    values = np.empty_like(old_values)
    values[0] = new_lower
    values[-1] = new_upper
    values[1:-1] = result_z.solution
    audits = tuple(
        audit for audit in (audit_y, audit_z) if audit is not None
    )
    return values, (result_y, result_z), audits


def _prepare_projected_lu(
    system: TridiagonalLCP,
) -> tuple[ProjectedLUFactorization, ProjectedLUEligibility, float, float]:
    setup_started = perf_counter()
    factorization = factorize_projected_lu(system, directions=("lu", "ul"))
    factorization_seconds = perf_counter() - setup_started
    check_started = perf_counter()
    eligibility = audit_projected_lu_eligibility(
        system, factorization=factorization
    )
    precheck_seconds = perf_counter() - check_started
    return factorization, eligibility, factorization_seconds, precheck_seconds


def _solve_lcp_stage(
    system: TridiagonalLCP,
    *,
    initial: np.ndarray,
    config: AmericanLCPConfig,
    lcp_solver: Literal["policy_iteration", "projected_lu_single"],
    step_index: int,
    stage_name: str,
    dt: float,
    factorization: ProjectedLUFactorization | None = None,
    pre_eligibility: ProjectedLUEligibility | None = None,
    factorization_seconds: float = 0.0,
    precheck_seconds: float = 0.0,
) -> tuple[LCPSolveResult, ProjectedLUDIRKStageAudit | None]:
    if lcp_solver == "policy_iteration":
        return (
            policy_iteration_lcp_solve(
                system,
                initial=initial,
                tolerance=config.tolerance,
                obstacle_tolerance=config.obstacle_tolerance,
                max_iter=config.max_iter,
            ),
            None,
        )
    if factorization is None or pre_eligibility is None:
        (
            factorization,
            pre_eligibility,
            factorization_seconds,
            precheck_seconds,
        ) = _prepare_projected_lu(system)
    mode = "single_put" if config.option_type == "put" else "single_call"
    result = projected_lu_lcp_solve(
        system,
        factorization,
        mode=mode,
        tolerance=config.tolerance,
        obstacle_tolerance=config.obstacle_tolerance,
    )
    postcheck_started = perf_counter()
    post_eligibility = audit_projected_lu_eligibility(
        system,
        result.solution,
        option_type=config.option_type,
        factorization=factorization,
    )
    postcheck_seconds = perf_counter() - postcheck_started
    audit = ProjectedLUDIRKStageAudit(
        step_index=step_index,
        stage_name=stage_name,
        dt=float(dt),
        mode=mode,
        matrix_sha256=factorization.matrix_sha256,
        factorization_seconds=float(factorization_seconds),
        precheck_seconds=float(precheck_seconds),
        postcheck_seconds=float(postcheck_seconds),
        pre_eligibility=pre_eligibility,
        post_eligibility=post_eligibility,
        result=result,
    )
    return result, audit


def _lobatto_penalty_step(
    old_values: np.ndarray,
    payoff: np.ndarray,
    coefficients: Any,
    dt: float,
    new_lower: float,
    new_upper: float,
    penalty: float,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, PenaltyStageResult]:
    try:
        from scipy.sparse import bmat, diags, eye
        from scipy.sparse.linalg import spsolve
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for the Lobatto penalty integrator") from exc

    size = len(old_values) - 2
    a_matrix = diags(
        (coefficients.lower[1:], coefficients.diagonal, coefficients.upper[:-1]),
        offsets=(-1, 0, 1),
        shape=(size, size),
        format="csc",
    )
    identity = eye(size, format="csc")
    b_matrix = identity - 0.5 * dt * a_matrix
    c_matrix = 0.5 * dt * a_matrix
    obstacle = payoff[1:-1]
    old = old_values[1:-1]
    boundary = np.zeros(size, dtype=float)
    boundary[0] = coefficients.lower[0] * new_lower
    boundary[-1] = coefficients.upper[-1] * new_upper
    y = old.copy()
    z = old.copy()
    previous_active: tuple[np.ndarray, np.ndarray] | None = None
    changes: list[int] = []
    final_update = float("inf")

    for iteration in range(1, max_iter + 1):
        active_y = y < obstacle
        active_z = z < obstacle
        p_diag = np.where(active_y, penalty, 0.0)
        q_diag = np.where(active_z, penalty, 0.0)
        p_matrix = diags(p_diag, format="csc")
        q_matrix = diags(q_diag, format="csc")
        matrix = bmat(
            [
                [b_matrix + p_matrix, c_matrix - q_matrix],
                [-c_matrix + p_matrix, b_matrix + q_matrix],
            ],
            format="csc",
        )
        rhs = np.concatenate(
            [
                old + (p_diag - q_diag) * obstacle,
                old + dt * boundary + (p_diag + q_diag) * obstacle,
            ]
        )
        solved = np.asarray(spsolve(matrix, rhs), dtype=float)
        new_y, new_z = solved[:size], solved[size:]
        scale = np.maximum(1.0, np.maximum(np.abs(new_y), np.abs(new_z)))
        final_update = float(
            max(
                np.max(np.abs(new_y - y) / scale),
                np.max(np.abs(new_z - z) / scale),
            )
        )
        if previous_active is None:
            changes.append(int(np.count_nonzero(active_y) + np.count_nonzero(active_z)))
        else:
            changes.append(
                int(
                    np.count_nonzero(active_y != previous_active[0])
                    + np.count_nonzero(active_z != previous_active[1])
                )
            )
        active_unchanged = previous_active is not None and np.array_equal(
            active_y, previous_active[0]
        ) and np.array_equal(active_z, previous_active[1])
        previous_active = (active_y.copy(), active_z.copy())
        y, z = new_y, new_z
        if final_update <= tolerance or active_unchanged:
            values = np.empty_like(old_values)
            values[0] = new_lower
            values[-1] = new_upper
            values[1:-1] = np.maximum(z, obstacle)
            return values, PenaltyStageResult(
                True, iteration, final_update, tuple(changes)
            )

    values = np.empty_like(old_values)
    values[0] = new_lower
    values[-1] = new_upper
    values[1:-1] = np.maximum(z, obstacle)
    return values, PenaltyStageResult(False, max_iter, final_update, tuple(changes))


def as_legacy_integrator_result(
    result: AmericanGreekIntegratorResult,
) -> AmericanCNPSORResult:
    """Adapt a time-integrator result for existing boundary and Greek diagnostics."""

    placeholders: list[PSORResult] = []
    for stages in result.stage_results:
        iterations = sum(stage.iterations for stage in stages)
        final_update = max(
            stage.residual.normalized_lcp_residual
            if isinstance(stage, LCPSolveResult)
            else stage.final_relative_update
            for stage in stages
        )
        placeholders.append(
            PSORResult(
                solution=np.empty(0),
                converged=all(stage.converged for stage in stages),
                iterations=iterations,
                final_update=final_update,
                tolerance=result.config.tolerance,
                omega=1.0,
                max_iter=result.config.max_iter,
            )
        )
    config = result.config
    return AmericanCNPSORResult(
        option_type=config.option_type,
        K=config.K,
        T=config.T,
        r=config.r,
        q=config.q,
        sigma=config.sigma,
        Smax=config.Smax,
        M=config.M,
        N=config.N,
        spot_grid=result.spot_grid.copy(),
        tau_grid=result.tau_grid.copy(),
        payoff=result.payoff.copy(),
        value_grid=result.value_grid.copy(),
        values=result.values.copy(),
        psor_results=tuple(placeholders),
        converged=result.converged,
        max_obstacle_violation=result.max_obstacle_violation,
    )


def _option_helpers(
    config: AmericanLCPConfig,
) -> tuple[Callable[..., Any], Callable[[float], tuple[float, float]]]:
    if config.option_type == "call":
        return call_payoff, lambda tau: american_call_boundaries(
            Smax=config.Smax, K=config.K, tau=tau, r=config.r, q=config.q
        )
    return put_payoff, lambda tau: american_put_boundaries(K=config.K, tau=tau)

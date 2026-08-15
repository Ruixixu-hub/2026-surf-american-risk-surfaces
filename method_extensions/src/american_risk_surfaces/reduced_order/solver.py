"""Affine online assembly and primal-dual active-set solver for RB variational inequalities."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from american_risk_surfaces.reduced_order.snapshots import boundary_lift_grid
from american_risk_surfaces.reduced_order.types import (
    AffineRBOperator,
    PrimalDualRBBasis,
    RBVISolveResult,
)
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    assemble_american_cn_lcp_step,
)
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid
from american_risk_surfaces.solvers.lcp import compute_lcp_residual
from american_risk_surfaces.solvers.operator import black_scholes_operator_coefficients


@dataclass(frozen=True)
class ReducedPDASResult:
    alpha: np.ndarray
    beta: np.ndarray
    converged: bool
    iterations: int
    residual: float
    active_set: np.ndarray
    failure_reason: str | None


@np.errstate(divide="ignore", over="ignore", invalid="ignore")
def assemble_affine_rb_operator(basis: PrimalDualRBBasis) -> AffineRBOperator:
    """Precompute the sigma-squared, rate, and dividend reduced components."""

    if basis.option_type not in {"put", "call"}:
        raise ValueError("basis option_type must be put or call")
    interior_size = basis.primal_basis.shape[0]
    spot_grid, spacing = uniform_spot_grid(4.0, interior_size + 1)
    primal = basis.primal_basis
    dual = basis.dual_generators
    components = []
    full_lift_projections = []
    left_unit = 1.0 - spot_grid / spot_grid[-1]
    right_unit = spot_grid / spot_grid[-1]
    for sigma, rate, dividend in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        matrix, left_boundary, right_boundary = _operator_dense_parts(
            spot_grid, spacing, sigma=sigma, r=rate, q=dividend
        )
        components.append(primal.T @ matrix @ primal)
        lift_columns = []
        for unit in (left_unit, right_unit):
            full_action = matrix @ unit[1:-1]
            full_action += left_boundary * unit[0] + right_boundary * unit[-1]
            lift_columns.append(primal.T @ full_action)
        full_lift_projections.append(np.column_stack(lift_columns))
    payoff_put = np.asarray(put_payoff(spot_grid, 1.0), dtype=float)[1:-1]
    payoff_call = np.asarray(call_payoff(spot_grid, 1.0), dtype=float)[1:-1]
    lift_units = np.column_stack((left_unit[1:-1], right_unit[1:-1]))
    return AffineRBOperator(
        basis=basis,
        spot_grid=spot_grid,
        mass_matrix=primal.T @ primal,
        operator_components=np.stack(components),
        primal_dual_coupling=primal.T @ dual,
        projected_lift_units=primal.T @ lift_units,
        projected_full_operator_lift_units=np.stack(full_lift_projections),
        dual_payoff_put=dual.T @ payoff_put,
        dual_payoff_call=dual.T @ payoff_call,
        dual_lift_units=dual.T @ lift_units,
        metadata={
            "component_order": ["sigma_squared", "r", "q"],
            "grid": {"K": 1.0, "Smax": 4.0, "M": interior_size + 1},
            "basis_protocol_hash": basis.metadata.get("protocol_hash"),
        },
    )


@np.errstate(divide="ignore", over="ignore", invalid="ignore")
def solve_reduced_american_vi(
    artifact: AffineRBOperator,
    config: AmericanLCPConfig,
    *,
    tolerance: float = 1e-12,
    max_iter: int = 100,
    condition_limit: float = 1e12,
) -> RBVISolveResult:
    """Solve an American-option trajectory in the reduced mixed cone."""

    started = perf_counter()
    if config.option_type != artifact.basis.option_type:
        raise ValueError("config and basis option types differ")
    if config.M != len(artifact.spot_grid) - 1:
        raise ValueError("config grid does not match the basis grid")
    if config.K != 1.0 or config.Smax != 4.0:
        raise ValueError("the frozen RB protocol requires K=1 and Smax=4")
    spot_grid = artifact.spot_grid
    tau_grid, dt = uniform_tau_grid(config.T, config.N)
    payoff_fn = put_payoff if config.option_type == "put" else call_payoff
    payoff = np.asarray(payoff_fn(spot_grid, config.K), dtype=float)
    lift = boundary_lift_grid(config, spot_grid, tau_grid)
    parameters = np.asarray((config.sigma**2, config.r, config.q), dtype=float)
    operator_reduced = np.tensordot(parameters, artifact.operator_components, axes=(0, 0))
    projected_operator_lifts = np.tensordot(
        parameters, artifact.projected_full_operator_lift_units, axes=(0, 0)
    )
    half_step = 0.5 * dt
    reduced_matrix = artifact.mass_matrix - half_step * operator_reduced
    coupling = artifact.primal_dual_coupling
    dual_payoff = (
        artifact.dual_payoff_put if config.option_type == "put" else artifact.dual_payoff_call
    )
    assembly_seconds = perf_counter() - started

    raw = np.empty((config.N + 1, config.M + 1), dtype=float)
    multipliers = np.zeros((config.N + 1, config.M - 1), dtype=float)
    raw[0] = payoff
    initial_state = payoff[1:-1] - lift[0, 1:-1]
    alpha = artifact.basis.primal_basis.T @ artifact.basis.gram_matrix @ initial_state
    beta = np.zeros(artifact.basis.dual_dimension, dtype=float)
    iterations: list[int] = []
    reduced_residual = 0.0
    pdas_seconds = 0.0
    reconstruction_seconds = 0.0
    failure_reason: str | None = None
    converged = True

    for step in range(1, config.N + 1):
        old_boundaries = np.asarray((lift[step - 1, 0], lift[step - 1, -1]))
        new_boundaries = np.asarray((lift[step, 0], lift[step, -1]))
        projected_old_lift = artifact.projected_lift_units @ old_boundaries
        projected_new_lift = artifact.projected_lift_units @ new_boundaries
        forcing = artifact.mass_matrix @ alpha
        forcing += projected_old_lift - projected_new_lift
        forcing += half_step * (
            operator_reduced @ alpha
            + projected_operator_lifts @ old_boundaries
            + projected_operator_lifts @ new_boundaries
        )
        dual_obstacle = dual_payoff - artifact.dual_lift_units @ new_boundaries
        solve_started = perf_counter()
        reduced = solve_reduced_mixed_lcp(
            reduced_matrix,
            coupling,
            forcing,
            dual_obstacle,
            initial_alpha=alpha,
            initial_beta=beta,
            tolerance=tolerance,
            max_iter=max_iter,
            condition_limit=condition_limit,
        )
        pdas_seconds += perf_counter() - solve_started
        iterations.append(reduced.iterations)
        reduced_residual = max(reduced_residual, reduced.residual)
        if not reduced.converged:
            converged = False
            failure_reason = f"time step {step}: {reduced.failure_reason}"
            raw[step:] = np.nan
            multipliers[step:] = np.nan
            break
        alpha, beta = reduced.alpha, reduced.beta
        reconstruct_started = perf_counter()
        raw[step, 0] = lift[step, 0]
        raw[step, -1] = lift[step, -1]
        raw[step, 1:-1] = lift[step, 1:-1] + artifact.basis.primal_basis @ alpha
        multipliers[step] = artifact.basis.dual_generators @ beta
        reconstruction_seconds += perf_counter() - reconstruct_started

    projection_started = perf_counter()
    projected = np.maximum(raw, payoff[np.newaxis, :]) if converged else raw.copy()
    if converged:
        projected[:, 0] = lift[:, 0]
        projected[:, -1] = lift[:, -1]
    projection_seconds = perf_counter() - projection_started
    audit_started = perf_counter()
    if converged:
        raw_audit = audit_rb_trajectory(raw, config)
        projected_audit = audit_rb_trajectory(projected, config)
        full_lcp_residual = projected_audit["normalized_lcp_residual_max"]
    else:
        raw_audit = _failed_audit()
        projected_audit = _failed_audit()
        full_lcp_residual = float("inf")
    audit_seconds = perf_counter() - audit_started
    total = perf_counter() - started
    return RBVISolveResult(
        raw_value_grid=raw,
        projected_value_grid=projected,
        reconstructed_multiplier_grid=multipliers,
        spot_grid=spot_grid.copy(),
        tau_grid=tau_grid,
        converged=converged,
        iterations_by_time=tuple(iterations),
        reduced_residual_max=float(reduced_residual),
        full_lcp_residual_max=float(full_lcp_residual),
        raw_audit=raw_audit,
        projected_audit=projected_audit,
        timing={
            "artifact_load_seconds": 0.0,
            "affine_assembly_seconds": float(assembly_seconds),
            "pdas_solve_seconds": float(pdas_seconds),
            "reconstruction_seconds": float(reconstruction_seconds),
            "projection_seconds": float(projection_seconds),
            "full_residual_audit_seconds": float(audit_seconds),
            "total_seconds": float(total),
        },
        failure_reason=failure_reason,
    )


def assemble_affine_reduced_step(
    artifact: AffineRBOperator,
    config: AmericanLCPConfig,
    previous_alpha: np.ndarray,
    old_boundaries: np.ndarray,
    new_boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one shifted reduced CN step using only precomputed components."""

    parameters = np.asarray((config.sigma**2, config.r, config.q), dtype=float)
    operator_reduced = np.tensordot(parameters, artifact.operator_components, axes=(0, 0))
    operator_lifts = np.tensordot(
        parameters, artifact.projected_full_operator_lift_units, axes=(0, 0)
    )
    half_step = 0.5 * config.T / config.N
    matrix = artifact.mass_matrix - half_step * operator_reduced
    forcing = artifact.mass_matrix @ np.asarray(previous_alpha, dtype=float)
    forcing += artifact.projected_lift_units @ (old_boundaries - new_boundaries)
    forcing += half_step * (
        operator_reduced @ previous_alpha
        + operator_lifts @ old_boundaries
        + operator_lifts @ new_boundaries
    )
    dual_payoff = (
        artifact.dual_payoff_put
        if config.option_type == "put"
        else artifact.dual_payoff_call
    )
    dual_obstacle = dual_payoff - artifact.dual_lift_units @ new_boundaries
    return matrix, forcing, dual_obstacle


def solve_reduced_mixed_lcp(
    matrix: np.ndarray,
    coupling: np.ndarray,
    forcing: np.ndarray,
    dual_obstacle: np.ndarray,
    *,
    initial_alpha: np.ndarray | None = None,
    initial_beta: np.ndarray | None = None,
    tolerance: float = 1e-12,
    max_iter: int = 100,
    condition_limit: float = 1e12,
) -> ReducedPDASResult:
    """Solve A alpha - B beta=f with beta and B'alpha-h complementary."""

    matrix = np.asarray(matrix, dtype=float)
    coupling = np.asarray(coupling, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    obstacle = np.asarray(dual_obstacle, dtype=float)
    primal_size = matrix.shape[0]
    dual_size = coupling.shape[1]
    if matrix.shape != (primal_size, primal_size) or coupling.shape[0] != primal_size:
        raise ValueError("incompatible reduced matrix and coupling shapes")
    if forcing.shape != (primal_size,) or obstacle.shape != (dual_size,):
        raise ValueError("incompatible forcing or obstacle shape")
    if tolerance <= 0.0 or max_iter < 1:
        raise ValueError("invalid PDAS controls")
    alpha = np.zeros(primal_size) if initial_alpha is None else np.asarray(initial_alpha, dtype=float).copy()
    beta = np.zeros(dual_size) if initial_beta is None else np.asarray(initial_beta, dtype=float).copy()
    if alpha.shape != (primal_size,) or beta.shape != (dual_size,):
        raise ValueError("incompatible initial coefficient shapes")
    active = beta - (coupling.T @ alpha - obstacle) >= 0.0
    residual = float("inf")
    for iteration in range(1, int(max_iter) + 1):
        block = np.zeros((primal_size + dual_size, primal_size + dual_size), dtype=float)
        rhs = np.zeros(primal_size + dual_size, dtype=float)
        block[:primal_size, :primal_size] = matrix
        block[:primal_size, primal_size:] = -coupling
        rhs[:primal_size] = forcing
        for index in range(dual_size):
            row = primal_size + index
            if active[index]:
                block[row, :primal_size] = coupling[:, index]
                rhs[row] = obstacle[index]
            else:
                block[row, primal_size + index] = 1.0
        condition = float(np.linalg.cond(block))
        if not np.isfinite(condition) or condition > condition_limit:
            return ReducedPDASResult(
                alpha, beta, False, iteration, residual, active, f"linear-system condition {condition:.3e}"
            )
        try:
            solution = np.linalg.solve(block, rhs)
        except np.linalg.LinAlgError as error:
            return ReducedPDASResult(alpha, beta, False, iteration, residual, active, str(error))
        if not np.all(np.isfinite(solution)):
            return ReducedPDASResult(alpha, beta, False, iteration, residual, active, "non-finite PDAS solution")
        alpha = solution[:primal_size]
        beta = solution[primal_size:]
        gap = coupling.T @ alpha - obstacle
        equation = matrix @ alpha - coupling @ beta - forcing
        scale = max(
            1.0,
            float(np.linalg.norm(forcing, ord=np.inf)),
            float(np.linalg.norm(matrix @ alpha, ord=np.inf)),
        )
        residual = max(
            float(np.linalg.norm(equation, ord=np.inf)) / scale,
            float(np.max(np.maximum(-beta, 0.0))) if dual_size else 0.0,
            float(np.max(np.maximum(-gap, 0.0))) if dual_size else 0.0,
            float(np.max(np.abs(beta * gap))) / scale if dual_size else 0.0,
        )
        new_active = beta - gap >= 0.0
        if residual <= tolerance:
            return ReducedPDASResult(alpha, beta, True, iteration, residual, new_active, None)
        active = new_active
    return ReducedPDASResult(alpha, beta, False, int(max_iter), residual, active, "maximum iterations reached")


def audit_rb_trajectory(value_grid: np.ndarray, config: AmericanLCPConfig) -> dict[str, float]:
    values = np.asarray(value_grid, dtype=float)
    if values.shape != (config.N + 1, config.M + 1) or not np.all(np.isfinite(values)):
        raise ValueError("value_grid must be a finite full trajectory")
    maxima = {
        "normalized_obstacle_violation_max": 0.0,
        "normalized_equation_violation_max": 0.0,
        "normalized_complementarity_max": 0.0,
        "normalized_lcp_residual_max": 0.0,
    }
    for step in range(1, config.N + 1):
        system = assemble_american_cn_lcp_step(config, values[step - 1], step)
        residual = compute_lcp_residual(system, values[step, 1:-1])
        maxima["normalized_obstacle_violation_max"] = max(
            maxima["normalized_obstacle_violation_max"], residual.normalized_obstacle_violation
        )
        maxima["normalized_equation_violation_max"] = max(
            maxima["normalized_equation_violation_max"], residual.normalized_equation_violation
        )
        maxima["normalized_complementarity_max"] = max(
            maxima["normalized_complementarity_max"], residual.normalized_complementarity
        )
        maxima["normalized_lcp_residual_max"] = max(
            maxima["normalized_lcp_residual_max"], residual.normalized_lcp_residual
        )
    return {key: float(value) for key, value in maxima.items()}


def direct_reduced_step(
    basis: PrimalDualRBBasis,
    config: AmericanLCPConfig,
    previous_alpha: np.ndarray,
    old_lift: np.ndarray,
    new_lift: np.ndarray,
    *,
    step_index: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Test-only full assembly then projection of one shifted CN step."""

    previous_full = old_lift.copy()
    previous_full[1:-1] += basis.primal_basis @ previous_alpha
    system = assemble_american_cn_lcp_step(config, previous_full, step_index)
    full_matrix = _tridiagonal_dense(system.lower, system.diagonal, system.upper)
    reduced_matrix = basis.primal_basis.T @ full_matrix @ basis.primal_basis
    forcing = basis.primal_basis.T @ (system.rhs - full_matrix @ new_lift[1:-1])
    return reduced_matrix, forcing


def _operator_dense_parts(
    spot_grid: np.ndarray,
    spacing: float,
    *,
    sigma: float,
    r: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficients = black_scholes_operator_coefficients(
        spot_grid, dS=spacing, r=r, q=q, sigma=sigma
    )
    matrix = _tridiagonal_dense(
        coefficients.lower[1:], coefficients.diagonal, coefficients.upper[:-1]
    )
    left = np.zeros(len(coefficients.diagonal))
    right = np.zeros(len(coefficients.diagonal))
    left[0] = coefficients.lower[0]
    right[-1] = coefficients.upper[-1]
    return matrix, left, right


def _tridiagonal_dense(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray) -> np.ndarray:
    result = np.diag(diagonal)
    if len(diagonal) > 1:
        result += np.diag(lower, -1) + np.diag(upper, 1)
    return result


def _failed_audit() -> dict[str, float]:
    return {
        "normalized_obstacle_violation_max": float("inf"),
        "normalized_equation_violation_max": float("inf"),
        "normalized_complementarity_max": float("inf"),
        "normalized_lcp_residual_max": float("inf"),
    }

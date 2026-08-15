"""Ticket 08: obstacle and complementarity diagnostics for American CN/PSOR results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from american_risk_surfaces.solvers.cn_psor import AmericanCNPSORResult, PSORResult
from american_risk_surfaces.solvers.operator import (
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
)

__all__ = (
    "LCPGapArrays",
    "ReconstructedLCPStep",
    "StepDiagnosticRow",
    "LCPDiagnosticSummary",
    "LCPDiagnostics",
    "compute_lcp_gap_arrays",
    "reconstruct_lcp_step",
    "summarize_lcp_step",
    "diagnose_lcp_result",
)


@dataclass(frozen=True)
class LCPGapArrays:
    """Interior-node LCP gap arrays for one American CN/PSOR time step."""

    value_gap: np.ndarray
    equation_gap: np.ndarray
    obstacle_violation: np.ndarray
    equation_violation: np.ndarray
    complementarity_product: np.ndarray


@dataclass(frozen=True)
class ReconstructedLCPStep:
    """Reconstructed interior-node LCP quantities for one time step."""

    step_index: int
    tau: float
    interior_spots: np.ndarray
    values: np.ndarray
    payoff: np.ndarray
    matrix_action: np.ndarray
    rhs: np.ndarray
    gaps: LCPGapArrays


@dataclass(frozen=True)
class StepDiagnosticRow:
    """Scalar diagnostics for one reconstructed American CN/PSOR step."""

    step_index: int
    tau: float
    psor_iterations: int
    psor_final_update: float
    min_value_gap: float
    max_obstacle_violation: float
    min_equation_gap: float
    max_equation_violation: float
    max_abs_complementarity_product: float
    mean_abs_complementarity_product: float
    exercise_like_node_count: int
    continuation_like_node_count: int
    ambiguous_node_count: int


@dataclass(frozen=True)
class LCPDiagnosticSummary:
    """Whole-result obstacle and complementarity diagnostic summary."""

    option_type: str
    K: float
    T: float
    r: float
    q: float
    sigma: float
    Smax: float
    M: int
    N: int
    value_gap_tolerance: float
    equation_gap_tolerance: float
    complementarity_tolerance: float
    all_psor_steps_converged: bool
    psor_step_count: int
    max_psor_iterations: int
    mean_psor_iterations: float
    max_final_update: float
    min_value_gap: float
    max_obstacle_violation: float
    min_equation_gap: float
    max_equation_violation: float
    max_abs_complementarity_product: float
    mean_max_abs_complementarity_product: float
    max_exercise_like_node_count: int
    max_continuation_like_node_count: int
    max_ambiguous_node_count: int
    status: str


@dataclass(frozen=True)
class LCPDiagnostics:
    """Diagnostic bundle for one American CN/PSOR result."""

    case_name: str
    summary: LCPDiagnosticSummary
    step_rows: tuple[StepDiagnosticRow, ...]


def compute_lcp_gap_arrays(
    values: Any,
    payoff: Any,
    matrix_action: Any,
    rhs: Any,
) -> LCPGapArrays:
    """Compute value, equation, violation, and complementarity arrays."""

    value_array = np.asarray(values, dtype=float)
    payoff_array = np.asarray(payoff, dtype=float)
    matrix_action_array = np.asarray(matrix_action, dtype=float)
    rhs_array = np.asarray(rhs, dtype=float)

    arrays = {
        "values": value_array,
        "payoff": payoff_array,
        "matrix_action": matrix_action_array,
        "rhs": rhs_array,
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

    if not (
        value_array.shape
        == payoff_array.shape
        == matrix_action_array.shape
        == rhs_array.shape
    ):
        raise ValueError("values, payoff, matrix_action, and rhs must have matching shapes.")
    if len(value_array) == 0:
        raise ValueError("diagnostic arrays must not be empty.")

    value_gap = value_array - payoff_array
    equation_gap = matrix_action_array - rhs_array
    return LCPGapArrays(
        value_gap=value_gap,
        equation_gap=equation_gap,
        obstacle_violation=np.maximum(-value_gap, 0.0),
        equation_violation=np.maximum(-equation_gap, 0.0),
        complementarity_product=value_gap * equation_gap,
    )


def reconstruct_lcp_step(result: AmericanCNPSORResult, step_index: int) -> ReconstructedLCPStep:
    """Reconstruct the interior-node LCP for one stored American CN/PSOR step."""

    _validate_american_result(result)
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index must be an integer.")
    if step_index < 1 or step_index >= len(result.tau_grid):
        raise ValueError("step_index must be between 1 and the final time-step index.")

    dS = _uniform_spacing(result.spot_grid, "spot_grid")
    dtau = _uniform_spacing(result.tau_grid, "tau_grid")
    if dtau <= 0.0:
        raise ValueError("result must contain positive time steps for LCP reconstruction.")

    coefficients = black_scholes_operator_coefficients(
        result.spot_grid,
        dS=dS,
        r=result.r,
        q=result.q,
        sigma=result.sigma,
    )
    half_step = 0.5 * dtau
    old_values = np.asarray(result.value_grid[step_index - 1], dtype=float)
    new_values = np.asarray(result.value_grid[step_index], dtype=float)
    rhs = old_values[1:-1] + half_step * apply_black_scholes_operator(
        old_values, coefficients
    )
    rhs[0] += half_step * coefficients.lower[0] * new_values[0]
    rhs[-1] += half_step * coefficients.upper[-1] * new_values[-1]

    lhs_lower = -half_step * coefficients.lower[1:]
    lhs_diagonal = 1.0 - half_step * coefficients.diagonal
    lhs_upper = -half_step * coefficients.upper[:-1]

    values = new_values[1:-1]
    matrix_action = lhs_diagonal * values
    matrix_action[:-1] += lhs_upper * values[1:]
    matrix_action[1:] += lhs_lower * values[:-1]

    payoff = np.asarray(result.payoff[1:-1], dtype=float)
    gaps = compute_lcp_gap_arrays(values, payoff, matrix_action, rhs)
    return ReconstructedLCPStep(
        step_index=step_index,
        tau=float(result.tau_grid[step_index]),
        interior_spots=np.asarray(result.spot_grid[1:-1], dtype=float),
        values=values.copy(),
        payoff=payoff.copy(),
        matrix_action=matrix_action.copy(),
        rhs=rhs.copy(),
        gaps=gaps,
    )


def summarize_lcp_step(
    reconstructed_step: ReconstructedLCPStep,
    psor_result: PSORResult,
    value_gap_tolerance: float = 1e-7,
    equation_gap_tolerance: float = 1e-7,
    complementarity_tolerance: float = 1e-7,
) -> StepDiagnosticRow:
    """Summarize one reconstructed LCP step into scalar diagnostic metrics."""

    value_tol = _validate_tolerance("value_gap_tolerance", value_gap_tolerance)
    equation_tol = _validate_tolerance("equation_gap_tolerance", equation_gap_tolerance)
    _validate_tolerance("complementarity_tolerance", complementarity_tolerance)
    if not isinstance(reconstructed_step, ReconstructedLCPStep):
        raise ValueError("reconstructed_step must be a ReconstructedLCPStep.")
    if not isinstance(psor_result, PSORResult):
        raise ValueError("psor_result must be a PSORResult.")

    gaps = reconstructed_step.gaps
    value_gap = gaps.value_gap
    equation_gap = gaps.equation_gap
    complementarity_abs = np.abs(gaps.complementarity_product)
    exercise_like = (value_gap <= value_tol) & (equation_gap >= -equation_tol)
    continuation_like = (value_gap > value_tol) & (np.abs(equation_gap) <= equation_tol)
    ambiguous = ~(exercise_like | continuation_like)

    return StepDiagnosticRow(
        step_index=reconstructed_step.step_index,
        tau=float(reconstructed_step.tau),
        psor_iterations=int(psor_result.iterations),
        psor_final_update=float(psor_result.final_update),
        min_value_gap=float(np.min(value_gap)),
        max_obstacle_violation=float(np.max(gaps.obstacle_violation)),
        min_equation_gap=float(np.min(equation_gap)),
        max_equation_violation=float(np.max(gaps.equation_violation)),
        max_abs_complementarity_product=float(np.max(complementarity_abs)),
        mean_abs_complementarity_product=float(np.mean(complementarity_abs)),
        exercise_like_node_count=int(np.count_nonzero(exercise_like)),
        continuation_like_node_count=int(np.count_nonzero(continuation_like)),
        ambiguous_node_count=int(np.count_nonzero(ambiguous)),
    )


def diagnose_lcp_result(
    result: AmericanCNPSORResult,
    case_name: str,
    value_gap_tolerance: float = 1e-7,
    equation_gap_tolerance: float = 1e-7,
    complementarity_tolerance: float = 1e-7,
) -> LCPDiagnostics:
    """Run obstacle and complementarity diagnostics over all stored time steps."""

    _validate_american_result(result)
    if not isinstance(case_name, str) or not case_name:
        raise ValueError("case_name must be a nonempty string.")
    value_tol = _validate_tolerance("value_gap_tolerance", value_gap_tolerance)
    equation_tol = _validate_tolerance("equation_gap_tolerance", equation_gap_tolerance)
    complementarity_tol = _validate_tolerance(
        "complementarity_tolerance", complementarity_tolerance
    )
    if len(result.psor_results) == 0:
        raise ValueError("result must contain at least one PSOR time step.")

    step_rows = []
    for step_index, psor_result in enumerate(result.psor_results, start=1):
        reconstructed = reconstruct_lcp_step(result, step_index)
        step_rows.append(
            summarize_lcp_step(
                reconstructed,
                psor_result,
                value_gap_tolerance=value_tol,
                equation_gap_tolerance=equation_tol,
                complementarity_tolerance=complementarity_tol,
            )
        )

    iterations = np.array([row.psor_iterations for row in step_rows], dtype=float)
    final_updates = np.array([row.psor_final_update for row in step_rows], dtype=float)
    max_abs_complementarity = np.array(
        [row.max_abs_complementarity_product for row in step_rows], dtype=float
    )
    max_obstacle_violation = max(row.max_obstacle_violation for row in step_rows)
    max_equation_violation = max(row.max_equation_violation for row in step_rows)
    max_abs_complementarity_product = float(np.max(max_abs_complementarity))
    status = (
        "PASS"
        if result.converged
        and max_obstacle_violation <= 1e-8
        and max_equation_violation <= 1e-6
        and max_abs_complementarity_product <= 1e-6
        else "REVIEW"
    )
    summary = LCPDiagnosticSummary(
        option_type=result.option_type,
        K=float(result.K),
        T=float(result.T),
        r=float(result.r),
        q=float(result.q),
        sigma=float(result.sigma),
        Smax=float(result.Smax),
        M=int(result.M),
        N=int(result.N),
        value_gap_tolerance=value_tol,
        equation_gap_tolerance=equation_tol,
        complementarity_tolerance=complementarity_tol,
        all_psor_steps_converged=bool(result.converged),
        psor_step_count=len(result.psor_results),
        max_psor_iterations=int(np.max(iterations)),
        mean_psor_iterations=float(np.mean(iterations)),
        max_final_update=float(np.max(final_updates)),
        min_value_gap=min(row.min_value_gap for row in step_rows),
        max_obstacle_violation=max_obstacle_violation,
        min_equation_gap=min(row.min_equation_gap for row in step_rows),
        max_equation_violation=max_equation_violation,
        max_abs_complementarity_product=max_abs_complementarity_product,
        mean_max_abs_complementarity_product=float(np.mean(max_abs_complementarity)),
        max_exercise_like_node_count=max(
            row.exercise_like_node_count for row in step_rows
        ),
        max_continuation_like_node_count=max(
            row.continuation_like_node_count for row in step_rows
        ),
        max_ambiguous_node_count=max(row.ambiguous_node_count for row in step_rows),
        status=status,
    )
    return LCPDiagnostics(
        case_name=case_name,
        summary=summary,
        step_rows=tuple(step_rows),
    )


def _validate_american_result(result: Any) -> None:
    if not isinstance(result, AmericanCNPSORResult):
        raise ValueError("result must be an AmericanCNPSORResult.")
    if result.value_grid.ndim != 2:
        raise ValueError("result.value_grid must be two-dimensional.")
    if len(result.tau_grid) != result.value_grid.shape[0]:
        raise ValueError("tau_grid length must match value_grid time dimension.")
    if len(result.spot_grid) != result.value_grid.shape[1]:
        raise ValueError("spot_grid length must match value_grid spot dimension.")
    if len(result.payoff) != len(result.spot_grid):
        raise ValueError("payoff length must match spot_grid length.")
    if len(result.psor_results) != len(result.tau_grid) - 1:
        raise ValueError("psor_results length must match the number of time steps.")
    if len(result.spot_grid) < 3:
        raise ValueError("result must include at least one interior spot node.")


def _uniform_spacing(grid: Any, name: str) -> float:
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must be a one-dimensional grid with at least two nodes.")
    spacings = np.diff(values)
    if np.any(spacings < 0.0):
        raise ValueError(f"{name} must be nondecreasing.")
    spacing = float(spacings[0])
    if not np.allclose(spacings, spacing, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced.")
    return spacing


def _validate_tolerance(name: str, value: float) -> float:
    tolerance = float(value)
    if tolerance < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return tolerance

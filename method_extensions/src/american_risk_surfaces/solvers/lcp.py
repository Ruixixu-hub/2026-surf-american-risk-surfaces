"""Shared tridiagonal LCP representation, residuals, and solver metadata."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


__all__ = (
    "LCPResidual",
    "LCPSolveResult",
    "TridiagonalLCP",
    "compute_lcp_residual",
    "psor_lcp_solve_residual",
    "tridiagonal_matvec",
)


@dataclass(frozen=True)
class TridiagonalLCP:
    """A tridiagonal LCP ``Au >= b, u >= obstacle``."""

    lower: np.ndarray
    diagonal: np.ndarray
    upper: np.ndarray
    rhs: np.ndarray
    obstacle: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "lower": np.asarray(self.lower, dtype=float),
            "diagonal": np.asarray(self.diagonal, dtype=float),
            "upper": np.asarray(self.upper, dtype=float),
            "rhs": np.asarray(self.rhs, dtype=float),
            "obstacle": np.asarray(self.obstacle, dtype=float),
        }
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")

        size = len(arrays["diagonal"])
        if size == 0:
            raise ValueError("diagonal must not be empty.")
        if len(arrays["lower"]) != size - 1:
            raise ValueError("lower length must be one less than diagonal length.")
        if len(arrays["upper"]) != size - 1:
            raise ValueError("upper length must be one less than diagonal length.")
        if len(arrays["rhs"]) != size:
            raise ValueError("rhs length must match diagonal length.")
        if len(arrays["obstacle"]) != size:
            raise ValueError("obstacle length must match diagonal length.")
        if np.any(np.abs(arrays["diagonal"]) <= 1e-14):
            raise ValueError("diagonal entries must be nonzero.")

        for name, values in arrays.items():
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def size(self) -> int:
        return int(self.diagonal.size)


@dataclass(frozen=True)
class LCPResidual:
    """Solver-independent feasibility and complementarity residuals."""

    max_obstacle_violation: float
    max_equation_violation: float
    max_abs_complementarity: float
    normalized_obstacle_violation: float
    normalized_equation_violation: float
    normalized_complementarity: float
    normalized_lcp_residual: float
    value_scale: float
    equation_scale: float


@dataclass(frozen=True)
class LCPSolveResult:
    """Common result type for residual-controlled LCP solvers."""

    solution: np.ndarray
    method: str
    converged: bool
    iterations: int
    active_set_changes: tuple[int, ...]
    residual: LCPResidual
    elapsed_seconds: float
    tolerance: float
    obstacle_tolerance: float
    max_iter: int
    component_timing: tuple[tuple[str, float], ...] = ()


def tridiagonal_matvec(system: TridiagonalLCP, values: Any) -> np.ndarray:
    """Multiply a tridiagonal LCP matrix by a vector."""

    vector = _validated_vector("values", values, system.size)
    result = system.diagonal * vector
    if system.size > 1:
        result[:-1] += system.upper * vector[1:]
        result[1:] += system.lower * vector[:-1]
    return result


def compute_lcp_residual(system: TridiagonalLCP, solution: Any) -> LCPResidual:
    """Compute normalized obstacle, equation, and complementarity residuals."""

    values = _validated_vector("solution", solution, system.size)
    matrix_action = tridiagonal_matvec(system, values)
    value_gap = values - system.obstacle
    equation_gap = matrix_action - system.rhs

    max_obstacle = float(np.max(np.maximum(-value_gap, 0.0)))
    max_equation = float(np.max(np.maximum(-equation_gap, 0.0)))
    max_complementarity = float(np.max(np.abs(value_gap * equation_gap)))
    value_scale = max(
        1.0,
        float(np.linalg.norm(values, ord=np.inf)),
        float(np.linalg.norm(system.obstacle, ord=np.inf)),
    )
    equation_scale = max(
        1.0,
        float(np.linalg.norm(matrix_action, ord=np.inf)),
        float(np.linalg.norm(system.rhs, ord=np.inf)),
    )
    normalized_obstacle = max_obstacle / value_scale
    normalized_equation = max_equation / equation_scale
    normalized_complementarity = max_complementarity / (value_scale * equation_scale)
    return LCPResidual(
        max_obstacle_violation=max_obstacle,
        max_equation_violation=max_equation,
        max_abs_complementarity=max_complementarity,
        normalized_obstacle_violation=normalized_obstacle,
        normalized_equation_violation=normalized_equation,
        normalized_complementarity=normalized_complementarity,
        normalized_lcp_residual=max(
            normalized_obstacle,
            normalized_equation,
            normalized_complementarity,
        ),
        value_scale=value_scale,
        equation_scale=equation_scale,
    )


def psor_lcp_solve_residual(
    system: TridiagonalLCP,
    initial: Any = None,
    *,
    omega: float = 1.2,
    tolerance: float = 1e-10,
    obstacle_tolerance: float = 1e-12,
    max_iter: int = 10000,
) -> LCPSolveResult:
    """Solve an LCP by PSOR using the shared residual stopping rule."""

    relaxation = float(omega)
    if not 0.0 < relaxation < 2.0:
        raise ValueError("omega must satisfy 0 < omega < 2.")
    tol, obstacle_tol, iterations_allowed = _validated_controls(
        tolerance, obstacle_tolerance, max_iter
    )
    if initial is None:
        solution = system.obstacle.copy()
    else:
        solution = np.maximum(
            _validated_vector("initial", initial, system.size).copy(),
            system.obstacle,
        )

    started = perf_counter()
    residual = compute_lcp_residual(system, solution)
    if _meets_tolerance(residual, tol, obstacle_tol):
        return _result(
            solution, "psor", True, 0, (), residual, started, tol, obstacle_tol, iterations_allowed
        )

    for iteration in range(1, iterations_allowed + 1):
        for row in range(system.size):
            row_rhs = system.rhs[row]
            if row > 0:
                row_rhs -= system.lower[row - 1] * solution[row - 1]
            if row < system.size - 1:
                row_rhs -= system.upper[row] * solution[row + 1]
            candidate = row_rhs / system.diagonal[row]
            relaxed = solution[row] + relaxation * (candidate - solution[row])
            solution[row] = max(system.obstacle[row], relaxed)

        residual = compute_lcp_residual(system, solution)
        if _meets_tolerance(residual, tol, obstacle_tol):
            return _result(
                solution,
                "psor",
                True,
                iteration,
                (),
                residual,
                started,
                tol,
                obstacle_tol,
                iterations_allowed,
            )

    return _result(
        solution,
        "psor",
        False,
        iterations_allowed,
        (),
        residual,
        started,
        tol,
        obstacle_tol,
        iterations_allowed,
    )


def _meets_tolerance(
    residual: LCPResidual, tolerance: float, obstacle_tolerance: float
) -> bool:
    return (
        residual.normalized_lcp_residual <= tolerance
        and residual.normalized_obstacle_violation <= obstacle_tolerance
    )


def _validated_controls(
    tolerance: float, obstacle_tolerance: float, max_iter: int
) -> tuple[float, float, int]:
    tol = float(tolerance)
    obstacle_tol = float(obstacle_tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be positive.")
    if obstacle_tol <= 0.0:
        raise ValueError("obstacle_tolerance must be positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    return tol, obstacle_tol, max_iter


def _validated_vector(name: str, values: Any, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must be one-dimensional with length {size}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _result(
    solution: np.ndarray,
    method: str,
    converged: bool,
    iterations: int,
    active_set_changes: tuple[int, ...],
    residual: LCPResidual,
    started: float,
    tolerance: float,
    obstacle_tolerance: float,
    max_iter: int,
    component_timing: tuple[tuple[str, float], ...] = (),
) -> LCPSolveResult:
    return LCPSolveResult(
        solution=solution.copy(),
        method=method,
        converged=converged,
        iterations=int(iterations),
        active_set_changes=active_set_changes,
        residual=residual,
        elapsed_seconds=float(perf_counter() - started),
        tolerance=tolerance,
        obstacle_tolerance=obstacle_tolerance,
        max_iter=max_iter,
        component_timing=tuple(
            (str(name), float(seconds)) for name, seconds in component_timing
        ),
    )

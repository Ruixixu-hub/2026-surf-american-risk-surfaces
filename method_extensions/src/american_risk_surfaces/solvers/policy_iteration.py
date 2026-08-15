"""Policy iteration for American-option tridiagonal LCPs."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from american_risk_surfaces.solvers.cn import solve_tridiagonal
from american_risk_surfaces.solvers.lcp import (
    LCPSolveResult,
    TridiagonalLCP,
    _meets_tolerance,
    _result,
    _validated_controls,
    _validated_vector,
    compute_lcp_residual,
    tridiagonal_matvec,
)


__all__ = ("policy_iteration_lcp_solve",)


def policy_iteration_lcp_solve(
    system: TridiagonalLCP,
    initial: Any = None,
    *,
    tolerance: float = 1e-10,
    obstacle_tolerance: float = 1e-12,
    max_iter: int = 10000,
) -> LCPSolveResult:
    """Solve ``min(Au-b, u-obstacle)=0`` by row-switching policy iteration."""

    if not isinstance(system, TridiagonalLCP):
        raise ValueError("system must be a TridiagonalLCP.")
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
            solution,
            "policy_iteration",
            True,
            0,
            (),
            residual,
            started,
            tol,
            obstacle_tol,
            iterations_allowed,
        )

    previous_active: np.ndarray | None = None
    changes: list[int] = []
    for iteration in range(1, iterations_allowed + 1):
        equation_gap = tridiagonal_matvec(system, solution) - system.rhs
        value_gap = solution - system.obstacle
        continuation = equation_gap <= value_gap
        active = ~continuation
        if previous_active is None:
            changes.append(int(np.count_nonzero(active)))
        else:
            changes.append(int(np.count_nonzero(active != previous_active)))
        previous_active = active.copy()

        diagonal = np.where(continuation, system.diagonal, 1.0)
        rhs = np.where(continuation, system.rhs, system.obstacle)
        lower = system.lower.copy()
        upper = system.upper.copy()
        if system.size > 1:
            lower[~continuation[1:]] = 0.0
            upper[~continuation[:-1]] = 0.0

        solution = solve_tridiagonal(lower, diagonal, upper, rhs)
        residual = compute_lcp_residual(system, solution)
        if _meets_tolerance(residual, tol, obstacle_tol):
            return _result(
                solution,
                "policy_iteration",
                True,
                iteration,
                tuple(changes),
                residual,
                started,
                tol,
                obstacle_tol,
                iterations_allowed,
            )

    return _result(
        solution,
        "policy_iteration",
        False,
        iterations_allowed,
        tuple(changes),
        residual,
        started,
        tol,
        obstacle_tol,
        iterations_allowed,
    )

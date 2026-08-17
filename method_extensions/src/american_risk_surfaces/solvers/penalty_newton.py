"""Penalty/semismooth-Newton candidate for tridiagonal obstacle LCPs.

The method solves the finite-penalty equation

``A u - b - penalty * max(obstacle - u, 0) = 0``

by freezing the active penalty set and solving a tridiagonal linear system.
It is deliberately certified against the same unpenalized LCP residual as
PSOR, Policy Iteration, and Projected LU.  Convergence of the penalized
equation alone never marks the method as a successful LCP solve.
"""

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


__all__ = ("penalty_newton_lcp_solve",)


def penalty_newton_lcp_solve(
    system: TridiagonalLCP,
    initial: Any = None,
    *,
    penalty: float = 1e10,
    tolerance: float = 1e-12,
    obstacle_tolerance: float = 1e-12,
    max_iter: int = 100,
) -> LCPSolveResult:
    """Solve a penalized obstacle equation and audit the original LCP.

    A stable penalty active set terminates the Newton iteration even when the
    original LCP tolerance is not met.  In that case ``converged`` is false,
    which makes finite-penalty bias visible to the common experiment gate.
    """

    if not isinstance(system, TridiagonalLCP):
        raise ValueError("system must be a TridiagonalLCP.")
    penalty_value = float(penalty)
    if not np.isfinite(penalty_value) or penalty_value <= 0.0:
        raise ValueError("penalty must be positive and finite.")
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
            "penalty_newton",
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
        active = solution < system.obstacle
        if previous_active is None:
            changes.append(int(np.count_nonzero(active)))
        else:
            changes.append(int(np.count_nonzero(active != previous_active)))

        penalized_diagonal = system.diagonal + penalty_value * active.astype(float)
        penalized_rhs = system.rhs + penalty_value * active.astype(float) * system.obstacle
        candidate = solve_tridiagonal(
            system.lower,
            penalized_diagonal,
            system.upper,
            penalized_rhs,
        )
        update = float(np.max(np.abs(candidate - solution)))
        solution = candidate
        residual = compute_lcp_residual(system, solution)
        if _meets_tolerance(residual, tol, obstacle_tol):
            return _result(
                solution,
                "penalty_newton",
                True,
                iteration,
                tuple(changes),
                residual,
                started,
                tol,
                obstacle_tol,
                iterations_allowed,
            )

        equation = (
            tridiagonal_matvec(system, solution)
            - system.rhs
            - penalty_value * np.maximum(system.obstacle - solution, 0.0)
        )
        equation_scale = max(
            1.0,
            float(np.linalg.norm(system.rhs, ord=np.inf)),
            float(np.linalg.norm(tridiagonal_matvec(system, solution), ord=np.inf)),
        )
        equation_residual = float(np.linalg.norm(equation, ord=np.inf)) / equation_scale
        stable = previous_active is not None and np.array_equal(active, previous_active)
        previous_active = active.copy()
        update_scale = max(1.0, float(np.linalg.norm(solution, ord=np.inf)))
        if stable and (equation_residual <= tol or update / update_scale <= tol):
            return _result(
                solution,
                "penalty_newton",
                False,
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
        "penalty_newton",
        False,
        iterations_allowed,
        tuple(changes),
        residual,
        started,
        tol,
        obstacle_tol,
        iterations_allowed,
    )

"""Projected LU/Brennan--Schwartz solvers for tridiagonal LCPs.

The implementation follows the obstacle-shifted formulation

``z = u - obstacle`` and ``v = rhs - A obstacle``

and applies the directional projected substitutions described by Ikonen and
Toivanen.  The double sweep follows Algorithm 1--2 of Le Floc'h (2022).
Every candidate is certified with the repository-wide LCP residual; no
iterative fallback or post-hoc payoff projection is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np

from american_risk_surfaces.solvers.lcp import (
    LCPResidual,
    LCPSolveResult,
    TridiagonalLCP,
    _meets_tolerance,
    _result,
    _validated_controls,
    _validated_vector,
    compute_lcp_residual,
    tridiagonal_matvec,
)


ProjectedLUDirection = Literal["lu", "ul"]
ProjectedLUMode = Literal["single_put", "single_call", "double"]

__all__ = (
    "ProjectedLUDirection",
    "ProjectedLUEligibility",
    "ProjectedLUFactorization",
    "ProjectedLUMode",
    "ProjectedLUStepAudit",
    "audit_projected_lu_eligibility",
    "audit_projected_lu_step",
    "factorize_projected_lu",
    "projected_lu_lcp_solve",
    "reconstruct_projected_lu_matrix",
)


@dataclass(frozen=True)
class ProjectedLUFactorization:
    """Reusable directional factorizations of one tridiagonal matrix."""

    lu_diagonal: np.ndarray
    lu_super: np.ndarray
    ul_diagonal: np.ndarray
    ul_lower: np.ndarray
    matrix_sha256: str
    pivot_tolerance: float
    available_directions: tuple[ProjectedLUDirection, ...]

    def __post_init__(self) -> None:
        for name in ("lu_diagonal", "lu_super", "ul_diagonal", "ul_lower"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite one-dimensional array.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        directions = tuple(self.available_directions)
        if not directions or any(direction not in {"lu", "ul"} for direction in directions):
            raise ValueError("available_directions must contain 'lu' and/or 'ul'.")
        object.__setattr__(self, "available_directions", directions)
        if float(self.pivot_tolerance) <= 0.0:
            raise ValueError("pivot_tolerance must be positive.")


@dataclass(frozen=True)
class ProjectedLUEligibility:
    """Sufficient-condition and contact-geometry audit for projected LU."""

    positive_diagonal: bool
    nonpositive_offdiagonals: bool
    strictly_diagonally_dominant: bool
    positive_lu_pivots: bool
    positive_ul_pivots: bool
    m_matrix_sufficient_conditions: bool
    contact_components: int | None
    expected_contact_geometry: bool | None
    theorem_eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProjectedLUStepAudit:
    """Per-time-step audit combining theorem and numerical certification."""

    mode: str
    sweeps: int
    theorem_eligible: bool
    numerically_certified: bool
    active_components: int
    residual: LCPResidual


def factorize_projected_lu(
    system: TridiagonalLCP,
    *,
    directions: Iterable[ProjectedLUDirection] = ("lu", "ul"),
    pivot_tolerance: float = 1e-14,
) -> ProjectedLUFactorization:
    """Factor the matrix once for directional projected substitutions."""

    if not isinstance(system, TridiagonalLCP):
        raise ValueError("system must be a TridiagonalLCP.")
    tolerance = float(pivot_tolerance)
    if tolerance <= 0.0:
        raise ValueError("pivot_tolerance must be positive.")
    requested = tuple(dict.fromkeys(str(direction) for direction in directions))
    if not requested or any(direction not in {"lu", "ul"} for direction in requested):
        raise ValueError("directions must contain 'lu' and/or 'ul'.")

    size = system.size
    lu_diagonal = np.array([], dtype=float)
    lu_super = np.array([], dtype=float)
    if "lu" in requested:
        lu_diagonal = np.empty(size, dtype=float)
        lu_super = np.empty(max(size - 1, 0), dtype=float)
        lu_diagonal[0] = system.diagonal[0]
        _require_pivot(lu_diagonal[0], tolerance, "LU", 0)
        for row in range(size - 1):
            lu_super[row] = system.upper[row] / lu_diagonal[row]
            lu_diagonal[row + 1] = (
                system.diagonal[row + 1] - system.lower[row] * lu_super[row]
            )
            _require_pivot(lu_diagonal[row + 1], tolerance, "LU", row + 1)

    ul_diagonal = np.array([], dtype=float)
    ul_lower = np.array([], dtype=float)
    if "ul" in requested:
        ul_diagonal = np.empty(size, dtype=float)
        ul_lower = np.empty(max(size - 1, 0), dtype=float)
        ul_diagonal[-1] = system.diagonal[-1]
        _require_pivot(ul_diagonal[-1], tolerance, "UL", size - 1)
        for row in range(size - 2, -1, -1):
            ul_lower[row] = system.lower[row] / ul_diagonal[row + 1]
            ul_diagonal[row] = (
                system.diagonal[row] - system.upper[row] * ul_lower[row]
            )
            _require_pivot(ul_diagonal[row], tolerance, "UL", row)

    return ProjectedLUFactorization(
        lu_diagonal=lu_diagonal,
        lu_super=lu_super,
        ul_diagonal=ul_diagonal,
        ul_lower=ul_lower,
        matrix_sha256=_matrix_sha256(system),
        pivot_tolerance=tolerance,
        available_directions=tuple(requested),  # type: ignore[arg-type]
    )


def reconstruct_projected_lu_matrix(
    system: TridiagonalLCP,
    factorization: ProjectedLUFactorization,
    *,
    direction: ProjectedLUDirection,
) -> np.ndarray:
    """Reconstruct a dense matrix for factorization tests and protocol audits."""

    _validate_factorization(system, factorization, direction)
    size = system.size
    reconstructed = np.zeros((size, size), dtype=float)
    if direction == "lu":
        reconstructed[np.arange(size), np.arange(size)] = factorization.lu_diagonal
        if size > 1:
            reconstructed[np.arange(1, size), np.arange(size - 1)] = system.lower
            reconstructed[np.arange(size - 1), np.arange(1, size)] = (
                factorization.lu_diagonal[:-1] * factorization.lu_super
            )
            reconstructed[np.arange(1, size), np.arange(1, size)] += (
                system.lower * factorization.lu_super
            )
        return reconstructed
    reconstructed[np.arange(size), np.arange(size)] = factorization.ul_diagonal
    if size > 1:
        reconstructed[np.arange(size - 1), np.arange(1, size)] = system.upper
        reconstructed[np.arange(1, size), np.arange(size - 1)] = (
            factorization.ul_diagonal[1:] * factorization.ul_lower
        )
        reconstructed[np.arange(size - 1), np.arange(size - 1)] += (
            system.upper * factorization.ul_lower
        )
    return reconstructed


def projected_lu_lcp_solve(
    system: TridiagonalLCP,
    factorization: ProjectedLUFactorization,
    *,
    mode: ProjectedLUMode,
    tolerance: float = 1e-12,
    obstacle_tolerance: float = 1e-12,
) -> LCPSolveResult:
    """Solve an obstacle-shifted tridiagonal LCP by one or two projected sweeps."""

    if not isinstance(system, TridiagonalLCP):
        raise ValueError("system must be a TridiagonalLCP.")
    if mode not in {"single_put", "single_call", "double"}:
        raise ValueError("mode must be 'single_put', 'single_call', or 'double'.")
    tol, obstacle_tol, _ = _validated_controls(tolerance, obstacle_tolerance, 1)
    required = ("ul",) if mode == "single_put" else ("lu",)
    if mode == "double":
        required = ("lu", "ul")
    for direction in required:
        _validate_factorization(system, factorization, direction)

    started = perf_counter()
    component_started = perf_counter()
    transformed_rhs = system.rhs - tridiagonal_matvec(system, system.obstacle)
    transformation_seconds = perf_counter() - component_started
    sweep_started = perf_counter()
    if mode == "single_call":
        shifted = _lu_projected_sweep(system, factorization, transformed_rhs)
        sweeps = 1
    elif mode == "single_put":
        shifted = _ul_projected_sweep(system, factorization, transformed_rhs)
        sweeps = 1
    else:
        call_shifted = _lu_projected_sweep(system, factorization, transformed_rhs)
        shifted = _ul_projected_sweep(
            system, factorization, transformed_rhs, base=call_shifted
        )
        sweeps = 2
    sweep_seconds = perf_counter() - sweep_started
    inverse_started = perf_counter()
    solution = system.obstacle + shifted
    inverse_seconds = perf_counter() - inverse_started
    residual_started = perf_counter()
    residual = compute_lcp_residual(system, solution)
    residual_seconds = perf_counter() - residual_started
    converged = _meets_tolerance(residual, tol, obstacle_tol)
    return _result(
        solution,
        f"projected_lu_{mode}",
        converged,
        sweeps,
        (),
        residual,
        started,
        tol,
        obstacle_tol,
        sweeps,
        (
            ("obstacle_transform", transformation_seconds),
            ("projected_sweep", sweep_seconds),
            ("inverse_transform", inverse_seconds),
            ("residual_audit", residual_seconds),
        ),
    )


def audit_projected_lu_eligibility(
    system: TridiagonalLCP,
    solution: Any = None,
    option_type: str | None = None,
    *,
    pivot_tolerance: float = 1e-14,
    sign_tolerance: float = 1e-15,
    contact_tolerance: float = 1e-10,
) -> ProjectedLUEligibility:
    """Audit standard M-matrix sufficient conditions and contact topology."""

    if not isinstance(system, TridiagonalLCP):
        raise ValueError("system must be a TridiagonalLCP.")
    if option_type is not None and str(option_type).lower() not in {"put", "call"}:
        raise ValueError("option_type must be 'put', 'call', or None.")
    if pivot_tolerance <= 0.0 or sign_tolerance < 0.0 or contact_tolerance <= 0.0:
        raise ValueError("audit tolerances are invalid.")

    positive_diagonal = bool(np.all(system.diagonal > pivot_tolerance))
    nonpositive_offdiagonals = bool(
        np.all(system.lower <= sign_tolerance)
        and np.all(system.upper <= sign_tolerance)
    )
    row_offdiagonal = np.zeros(system.size, dtype=float)
    if system.size > 1:
        row_offdiagonal[1:] += np.abs(system.lower)
        row_offdiagonal[:-1] += np.abs(system.upper)
    strict_dominance = bool(np.all(system.diagonal > row_offdiagonal))

    positive_lu_pivots = _positive_pivots(system, "lu", pivot_tolerance)
    positive_ul_pivots = _positive_pivots(system, "ul", pivot_tolerance)
    m_matrix = bool(
        positive_diagonal
        and nonpositive_offdiagonals
        and strict_dominance
        and positive_lu_pivots
        and positive_ul_pivots
    )

    contact_components: int | None = None
    expected_geometry: bool | None = None
    if solution is not None:
        values = _validated_vector("solution", solution, system.size)
        scale = max(
            1.0,
            float(np.linalg.norm(values, ord=np.inf)),
            float(np.linalg.norm(system.obstacle, ord=np.inf)),
        )
        # In a truncated vanilla grid, both value and payoff can be numerically
        # zero in the far out-of-the-money tail.  Those nodes are not an
        # economically meaningful exercise set and would create a false second
        # contact component under a pure value-gap threshold.  Restrict the
        # topology audit to positive-payoff nodes; the full LCP residual still
        # audits every node without this restriction.
        active = (
            values - system.obstacle <= contact_tolerance * scale
        ) & (system.obstacle > contact_tolerance * scale)
        contact_components = _component_count(active)
        option = str(option_type).lower() if option_type is not None else None
        expected_geometry = _expected_contact_geometry(active, option)

    reasons: list[str] = []
    if not positive_diagonal:
        reasons.append("nonpositive_or_small_diagonal")
    if not nonpositive_offdiagonals:
        reasons.append("positive_offdiagonal")
    if not strict_dominance:
        reasons.append("not_strictly_diagonally_dominant")
    if not positive_lu_pivots:
        reasons.append("nonpositive_lu_pivot")
    if not positive_ul_pivots:
        reasons.append("nonpositive_ul_pivot")
    if expected_geometry is False:
        reasons.append("unexpected_contact_geometry")
    theorem_eligible = bool(m_matrix and expected_geometry is not False)
    return ProjectedLUEligibility(
        positive_diagonal=positive_diagonal,
        nonpositive_offdiagonals=nonpositive_offdiagonals,
        strictly_diagonally_dominant=strict_dominance,
        positive_lu_pivots=positive_lu_pivots,
        positive_ul_pivots=positive_ul_pivots,
        m_matrix_sufficient_conditions=m_matrix,
        contact_components=contact_components,
        expected_contact_geometry=expected_geometry,
        theorem_eligible=theorem_eligible,
        reasons=tuple(reasons),
    )


def audit_projected_lu_step(
    system: TridiagonalLCP,
    result: LCPSolveResult,
    *,
    option_type: str,
    reference: Any = None,
    comparison_tolerance: float = 1e-9,
) -> ProjectedLUStepAudit:
    """Combine eligibility, shared residual, and optional reference agreement."""

    if not isinstance(result, LCPSolveResult):
        raise ValueError("result must be an LCPSolveResult.")
    eligibility = audit_projected_lu_eligibility(
        system, result.solution, option_type=option_type
    )
    reference_ok = True
    if reference is not None:
        target = _validated_vector("reference", reference, system.size)
        reference_ok = bool(
            np.max(np.abs(result.solution - target)) <= float(comparison_tolerance)
        )
    certified = bool(result.converged and reference_ok)
    return ProjectedLUStepAudit(
        mode=result.method,
        sweeps=result.iterations,
        theorem_eligible=eligibility.theorem_eligible,
        numerically_certified=certified,
        active_components=eligibility.contact_components or 0,
        residual=result.residual,
    )


def _lu_projected_sweep(
    system: TridiagonalLCP,
    factorization: ProjectedLUFactorization,
    transformed_rhs: np.ndarray,
) -> np.ndarray:
    size = system.size
    intermediate = np.empty(size, dtype=float)
    intermediate[0] = transformed_rhs[0] / factorization.lu_diagonal[0]
    for row in range(1, size):
        intermediate[row] = (
            transformed_rhs[row] - system.lower[row - 1] * intermediate[row - 1]
        ) / factorization.lu_diagonal[row]
    shifted = np.empty(size, dtype=float)
    shifted[-1] = max(intermediate[-1], 0.0)
    for row in range(size - 2, -1, -1):
        shifted[row] = max(
            intermediate[row] - factorization.lu_super[row] * shifted[row + 1],
            0.0,
        )
    return shifted


def _ul_projected_sweep(
    system: TridiagonalLCP,
    factorization: ProjectedLUFactorization,
    transformed_rhs: np.ndarray,
    *,
    base: np.ndarray | None = None,
) -> np.ndarray:
    size = system.size
    intermediate = np.empty(size, dtype=float)
    intermediate[-1] = transformed_rhs[-1] / factorization.ul_diagonal[-1]
    for row in range(size - 2, -1, -1):
        intermediate[row] = (
            transformed_rhs[row] - system.upper[row] * intermediate[row + 1]
        ) / factorization.ul_diagonal[row]
    shifted = np.zeros(size, dtype=float) if base is None else np.asarray(base, dtype=float).copy()
    shifted[0] = max(shifted[0], intermediate[0], 0.0)
    for row in range(1, size):
        reverse_candidate = (
            intermediate[row] - factorization.ul_lower[row - 1] * shifted[row - 1]
        )
        shifted[row] = max(shifted[row], reverse_candidate, 0.0)
    return shifted


def _positive_pivots(
    system: TridiagonalLCP,
    direction: ProjectedLUDirection,
    pivot_tolerance: float,
) -> bool:
    try:
        factorization = factorize_projected_lu(
            system, directions=(direction,), pivot_tolerance=pivot_tolerance
        )
    except ValueError:
        return False
    pivots = (
        factorization.lu_diagonal
        if direction == "lu"
        else factorization.ul_diagonal
    )
    return bool(np.all(pivots > pivot_tolerance))


def _validate_factorization(
    system: TridiagonalLCP,
    factorization: ProjectedLUFactorization,
    direction: ProjectedLUDirection,
) -> None:
    if not isinstance(factorization, ProjectedLUFactorization):
        raise ValueError("factorization must be a ProjectedLUFactorization.")
    if direction not in factorization.available_directions:
        raise ValueError(f"factorization does not contain the {direction!r} direction.")
    if factorization.matrix_sha256 != _matrix_sha256(system):
        raise ValueError("factorization matrix hash does not match the LCP matrix.")
    expected_diagonal = (
        factorization.lu_diagonal if direction == "lu" else factorization.ul_diagonal
    )
    expected_offdiagonal = (
        factorization.lu_super if direction == "lu" else factorization.ul_lower
    )
    if len(expected_diagonal) != system.size or len(expected_offdiagonal) != system.size - 1:
        raise ValueError("factorization shape does not match the LCP matrix.")


def _matrix_sha256(system: TridiagonalLCP) -> str:
    digest = hashlib.sha256()
    for values in (system.lower, system.diagonal, system.upper):
        contiguous = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _require_pivot(value: float, tolerance: float, direction: str, row: int) -> None:
    if not np.isfinite(value) or abs(float(value)) <= tolerance:
        raise ValueError(f"{direction} factorization has a zero/small pivot at row {row}.")


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    starts = mask & ~np.r_[False, mask[:-1]]
    return int(np.count_nonzero(starts))


def _expected_contact_geometry(mask: np.ndarray, option_type: str | None) -> bool:
    active_indices = np.flatnonzero(mask)
    if option_type == "put":
        if len(active_indices) == 0:
            return False
        return bool(np.array_equal(active_indices, np.arange(active_indices[-1] + 1)))
    if option_type == "call":
        if len(active_indices) == 0:
            return True
        return bool(
            np.array_equal(
                active_indices,
                np.arange(active_indices[0], len(mask)),
            )
        )
    return _component_count(mask) <= 1

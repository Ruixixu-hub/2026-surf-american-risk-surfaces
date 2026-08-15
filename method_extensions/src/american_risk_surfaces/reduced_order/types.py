"""Immutable public result types for primal/dual reduced-basis VI models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RBFOMSnapshot:
    regime_id: str
    option_type: str
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    payoff: np.ndarray
    value_grid: np.ndarray
    boundary_lift_grid: np.ndarray
    lifted_state_grid: np.ndarray
    multiplier_grid: np.ndarray
    active_set_grid: np.ndarray
    residual_by_time: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class PrimalDualRBBasis:
    option_type: str
    primal_basis: np.ndarray
    dual_generators: np.ndarray
    gram_matrix: np.ndarray
    primal_dimension: int
    dual_dimension: int
    inf_sup_constant: float
    condition_number: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class RBVISolveResult:
    raw_value_grid: np.ndarray
    projected_value_grid: np.ndarray
    reconstructed_multiplier_grid: np.ndarray
    spot_grid: np.ndarray
    tau_grid: np.ndarray
    converged: bool
    iterations_by_time: tuple[int, ...]
    reduced_residual_max: float
    full_lcp_residual_max: float
    raw_audit: dict[str, float]
    projected_audit: dict[str, float]
    timing: dict[str, float]
    failure_reason: str | None


@dataclass(frozen=True)
class RBBasisArtifact:
    path: Path
    basis: PrimalDualRBBasis


@dataclass(frozen=True)
class AffineRBOperator:
    basis: PrimalDualRBBasis
    spot_grid: np.ndarray
    mass_matrix: np.ndarray
    operator_components: np.ndarray
    primal_dual_coupling: np.ndarray
    projected_lift_units: np.ndarray
    projected_full_operator_lift_units: np.ndarray
    dual_payoff_put: np.ndarray
    dual_payoff_call: np.ndarray
    dual_lift_units: np.ndarray
    metadata: dict[str, object]

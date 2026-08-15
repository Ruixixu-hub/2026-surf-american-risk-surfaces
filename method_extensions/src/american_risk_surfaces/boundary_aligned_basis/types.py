"""Public immutable types for the oracle boundary-alignment experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BoundaryAlignmentConfig:
    canonical_points: int = 1921
    boundary_threshold: float = 1e-6
    interpolation: Literal["pchip"] = "pchip"
    pairing_tolerance: float = 1e-4


@dataclass(frozen=True)
class BoundaryAlignmentMap:
    boundary_spot: float
    physical_grid: np.ndarray
    canonical_grid: np.ndarray
    physical_at_canonical: np.ndarray
    jacobian: np.ndarray
    boundary_found: bool


@dataclass(frozen=True)
class OracleBasisArtifact:
    arm: Literal["U", "A", "L", "AL"]
    option_type: str
    primal_bases: tuple[np.ndarray, ...]
    dual_generators: tuple[np.ndarray, ...]
    bin_edges: np.ndarray
    bin_labels: tuple[str, ...]
    active_dimension: int
    total_stored_modes: int
    metric_grids: tuple[np.ndarray, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class OracleFalsificationResult:
    raw_value_grid: np.ndarray
    projected_value_grid: np.ndarray
    reconstructed_multiplier_grid: np.ndarray
    metrics: dict[str, float]
    oracle_information_used: tuple[str, ...]

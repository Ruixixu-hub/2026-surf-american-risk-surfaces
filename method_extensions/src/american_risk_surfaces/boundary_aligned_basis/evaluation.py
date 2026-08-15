"""Oracle projection/reconstruction and full structural audit on validation."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls

from american_risk_surfaces.boundary_aligned_basis.alignment import (
    align_dual_multiplier,
    align_primal_state,
    build_boundary_alignment_map,
    inverse_align_dual_multiplier,
    inverse_align_primal_state,
)
from american_risk_surfaces.boundary_aligned_basis.metric import WeightedH1Metric
from american_risk_surfaces.boundary_aligned_basis.types import (
    BoundaryAlignmentConfig,
    OracleBasisArtifact,
    OracleFalsificationResult,
)
from american_risk_surfaces.reduced_order.metrics import score_value_trajectory
from american_risk_surfaces.reduced_order.snapshots import (
    boundary_lift_grid,
    trajectory_multipliers,
)
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig


def evaluate_oracle_basis(
    artifact: OracleBasisArtifact,
    config: AmericanLCPConfig,
    value_grid: np.ndarray,
    *,
    alignment_config: BoundaryAlignmentConfig | None = None,
    boundary_path: np.ndarray,
    boundary_found: np.ndarray,
) -> OracleFalsificationResult:
    if artifact.option_type != config.option_type:
        raise ValueError("artifact and option family differ")
    if artifact.metadata.get("protocol_hash") is None:
        raise ValueError("artifact lacks a frozen protocol hash")
    aligned = artifact.arm in {"A", "AL"}
    localized = artifact.arm in {"L", "AL"}
    align_config = alignment_config or BoundaryAlignmentConfig(
        canonical_points=len(artifact.metric_grids[0])
    )
    values = np.asarray(value_grid, dtype=float)
    spots = np.linspace(0.0, config.Smax, config.M + 1)
    taus = np.linspace(0.0, config.T, config.N + 1)
    lift = boundary_lift_grid(config, spots, taus)
    payoff = values[0].copy()
    state = values[:, 1:-1] - lift[:, 1:-1]
    multipliers, _, _ = trajectory_multipliers(config, values)
    multipliers = np.maximum(multipliers, 0.0)
    raw = np.empty_like(values)
    raw[0] = payoff
    raw[:, 0] = values[:, 0]
    raw[:, -1] = values[:, -1]
    predicted_multiplier = np.zeros_like(multipliers)
    transform_seconds = 0.0
    projection_seconds = 0.0
    reconstruction_seconds = 0.0
    for index in range(1, len(taus)):
        bin_index = _artifact_bin_index(
            artifact,
            config.option_type,
            boundary_path[index],
            bool(boundary_found[index]),
            localized,
        )
        primal = artifact.primal_bases[bin_index]
        dual = artifact.dual_generators[bin_index]
        metric = WeightedH1Metric.from_grid(artifact.metric_grids[bin_index])
        mapping = None
        source_state = state[index]
        source_multiplier = multipliers[index]
        started = perf_counter()
        if aligned:
            mapping = build_boundary_alignment_map(
                boundary_path[index] if boundary_found[index] else None,
                align_config,
                physical_grid=spots,
            )
            source_state = align_primal_state(source_state, mapping)
            source_multiplier = align_dual_multiplier(source_multiplier, mapping)
        transform_seconds += perf_counter() - started
        started = perf_counter()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            coefficients = primal.T @ metric.apply(source_state)
            reduced_state = primal @ coefficients
        reduced_multiplier = _dual_nnls_projection(source_multiplier, dual, metric)
        projection_seconds += perf_counter() - started
        started = perf_counter()
        if aligned:
            assert mapping is not None
            reduced_state = inverse_align_primal_state(reduced_state, mapping)
            reduced_multiplier = inverse_align_dual_multiplier(reduced_multiplier, mapping)
        raw[index, 1:-1] = lift[index, 1:-1] + reduced_state
        predicted_multiplier[index] = reduced_multiplier
        reconstruction_seconds += perf_counter() - started
    projected = np.maximum(raw, payoff[None, :])
    projected[0] = payoff
    projected[:, 0] = values[:, 0]
    projected[:, -1] = values[:, -1]
    _, predicted_active, raw_residual = trajectory_multipliers(config, raw)
    _, _, projected_residual = trajectory_multipliers(config, projected)
    reference_active = multipliers > 1e-10
    predicted_active_from_lambda = predicted_multiplier > 1e-10
    tp = int(np.sum(predicted_active_from_lambda & reference_active))
    fp = int(np.sum(predicted_active_from_lambda & ~reference_active))
    fn = int(np.sum(~predicted_active_from_lambda & reference_active))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = score_value_trajectory(
        raw,
        values,
        payoff,
        spots,
        taus,
        config.option_type,
        predicted_multiplier=predicted_multiplier,
        reference_multiplier=multipliers,
    )
    metrics.update(
        {
            "projected_price_rmse": float(np.sqrt(np.mean((projected - values) ** 2))),
            "raw_obstacle_violation": float(np.max(np.maximum(payoff[None, :] - raw, 0.0))),
            "projected_obstacle_violation": float(
                np.max(np.maximum(payoff[None, :] - projected, 0.0))
            ),
            "raw_full_lcp_residual": float(np.max(raw_residual[:, 3])),
            "projected_full_lcp_residual": float(np.max(projected_residual[:, 3])),
            "active_set_precision": precision,
            "active_set_recall": recall,
            "active_set_f1": 2.0 * precision * recall / max(precision + recall, 1e-15),
            "active_set_disagreement": float(np.mean(predicted_active != reference_active)),
            "timing_transform_seconds": transform_seconds,
            "timing_projection_seconds": projection_seconds,
            "timing_reconstruction_seconds": reconstruction_seconds,
        }
    )
    information = ["validation_true_boundary"] if aligned or localized else []
    if localized:
        information.append("validation_true_boundary_bin")
    return OracleFalsificationResult(
        raw,
        projected,
        predicted_multiplier,
        metrics,
        tuple(information),
    )


def _dual_nnls_projection(
    multiplier: np.ndarray,
    generators: np.ndarray,
    metric: WeightedH1Metric,
) -> np.ndarray:
    if generators.shape[1] == 0:
        return np.zeros_like(multiplier)
    inverse_generators = metric.solve(generators.T).T
    inverse_multiplier = metric.solve(multiplier)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        hessian = generators.T @ inverse_generators
        linear = generators.T @ inverse_multiplier
    try:
        root = cholesky(hessian, lower=False, check_finite=True)
    except np.linalg.LinAlgError:
        # A generator can become nearly dependent without invalidating the cone;
        # report it through the reconstruction error rather than regularize it.
        raise RuntimeError("dual cone Gram matrix is singular") from None
    target = solve_triangular(root.T, linear, lower=True)
    coefficients, _ = nnls(root, target, maxiter=20 * generators.shape[1])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return generators @ coefficients


def _artifact_bin_index(
    artifact: OracleBasisArtifact,
    family: str,
    boundary: float,
    found: bool,
    localized: bool,
) -> int:
    if not localized:
        return 0
    if not found:
        if "no_boundary" not in artifact.bin_labels:
            raise RuntimeError("validation row has no boundary but artifact lacks that bin")
        return artifact.bin_labels.index("no_boundary")
    value = float(boundary) if family == "put" else float((boundary - 1.0) / 3.0)
    return int(np.clip(np.digitize(value, artifact.bin_edges[1:-1]), 0, len(artifact.bin_edges) - 2))

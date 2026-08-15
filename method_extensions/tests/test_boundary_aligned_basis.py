from __future__ import annotations

import numpy as np
import pytest

from american_risk_surfaces.boundary_aligned_basis import (
    BoundaryAlignmentConfig,
    align_dual_multiplier,
    align_primal_state,
    build_boundary_alignment_map,
    inverse_align_dual_multiplier,
    inverse_align_primal_state,
)
from american_risk_surfaces.boundary_aligned_basis.basis import (
    _bin_assignments,
    _TrainingRows,
    angle_greedy_rows,
    pod_greedy_rows,
)
from american_risk_surfaces.boundary_aligned_basis.metric import WeightedH1Metric
from american_risk_surfaces.boundary_aligned_basis.protocol import assert_oracle_regime_allowed


def test_alignment_map_endpoints_monotonicity_and_boundary_to_strike() -> None:
    config = BoundaryAlignmentConfig(canonical_points=1921)
    mapping = build_boundary_alignment_map(0.8, config)
    assert mapping.physical_at_canonical[0] == 0.0
    assert mapping.physical_at_canonical[-1] == 4.0
    assert np.all(np.diff(mapping.physical_at_canonical) > 0.0)
    assert np.interp(1.0, mapping.canonical_grid, mapping.physical_at_canonical) == pytest.approx(0.8)


def test_primal_and_dual_roundtrip_and_nonnegativity() -> None:
    mapping = build_boundary_alignment_map(0.75, BoundaryAlignmentConfig(canonical_points=1921))
    spots = mapping.physical_grid[1:-1]
    state = spots * (4.0 - spots) * np.exp(-spots)
    multiplier = np.maximum(1.0 - spots, 0.0)
    aligned_state = align_primal_state(state, mapping)
    aligned_multiplier = align_dual_multiplier(multiplier, mapping)
    restored_state = inverse_align_primal_state(aligned_state, mapping)
    restored_multiplier = inverse_align_dual_multiplier(aligned_multiplier, mapping)
    assert np.sqrt(np.mean((state - restored_state) ** 2)) < 1e-5
    assert np.sqrt(np.mean((multiplier - restored_multiplier) ** 2)) < 2.5e-5
    assert np.min(aligned_multiplier) >= 0.0


def test_identity_map_and_strict_negative_multiplier_rule() -> None:
    mapping = build_boundary_alignment_map(None, BoundaryAlignmentConfig(canonical_points=1921))
    assert not mapping.boundary_found
    multiplier = np.zeros(119)
    multiplier[10] = -2e-14
    with pytest.raises(ValueError):
        align_dual_multiplier(multiplier, mapping)


def test_weighted_h1_metric_and_deterministic_greedy() -> None:
    grid = np.linspace(0.0, 4.0, 21)
    metric = WeightedH1Metric.from_grid(grid)
    assert np.min(metric.eigenvalues()) > 0.0
    x = grid[1:-1]
    states = np.vstack([np.sin((index + 1) * np.pi * x / 4.0) for index in range(6)])
    regimes = np.asarray([f"r{index // 2}" for index in range(6)])
    basis1, history1 = pod_greedy_rows(states, regimes, metric, 3)
    basis2, history2 = pod_greedy_rows(states, regimes, metric, 3)
    assert np.allclose(basis1, basis2)
    assert history1 == history2
    identity = basis1.T @ metric.apply(basis1.T).T
    assert np.allclose(identity, np.eye(3), atol=1e-9)
    multipliers = np.maximum(states, 0.0)
    dual, _ = angle_greedy_rows(
        multipliers, regimes, np.arange(6), metric, 3
    )
    assert np.min(dual) >= 0.0


def test_quantile_assignment_no_boundary_and_budget_inputs() -> None:
    grid = np.linspace(0.0, 4.0, 11)
    rows = _TrainingRows(
        np.ones((5, 9)),
        np.ones((5, 9)),
        np.asarray(["a", "a", "b", "b", "c"]),
        np.arange(5),
        np.asarray([0.1, 0.2, 0.8, 0.9, np.nan]),
        np.asarray([True, True, True, True, False]),
        grid,
        0.0,
    )
    assignments, _, labels = _bin_assignments(rows, "call", 2)
    assert labels[-1] == "no_boundary"
    assert assignments[-1] == len(labels) - 1


def test_static_heldout_and_no_dividend_call_seal() -> None:
    assert_oracle_regime_allowed("validation", "call", 0.03)
    with pytest.raises(PermissionError):
        assert_oracle_regime_allowed("test", "put", 0.0)
    with pytest.raises(PermissionError):
        assert_oracle_regime_allowed("validation", "call", 0.0)

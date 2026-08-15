"""Unit, algebraic reproduction, leakage, and smoke tests for primal/dual RB-VI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.reduced_order import (
    RBRegime,
    assemble_affine_rb_operator,
    assemble_affine_reduced_step,
    build_primal_dual_basis,
    direct_reduced_step,
    g_orthonormalize,
    load_snapshot,
    reduced_inf_sup_constant,
    solve_reduced_american_vi,
    solve_reduced_mixed_lcp,
    weighted_h1_gram,
)
from american_risk_surfaces.reduced_order.basis import angle_greedy
from american_risk_surfaces.reduced_order.snapshots import (
    boundary_lift_grid,
    generate_fom_snapshot,
)
from american_risk_surfaces.reduced_order.study import create_scoring_marker
from american_risk_surfaces.reduced_order.types import PrimalDualRBBasis
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.grid import uniform_spot_grid, uniform_tau_grid


def _regime(identifier: str, option: str = "put", *, q: float = 0.02, sigma: float = 0.2) -> RBRegime:
    return RBRegime(identifier, option, 0.25, sigma, 0.03, q, 1.0, 4.0, 16, 12, "train")


class ReducedBasisAlgebraTests(unittest.TestCase):
    def test_boundary_lift_matches_both_exact_endpoints(self) -> None:
        spots, _ = uniform_spot_grid(4.0, 16)
        taus, _ = uniform_tau_grid(0.5, 8)
        put = AmericanLCPConfig("put", 1.0, 0.5, 0.05, 0.02, 0.2, 4.0, 16, 8)
        call = AmericanLCPConfig("call", 1.0, 0.5, 0.05, 0.08, 0.2, 4.0, 16, 8)
        put_lift = boundary_lift_grid(put, spots, taus)
        call_lift = boundary_lift_grid(call, spots, taus)
        npt.assert_allclose(put_lift[:, 0], 1.0)
        npt.assert_allclose(put_lift[:, -1], 0.0)
        npt.assert_allclose(call_lift[:, 0], 0.0)
        self.assertTrue(np.all(call_lift[:, -1] >= 3.0))

    def test_gram_spd_orthonormality_supremizer_and_inf_sup(self) -> None:
        spots, _ = uniform_spot_grid(4.0, 12)
        gram = weighted_h1_gram(spots)
        self.assertTrue(np.all(np.linalg.eigvalsh(gram) > 0.0))
        rng = np.random.default_rng(17)
        dual = np.abs(rng.normal(size=(11, 3)))
        dual /= np.sqrt(np.sum(dual * np.linalg.solve(gram, dual), axis=0))
        pod = g_orthonormalize(rng.normal(size=(11, 3)), gram)
        supremizers = np.linalg.solve(gram, dual)
        primal = g_orthonormalize(np.column_stack((pod, supremizers)), gram)
        npt.assert_allclose(primal.T @ gram @ primal, np.eye(primal.shape[1]), atol=1e-10)
        npt.assert_allclose(gram @ supremizers, dual, atol=1e-12)
        self.assertGreater(reduced_inf_sup_constant(primal, dual, gram), 1e-8)

    def test_angle_greedy_is_deterministic_nonnegative(self) -> None:
        spots, _ = uniform_spot_grid(4.0, 10)
        gram = weighted_h1_gram(spots)
        rng = np.random.default_rng(29)
        snapshots = [(f"r{index}", index, np.abs(rng.normal(size=9))) for index in range(8)]
        first, history_a = angle_greedy(snapshots, gram, 4)
        second, history_b = angle_greedy(reversed(snapshots), gram, 4)
        npt.assert_allclose(first, second)
        self.assertEqual(history_a, history_b)
        self.assertGreaterEqual(float(np.min(first)), 0.0)

    def test_pdas_known_solution_tie_singular_and_nonconvergence(self) -> None:
        result = solve_reduced_mixed_lcp(
            np.eye(2), np.eye(2), np.array([0.0, 2.0]), np.array([1.0, 1.0])
        )
        self.assertTrue(result.converged)
        npt.assert_allclose(result.alpha, [1.0, 2.0])
        npt.assert_allclose(result.beta, [1.0, 0.0])
        tie = solve_reduced_mixed_lcp(np.ones((1, 1)), np.ones((1, 1)), np.ones(1), np.ones(1))
        self.assertTrue(tie.converged)
        singular = solve_reduced_mixed_lcp(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.zeros(1)
        )
        self.assertFalse(singular.converged)
        capped = solve_reduced_mixed_lcp(
            np.eye(2), np.array([[1.0, 1.0], [0.0, 0.0]]), np.zeros(2), np.ones(2), max_iter=1
        )
        self.assertFalse(capped.converged)


class ReducedBasisSnapshotAndSolverTests(unittest.TestCase):
    def test_snapshot_multiplier_sign_train_only_and_call_zero_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            put_path = Path(directory) / "put.npz"
            snapshot = generate_fom_snapshot(_regime("put_a"), put_path)
            self.assertGreaterEqual(float(np.min(snapshot.multiplier_grid)), -1e-12)
            self.assertLessEqual(float(np.max(snapshot.residual_by_time[:, 3])), 1e-12)
            loaded = load_snapshot(put_path)
            npt.assert_allclose(loaded.value_grid, snapshot.value_grid)
            with self.assertRaises(ValueError):
                invalid = RBRegime("heldout", "put", 0.25, 0.2, 0.03, 0.02, 1.0, 4.0, 16, 12, "test")
                generate_fom_snapshot(invalid, Path(directory) / "invalid.npz")
            call = generate_fom_snapshot(
                _regime("call_zero", "call", q=0.0), Path(directory) / "call.npz"
            )
            self.assertLessEqual(float(np.max(np.abs(call.multiplier_grid))), 1e-10)

    def test_affine_matches_full_projection_and_rb_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, sigma in enumerate((0.16, 0.22, 0.30)):
                path = Path(directory) / f"put_{index}.npz"
                generate_fom_snapshot(_regime(f"put_{index}", sigma=sigma), path)
                paths.append(path)
            basis = build_primal_dual_basis(paths, "put", 2)
            artifact = assemble_affine_rb_operator(basis)
            snapshot = load_snapshot(paths[0])
            config = _regime("target", sigma=0.16).config()
            alpha = basis.primal_basis.T @ basis.gram_matrix @ snapshot.lifted_state_grid[0]
            affine_matrix, affine_forcing, _ = assemble_affine_reduced_step(
                artifact,
                config,
                alpha,
                snapshot.boundary_lift_grid[0, [0, -1]],
                snapshot.boundary_lift_grid[1, [0, -1]],
            )
            direct_matrix, direct_forcing = direct_reduced_step(
                basis,
                config,
                alpha,
                snapshot.boundary_lift_grid[0],
                snapshot.boundary_lift_grid[1],
            )
            npt.assert_allclose(affine_matrix, direct_matrix, atol=1e-12, rtol=0.0)
            npt.assert_allclose(affine_forcing, direct_forcing, atol=1e-12, rtol=0.0)
            result = solve_reduced_american_vi(artifact, config)
            self.assertTrue(result.converged, result.failure_reason)
            self.assertTrue(np.all(np.isfinite(result.raw_value_grid)))
            self.assertLessEqual(
                float(np.max(snapshot.payoff[np.newaxis, :] - result.projected_value_grid)),
                1e-12,
            )
            self.assertIn("normalized_lcp_residual_max", result.raw_audit)
            self.assertIn("normalized_lcp_residual_max", result.projected_audit)

    def test_basis_rejects_nontrain_snapshot_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            generate_fom_snapshot(_regime("put_train"), path)
            with np.load(path, allow_pickle=False) as source:
                values = {name: source[name] for name in source.files}
            metadata = json.loads(str(values["metadata_json"]))
            metadata["regime"]["split"] = "validation"
            values["metadata_json"] = json.dumps(metadata)
            leaked = Path(directory) / "leaked.npz"
            np.savez_compressed(leaked, **values)
            with self.assertRaises(ValueError):
                build_primal_dual_basis([leaked], "put", 1)

    def test_scoring_marker_is_immutable_for_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.json"
            frozen.write_text('{"dimension": 4}', encoding="utf-8")
            marker = create_scoring_marker(frozen, output_dir=root / "heldout")
            first = marker.read_text(encoding="utf-8")
            self.assertEqual(marker, create_scoring_marker(frozen, output_dir=root / "heldout"))
            self.assertEqual(first, marker.read_text(encoding="utf-8"))
            frozen.write_text('{"dimension": 8}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                create_scoring_marker(frozen, output_dir=root / "heldout")


if __name__ == "__main__":
    unittest.main()

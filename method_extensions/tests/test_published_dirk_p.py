"""Tests for the published in 't Hout DIRK-P implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.grid import inthout_published_spot_grid
from american_risk_surfaces.solvers.lcp import TridiagonalLCP
from american_risk_surfaces.solvers.published_dirk_p import (
    PUBLISHED_DIRK_THETA,
    american_published_dirk_p_price,
    published_penalty_lcp_solve,
)


class PublishedDIRKPTests(unittest.TestCase):
    def test_published_grid_matches_piecewise_formula(self) -> None:
        grid = inthout_published_spot_grid(5.0, 1.0, 50)
        d = 0.1
        xi_int = 2.0 / d
        xi_max = xi_int + np.arcsinh(5.0 / d - xi_int)
        xi = np.linspace(0.0, xi_max, 51)
        expected = np.where(
            xi <= xi_int,
            d * xi,
            2.0 + d * np.sinh(xi - xi_int),
        )
        expected[0], expected[-1] = 0.0, 5.0
        np.testing.assert_allclose(grid, expected, rtol=0.0, atol=1e-14)
        self.assertTrue(np.all(np.diff(grid) > 0.0))

    def test_published_grid_is_not_surf_strike_centered_grid(self) -> None:
        from american_risk_surfaces.solvers.grid import sinh_spot_grid

        paper = inthout_published_spot_grid(4.0, 1.0, 120)
        surf = sinh_spot_grid(4.0, 1.0, 120)
        self.assertGreater(float(np.max(np.abs(paper - surf))), 1e-3)

    def test_penalty_iteration_uses_finite_paper_penalty(self) -> None:
        system = TridiagonalLCP(
            lower=np.array([0.0]),
            diagonal=np.ones(2),
            upper=np.array([0.0]),
            rhs=np.array([2.0, 0.0]),
            obstacle=np.ones(2),
        )
        result = published_penalty_lcp_solve(system, np.ones(2))
        self.assertTrue(result.converged)
        self.assertEqual("penalty_matrix_unchanged", result.stopping_reason)
        self.assertAlmostEqual(2.0, result.solution[0], places=12)
        self.assertAlmostEqual(1.0e7 / (1.0 + 1.0e7), result.solution[1], places=14)
        self.assertGreater(result.residual.normalized_obstacle_violation, 0.0)
        self.assertLess(result.residual.normalized_lcp_residual, 1e-6)

    def test_small_put_and_call_runs_are_finite(self) -> None:
        for option_type, q in (("put", 0.0), ("call", 0.06), ("call", 0.0)):
            with self.subTest(option_type=option_type, q=q):
                config = AmericanLCPConfig(
                    option_type=option_type,
                    K=1.0,
                    T=0.5,
                    r=0.05,
                    q=q,
                    sigma=0.2,
                    Smax=4.0,
                    M=40,
                    N=20,
                    tolerance=1e-12,
                    obstacle_tolerance=1e-12,
                )
                result = american_published_dirk_p_price(config)
                self.assertTrue(result.converged)
                self.assertTrue(np.all(np.isfinite(result.value_grid)))
                self.assertEqual("quadratic_published", result.time_grid)
                self.assertEqual(2, result.damping_steps)
                self.assertEqual(config.N + 1, len(result.tau_grid))
                self.assertAlmostEqual(config.T, result.tau_grid[-1])
                self.assertGreaterEqual(
                    float(np.min(result.value_grid - result.payoff[np.newaxis, :])),
                    -1e-6,
                )

    def test_published_parameters_cannot_be_retuned(self) -> None:
        config = AmericanLCPConfig(
            option_type="put",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.0,
            sigma=0.2,
            Smax=4.0,
            M=20,
            N=10,
        )
        self.assertAlmostEqual(1.0 - math.sqrt(2.0) / 2.0, PUBLISHED_DIRK_THETA)
        with self.assertRaises(ValueError):
            american_published_dirk_p_price(config, penalty_large=1e8)
        with self.assertRaises(ValueError):
            american_published_dirk_p_price(config, penalty_tolerance=1e-8)
        with self.assertRaises(ValueError):
            american_published_dirk_p_price(config, damping_steps=0)


if __name__ == "__main__":
    unittest.main()

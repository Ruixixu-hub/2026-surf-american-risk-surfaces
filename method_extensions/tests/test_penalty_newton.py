"""Tests for the finite-penalty semismooth-Newton LCP candidate."""

from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
)
from american_risk_surfaces.solvers.lcp import TridiagonalLCP
from american_risk_surfaces.solvers.penalty_newton import penalty_newton_lcp_solve
from american_risk_surfaces.solvers.policy_iteration import policy_iteration_lcp_solve


class PenaltyNewtonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = TridiagonalLCP(
            lower=np.array([-1.0]),
            diagonal=np.array([2.0, 2.0]),
            upper=np.array([-1.0]),
            rhs=np.array([1.0, -2.0]),
            obstacle=np.zeros(2),
        )

    def test_large_penalty_matches_known_lcp_at_declared_tolerance(self) -> None:
        result = penalty_newton_lcp_solve(
            self.system,
            penalty=1e14,
            tolerance=2e-12,
            obstacle_tolerance=2e-12,
        )
        expected = policy_iteration_lcp_solve(
            self.system,
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        self.assertTrue(result.converged)
        npt.assert_allclose(result.solution, expected.solution, atol=2e-12, rtol=0.0)

    def test_penalized_equation_does_not_hide_lcp_failure(self) -> None:
        result = penalty_newton_lcp_solve(
            self.system,
            penalty=1.0,
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        self.assertFalse(result.converged)
        self.assertGreater(result.residual.normalized_lcp_residual, 1e-12)

    def test_invalid_inputs_fail_explicitly(self) -> None:
        for penalty in (0.0, -1.0, float("nan")):
            with self.subTest(penalty=penalty):
                with self.assertRaises(ValueError):
                    penalty_newton_lcp_solve(self.system, penalty=penalty)

    def test_cn_marcher_exposes_penalty_candidate_without_fallback(self) -> None:
        config = AmericanLCPConfig(
            "put",
            1.0,
            0.25,
            0.05,
            0.02,
            0.2,
            4.0,
            30,
            30,
            tolerance=1e-10,
            obstacle_tolerance=1e-10,
            penalty=1e12,
            penalty_newton_max_iter=100,
        )
        result = american_cn_lcp_price(config, lcp_solver="penalty_newton")
        self.assertEqual("penalty_newton", result.solver)
        self.assertEqual("previous_slice", result.initializer)
        self.assertTrue(all(step.method == "penalty_newton" for step in result.lcp_results))
        self.assertTrue(np.all(np.isfinite(result.value_grid)))


if __name__ == "__main__":
    unittest.main()

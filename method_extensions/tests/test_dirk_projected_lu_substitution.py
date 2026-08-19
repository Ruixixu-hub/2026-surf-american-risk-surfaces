"""Tests for the frozen DIRK+sinh Projected-LU substitution experiment."""

from __future__ import annotations

from unittest import mock
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.method_extensions.dirk_projected_lu_study import (
    FROZEN_M,
    VALUE_MATCH_TOLERANCE,
    _interpolated_operator_l1_bound,
    _load_existing_reference_rows,
)
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.greek_integrators import (
    american_dirk_policy_price,
    american_dirk_projected_lu_price,
)
from american_risk_surfaces.solvers.grid import sinh_spot_grid


class DIRKProjectedLUSubstitutionTests(unittest.TestCase):
    def test_all_twelve_preexisting_reference_errors_are_valid(self) -> None:
        rows = _load_existing_reference_rows()
        self.assertEqual(12, len(rows))
        for row in rows.values():
            self.assertGreater(float(row["delta_max_error"]), 0.0)
            self.assertGreater(float(row["gamma_max_error"]), 0.0)

    def test_price_operator_bound_dominates_exact_fd_perturbation(self) -> None:
        grid = sinh_spot_grid(4.0, 1.0, FROZEN_M)
        query = np.linspace(0.8, 1.2, 37)
        rng = np.random.default_rng(17)
        perturbation = rng.uniform(
            -VALUE_MATCH_TOLERANCE,
            VALUE_MATCH_TOLERANCE,
            size=len(grid),
        )
        for order in (1, 2):
            bound = VALUE_MATCH_TOLERANCE * _interpolated_operator_l1_bound(
                grid, query, derivative_order=order
            )
            derivative = _fd(grid, perturbation, order)
            observed = float(np.max(np.abs(np.interp(query, grid, derivative))))
            self.assertLessEqual(observed, bound * (1.0 + 1e-12))

    def test_projected_lu_reuses_frozen_dirk_path_and_matches_policy(self) -> None:
        for option_type, q in (("put", 0.03), ("call", 0.06), ("call", 0.0)):
            with self.subTest(option_type=option_type, q=q):
                config = AmericanLCPConfig(
                    option_type,
                    1.0,
                    0.5,
                    0.05,
                    q,
                    0.2,
                    4.0,
                    60,
                    40,
                    tolerance=1e-12,
                    obstacle_tolerance=1e-12,
                )
                grid = sinh_spot_grid(4.0, 1.0, 60)
                policy = american_dirk_policy_price(config, spot_grid=grid)
                candidate = american_dirk_projected_lu_price(config, spot_grid=grid)
                self.assertTrue(policy.converged)
                self.assertTrue(candidate.converged)
                npt.assert_array_equal(candidate.spot_grid, policy.spot_grid)
                npt.assert_array_equal(candidate.tau_grid, policy.tau_grid)
                npt.assert_allclose(
                    candidate.value_grid,
                    policy.value_grid,
                    atol=1e-9,
                    rtol=0.0,
                )
                self.assertEqual(2 + 2 * (config.N - 2), len(candidate.projected_lu_stage_audits))
                self.assertEqual(0.0, np.max(np.abs(candidate.value_grid[:, (0, -1)] - policy.value_grid[:, (0, -1)])))

    def test_no_policy_fallback_and_one_factorization_per_time_step(self) -> None:
        config = AmericanLCPConfig(
            "put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 40, 20,
            tolerance=1e-12, obstacle_tolerance=1e-12,
        )
        grid = sinh_spot_grid(4.0, 1.0, 40)
        from american_risk_surfaces.solvers import greek_integrators as integrators

        with mock.patch.object(
            integrators,
            "policy_iteration_lcp_solve",
            side_effect=AssertionError("Policy fallback is forbidden"),
        ), mock.patch.object(
            integrators,
            "factorize_projected_lu",
            wraps=integrators.factorize_projected_lu,
        ) as factorize:
            result = integrators.american_dirk_projected_lu_price(
                config, spot_grid=grid
            )
        self.assertTrue(result.converged)
        self.assertEqual(config.N, factorize.call_count)


def _fd(grid: np.ndarray, values: np.ndarray, order: int) -> np.ndarray:
    left = grid[1:-1] - grid[:-2]
    right = grid[2:] - grid[1:-1]
    result = np.full_like(values, np.nan)
    if order == 1:
        a = -right / (left * (left + right))
        b = (right - left) / (left * right)
        c = left / (right * (left + right))
    else:
        a = 2.0 / (left * (left + right))
        b = -2.0 / (left * right)
        c = 2.0 / (right * (left + right))
    result[1:-1] = a * values[:-2] + b * values[1:-1] + c * values[2:]
    return result


if __name__ == "__main__":
    unittest.main()

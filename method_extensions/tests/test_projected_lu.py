"""Tests for projected LU/Brennan--Schwartz LCP solvers."""

from __future__ import annotations

from unittest import mock
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
)
from american_risk_surfaces.solvers.lcp import TridiagonalLCP
from american_risk_surfaces.solvers.policy_iteration import policy_iteration_lcp_solve
from american_risk_surfaces.solvers.projected_lu import (
    audit_projected_lu_eligibility,
    factorize_projected_lu,
    projected_lu_lcp_solve,
    reconstruct_projected_lu_matrix,
)


class ProjectedLUFactorizationTests(unittest.TestCase):
    def test_lu_and_ul_reconstruct_matrix(self) -> None:
        system = _system(np.array([0.2, -0.3, 1.1, -0.4, 0.8]))
        factorization = factorize_projected_lu(system)
        expected = _dense(system)
        for direction in ("lu", "ul"):
            with self.subTest(direction=direction):
                reconstructed = reconstruct_projected_lu_matrix(
                    system, factorization, direction=direction
                )
                npt.assert_allclose(reconstructed, expected, atol=1e-13, rtol=0.0)

    def test_size_one_factorization_and_lcp(self) -> None:
        system = TridiagonalLCP(
            lower=np.array([]),
            diagonal=np.array([2.0]),
            upper=np.array([]),
            rhs=np.array([3.0]),
            obstacle=np.array([2.0]),
        )
        factorization = factorize_projected_lu(system)
        for mode in ("single_put", "single_call", "double"):
            result = projected_lu_lcp_solve(system, factorization, mode=mode)
            self.assertTrue(result.converged)
            npt.assert_allclose(result.solution, np.array([2.0]))

    def test_factorization_hash_rejects_different_matrix(self) -> None:
        system = _system(np.ones(5))
        factorization = factorize_projected_lu(system)
        changed = TridiagonalLCP(
            system.lower,
            system.diagonal + 0.01,
            system.upper,
            system.rhs,
            system.obstacle,
        )
        with self.assertRaisesRegex(ValueError, "hash"):
            projected_lu_lcp_solve(changed, factorization, mode="double")

    def test_zero_pivot_and_invalid_inputs_fail_explicitly(self) -> None:
        zero_pivot = TridiagonalLCP(
            lower=np.array([1.0]),
            diagonal=np.array([1.0, 1.0]),
            upper=np.array([1.0]),
            rhs=np.ones(2),
            obstacle=np.zeros(2),
        )
        with self.assertRaisesRegex(ValueError, "pivot"):
            factorize_projected_lu(zero_pivot)
        with self.assertRaises(ValueError):
            factorize_projected_lu(_system(np.ones(5)), directions=("bad",))


class ProjectedLUSolveTests(unittest.TestCase):
    def test_put_direction_passes_and_wrong_direction_is_detected(self) -> None:
        rhs = np.array(
            [-0.8019314252534474, -1.324358995628145, -0.24836162209524854,
             0.4204452380655215, 1.1360465324896427]
        )
        system = _system(rhs)
        factorization = factorize_projected_lu(system)
        policy = policy_iteration_lcp_solve(system, tolerance=1e-12)
        correct = projected_lu_lcp_solve(system, factorization, mode="single_put")
        wrong = projected_lu_lcp_solve(system, factorization, mode="single_call")
        self.assertTrue(correct.converged)
        self.assertFalse(wrong.converged)
        npt.assert_allclose(correct.solution, policy.solution, atol=1e-13, rtol=0.0)

    def test_call_direction_passes_and_wrong_direction_is_detected(self) -> None:
        rhs = np.array(
            [0.0012301533574825742, 0.2987455375084699, -0.2741378553622176,
             -0.8905918387572742, -0.45467078517172255]
        )
        system = _system(rhs)
        factorization = factorize_projected_lu(system)
        policy = policy_iteration_lcp_solve(system, tolerance=1e-12)
        correct = projected_lu_lcp_solve(system, factorization, mode="single_call")
        wrong = projected_lu_lcp_solve(system, factorization, mode="single_put")
        self.assertTrue(correct.converged)
        self.assertFalse(wrong.converged)
        npt.assert_allclose(correct.solution, policy.solution, atol=1e-13, rtol=0.0)

    def test_double_sweep_solves_two_sided_contact(self) -> None:
        rhs = np.array(
            [0.03419276725318417, 1.3597475403099617, 1.2247210785859324,
             -0.5103070767876675, -0.2979695111064471,
             -0.5273841930334252, 0.5697263575719601]
        )
        size = len(rhs)
        system = TridiagonalLCP(
            lower=np.full(size - 1, -0.4),
            diagonal=np.full(size, 2.0),
            upper=np.full(size - 1, -0.3),
            rhs=rhs,
            obstacle=np.zeros(size),
        )
        factorization = factorize_projected_lu(system)
        policy = policy_iteration_lcp_solve(system, tolerance=1e-12)
        put = projected_lu_lcp_solve(system, factorization, mode="single_put")
        call = projected_lu_lcp_solve(system, factorization, mode="single_call")
        double = projected_lu_lcp_solve(system, factorization, mode="double")
        self.assertFalse(put.converged)
        self.assertFalse(call.converged)
        self.assertTrue(double.converged)
        self.assertEqual(2, double.iterations)
        npt.assert_allclose(double.solution, policy.solution, atol=1e-13, rtol=0.0)

    def test_non_m_matrix_is_reported_without_blocking_diagnostic_run(self) -> None:
        system = TridiagonalLCP(
            lower=np.array([0.1]),
            diagonal=np.array([2.0, 2.0]),
            upper=np.array([-0.2]),
            rhs=np.array([1.0, 1.0]),
            obstacle=np.zeros(2),
        )
        eligibility = audit_projected_lu_eligibility(system)
        self.assertFalse(eligibility.nonpositive_offdiagonals)
        self.assertFalse(eligibility.theorem_eligible)
        result = projected_lu_lcp_solve(
            system, factorize_projected_lu(system), mode="double"
        )
        self.assertTrue(np.all(np.isfinite(result.solution)))


class ProjectedLUMarcherTests(unittest.TestCase):
    def test_single_sweep_matches_policy_for_put_call_and_q0_call(self) -> None:
        for option_type, q in (("put", 0.03), ("call", 0.06), ("call", 0.0)):
            with self.subTest(option_type=option_type, q=q):
                config = AmericanLCPConfig(
                    option_type, 1.0, 0.5, 0.05, q, 0.2, 4.0, 60, 60,
                    tolerance=1e-12, obstacle_tolerance=1e-12,
                )
                policy = american_cn_lcp_price(config, lcp_solver="policy_iteration")
                projected = american_cn_lcp_price(
                    config, lcp_solver="projected_lu_single"
                )
                self.assertTrue(projected.converged)
                self.assertEqual("none_direct", projected.initializer)
                npt.assert_allclose(
                    projected.value_grid, policy.value_grid, atol=1e-9, rtol=0.0
                )

    def test_factorization_is_built_once_per_surface(self) -> None:
        config = AmericanLCPConfig(
            "put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 30, 20,
            tolerance=1e-12, obstacle_tolerance=1e-12,
        )
        from american_risk_surfaces.solvers import american_lcp as marcher

        with mock.patch.object(
            marcher,
            "factorize_projected_lu",
            wraps=marcher.factorize_projected_lu,
        ) as factorize:
            result = marcher.american_cn_lcp_price(
                config, lcp_solver="projected_lu_single"
            )
        self.assertTrue(result.converged)
        factorize.assert_called_once()
        self.assertGreaterEqual(result.solver_setup_seconds, 0.0)

    def test_projected_lu_rejects_custom_initializer(self) -> None:
        config = AmericanLCPConfig("put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 20, 20)

        def initializer(_step, _tau, previous, _obstacle):
            return previous

        with self.assertRaisesRegex(ValueError, "initializer"):
            american_cn_lcp_price(
                config,
                lcp_solver="projected_lu_single",
                initializer=initializer,
            )


def _system(rhs: np.ndarray) -> TridiagonalLCP:
    size = len(rhs)
    return TridiagonalLCP(
        lower=np.full(size - 1, -0.4),
        diagonal=np.full(size, 2.0),
        upper=np.full(size - 1, -0.3),
        rhs=rhs,
        obstacle=np.zeros(size),
    )


def _dense(system: TridiagonalLCP) -> np.ndarray:
    dense = np.diag(system.diagonal)
    if system.size > 1:
        dense += np.diag(system.lower, -1)
        dense += np.diag(system.upper, 1)
    return dense


if __name__ == "__main__":
    unittest.main()

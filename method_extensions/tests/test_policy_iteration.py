"""Tests for the residual-controlled LCP solvers and common CN marcher."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.diagnostics.boundary import extract_boundary_curve
from american_risk_surfaces.diagnostics.greeks import diagnose_greek_result
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta_nonuniform,
    finite_difference_gamma_nonuniform,
)
from american_risk_surfaces.solvers.american_lcp import (
    AmericanLCPConfig,
    american_cn_lcp_price,
    as_legacy_cn_psor_result,
)
from american_risk_surfaces.solvers.black_scholes import european_call_price
from american_risk_surfaces.solvers.lcp import (
    TridiagonalLCP,
    compute_lcp_residual,
    psor_lcp_solve_residual,
)
from american_risk_surfaces.solvers.policy_iteration import policy_iteration_lcp_solve
from american_risk_surfaces.method_extensions.premium_warmstart import (
    GatedSurfaceInitializer,
    PositivePremiumSurfaceModel,
    train_positive_premium_checkpoint,
)
from american_risk_surfaces.solvers.greek_integrators import (
    american_dirk_policy_price,
    american_lobatto_penalty_price,
    quadratic_tau_grid,
)
from american_risk_surfaces.solvers.grid import (
    sinh_spot_grid,
    uniform_spot_grid,
    uniform_tau_grid,
)
from american_risk_surfaces.solvers.operator import (
    black_scholes_operator_coefficients,
    black_scholes_operator_coefficients_nonuniform,
)
from american_risk_surfaces.surrogates import price_premium as stage3


class PolicyIterationTests(unittest.TestCase):
    def test_toy_lcp_has_known_solution(self) -> None:
        system = TridiagonalLCP(
            lower=np.array([0.0, 0.0]),
            diagonal=np.ones(3),
            upper=np.array([0.0, 0.0]),
            rhs=np.array([0.0, 2.0, -1.0]),
            obstacle=np.array([1.0, 1.0, 0.0]),
        )
        result = policy_iteration_lcp_solve(system, tolerance=1e-12)
        self.assertTrue(result.converged)
        npt.assert_allclose(result.solution, np.array([1.0, 2.0, 0.0]))
        self.assertEqual("policy_iteration", result.method)
        self.assertLessEqual(result.residual.normalized_lcp_residual, 1e-12)

    def test_shared_residual_detects_all_three_failure_modes(self) -> None:
        system = TridiagonalLCP(
            lower=np.array([0.0]),
            diagonal=np.ones(2),
            upper=np.array([0.0]),
            rhs=np.array([1.0, 1.0]),
            obstacle=np.array([0.5, 0.5]),
        )
        below = compute_lcp_residual(system, np.array([0.0, 0.0]))
        above = compute_lcp_residual(system, np.array([2.0, 2.0]))
        self.assertGreater(below.max_obstacle_violation, 0.0)
        self.assertGreater(below.max_equation_violation, 0.0)
        self.assertGreater(above.max_abs_complementarity, 0.0)

    def test_policy_and_psor_agree_on_put_and_dividend_call(self) -> None:
        for option_type, q in (("put", 0.02), ("call", 0.08)):
            with self.subTest(option_type=option_type):
                config = AmericanLCPConfig(
                    option_type=option_type,
                    K=1.0,
                    T=0.5,
                    r=0.05,
                    q=q,
                    sigma=0.2,
                    Smax=4.0,
                    M=40,
                    N=40,
                    tolerance=1e-11,
                )
                psor = american_cn_lcp_price(config, lcp_solver="psor")
                policy = american_cn_lcp_price(config, lcp_solver="policy_iteration")
                self.assertTrue(psor.converged)
                self.assertTrue(policy.converged)
                npt.assert_allclose(psor.value_grid, policy.value_grid, atol=5e-8, rtol=0.0)
                self.assertLessEqual(policy.max_obstacle_violation, 1e-12)

    def test_no_dividend_call_matches_european_control(self) -> None:
        config = AmericanLCPConfig("call", 1.0, 1.0, 0.05, 0.0, 0.2, 4.0, 100, 100)
        result = american_cn_lcp_price(config, lcp_solver="policy_iteration")
        exact = european_call_price(
            result.spot_grid, K=1.0, T=1.0, r=0.05, q=0.0, sigma=0.2
        )
        target = (result.spot_grid >= 0.4) & (result.spot_grid <= 1.8)
        self.assertLess(float(np.max(np.abs(result.values[target] - exact[target]))), 2e-3)

    def test_custom_initializer_is_projected_and_shape_checked(self) -> None:
        config = AmericanLCPConfig("put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 20, 10)

        def below_obstacle(_step, _tau, previous, _obstacle):
            return np.full_like(previous, -100.0)

        result = american_cn_lcp_price(
            config, lcp_solver="policy_iteration", initializer=below_obstacle
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(result.max_obstacle_violation, 1e-12)

        def wrong_shape(_step, _tau, _previous, _obstacle):
            return np.zeros(2)

        with self.assertRaises(ValueError):
            american_cn_lcp_price(config, initializer=wrong_shape)

    def test_nonconvergence_and_invalid_inputs_are_reported(self) -> None:
        system = TridiagonalLCP(
            lower=np.array([-1.0]),
            diagonal=np.array([2.0, 2.0]),
            upper=np.array([-1.0]),
            rhs=np.array([1.0, 1.0]),
            obstacle=np.zeros(2),
        )
        result = psor_lcp_solve_residual(system, max_iter=1, tolerance=1e-16)
        self.assertFalse(result.converged)
        with self.assertRaises(ValueError):
            TridiagonalLCP(np.zeros(2), np.ones(2), np.zeros(1), np.ones(2), np.zeros(2))

    def test_zero_maturity_and_legacy_diagnostic_adapter(self) -> None:
        zero = american_cn_lcp_price(
            AmericanLCPConfig("put", 1.0, 0.0, 0.05, 0.02, 0.2, 4.0, 20, 3)
        )
        npt.assert_allclose(zero.values, zero.payoff)
        self.assertEqual((), zero.lcp_results)

        priced = american_cn_lcp_price(
            AmericanLCPConfig("put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 30, 30)
        )
        legacy = as_legacy_cn_psor_result(priced)
        boundary = extract_boundary_curve(legacy, "policy_put")
        greeks = diagnose_greek_result(legacy, "policy_put", boundary_curve=boundary)
        self.assertEqual(priced.value_grid.shape, greeks.value_grid.shape)

    def test_nonuniform_grid_operator_and_greeks(self) -> None:
        uniform, spacing = uniform_spot_grid(4.0, 40)
        classic = black_scholes_operator_coefficients(
            uniform, spacing, r=0.05, q=0.02, sigma=0.2
        )
        generalized = black_scholes_operator_coefficients_nonuniform(
            uniform, r=0.05, q=0.02, sigma=0.2
        )
        npt.assert_allclose(generalized.lower, classic.lower, atol=1e-12)
        npt.assert_allclose(generalized.diagonal, classic.diagonal, atol=1e-12)
        npt.assert_allclose(generalized.upper, classic.upper, atol=1e-12)

        concentrated = sinh_spot_grid(4.0, 1.0, 80)
        self.assertEqual(0.0, concentrated[0])
        self.assertEqual(4.0, concentrated[-1])
        self.assertTrue(np.all(np.diff(concentrated) > 0.0))
        self.assertLess(float(np.min(np.abs(concentrated - 1.0))), 1e-14)
        quadratic = concentrated**2
        delta = finite_difference_delta_nonuniform(concentrated, quadratic)
        gamma = finite_difference_gamma_nonuniform(concentrated, quadratic)
        npt.assert_allclose(delta[1:-1], 2.0 * concentrated[1:-1], atol=1e-11)
        npt.assert_allclose(gamma[1:-1], 2.0, atol=1e-9)

    def test_greek_integrators_smoke_and_time_grid(self) -> None:
        times = quadratic_tau_grid(1.0, 10)
        self.assertTrue(np.all(np.diff(times) > 0.0))
        self.assertEqual(0.0, times[0])
        self.assertEqual(1.0, times[-1])
        config = AmericanLCPConfig(
            "put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 30, 20, tolerance=1e-10
        )
        for solver in (american_dirk_policy_price, american_lobatto_penalty_price):
            with self.subTest(solver=solver.__name__):
                result = solver(config)
                self.assertTrue(result.converged)
                self.assertLessEqual(result.max_obstacle_violation, 1e-9)
                self.assertEqual((21, 31), result.value_grid.shape)

    @unittest.skipIf(stage3.torch is None, "PyTorch is unavailable")
    def test_positive_premium_checkpoint_round_trip_and_support_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "premium.pt"
            outputs = train_positive_premium_checkpoint(
                path,
                train_cap=256,
                epochs=1,
                batch_size=64,
                hidden_units=8,
            )
            self.assertTrue(outputs["checkpoint"].exists())
            self.assertTrue(outputs["manifest"].exists())
            model = PositivePremiumSurfaceModel.load(path)
            config = AmericanLCPConfig("put", 1.0, 0.25, 0.05, 0.02, 0.2, 4.0, 20, 10)
            spots, _ = uniform_spot_grid(config.Smax, config.M)
            taus, _ = uniform_tau_grid(config.T, config.N)
            prediction = model.predict_surface(config, spots, taus)
            self.assertEqual((11, 19), prediction.premium_grid.shape)
            self.assertTrue(np.all(prediction.premium_grid >= 0.0))

            initializer = GatedSurfaceInitializer(
                prediction.value_grid, spots[1:-1], config.K
            )
            previous = np.full(19, 7.0)
            obstacle = np.zeros(19)
            initialized = initializer(1, taus[1], previous, obstacle)
            outside = (spots[1:-1] / config.K < 0.4) | (spots[1:-1] / config.K > 1.8)
            npt.assert_allclose(initialized[outside], previous[outside])


if __name__ == "__main__":
    unittest.main()

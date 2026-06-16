import unittest

import numpy as np
import numpy.testing as npt

import american_risk_surfaces.solvers.cn_psor as cn_psor
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.cn_psor import (
    american_crank_nicolson_psor_price,
    psor_lcp_solve,
)


class PSORCoreTests(unittest.TestCase):
    def test_psor_lcp_projects_known_toy_solution(self):
        result = psor_lcp_solve(
            lower=np.array([0.0, 0.0]),
            diagonal=np.array([1.0, 1.0, 1.0]),
            upper=np.array([0.0, 0.0]),
            rhs=np.array([0.0, 2.0, -1.0]),
            payoff=np.array([1.0, 1.0, 0.0]),
            omega=1.0,
            tolerance=1e-12,
            max_iter=10,
        )

        self.assertTrue(result.converged)
        npt.assert_allclose(result.solution, np.array([1.0, 2.0, 0.0]))

    def test_american_put_smoke_has_no_obstacle_violation(self):
        result = american_crank_nicolson_psor_price(
            option_type="put",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=4.0,
            M=40,
            N=40,
        )

        self.assertTrue(result.converged)
        self.assertLessEqual(result.max_obstacle_violation, 1e-8)
        self.assertEqual(result.value_grid.shape, (41, 41))
        npt.assert_array_less(
            result.payoff[np.newaxis, :] - result.value_grid,
            np.full_like(result.value_grid, 1e-8),
        )

    def test_convergence_metadata_is_exposed(self):
        result = psor_lcp_solve(
            lower=np.array([0.0]),
            diagonal=np.array([2.0, 2.0]),
            upper=np.array([0.0]),
            rhs=np.array([2.0, 4.0]),
            payoff=np.array([0.0, 0.0]),
            omega=1.1,
            tolerance=1e-10,
            max_iter=100,
        )

        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertLessEqual(result.final_update, result.tolerance)
        self.assertAlmostEqual(result.tolerance, 1e-10)
        self.assertAlmostEqual(result.omega, 1.1)
        self.assertEqual(result.max_iter, 100)

    def test_non_convergence_is_reported(self):
        result = psor_lcp_solve(
            lower=np.array([0.0]),
            diagonal=np.array([1.0, 1.0]),
            upper=np.array([0.0]),
            rhs=np.array([10.0, 20.0]),
            payoff=np.array([0.0, 0.0]),
            initial=np.array([0.0, 0.0]),
            omega=1.0,
            tolerance=1e-14,
            max_iter=1,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertGreater(result.final_update, result.tolerance)

    def test_zero_maturity_returns_payoff(self):
        put_result = american_crank_nicolson_psor_price(
            option_type="put",
            K=1.0,
            T=0.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            Smax=4.0,
            M=20,
            N=3,
        )
        call_result = american_crank_nicolson_psor_price(
            option_type="call",
            K=1.0,
            T=0.0,
            r=0.05,
            q=0.08,
            sigma=0.2,
            Smax=4.0,
            M=20,
            N=3,
        )

        npt.assert_allclose(put_result.values, put_payoff(put_result.spot_grid, 1.0))
        npt.assert_allclose(call_result.values, call_payoff(call_result.spot_grid, 1.0))
        self.assertEqual(put_result.psor_results, ())
        self.assertEqual(call_result.psor_results, ())
        self.assertAlmostEqual(put_result.max_obstacle_violation, 0.0)
        self.assertAlmostEqual(call_result.max_obstacle_violation, 0.0)

    def test_american_call_with_dividends_runs_with_call_payoff_and_boundaries(self):
        result = american_crank_nicolson_psor_price(
            option_type="call",
            K=1.0,
            T=0.5,
            r=0.05,
            q=0.08,
            sigma=0.25,
            Smax=4.0,
            M=30,
            N=30,
        )

        self.assertTrue(result.converged)
        npt.assert_allclose(result.payoff, call_payoff(result.spot_grid, 1.0))
        npt.assert_allclose(result.value_grid[:, 0], np.zeros_like(result.tau_grid))
        self.assertTrue(np.all(result.value_grid >= result.payoff[np.newaxis, :] - 1e-8))
        self.assertGreater(result.values[-1], 0.0)

    def test_invalid_inputs_raise_value_error(self):
        invalid_solver_calls = [
            {"option_type": "straddle", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 0.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": -0.1, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": -0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 0.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 1, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 0},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20, "omega": 0.0},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20, "omega": 2.0},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20, "tolerance": 0.0},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20, "max_iter": 0},
        ]

        for kwargs in invalid_solver_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    american_crank_nicolson_psor_price(**kwargs)

        invalid_lcp_calls = [
            (np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])),
            (np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0])),
            (np.array([0.0]), np.array([1.0, 1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0, 0.0])),
        ]
        for lower, diagonal, upper, rhs, payoff in invalid_lcp_calls:
            with self.subTest(diagonal=diagonal, rhs=rhs):
                with self.assertRaises(ValueError):
                    psor_lcp_solve(lower, diagonal, upper, rhs, payoff)

        with self.assertRaises(ValueError):
            psor_lcp_solve(
                np.array([0.0]),
                np.array([1.0, 1.0]),
                np.array([0.0]),
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
                initial=np.array([0.0]),
            )

    def test_public_api_has_no_boundary_greek_stress_dataset_or_neural_surface(self):
        forbidden_fragments = (
            "boundary",
            "extract",
            "premium",
            "greek",
            "stress",
            "dataset",
            "neural",
        )

        for public_name in cn_psor.__all__:
            with self.subTest(public_name=public_name):
                lowered = public_name.lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden_fragments))


if __name__ == "__main__":
    unittest.main()

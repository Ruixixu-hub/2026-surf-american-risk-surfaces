import math
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers import cn
from american_risk_surfaces.solvers.cn import (
    european_crank_nicolson_price,
    solve_tridiagonal,
    target_region_error_metrics,
)


class EuropeanCrankNicolsonTests(unittest.TestCase):
    def test_tridiagonal_solver_matches_known_system(self):
        lower = np.array([-1.0, -1.0])
        diagonal = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0])
        expected = np.array([1.0, 2.0, 3.0])
        rhs = np.array([0.0, 0.0, 4.0])

        result = solve_tridiagonal(lower, diagonal, upper, rhs)

        npt.assert_allclose(result, expected)

    def test_zero_maturity_returns_payoff_for_call_and_put(self):
        call_result = european_crank_nicolson_price(
            option_type="call", K=1.0, T=0.0, r=0.05, q=0.02, sigma=0.2, Smax=4.0, M=20, N=1
        )
        put_result = european_crank_nicolson_price(
            option_type="put", K=1.0, T=0.0, r=0.05, q=0.02, sigma=0.2, Smax=4.0, M=20, N=1
        )

        npt.assert_allclose(call_result.values, call_payoff(call_result.spot_grid, K=1.0))
        npt.assert_allclose(call_result.values, call_result.closed_form_values)
        npt.assert_allclose(put_result.values, put_payoff(put_result.spot_grid, K=1.0))
        npt.assert_allclose(put_result.values, put_result.closed_form_values)
        self.assertAlmostEqual(call_result.metrics.max_abs_error, 0.0)
        self.assertAlmostEqual(put_result.metrics.rmse, 0.0)

    def test_european_put_cn_close_to_closed_form(self):
        result = european_crank_nicolson_price(
            option_type="put", K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2, Smax=4.0, M=120, N=120
        )

        self.assertLess(result.metrics.max_abs_error, 0.025)
        self.assertLess(result.metrics.rmse, 0.010)

    def test_european_call_cn_close_to_closed_form(self):
        result = european_crank_nicolson_price(
            option_type="call", K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2, Smax=4.0, M=120, N=120
        )

        self.assertLess(result.metrics.max_abs_error, 0.025)
        self.assertLess(result.metrics.rmse, 0.010)

    def test_target_region_error_metrics_known_values(self):
        spots = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
        numerical = np.array([10.0, 1.0, 2.0, 4.0, 20.0])
        reference = np.array([0.0, 1.0, 3.0, 2.0, 0.0])

        metrics = target_region_error_metrics(
            spots,
            numerical,
            reference,
            K=1.0,
            lower_moneyness=0.4,
            upper_moneyness=1.8,
        )

        self.assertAlmostEqual(metrics.max_abs_error, 2.0)
        self.assertAlmostEqual(metrics.rmse, math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(metrics.max_error_spot, 1.5)
        self.assertAlmostEqual(metrics.target_lower, 0.4)
        self.assertAlmostEqual(metrics.target_upper, 1.8)

    def test_invalid_option_type_and_grid_parameters_raise_value_error(self):
        invalid_solver_calls = [
            {"option_type": "american_put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 0.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": -1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": -0.2, "Smax": 4.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 0.0, "M": 20, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 1, "N": 20},
            {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2, "Smax": 4.0, "M": 20, "N": 0},
        ]

        for kwargs in invalid_solver_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    european_crank_nicolson_price(**kwargs)

        invalid_tridiagonal_calls = [
            (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])),
            (np.array([1.0]), np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0, 1.0])),
        ]
        for lower, diagonal, upper, rhs in invalid_tridiagonal_calls:
            with self.subTest(diagonal=diagonal):
                with self.assertRaises(ValueError):
                    solve_tridiagonal(lower, diagonal, upper, rhs)

        with self.assertRaises(ValueError):
            target_region_error_metrics(
                np.array([0.1, 0.2]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                K=1.0,
            )

    def test_public_api_contains_no_psor_or_american_projection_logic(self):
        public_names = [name.lower() for name in dir(cn) if not name.startswith("_")]
        forbidden_fragments = ("psor", "american", "projection", "obstacle")

        for name in public_names:
            with self.subTest(name=name):
                self.assertFalse(any(fragment in name for fragment in forbidden_fragments))


if __name__ == "__main__":
    unittest.main()

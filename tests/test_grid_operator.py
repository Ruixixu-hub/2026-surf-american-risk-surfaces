import math
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.solvers.grid import (
    interior_indices,
    uniform_spot_grid,
    uniform_tau_grid,
)
from american_risk_surfaces.solvers.operator import (
    american_call_boundaries,
    american_put_boundaries,
    apply_black_scholes_operator,
    black_scholes_operator_coefficients,
    european_call_boundaries,
    european_put_boundaries,
)


class GridAndOperatorTests(unittest.TestCase):
    def test_spot_grid_endpoints_spacing_and_length(self):
        spots, dS = uniform_spot_grid(Smax=4.0, M=8)

        self.assertEqual(len(spots), 9)
        self.assertAlmostEqual(dS, 0.5)
        self.assertAlmostEqual(spots[0], 0.0)
        self.assertAlmostEqual(spots[-1], 4.0)
        npt.assert_allclose(spots, np.linspace(0.0, 4.0, 9))

    def test_tau_grid_endpoints_spacing_and_length(self):
        taus, dtau = uniform_tau_grid(T=2.0, N=4)

        self.assertEqual(len(taus), 5)
        self.assertAlmostEqual(dtau, 0.5)
        self.assertAlmostEqual(taus[0], 0.0)
        self.assertAlmostEqual(taus[-1], 2.0)
        npt.assert_allclose(taus, np.linspace(0.0, 2.0, 5))

    def test_zero_maturity_tau_grid_is_allowed(self):
        taus, dtau = uniform_tau_grid(T=0.0, N=3)

        self.assertAlmostEqual(dtau, 0.0)
        npt.assert_allclose(taus, np.zeros(4))

    def test_invalid_grid_parameters_raise_value_error(self):
        invalid_calls = [
            (uniform_spot_grid, {"Smax": 0.0, "M": 2}),
            (uniform_spot_grid, {"Smax": 1.0, "M": 1}),
            (uniform_tau_grid, {"T": -1.0, "N": 1}),
            (uniform_tau_grid, {"T": 1.0, "N": 0}),
            (interior_indices, {"M": 1}),
        ]

        for function, kwargs in invalid_calls:
            with self.subTest(function=function.__name__, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    function(**kwargs)

    def test_interior_indices_exclude_boundary_nodes(self):
        npt.assert_array_equal(interior_indices(M=5), np.array([1, 2, 3, 4]))

    def test_operator_coefficient_shapes(self):
        spots, dS = uniform_spot_grid(Smax=3.0, M=3)

        coefficients = black_scholes_operator_coefficients(
            spots, dS=dS, r=0.05, q=0.02, sigma=0.2
        )

        self.assertEqual(coefficients.lower.shape, (2,))
        self.assertEqual(coefficients.diagonal.shape, (2,))
        self.assertEqual(coefficients.upper.shape, (2,))
        npt.assert_allclose(coefficients.interior_spots, np.array([1.0, 2.0]))

    def test_operator_coefficients_match_manual_small_example(self):
        spots, dS = uniform_spot_grid(Smax=3.0, M=3)

        coefficients = black_scholes_operator_coefficients(
            spots, dS=dS, r=0.05, q=0.02, sigma=0.2
        )

        npt.assert_allclose(coefficients.lower, np.array([0.005, 0.05]))
        npt.assert_allclose(coefficients.diagonal, np.array([-0.09, -0.21]))
        npt.assert_allclose(coefficients.upper, np.array([0.035, 0.11]))

    def test_operator_rejects_non_increasing_spot_grid(self):
        invalid_grids = [
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 1.0, 3.0]),
        ]

        for spots in invalid_grids:
            with self.subTest(spots=spots):
                with self.assertRaises(ValueError):
                    black_scholes_operator_coefficients(
                        spots, dS=1.0, r=0.05, q=0.02, sigma=0.2
                    )

    def test_operator_rejects_nonuniform_spot_grid(self):
        with self.assertRaises(ValueError):
            black_scholes_operator_coefficients(
                np.array([0.0, 1.0, 2.5, 3.0]), dS=1.0, r=0.05, q=0.02, sigma=0.2
            )

    def test_operator_rejects_dS_mismatch(self):
        with self.assertRaises(ValueError):
            black_scholes_operator_coefficients(
                np.array([0.0, 1.0, 2.0, 3.0]), dS=0.5, r=0.05, q=0.02, sigma=0.2
            )

    def test_apply_operator_to_constant_returns_minus_r(self):
        spots, dS = uniform_spot_grid(Smax=2.0, M=4)
        rate = 0.07
        coefficients = black_scholes_operator_coefficients(
            spots, dS=dS, r=rate, q=0.02, sigma=0.25
        )

        result = apply_black_scholes_operator(np.ones_like(spots), coefficients)

        npt.assert_allclose(result, -rate * np.ones_like(coefficients.interior_spots))

    def test_apply_operator_to_linear_spot_returns_minus_q_times_spot(self):
        spots, dS = uniform_spot_grid(Smax=2.0, M=4)
        dividend = 0.03
        coefficients = black_scholes_operator_coefficients(
            spots, dS=dS, r=0.07, q=dividend, sigma=0.25
        )

        result = apply_black_scholes_operator(spots, coefficients)

        npt.assert_allclose(result, -dividend * coefficients.interior_spots)

    def test_put_boundary_helpers(self):
        european_lower, european_upper = european_put_boundaries(K=1.0, tau=2.0, r=0.05)
        american_lower, american_upper = american_put_boundaries(K=1.0, tau=2.0)

        self.assertAlmostEqual(european_lower, math.exp(-0.1))
        self.assertAlmostEqual(european_upper, 0.0)
        self.assertAlmostEqual(american_lower, 1.0)
        self.assertAlmostEqual(american_upper, 0.0)

    def test_call_boundary_helpers(self):
        Smax = 4.0
        strike = 1.0
        tau = 2.0
        rate = 0.05
        dividend = 0.02
        asymptotic = Smax * math.exp(-dividend * tau) - strike * math.exp(-rate * tau)

        european_lower, european_upper = european_call_boundaries(
            Smax=Smax, K=strike, tau=tau, r=rate, q=dividend
        )
        american_lower, american_upper = american_call_boundaries(
            Smax=Smax, K=strike, tau=tau, r=rate, q=dividend
        )

        self.assertAlmostEqual(european_lower, 0.0)
        self.assertAlmostEqual(european_upper, max(asymptotic, 0.0))
        self.assertAlmostEqual(american_lower, 0.0)
        self.assertAlmostEqual(american_upper, max(Smax - strike, asymptotic))

    def test_american_call_boundary_upper_is_nonnegative_for_small_domain(self):
        _, upper = american_call_boundaries(Smax=0.25, K=1.0, tau=1.0, r=0.0, q=1.0)

        self.assertAlmostEqual(upper, 0.0)

    def test_invalid_operator_and_boundary_inputs_raise_value_error(self):
        invalid_calls = [
            (
                black_scholes_operator_coefficients,
                {
                    "spot_grid": np.array([[0.0, 1.0], [2.0, 3.0]]),
                    "dS": 1.0,
                    "r": 0.05,
                    "q": 0.02,
                    "sigma": 0.2,
                },
            ),
            (
                black_scholes_operator_coefficients,
                {
                    "spot_grid": np.array([0.0, 1.0]),
                    "dS": 1.0,
                    "r": 0.05,
                    "q": 0.02,
                    "sigma": 0.2,
                },
            ),
            (
                black_scholes_operator_coefficients,
                {
                    "spot_grid": np.array([0.0, -1.0, 2.0]),
                    "dS": 1.0,
                    "r": 0.05,
                    "q": 0.02,
                    "sigma": 0.2,
                },
            ),
            (
                black_scholes_operator_coefficients,
                {
                    "spot_grid": np.array([0.0, 1.0, 2.0]),
                    "dS": 0.0,
                    "r": 0.05,
                    "q": 0.02,
                    "sigma": 0.2,
                },
            ),
            (
                black_scholes_operator_coefficients,
                {
                    "spot_grid": np.array([0.0, 1.0, 2.0]),
                    "dS": 1.0,
                    "r": 0.05,
                    "q": 0.02,
                    "sigma": -0.2,
                },
            ),
            (european_put_boundaries, {"K": 0.0, "tau": 1.0, "r": 0.05}),
            (american_put_boundaries, {"K": 1.0, "tau": -1.0}),
            (
                european_call_boundaries,
                {"Smax": 0.0, "K": 1.0, "tau": 1.0, "r": 0.05, "q": 0.02},
            ),
            (
                american_call_boundaries,
                {"Smax": 4.0, "K": -1.0, "tau": 1.0, "r": 0.05, "q": 0.02},
            ),
        ]

        for function, kwargs in invalid_calls:
            with self.subTest(function=function.__name__, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    function(**kwargs)


if __name__ == "__main__":
    unittest.main()

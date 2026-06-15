import math
import unittest

import numpy as np
import numpy.testing as npt

from american_risk_surfaces.solvers.black_scholes import (
    call_payoff,
    european_call_price,
    european_put_price,
    put_payoff,
)


class BlackScholesUtilityTests(unittest.TestCase):
    def test_call_payoff_below_at_above_strike(self):
        spots = np.array([0.8, 1.0, 1.2])

        result = call_payoff(spots, K=1.0)

        npt.assert_allclose(result, np.array([0.0, 0.0, 0.2]))

    def test_put_payoff_below_at_above_strike(self):
        spots = np.array([0.8, 1.0, 1.2])

        result = put_payoff(spots, K=1.0)

        npt.assert_allclose(result, np.array([0.2, 0.0, 0.0]))

    def test_scalar_inputs_return_scalars(self):
        self.assertIsInstance(call_payoff(1.2, K=1.0), float)
        self.assertIsInstance(put_payoff(0.8, K=1.0), float)
        self.assertIsInstance(
            european_call_price(1.1, K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2),
            float,
        )
        self.assertIsInstance(
            european_put_price(0.9, K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2),
            float,
        )

    def test_numpy_array_inputs_preserve_shape(self):
        spots = np.array([[0.8, 1.0], [1.2, 1.4]])

        call_prices = european_call_price(
            spots, K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2
        )
        put_prices = european_put_price(
            spots, K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2
        )

        self.assertIsInstance(call_prices, np.ndarray)
        self.assertIsInstance(put_prices, np.ndarray)
        self.assertEqual(call_prices.shape, spots.shape)
        self.assertEqual(put_prices.shape, spots.shape)

    def test_zero_maturity_prices_equal_payoffs(self):
        spots = np.array([0.8, 1.0, 1.2])

        call_prices = european_call_price(
            spots, K=1.0, T=0.0, r=0.05, q=0.02, sigma=0.2
        )
        put_prices = european_put_price(
            spots, K=1.0, T=0.0, r=0.05, q=0.02, sigma=0.2
        )

        npt.assert_allclose(call_prices, call_payoff(spots, K=1.0))
        npt.assert_allclose(put_prices, put_payoff(spots, K=1.0))

    def test_zero_volatility_uses_deterministic_discounted_payoff(self):
        spots = np.array([0.8, 1.0, 1.2])
        strike = 1.0
        maturity = 2.0
        rate = 0.05
        dividend = 0.02

        discounted_spots = spots * math.exp(-dividend * maturity)
        discounted_strike = strike * math.exp(-rate * maturity)
        expected_calls = np.maximum(discounted_spots - discounted_strike, 0.0)
        expected_puts = np.maximum(discounted_strike - discounted_spots, 0.0)

        call_prices = european_call_price(
            spots, K=strike, T=maturity, r=rate, q=dividend, sigma=0.0
        )
        put_prices = european_put_price(
            spots, K=strike, T=maturity, r=rate, q=dividend, sigma=0.0
        )

        npt.assert_allclose(call_prices, expected_calls)
        npt.assert_allclose(put_prices, expected_puts)

    def test_put_call_parity_with_continuous_dividends(self):
        spots = np.array([0.7, 1.0, 1.3])
        strike = 1.0
        maturity = 1.5
        rate = 0.04
        dividend = 0.01
        volatility = 0.25

        call_prices = european_call_price(
            spots, K=strike, T=maturity, r=rate, q=dividend, sigma=volatility
        )
        put_prices = european_put_price(
            spots, K=strike, T=maturity, r=rate, q=dividend, sigma=volatility
        )
        parity_rhs = spots * np.exp(-dividend * maturity) - strike * np.exp(
            -rate * maturity
        )

        npt.assert_allclose(call_prices - put_prices, parity_rhs, rtol=1e-12, atol=1e-12)

    def test_financial_sanity_nonnegative_and_monotone(self):
        spots = np.array([0.8, 1.0, 1.2])

        call_prices = european_call_price(
            spots, K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2
        )
        put_prices = european_put_price(
            spots, K=1.0, T=1.0, r=0.05, q=0.02, sigma=0.2
        )

        self.assertTrue(np.all(call_prices >= 0.0))
        self.assertTrue(np.all(put_prices >= 0.0))
        self.assertTrue(np.all(np.diff(call_prices) > 0.0))
        self.assertTrue(np.all(np.diff(put_prices) < 0.0))

    def test_invalid_inputs_raise_value_error(self):
        invalid_cases = [
            (call_payoff, (1.0,), {"K": 0.0}),
            (put_payoff, (-0.1,), {"K": 1.0}),
            (
                european_call_price,
                (1.0,),
                {"K": -1.0, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2},
            ),
            (
                european_put_price,
                (1.0,),
                {"K": 1.0, "T": -0.1, "r": 0.05, "q": 0.0, "sigma": 0.2},
            ),
            (
                european_call_price,
                (1.0,),
                {"K": 1.0, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": -0.2},
            ),
        ]

        for function, args, kwargs in invalid_cases:
            with self.subTest(function=function.__name__, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    function(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()

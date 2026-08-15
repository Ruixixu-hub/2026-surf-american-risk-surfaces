"""Payoff and European Black-Scholes utilities with continuous dividends."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def call_payoff(S: Any, K: float):
    """Return the call payoff max(S - K, 0).

    A call option gives the holder the right to buy the underlying asset at
    strike K. Its immediate exercise value is positive only when the spot S is
    above the strike.
    """

    spots, scalar_input = _validated_spots(S)
    strike = _validated_strike(K)
    return _return_like_input(np.maximum(spots - strike, 0.0), scalar_input)


def put_payoff(S: Any, K: float):
    """Return the put payoff max(K - S, 0).

    A put option gives the holder the right to sell the underlying asset at
    strike K. Its immediate exercise value is positive only when the spot S is
    below the strike.
    """

    spots, scalar_input = _validated_spots(S)
    strike = _validated_strike(K)
    return _return_like_input(np.maximum(strike - spots, 0.0), scalar_input)


def european_call_price(S: Any, K: float, T: float, r: float, q: float, sigma: float):
    """Return the European call price under Black-Scholes with dividends.

    The option can be exercised only at maturity. The continuous dividend yield
    q discounts the stock leg, while the risk-free rate r discounts the strike
    payment. At T = 0 the price equals payoff. At sigma = 0 the random
    diffusion vanishes, so the price is the discounted deterministic payoff.
    """

    spots, scalar_input = _validated_spots(S)
    strike, maturity, volatility = _validated_price_inputs(K, T, sigma)

    if maturity == 0.0:
        return call_payoff(S, strike)

    discounted_spots = spots * np.exp(-float(q) * maturity)
    discounted_strike = strike * np.exp(-float(r) * maturity)
    if volatility == 0.0:
        price = np.maximum(discounted_spots - discounted_strike, 0.0)
        return _return_like_input(price, scalar_input)

    d1, d2 = _d1_d2(spots, strike, maturity, r, q, volatility)
    price = discounted_spots * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return _return_like_input(price, scalar_input)


def european_put_price(S: Any, K: float, T: float, r: float, q: float, sigma: float):
    """Return the European put price under Black-Scholes with dividends.

    The option can be exercised only at maturity. The continuous dividend yield
    q discounts the stock leg, while the risk-free rate r discounts the strike
    payment. At T = 0 the price equals payoff. At sigma = 0 the random
    diffusion vanishes, so the price is the discounted deterministic payoff.
    """

    spots, scalar_input = _validated_spots(S)
    strike, maturity, volatility = _validated_price_inputs(K, T, sigma)

    if maturity == 0.0:
        return put_payoff(S, strike)

    discounted_spots = spots * np.exp(-float(q) * maturity)
    discounted_strike = strike * np.exp(-float(r) * maturity)
    if volatility == 0.0:
        price = np.maximum(discounted_strike - discounted_spots, 0.0)
        return _return_like_input(price, scalar_input)

    d1, d2 = _d1_d2(spots, strike, maturity, r, q, volatility)
    price = discounted_strike * _normal_cdf(-d2) - discounted_spots * _normal_cdf(-d1)
    return _return_like_input(price, scalar_input)


def _validated_spots(S: Any) -> tuple[np.ndarray, bool]:
    spots = np.asarray(S, dtype=float)
    scalar_input = spots.shape == ()
    if np.any(spots < 0.0):
        raise ValueError("S must be nonnegative.")
    return spots, scalar_input


def _validated_strike(K: float) -> float:
    strike = float(K)
    if strike <= 0.0:
        raise ValueError("K must be positive.")
    return strike


def _validated_price_inputs(K: float, T: float, sigma: float) -> tuple[float, float, float]:
    strike = _validated_strike(K)
    maturity = float(T)
    volatility = float(sigma)
    if maturity < 0.0:
        raise ValueError("T must be nonnegative.")
    if volatility < 0.0:
        raise ValueError("sigma must be nonnegative.")
    return strike, maturity, volatility


def _d1_d2(
    spots: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = volatility * math.sqrt(maturity)
    with np.errstate(divide="ignore"):
        d1 = (
            np.log(spots / strike)
            + (float(rate) - float(dividend) + 0.5 * volatility**2) * maturity
        ) / denominator
    return d1, d1 - denominator


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def _return_like_input(values: np.ndarray, scalar_input: bool):
    if scalar_input:
        return float(np.asarray(values))
    return values
